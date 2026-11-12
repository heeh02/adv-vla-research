"""Bounded local native tree codec for this bridge, never pickle or cloud IO."""
from pathlib import Path
import hashlib
import json
import zipfile
import numpy as np

MAX_NATIVE_BYTES = 512 * 1024**2
MAX_FILE_BYTES = 512 * 1024**2
MAX_METADATA_BYTES = 1024**2


def need(ok, code):
    if not ok:
        raise ValueError(code)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def file_ref(path):
    p = Path(path)
    h = hashlib.sha256()
    with p.open('rb') as handle:
        for raw in iter(lambda: handle.read(1024**2), b''):
            h.update(raw)
    return {'bytes': p.stat().st_size, 'sha256': h.hexdigest()}


def save_tree(path, value, *, allow_nonfinite=False):
    path = Path(path)
    arrays = {}
    total = 0

    def pack(x, depth=0):
        nonlocal total
        need(depth < 40, 'native_tree_depth')
        if isinstance(x, dict):
            need(all(type(k) is str for k in x), 'native_tree_string_keys')
            return {'kind': 'dict', 'items': {k: pack(v, depth+1) for k, v in x.items()}}
        if isinstance(x, (tuple, list)):
            return {'kind': 'tuple' if isinstance(x, tuple) else 'list', 'items': [pack(v, depth+1) for v in x]}
        if isinstance(x, np.ndarray) or hasattr(x, 'dtype'):
            a = np.asarray(x)
            need(a.dtype.name in ('bfloat16', 'float32', 'float64', 'int32', 'int64', 'uint16', 'uint8', 'uint32', 'bool'),
                 'native_leaf_dtype_allowlist')
            nonfinite = int(np.count_nonzero(~np.isfinite(a)))
            need(allow_nonfinite or nonfinite == 0, 'native_leaf_finite')
            storage = np.array(a.view(np.uint16) if a.dtype.name == 'bfloat16' else a, copy=True, order='C')
            total += storage.nbytes
            need(total <= MAX_NATIVE_BYTES and len(arrays) < 1000, 'bounded_native_tree')
            key = 'a_%06d' % len(arrays)
            arrays[key] = storage
            return {'kind': 'array', 'key': key, 'shape': list(a.shape), 'logical_dtype': a.dtype.name,
                    'storage_dtype': storage.dtype.str, 'bytes': storage.nbytes, 'sha256': sha(storage.tobytes()),
                    'nonfinite_elements': nonfinite}
        need(x is None or type(x) in (str, int, bool, float), 'native_json_scalar')
        return {'kind': 'scalar', 'value': x}

    document = {'schema': 'B-BRIDGE-native-tree-v1', 'tree': pack(value), 'native_bytes': total,
                'unqualified_nonfinite_preservation_enabled': allow_nonfinite}
    raw = json.dumps(document, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    need(len(raw) <= MAX_METADATA_BYTES, 'bounded_native_tree_metadata')
    arrays['metadata_json'] = np.frombuffer(raw, dtype=np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as handle:
        np.savez_compressed(handle, **arrays)
    ref = file_ref(path)
    need(ref['bytes'] <= MAX_FILE_BYTES, 'bounded_native_archive')
    return {**ref, 'native_bytes': total, 'schema': document['schema']}


def load_tree(path, ref, *, allow_nonfinite=False):
    path = Path(path)
    need(path.is_file() and not path.is_symlink() and path.stat().st_size <= MAX_FILE_BYTES, 'bounded_native_input')
    need(file_ref(path) == {k: ref[k] for k in ('bytes', 'sha256')}, 'native_wholefile_SHA')
    with zipfile.ZipFile(path) as archive:
        entries = archive.infolist()
        need(len(entries) <= 1001 and len({x.filename for x in entries}) == len(entries), 'native_unique_members')
        need(all(not x.is_dir() and '/' not in x.filename and not x.flag_bits & 1 for x in entries), 'native_regular_flat_members')
        need(sum(x.file_size for x in entries) <= MAX_NATIVE_BYTES + MAX_METADATA_BYTES + 256000, 'native_decompression_bound')
    with np.load(path, allow_pickle=False) as archive:
        raw = archive['metadata_json']
        need(raw.dtype == np.uint8 and raw.ndim == 1 and raw.nbytes <= MAX_METADATA_BYTES, 'native_metadata_ABI')
        doc = json.loads(raw.tobytes())
        need(doc['schema'] == 'B-BRIDGE-native-tree-v1', 'native_schema')
        used = set()
        total = 0

        def unpack(node, depth=0):
            nonlocal total
            need(depth < 40, 'native_tree_depth')
            kind = node['kind']
            if kind == 'dict':
                return {k: unpack(v, depth+1) for k, v in node['items'].items()}
            if kind in ('list', 'tuple'):
                values = [unpack(v, depth+1) for v in node['items']]
                return tuple(values) if kind == 'tuple' else values
            if kind == 'scalar':
                return node['value']
            need(kind == 'array' and node['key'] not in used, 'native_unique_leaf')
            used.add(node['key'])
            a = archive[node['key']]
            need(list(a.shape) == node['shape'] and a.dtype.str == node['storage_dtype'] and
                 a.nbytes == node['bytes'] and sha(a.tobytes()) == node['sha256'], 'native_leaf_SHA_ABI')
            total += a.nbytes
            need(total <= MAX_NATIVE_BYTES, 'native_tree_total_bound')
            if node['logical_dtype'] == 'bfloat16':
                import ml_dtypes
                need(a.dtype == np.uint16, 'BF16_original_bits')
                a = a.view(ml_dtypes.bfloat16)
            need(a.dtype.name == node['logical_dtype'], 'native_logical_dtype')
            nonfinite = int(np.count_nonzero(~np.isfinite(a)))
            need(nonfinite == node['nonfinite_elements'], 'native_nonfinite_count')
            need(allow_nonfinite or nonfinite == 0, 'native_logical_finite_dtype')
            return np.array(a, copy=True)

        value = unpack(doc['tree'])
        need(used | {'metadata_json'} == set(archive.files) and total == doc['native_bytes'], 'native_complete_roster')
    return value
