"""Fail-closed CPU adapter for the already archived 24-query Grams.

This file never restores objects, requests cloud jobs, writes permissions, or
loads a model. The caller supplies an exact, cloud-local stage and source
manifest. Actual array processing is Linux-only. Unit tests use synthetic
arrays through the independent, non-I/O analyze_query function.
"""
from pathlib import Path, PurePosixPath
import argparse
import gzip
import hashlib
import importlib.util
import io
import json
import math
import os
import stat
import sys
import time
import zipfile

import numpy as np

HERE = Path(__file__).resolve().parent
PARENT_RUN = 'jinwen-sru-natural-reading-20261016-v1'
PINNED_METADATA = {
    'PARENT-RESULT.json': (21725, '4990156b6f20c3da37719e13dfee01e052608f52b6095f9925e5a85eceb91480'),
    'PARENT-MANIFEST.json': (43361, 'e852ff4392168203d53310bc4f4b61e74f941869b4eaef77ebef679b9755ee44'),
    'PARENT-SOURCE.json': (159812, 'db14472ddf28bd5792d456b2c0403b5d7e471df73093749328302f73bd42f274'),
    'PARENT-SUMMARY.json': (88120, 'b4c3a2d6af8ff3a1e55126400e68c3c32b9d17ebea82252afa3d6c87d7673dcc'),
}
FIELDS = ('raw_gram_upper', 'masses', 'corner_defined', 'projection_gram_upper', 'group_corner_defined')
ABI = {
    'raw_gram_upper': ((2, 3, 3, 8, 10, 10), 'float64'),
    'masses': ((2, 3, 3, 2, 8, 10, 3), 'float64'),
    'corner_defined': ((2, 3, 3, 4, 8, 10), 'bool'),
    'projection_gram_upper': ((2, 3, 3, 2, 2, 8, 10, 10), 'float64'),
    'group_corner_defined': ((2, 3, 3, 4, 8, 10, 3), 'bool'),
}
LAYERS = (0, 8, 17)
PAIRS = ((0, 5), (5, 9), (0, 9))
TASKS = {f'CCONF-T{t:02d}': tuple(f'SDRU-CCONF-T{t:02d}-V{v:05d}-I00-P{p}' for p in ('C', 'T'))
         for t, v in ((6, 700), (7, 774), (8, 1029), (9, 827))}
EXPECTED = {(root, q) for roots in TASKS.values() for root in roots for q in (11, 16, 22)}
SAVED_FIELDS = ('prefix_mass', 'scales', 'scale_defined', 'scale_identified',
                'target_energy', 'baseline_SSE', 'baseline_defined')
OUTPUT_LIMITS = {'output_bytes': 64 * 1024**2, 'output_raw_array_bytes': 48 * 1024**2,
                 'single_output_bytes': 4 * 1024**2}


def need(ok, code):
    if not ok:
        raise ValueError(code)


def identity(raw):
    return {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


def same_identity(ref, other):
    return all(ref[k] == other[k] for k in ('bytes', 'sha256'))


def safe_json(value):
    if isinstance(value, np.ndarray):
        return safe_json(value.tolist())
    if isinstance(value, np.generic):
        return safe_json(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: safe_json(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [safe_json(v) for v in value]
    return value


def encode(value):
    return (json.dumps(safe_json(value), ensure_ascii=False, sort_keys=True,
                       separators=(',', ':'), allow_nan=False) + '\n').encode()


def relative(value):
    need(type(value) is str and value and '\\' not in value, 'relative_path_string')
    path = PurePosixPath(value)
    need(not path.is_absolute() and str(path) == value and
         all(part not in ('', '.', '..') for part in value.split('/')), 'canonical_relative_path')
    return path


def read_exact(base, name, ref, *, cap):
    """Read a regular owned-stage file once, with no symlink or size race."""
    base = Path(base)
    need(base.is_dir() and not base.is_symlink(), 'regular_stage_directory')
    parts = relative(name).parts
    path = base
    for part in parts:
        path = path / part
        need(not path.is_symlink(), 'no_stage_symlink')
    need(type(ref['bytes']) is int and 0 < ref['bytes'] <= cap, 'bounded_input_size')
    flags = os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0)
    fd = os.open(path, flags)
    with os.fdopen(fd, 'rb') as stream:
        before = os.fstat(stream.fileno())
        need(stat.S_ISREG(before.st_mode) and before.st_size == ref['bytes'], 'regular_exact_input_size')
        raw = stream.read(ref['bytes'] + 1)
        after = os.fstat(stream.fileno())
    need((before.st_ino, before.st_size, before.st_mtime_ns) ==
         (after.st_ino, after.st_size, after.st_mtime_ns), 'input_unchanged_during_read')
    need(same_identity(identity(raw), ref), 'wholefile_SHA_and_bytes')
    return raw


def verify_source_manifest(path, expected_sha256):
    path = Path(path)
    need(len(expected_sha256) == 64 and all(c in '0123456789abcdef' for c in expected_sha256), 'source_manifest_SHA_required')
    need(path.is_file() and not path.is_symlink() and path.stat().st_size <= 1024**2, 'small_regular_source_manifest')
    raw = path.read_bytes()
    need(hashlib.sha256(raw).hexdigest() == expected_sha256, 'new_source_manifest_SHA')
    manifest = json.loads(raw)
    need(manifest['schema'] == 'PAPER2-mass-template-source-manifest-v1', 'new_source_manifest_schema')
    rows = manifest['files']
    need(len({r['path'] for r in rows}) == len(rows), 'unique_source_files')
    by = {r['path']: r for r in rows}
    required = {'adapter.py', 'kernel.py', 'SOURCE-INPUT-INVENTORY.json'}
    need(required <= set(by), 'all_runtime_sources_pinned')
    need(path.parent.resolve() == HERE, 'executed_adapter_inside_source_bundle')
    loaded = {name: read_exact(path.parent, name, by[name], cap=2 * 1024**2) for name in required}
    return manifest, loaded, identity(raw)


def qualify_context(inventory, stage):
    need(inventory['schema'] == 'PAPER2-saved-Gram-input-inventory-v1' and inventory['synthetic'] is False,
         'actual_fixed_inventory')
    need(inventory['source_run_id'] == PARENT_RUN and inventory['selected_fields'] == list(FIELDS), 'exact_parent_and_fields')
    need(set(inventory['metadata']) == set(PINNED_METADATA), 'four_exact_parent_metadata_files')
    documents = {}
    for name, (size, digest) in PINNED_METADATA.items():
        ref = inventory['metadata'][name]
        need((ref['bytes'], ref['sha256']) == (size, digest), 'historical_parent_identity_pinned')
        documents[name] = json.loads(read_exact(stage, name, ref, cap=2 * 1024**2))
    result, manifest, source, summary = (documents[n] for n in PINNED_METADATA)
    need(result['run_id'] == PARENT_RUN and result['scientific_complete'] is True and
         result['model_calls'] == result['new_env_steps'] == 0, 'actual_complete_parent_result')
    need(summary['analysis_complete'] is True and summary['synthetic'] is False and
         summary['qualified_queries'] == 24 and summary['qualified_comparisons'] == 432 and
         summary['missing_queries'] == summary['numerical_failures'] == 0, 'actual_complete_parent_summary')
    need(source['schema'] == 'CONF-natural-reading-use-source-v1' and
         source['source_context']['source_qualified'] is True and
         source['source_context']['qualification_proofs']['CONF_comparison_posthoc'] is True,
         'original_source_qualified_and_CONF_posthoc')
    cloud = {r['relative']: r for r in result['cloud_objects']}
    need(len(cloud) == len(result['cloud_objects']), 'unique_parent_cloud_objects')
    listed = {r['path']: r for r in manifest['files']}
    need(len(listed) == len(manifest['files']), 'unique_parent_manifest_paths')
    for name, ref in inventory['metadata'].items():
        if name == 'PARENT-RESULT.json':
            continue
        row = cloud[ref['parent_relative']]
        need(same_identity(ref, row) and ref['uri'] == row['uri'], 'metadata_in_parent_RESULT')
        if name != 'PARENT-MANIFEST.json':
            need(same_identity(ref, listed[ref['parent_relative'].removeprefix('science/')]), 'metadata_in_science_MANIFEST')
    selected = {(q['root_id'], q['query_index']): q for q in inventory['queries']}
    indexed = {(q['root_id'], q['query_index']): q for q in source['queries']}
    need(len(selected) == len(indexed) == len(inventory['queries']) == len(source['queries']) == 24 and
         set(selected) == set(indexed) == EXPECTED, 'all24_original_queries_no_duplicates')
    for key, ref in selected.items():
        q = indexed[key]
        need(ref['root_id'] in TASKS[ref['task_id']] and
             all(ref[k] == q[k] for k in ('root_id', 'query_index', 'task_id', 'source_identity')),
             'original_task_query_source_identity')
        need(q['initial_noise_same_CT'] is True, 'original_CT_noise_proof')
        name = q['array_file']['path']
        need(ref['path'] == name and name.startswith('QUERY-') and name.endswith('.npz'), 'original_query_filename')
        cloud_ref = cloud['science/analysis/' + name]
        need(same_identity(ref, q['array_file']) and same_identity(ref, cloud_ref) and
             same_identity(ref, listed['analysis/' + name]) and ref['uri'] == cloud_ref['uri'],
             'RESULT_MANIFEST_SOURCE_query_identity_chain')
        for field, (shape, dtype) in ABI.items():
            desc = q['arrays'][field]
            need(desc['shape'] == list(shape) and desc['dtype'] == dtype and
                 desc['bytes'] == math.prod(shape) * np.dtype(dtype).itemsize, 'original_field_ABI_' + field)
    need(sum(q['bytes'] for q in selected.values()) == 25923751 and
         inventory['budgets']['total_input_body_bytes'] == 26236769 and
         inventory['budgets']['selected_decoded_bytes_per_query'] == 668160, 'exact_input_budget')
    return documents, indexed


def load_query(stage, ref, query_meta, *, on_verified=None):
    raw = read_exact(stage, ref['path'], ref, cap=2 * 1024**2)
    if on_verified is not None:
        on_verified(ref)
    descriptors = query_meta['arrays']
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        infos = archive.infolist()
        names = [r.filename for r in infos]
        need(len(set(names)) == len(names) and set(names) == {f + '.npy' for f in descriptors}, 'original_unique_NPZ_members')
        for item in infos:
            desc = descriptors[item.filename[:-4]]
            need(item.file_size <= desc['bytes'] + 4096 and not item.flag_bits & 1, 'NPZ_member_expansion_bound')
    arrays = {}
    with np.load(io.BytesIO(raw), allow_pickle=False) as loaded:
        for field in FIELDS:
            desc = descriptors[field]
            array = loaded[field]
            shape, dtype = ABI[field]
            need(array.shape == shape and array.dtype == np.dtype(dtype) and array.nbytes == desc['bytes'],
                 'typed_array_shape_dtype_bytes_' + field)
            need(hashlib.sha256(array.tobytes(order='C')).hexdigest() == desc['payload_sha256'], 'typed_payload_SHA_' + field)
            array.setflags(write=False)
            arrays[field] = array
    need(sum(a.nbytes for a in arrays.values()) == 668160, 'selected_decode_byte_budget')
    return arrays


def analyze_query(arrays, query_meta, kernel, *, deadline=None):
    """Already verified arrays; no file reads and no source identity invention."""
    rows, saved, checks = [], {}, []
    identity_tags = {k: query_meta[k] for k in ('task_id', 'root_id', 'query_index', 'source_identity')}
    comparisons = 0
    for ci, condition in enumerate(('C', 'T')):
        for li, layer in enumerate(LAYERS):
            for pi, pair in enumerate(PAIRS):
                if deadline is not None:
                    deadline.check()
                idx = (ci, li, pi)
                value = kernel.analyze_saved_grams(arrays['raw_gram_upper'][idx], arrays['masses'][idx],
                    arrays['corner_defined'][idx], projection_gram_upper=arrays['projection_gram_upper'][idx],
                    group_corner_defined=arrays['group_corner_defined'][idx], flow_pair=pair)
                tags = {**identity_tags, 'condition': condition, 'layer_index': layer,
                        'flow_from': pair[0], 'flow_to': pair[1]}
                rows.extend({**r, **tags, 'comparison_role': comparison_role({**r, **tags})} for r in value['rows'])
                checks.append({'tags': tags, 'checks': value['checks']})
                prefix = f'{condition}_L{layer}_j{pair[0]}to{pair[1]}__'
                for name, array in value['arrays'].items():
                    if name not in SAVED_FIELDS and not name.startswith('oracle_'):
                        continue
                    need(type(name) is str and name.replace('_', '').isalnum(), 'safe_derived_field_name')
                    a = np.asarray(array)
                    need(not a.dtype.hasobject and a.dtype.kind in 'biuf', 'numeric_derived_array')
                    saved[prefix + name] = a
                comparisons += 1
    need(comparisons == 18, 'all18_original_views_per_query')
    return {'rows': rows, 'arrays': saved, 'checks': checks, 'comparisons': comparisons}


def comparison_role(row):
    main = (row['condition'], row['layer_index'], row['flow_from'], row['flow_to']) == ('C', 17, 0, 9)
    if main and row['view'] == 'full':
        return 'primary' if row['window'] == 'full10' else 'support_window'
    return 'descriptive_control'


class Deadline:
    def __init__(self, epoch, max_seconds):
        need(isinstance(epoch, (float, int)) and math.isfinite(epoch) and epoch > time.time(),
             'future_absolute_science_deadline')
        need(0 < max_seconds <= 900, 'bounded_science_runtime')
        self.epoch = epoch
        self.start = time.monotonic()
        self.monotonic_end = self.start + max_seconds

    def check(self):
        need(time.time() < self.epoch and time.monotonic() < self.monotonic_end,
             'absolute_or_runtime_science_deadline')


class Output:
    def __init__(self, directory, limits, deadline=None):
        self.path = Path(directory)
        self.path.mkdir(parents=True, exist_ok=False)
        self.limits = limits
        self.files = []
        self.raw_bytes = 0
        self.deadline = deadline

    def put(self, name, raw, *, finalizing=False):
        if self.deadline is not None and not finalizing:
            self.deadline.check()
        need('/' not in name and name not in ('.', '..'), 'owned_output_basename')
        need(len(raw) <= self.limits['single_output_bytes'], 'single_output_budget')
        if finalizing:
            need(name in ('FAILURE.json', 'SUMMARY.json', 'MANIFEST.json', 'FINALIZATION-FAILURE.json'),
                 'failure_reserve_only_for_final_receipts')
        cap = self.limits['output_bytes'] if finalizing else self.limits['output_bytes'] - 65536
        need(sum(r['bytes'] for r in self.files) + len(raw) <= cap,
             'total_output_budget_with_failure_reserve')
        path = self.path / name
        with path.open('xb') as stream:
            stream.write(raw)
        ref = {'path': name, **identity(raw)}
        self.files.append(ref)
        return ref

    def arrays(self, name, arrays):
        if self.deadline is not None:
            self.deadline.check()
        size = sum(a.nbytes for a in arrays.values())
        need(self.raw_bytes + size <= self.limits['output_raw_array_bytes'], 'derived_raw_array_budget')
        buf = io.BytesIO()
        np.savez_compressed(buf, **arrays)
        ref = self.put(name, buf.getvalue())
        self.raw_bytes += size
        return ref


def aggregate_rows(rows, kernel):
    """Fixed 3-query root means, then equal roots; missing values stay null."""
    key_fields = ('condition', 'layer_index', 'flow_from', 'flow_to', 'view', 'window')
    indexed = {}
    for row in rows:
        key = tuple(row[k] for k in key_fields) + (row['root_id'], row['query_index'])
        need(key not in indexed and (row['root_id'], row['query_index']) in EXPECTED,
             'unique_original_query_aggregate_row')
        indexed[key] = row
    roots = sorted({r for rr in TASKS.values() for r in rr})
    output = {'schema': 'PAPER2-fixed-membership-aggregate-v1',
              'weighting': '3 planned queries equally within each root; roots equally within task or population; absolute means before ratio; any missing operand stays null',
              'root': [], 'task': [], 'population': []}
    for condition in ('C', 'T'):
        for layer in LAYERS:
            for a, b in PAIRS:
                for view in kernel.VIEWS:
                    for window, _, _ in kernel.WINDOWS:
                        tags = dict(condition=condition, layer_index=layer, flow_from=a, flow_to=b,
                                    view=view, window=window)
                        tags['comparison_role'] = comparison_role(tags)
                        identity = tuple(tags[k] for k in key_fields)
                        missing = {'view': view, 'window': window, 'target_delta_energy': None,
                                   'full_target_delta_energy': None,
                                   'baselines': {name: {'SSE': None} for name in kernel.BASELINES}}
                        per_root = {}
                        for root in roots:
                            planned = [indexed.get(identity + (root, q), missing) for q in (11, 16, 22)]
                            result = kernel.aggregate_absolute(planned)
                            observed = sum(identity + (root, q) in indexed for q in (11, 16, 22))
                            result.update(tags, aggregation_level='root', aggregation_id=root,
                                          planned_queries=3, observed_queries=observed,
                                          planned_roots=1, observed_roots=int(observed == 3))
                            # Signed MAI terms remain in perquery rows. This aggregate
                            # is an energy/SSE estimand, never survivor-weighted.
                            output['root'].append(result)
                            per_root[root] = result
                        for level, memberships in (('task', TASKS), ('population', {'all': tuple(roots)})):
                            for label, members in memberships.items():
                                result = kernel.aggregate_absolute([per_root[r] for r in members])
                                result.update(tags, aggregation_level=level, aggregation_id=label,
                                              planned_queries=3 * len(members),
                                              observed_queries=sum(per_root[r]['observed_queries'] for r in members),
                                              planned_roots=len(members),
                                              observed_roots=sum(per_root[r]['observed_roots'] for r in members))
                                output[level].append(result)
    need(len(output['root']) == 2160 and len(output['task']) == 1080 and
         len(output['population']) == 270, 'fixed_all_condition_aggregate_support')
    return output


def run(stage, output, source_manifest, source_manifest_sha256, *, deadline_epoch, max_runtime_seconds=900):
    need(sys.platform.startswith('linux'), 'actual_arrays_cloud_Linux_only')
    need(Path(stage).resolve() != Path(output).resolve(), 'input_output_separate')
    out = Output(output, OUTPUT_LIMITS)
    try:
        deadline = Deadline(deadline_epoch, max_runtime_seconds)
        out.deadline = deadline
        _, loaded, source_ref = verify_source_manifest(source_manifest, source_manifest_sha256)
        inventory = json.loads(loaded['SOURCE-INPUT-INVENTORY.json'])
        need(all(inventory['limits'][k] == v for k, v in OUTPUT_LIMITS.items()), 'fixed_output_caps')
        documents, indexed = qualify_context(inventory, stage)
        deadline.check()
        spec = importlib.util.spec_from_file_location('_paper2_pinned_saved_gram_kernel', HERE / 'kernel.py')
        kernel = importlib.util.module_from_spec(spec)
        # Execute verified bytes rather than reopening a potentially replaced source.
        exec(compile(loaded['kernel.py'], str(HERE / 'kernel.py'), 'exec'), kernel.__dict__)
    except Exception as error:
        failure = {'schema': 'PAPER2-saved-Gram-preflight-failure-v1', 'analysis_complete': False,
            'stage': 'preflight', 'error_type': type(error).__name__, 'error': str(error)[:240],
            'requested_source_manifest_sha256': source_manifest_sha256, 'deadline_epoch': deadline_epoch,
            'completed_queries': 0, 'completed_comparisons': 0, 'failures': 1,
            'input_identity_qualified': False, 'new_model_calls': 0, 'new_env_steps': 0, 'cloud_calls': 0}
        out.put('FAILURE.json', encode(failure), finalizing=True)
        out.put('SUMMARY.json', encode(failure), finalizing=True)
        out.put('MANIFEST.json', encode({'analysis_complete': False, 'files': out.files.copy()}), finalizing=True)
        return failure
    completed, failures = [], []
    verified_query_reads, all_rows = [], []
    for ref in inventory['queries']:
        key = (ref['root_id'], ref['query_index'])
        try:
            deadline.check()
            arrays = load_query(stage, ref, indexed[key], on_verified=verified_query_reads.append)
            analyzed = analyze_query(arrays, indexed[key], kernel, deadline=deadline)
            number = ref['path'].removeprefix('QUERY-').removesuffix('.npz')
            derived = out.arrays('DERIVED-' + number + '.npz', analyzed['arrays'])
            detail = out.put('QUERY-' + number + '.json.gz', gzip.compress(encode({
                'input': ref, 'source_manifest': source_ref, 'derived': derived,
                'rows': analyzed['rows'], 'checks': analyzed['checks'],
                'array_descriptors': {k: {'shape': list(a.shape), 'dtype': str(a.dtype), 'bytes': a.nbytes,
                    'payload_sha256': hashlib.sha256(a.tobytes(order='C')).hexdigest()} for k, a in analyzed['arrays'].items()},
                'posthoc_reuse': True, 'signed_vector_reconstructed': False, 'native_W_inference': False}), mtime=0))
            completed.append({'root_id': key[0], 'query_index': key[1], 'comparisons': analyzed['comparisons'],
                              'detail': detail, 'derived': derived})
            all_rows.extend(analyzed['rows'])
            del arrays, analyzed
        except Exception as error:
            failures.append({'root_id': key[0], 'query_index': key[1], 'error_type': type(error).__name__,
                             'error': str(error)[:240]})
            break  # No silent skipping, retry, survivor aggregation, or scope expansion.
    aggregate_ref = None
    try:
        deadline.check()
        aggregates = aggregate_rows(all_rows, kernel)
        aggregate_ref = out.put('AGGREGATES.json.gz', gzip.compress(encode(aggregates), mtime=0))
    except Exception as error:
        failures.append({'stage': 'aggregation_or_write', 'error_type': type(error).__name__,
                         'error': str(error)[:240]})
    summary = {'schema': 'PAPER2-saved-Gram-run-v1', 'analysis_complete': len(completed) == 24 and not failures,
        'synthetic': False, 'source_manifest': source_ref, 'input_inventory': identity(loaded['SOURCE-INPUT-INVENTORY.json']),
        'parent_run_id': PARENT_RUN, 'parent_metadata': inventory['metadata'], 'completed': completed,
        'failures': failures, 'planned_queries': 24, 'completed_queries': len(completed),
        'planned_comparisons': 432, 'completed_comparisons': sum(q['comparisons'] for q in completed),
        'missing_queries': [{'root_id': r, 'query_index': q} for r, q in sorted(EXPECTED - {(x['root_id'], x['query_index']) for x in completed})],
        'planned_input_body_bytes': 26236769, 'verified_local_input_body_bytes': 313018 + sum(q['bytes'] for q in verified_query_reads),
        'verified_query_reads': verified_query_reads, 'failed_unverified_read_bytes': 'not_measured' if failures else 0,
        'elapsed_seconds': time.monotonic() - deadline.start, 'deadline_epoch': deadline_epoch,
        'max_runtime_seconds': max_runtime_seconds, 'derived_raw_array_bytes': out.raw_bytes,
        'aggregates': aggregate_ref,
        'new_model_calls': 0, 'new_env_steps': 0, 'new_fit': False, 'cloud_calls': 0,
        'input_deletions': 0, 'posthoc_CONF_reuse': True,
        'scope': 'Energy/SSE/cross statistics identifiable from saved per-head Grams. No signed vector reconstruction, WO, native W, new behavior, or new confirmation.'}
    try:
        if failures:
            out.put('FAILURE.json', encode({'source_manifest': source_ref, 'failures': failures,
                                          'partial_outputs_retained': True}), finalizing=True)
        out.put('SUMMARY.json', encode(summary), finalizing=True)
        manifest = {'schema': 'PAPER2-saved-Gram-output-manifest-v1', 'source_manifest': source_ref,
                    'analysis_complete': summary['analysis_complete'], 'files': out.files.copy()}
        out.put('MANIFEST.json', encode(manifest), finalizing=True)
    except Exception as error:
        summary['analysis_complete'] = False
        failure = {'stage': 'final_receipts', 'error_type': type(error).__name__, 'error': str(error)[:240]}
        summary['failures'].append(failure)
        out.put('FINALIZATION-FAILURE.json', encode({'analysis_complete': False,
            'source_manifest': source_ref, 'failure': failure, 'partial_outputs_retained': True}), finalizing=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--source-manifest', type=Path, required=True)
    parser.add_argument('--source-manifest-sha256', required=True)
    parser.add_argument('--deadline-epoch', type=float, required=True)
    parser.add_argument('--max-runtime-seconds', type=int, default=900)
    args = parser.parse_args()
    summary = run(args.stage, args.out, args.source_manifest, args.source_manifest_sha256,
                  deadline_epoch=args.deadline_epoch, max_runtime_seconds=args.max_runtime_seconds)
    print(json.dumps({k: summary[k] for k in ('analysis_complete', 'completed_queries', 'completed_comparisons', 'failures')}))
    return 0 if summary['analysis_complete'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
