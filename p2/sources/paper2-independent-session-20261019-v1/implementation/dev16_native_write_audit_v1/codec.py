"""Bounded typed reads; no policy, simulator, network or fitting."""
from dataclasses import dataclass
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import zipfile
import numpy as np

HERE = Path(__file__).resolve().parent


def need(ok, code):
    if not ok:
        raise ValueError(code)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def json_read(raw):
    def pairs(items):
        out = {}
        for key, value in items:
            need(key not in out, "duplicate_JSON_key")
            out[key] = value
        return out
    def nonfinite(_):
        raise ValueError("nonfinite_JSON")
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=nonfinite)


@dataclass
class Leaf:
    storage: np.ndarray
    logical: str

    def numeric(self):
        if self.logical == "bfloat16":
            return (self.storage.astype(np.uint32) << 16).view(np.float32)
        return self.storage

    def select(self, index):
        return Leaf(np.asarray(self.storage[index]), self.logical)

    def identity(self):
        a = self.storage
        return {"shape": list(a.shape), "logical_dtype": self.logical,
                "storage_dtype": str(a.dtype), "bytes": a.nbytes,
                "payload_sha256": sha(a.tobytes(order="C"))}

    def input_identity(self):
        a = self.storage
        header = {"shape": list(a.shape), "dtype": "<V2" if self.logical == "bfloat16" else a.dtype.str,
                  "logical_dtype": self.logical, "bytes": a.nbytes}
        return {**header, "sha256": sha(json.dumps(header, sort_keys=True).encode() +
                                       b"\0" + a.tobytes(order="C"))}


def from_array(a):
    a = np.asarray(a)
    need(not a.dtype.hasobject and np.isfinite(a).all(), "finite_numeric_array")
    return Leaf(a, a.dtype.name)


def zip_guard(path, cap):
    with zipfile.ZipFile(path) as z:
        entries = z.infolist()
        names = [i.filename for i in entries]
        need(len(names) == len(set(names)) and 0 < len(names) <= 4096,
             "bounded_unique_NPZ_entries")
        need(all(n.endswith(".npy") and not n.startswith("/") and ".." not in n.split("/")
                 for n in names), "safe_NPY_names")
        need(all(not (i.flag_bits & 1) for i in entries) and
             sum(i.file_size for i in entries) <= cap, "bounded_uncompressed_NPZ")


def typed(path, ref, *, synthetic=False):
    need(synthetic or sys.platform.startswith("linux"), "real_arrays_cloud_Linux_only")
    zip_guard(path, 16 * 1024**2)
    pin = json_read((HERE / "CODEC-PINS.json").read_bytes())["files"][0]
    p = HERE / pin["path"]
    raw = p.read_bytes()
    need(len(raw) == pin["bytes"] and sha(raw) == pin["sha256"], "pinned_reader_identity")
    spec = importlib.util.spec_from_file_location("_dev16_bridge_retained_reader", p)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    reader = module.Record(path, ref)
    leaves = {}
    try:
        # Parse again with duplicate/nonfinite rejection, then verify every typed
        # payload, including stored leaves not selected for scientific features.
        tree = json_read(reader.z["__schema__"].tobytes())
        need(set(tree) == {"metadata", "arrays"}, "typed_root_schema")
        declared = set()
        def walk(value, keys=()):
            if isinstance(value, dict) and "__array__" in value:
                need(set(value) == {"__array__", "shape", "logical_dtype", "storage_dtype",
                                   "bytes", "payload_sha256"}, "typed_descriptor_schema")
                need(value["__array__"] not in declared, "unique_typed_payload")
                declared.add(value["__array__"])
                a, descriptor = reader.leaf(*keys)
                leaves[keys] = Leaf(a, descriptor["logical_dtype"])
            elif isinstance(value, dict):
                for k, v in value.items():
                    walk(v, keys + (k,))
            else:
                need(False, "arrays_tree_contains_only_typed_leaves")
        walk(tree["arrays"])
        need(declared == set(reader.z.files) - {"__schema__"}, "no_orphan_NPZ_payload")
        return reader.meta, leaves
    finally:
        reader.close()


def flat_npz(path, *, synthetic=False):
    need(synthetic or sys.platform.startswith("linux"), "real_arrays_cloud_Linux_only")
    zip_guard(path, 256 * 1024**2)
    with np.load(path, allow_pickle=False) as z:
        result = {}
        for k in z.files:
            a = np.asarray(z[k])
            need(not a.dtype.hasobject and a.dtype.kind in "biufSU", "safe_flat_array_dtype")
            need(a.dtype.kind in "SU" or np.isfinite(a).all(), "finite_flat_array")
            result[k] = np.array(a, copy=True)
        return result
