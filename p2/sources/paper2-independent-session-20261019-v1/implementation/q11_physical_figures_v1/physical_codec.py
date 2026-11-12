"""Bounded inverse of the pinned physical JSON writer, with payload verification."""
import gzip
import hashlib
import json
import math
import numpy as np


def need(ok, code):
    if not ok:
        raise ValueError(code)


def identity(raw):
    return {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def array_identity(value):
    value = np.asarray(value)
    return {"shape": list(value.shape), "dtype": value.dtype.str,
            **identity(value.tobytes(order="C"))}


def json_bytes(raw):
    def unique(items):
        result = {}
        for key, value in items:
            need(key not in result, "duplicate_JSON_key")
            result[key] = value
        return result
    return json.loads(raw, object_pairs_hook=unique,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite_JSON")))


def decode(value, counters, depth=0):
    need(depth <= 40, "bounded_physical_tree_depth")
    if isinstance(value, dict) and "__physical_array__" in value:
        need(set(value) == {"__physical_array__", "dtype", "shape", "payload_sha256", "values"}
             and value["__physical_array__"] is True, "exact_physical_array_wrapper")
        shape, dtype = value["shape"], np.dtype(value["dtype"])
        need(isinstance(shape, list) and len(shape) <= 8 and
             all(type(x) is int and 0 <= x <= 4096 for x in shape) and
             math.prod(shape) <= 4096 and dtype.kind in "biuf" and dtype.itemsize <= 8,
             "bounded_numeric_physical_array")
        array = np.asarray(value["values"], dtype=dtype)
        need(list(array.shape) == shape and np.isfinite(array).all(), "physical_array_shape_and_finite")
        need(hashlib.sha256(array.tobytes(order="C")).hexdigest() == value["payload_sha256"],
             "physical_array_payload_SHA")
        counters["arrays"] += 1
        counters["numeric_bytes"] += array.nbytes
        need(counters["arrays"] <= 10000 and counters["numeric_bytes"] <= 16*1024**2,
             "bounded_physical_numeric_total")
        return array
    if isinstance(value, dict):
        return {k: decode(v, counters, depth+1) for k, v in value.items()}
    if isinstance(value, list):
        need(len(value) <= 20000, "bounded_physical_sequence")
        return [decode(v, counters, depth+1) for v in value]
    need(value is None or type(value) in (str, int, float, bool), "physical_scalar_type")
    need(type(value) is not float or math.isfinite(value), "physical_scalar_finite")
    return value


def load(path):
    with gzip.open(path, "rb") as stream:
        raw = stream.read(32*1024**2+1)
    need(len(raw) <= 32*1024**2, "bounded_physical_JSON_inflation")
    counters = {"arrays": 0, "numeric_bytes": 0}
    value = decode(json_bytes(raw), counters)
    return value, {**counters, "uncompressed_bytes": len(raw)}
