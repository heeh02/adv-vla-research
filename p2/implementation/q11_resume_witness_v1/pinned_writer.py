"""Bounded JSON evidence writer; retains actual numerical dtype and payload SHA."""
import gzip
import hashlib
import json
import math
from pathlib import Path
import numpy as np


def public_tree(value):
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject or value.dtype.kind not in "biuf" or value.size > 4096:
            raise ValueError("bounded_physical_numeric_array_required")
        if not np.isfinite(value).all():
            raise ValueError("nonfinite_physical_array")
        return {"__physical_array__": True, "dtype": value.dtype.str, "shape": list(value.shape),
                "payload_sha256": hashlib.sha256(value.tobytes(order="C")).hexdigest(),
                "values": value.tolist()}
    if isinstance(value, np.generic):
        return public_tree(np.asarray(value))
    if isinstance(value, dict):
        if not all(isinstance(k, str) for k in value):
            raise ValueError("named_physical_fields_required")
        return {k: public_tree(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        if len(value) > 20000:
            raise ValueError("bounded_physical_sequence")
        return [public_tree(v) for v in value]
    if type(value) is float and not math.isfinite(value):
        raise ValueError("nonfinite_physical_scalar")
    if value is not None and type(value) not in (str, int, float, bool):
        raise ValueError("unsupported_physical_value")
    return value


def encoded(value):
    return (json.dumps(public_tree(value), ensure_ascii=False, sort_keys=True,
                       separators=(",", ":"), allow_nan=False) + "\n").encode()


class Events:
    def __init__(self, path, *, check, max_uncompressed=64*1024**2, max_compressed=16*1024**2):
        self.path = Path(path)
        if self.path.exists():
            raise ValueError("new_event_file_required")
        # Concatenated gzip members permit checking the compressed size BEFORE
        # every write; a rejected event never creates an over-budget file.
        self.stream = self.path.open("xb")
        self.check = check
        self.limit = max_uncompressed
        self.compressed_limit = max_compressed
        self.bytes = self.compressed_bytes = self.count = 0
        self.closed = False

    def emit(self, event, metadata):
        self.check()
        raw = encoded({"sequence": self.count + 1, "event": event, "metadata": metadata})
        if len(raw) > 2*1024**2 or self.bytes + len(raw) > self.limit:
            raise ValueError("physical_event_byte_cap")
        compressed = gzip.compress(raw, mtime=0)
        if self.compressed_bytes + len(compressed) > self.compressed_limit:
            raise ValueError("physical_event_compressed_cap")
        self.stream.write(compressed)
        self.stream.flush()
        self.count += 1
        self.bytes += len(raw)
        self.compressed_bytes += len(compressed)
        return "events:" + str(self.count)

    def close(self):
        if not self.closed:
            self.closed = True
            self.stream.close()


def write_new(path, value, *, limit=8*1024**2, compressed=False):
    path = Path(path)
    raw = encoded(value)
    if len(raw) > 32*1024**2:
        raise ValueError("physical_document_uncompressed_cap")
    body = gzip.compress(raw, mtime=0) if compressed else raw
    if len(body) > limit:
        raise ValueError("physical_document_output_cap")
    with path.open("xb") as stream:
        stream.write(body)
    return {"path": path.name, "bytes": len(body), "sha256": hashlib.sha256(body).hexdigest()}
