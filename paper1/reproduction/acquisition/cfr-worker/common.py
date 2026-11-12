"""Small shared helpers for the cf-repair-v1 JD worker (no cloud, no model imports).

``difference`` and ``within`` are adapted from NSA/execution/qualification_worker.py:36-66
(SHA 8352f0d0...): exact-byte comparison plus absolute/RMS-relative statistics.
``write_json`` follows the closeout capsule atomic write (CO/execution/capsule/common.py:96-103).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np


class TechnicalFailure(RuntimeError):
    """A technical (engine/qualification) failure; retained as data, never retried."""

    def __init__(self, code, detail=None):
        super().__init__(code)
        self.code = code
        self.detail = detail or {}


def need(ok, code):
    if not ok:
        raise TechnicalFailure(code)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def sha_bytes(raw):
    return hashlib.sha256(raw).hexdigest()


def file_identity(path):
    h = hashlib.sha256()
    size = 0
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
            size += len(block)
    return {"bytes": size, "sha256": h.hexdigest()}


def array_id(value):
    """Typed identity of a numeric array (dtype string, shape, raw C-order bytes)."""
    a = np.asarray(value)
    if a.dtype.hasobject:
        raise TypeError("object arrays are not scientific values")
    return {"dtype": a.dtype.str, "shape": list(a.shape),
            "sha256": hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()}


def float64_le_sha(value):
    """SHA-256 over C-contiguous little-endian float64 bytes (I-1 array identity domain)."""
    a = np.ascontiguousarray(np.asarray(value), dtype="<f8")
    return hashlib.sha256(a.tobytes()).hexdigest()


def bf16_bits(value):
    """uint16 view of a BF16 array (native-adapter storage convention)."""
    a = np.ascontiguousarray(np.asarray(value))
    need(a.dtype.name == "bfloat16", "bf16_storage_requires_native_bfloat16")
    return a.view(np.uint16)


def to_f64(value):
    return np.asarray(value).astype(np.float64)


def safe_error(error):
    code = getattr(error, "code", None) or str(error)
    ok = isinstance(code, str) and 0 < len(code) <= 160 and all(ch.isalnum() or ch in "_-:.,/ " for ch in code)
    return {"exception_type": type(error).__name__, "code": code if ok else "unclassified_exception"}


def write_json(path, value):
    """Atomic, no-overwrite JSON write (part file, fsync, hard link)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".part")
    with part.open("xb") as handle:
        handle.write(canonical(value) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.link(part, path)
    part.unlink()


def jsonable(value):
    """Convert numpy scalars/arrays (small) to JSON-safe Python values; NaN/inf -> None."""
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        f = float(value)
        return f if math.isfinite(f) else None
    return value


def _bitwise_differs(a, b):
    """Per-element bit inequality (treats +0/-0 and NaN payloads as different bits)."""
    if a.size == 0:
        return np.zeros(a.shape, bool)
    width = a.dtype.itemsize
    ua = np.ascontiguousarray(a).view(np.uint8).reshape(a.size, width)
    ub = np.ascontiguousarray(b).view(np.uint8).reshape(b.size, width)
    return (ua != ub).any(axis=1).reshape(a.shape)


def difference(left, right):
    """Exact-byte comparison with absolute/RMS statistics (qualification_worker._difference)."""
    a, b = np.asarray(left), np.asarray(right)
    if a.shape != b.shape or a.dtype != b.dtype:
        return {"comparable": False, "exact": False, "left": array_id(a), "right": array_id(b)}
    af, bf = a.astype(np.float64), b.astype(np.float64)
    finite = bool(np.isfinite(af).all() and np.isfinite(bf).all())
    distance = np.abs(af - bf) if finite else np.zeros(1)
    reference_rms = float(np.sqrt(np.mean(np.square(af)))) if (af.size and finite) else 0.0
    max_abs = float(distance.max(initial=0)) if finite else None
    bit_differs = _bitwise_differs(a, b)
    # Elements whose bits differ only by the sign of zero (+0 vs -0); diagnostic only.
    signed_zero_only = int(np.count_nonzero(bit_differs & (af == 0) & (bf == 0))) if a.size else 0
    return {"comparable": True, "exact": bool(not bit_differs.any()),
            "different_bit_elements": int(np.count_nonzero(bit_differs)),
            "signed_zero_only_elements": signed_zero_only,
            "finite": finite, "max_abs": max_abs,
            "mean_abs": float(distance.mean()) if (finite and distance.size) else None,
            "reference_rms": reference_rms,
            "left": array_id(a), "right": array_id(b)}


def within(stats, tolerance):
    """Tolerance verdict; default tolerance is exact bytes (TECH-CONTRACT-v1 7.0)."""
    if tolerance is None or tolerance == "exact_bytes":
        return {"pass": bool(stats.get("exact")), "rule": "exact_bytes"}
    allowed = float(tolerance["abs"]) + float(tolerance["rel_rms"]) * float(stats.get("reference_rms") or 0.0)
    ok = bool(stats.get("comparable")) and stats.get("max_abs") is not None and stats["max_abs"] <= allowed
    return {"pass": ok, "rule": "abs_plus_rel_rms", "allowed_max_abs": allowed,
            "observed_max_abs": stats.get("max_abs"), "tolerance": dict(tolerance)}
