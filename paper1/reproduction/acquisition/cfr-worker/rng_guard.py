"""Process-global RNG isolation (TECH-CONTRACT-v1 3.5; test T12).

Adapted from P/parallel-collection-runtime-v6/production.py:63-96 (``invariant_call``) and
A/adapter.py:57-60,151,165: save the global NumPy and Python RNG before a neural call or a
counterfactual render, assert they are unchanged afterwards, restore on exception or
change, and keep an audit record. A change is a technical failure (never silently restored
and continued).
"""

from __future__ import annotations

import copy
import hashlib
import random

import numpy as np

from .common import TechnicalFailure


def rng_state():
    return {"numpy": copy.deepcopy(np.random.get_state()), "python": copy.deepcopy(random.getstate())}


def rng_fingerprint(state=None):
    state = rng_state() if state is None else state
    n, p = state["numpy"], state["python"]
    return {"numpy_algorithm": n[0], "numpy_keys_sha256": hashlib.sha256(np.asarray(n[1]).tobytes()).hexdigest(),
            "numpy_position": int(n[2]), "numpy_has_gauss": int(n[3]), "numpy_cached_gaussian_hex": float(n[4]).hex(),
            "python_sha256": hashlib.sha256(repr(p).encode()).hexdigest()}


def restore(state):
    np.random.set_state(copy.deepcopy(state["numpy"]))
    random.setstate(copy.deepcopy(state["python"]))


class RngAudit:
    """Collects one record per guarded call; ``summary`` feeds test T12."""

    def __init__(self):
        self.records = []

    def call(self, label, fn, *args, **kwargs):
        saved = rng_state()
        before = rng_fingerprint(saved)
        error = None
        try:
            value = fn(*args, **kwargs)
        except BaseException as caught:  # noqa: BLE001 - restore then re-raise
            error = caught
            value = None
        after = rng_fingerprint()
        changed = before != after
        if changed or error is not None:
            restore(saved)
        self.records.append({"label": label, "changed": changed, "exception_type": type(error).__name__ if error else None})
        if error is not None:
            raise error
        if changed:
            raise TechnicalFailure("global_rng_changed_during_" + label, {"before": before, "after": after})
        return value

    def summary(self):
        labels = {}
        for record in self.records:
            row = labels.setdefault(record["label"], {"calls": 0, "changed": 0, "exceptions": 0})
            row["calls"] += 1
            row["changed"] += int(record["changed"])
            row["exceptions"] += int(record["exception_type"] is not None)
        return {"guarded_calls": len(self.records), "changed_calls": sum(r["changed"] for r in self.records),
                "by_label": labels, "pass": all(not r["changed"] for r in self.records)}
