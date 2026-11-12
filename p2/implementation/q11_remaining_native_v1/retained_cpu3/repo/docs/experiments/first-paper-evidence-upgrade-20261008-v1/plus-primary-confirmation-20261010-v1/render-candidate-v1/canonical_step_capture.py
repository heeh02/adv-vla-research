"""Observe the original wrapper step without reimplementing its sensor noise.

Cloud runtime qualification must compare a tapped and an untapped execution.
The synthetic tests of this module cannot establish MuJoCo/render parity.
"""
from __future__ import annotations

from contextlib import contextmanager
import copy
import hashlib
import random

import numpy as np


def array_id(value):
    a = np.asarray(value)
    if a.dtype.hasobject:
        raise TypeError("Object arrays are not scientific observations")
    return {"dtype": a.dtype.str, "shape": list(a.shape),
            "sha256": hashlib.sha256(a.tobytes(order="C")).hexdigest()}


def rng_copy():
    # No random draws; capture both process-global streams used by the wrapper.
    return {"numpy": copy.deepcopy(np.random.get_state()),
            "python": copy.deepcopy(random.getstate())}


def rng_id(state):
    n = state["numpy"]
    return {"numpy": {"algorithm": n[0], "keys": array_id(n[1]),
                       "position": n[2], "has_gauss": n[3],
                       "cached_gaussian_hex": float(n[4]).hex()},
            "python": hashlib.sha256(repr(state["python"]).encode()).hexdigest()}


def rng_arrays(state):
    """Lossless no-pickle RNG payload, not just a provenance fingerprint."""
    n, p = state["numpy"], state["python"]
    return {"rng_numpy_algorithm": np.asarray(n[0]),
            "rng_numpy_keys": np.array(n[1], copy=True),
            "rng_numpy_position": np.asarray(n[2], np.int64),
            "rng_numpy_has_gauss": np.asarray(n[3], np.int64),
            "rng_numpy_cached_gaussian": np.asarray(n[4], np.float64),
            "rng_python_version": np.asarray(p[0], np.int64),
            "rng_python_internal": np.asarray(p[1], np.uint64),
            "rng_python_gaussian_present": np.asarray(p[2] is not None, np.bool_),
            "rng_python_gaussian": np.asarray(0.0 if p[2] is None else p[2], np.float64)}


def observation_ids(obs):
    return {k: array_id(v) for k, v in obs.items()
            if isinstance(v, np.ndarray)}


class StepCapture:
    """One selected canonical wrapper step, retaining pre/post sensor inputs."""

    def __init__(self, wrapper, physics_state):
        self.wrapper = wrapper
        self.physics_state = physics_state
        self.active = False
        self.record = None

    @contextmanager
    def _tap(self):
        if self.active:
            raise RuntimeError("Nested capture on the same step is forbidden")
        self.active = True
        underlying = self.wrapper.env
        if getattr(underlying, "_adv_plus_step_capture_active", False):
            self.active = False
            raise RuntimeError("Another observer already owns the underlying step")
        had_own_step = "step" in vars(underlying)
        own_step = vars(underlying).get("step")
        original = underlying.step
        calls = []

        def observed_step(*args, **kwargs):
            result = original(*args, **kwargs)
            if calls:
                raise RuntimeError("Expected exactly one underlying physics step")
            if not isinstance(result, tuple) or len(result) != 4:
                raise TypeError("Expected official four-field step result")
            obs = result[0]
            if not isinstance(obs, dict):
                raise TypeError("Expected official observation dictionary")
            calls.append({"observation": copy.deepcopy(obs),
                          "rng": rng_copy(),
                          "physics": np.array(self.physics_state(), copy=True)})
            # Return the original result; never replace, cast, or re-render it.
            return result

        underlying._adv_plus_step_capture_active = True
        underlying.step = observed_step
        try:
            yield calls
        finally:
            if had_own_step:
                underlying.step = own_step
            else:
                delattr(underlying, "step")
            delattr(underlying, "_adv_plus_step_capture_active")
            self.active = False

    def step(self, action, *, invoke=None):
        self.record = None
        action_before = array_id(action)
        rng_before = rng_copy()
        physics_before = np.array(self.physics_state(), copy=True)
        with self._tap() as calls:
            # Optional outer adapter retains its ordinary validation/delegation.
            returned = (invoke or self.wrapper.step)(action)
            if len(calls) != 1:
                raise RuntimeError("Canonical wrapper did not call its physics step once")
            pre = calls[0]
            post = {"observation": copy.deepcopy(returned[0]),
                    "rng": rng_copy(),
                    "physics": np.array(self.physics_state(), copy=True)}
        if array_id(action) != action_before:
            raise RuntimeError("Canonical action input was modified")
        if array_id(pre["physics"]) != array_id(post["physics"]):
            raise RuntimeError("Physical state changed after underlying step returned")
        self.record = {"schema": "plus-canonical-step-capture-v1",
                       "role": "qualification_original_wrapper_step",
                       "before_step": {"rng": rng_before, "physics": physics_before},
                       "pre_sensor": pre, "post_sensor": post,
                       "action": action_before,
                       "underlying_step_calls": len(calls),
                       "native_runtime_parity_established": False}
        return returned

    def receipt(self):
        if self.record is None:
            raise RuntimeError("No successful capture")
        r = self.record
        return {"schema": r["schema"], "role": r["role"], "action": r["action"],
                "underlying_step_calls": r["underlying_step_calls"],
                "native_runtime_parity_established": False,
                "before_step": {"rng": rng_id(r["before_step"]["rng"]),
                                "physics": array_id(r["before_step"]["physics"])},
                **{name: {"observations": observation_ids(r[name]["observation"]),
                          "rng": rng_id(r[name]["rng"]),
                          "physics": array_id(r[name]["physics"])}
                   for name in ("pre_sensor", "post_sensor")}}
