"""Warm-up snapshot S0: capture, restore and verification (TECH-CONTRACT-v1 3.2-3.3, test T2).

Capture adapts the read-only walker of E/libero_light_backend.py:14-45 (``numeric_snapshot``,
SHA 80e21aad...) and the field list of its ``snapshot`` (:139-167); restore is new. Changes:
  * the walker records unknown/opaque members in an explicit ``unobserved`` list instead
    of raising (plan 3.2 item 7), and keeps dict key sets so restore can drop extra keys;
  * the MuJoCo integration state uses ``mj_getState(mjSTATE_INTEGRATION)`` when the
    installed MuJoCo exposes it (T11), else the explicit list of TECH-CONTRACT-v1 3.2;
  * robot capture covers vars(robot) (controller, gripper, history buffers) excluding
    sim/model references; environment capture covers timestep/cur_time/done, _obs_cache,
    _observables and top-level numeric attributes of the LIBERO problem env;
  * RNG (global NumPy and Python) is restored last, losslessly.
Restore order (3.3): integration state -> mj_forward -> integration state again (mj_forward
rewrites qacc_warmstart; this re-write is recorded) -> runtime model fields -> robot trees
-> environment trees -> RNG. Arrays are restored in place with np.copyto so references stay.
"""

from __future__ import annotations

import copy
import hashlib
import random
import types

import numpy as np

from .common import TechnicalFailure, array_id, need

EXCLUDED_KEYS = frozenset({"sim", "_sim", "model", "_model", "robot_model", "mujoco_robot",
                           "mujoco_arena", "viewer", "renderer", "_renderer", "env", "_env"})
ALLOWED_MODULES = ("robosuite", "libero")
EXPLICIT_INTEGRATION_FIELDS = ("time", "qpos", "qvel", "act", "qacc_warmstart", "ctrl", "qfrc_applied",
                               "xfrc_applied", "mocap_pos", "mocap_quat", "userdata", "plugin_state", "eq_active")
DEFAULT_RUNTIME_MODEL_FIELDS = ("site_rgba",)
ENV_SCALARS = ("timestep", "cur_time", "done")
SKIP_TYPES = (types.FunctionType, types.MethodType, types.BuiltinFunctionType, types.BuiltinMethodType,
              type, types.ModuleType)


class SimHandles:
    """Read handles on one live LIBERO ControlEnv wrapper (no mutation on construction)."""

    def __init__(self, wrapper, mujoco_module):
        self.wrapper = wrapper
        self.env = wrapper.env
        self.sim = self.env.sim
        self.raw_model = getattr(self.sim.model, "_model", self.sim.model)
        self.raw_data = getattr(self.sim.data, "_data", self.sim.data)
        self.mujoco = mujoco_module


# ---------------------------------------------------------------- integration state

def integration_mode(mujoco_module):
    spec = getattr(getattr(mujoco_module, "mjtState", None), "mjSTATE_INTEGRATION", None)
    ok = spec is not None and all(hasattr(mujoco_module, n) for n in ("mj_getState", "mj_setState", "mj_stateSize"))
    return "mj_getState" if ok else "explicit"


def capture_integration(h, mode):
    if mode == "mj_getState":
        spec = h.mujoco.mjtState.mjSTATE_INTEGRATION
        state = np.empty(h.mujoco.mj_stateSize(h.raw_model, spec), np.float64)
        h.mujoco.mj_getState(h.raw_model, h.raw_data, state, spec)
        return {"mj_state_integration": state}, []
    out, missing = {}, []
    for name in EXPLICIT_INTEGRATION_FIELDS:
        if not hasattr(h.raw_data, name):
            missing.append("integration:" + name)
            continue
        value = getattr(h.raw_data, name)
        out[name] = np.array(value, copy=True) if not np.isscalar(value) else np.asarray(value).copy()
    return out, missing


def write_integration(h, mode, saved):
    if mode == "mj_getState":
        spec = h.mujoco.mjtState.mjSTATE_INTEGRATION
        h.mujoco.mj_setState(h.raw_model, h.raw_data, np.asarray(saved["mj_state_integration"]), spec)
        return
    for name, value in saved.items():
        current = getattr(h.raw_data, name)
        if isinstance(current, np.ndarray) and current.shape != ():
            np.copyto(current, value)
        else:
            setattr(h.raw_data, name, type(current)(np.asarray(value).item()))


# ---------------------------------------------------------------- generic tree walker

def _is_scalar(value):
    return isinstance(value, (bool, int, float, np.bool_, np.integer, np.floating))


def capture_tree(root, *, allowed_modules=ALLOWED_MODULES, max_depth=12, root_kind="object"):
    """Leaves: list of (path, kind, value). kind in {'ndarray','scalar'}. Read-only."""
    leaves, unobserved, dict_keys = [], [], []
    seen = set()

    def visit(obj, path, depth):
        if isinstance(obj, np.ndarray):
            if obj.dtype.hasobject:
                unobserved.append({"path": _path_text(path), "reason": "object_ndarray"})
            else:
                leaves.append((path, "ndarray", np.array(obj, copy=True)))
            return
        if _is_scalar(obj):
            leaves.append((path, "scalar", (type(obj), copy.copy(obj))))
            return
        if obj is None or isinstance(obj, (str, bytes)) or isinstance(obj, SKIP_TYPES):
            return
        if depth > max_depth:
            unobserved.append({"path": _path_text(path), "reason": "depth_limit"})
            return
        if id(obj) in seen:
            return
        seen.add(id(obj))
        if isinstance(obj, dict):
            dict_keys.append((path, frozenset(obj.keys())))
            for key, value in list(obj.items()):
                if str(key) not in EXCLUDED_KEYS:
                    visit(value, path + (("key", key),), depth + 1)
        elif isinstance(obj, list):
            for index, value in enumerate(obj):
                visit(value, path + (("idx", index),), depth + 1)
        elif isinstance(obj, tuple):
            for index, value in enumerate(obj):
                visit(value, path + (("tidx", index),), depth + 1)
        elif type(obj).__module__.split(".")[0] in allowed_modules and hasattr(obj, "__dict__"):
            for key, value in list(vars(obj).items()):
                if key not in EXCLUDED_KEYS:
                    visit(value, path + (("attr", key),), depth + 1)
        else:
            unobserved.append({"path": _path_text(path), "reason": "opaque_type",
                               "type": type(obj).__module__ + "." + type(obj).__qualname__})

    visit(root, (), 0)
    return {"leaves": leaves, "unobserved": unobserved, "dict_keys": dict_keys}


def _path_text(path):
    return "/".join(f"{kind}:{name}" for kind, name in path) or "<root>"


def _get(parent, step):
    kind, name = step
    if kind == "attr":
        return getattr(parent, name)
    return parent[name]


def _set(parent, step, value):
    kind, name = step
    if kind == "attr":
        setattr(parent, name, value)
    elif kind in ("key", "idx"):
        parent[name] = value
    else:
        raise TechnicalFailure("immutable_tuple_member_differs:" + str(name))


def _navigate(root, path):
    node = root
    for step in path:
        node = _get(node, step)
    return node


def restore_tree(root, tree):
    """Write captured leaves back in place; drop dict keys absent at capture."""
    replaced = []
    for path, keys in tree["dict_keys"]:
        node = _navigate(root, path)
        if isinstance(node, dict):
            for extra in [k for k in node if k not in keys]:
                del node[extra]
    for path, kind, saved in tree["leaves"]:
        if not path:
            continue
        parent = _navigate(root, path[:-1])
        step = path[-1]
        try:
            current = _get(parent, step)
        except (KeyError, IndexError, AttributeError):
            current = None
        if kind == "ndarray":
            if (isinstance(current, np.ndarray) and current.shape == saved.shape and current.dtype == saved.dtype
                    and current.flags.writeable):
                np.copyto(current, saved)
            else:
                if step[0] == "tidx":
                    need(isinstance(current, np.ndarray) and current.tobytes() == saved.tobytes(),
                         "immutable_tuple_array_differs:" + _path_text(path))
                    continue
                _set(parent, step, np.array(saved, copy=True))
                replaced.append(_path_text(path))
        else:
            value_type, value = saved
            if type(current) is value_type and _scalar_equal(current, value):
                continue
            if step[0] == "tidx":
                raise TechnicalFailure("immutable_tuple_scalar_differs:" + _path_text(path))
            _set(parent, step, value_type(value) if value_type not in (np.bool_,) else np.bool_(value))
    return replaced


def _scalar_equal(a, b):
    try:
        return np.asarray(a).tobytes() == np.asarray(b).tobytes() and np.asarray(a).dtype == np.asarray(b).dtype
    except Exception:  # noqa: BLE001
        return False


def tree_digest(tree):
    rows = {}
    for path, kind, value in tree["leaves"]:
        if kind == "ndarray":
            rows[_path_text(path)] = "a:" + array_id(value)["sha256"] + ":" + value.dtype.str + ":" + str(value.shape)
        else:
            value_type, scalar = value
            rows[_path_text(path)] = "s:" + value_type.__name__ + ":" + hashlib.sha256(np.asarray(scalar).tobytes()).hexdigest()
    rows.update({"dictkeys:" + _path_text(p): repr(sorted(map(str, k))) for p, k in tree["dict_keys"]})
    return rows


# ---------------------------------------------------------------- S0

def rng_state():
    return {"numpy": copy.deepcopy(np.random.get_state()), "python": copy.deepcopy(random.getstate())}


def capture(h, *, mode, runtime_model_fields=DEFAULT_RUNTIME_MODEL_FIELDS, observations=None):
    integration, missing = capture_integration(h, mode)
    model_runtime = {}
    for name in runtime_model_fields:
        if hasattr(h.sim.model, name):
            model_runtime[name] = np.array(getattr(h.sim.model, name), copy=True)
        else:
            missing.append("model_runtime:" + name)
    robots = [capture_tree(robot) for robot in h.env.robots]
    env_scalars = {name: copy.copy(getattr(h.env, name)) for name in ENV_SCALARS if hasattr(h.env, name)}
    obs_cache = capture_tree(h.env._obs_cache, root_kind="dict")  # noqa: SLF001
    observables = capture_tree(h.env._observables, root_kind="dict")  # noqa: SLF001
    libero_numeric, libero_skipped = {}, []
    for key, value in vars(h.env).items():
        if key in ENV_SCALARS or key.startswith("_obs") or key in EXCLUDED_KEYS:
            continue
        if isinstance(value, np.ndarray) and not value.dtype.hasobject:
            libero_numeric[key] = np.array(value, copy=True)
        elif _is_scalar(value):
            libero_numeric[key] = copy.copy(value)
        elif not (value is None or isinstance(value, (str, bytes)) or isinstance(value, SKIP_TYPES)):
            libero_skipped.append({"attr": key, "type": type(value).__module__ + "." + type(value).__qualname__})
    unobserved = [{"path": m, "reason": "absent_in_runtime"} for m in missing]
    for label, tree in [("robot%d" % i, t) for i, t in enumerate(robots)] + [("obs_cache", obs_cache), ("observables", observables)]:
        unobserved += [{**row, "tree": label} for row in tree["unobserved"]]
    unobserved += [{"path": "env/" + row["attr"], "reason": "non_numeric_env_member_not_captured", "type": row["type"]}
                   for row in libero_skipped]
    unobserved += [{"path": "callables_and_closures", "reason": "functions/closures (e.g. observable sensors) are not state-captured"},
                   {"path": "renderer_internal", "reason": "offscreen render context is not captured"}]
    return {"schema": "cf-repair-S0-v1", "mode": mode, "integration": integration, "model_runtime": model_runtime,
            "robots": robots, "env_scalars": env_scalars, "obs_cache": obs_cache, "observables": observables,
            "env_numeric": libero_numeric, "wrapper_noise": getattr(h.wrapper, "noise", None),
            "rng": rng_state(), "unobserved": unobserved, "observations": observations}


def fingerprint(snapshot):
    """Per-field digests of every captured section (S0 or a re-capture)."""
    rows = {}
    for name, value in snapshot["integration"].items():
        rows["integration/" + name] = array_id(value)["sha256"]
    for name, value in snapshot["model_runtime"].items():
        rows["model_runtime/" + name] = array_id(value)["sha256"]
    for i, tree in enumerate(snapshot["robots"]):
        rows.update({f"robot{i}/" + k: v for k, v in tree_digest(tree).items()})
    for name, value in snapshot["env_scalars"].items():
        rows["env/" + name] = type(value).__name__ + ":" + hashlib.sha256(np.asarray(value).tobytes()).hexdigest()
    rows.update({"obs_cache/" + k: v for k, v in tree_digest(snapshot["obs_cache"]).items()})
    rows.update({"observables/" + k: v for k, v in tree_digest(snapshot["observables"]).items()})
    for name, value in snapshot["env_numeric"].items():
        rows["env_numeric/" + name] = type(value).__name__ + ":" + hashlib.sha256(np.asarray(value).tobytes()).hexdigest()
    rows["wrapper/noise"] = repr(snapshot["wrapper_noise"])
    n, p = snapshot["rng"]["numpy"], snapshot["rng"]["python"]
    rows["rng/numpy"] = hashlib.sha256(np.asarray(n[1]).tobytes() + repr((n[0], n[2], n[3], float(n[4]).hex())).encode()).hexdigest()
    rows["rng/python"] = hashlib.sha256(repr(p).encode()).hexdigest()
    return rows


def compare(a, b, *, exclude_prefixes=()):
    keys = set(a) | set(b)
    diffs = sorted(k for k in keys if not k.startswith(tuple(exclude_prefixes)) and a.get(k) != b.get(k))
    return {"equal": not diffs, "different_fields": diffs[:200], "different_count": len(diffs), "compared": len(keys)}


SHARED_EXCLUDE = ("obs_cache/", "observables/")


def restore(h, snapshot, *, into_base=False):
    """Restore S0 into a live wrapper. ``into_base``: the paired base env E_C keeps its own
    LIBERO problem-env numeric attributes (variant view configuration) and wrapper noise id."""
    mode = snapshot["mode"]
    write_integration(h, mode, snapshot["integration"])
    h.mujoco.mj_forward(h.raw_model, h.raw_data)
    write_integration(h, mode, snapshot["integration"])
    for name, value in snapshot["model_runtime"].items():
        np.copyto(getattr(h.sim.model, name), value)
    replaced = []
    for robot, tree in zip(h.env.robots, snapshot["robots"], strict=True):
        replaced += restore_tree(robot, tree)
    for name, value in snapshot["env_scalars"].items():
        setattr(h.env, name, copy.copy(value))
    replaced += restore_tree(h.env._obs_cache, snapshot["obs_cache"])  # noqa: SLF001
    replaced += restore_tree(h.env._observables, snapshot["observables"])  # noqa: SLF001
    for name, value in ({} if into_base else snapshot["env_numeric"]).items():
        current = getattr(h.env, name, None)
        if isinstance(value, np.ndarray) and isinstance(current, np.ndarray) and current.shape == value.shape \
                and current.dtype == value.dtype and current.flags.writeable:
            np.copyto(current, value)
        else:
            setattr(h.env, name, copy.copy(value))
    np.random.set_state(copy.deepcopy(snapshot["rng"]["numpy"]))
    random.setstate(copy.deepcopy(snapshot["rng"]["python"]))
    return {"restore_order": ["integration", "mj_forward", "integration_rewrite_after_forward", "model_runtime",
                              "robots", "env_scalars", "obs_cache", "observables", "env_numeric", "rng"],
            "replaced_paths": replaced, "integration_rewritten_after_forward": True,
            "into_base": into_base, "env_numeric_restored": not into_base}


BASE_EXCLUDE = ("env_numeric/", "wrapper/")


def recapture(h, snapshot):
    """Re-capture with the same mode/fields as ``snapshot`` (for verification)."""
    return capture(h, mode=snapshot["mode"], runtime_model_fields=tuple(snapshot["model_runtime"]))


def light_fingerprint(h):
    """Cheap per-transaction guard: integration state, robot trees, RNG (TECH-CONTRACT-v1 2.3)."""
    integration, _ = capture_integration(h, integration_mode(h.mujoco))
    rows = {"integration/" + k: array_id(v)["sha256"] for k, v in integration.items()}
    for i, robot in enumerate(h.env.robots):
        rows.update({f"robot{i}/" + k: v for k, v in tree_digest(capture_tree(robot)).items()})
    state = rng_state()
    n = state["numpy"]
    rows["rng/numpy"] = hashlib.sha256(
        np.asarray(n[1]).tobytes() + repr((n[0], n[2], n[3], float(n[4]).hex())).encode()).hexdigest()
    rows["rng/python"] = hashlib.sha256(repr(state["python"]).encode()).hexdigest()
    return hashlib.sha256(repr(sorted(rows.items())).encode()).hexdigest()
