"""Same-state clean counterfactual observations (TECH-CONTRACT-v1 2.2-2.3; tests T3, T12).

Sensor Noise: ``SensorTap`` is adapted from P/render-candidate-v1/canonical_step_capture.py:57-132
(SHA see PROVENANCE-v1.json): it wraps the underlying robosuite ``step`` of ONE chunk-final
wrapper step, deep-copies the observation BEFORE the LIBERO-Plus wrapper replaces
``agentview_image``, and returns the original result unchanged. It performs no render and no
RNG draw. Addition: a copy of the render-relevant MjData arrays right after the underlying
step returns (diagnostic d' of T3 only).

Camera: ``CameraCallbackCapture`` adapts the in-callback triple of
E/native_visual_campaign.py:235-292 and the camera-only write/refresh/restore of
E/native_visual_plus.py:84-91,178-293 with the field lists of E/selective_camera_reference.py:20-29:
inside the native agentview render callback it renders the native frame, writes ONLY the
clean compiled agentview cam_pos/cam_quat/cam_fovy (from the base environment), refreshes
derived poses with mj_camlight (never mj_forward or a step), renders the clean frame,
restores every written model/data field bytewise, re-renders and requires the restored
frame to equal the native frame. Per transaction only the written/guarded camera, light,
headlight and material fields plus a physics/controller/RNG fingerprint are hashed; the
immutable model digest is computed once per unit (no per-render texture digests).
"""

from __future__ import annotations

from contextlib import contextmanager
import copy
import hashlib
import time

import numpy as np

from .common import TechnicalFailure, array_id, need
from .snapshot import light_fingerprint

CAMERAS = ("agentview", "robot0_eye_in_hand")
EDIT = ("cam_pos", "cam_quat", "cam_fovy")
STATIC = ("cam_bodyid", "cam_mode", "cam_targetbodyid")
EXTENDED_OPTICS = ("cam_intrinsic", "cam_sensorsize", "cam_resolution", "cam_ipd", "cam_orthographic", "cam_projection")
LIGHT = ("light_pos", "light_dir", "light_ambient", "light_diffuse", "light_specular", "light_active")
MATERIAL = ("mat_rgba", "mat_texid", "geom_matid", "geom_rgba", "site_rgba")
HEADLIGHT = ("active", "ambient", "diffuse", "specular")
IMMUTABLE = ("tex_data", "tex_rgb", "mat_texid", "geom_matid", "geom_size", "geom_type", "geom_pos", "geom_quat",
             "body_pos", "body_quat", "light_pos", "light_dir", "light_ambient", "light_diffuse", "light_specular",
             "light_active", "cam_bodyid", "cam_mode", "cam_targetbodyid", "cam_fovy") + EXTENDED_OPTICS
RENDER_DATA = ("qpos", "qvel", "act", "mocap_pos", "mocap_quat", "xpos", "xquat", "xmat", "xipos", "ximat",
               "geom_xpos", "geom_xmat", "site_xpos", "site_xmat", "cam_xpos", "cam_xmat", "light_xpos", "light_xdir")


def _same(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()


def _digest(arrays):
    return hashlib.sha256(b"".join(k.encode() + b":" + array_id(v)["sha256"].encode() for k, v in sorted(arrays.items()))).hexdigest()


def render_data_copy(raw_data):
    return {name: np.array(getattr(raw_data, name), copy=True) for name in RENDER_DATA if hasattr(raw_data, name)}


# ---------------------------------------------------------------- Sensor Noise

class SensorTap:
    """One tapped chunk-final wrapper step (canonical_step_capture.StepCapture adaptation)."""

    def __init__(self, wrapper, physics_state, raw_data=None):
        self.wrapper = wrapper
        self.physics_state = physics_state
        self.raw_data = raw_data
        self.record = None

    @contextmanager
    def _tap(self):
        underlying = self.wrapper.env
        if getattr(underlying, "_adv_plus_step_capture_active", False):
            raise TechnicalFailure("another_observer_owns_underlying_step")
        had_own = "step" in vars(underlying)
        own = vars(underlying).get("step")
        original = underlying.step
        calls = []

        def observed_step(*args, **kwargs):
            result = original(*args, **kwargs)
            if calls:
                raise TechnicalFailure("expected_exactly_one_underlying_physics_step")
            if not isinstance(result, tuple) or len(result) != 4 or not isinstance(result[0], dict):
                raise TechnicalFailure("expected_official_four_field_step_result")
            calls.append({"observation": copy.deepcopy(result[0]),
                          "physics": np.array(self.physics_state(), copy=True),
                          "render_data": render_data_copy(self.raw_data) if self.raw_data is not None else None,
                          "sim_time": float(getattr(self.raw_data, "time", np.nan)) if self.raw_data is not None else None})
            return result  # never replace, cast or re-render

        underlying._adv_plus_step_capture_active = True
        underlying.step = observed_step
        try:
            yield calls
        finally:
            if had_own:
                underlying.step = own
            else:
                delattr(underlying, "step")
            delattr(underlying, "_adv_plus_step_capture_active")

    def step(self, action, *, invoke):
        action_before = array_id(action)
        with self._tap() as calls:
            returned = invoke(action)
            need(len(calls) == 1, "wrapper_did_not_call_underlying_step_once")
            pre = calls[0]
        need(array_id(action) == action_before, "canonical_action_input_modified")
        post_obs = returned[0]
        need(_same(np.asarray(pre["physics"]), np.asarray(self.physics_state())), "physics_changed_after_underlying_step")
        self.record = {"pre_sensor_observation": pre["observation"], "post_sensor_observation": copy.deepcopy(post_obs),
                       "source_physics": pre["physics"], "render_data": pre["render_data"], "sim_time": pre["sim_time"]}
        return returned

    def clean_observation(self):
        """Pre-transform frame set; all non-agentview keys must equal the post-sensor observation."""
        need(self.record is not None, "no_sensor_tap_record")
        pre, post = self.record["pre_sensor_observation"], self.record["post_sensor_observation"]
        need(set(pre) == set(post), "sensor_tap_observation_keys_differ")
        others_equal = {k: _same(pre[k], post[k]) for k in pre if k != "agentview_image" and isinstance(pre[k], np.ndarray)}
        need(all(others_equal.values()), "sensor_tap_nonagentview_keys_differ")
        clean = {k: (np.array(v, copy=True) if isinstance(v, np.ndarray) else copy.deepcopy(v)) for k, v in pre.items()}
        receipt = {"wrist_pre_equals_post": others_equal.get("robot0_eye_in_hand_image"),
                   "nonagentview_keys_equal": all(others_equal.values()),
                   "agentview_changed_by_wrapper": not _same(pre["agentview_image"], post["agentview_image"]),
                   "source_physics_sha256": array_id(self.record["source_physics"])["sha256"],
                   "extra_renders": 0, "extra_rng_draws": 0}
        return clean, receipt


# ---------------------------------------------------------------- Camera

def camera_ids(mujoco_module, raw_model):
    ids = {name: int(mujoco_module.mj_name2id(raw_model, mujoco_module.mjtObj.mjOBJ_CAMERA, name)) for name in CAMERAS}
    need(min(ids.values()) >= 0 and ids["agentview"] != ids["robot0_eye_in_hand"], "compiled_camera_names_unavailable")
    return ids


def clean_camera_fields(base_model, agentview_id):
    """Compiled clean agentview fields read from the base-environment model (never inverted)."""
    fields = {}
    for name in EDIT + STATIC + EXTENDED_OPTICS:
        if hasattr(base_model, name):
            fields[name] = np.array(np.asarray(getattr(base_model, name))[agentview_id], copy=True)
    return fields


def camera_compatibility(variant_model, base_fields, agentview_id):
    """Recoverability guard of TECH-CONTRACT-v1 2.3/2.4 for a camera-only transaction."""
    problems = []
    for name in STATIC + EXTENDED_OPTICS:
        if name in base_fields:
            if not hasattr(variant_model, name):
                problems.append("variant_lacks:" + name)
            elif not _same(np.asarray(getattr(variant_model, name))[agentview_id], base_fields[name]):
                problems.append("differs:" + name)
        elif hasattr(variant_model, name):
            problems.append("base_lacks:" + name)
    for name, expected in (("cam_bodyid", 0), ("cam_mode", 0), ("cam_targetbodyid", -1)):
        if name in base_fields and int(base_fields[name]) != expected:
            problems.append("not_world_fixed:" + name)
    quat = np.asarray(base_fields.get("cam_quat", np.zeros(4)), np.float64)
    if abs(float(np.linalg.norm(quat)) - 1.0) >= 1e-5:
        problems.append("clean_quat_not_unit")
    fovy = float(np.asarray(base_fields.get("cam_fovy", 0.0)))
    if not 0 < fovy < 180:
        problems.append("invalid_clean_fovy")
    return {"recoverable_by_camera_only_transaction": not problems, "problems": problems}


def _quat_matrix_wxyz(q):
    w, x, y, z = np.asarray(q, np.float64)
    return np.asarray([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
                       2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
                       2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)])


class CameraCallbackCapture:
    """Installed on the treated env's ``sim.render``; enabled only on chunk-final steps."""

    def __init__(self, handles, clean_fields, *, convention, keep_render_data=False):
        self.h = handles
        self.model = handles.sim.model
        self.data = handles.sim.data
        self.clean = {k: np.array(v, copy=True) for k, v in clean_fields.items() if k in EDIT}
        self.ids = camera_ids(handles.mujoco, handles.raw_model)
        self.convention = convention
        self.keep_render_data = keep_render_data
        self.enabled = False
        self.step = 0
        self.records = {}
        self.transactions = 0
        self.seconds = 0.0
        self.setup_digest = self.immutable_digest()
        self.original_render = handles.sim.render
        handles.sim.render = self.render

    def close(self):
        self.h.sim.render = self.original_render

    def immutable_digest(self):
        arrays = {name: np.asarray(getattr(self.model, name)) for name in IMMUTABLE if hasattr(self.model, name)}
        arrays.update({"headlight_" + n: np.asarray(getattr(self.model.vis.headlight, n)) for n in HEADLIGHT})
        return _digest(arrays)

    def guard_arrays(self):
        m, d = self.model, self.data
        arrays = {name: np.array(getattr(m, name), copy=True) for name in EDIT + STATIC + EXTENDED_OPTICS + LIGHT + MATERIAL
                  if hasattr(m, name)}
        arrays.update({"headlight_" + n: np.array(getattr(m.vis.headlight, n), copy=True) for n in HEADLIGHT})
        arrays.update({"data_" + n: np.array(getattr(d, n), copy=True) for n in ("cam_xpos", "cam_xmat", "light_xpos", "light_xdir")})
        return arrays

    def _restore(self, saved):
        m, d, cam = self.model, self.data, self.ids["agentview"]
        for name in EDIT:
            getattr(m, name)[cam] = saved[name][cam]
        for name in ("cam_xpos", "cam_xmat", "light_xpos", "light_xdir"):
            getattr(d, name)[...] = saved["data_" + name]

    def render(self, *args, **kwargs):
        native = self.original_render(*args, **kwargs)
        camera = kwargs.get("camera_name")
        if not self.enabled or camera not in CAMERAS:
            return native
        if kwargs.get("depth", False) or kwargs.get("segmentation", False):
            raise TechnicalFailure("policy_capture_requires_rgb_sensor")
        flipped = np.array(np.asarray(native)[::self.convention], copy=True)
        if camera == "robot0_eye_in_hand":
            self.records[camera] = {"step": self.step, "native": flipped, "sim_time": float(self.data.time)}
            return native
        started = time.monotonic()
        m, d, cam, wrist = self.model, self.data, self.ids["agentview"], self.ids["robot0_eye_in_hand"]
        source_physics = np.array(self.h.sim.get_state().flatten(), copy=True)
        render_data = render_data_copy(self.h.raw_data) if self.keep_render_data else None
        fp_before = light_fingerprint(self.h)
        saved = self.guard_arrays()
        guard_before = _digest(saved)
        try:
            for name in EDIT:
                getattr(m, name)[cam] = self.clean[name]
            self.h.mujoco.mj_camlight(self.h.raw_model, self.h.raw_data)
            pose_ok = (np.allclose(d.cam_xpos[cam], self.clean["cam_pos"], atol=1e-6, rtol=1e-6)
                       and np.allclose(np.asarray(d.cam_xmat[cam]).reshape(9), _quat_matrix_wxyz(self.clean["cam_quat"]),
                                       atol=1e-6, rtol=1e-6))
            others = [i for i in range(np.asarray(d.cam_xpos).shape[0]) if i != cam]
            others_ok = (_same(np.asarray(d.cam_xpos)[others], saved["data_cam_xpos"][others])
                         and _same(np.asarray(d.cam_xmat)[others], saved["data_cam_xmat"][others]))
            lights_ok = _same(d.light_xpos, saved["data_light_xpos"]) and _same(d.light_xdir, saved["data_light_xdir"])
            clean = np.array(np.asarray(self.original_render(*args, **kwargs))[::self.convention], copy=True)
        finally:
            self._restore(saved)
        restored = np.array(np.asarray(self.original_render(*args, **kwargs))[::self.convention], copy=True)
        guard_after = _digest(self.guard_arrays())
        fp_after = light_fingerprint(self.h)
        self.transactions += 1
        self.seconds += time.monotonic() - started
        self.records[camera] = {
            "step": self.step, "native": flipped, "clean": clean, "sim_time": float(d.time),
            "source_physics": source_physics, "render_data": render_data,
            "checks": {"a_restored_equals_native": _same(restored, flipped),
                       "clean_pose_refreshed": bool(pose_ok), "other_cameras_derived_unchanged": bool(others_ok),
                       "light_derived_unchanged": bool(lights_ok),
                       "c_guarded_fields_restored": guard_before == guard_after,
                       "c_physics_controller_rng_unchanged": fp_before == fp_after,
                       "wrist_derived_index": wrist},
        }
        return native

    def take(self, step, observation):
        """Return (clean_observation, receipt) for the chunk-final observation at ``step``."""
        agent, wrist = self.records.get("agentview"), self.records.get("robot0_eye_in_hand")
        need(agent is not None and wrist is not None and agent["step"] == step and wrist["step"] == step,
             "camera_capture_missing_at_policy_sensor_step")
        need(_same(observation["agentview_image"], agent["native"]), "policy_agentview_differs_from_callback_native")
        need(_same(observation["robot0_eye_in_hand_image"], wrist["native"]), "policy_wrist_differs_from_callback_native")
        checks = dict(agent["checks"])
        checks["b_wrist_unchanged_by_transaction"] = True  # wrist is rendered natively, never inside the transaction
        need(checks["a_restored_equals_native"] and checks["c_guarded_fields_restored"]
             and checks["c_physics_controller_rng_unchanged"] and checks["clean_pose_refreshed"]
             and checks["other_cameras_derived_unchanged"] and checks["light_derived_unchanged"],
             "camera_transaction_guard_failed")
        clean = {k: (np.array(v, copy=True) if isinstance(v, np.ndarray) else copy.deepcopy(v)) for k, v in observation.items()}
        clean["agentview_image"] = np.array(agent["clean"], copy=True)
        receipt = {"checks": checks, "source_physics_sha256": array_id(agent["source_physics"])["sha256"],
                   "sim_time": agent["sim_time"], "extra_agentview_renders": 2, "mj_camlight_calls": 1,
                   "per_render_full_model_digest": False}
        return clean, receipt
