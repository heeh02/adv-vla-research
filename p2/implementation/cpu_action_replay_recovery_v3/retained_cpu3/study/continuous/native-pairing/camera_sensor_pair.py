"""Camera-only sensor callback candidate; no environment/model/rollout factory.

C is the original camera_rgb callback's renderer result, returned unchanged.
T and restored-C use that same renderer INSIDE the original callback scope.
The query-side supplier only reads captured buffers; it never renders.
Native hook/untapped and next-step parity remain unqualified by this source.
"""
from contextlib import contextmanager
import copy
import hashlib
import inspect
from pathlib import Path
import re
import sys

import numpy as np
from openpi.eval import selective_reference as R
from openpi.eval.selective_camera_reference import EDIT, STATIC, EXTENDED_OPTICS
from openpi.eval.selective_camera_reference import _array_binding, _attribute_binding, _quat_matrix_wxyz, LIGHT, MATERIAL, HEADLIGHT

CAMERAS = ("agentview", "robot0_eye_in_hand")
CASES = {"SDRU-Q01": (0, 0, 100, 4, 8, 0, 0), "SDRU-Q02": (0, 0, 100, 354, 2, 0, 0)}
ROBOT_ENV_SHA = "a0b39012c175272a2037a7a66a771d416b905a5a7b847e53c60a0588407398af"


def need(ok, code):
    if not ok:
        raise ValueError(code)


def same(a, b):
    return R._identity(a) == R._identity(b)


class CameraParameters:
    """Use supplied, independently source-bound compiled camera fields.

    No angle-to-pose implementation and no guessed geometry. Required source
    receipts bind clean/variant compiled payloads to one of the two old pairs.
    This validates the binding shape; root must verify the immutable receipts.
    """
    def __init__(self, sim, *, case_id, official_parameters, clean_fields, variant_fields,
                 source_receipt, camlight, activity_counter, passive_bindings):
        need(case_id in CASES and tuple(official_parameters) == CASES[case_id], "only_two_declared_camera_variants")
        need(source_receipt.get("case_id") == case_id and source_receipt.get("qualified_q0_pair") is True,
             "actual_old_q0_pair_receipt_required")
        need(source_receipt.get("clean_fields_id") == R._identity(clean_fields)
             and source_receipt.get("variant_fields_id") == R._identity(variant_fields), "compiled_camera_payload_binding")
        need(all(type(source_receipt.get(k)) is str and re.fullmatch(r"[0-9a-f]{64}", source_receipt[k]) for k in
                 ("pair_receipt_sha256", "clean_array_member_sha256", "variant_array_member_sha256")), "exact_compiled_source_refs")
        self.sim, self.case_id = sim, case_id
        self.clean, self.variant = copy.deepcopy(clean_fields), copy.deepcopy(variant_fields)
        self.camlight, self.activity = camlight, activity_counter
        self.passive = tuple(passive_bindings)
        need(callable(camlight) and callable(activity_counter) and self.passive
             and any(b.name == "physics.current" for b in self.passive)
             and all(not b.allow_temporary_change for b in self.passive), "protected_physics_and_declared_passive_witnesses")
        self.camera = sim.model.camera_name2id("agentview")
        need(sim.model.camera_name2id("robot0_eye_in_hand") != self.camera, "distinct_native_camera_ids")
        self.last_reports = []
        self.check_clean()

    def check_clean(self):
        model = self.sim.model
        fields = set(EDIT) | set(STATIC) | {k for k in EXTENDED_OPTICS if hasattr(model, k)}
        need(set(self.clean) == set(self.variant) == fields, "all_exposed_camera_fields_required")
        for name in fields:
            actual = np.asarray(getattr(model, name)[self.camera])
            clean, variant = np.asarray(self.clean[name]), np.asarray(self.variant[name])
            need(actual.dtype == clean.dtype == variant.dtype and actual.shape == clean.shape == variant.shape
                 and np.isfinite(clean).all() and np.isfinite(variant).all(), "native_compiled_camera_ABI")
            need(same(actual, clean), "root_must_still_use_exact_clean_camera")
            if name not in EDIT:
                need(same(clean, variant), "only_pose_and_fovy_may_differ")
        for name, expected in (("cam_bodyid", 0), ("cam_mode", 0), ("cam_targetbodyid", -1)):
            need(int(self.clean[name]) == expected, "fixed_world_camera_only")
        need(model.body_id2name(0) == "world", "world_body_identity")
        for profile in (self.clean, self.variant):
            need(0 < float(profile["cam_fovy"]) < 180 and abs(np.linalg.norm(profile["cam_quat"]) - 1) < 1e-5,
                 "actual_fovy_and_wxyz_unit_quaternion")

    @contextmanager
    def temporary(self, role):
        need(role in ("nativeperturb", "clean"), "two_optical_roles_only")
        self.check_clean()
        model, data = self.sim.model, self.sim.data
        other = np.asarray([i for i in range(len(model.cam_pos)) if i != self.camera])
        bindings = [
            _array_binding("camera.target", model, EDIT, selection=self.camera, temporary=True),
            _array_binding("camera.derived_target", data, ("cam_xpos", "cam_xmat"), selection=self.camera, temporary=True),
            _array_binding("camera.other", model, EDIT, selection=other),
            _array_binding("camera.static", model, STATIC),
            _array_binding("camera.extended", model, tuple(k for k in EXTENDED_OPTICS if hasattr(model, k))),
            _array_binding("camera.derived_other", data, ("cam_xpos", "cam_xmat"), selection=other),
            _array_binding("light.model", model, LIGHT),
            _array_binding("light.derived", data, ("light_xpos", "light_xdir")),
            _array_binding("material", model, MATERIAL),
            _attribute_binding("headlight", model.vis.headlight, HEADLIGHT), *self.passive]
        transaction = R.ReferenceTransaction(bindings, required_states=[b.name for b in bindings],
                                              physical_step_counter=self.activity)
        self.last_reports.append(transaction.report)
        with transaction:
            wanted = self.variant if role == "nativeperturb" else self.clean
            for field in EDIT:
                getattr(model, field)[self.camera] = wanted[field]
            self.camlight(self.sim)  # qualified mj_camlight only, never forward
            def readback():
                need(all(same(np.asarray(getattr(model, field)[self.camera]), np.asarray(wanted[field])) for field in EDIT),
                     "actual_camera_write_readback")
                need(np.allclose(data.cam_xpos[self.camera], wanted["cam_pos"], atol=1e-6, rtol=1e-6)
                     and np.allclose(data.cam_xmat[self.camera], _quat_matrix_wxyz(wanted["cam_quat"]), atol=1e-6, rtol=1e-6),
                     "actual_camlight_world_pose_readback")
            readback()
            yield
            readback()
        self.check_clean()


class NativeCameraPairTap:
    """Instance-local hook following actual native SensorLightCapture ordering.

    clock_state runs inside camera_rgb's sim.render call and supplies actual
    sim_time + physical_state. parent_step_index is declared by the existing
    root immediately before its ordinary step, not inferred after the query.
    """
    def __init__(self, sim, observables, optics, *, clock_state, synthetic=False):
        need(synthetic or sys.platform.startswith("linux"), "native_capture_requires_cloud_Linux")
        need(set(observables) == {c + "_image" for c in CAMERAS}, "exact_two_RGB_observables")
        need(callable(clock_state), "sensor_boundary_clock_reader")
        self.sim, self.observables, self.optics = sim, observables, optics
        self.clock_state, self.synthetic = clock_state, synthetic
        self.active, self.armed, self.records, self.halted = None, None, {}, False
        self.partial_records = {}  # at most the latest actual attempt per camera
        self._old_render, self._old_sensors = None, {}
        self.conventions = {}

    def _validate_sensor(self, camera, callback):
        if self.synthetic:
            return callback.mock_convention
        need(inspect.isfunction(callback) and callback.__code__.co_name == "camera_rgb", "native_camera_rgb_callback_required")
        source = Path(callback.__code__.co_filename)
        need(source.is_file() and hashlib.sha256(source.read_bytes()).hexdigest() == ROBOT_ENV_SHA,
             "actual_Plus_camera_producer_source_not_yet_bound")
        closure = inspect.getclosurevars(callback).nonlocals
        need(closure.get("cam_name") == camera and closure.get("cam_w") == closure.get("cam_h") == 256
             and closure.get("cam_d") is False and closure.get("convention") in (-1, 1)
             and getattr(closure.get("self"), "sim", None) is self.sim, "exact_RGB_sensor_closure")
        return closure["convention"]

    @contextmanager
    def installed(self):
        need(self._old_render is None and not self.halted and not hasattr(self.sim, "_sru_native_camera_pair_owner"), "one_sensor_hook_owner")
        self._old_render = self.sim.render
        had_render = "render" in vars(self.sim)
        own_render = vars(self.sim).get("render")
        try:
            for camera in CAMERAS:
                obs = self.observables[camera + "_image"]
                original = obs._sensor
                self.conventions[camera] = self._validate_sensor(camera, original)
                self._old_sensors[camera] = original
            self.sim._sru_native_camera_pair_owner = self
            self.sim.render = self._render
            for camera in CAMERAS:
                def sensor(cache, camera=camera):
                    if self.armed is None:
                        return self._old_sensors[camera](cache)
                    need(self.active is None and not self.halted, "no_nested_camera_sensor")
                    cache_before = R._identity(cache)
                    self.active = {"camera": camera, "renders": 0}
                    try:
                        result = self._old_sensors[camera](cache)
                        need(self.active["renders"] == 1 and same(result, self.records[camera]["clean"]), "original_sensor_output_preserved")
                        need(R._identity(cache) == cache_before, "RGB_callback_must_not_mutate_obs_cache")
                        return result
                    except BaseException:
                        self.halted = True
                        if camera in self.partial_records:
                            self.partial_records[camera]["status"] = "failed_preserved"
                        raise
                    finally:
                        self.active = None
                self.observables[camera + "_image"]._sensor = sensor
            yield self
        finally:
            for camera, original in self._old_sensors.items():
                self.observables[camera + "_image"]._sensor = original
            if had_render:
                self.sim.render = own_render
            elif "render" in vars(self.sim):
                delattr(self.sim, "render")
            if getattr(self.sim, "_sru_native_camera_pair_owner", None) is self:
                delattr(self.sim, "_sru_native_camera_pair_owner")
            self._old_render = None; self._old_sensors.clear(); self.armed = None

    @contextmanager
    def during_parent_step(self, env_step_index):
        need(self._old_render is not None and self.armed is None and type(env_step_index) is int
             and env_step_index >= 0 and not self.halted, "explicit_existing_parent_step_scope")
        self.armed = env_step_index
        try:
            yield  # caller alone invokes its already scheduled ordinary step
        finally:
            self.armed = None

    def _render(self, *args, **kwargs):
        if self.armed is None:
            return self._old_render(*args, **kwargs)
        camera = kwargs.get("camera_name")
        if self.active is None or camera != self.active["camera"]:
            self.halted = True
            raise ValueError("query_side_render_is_not_sensor_capture")
        need(not kwargs.get("depth", False) and not kwargs.get("segmentation", False)
             and kwargs.get("width") == kwargs.get("height") == 256, "native_RGB_only")
        self.active["renders"] += 1
        source = copy.deepcopy(self.clock_state())
        need(set(source) == {"sim_time", "physical_state"} and np.asarray(source["sim_time"]).shape == ()
             and np.asarray(source["sim_time"]).dtype.kind == "f" and np.isfinite(source["sim_time"]), "actual_sensor_clock_state")
        native = self._old_render(*args, **kwargs)
        convention = self.conventions[camera]
        clean = np.array(native[::convention], copy=True)
        self.partial_records[camera] = {"status": "capturing", "clean": clean.copy(), "nativeperturb": None,
            "source": copy.deepcopy(source), "sensor_env_step_index": self.armed, "native_qualified": False}
        need(clean.dtype == np.uint8 and clean.shape == (256, 256, 3), "native_sensor_image_ABI")
        need(same(source, self.clock_state()), "native_C_render_changed_physics")
        with self.optics.temporary("nativeperturb"):
            treated = np.array(self._old_render(*args, **kwargs)[::convention], copy=True)
            self.partial_records[camera]["nativeperturb"] = treated.copy()
            need(treated.shape == clean.shape and treated.dtype == clean.dtype, "paired_native_RGB_ABI")
            need(same(source, self.clock_state()), "camera_T_render_changed_sensor_source")
        with self.optics.temporary("clean"):
            restored = np.array(self._old_render(*args, **kwargs)[::convention], copy=True)
        need(same(clean, restored) and same(clean, native[::convention]) and same(source, self.clock_state()), "restored_C_or_original_return_buffer_changed")
        self.records[camera] = {"clean": clean, "nativeperturb": treated, "source": source,
                                "sensor_env_step_index": self.armed, "sensor_time": copy.deepcopy(source["sim_time"])}
        self.partial_records[camera]["status"] = "candidate_sensor_transaction_restored"
        return native  # exact original object; no copied/perturbed parent frame

    def supplier(self, parent_observation):
        """Build state_pairing's supplier from actual sampled buffers, no render."""
        need(not self.halted and all(c in self.records for c in CAMERAS), "two_actual_sensor_records_required")
        records = copy.deepcopy(self.records)
        steps = {r["sensor_env_step_index"] for r in records.values()}
        need(len(steps) == 1, "different_camera_control_steps_require_explicit_protocol_extension")
        for camera in CAMERAS:
            need(same(parent_observation[camera + "_image"], records[camera]["clean"]), "parent_cached_observation_not_this_sensor_output")
        wrist = records["robot0_eye_in_hand"]
        need(same(wrist["clean"], wrist["nativeperturb"]), "agentview_only_variant_changed_wrist")
        source = {camera: records[camera]["source"]["physical_state"] for camera in CAMERAS}
        times = {camera: records[camera]["sensor_time"] for camera in CAMERAS}
        pair = {}
        for role in ("clean", "nativeperturb"):
            observation = copy.deepcopy(parent_observation)
            for camera in CAMERAS:
                observation[camera + "_image"] = records[camera][role].copy()
            pair[role] = {"observation": observation, "physical_source": copy.deepcopy(source),
                          "sensor_env_step_index": next(iter(steps)), "sensor_time_actual": copy.deepcopy(times)}
        return lambda: copy.deepcopy(pair)
