"""Bounded Camera-only transaction candidate; no native runtime is imported.

Supports only an explicitly paired, fixed, world-attached camera with unchanged
extended optical fields; it never switches or interprets an unknown lens mode.
The injected render/camlight and caller state bindings need separate native
qualification. Never substitutes a stored sensor frame or another rollout.
"""

from collections.abc import Mapping, Sequence
import copy
import dataclasses
import re

import numpy as np

from openpi.eval.selective_reference import ReferenceTransaction
from openpi.eval.selective_reference import StateBinding
from openpi.eval.selective_reference import _identity

EDIT = {"cam_pos": (3,), "cam_quat": (4,), "cam_fovy": ()}
STATIC = {"cam_bodyid": (), "cam_mode": (), "cam_targetbodyid": ()}
MATERIAL = ("mat_rgba", "mat_texid", "geom_matid", "geom_rgba")
LIGHT = ("light_pos", "light_dir", "light_ambient", "light_diffuse", "light_specular", "light_active")
HEADLIGHT = ("active", "ambient", "diffuse", "specular")
EXTENDED_OPTICS = ("cam_intrinsic", "cam_sensorsize", "cam_resolution", "cam_ipd", "cam_orthographic", "cam_projection")
UNOBSERVED = (
    "controller", "observables", "environment", "renderer.hidden",
    "physics.unregistered", "model.other_fields", "material.texture_bytes", "policy_rng",
)


class CameraReferenceBindingError(ValueError):
    """A source/ABI/state requirement is unavailable; do not infer a substitute."""


def _need(ok, code):
    if not ok:
        raise CameraReferenceBindingError(code)


@dataclasses.dataclass(frozen=True)
class CompiledCleanCamera:
    """Actual clean compiled field payloads, with an external pairing receipt.

    Hash strings identify the supplied evidence; this module does not fetch or
    independently verify that receipt or the source model. All six core fields
    and every exposed EXTENDED_OPTICS field must be explicitly supplied as
    NumPy scalars/arrays with the exact target model field dtype. Extended fields
    must be byte-identical to the target; differing optics need another adapter.
    """

    clean_model_sha256: str
    paired_target_model_sha256: str
    pairing_receipt_sha256: str
    camera_name: str
    body_name: str
    coordinate_frame: str
    fields: Mapping[str, np.ndarray | np.generic]


def _array_binding(name, owner, fields, *, selection=None, temporary=False):
    def snapshot():
        return {field: np.array(getattr(owner, field)[selection] if selection is not None
                                else getattr(owner, field), copy=True) for field in fields}

    def restore(saved):
        for field, value in saved.items():
            actual = getattr(owner, field)
            view = actual[selection] if selection is not None else actual
            _need(np.asarray(view).dtype == value.dtype and np.shape(view) == value.shape,
                  "restore_field_ABI_changed:" + field)
            if _identity(np.asarray(view)) == _identity(value):
                continue  # An unchanged read-only native array needs no write.
            if selection is None:
                actual[...] = value
            else:
                actual[selection] = value

    return StateBinding(name, snapshot, restore, temporary)


def _attribute_binding(name, owner, fields):
    """Protect explicit scalar/array properties, never copy a native owner."""
    def copied(value):
        _need(type(value) in (bool, int, float) or isinstance(value, np.ndarray | np.generic),
              "native_attribute_requires_scalar_or_array_binding")
        _identity(value)  # Reject object arrays without serializing their items.
        return copy.deepcopy(value)

    def snapshot():
        return {field: copied(getattr(owner, field)) for field in fields}

    def restore(saved):
        for field, value in saved.items():
            current = copied(getattr(owner, field))
            if _identity(current) == _identity(value):
                continue
            if isinstance(value, np.ndarray):
                actual = getattr(owner, field)
                _need(isinstance(actual, np.ndarray) and actual.shape == value.shape and actual.dtype == value.dtype,
                      "restore_attribute_array_ABI_changed:" + field)
                actual[...] = value
            else:
                setattr(owner, field, copied(value))  # Preserve the captured Python/NumPy scalar type.
            _need(_identity(copied(getattr(owner, field))) == _identity(value),
                  "restore_attribute_type_or_value_mismatch:" + field)

    return StateBinding(name, snapshot, restore)


def _quat_matrix_wxyz(quaternion):
    w, x, y, z = quaternion
    return np.asarray([
        1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w),
        2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w),
        2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y),
    ])


class CameraReferenceCandidate:
    """Two fresh current-state RGB renders in one guarded camera transaction.

    Required callbacks are ``native_render(sim, *, camera_name, width, height)``
    and ``camlight(sim)``. They must use the already bound native runtime; the
    latter must only refresh camera/light derived fields, never sim.forward.
    The caller supplies passive camera_id2name/body_id2name methods on model,
    an external attempted-step counter, and a ``physics.current`` StateBinding
    returning ``{'sim_time': scalar, 'physical_state': typed_data_tree}``.

    Exposed extended optics require explicit equal clean/target payloads and
    are guarded unchanged, including mode fields whose codes are not interpreted.
    Unknown/different optical fields fail pending a new adapter. Other unknown
    state stays unobserved unless separately bound; success is software restoration,
    never qualification of a real Plus renderer. The actual guard report is
    available in ``last_receipt`` even when rendering or restoration fails.
    """

    def __init__(
        self, *, sim, baseline: CompiledCleanCamera, target_model_sha256: str,
        agentview_id: int, wrist_id: int, native_render, camlight,
        physics: StateBinding, physical_step_counter,
        extra_bindings: Sequence[StateBinding] = (), generators=None,
    ):
        _need(isinstance(baseline, CompiledCleanCamera), "explicit_compiled_clean_camera_required")
        for value in (baseline.clean_model_sha256, baseline.paired_target_model_sha256,
                      baseline.pairing_receipt_sha256, target_model_sha256):
            _need(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value), "explicit_source_SHA_required")
        _need(baseline.paired_target_model_sha256 == target_model_sha256, "pair_receipt_target_model_mismatch")
        _need((baseline.camera_name, baseline.body_name, baseline.coordinate_frame)
              == ("agentview", "world", "world"), "only_agentview_world_frame_supported")
        _need(isinstance(baseline.fields, Mapping), "explicit_compiled_field_mapping_required")
        for value in baseline.fields.values():
            _need(isinstance(value, np.ndarray | np.generic), "compiled_payload_must_be_numeric_array_or_scalar")
            _identity(value)  # Reject native/opaque objects before deepcopy.
        _need(type(agentview_id) is int and type(wrist_id) is int and min(agentview_id, wrist_id) >= 0
              and agentview_id != wrist_id, "distinct_explicit_camera_ids_required")
        _need(callable(native_render) and callable(camlight), "explicit_native_callbacks_required")
        _need(isinstance(physics, StateBinding) and physics.name == "physics.current"
              and not physics.allow_temporary_change, "protected_current_physics_binding_required")
        self.sim, self.baseline = sim, copy.deepcopy(baseline)
        self.agentview_id, self.wrist_id = agentview_id, wrist_id
        self.native_render, self.camlight = native_render, camlight
        self.physics, self.step_counter = physics, physical_step_counter
        self.extra_bindings, self.generators = tuple(extra_bindings), generators
        self.last_receipt = None
        self._validate()

    def _validate(self):
        model, data = self.sim.model, self.sim.data
        extension = tuple(field for field in EXTENDED_OPTICS if hasattr(model, field))
        _need(set(self.baseline.fields) == set(EDIT) | set(STATIC) | set(extension),
              "complete_exact_compiled_camera_fields_required")
        _need(callable(getattr(model, "camera_id2name", None)) and callable(getattr(model, "body_id2name", None)),
              "native_camera_body_name_binding_unavailable")
        _need(model.camera_id2name(self.agentview_id) == "agentview"
              and model.camera_id2name(self.wrist_id) == "robot0_eye_in_hand"
              and model.body_id2name(0) == "world", "actual_camera_or_body_identity_mismatch")
        count = None
        for field, shape in {**EDIT, **STATIC}.items():
            actual, clean = getattr(model, field, None), self.baseline.fields[field]
            _need(isinstance(actual, np.ndarray) and isinstance(clean, np.ndarray | np.generic),
                  "compiled_field_array_unavailable:" + field)
            clean = np.asarray(clean)
            _need(actual.ndim == len(shape) + 1 and actual.shape[1:] == shape,
                  "compiled_camera_shape_mismatch:" + field)
            count = actual.shape[0] if count is None else count
            _need(actual.shape[0] == count and max(self.agentview_id, self.wrist_id) < count,
                  "compiled_camera_count_mismatch")
            _need(clean.shape == shape and clean.dtype == actual.dtype and np.isfinite(clean).all(),
                  "clean_compiled_shape_dtype_or_finite_mismatch:" + field)
            _need(actual.dtype in (np.dtype("float32"), np.dtype("float64")) if field in EDIT
                  else actual.dtype.kind in "iu", "compiled_field_dtype_mismatch:" + field)
        for field, expected in (("cam_bodyid", 0), ("cam_mode", 0), ("cam_targetbodyid", -1)):
            _need(int(getattr(model, field)[self.agentview_id]) == int(self.baseline.fields[field]) == expected,
                  "fixed_world_camera_compatibility_failed:" + field)
        for field in extension:
            actual, clean = getattr(model, field), self.baseline.fields[field]
            _need(isinstance(actual, np.ndarray) and actual.ndim >= 1 and actual.shape[0] == count
                  and actual.dtype.kind in "biuf" and isinstance(clean, np.ndarray | np.generic),
                  "extended_camera_field_ABI_unavailable:" + field)
            clean = np.asarray(clean)
            _need(clean.shape == actual.shape[1:] and clean.dtype == actual.dtype and np.isfinite(clean).all(),
                  "extended_clean_shape_dtype_or_finite_mismatch:" + field)
            _need(_identity(np.asarray(actual[self.agentview_id])) == _identity(clean),
                  "different_extended_optics_requires_adapter:" + field)
        _need(0 < float(self.baseline.fields["cam_fovy"]) < 180, "invalid_clean_perspective_fovy")
        quat = np.asarray(self.baseline.fields["cam_quat"])
        # Validation only: never renormalize or synthesize a camera pose.
        _need(abs(float(np.linalg.norm(quat)) - 1.0) < 1e-5, "clean_wxyz_quaternion_not_unit")
        for field, tail in (("cam_xpos", (3,)), ("cam_xmat", (9,)),
                            ("light_xpos", (3,)), ("light_xdir", (3,))):
            value = getattr(data, field, None)
            _need(isinstance(value, np.ndarray) and value.dtype.kind == "f" and value.ndim == 2
                  and value.shape[1:] == tail, "derived_renderer_ABI_unavailable:" + field)
            if field.startswith("cam_"):
                _need(value.shape[0] == count, "derived_camera_count_mismatch")
        _need(all(isinstance(getattr(model, field, None), np.ndarray) for field in MATERIAL),
              "material_guard_fields_unavailable")
        _need(all(isinstance(getattr(model, field, None), np.ndarray) for field in LIGHT),
              "light_model_guard_fields_unavailable")
        headlight = getattr(getattr(model, "vis", None), "headlight", None)
        _need(headlight is not None and all(hasattr(headlight, field) for field in HEADLIGHT),
              "headlight_guard_fields_unavailable")
        _attribute_binding("light.headlight", headlight, HEADLIGHT).snapshot()
        return count

    def _readback_clean(self):
        model, data, camera = self.sim.model, self.sim.data, self.agentview_id
        for field in EDIT:
            _need(_identity(np.asarray(getattr(model, field)[camera]))
                  == _identity(np.asarray(self.baseline.fields[field])), "clean_camera_write_not_retained:" + field)
        for field in EXTENDED_OPTICS:
            if field in self.baseline.fields:
                _need(_identity(np.asarray(getattr(model, field)[camera]))
                      == _identity(np.asarray(self.baseline.fields[field])), "extended_optics_changed:" + field)
        _need(np.allclose(data.cam_xpos[camera], self.baseline.fields["cam_pos"], atol=1e-6, rtol=1e-6),
              "camlight_did_not_refresh_world_camera_position")
        _need(np.allclose(data.cam_xmat[camera], _quat_matrix_wxyz(self.baseline.fields["cam_quat"]),
                          atol=1e-6, rtol=1e-6), "camlight_did_not_refresh_world_camera_orientation")

    def render(self, *, query_state, query_identity, width=256, height=256):
        _need(type(width) is int and type(height) is int and 0 < width <= 256 and 0 < height <= 256,
              "bounded_native_RGB_dimensions_required")
        _need(isinstance(query_identity, dict) and set(query_identity) == {"run", "case", "arm", "query"}
              and all(isinstance(query_identity[k], str) and query_identity[k] for k in ("run", "case", "arm"))
              and type(query_identity["query"]) is int and query_identity["query"] >= 0, "explicit_query_identity_required")
        _need(isinstance(query_state, dict) and set(query_state) == {"sim_time", "physical_state"},
              "current_query_state_and_time_required")
        _need(np.asarray(query_state["sim_time"]).shape == ()
              and np.asarray(query_state["sim_time"]).dtype.kind == "f"
              and np.isfinite(query_state["sim_time"]) and query_state["physical_state"] is not None,
              "finite_current_simulation_time_and_state_required")
        expected_state = _identity(query_state)
        model, data, camera = self.sim.model, self.sim.data, self.agentview_id
        other = np.asarray([i for i in range(self._validate()) if i != camera])
        bindings = [
            _array_binding("camera.target", model, EDIT, selection=camera, temporary=True),
            _array_binding("camera.derived_target", data, ("cam_xpos", "cam_xmat"), selection=camera, temporary=True),
            _array_binding("camera.other_including_wrist", model, EDIT, selection=other),
            _array_binding("camera.static", model, STATIC),
            _array_binding("camera.extended_optics", model,
                           tuple(field for field in EXTENDED_OPTICS if field in self.baseline.fields)),
            _array_binding("camera.derived_other_including_wrist", data, ("cam_xpos", "cam_xmat"), selection=other),
            _array_binding("light.derived", data, ("light_xpos", "light_xdir")),
            _array_binding("light.compiled", model, LIGHT),
            _attribute_binding("light.headlight", model.vis.headlight, HEADLIGHT),
            _array_binding("material.compiled", model, MATERIAL),
            self.physics, *self.extra_bindings,
        ]
        transaction = ReferenceTransaction(
            bindings, required_states=[*(binding.name for binding in bindings), *UNOBSERVED],
            physical_step_counter=self.step_counter, generators=self.generators,
        )
        receipt = {
            "schema": "selective-camera-reference-candidate-v2", "native_renderer_qualified": False,
            "query_identity": copy.deepcopy(query_identity), "query_state_identity": expected_state,
            "query_sim_time": float(query_state["sim_time"]),
            "query_sim_time_identity": _identity(query_state["sim_time"]), "render_source_states": {},
            "clean_model_sha256": self.baseline.clean_model_sha256,
            "target_model_sha256": self.baseline.paired_target_model_sha256,
            "pairing_receipt_sha256": self.baseline.pairing_receipt_sha256,
            "clean_compiled_fields_identity": _identity(dict(self.baseline.fields)),
            "extended_optics_fields": sorted(set(self.baseline.fields) & set(EXTENDED_OPTICS)),
            "extended_optics_contract": "explicit_clean_target_typed_exact_never_modified",
            "other_camera_light_derived_contract": "current_query_state_must_remain_typed_exact",
            "pixel_convention": "raw_native_no_rotation", "image_identities": {},
            "derived_pose_check": {"atol": 1e-6, "rtol": 1e-6, "frame": "world", "quaternion": "wxyz"},
            "restoration_check": "typed_bytes_exact",
            "render_calls_attempted": 0, "camlight_calls_attempted": 0, "exception_type": None,
            "transaction": transaction.report,
        }
        self.last_receipt = receipt
        images = {}
        try:
            with transaction:
                self._validate()
                _need(_identity(self.physics.snapshot()) == expected_state, "stale_or_foreign_query_state")
                for field in EDIT:
                    getattr(model, field)[camera] = self.baseline.fields[field]
                receipt["camlight_calls_attempted"] += 1
                self.camlight(self.sim)
                self._readback_clean()
                for name in ("agentview", "robot0_eye_in_hand"):
                    source = self.physics.snapshot()
                    _need(_identity(source) == expected_state, "render_source_not_current_query_state")
                    receipt["render_source_states"][name] = {
                        "identity": _identity(source), "sim_time": float(source["sim_time"]),
                        "sim_time_identity": _identity(source["sim_time"]),
                    }
                    receipt["render_calls_attempted"] += 1
                    image = self.native_render(self.sim, camera_name=name, width=width, height=height)
                    self._readback_clean()
                    _need(isinstance(image, np.ndarray) and image.shape == (height, width, 3)
                          and image.dtype == np.uint8, "native_RGB_ABI_mismatch")
                    images[name + "_image"] = image.copy()
                    receipt["image_identities"][name] = _identity(image)
            return images, copy.deepcopy(receipt)
        except BaseException as error:
            receipt["exception_type"] = type(error).__name__
            raise
