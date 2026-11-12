"""Passive source-state witness for official Sensor Noise reference pairs.

Only observes renders already requested by the native sensor. It neither steps,
renders an extra image, nor implements/replicates the corruption transform.
Pairing is rejected unless witnessed raw frames and current physical identity
agree. Actual simulator/observer qualification is a separate B requirement.
"""

from contextlib import contextmanager
import copy

import numpy as np

from openpi.eval.selective_reference import _identity


class SensorPairUnavailable(ValueError):
    pass


class NativeRenderWitness:
    """Own one instance render hook for a single existing transition.

    physical_snapshot is a passive explicit data tree containing the native
    simulation time and physical state. Callers must declare its actual field
    coverage; this object cannot discover unobserved simulator internals.
    convention is the actual native sensor row convention, never guessed.
    """

    def __init__(self, sim, physical_snapshot, *, convention, query_identity):
        if convention not in (-1, 1) or type(convention) is not int:
            raise ValueError("An explicit native image row convention is required")
        if not callable(physical_snapshot) or not query_identity:
            raise ValueError("Passive physical snapshot and query identity are required")
        self.sim = sim
        self.snapshot = physical_snapshot
        self.convention = convention
        self.query_identity = copy.deepcopy(query_identity)
        self.records = {}
        self._active = False
        self._used = False
        self._finished = False

    @contextmanager
    def observe_native_transition(self):
        if self._used or getattr(self.sim, "_selective_render_witness_active", False):
            raise RuntimeError("Render witness is single-use and instance-exclusive")
        self._used = True
        original = self.sim.render
        had_own = "render" in vars(self.sim)
        own = vars(self.sim).get("render")
        self.sim._selective_render_witness_active = True
        self._active = True

        def witnessed(*args, **kwargs):
            camera = kwargs.get("camera_name")
            tracked = camera in ("agentview", "robot0_eye_in_hand")
            if tracked and (kwargs.get("depth", False) or kwargs.get("segmentation", False)):
                raise SensorPairUnavailable("This witness is bound to RGB sensors only")
            before = copy.deepcopy(self.snapshot()) if tracked else None
            result = original(*args, **kwargs)
            if not tracked:
                return result
            after = copy.deepcopy(self.snapshot())
            if _identity(before) != _identity(after):
                raise SensorPairUnavailable("Native render changed the declared physical state")
            image = np.asarray(result)
            if image.ndim != 3 or image.shape[-1] != 3 or image.dtype != np.uint8:
                raise SensorPairUnavailable("Expected native uint8 HxWx3 sensor output")
            previous = self.records.get(camera)
            self.records[camera] = {
                "image": np.array(image[::self.convention], copy=True),
                "physical_state": before,
                "physical_identity": _identity(before),
                "native_render_count": 1 if previous is None else previous["native_render_count"] + 1,
            }
            return result  # preserve the very original native object

        self.sim.render = witnessed
        try:
            yield self
            self._finished = True
        finally:
            if had_own:
                self.sim.render = own
            else:
                delattr(self.sim, "render")
            delattr(self.sim, "_selective_render_witness_active")
            self._active = False

    def current_clean_pair(self, pre_sensor_observation, treatment_observation, *, query_identity):
        """Return existing uncorrupted frames only after source/current match.

        Inputs are the original canonical step observer's pre/post sensor obs.
        No source-state rollback, re-corruption, resampling, or interpolation.
        """
        if self._active or not self._finished:
            raise SensorPairUnavailable("A successful native transition must finish before pairing")
        if _identity(query_identity) != _identity(self.query_identity):
            raise SensorPairUnavailable("Witness belongs to another query")
        current = copy.deepcopy(self.snapshot())
        current_id = _identity(current)
        images, receipt = {}, {}
        for camera in ("agentview", "robot0_eye_in_hand"):
            record = self.records.get(camera)
            key = camera + "_image"
            if record is None or key not in pre_sensor_observation or key not in treatment_observation:
                raise SensorPairUnavailable("Both original camera frames must be witnessed")
            if record["physical_identity"] != current_id:
                raise SensorPairUnavailable("Sensor source-state differs from query current-state")
            if _identity(pre_sensor_observation[key]) != _identity(record["image"]):
                raise SensorPairUnavailable("Canonical pre-corruption frame is not the witnessed native frame")
            if camera == "robot0_eye_in_hand" and _identity(pre_sensor_observation[key]) != _identity(treatment_observation[key]):
                raise SensorPairUnavailable("Official Sensor Noise contract does not corrupt the wrist image")
            actual = np.asarray(treatment_observation[key])
            if actual.shape != record["image"].shape or actual.dtype != record["image"].dtype:
                raise SensorPairUnavailable("Treatment frame changed the original image ABI")
            images[key] = np.array(record["image"], copy=True)
            receipt[camera] = {
                "source_physical_identity": record["physical_identity"],
                "query_physical_identity": current_id,
                "raw_clean_identity": _identity(images[key]),
                "raw_treatment_identity": _identity(actual),
                "native_render_count": record["native_render_count"],
            }
        # Preserve other current observation fields; the adapter still owns its
        # usual deepcopy/tokenization/normalization and source qualification.
        clean = copy.deepcopy(treatment_observation)
        clean.update(images)
        return clean, {
            "schema": "selective-sensor-current-pair-v1",
            "query_identity": copy.deepcopy(query_identity),
            "cameras": receipt,
            "extra_render_calls": 0, "extra_physical_steps": 0,
            "corruption_replayed": False,
            "scope": "declared physical fields and observed frames only",
            "native_renderer_qualified": False,
        }
