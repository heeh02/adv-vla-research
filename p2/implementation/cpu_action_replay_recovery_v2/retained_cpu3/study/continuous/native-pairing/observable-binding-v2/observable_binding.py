"""Bind existing camera-tap events to real Observable publication and query.

Wrap update only, calling the original once. No sensor/render/env/model calls,
resampling, queue simulation, state restoration or new camera wrapper.
"""
from contextlib import contextmanager
import copy
import hashlib
import json
from pathlib import Path

import numpy as np

SOURCE_SHA = "19ea2e0e2cfb4ebcd5911cd2526dd645859535d25fc7bc55944c54a90e6672dc"
CAMERAS = ("agentview", "robot0_eye_in_hand")


class BindingError(RuntimeError):
    pass


def need(ok, code):
    if not ok:
        raise BindingError(code)


def array_id(value):
    a = np.asarray(value)
    need(not a.dtype.hasobject and a.dtype.kind in "biuf" and np.isfinite(a).all(), "finite_native_array")
    return {"shape": list(a.shape), "dtype": a.dtype.str,
            "bytes": a.nbytes, "sha256_payload_C": hashlib.sha256(a.tobytes(order="C")).hexdigest()}


def scalar(value):
    a = np.asarray(value)
    need(a.shape == () and a.dtype.kind in "biuf" and np.isfinite(a), "finite_actual_scalar")
    return {"dtype": a.dtype.str, "value": a.item(), "bytes_hex": a.tobytes().hex()}


def tree_id(value):
    def encode(x):
        if isinstance(x, dict):
            need(all(isinstance(k, str) for k in x), "string_state_keys")
            return {k: encode(v) for k, v in sorted(x.items())}
        if isinstance(x, (tuple, list)):
            return {"sequence": type(x).__name__, "values": [encode(v) for v in x]}
        if x is None or isinstance(x, str):
            return x
        return array_id(x)
    return hashlib.sha256(json.dumps(encode(value), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def callable_id(fn):
    code = getattr(fn, "__code__", None)
    return {"module": getattr(fn, "__module__", None), "qualname": getattr(fn, "__qualname__", None),
            "code_filename": None if code is None else code.co_filename,
            "code_firstlineno": None if code is None else code.co_firstlineno}


class ObservableEventBinding:
    """Install inside the existing tap.installed() context, before original waits.

    tap must be the caller's existing NativeCameraPairTap. emit(event, metadata)
    durably records small evidence. It must not step/render or mutate inputs.
    Only actual sentinel identity filter/corrupter are supported in this version;
    any delay is read from the live instance, never invoked by this helper.
    """
    def __init__(self, tap, *, observable_module, emit):
        path = Path(observable_module.__file__).resolve(strict=True)
        need(path.is_file() and path.stat().st_size <= 128 * 1024
             and hashlib.sha256(path.read_bytes()).hexdigest() == SOURCE_SHA, "exact_Observable_source_required")
        need(callable(emit), "durable_small_emitter_required")
        need(set(tap.observables) == {c + "_image" for c in CAMERAS}, "actual_two_camera_observables")
        self.tap, self.module, self.emit = tap, observable_module, emit
        self.source = {"path": str(path), "bytes": path.stat().st_size, "sha256": SOURCE_SHA}
        self.published = {c: None for c in CAMERAS}
        self.serial = {c: 0 for c in CAMERAS}
        self.update_count = {c: 0 for c in CAMERAS}
        self.config = {}
        self.installed_now = self.halted = False
        self.failure = None
        self._saved = {}
        self.returned = None
        self.return_serial = 0

    def _snapshot(self, camera):
        o = self.tap.observables[camera + "_image"]
        need(type(o) is self.module.Observable, "actual_Observable_class_not_substitute")
        need(type(o._enabled) is bool and type(o._active) is bool and type(o._sampled) is bool,
             "actual_Observable_boolean_fields")
        need(o._corrupter is self.module.NO_CORRUPTION and o._filter is self.module.NO_FILTER,
             "nonidentity_filter_or_corrupter_not_supported")
        need(callable(o._sensor) and callable(o._delayer), "actual_sensor_delayer_callables")
        need(getattr(o._sensor, "__modality__", None) == "image", "sensor_wrapper_must_preserve_actual_modality")
        need(o.name == camera + "_image" and not o._is_number, "actual_camera_observable_name_and_array")
        need(float(o._sampling_timestep) > 0 and float(o._current_delay) >= 0
             and float(o._time_since_last_sample) >= 0, "actual_nonnegative_clock_and_positive_period")
        callbacks = (o._sensor, o._corrupter, o._filter, o._delayer)
        fixed = (callbacks, o.name, tuple(o._data_shape), float(o._sampling_timestep), o._enabled, o._active)
        if camera in self.config:
            need(fixed == self.config[camera], "Observable_config_changed_during_binding")
        return {"name": o.name, "modality": o.modality, "enabled": o._enabled, "active": o._active,
                "sampling_timestep_actual": scalar(o._sampling_timestep),
                "time_since_last_sample_actual": scalar(o._time_since_last_sample),
                "current_delay_actual": scalar(o._current_delay), "sampled_actual": o._sampled,
                "data_shape_actual": list(o._data_shape),
                "actual_NO_CORRUPTION": o._corrupter is self.module.NO_CORRUPTION,
                "actual_NO_FILTER": o._filter is self.module.NO_FILTER,
                "actual_NO_DELAY": o._delayer is self.module.NO_DELAY,
                "callbacks": {k: callable_id(v) for k, v in zip(("sensor", "corrupter", "filter", "delayer"), callbacks)}}, fixed

    def _failure(self, camera, phase, error):
        self.halted = True
        self.failure = {"camera": camera, "phase": phase, "exception_type": type(error).__name__,
                        "code": str(error) if isinstance(error, BindingError) else "unexpected_failure",
                        "reset_performed": False, "automatic_retry": False}

    @contextmanager
    def installed(self):
        need(not self.installed_now and not self.halted and getattr(self.tap, "_old_render", None) is not None,
             "install_inside_existing_tap_context")
        need(not any(hasattr(o, "_sru_observable_binding_owner") for o in self.tap.observables.values()),
             "one_update_binding_owner")
        try:
            for camera in CAMERAS:
                o = self.tap.observables[camera + "_image"]
                need(getattr(o.update, "__func__", None) is self.module.Observable.update,
                     "original_update_dispatch_required")
                snapshot, fixed = self._snapshot(camera)
                self.config[camera] = fixed
                self._saved[camera] = ("update" in vars(o), vars(o).get("update"), o.update)
                original = o.update
                def wrapped(timestep, obs_cache, force=False, camera=camera, original=original):
                    return self._update(camera, original, timestep, obs_cache, force)
                o.update = wrapped
                o._sru_observable_binding_owner = self
                self.emit("OBSERVABLE_BOUND", {"camera": camera, "source": self.source, "instance": snapshot,
                                             "native_qualification_granted": False})
            self.installed_now = True
            yield self
        finally:
            for camera, (owned, value, _) in self._saved.items():
                o = self.tap.observables[camera + "_image"]
                if owned:
                    o.update = value
                elif "update" in vars(o):
                    delattr(o, "update")
                if getattr(o, "_sru_observable_binding_owner", None) is self:
                    delattr(o, "_sru_observable_binding_owner")
            self._saved.clear()
            self.installed_now = False

    def _update(self, camera, original, timestep, obs_cache, force):
        need(not self.halted, "binding_failed_no_continuation")
        o = self.tap.observables[camera + "_image"]
        try:
            before, _ = self._snapshot(camera)
            need(type(force) in (bool, np.bool_), "actual_update_force_bool")
            dt = scalar(timestep)
            old_record = self.tap.records.get(camera)
            old_buffer = o._current_observed_value
            if self.published[camera] is not None and old_buffer is not self.published[camera]["buffer"]:
                self.published[camera] = None  # reset or publication outside this bound update
            result = original(timestep, obs_cache, force=force)  # exactly one original invocation
            after, _ = self._snapshot(camera)
            self.update_count[camera] += 1
            record = self.tap.records.get(camera)
            if record is not old_record:
                need(record is not None and o._enabled, "new_tap_event_requires_enabled_publication")
                clean = array_id(record["clean"])
                need(np.asarray(record["clean"]).dtype == np.uint8
                     and tuple(np.asarray(record["clean"]).shape) == tuple(o._data_shape), "actual_camera_image_ABI")
                need(array_id(o._current_observed_value) == clean and o.name in obs_cache
                     and array_id(obs_cache[o.name]) == clean, "sample_event_not_current_and_cache_publication")
                self.serial[camera] += 1
                event = {"camera": camera, "event_id": camera + ":" + str(self.serial[camera]),
                         "sensor_env_step_index": record["sensor_env_step_index"],
                         "sensor_time_actual": scalar(record["sensor_time"]),
                         "sensor_physical_source_identity": tree_id(record["source"]["physical_state"]),
                         "sensor_physical_identity_domain": "observable-binding-tree-v2",
                         "clean_image_identity": clean, "update_call_index": self.update_count[camera]}
                self.published[camera] = {"event": event, "buffer": o._current_observed_value, "record": record}
                disposition = "paired_event_published"
            elif o._current_observed_value is not old_buffer:
                # Unarmed sampling/reset is never joined to a previous equal-pixel event.
                self.published[camera] = None
                disposition = "unpaired_current_buffer_unavailable"
            else:
                disposition = "held_previous_buffer"
            published = self.published[camera]
            self.emit("OBSERVABLE_UPDATE", {"camera": camera, "update_call_index": self.update_count[camera],
                "timestep_actual": dt, "force_actual": bool(force), "before": before, "after": after,
                "disposition": disposition, "published_event": None if published is None else published["event"],
                "delay_queue_in_this_source": False})
            return result
        except BaseException as error:
            self._failure(camera, "original_update_and_publication", error)
            raise

    def bind_query(self, parent_observation, *, query_index, env_step_index, absolute_env_step_index, query_time_actual):
        """No render/update. Prove current buffer lineage, then returned bytes.

        Query time is never substituted for the actual sensor event time.
        Different camera event clocks remain separate in the receipt.
        """
        need(self.installed_now and not self.halted, "live_unfailed_update_binding")
        need((query_index is None or type(query_index) is int and query_index >= 0)
             and all(type(x) is int and x >= 0 for x in (env_step_index, absolute_env_step_index)),
             "actual_parent_integer_clocks")
        query_time = None if query_time_actual is None else scalar(query_time_actual)
        rows = {}
        try:
            need(self.returned is not None and parent_observation is self.returned["observation"],
                 "query_requires_exact_actual_step_return_object")
            need(self.returned["paired"] and env_step_index == self.returned["receipt"]["env_step_index"]
                 and absolute_env_step_index == self.returned["receipt"]["absolute_env_step_index"],
                 "query_clock_not_bound_to_last_step_return")
            for camera in CAMERAS:
                o = self.tap.observables[camera + "_image"]
                current, _ = self._snapshot(camera)
                p = self.published[camera]
                need(p is not None and o._enabled and o._active, "no_paired_published_event_for_camera")
                need(p is self.returned["publications"][camera], "published_event_changed_since_actual_step_return")
                need(o._current_observed_value is p["buffer"] and o.obs is p["buffer"],
                     "published_buffer_replaced_reset_or_unobserved_update")
                need(array_id(p["buffer"]) == p["event"]["clean_image_identity"], "published_buffer_mutated")
                need(self.tap.records.get(camera) is p["record"], "tap_event_changed_outside_bound_update")
                name = camera + "_image"
                need(name in parent_observation and array_id(parent_observation[name]) == p["event"]["clean_image_identity"],
                     "parent_returned_RGB_not_bound_publication")
                need(p["event"]["sensor_env_step_index"] <= env_step_index
                     and (query_time is None or p["event"]["sensor_time_actual"]["value"] <= query_time["value"]), "future_sensor_event")
                rows[camera] = {"event": copy.deepcopy(p["event"]), "instance_at_query": current,
                                "returned_image_identity": array_id(parent_observation[name]),
                                "published_buffer_object_retained": True}
            receipt = {"schema": "observable-query-binding-v2", "source": self.source,
                       "query_index": query_index, "env_step_index": env_step_index,
                       "absolute_env_step_index": absolute_env_step_index, "query_time_actual": query_time,
                       "cameras": rows, "extra_update_render_or_env_calls": 0,
                       "native_qualification_granted": False, "full_snapshot_qualified": False}
            self.emit("OBSERVABLE_QUERY_BOUND", receipt)
            return receipt
        except BaseException as error:
            self._failure(camera if "camera" in locals() else None, "query_consumed_buffer", error)
            raise

    def record_step_return(self, parent_observation, *, env_step_index, absolute_env_step_index, return_time_actual):
        try:
            return self._record_step_return(parent_observation, env_step_index=env_step_index,
                absolute_env_step_index=absolute_env_step_index, return_time_actual=return_time_actual)
        except BaseException as error:
            self._failure(None, "actual_step_return", error)
            raise

    def _record_step_return(self, parent_observation, *, env_step_index, absolute_env_step_index, return_time_actual):
        """Caller invokes immediately after its ORIGINAL env.step returns.

        Early unarmed/dummy returns may be unpaired; they never qualify a query.
        Keep the actual returned dict reference, not a pixel-matched substitute.
        No arrays are copied into receipts, and only the latest return is held.
        """
        need(self.installed_now and not self.halted, "live_unfailed_update_binding")
        need(isinstance(parent_observation, dict)
             and all(type(x) is int and x >= 0 for x in (env_step_index, absolute_env_step_index)),
             "actual_step_return_dict_and_integer_clocks")
        time_value = scalar(return_time_actual)
        rows, paired = {}, True
        for camera in CAMERAS:
            o = self.tap.observables[camera + "_image"]
            snapshot, _ = self._snapshot(camera)
            p = self.published[camera]
            if p is None or o._current_observed_value is not p["buffer"]:
                paired = False
                rows[camera] = {"status": "unpaired_current_buffer", "instance": snapshot}
                continue
            name = camera + "_image"
            need(o._enabled and o._active and o.obs is p["buffer"], "actual_return_camera_disabled_or_inactive")
            need(self.tap.records.get(camera) is p["record"], "tap_event_not_bound_to_current_update")
            need(array_id(p["buffer"]) == p["event"]["clean_image_identity"] and name in parent_observation
                 and array_id(parent_observation[name]) == p["event"]["clean_image_identity"],
                 "actual_step_return_RGB_does_not_match_published_buffer")
            need(p["event"]["sensor_env_step_index"] <= env_step_index
                 and p["event"]["sensor_time_actual"]["value"] <= time_value["value"], "future_sensor_at_step_return")
            rows[camera] = {"status": "paired", "event": copy.deepcopy(p["event"]), "instance": snapshot}
        self.return_serial += 1
        receipt = {"schema": "observable-original-step-return-v2", "return_id": self.return_serial,
                   "env_step_index": env_step_index, "absolute_env_step_index": absolute_env_step_index,
                   "return_time_actual": time_value, "cameras": rows, "paired": paired,
                   "extra_update_render_or_env_calls": 0}
        self.returned = {"observation": parent_observation, "publications": dict(self.published),
                         "paired": paired, "receipt": receipt}
        self.emit("OBSERVABLE_STEP_RETURN_BOUND", receipt)
        return receipt

    def query_gate(self, parent_observation, tap_records):
        """Collector's two-argument gate; actual k remains the collector's field."""
        need(tap_records is self.tap.records and self.returned is not None, "actual_tap_records_and_step_return")
        r = self.returned["receipt"]
        proof = self.bind_query(parent_observation, query_index=None,
            env_step_index=r["env_step_index"], absolute_env_step_index=r["absolute_env_step_index"],
            query_time_actual=None)
        # Null k/time is explicit: only the collector knows the actual query clock.
        return {"accepted": True, "binding": proof, "actual_query_clock_bound_by_collector": True,
                "extra_update_render_or_env_calls": 0, "native_qualification_granted": False}
