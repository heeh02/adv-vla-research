"""Glue existing camera tap to TwelveQueryRecorder; never invoke step/render/model.

Caller owns the original loop, actual clock updates, consumed observation, and
durable writer. Install the existing tap once. Wrap ordinary steps BEFORE their
sensor callbacks, then call record_query before the parent consumes that input.
No generic replay, new scheduler, or production qualification is provided here.
"""
from contextlib import contextmanager
import copy
from pathlib import Path
import sys

import numpy as np

# These are exact sibling candidates, not a reconstructed environment package.
_CONTINUOUS = str(Path(__file__).resolve().parent.parent)
if _CONTINUOUS not in sys.path:
    sys.path.insert(0, _CONTINUOUS)
import state_pairing as S
from camera_sensor_pair import CAMERAS, NativeCameraPairTap


class NativePairBridge:
    def __init__(self, recorder, state, tap):
        S.need(isinstance(tap, NativeCameraPairTap), "reuse_existing_native_camera_tap_only")
        S.need(isinstance(recorder, S.TwelveQueryRecorder) and not recorder.closed
               and recorder.next == 0 and recorder.planned_receipt, "existing_durable_planned_recorder_required")
        S.need(recorder.startup == 10 and len(recorder.rows) == 12
               and [r["planned_env_step_index"] for r in recorder.rows] == list(range(55, 115, 5)),
               "current_pilot_12_slots_n55_and_10_waits_only")
        self.rec, self.state, self.tap = recorder, state, tap
        self.bindings = {b.name: b for b in state.bindings}
        self.pending = None
        self.failure_in_memory = None  # retained even if the durable writer fails
        self._in_step = False

    def _clock(self):
        return self.bindings["root.clock"].snapshot()

    def _emit(self, event, metadata, arrays):
        receipt = self.rec.emit(event, metadata, arrays)
        S.need(isinstance(receipt, str) and receipt, "durable_bridge_receipt_required")
        return receipt

    def _failure(self, phase, error, attempted_slot):
        # Keep actual per-camera source/index on stale/partial records. They are
        # never relabelled as the failed query's successful input.
        evidence = {"sensor_attempt_records": copy.deepcopy(self.tap.partial_records)}
        failure = {"phase": phase, "exception_type": type(error).__name__,
                   "report": copy.deepcopy(getattr(error, "report", None)),
                   "automatic_retry": False, "attempted_window_slot": attempted_slot}
        self.failure_in_memory = {"metadata": failure, "arrays": evidence}
        self.tap.halted = True
        if not self.rec.closed:
            if attempted_slot is None:
                self.rec.finish("root_failure")  # no query attempt: all remaining slots unreached
            else:
                row = self.rec.rows[self.rec.next]
                row.update(status="failed", failure=failure)
                for later in self.rec.rows[self.rec.next + 1:]:
                    later.update(status="unreached", reason="prior_technical_failure")
                self.rec.closed = True
                self._emit("WINDOW_FAILED", self.rec.summary(), {})
        self._emit("BRIDGE_FAILED", {**failure, "window": self.rec.summary()}, evidence)

    @contextmanager
    def parent_step(self, *, produces_env_step_index, absolute_env_step_index):
        """Arm before ONE existing env.step; caller updates actual n before exit.

        produces_env_step_index is the count AFTER this pending step, excluding
        the 10 waits. Only n=55,60,...,110 arm the tap; other steps pass through.
        The context yields an arm boolean, not an observation or an action.
        """
        S.need(not self.rec.closed and not self._in_step, "open_bridge_one_parent_step_only")
        row = self.rec.rows[self.rec.next] if self.rec.next < 12 else None
        armed = row is not None and produces_env_step_index == row["planned_env_step_index"]
        slot = self.rec.next if armed else None
        try:
            clock = self._clock()
            S.need(type(produces_env_step_index) is int and type(absolute_env_step_index) is int
                   and produces_env_step_index == clock["env_step_index"] + 1
                   and absolute_env_step_index == clock["absolute_env_step_index"] + 1
                   and absolute_env_step_index == self.rec.startup + produces_env_step_index,
                   "pending_step_must_match_actual_root_counts")
            S.need(self.pending is None, "paired_sensor_record_not_yet_consumed")
            if row is not None:
                S.need(produces_env_step_index <= row["planned_env_step_index"], "planned_query_was_skipped")
            self._in_step = True
            if armed:
                old_records = {c: self.tap.records.get(c) for c in CAMERAS}
                before_time = copy.deepcopy(self.tap.clock_state()["sim_time"])
                with self.tap.during_parent_step(produces_env_step_index):
                    yield True  # only the caller invokes its already scheduled env.step
                after_time = self.tap.clock_state()["sim_time"]
                S.need(np.asarray(before_time).dtype.kind == np.asarray(after_time).dtype.kind == "f"
                       and np.isfinite(before_time) and np.isfinite(after_time)
                       and float(after_time) >= float(before_time), "actual_step_clock_not_monotonic")
                for camera in CAMERAS:
                    actual = self.tap.records.get(camera)
                    S.need(actual is not None and actual is not old_records[camera]
                           and actual["sensor_env_step_index"] == produces_env_step_index,
                           "each_camera_must_really_sample_this_armed_cycle")
                    S.need(float(before_time) <= float(actual["sensor_time"]) <= float(after_time),
                           "camera_time_outside_actual_producing_step")
                self.pending = {"window_slot": slot, "env_step_index": produces_env_step_index,
                                "records": {c: self.tap.records[c] for c in CAMERAS}}
            else:
                yield False
            clock = self._clock()
            S.need(clock["env_step_index"] == produces_env_step_index
                   and clock["absolute_env_step_index"] == absolute_env_step_index,
                   "caller_must_publish_actual_completed_step_counts")
        except BaseException as error:
            self._failure("native_sensor_cycle" if armed else "ordinary_parent_step", error, slot)
            raise
        finally:
            self._in_step = False

    def record_query(self, parent_observation, *, query_index, env_step_index,
                     absolute_env_step_index, query_time_actual):
        """Consume native buffers for the actual parent input; no query re-render.

        Invoke at every parent query boundary BEFORE inference/queue refill.
        Before the frozen window returns None. At window slots, the unchanged
        recorder checks actual k/n/absolute clocks and the empty action deque.
        """
        S.need(not self.rec.closed and not self._in_step, "query_outside_step_and_open_window")
        if self.rec.next == 12:
            return None
        row = self.rec.rows[self.rec.next]
        try:
            if env_step_index < row["planned_env_step_index"]:
                S.need(query_index < row["planned_query_index"] and self.pending is None,
                       "early_query_disagrees_with_predeclared_window")
                return None
            def supplied():
                S.need(self.pending is not None and self.pending["window_slot"] == self.rec.next
                       and self.pending["env_step_index"] == env_step_index, "query_requires_prior_native_sensor_cycle")
                S.need(np.asarray(query_time_actual).dtype.kind == "f", "actual_query_time_must_be_typed_float")
                S.need(S.R._identity(self.bindings["root.observation"].snapshot()) == S.R._identity(parent_observation),
                       "pair_input_not_bound_actual_parent_observation")
                S.need(all(self.tap.records.get(c) is self.pending["records"][c] for c in CAMERAS),
                       "sensor_record_replaced_before_parent_consumption")
                return self.tap.supplier(parent_observation)()
            result = self.rec.record(window_slot=self.rec.next, query_index=query_index,
                env_step_index=env_step_index, absolute_env_step_index=absolute_env_step_index,
                query_time_actual=query_time_actual, state=self.state, pair_supplier=supplied)
            self.pending = None
            return result
        except BaseException as error:
            self._failure("query_consumed_sensor_pair", error, self.rec.next)
            raise

    def finish(self, reason):
        """Caller reports terminal/failure/deadline; never turn unused frames into queries."""
        S.need(not self._in_step and not self.rec.closed, "one_bridge_close_outside_step")
        if self.pending is not None:
            self._emit("UNCONSUMED_SENSOR", {"window_slot": self.pending["window_slot"],
                "reason": reason, "query_reached": False, "paired_query": False},
                {"sensor_attempt_records": copy.deepcopy(self.tap.partial_records)})
        value = self.rec.finish(reason)
        self.pending = None
        return value
