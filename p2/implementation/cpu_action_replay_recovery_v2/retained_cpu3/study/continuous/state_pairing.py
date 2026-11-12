"""Small CPU contracts for continuous same-state paired observation.

No environment/model construction, reset, step, forward, render, checkpoint,
cloud, or neural capture implementation. The existing root loop supplies a
native pair callback and advances the root only after this function returns.
Actual execution remains separately authorized/qualified. Generic physical
branch restore is deliberately not a prerequisite for observation diagnostics.
"""
from __future__ import annotations

from collections import deque
import copy
from dataclasses import dataclass
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
from openpi.eval import selective_reference as R
from observation_witness import observation_binding, observation_identity


DIAGNOSTIC_STATES = (
    "physics.current", "render.parameters", "root.clock", "root.action_buffer", "root.observation",
    "rng.python_global", "rng.numpy_global",
)
REPLAY_STATES = (*DIAGNOSTIC_STATES, "environment", "controller", "observables", "policy_rng",
                 "renderer.hidden", "physics.unregistered", "model.unregistered", "callable.closure_state")
FACTORY = "docs/experiments/selective-correction-20261012-v1/execution/B/runtime/environment_factory.py"


class PairingError(RuntimeError):
    pass


def need(ok, code):
    if not ok:
        raise PairingError(code)


def mapping_binding(name, mapping, *, temporary=False):
    """Bind caller-owned data in place; do not replace its root object."""
    need(type(mapping) is dict and not any(k.startswith("attempt") for k in mapping),
         "external_attempt_witness_must_not_be_restorable")
    R._identity(mapping)
    keys = set(mapping)
    def snapshot():
        need(set(mapping) == keys, "declared_mapping_fields_changed")
        return copy.deepcopy(mapping)
    def restore(value):
        need(set(value) == keys and set(mapping) == keys, "declared_mapping_restore_fields_changed")
        mapping.clear(); mapping.update(copy.deepcopy(value))
    return R.StateBinding(name, snapshot, restore, temporary)


def passive_witness_binding(name, read):
    """Observe protected state without installing any physical-state setter.

    For render-only diagnosis, a changed physical/controller witness means stop,
    not 'repair' by restoring a guessed subset. No generic replay is required.
    """
    need(callable(read), "passive_witness_reader_required")
    def snapshot():
        value = copy.deepcopy(read())
        R._identity(value)
        return value
    def verify_unchanged(saved):
        need(R._identity(snapshot()) == R._identity(saved), "passive_witness_changed_no_physical_repair")
    return R.StateBinding(name, snapshot, verify_unchanged)


def action_buffer_binding(queue):
    """Expose campaign_episode's existing deque, including unissued commands.

    No new action scheduler: campaign_episode uses actions[:5] and popleft.
    Preserve exact native arrays, queue order and maxlen; never cast actions.
    """
    need(isinstance(queue, deque), "actual_existing_action_deque_required")
    maxlen = queue.maxlen
    def snapshot():
        need(queue.maxlen == maxlen, "action_buffer_ABI_changed")
        return {"items": copy.deepcopy(tuple(queue)), "maxlen": maxlen}
    def restore(value):
        need(value["maxlen"] == queue.maxlen == maxlen, "action_buffer_maxlen_changed")
        queue.clear(); queue.extend(copy.deepcopy(value["items"]))
    return R.StateBinding("root.action_buffer", snapshot, restore)


def source_dependencies(repo_root, manifest_path=None):
    """Verify only the exact listed small source files; no runtime payload."""
    root = Path(repo_root).resolve()
    path = Path(manifest_path or Path(__file__).with_name("SNAPSHOT-DEPENDENCIES-v1.json"))
    manifest = json.loads(path.read_bytes())
    for row in manifest["files"]:
        p = root / row["path"]
        need(p.resolve().is_relative_to(root) and p.is_file() and not p.is_symlink(), "exact_source_path")
        raw = p.read_bytes()
        need(len(raw) == row["bytes"] and hashlib.sha256(raw).hexdigest() == row["sha256"], "snapshot_dependency_changed")
    return manifest


def existing_native_bindings(repo_root, official, policy_rng, *, synthetic=False):
    """Reuse existing explicit controller/interpolator/observable inventories.

    The object must already exist. This does not call the old make_environment
    or any old worker. It is optional for observation diagnostics and NOT proof
    that a physical continuation from restored state is equivalent.
    """
    source_dependencies(repo_root)
    path = Path(repo_root) / FACTORY
    spec = importlib.util.spec_from_file_location("_sru_existing_state_factory", path)
    factory = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = factory
    spec.loader.exec_module(factory)
    return factory.build_bindings(official, policy_rng, synthetic=synthetic)


@dataclass(frozen=True)
class DeclaredSnapshot:
    owner: object
    source_id: str
    states: dict
    state_id: str
    step_attempts_observed: int


class DeclaredState:
    """Compose the existing StateBinding/ReferenceTransaction, not a simulator.

    The monotonic witness must count attempted env.step/sim.step/forward/reset
    calls and live outside every restorable state. Coverage is explicit; missing
    hidden state is not inferred unchanged. A receipt never proves full replay.
    """
    def __init__(self, bindings, *, source_id, step_attempts, generators=None, synthetic=False):
        self.bindings = tuple(bindings)
        need(source_id and callable(step_attempts), "source_and_monotonic_witness_required")
        need(len({b.name for b in self.bindings}) == len(self.bindings), "unique_state_bindings")
        byname = {b.name: b for b in self.bindings}
        need(set(DIAGNOSTIC_STATES)-{"rng.python_global", "rng.numpy_global"} <= byname.keys(),
             "minimal_diagnostic_state_coverage")
        need(byname["render.parameters"].allow_temporary_change and all(
            not b.allow_temporary_change for b in self.bindings if b.name != "render.parameters"),
            "only_render_parameters_may_temporarily_change")
        self.source_id, self.step_attempts = source_id, step_attempts
        self.generators, self.synthetic = dict(generators or {}), synthetic
        self._owner, self._halted = object(), False

    def transaction(self):
        need(not self._halted, "state_owner_halted")
        return R.ReferenceTransaction(self.bindings, required_states=DIAGNOSTIC_STATES,
                                      physical_step_counter=self.step_attempts, generators=self.generators)

    def capture(self):
        tx = self.transaction()
        # Reuse the transaction's pinned Python/NumPy/Generator bindings and
        # process-global lock. No second RNG implementation or random draw.
        with tx:
            states = {b.name: copy.deepcopy(b.snapshot()) for b in tx._bindings}
            result = DeclaredSnapshot(self._owner, self.source_id, states, R._identity(states), self.step_attempts())
        return result

    def restore_for_synthetic_check(self, snapshot):
        """Exercise actual restore callbacks on CPU fixtures only.

        Production physical-fork restore is NOT enabled by a caller boolean
        receipt. Native restore+next-step/observation parity is still pending.
        This method is never called by same_state_pair or TwelveQueryRecorder.
        """
        need(self.synthetic is True and not self._halted, "native_physical_restore_not_qualified")
        need(snapshot.owner is self._owner and snapshot.source_id == self.source_id, "foreign_snapshot_owner_or_source")
        need(R._identity(snapshot.states) == snapshot.state_id, "snapshot_payload_mutated")
        before = self.step_attempts()
        tx = self.transaction()
        failures = []
        # Explicit restoration allows intentional changes only in this mock
        # test. All state is verified afterward; external attempt count stays.
        for binding in reversed(tx._bindings):
            try:
                binding.restore(copy.deepcopy(snapshot.states[binding.name]))
            except BaseException as error:
                failures.append({"state": binding.name, "exception_type": type(error).__name__})
        try:
            after = {b.name: b.snapshot() for b in tx._bindings}
            need(self.step_attempts() == before, "restore_performed_physical_activity")
            need(R._identity(after) == snapshot.state_id, "declared_restore_not_exact")
        except BaseException as error:
            failures.append({"state": "verification", "exception_type": type(error).__name__})
        if failures:
            self._halted = True
            raise PairingError("synthetic_restore_failed_owner_halted")
        return {"declared_fields_exact": True, "native_replay_qualified": False,
                "step_attempts_unchanged": True, "synthetic_only": True}


def same_state_pair(state, pair_supplier, *, query_env_step, query_time):
    """Read paired native views without stepping the root or using a model.

    Supplier returns clean/nativeperturb views with observation, physical_source
    (sensor-producing state, not just query state), sensor_env_step_index, and
    sensor_time_actual keyed by the two real consumed cameras. It must use the
    existing native operator/step sensor capture; this contract renders nothing.
    """
    before = state.capture()
    tx = state.transaction()
    try:
        with tx:
            pair = copy.deepcopy(pair_supplier())
            need(set(pair) == {"clean", "nativeperturb"}, "two_explicit_view_roles")
            ids = []
            for role in ("clean", "nativeperturb"):
                view = pair[role]
                need(set(view) == {"observation", "physical_source", "sensor_env_step_index", "sensor_time_actual"},
                     "explicit_view_source_fields")
                need(type(view["sensor_env_step_index"]) is int and 0 <= view["sensor_env_step_index"] <= query_env_step,
                     "sensor_control_clock_after_query")
                times = view["sensor_time_actual"]
                need(set(times) == {"agentview", "robot0_eye_in_hand"}, "two_actual_camera_times")
                for tau in times.values():
                    actual = np.asarray(tau)
                    need(actual.shape == () and actual.dtype.kind == "f" and np.isfinite(actual)
                         and float(actual) <= float(np.asarray(query_time)), "actual_sensor_time_after_query_or_invalid")
                ids.append(R._identity({k: view[k] for k in ("physical_source", "sensor_env_step_index", "sensor_time_actual")}))
                observation_identity(view["observation"])
            need(ids[0] == ids[1], "clean_treated_different_sensor_source_state_or_time")
        return pair, {"schema": "sru-same-state-observation-pair-v1", "status": "declared_state_restored",
            "query_physical_snapshot_id": R._identity(before.states["physics.current"]),
            "sensor_source_physical_snapshot_id": R._identity(pair["clean"]["physical_source"]),
            "declared_runtime_snapshot_id": before.state_id, "full_runtime_snapshot_id": None,
            "declared_binding_groups": sorted(before.states),
            "unobserved_for_full_replay": sorted(set(REPLAY_STATES)-set(before.states)),
            "transaction": copy.deepcopy(tx.report), "full_replay_qualified": False,
            "native_render_parity_qualified_by_this_module": False,
            "root_trajectory_advanced": False, "PG_AE_capture_status": "not_called_pending_native_qualification"}
    except BaseException as error:
        state._halted = True
        # Safe structured fields only; preserve the existing transaction report.
        raise PairFailure({"status": "failed", "exception_type": type(error).__name__,
                           "transaction": copy.deepcopy(tx.report), "automatic_retry": False}) from error


class PairFailure(PairingError):
    def __init__(self, report):
        self.report = report
        super().__init__("paired_observation_failed_preserved")


class TwelveQueryRecorder:
    """Record window slots in the existing root loop; never run a rollout.

    emit(event, metadata, arrays)->durable receipt ID is supplied by the owner.
    Query slots are not environment steps or diffusion calls. Missing slots stay
    in the denominator. No inference callback, simulation callback or scheduler.
    """
    def __init__(self, *, root_trajectory_id, first_query_index, first_env_step_index, startup_controls, emit):
        need(root_trajectory_id and callable(emit), "trajectory_and_durable_writer_required")
        need(all(type(v) is int and v >= 0 for v in (first_query_index, first_env_step_index, startup_controls)),
             "explicit_nonnegative_clocks")
        self.root_id, self.emit, self.startup = root_trajectory_id, emit, startup_controls
        self.rows = [{"window_slot": i, "planned_query_index": first_query_index+i,
                      "planned_env_step_index": first_env_step_index+5*i, "status": "planned"} for i in range(12)]
        self.next, self.closed = 0, False
        # Construct each parent's recorder before its rollout, not on reaching
        # the window. Even a parent that never reaches slot0 has a durable row.
        self.planned_receipt = None
        try:
            receipt = self.emit("WINDOW_PLANNED", self.summary(), {})
            need(isinstance(receipt, str) and receipt, "durable_planned_window_receipt_required")
            self.planned_receipt = receipt
        except BaseException:
            self.closed = True
            raise

    def record(self, *, window_slot, query_index, env_step_index, absolute_env_step_index, query_time_actual,
               state, pair_supplier):
        need(not self.closed and self.next < 12, "next_frozen_window_slot_only")
        row = self.rows[self.next]
        stage = "clock_and_buffer_validation"
        try:
            need(type(window_slot) is int and window_slot == self.next, "next_frozen_window_slot_only")
            need(query_index == row["planned_query_index"] and env_step_index == row["planned_env_step_index"]
                 and absolute_env_step_index == self.startup+env_step_index, "actual_query_control_clocks_disagree")
            need(np.asarray(query_time_actual).shape == () and np.isfinite(query_time_actual), "actual_scalar_query_time")
            bindings = {b.name: b for b in state.bindings}
            current_clock = bindings["root.clock"].snapshot()
            need(all(current_clock.get(k) == v for k, v in (("query_index", query_index),
                 ("env_step_index", env_step_index), ("absolute_env_step_index", absolute_env_step_index))) and
                 "query_time_actual" in current_clock and
                 R._identity(np.asarray(current_clock["query_time_actual"])) == R._identity(np.asarray(query_time_actual)),
                 "record_clocks_not_bound_to_current_root")
            need(bindings["root.action_buffer"].snapshot()["items"] == (), "query_cannot_replace_pending_root_actions")
            clocks = {"root_trajectory_id": self.root_id, "window_slot": window_slot, "query_index": query_index,
                      "env_step_index": env_step_index, "absolute_env_step_index": absolute_env_step_index,
                      "trajectory_role": "root_clean_visited", "capture_phase": "query_input_same_sensor_source",
                      "query_time_actual_id": R._identity(np.asarray(query_time_actual)),
                      "flow_call_index": None, "flow_time_actual": None, "action_slot": None}
            row.update(status="reached", **clocks)
            stage = "paired_input_capture"
            receipt = self.emit("QUERY_REACHED", dict(row), {})
            need(isinstance(receipt, str) and receipt, "durable_reached_receipt_required")
            pair, proof = same_state_pair(state, pair_supplier, query_env_step=env_step_index, query_time=query_time_actual)
            arrays = {"query_time_actual": np.asarray(query_time_actual).copy(), "views": pair}
            receipt = self.emit("PAIRED_INPUT", {**clocks, **proof, "branch_roles": ["clean", "nativeperturb"],
                                 "shadow_role": "same_state_observation_only"}, arrays)
            need(isinstance(receipt, str) and receipt, "durable_pair_receipt_required")
            row.update(status="recorded", pair_receipt=receipt)
            self.next += 1
            return pair, proof
        except BaseException as error:
            row.update(status="failed", failure={"phase": stage, "exception_type": type(error).__name__,
                       "report": getattr(error, "report", None), "automatic_retry": False})
            self.closed = True
            for pending in self.rows[self.next+1:]:
                pending.update(status="unreached", reason="prior_technical_failure")
            receipt = self.emit("WINDOW_FAILED", self.summary(), {})
            need(isinstance(receipt, str) and receipt, "durable_window_failure_receipt_required")
            raise

    def finish(self, reason):
        need(not self.closed and reason in ("window_complete", "root_terminal", "root_failure", "deadline"), "one_explicit_window_close")
        need(reason != "window_complete" or self.next == 12, "cannot_complete_with_missing_slots")
        self.closed = True
        for row in self.rows[self.next:]:
            row.update(status="unreached", reason=reason)
        value = self.summary()
        receipt = self.emit("WINDOW_CLOSED", value, {})
        need(isinstance(receipt, str) and receipt, "durable_window_close_receipt_required")
        return value

    def summary(self):
        return {"schema": "sru-twelve-query-window-v1", "root_trajectory_id": self.root_id,
                "planned_queries": 12, "planned_receipt": self.planned_receipt, "rows": copy.deepcopy(self.rows),
                "full_replay_qualified": False, "PG_AE_capture_status": "not_called_pending_native_qualification",
                "new_model_calls_by_module": 0, "new_environment_steps_by_module": 0}
