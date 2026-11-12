"""v2: bind the actual Plus collect.py generated['actions'][0, :5] loop.

Reuse bridge v1's camera/window handling. Only the issued-action witness changes:
no deque, fake empty buffer, model call, simulation call, or new scheduler.
"""
from contextlib import contextmanager
import copy

import numpy as np
from bridge import NativePairBridge, S


class IssuedSliceCursor:
    """Passive witness for the exact generated array and actual for-loop slot.

    Bind after the original validate_output, before enumerate. Completion is
    updated only AFTER the caller's env.step has returned and its actual steps
    count has advanced by one. Failed/partial issuance is never declared empty.
    """
    def __init__(self, read_actual_counts):
        S.need(callable(read_actual_counts), "actual_parent_count_reader_required")
        self.read_counts = read_actual_counts
        self.generated = self.actions = None
        self.generated_query = self.authorized_query = None
        self.completed = 0
        self.in_flight = None
        self.failed = False
        self.actions_id = None
        self.start_counts = None
        self.binding = S.R.StateBinding("root.action_buffer", self.snapshot, self._verify_unchanged)

    def _check_array(self):
        S.need(self.generated["actions"] is self.actions
               and S.R._identity(self.actions) == self.actions_id, "actual_generated_array_replaced_or_mutated")

    def snapshot(self):
        if self.actions is not None:
            self._check_array()
        return {"items": tuple(np.array(a, copy=True) for a in self.actions[0, self.completed:5])
                    if self.actions is not None else (),
                "representation": "actual_generated_actions_0_first5_and_completed_slot_cursor",
                "generated_query_index": self.generated_query, "actual_array_identity": self.actions_id,
                "chunk_start_actual_counts": copy.deepcopy(self.start_counts),
                "completed_slots": self.completed, "in_flight_slot": self.in_flight,
                "issuance_failed": self.failed, "authorized_query_index": self.authorized_query}

    def _verify_unchanged(self, saved):
        S.need(S.R._identity(self.snapshot()) == S.R._identity(saved), "issued_slice_changed_no_restore")

    def require_query_ready(self, query_index):
        S.need(type(query_index) is int and not self.failed and self.in_flight is None,
               "no_query_during_or_after_failed_action")
        counts = self.read_counts()
        if self.actions is None:
            S.need(query_index == 0 and counts["env_step_index"] == 0
                   and counts["absolute_env_step_index"] == 10, "only_real_first_query_has_no_previous_chunk")
        else:
            self._check_array()
            S.need(self.completed == 5 and query_index == self.generated_query + 1,
                   "all_five_actual_slots_must_finish_before_next_query")
            S.need(all(counts[k] == self.start_counts[k] + 5 for k in
                       ("env_step_index", "absolute_env_step_index")), "next_query_requires_five_actual_completed_steps")
        S.need(self.authorized_query != query_index, "one_parent_query_admission_only")

    def bind_generated(self, generated, *, query_index):
        """Pass the ORIGINAL generated dict at collect.py:95–96, no copied slice."""
        S.need(self.authorized_query == query_index and not self.failed and self.in_flight is None,
               "generated_chunk_requires_admitted_actual_query")
        S.need(self.generated_query != query_index, "actual_query_chunk_bound_once_only")
        S.need(self.actions is None or self.completed == 5, "cannot_replace_partly_issued_chunk")
        actions = generated["actions"]
        S.need(isinstance(generated, dict) and isinstance(actions, np.ndarray)
               and actions.shape == (1, 10, 7) and actions.dtype == np.float64
               and np.isfinite(actions).all(), "actual_native_adapter_chunk_required")
        self.generated, self.actions = generated, actions
        self.generated_query, self.completed = query_index, 0
        self.actions_id = S.R._identity(actions)
        self.start_counts = copy.deepcopy(self.read_counts())

    @contextmanager
    def issuing(self, slot, action):
        """Wrap the actual enumerate slot, not a fabricated 'pending=0' flag."""
        S.need(self.actions is not None and not self.failed and self.in_flight is None,
               "bound_unfailed_actual_action_chunk_required")
        self._check_array()
        expected = self.actions[0, self.completed] if self.completed < 5 else None
        S.need(type(slot) is int and slot == self.completed and expected is not None
               and isinstance(action, np.ndarray) and action.shape == expected.shape
               and action.dtype == expected.dtype and np.shares_memory(action, self.actions)
               and action.ctypes.data == expected.ctypes.data
               and S.R._identity(action) == S.R._identity(expected), "actual_for_slot_view_required")
        before = copy.deepcopy(self.read_counts())
        S.need(all(before[k] == self.start_counts[k] + self.completed for k in
                   ("env_step_index", "absolute_env_step_index")), "slot_cursor_disagrees_with_actual_completed_steps")
        self.in_flight = slot
        try:
            yield
            after = self.read_counts()
            S.need(after["env_step_index"] == before["env_step_index"] + 1
                   and after["absolute_env_step_index"] == before["absolute_env_step_index"] + 1,
                   "slot_completion_requires_actual_one_step_count")
            self._check_array()
            self.completed = slot + 1
            self.in_flight = None
        except BaseException:
            self.failed = True  # retain slot and remaining actions; never restore/advance them
            raise


class NativePairBridgeV2(NativePairBridge):
    def __init__(self, recorder, state, tap, issued_cursor):
        S.need(isinstance(issued_cursor, IssuedSliceCursor), "actual_issued_slice_cursor_required")
        binding = next((b for b in state.bindings if b.name == "root.action_buffer"), None)
        S.need(binding is issued_cursor.binding, "state_must_bind_actual_slice_not_empty_deque")
        super().__init__(recorder, state, tap)
        self.issued = issued_cursor

    @contextmanager
    def parent_step(self, *, slot, action, produces_env_step_index, absolute_env_step_index):
        try:
            with self.issued.issuing(slot, action):
                with super().parent_step(produces_env_step_index=produces_env_step_index,
                                         absolute_env_step_index=absolute_env_step_index) as armed:
                    yield armed
        except BaseException as error:
            if not self.rec.closed:
                attempted = self.rec.next if self.rec.next < 12 and produces_env_step_index == self.rec.rows[self.rec.next]["planned_env_step_index"] else None
                self._failure("actual_issued_slice", error, attempted)
            raise

    def record_query(self, parent_observation, *, query_index, **clocks):
        try:
            self.issued.require_query_ready(query_index)
            result = super().record_query(parent_observation, query_index=query_index, **clocks)
            self.issued.authorized_query = query_index
            return result
        except BaseException as error:
            if not self.rec.closed:
                # Actual query attempt, including incorrect pending-action state.
                attempted = self.rec.next if self.rec.next < 12 and clocks["env_step_index"] >= self.rec.rows[self.rec.next]["planned_env_step_index"] else None
                self._failure("actual_query_issued_slice_not_empty", error, attempted)
            raise
