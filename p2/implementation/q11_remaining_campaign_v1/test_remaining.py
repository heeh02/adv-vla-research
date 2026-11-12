"""Synthetic continuation and exact retained-recorder checks; no native runtime."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import time
from types import SimpleNamespace as NS
import unittest
import numpy as np
import campaign_core as P
import fake_runtime as F
import resume_core as R
from close_protocol import close_original_recorder

HERE = Path(__file__).resolve().parent


class Runtime(F.Runtime):
    def __init__(self, role, fault=None):
        super().__init__(role, fault)
        self.model_attempts = 0
    def step_snapshot(self):
        return F.state(self.steps)
    def close(self):
        self.closed = True
        if self.fault == "close":
            raise ValueError("injected_close_failure")
        if self.fault == "close_model":
            self.model_attempts += 1


def setup():
    source = F.Source()
    cached = {arm: {"actions": np.full((1, 10, 7), i/10, np.float64),
                    "raw32": np.zeros((1, 10, 32), np.float32)}
              for i, arm in enumerate(("N", "S", "R"))}
    old = Runtime("N")
    old.open(source, check=lambda: None)
    old.query(11)
    old.steps = 55
    critical = old.critical_state()
    start = P.digest({"declared": old.group_ids(), "critical": critical})
    before = P.digest(old.step_snapshot())
    old.bind_cached_commands(cached["N"]["actions"], cached["N"]["raw32"], 11, "N")
    rows = [old.step(a, query=11, slot=i, steps=55+i) for i, a in enumerate(cached["N"]["actions"][0, :5])]
    witness = R.load_witness((HERE/"RECEIVED-WITNESS.json").read_bytes())
    witness.update(recorded_complete_q11_start_identity=start,
                   critical_state_identity_recomputed=P.digest(critical),
                   first_before_physics_time_pose_identity=before,
                   N_trajectory_identity_recomputed=P.trajectory_identity(rows))
    return source, cached, witness


class Remaining(unittest.TestCase):
    def run_fake(self, fault=None, witness_change=None):
        source, cached, witness = setup()
        if witness_change:
            witness_change(witness)
        created = []
        def factory(role):
            runtime = Runtime(role, fault)
            created.append(runtime)
            return runtime
        result = R.run_remaining(source, cached, factory, witness=witness, authority=None,
            emit=lambda *_: None, deadline_epoch=time.time()+20, synthetic=True)
        return result, created

    def test_exactly_three_missing_roles210_new350_cumulative(self):
        result, created = self.run_fake()
        self.assertEqual([x.role for x in created], ["N_repeat", "S", "R"])
        self.assertEqual(result["new_environment_steps"], 210)
        self.assertEqual(result["cumulative_environment_steps"], 350)
        self.assertTrue(result["combined_physical_readout_available"])
        self.assertEqual(result["retained_N_original_status"], "FAILED_NO_RETRY")
        self.assertFalse(result["old_run_status_modified"])
        self.assertTrue(all(x.closed for x in created))

    def test_changed_cross_job_start_stops_before_target_and_S_R(self):
        result, created = self.run_fake(witness_change=lambda w: w.update(recorded_complete_q11_start_identity="0"*64))
        self.assertEqual(len(created), 1)
        self.assertEqual(result["new_environment_steps"], 65)
        self.assertEqual(result["roles"][0]["failure"]["code"], "cross_job_exact_start_critical_physics_time_pose")
        self.assertFalse(result["combined_physical_readout_available"])

    def test_changed_N_repeat_stops_before_S_R(self):
        result, created = self.run_fake("repeat")
        self.assertEqual(len(created), 1)
        self.assertEqual(result["roles"][0]["failure"]["code"], "N_repeat_exact_retained_observation_before_S_R")
        self.assertEqual(result["new_environment_steps"], 70)

    def test_close_failure_keeps_complete_observation_but_stops_next_role(self):
        result, created = self.run_fake("close")
        self.assertEqual(len(created), 1)
        self.assertEqual(result["roles"][0]["observed_continuation_steps"], 5)
        self.assertIn("close_failure", result["roles"][0])
        self.assertFalse(result["completed_remaining_roles"])

    def test_model_attempt_during_cleanup_is_counted_and_aborts(self):
        result, created = self.run_fake("close_model")
        self.assertEqual(len(created), 1)
        self.assertEqual(result["model_attempts"], 1)
        self.assertFalse(result["completed_remaining_roles"])

    def test_old_full_pilot_authority_does_not_authorize_new_time_window(self):
        source, cached, _ = setup()
        with self.assertRaisesRegex(ValueError, "new_exact_continuation_authority"):
            R.run_remaining(source, cached, lambda _: self.fail("factory must not run"),
                witness=(HERE/"RECEIVED-WITNESS.json").read_bytes(),
                authority={"direct_human_approved": True, "scope": "q11_cached_command_physical_pilot"},
                emit=lambda *_: None, deadline_epoch=time.time()+20)

    def test_real_cloud_witness_tampering_rejected(self):
        raw = (HERE/"RECEIVED-WITNESS.json").read_bytes()
        R.witness_gate(R.load_witness(raw))
        with self.assertRaisesRegex(ValueError, "exact_actual_cloud_resume_witness"):
            R.load_witness(raw+b" ")

    def test_native_budget_starts_at_prior_usage_and_stops_before_limit(self):
        witness = R.load_witness((HERE/"RECEIVED-WITNESS.json").read_bytes())
        budget = R.cumulative_budget(witness, time.time()+20)
        self.assertEqual(budget.counts, R.PREVIOUS_CALLS)
        self.assertEqual(budget.resets.attempted, 4)
        budget.charge("mj_step", 5250)
        self.assertEqual(budget.counts["mj_step"], 8750)
        with self.assertRaisesRegex(RuntimeError, "native_cap_mj_step"):
            budget.charge("mj_step")


def original_recorder():
    root = HERE.parents[1]/"implementation/cpu_action_replay_recovery_v3/retained_cpu3"
    name = "study/continuous/collector-v2/collector.py"
    raw = (root/name).read_bytes()
    ref = next(r for r in json.loads((root/"metadata/SOURCE_MANIFEST-v1.json").read_bytes())["files"]
               if r["path"] == name)
    assert len(raw) == ref["bytes"] and hashlib.sha256(raw).hexdigest() == ref["sha256"]
    node = next(n for n in ast.parse(raw).body if isinstance(n, ast.ClassDef) and n.name == "ScheduledRecorder")
    def need(ok, code):
        if not ok:
            raise ValueError(code)
    namespace = {"copy": copy, "S": NS(need=need)}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(root/name), "exec"), namespace)
    return namespace["ScheduledRecorder"]


class Closure(unittest.TestCase):
    def test_original13_slot_schedule_retained_and_post_action_close_valid(self):
        events = []
        plan = {"root_trajectory_id": P.ROOT, "max_policy_steps": 520,
                "recorded_query_indices": list(range(13)),
                "tiers": {str(q): {} for q in range(13)}}
        rec = original_recorder()(plan, lambda *_: "synthetic-receipt")
        rec.next = 12
        # Original ScheduledBridge can still arm q12 at policy step60.
        self.assertEqual(rec.rows[rec.next]["planned_env_step_index"], 60)
        result = close_original_recorder(rec, rec, terminal=False, role="N_repeat", env_steps=70,
                                         emit=lambda *args: events.append(args))
        self.assertTrue(rec.closed)
        self.assertEqual(result["rows"][12]["status"], "unreached")
        self.assertEqual(result["rows"][12]["reason"], "root_failure")
        self.assertTrue(events[0][1]["target_five_commands_observed"])
        self.assertFalse(events[0][1]["environment_failure_inferred"])

    def test_partial_failure_does_not_receive_five_command_annotation(self):
        plan = {"root_trajectory_id": P.ROOT, "max_policy_steps": 520,
                "recorded_query_indices": list(range(13)), "tiers": {str(q): {} for q in range(13)}}
        rec = original_recorder()(plan, lambda *_: "synthetic-receipt")
        rec.next = 10
        events = []
        close_original_recorder(rec, rec, terminal=False, role="S", env_steps=55,
                                emit=lambda *args: events.append(args))
        self.assertEqual(events, [])


if __name__ == "__main__":
    unittest.main()
