"""Candidate integration with fake dynamics; never imports a native environment."""
import copy
import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch
import numpy as np
import campaign_core as P
import native_child as N
import native_facade
import pilot_contract as C
import reference_loader
import resume_core as R
import test_remaining as F

HERE = Path(__file__).resolve().parent


def authority(manifest, inputs, witness):
    return {"direct_human_approved": True, "scope": C.SCOPE, "run_id": C.RUN,
        "root_id": P.ROOT, "query_index": 11, "maximum_new_environment_steps": 210,
        "maximum_cumulative_environment_steps": 350, "previous_environment_steps": 140,
        "resume_witness_identity": witness, "new_model_calls": 0, "GPU": 0,
        "resources": C.RESOURCES, "source_manifest": manifest, "inputs_identity": inputs,
        "science_seconds": 480, "archive_seconds": 540, "allocation_seconds": 600,
        "automatic_retries": 0, "direct_user_evidence": {"bytes": 1, "sha256": "e"*64}}


class FakePhysical(F.Runtime):
    synthetic = False  # Test injects this fake explicitly at the native boundary.
    def __init__(self, context, emit):
        super().__init__(context["role"])
        self.context, self.budget = context, context.get("native_budget")
    def open(self, source, check):
        super().open(source, check)
        if self.budget:
            for name in ("construct", "reset", "set_init_state"):
                self.budget.charge(name, 2)
            self.budget.resets.attempted += 2
            self.budget.resets.successful += 2
            self.budget.charge("mj_step", 250)
            self.budget.charge("mj_forward", 250)
    def step(self, action, **kw):
        if self.budget:
            self.budget.charge("mj_step", 25)
            self.budget.charge("mj_forward", 25)
        result = super().step(action, **kw)
        physical = result["physical_outputs"]["synthetic_EEF"]
        result["physical_outputs"] = {"returned_nonimage_observations": {
            "robot0_eef_pos": physical, "robot0_eef_quat": np.array([0., 0., 0., 1.]),
            "robot0_gripper_qpos": np.zeros(2)}, "gripper_actuator_ctrl": np.array([1., -1.]),
            "contact_count": 0}
        return result


def synthetic_reference(source, cached, witness):
    prior = FakePhysical({"role": "N"}, lambda *_: None)
    prior.open(source, lambda: None)
    prior.query(11)
    prior.steps = 55
    critical = prior.critical_state()
    witness.update(recorded_complete_q11_start_identity=P.digest({"declared": prior.group_ids(), "critical": critical}),
        critical_state_identity_recomputed=P.digest(critical),
        first_before_physics_time_pose_identity=P.digest(prior.step_snapshot()))
    prior.bind_cached_commands(cached["N"]["actions"], cached["N"]["raw32"], 11, "N")
    rows = [prior.step(action, query=11, slot=slot, steps=55+slot)
            for slot, action in enumerate(cached["N"]["actions"][0, :5])]
    witness["N_trajectory_identity_recomputed"] = P.trajectory_identity(rows)
    retained = {"roles": [{"role": "source_qualification"}, {"role": "N", "status": "FAILED_NO_RETRY",
        "close_failure": {"code": "one_window_finish", "type": "PairingError"},
        "trajectory_identity": witness["N_trajectory_identity_recomputed"], "trajectory": rows}]}
    return retained, C.encoded(witness)


class Integration(unittest.TestCase):
    def test_native_child_all_components_to_outputs_with_fake_boundary_only(self):
        source, cached, witness = F.setup()
        retained, witness_raw = synthetic_reference(source, cached, witness)
        witness_ref = C.ident(witness_raw)
        created = []
        def factory(context, emit):
            instance = FakePhysical(context, emit)
            created.append(instance)
            return instance
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            out = root/"science"
            out.mkdir()
            (root/"INPUTS.json").write_text("{}")
            md = root/"retained_cpu3/metadata"
            (md/"roots").mkdir(parents=True)
            (md/"SOURCE_MANIFEST-v1.json").write_text('{}')
            (md/"roots"/(P.ROOT+".json")).write_text('{}')
            manifest = {"bytes": 1, "sha256": "s"*64}
            grant = authority(manifest, C.ident(root/"INPUTS.json"), witness_ref)
            deadline = time.time()+30
            ctx = {"package": str(root), "science": str(out), "stage": str(root/"stage"),
                "authority": grant, "source_manifest": manifest, "T0_epoch": deadline-480,
                "deadline_epoch": deadline, "archive_deadline_epoch": deadline+60,
                "restored": {}, "EGL_probe": {"status": "SOFTWARE_EGL_AVAILABLE"}}
            context = root/"CONTEXT.json"
            context.write_bytes(C.encoded(ctx))
            with patch.object(N.sys, "platform", "linux"), patch.object(N.os, "environ", {}), \
                 patch.object(C, "WITNESS", witness_ref), patch.object(R, "WITNESS", witness_ref), \
                 patch.object(C, "verify_sources", return_value={}), \
                 patch.object(N, "validate_image_sources", return_value={"test_fixture_only": True}), \
                 patch.object(N, "load_data", return_value=(source, cached, {"fixture": True})), \
                 patch.object(reference_loader, "load_reference", return_value=(retained, witness_raw, {"test_fixture_only": True})), \
                 patch.object(native_facade, "NativeReplay", side_effect=factory):
                result = N.run(context)
            self.assertTrue(result["completed_remaining_roles"])
            self.assertEqual([r.role for r in created], ["N_repeat", "S", "R"])
            self.assertEqual(result["new_environment_steps"], 210)
            self.assertEqual(result["cumulative_environment_steps"], 350)
            self.assertEqual(result["native_budget_cumulative"]["entered_calls"]["mj_step"], 8750)
            self.assertEqual(result["new_native_calls"]["mj_step"], 5250)
            contrasts = json.loads((out/"PHYSICAL-CONTRASTS.json").read_bytes())
            self.assertEqual(len(contrasts["comparisons"]), 10)
            self.assertEqual(contrasts["retained_N_original_status"], "FAILED_NO_RETRY")
            self.assertFalse(contrasts["old_run_status_modified"])
            self.assertEqual(contrasts["comparisons"][0]["eef_position_distance_m"], .1)
            self.assertFalse(any(n in __import__('sys').modules for n in ('mujoco','robosuite','libero')))

    def test_exact_original_data_rows_preserved_with_ten_new_reference_objects(self):
        inputs = json.loads((HERE/"INPUTS.json").read_bytes())
        original = json.loads((HERE.parents[1]/"implementation/cpu_action_replay_recovery_v3/INPUTS.json").read_bytes())
        current = {r["path"]: r for r in inputs["input_objects"]}
        self.assertEqual(len(current), 253)
        self.assertTrue(all(current[r["path"]] == r for r in original["input_objects"]))
        self.assertEqual(len(inputs["reference_paths"]), 10)
        self.assertEqual(sum(r["bytes"] for r in current.values()), C.INPUT_BYTES)

    def test_actual_facade_close_uses_original_recorder_without_new_sensor_reads(self):
        plan = {"root_trajectory_id": P.ROOT, "max_policy_steps": 520,
                "recorded_query_indices": list(range(13)), "tiers": {str(q): {} for q in range(13)}}
        rec = F.original_recorder()(plan, lambda *_: "fixture")
        rec.next = 12
        events, closed = [], []
        runtime = native_facade.NativeReplay({"role": "N_repeat"}, emit=lambda *x: events.append(x))
        runtime.env_step_calls = 70
        runtime.collector = NS(rec=rec, bridge=rec)
        runtime.storage = NS(active_query=None)
        runtime.parent = NS(close=lambda: closed.append(True))
        runtime.close()
        self.assertTrue(rec.closed)
        self.assertEqual(closed, [True])
        self.assertEqual(events[0][0], "FIXED_HORIZON_LEGACY_PARTIAL_WINDOW_CLOSE")
        runtime.close()
        self.assertEqual(closed, [True])

    def test_old_scope_or_zero_precharge_rejected_before_native_import(self):
        from native_budget import Budget
        grant = authority({}, {}, R.WITNESS)
        runtime = native_facade.NativeReplay({"role": "N_repeat", "authority": grant,
            "resources": C.RESOURCES, "native_budget": Budget(time.time()+10)}, emit=lambda *_: None)
        with self.assertRaisesRegex(ValueError, "prior_native_usage_precharged"):
            runtime._require_native_authority()
        runtime.context["authority"] = {**grant, "scope": "q11_cached_command_physical_pilot"}
        with self.assertRaisesRegex(ValueError, "new_direct_environment_authority"):
            runtime._require_native_authority()


if __name__ == "__main__":
    unittest.main()
