"""Entry-gate, source-binding and physical-readout tests with synthetic values."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch
import numpy as np
import pilot_contract as C
import native_child
import science_adapter
from physical_contrasts import summarize
from physical_writer import write_new
import test_campaign_core as fake_campaign


class EntrypointTests(unittest.TestCase):
    def test_both_native_entrypoints_reject_before_runtime_or_output(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root/"INPUTS.json").write_text("{}")
            context = {"package": str(root), "science": str(root/"out"),
                       "stage": str(root/"stage"), "runtime_work": str(root/"runtime"),
                       "authority": {"direct_human_approved": False},
                       "source_manifest": {}, "T0_epoch": 1, "deadline_epoch": 1501}
            path = root/"context.json"
            path.write_text(json.dumps(context))
            for function in (native_child.run, science_adapter.run):
                with self.subTest(function=function.__module__):
                    with self.assertRaisesRegex(ValueError, "new_direct_environment_authority"):
                        function(path)
            self.assertFalse((root/"out").exists())
            self.assertFalse((root/"runtime").exists())

    def test_authority_source_or_time_change_rejected(self):
        manifest, inputs = {"bytes": 1, "sha256": "a"*64}, {"bytes": 2, "sha256": "b"*64}
        authority = {"direct_human_approved": True, "scope": C.SCOPE, "run_id": C.RUN,
            "root_id": C.ROOT, "query_index": 11, "maximum_explicit_env_steps": 350,
            "new_model_calls": 0, "GPU": 0, "resources": C.RESOURCES,
            "source_manifest": manifest, "inputs_identity": inputs,
            "science_seconds": 1500, "archive_seconds": 1680, "allocation_seconds": 1800,
            "automatic_retries": 0, "direct_user_evidence": {"bytes": 1, "sha256": "c"*64}}
        C.authority_gate(authority, manifest, inputs, t0=100, deadline=1600)
        for change in ({"GPU": 1}, {"automatic_retries": 1}, {"source_manifest": inputs}):
            with self.assertRaisesRegex(ValueError, "authority_binding"):
                C.authority_gate({**authority, **change}, manifest, inputs, t0=100, deadline=1600)
        with self.assertRaisesRegex(ValueError, "creation_anchored"):
            C.authority_gate(authority, manifest, inputs, t0=100, deadline=1601)

    def test_source_member_corruption_detected_before_import(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root/"helper.py").write_bytes(b"raise AssertionError('never execute')\n")
            doc = {"schema": "paper2-q11-physical-source-v1",
                   "files": [{"path": "helper.py", **C.ident(root/"helper.py")}]}
            (root/"SOURCE-MANIFEST.json").write_bytes(C.encoded(doc))
            ref = C.ident(root/"SOURCE-MANIFEST.json")
            C.verify_sources(root, ref)
            (root/"helper.py").write_bytes(b"changed\n")
            with self.assertRaisesRegex(ValueError, "source_member"):
                C.verify_sources(root, ref)

    def test_stage_whole_SHA_and_extra_file_rejection(self):
        with tempfile.TemporaryDirectory() as td:
            stage = Path(td)
            (stage/"a.json").write_bytes(b"abc")
            inputs = {"schema": "paper2-q11-cached-physical-inputs-v1",
                      "input_objects": [{"path": "a.json", **C.ident(stage/"a.json")}]}
            with patch.object(C, "INPUT_COUNT", 1), patch.object(C, "INPUT_BYTES", 3):
                science_adapter.stage_gate(stage, inputs, lambda: None)
                (stage/"a.json").write_bytes(b"abd")
                with self.assertRaisesRegex(ValueError, "input_SHA"):
                    science_adapter.stage_gate(stage, inputs, lambda: None)
                (stage/"extra").write_bytes(b"x")
                with self.assertRaisesRegex(ValueError, "exact_physical_stage"):
                    science_adapter.stage_gate(stage, inputs, lambda: None)

    def test_model_guard_records_rejected_attempt(self):
        guard = native_child.NoModelImports()
        for name in ("jax", "openpi.models.pi0", "openpi.policies.policy", "flax"):
            with self.assertRaisesRegex(RuntimeError, "model_import_forbidden"):
                guard.find_spec(name)
        self.assertEqual(guard.attempts, 4)
        self.assertIsNone(guard.find_spec("robosuite"))

    def test_complete_fake_campaign_serializes_without_real_arrays_or_runtime(self):
        result, _, _ = fake_campaign.Campaign().run_fake()
        with tempfile.TemporaryDirectory() as td:
            ref = write_new(Path(td)/"synthetic-campaign.json.gz", result, compressed=True)
            self.assertLess(ref["bytes"], 1024**2)
            self.assertTrue(result["synthetic"])

    def test_log_limit_interrupts_child_wait_before_further_progress(self):
        child = NS(poll=lambda: None, wait=lambda **_: self.fail("must stop before wait"))
        with self.assertRaisesRegex(ValueError, "native_child_log_2MiB"):
            science_adapter.bounded_wait(child, [2*1024**2+1], lambda: None)


def physical(position=(0., 0., 0.), quaternion=(0., 0., 0., 1.), finger=(0., 0.)):
    return {"physical_outputs": {
        "returned_nonimage_observations": {"robot0_eef_pos": np.array(position),
            "robot0_eef_quat": np.array(quaternion), "robot0_gripper_qpos": np.array(finger)},
        "gripper_actuator_ctrl": np.array([1., -1.]), "contact_count": 0},
        "done": False, "official_success": False}


class ContrastTests(unittest.TestCase):
    def test_positions_rotations_and_fingers_kept_separate(self):
        campaign = {"completed_all_roles": True, "roles": [
            {"role": "N", "trajectory": [physical()]},
            {"role": "S", "trajectory": [physical((.003, .004, 0.), (0., 0., 1., 0.), (.001, 0.))]},
            {"role": "R", "trajectory": [physical(quaternion=(0., 0., 0., -1.))]}]}
        result = summarize(campaign)
        s, r = result["comparisons"]
        self.assertAlmostEqual(s["eef_position_distance_m"], .005)
        self.assertAlmostEqual(s["eef_orientation_geodesic_rad"], np.pi)
        self.assertTrue(s["gripper_actuator_target_equal"])
        self.assertEqual(s["gripper_qpos_delta"][0], .001)
        self.assertEqual(r["eef_orientation_geodesic_rad"], 0.)
        self.assertFalse(result["success_rate_or_task_transfer"])

    def test_failed_qualification_yields_no_scientific_contrasts(self):
        result = summarize({"completed_all_roles": False})
        self.assertEqual(result["comparisons"], [])

    def test_only_observed_common_prefix_and_no_padding(self):
        campaign = {"completed_all_roles": True, "roles": [
            {"role": "N", "trajectory": [physical(), physical()]},
            {"role": "S", "trajectory": [physical()]},
            {"role": "R", "trajectory": [physical(), physical()]}]}
        result = summarize(campaign)
        self.assertEqual(len(result["comparisons"]), 3)
        self.assertTrue(result["no_unobserved_tail_filled"])


if __name__ == "__main__":
    unittest.main()
