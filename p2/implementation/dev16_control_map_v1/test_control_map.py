"""Meaningful boundary and provenance tests with synthetic commands only."""
import copy
import gzip
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import adapter as D
import analysis as A
import binding as B
import codec as C
import synthetic_fixture as F


class MappingTests(unittest.TestCase):
    def test_signedzero_raw_payload_and_numeric_are_distinct(self):
        row = A.scalar_pair(-0.0, +0.0, 0)
        self.assertTrue(row["raw_numeric_equal"])
        self.assertFalse(row["raw_bitwise_equal"])
        self.assertTrue(row["clipped_numeric_equal"])
        self.assertFalse(row["clipped_bitwise_equal"])
        self.assertTrue(row["mapped_bitwise_equal"])  # source final +0 canonicalizes here
        self.assertFalse(row["raw_difference_erased_by_mapping"])

    def test_gripper_signedzero_is_same_canonical_symbol(self):
        row = A.scalar_pair(-0.0, +0.0, 6)
        self.assertTrue(row["both_zero"])
        self.assertEqual(row["reference_gripper_sign"], 0)
        self.assertFalse(row["raw_bitwise_equal"])
        self.assertTrue(row["mapped_numeric_equal"])

    def test_clip_erases_different_saturated_values(self):
        row = A.scalar_pair(1.1, 1.5, 1)
        self.assertTrue(row["erased_at_clip"])
        self.assertFalse(row["erased_only_at_scale_rounding"])
        self.assertEqual(row["reference_mapped"], .05)

    def test_negative_saturation_preserved(self):
        row = A.scalar_pair(-2., -3., 5)
        self.assertLess(row["raw_delta"], 0)
        self.assertTrue(row["erased_at_clip"])
        self.assertEqual(row["reference_mapped"], -.5)

    def test_exact_boundary_is_not_marked_clipped(self):
        row = A.scalar_pair(1., np.nextafter(1., 2.), 0)
        self.assertTrue(row["reference_at_input_boundary"])
        self.assertEqual(row["reference_clip_state"], "inside_including_boundary")
        self.assertEqual(row["candidate_clip_state"], "above_+1")
        self.assertTrue(row["erased_at_clip"])

    def test_tiny_scale_underflow_separate_from_clip(self):
        row = A.scalar_pair(0., np.nextafter(0., 1.), 0)
        self.assertFalse(row["raw_numeric_equal"])
        self.assertFalse(row["clipped_numeric_equal"])
        self.assertTrue(row["mapped_numeric_equal"])
        self.assertFalse(row["erased_at_clip"])
        self.assertTrue(row["erased_only_at_scale_rounding"])

    def test_gripper_magnitude_is_erased_but_not_motion_claim(self):
        row = A.scalar_pair(.001, 100., 6)
        self.assertTrue(row["same_nonzero_sign"])
        self.assertTrue(row["raw_magnitude_difference_erased_by_sign"])
        self.assertEqual(row["mapping"], "gripper_sign_symbol_not_stateful_target")
        self.assertNotIn("current_action", row)

    def test_gripper_cross_zero_and_explicit_zero_are_separate(self):
        cross = A.scalar_pair(-1e-12, 1e-12, 6)
        self.assertTrue(cross["opposite_nonzero_signs"])
        self.assertEqual(cross["mapped_delta"], 2)
        touch = A.scalar_pair(0., 1e-12, 6)
        self.assertTrue(touch["one_zero_other_nonzero"])
        self.assertFalse(touch["opposite_nonzero_signs"])

    def test_negative_difference_sign_retained(self):
        row = A.scalar_pair(.5, .25, 4)
        self.assertEqual(row["raw_delta"], -.25)
        self.assertEqual(row["mapped_delta"], -.125)

    def test_symmetric_opposite_arm_values_not_erased(self):
        row = A.scalar_pair(-2., 2., 0)
        self.assertFalse(row["mapped_numeric_equal"])
        self.assertEqual(row["mapped_delta"], .1)

    def test_nonfinite_rejected(self):
        for bad in (float("nan"), float("inf"), -float("inf")):
            with self.assertRaisesRegex(ValueError, "finite_command_scalar"):
                A.scalar_pair(0., bad, 6)

    def test_adapter_dtype_shape_are_not_silently_changed(self):
        for value in (np.zeros((10, 7), np.float32), np.zeros((5, 7), np.float64)):
            with self.assertRaisesRegex(ValueError, "finite_native_adapter7_ABI"):
                A.mapping(value)

    def test_rms_avoids_square_underflow(self):
        result = A.rms([1e-300, 1e-300])
        self.assertGreater(result, 0)
        self.assertAlmostEqual(result / 1e-300, 1., places=14)

    def test_same_gripper_different_arm_not_joint_equivalence(self):
        n = np.zeros((10, 7), dtype=np.float64)
        n[:, 6] = .01
        s, r = n.copy(), n.copy()
        s[:, 0] = .1
        s[:, 6] = .5
        points, windows, cases = A.snapshot(B.KEYS[0], {"N": n, "S": s, "R": r},
                                           semantic_claim_eligible=True)
        row = next(c for c in cases if c["contrast"] == "S-N" and c["window"] == "first5")
        self.assertTrue(row["gripper_all_signs_equal"])
        self.assertFalse(row["arm6_all_mapped_numeric_equal"])
        self.assertFalse(row["all7_frontend_numeric_equivalent"])
        self.assertFalse(row["physical_equivalence_measured"])
        self.assertEqual((len(points), len(windows), len(cases)), (210, 42, 6))

    def test_zero_cases_and_all_windows_retained(self):
        z = np.zeros((10, 7), np.float64)
        points, windows, cases = A.snapshot(B.KEYS[0], {"N": z, "S": z, "R": z},
                                           semantic_claim_eligible=False)
        self.assertEqual(len(points), 210)
        self.assertTrue(all(r["raw_numeric_equal"] for r in points))
        self.assertTrue(all(r["raw_RMS_difference"] == 0 for r in windows))
        self.assertTrue(all(c["all7_frontend_numeric_equivalent"] for c in cases))
        self.assertFalse(any(r["static_controller_binding_eligible"] for r in points))


class IdentityTests(unittest.TestCase):
    def test_complete_fabricated54object_pipeline_and_caps(self):
        with tempfile.TemporaryDirectory() as td:
            stage = Path(td) / "stage"
            inputs, proof = F.stage_fixture(stage)
            docs = D.execute(stage, inputs, lambda: None, synthetic=True, root_proof=proof)
            self.assertEqual(docs["QA.json"]["counts"], D.COUNTS)
            self.assertTrue(docs["QA.json"]["pass"])
            self.assertEqual(len(docs["SUMMARY.json"]["groups"]), 42)
            self.assertEqual(len(docs["SUMMARY.json"]["all_case_window_equivalence"]), 48)
            self.assertFalse(docs["QA.json"]["specific_current_action_or_actuator_target_computed"])
            size = 0
            for name, doc in docs.items():
                raw = D.encoded(doc)
                if name.endswith(".gz"):
                    raw = gzip.compress(raw, mtime=0)
                self.assertLessEqual(len(raw), 2 * 1024**2)
                size += len(raw)
            self.assertLess(size, 8 * 1024**2)

    def test_static_proof_duplicate_root_rejected(self):
        inputs = json.loads((D.HERE / "INPUTS.json").read_text())
        proof = json.loads((D.HERE / "ALL-ROOT-CONTROL-BINDING.json").read_text())
        proof["roots"].append(proof["roots"][0])
        with self.assertRaisesRegex(ValueError, "all4_static_root_bindings"):
            D.root_semantics(proof, inputs["snapshots"][0], {r["path"]: r for r in inputs["input_objects"]})

    def test_static_source_domains_must_each_match(self):
        inputs = json.loads((D.HERE / "INPUTS.json").read_text())
        for domain in ("saved_input_source_identity", "historical_query_source_identity", "input_identity"):
            proof = json.loads((D.HERE / "ALL-ROOT-CONTROL-BINDING.json").read_text())
            proof["snapshots"][0][domain] = "wrong-domain"
            with self.assertRaisesRegex(ValueError, "static_root_.*identity"):
                D.root_semantics(proof, inputs["snapshots"][0],
                                 {r["path"]: r for r in inputs["input_objects"]})

    def test_check_event_swap_rejected(self):
        entry = {"arm": "S"}
        inp = {"path": "input", "bytes": 4, "sha256": "i"}
        call = {"path": "call", "bytes": 8, "sha256": "c"}
        meta = {"schema": "rss-pulse-compact-call-v1", "entry": entry, "input_ref": inp,
                "native_validation_errors": []}
        event = {"schema": "rss-pulse-compact-event-v1", "event": "CALL_CHECKS",
                 "payload": {"entry": {"arm": "R"}, "passed": True, "output": call}}
        with self.assertRaisesRegex(ValueError, "later_passed_checks"):
            B.validate_call(meta, event, entry, inp, call)

    def test_compact_role_swap_rejected_after_bound_inputs(self):
        with tempfile.TemporaryDirectory() as td:
            stage = Path(td) / "stage"
            inputs, proof = F.stage_fixture(stage)
            inputs["snapshots"][0]["S_entry"], inputs["snapshots"][0]["R_entry"] = (
                inputs["snapshots"][0]["R_entry"], inputs["snapshots"][0]["S_entry"])
            with self.assertRaisesRegex(ValueError, "exact_native_role_not_filename"):
                D.execute(stage, inputs, lambda: None, synthetic=True, root_proof=proof)

    def test_corrupted_input_rejected_before_decode(self):
        with tempfile.TemporaryDirectory() as td:
            stage = Path(td) / "stage"
            inputs, proof = F.stage_fixture(stage)
            path = stage / inputs["snapshots"][0]["new"]["N"]
            raw = path.read_bytes()
            path.write_bytes(raw[:-1] + bytes([raw[-1] ^ 1]))
            with self.assertRaisesRegex(ValueError, "input_whole_SHA"):
                D.execute(stage, inputs, lambda: None, synthetic=True, root_proof=proof)

    def test_actual_array_local_gate(self):
        with patch.object(C.sys, "platform", "darwin"):
            with self.assertRaisesRegex(ValueError, "real_arrays_cloud_Linux_only"):
                C.typed(Path("never_opened.npz"), {})

    def test_deadline_callback_checked_before_input_access(self):
        with tempfile.TemporaryDirectory() as td:
            stage = Path(td) / "stage"
            inputs, proof = F.stage_fixture(stage)
            def expired():
                raise ValueError("absolute_science_deadline")
            with self.assertRaisesRegex(ValueError, "absolute_science_deadline"):
                D.execute(stage, inputs, expired, synthetic=True, root_proof=proof)


if __name__ == "__main__":
    unittest.main()
