import copy
import gzip
import json
import math
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch
import numpy as np
import adapter as D
import analysis as A
import binding as B
import codec as C
import synthetic_fixture as F


class AuditTests(unittest.TestCase):
    def test_zero_denominators_preserved(self):
        q = A.vector_metrics(np.zeros(4), np.zeros(4))
        self.assertIsNone(q["cosine"])
        self.assertIsNone(q["relative_vector_error_to_reference"])
        self.assertIsNone(q["same_sign_fraction_of_both_nonzero"])
        self.assertEqual(q["both_zero_coordinates"], 4)

    def test_sign_support_denominator(self):
        q = A.vector_metrics([0, 1, -1, 2, 3], [1, 0, 1, 4, -1])
        self.assertEqual(q["both_nonzero_coordinates"], 3)
        self.assertEqual(q["same_sign_both_nonzero"], 1)
        self.assertEqual(q["opposite_sign_both_nonzero"], 2)
        self.assertEqual(q["reference_zero_candidate_nonzero"], 1)
        self.assertEqual(q["reference_nonzero_candidate_zero"], 1)

    def test_scalar_gain_is_descriptive(self):
        q = A.vector_metrics([1, 2, 3], [-2, -4, -6])
        self.assertEqual(q["descriptive_best_scalar_candidate_from_reference"], -2)
        self.assertEqual(q["descriptive_best_scalar_residual_norm"], 0)
        self.assertAlmostEqual(q["cosine"], -1)
        self.assertTrue(q["projection_is_target_visible_not_prediction"])

    def test_different_memory_order_same_typed_payload(self):
        rng = np.random.default_rng(730)
        a = rng.normal(size=(37, 29)).astype(np.float64)
        b = np.array(a, order="F")
        self.assertNotEqual(a.flags.c_contiguous, b.flags.c_contiguous)
        self.assertEqual(C.from_array(a).identity(), C.from_array(b).identity())
        q = A.scalar_check(float(np.sum(a*a)), float(np.sum(b*b)))
        self.assertTrue(q["scalar_close"])
        # A last-bit scalar reduction difference never defines typed equality.
        self.assertTrue(A.scalar_check(1., np.nextafter(1., 2.))["scalar_close"])
        self.assertFalse(A.compare_typed(C.from_array(np.array([1.])),
                                        C.from_array(np.array([np.nextafter(1., 2.)])))["typed_exact"])

    def test_distinct_hash_domains(self):
        saved = "synthetic_saved"
        cr, tr = {"path": "C", "bytes": 1, "sha256": "C"}, {"path": "T", "bytes": 2, "sha256": "T"}
        historical = B.query_source_identity(saved, cr, tr)
        self.assertNotEqual(saved, historical)
        self.assertNotEqual(historical, B.query_source_identity(saved, tr, cr))

    def test_call_event_swap_rejected(self):
        entry = {"arm": "S"}
        inp = {"path": "input", "bytes": 4, "sha256": "i"}
        call = {"path": "call", "bytes": 8, "sha256": "c"}
        m = {"schema": "rss-pulse-compact-call-v1", "entry": entry, "input_ref": inp,
             "native_validation_errors": []}
        event = {"schema": "rss-pulse-compact-event-v1", "event": "CALL_CHECKS",
                 "payload": {"entry": {"arm": "R"}, "passed": True, "output": call}}
        with self.assertRaisesRegex(ValueError, "later_passed_checks"):
            B.validate_call(m, event, entry, inp, call)

    def test_full_denominator_no_prefix_renormalization(self):
        p = np.ones((8, 10, 16)) / 16
        e = np.ones((6, 256))
        w = np.zeros((8, 256, 1024))
        w[:, 0, 0] = 1
        raw, gated = A.proxy(p, e, w, np.ones(1024) * .5)
        self.assertEqual(raw[0, 0], 3)
        self.assertEqual(gated[0, 0], 1.5)

    def test_exact_signed_energy_requires_cross(self):
        A.verify_energy_closure(1, 4, 1, -4)
        with self.assertRaisesRegex(ValueError, "exact_proxy_native_signed_energy_closure"):
            A.verify_energy_closure(1, 4, 1, 0)

    def test_dyadic_proxy_value_exact(self):
        self.assertEqual(A.dyadic_float_units(.5), 1 << 1073)
        self.assertEqual(A.dyadic_float_units(np.nextafter(0., 1.)), 1)
        self.assertEqual(A.dyadic_float_units(-.5), -(1 << 1073))

    def test_FP64_gate_product_is_not_exact_dyadic_product(self):
        proxy, gate = np.nextafter(1., 2.), 1.5
        exact = A.dyadic_float_units(proxy) * A.Q.units(F.bf([gate]).storage[0])
        rounded = A.dyadic_float_units(float(proxy * gate)) << 133
        self.assertNotEqual(exact, rounded)

    def test_exact_decomposition_and_zero(self):
        zeros = np.zeros((10, 1024), np.uint16)
        gate = F.bf(np.full(1024, .5)).storage
        raw = F.bf(np.ones((10, 1024))).storage
        h = F.bf(np.full((10, 1024), .25)).storage
        rows = A.exact_decomposition(zeros, raw, zeros, h, gate, np.full((10, 1024), .75))
        full = rows[-1]
        self.assertTrue(full["exact_energy_closure"])
        self.assertEqual(full["terms"]["total"]["squared_norm"], 160)
        self.assertEqual(full["terms"]["projection"]["squared_norm"], 160)
        self.assertEqual(full["terms"]["gate_residual"]["squared_norm"], 640)
        self.assertEqual(full["signed_cross_term"], -640)
        z = A.exact_decomposition(zeros, zeros, zeros, zeros, gate, np.zeros((10, 1024)))[-1]
        self.assertEqual(z["terms"]["total"]["norm"], 0)

    def test_same_state_mismatch_preserved(self):
        native, edits, wo, _, _, _ = F.example()
        obs = F.observability(A.KEYS[0], native, edits)
        native["S"][("reads__attention_gate",)] = F.bf(np.full((5, 3, 1, 1, 1024), .25))
        answer = A.snapshot(A.KEYS[0], native,
            {"S": edits[("actual_E", "shared")].numeric()[0, :, 0],
             "R": edits[("actual_E", "random_matched")].numeric()[0, :, 0]}, wo.numeric(), obs)
        self.assertEqual(len(answer[1]), 117)
        self.assertFalse(next(x for x in answer[0] if x["arm"] == "S" and x["field"] == "gate")["typed_exact"])
        self.assertFalse(next(x for x in answer[1] if x["contrast"] == "S-N")["same_state_interpretation_eligible"])
        self.assertTrue(next(x for x in answer[1] if x["contrast"] == "R-N")["same_state_interpretation_eligible"])

    def test_local_real_array_gate(self):
        with patch.object(C.sys, "platform", "darwin"):
            with self.assertRaisesRegex(ValueError, "real_arrays_cloud_Linux_only"):
                C.typed(Path("not_opened.npz"), {})

    def test_complete_synthetic_stage_and_output_caps(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "stage"
            inputs = F.stage_fixture(path)
            result = D.execute(path, inputs, lambda: None, synthetic=True)
            self.assertEqual(result["QA.json"]["counts"], D.COUNTS)
            self.assertTrue(result["QA.json"]["pass"])
            self.assertEqual(result["SUMMARY.json"]["state_mismatch_count"], 0)
            for name, doc in result.items():
                raw = D.encoded(doc)
                if name.endswith(".gz"):
                    raw = gzip.compress(raw, mtime=0)
                self.assertLessEqual(len(raw), 2 * 1024**2)
            bad = copy.deepcopy(inputs)
            bad["snapshots"][0]["S_entry"], bad["snapshots"][0]["R_entry"] = (
                bad["snapshots"][0]["R_entry"], bad["snapshots"][0]["S_entry"])
            with self.assertRaisesRegex(ValueError, "exact_native_role_not_filename"):
                D.execute(path, bad, lambda: None, synthetic=True)

    def test_failed_receipt_preserved(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "SOURCE-MANIFEST.json").write_text("{}")
            with patch.object(D, "HERE", root), patch.object(D.sys, "platform", "linux"):
                result = D.run(root / "stage", root / "out", "wrong", time.time() + 60)
            self.assertEqual(result["status"], "FAILED_PRESERVED")
            self.assertFalse(result["analysis_complete"])
            self.assertEqual(json.loads((root / "out" / D.RESULT).read_text())["failure"]["code"],
                             "externally_bound_source_SHA")


if __name__ == "__main__":
    unittest.main()
