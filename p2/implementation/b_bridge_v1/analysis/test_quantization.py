"""Synthetic-only exact rounding and captured-leaf readout tests."""
import unittest
import numpy as np
import quantization as Q


class Tests(unittest.TestCase):
    def test_every_finite_bf16_roundtrips_and_nonfinite_rejected(self):
        for bits in range(65536):
            if ((bits >> 7) & 255) == 255:
                with self.assertRaises(Q.QuantizationError):
                    Q.units(bits)
            else:
                self.assertEqual(Q.round_product_units_to_bf16(Q.units(bits) << 133, zero_sign=bits >> 15), bits)

    def test_midpoints_round_to_even_both_signs_and_subnormal(self):
        for low in (0, 1, 2, 126, 127, 128, 129, 0x3f7f, 0x3f80, 0x3f81, 0x7f7e):
            upper = low+1
            midpoint = (Q.units(low)+Q.units(upper)) << 132
            expected = low if low % 2 == 0 else upper
            self.assertEqual(Q.round_product_units_to_bf16(midpoint), expected)
            self.assertEqual(Q.round_product_units_to_bf16(-midpoint), expected | 0x8000)
        # Positive and negative half of the minimum subnormal become signed0.
        self.assertEqual(Q.round_product_units_to_bf16(1 << 132), 0)
        self.assertEqual(Q.round_product_units_to_bf16(-(1 << 132)), 0x8000)

    def test_overflow_rounding_threshold_not_saturation(self):
        maximum = Q.units(0x7f7f)
        threshold = (maximum+(1 << 252)) << 133
        self.assertEqual(Q.round_product_units_to_bf16(threshold-1), 0x7f7f)
        self.assertEqual(Q.round_product_units_to_bf16(threshold), 0x7f80)
        self.assertEqual(Q.round_product_units_to_bf16(-threshold), 0xff80)

    def test_fused_split_have_known_cancellation_counterexample(self):
        # (129/128)^2 - 130/128 = 2^-14 exactly; intermediate BF16 rounding
        # loses the last 2^-14 before adding the negative residual.
        p = Q.rounding_candidates(0x3f81, 0x3f81, 0xbf82)
        self.assertEqual(p["fused"], 0x3880)
        self.assertEqual(p["split"], 0)
        self.assertFalse(p["split_product_overflow"])

    def test_zero_sign_rules_and_model_ambiguity(self):
        self.assertEqual(Q.rounding_candidates(0x8000, 0x3f80, 0x8000)["fused"], 0x8000)
        self.assertEqual(Q.rounding_candidates(0x8000, 0x3f80, 0)["fused"], 0)
        a = lambda value: np.full((10,1024), value, np.uint16)
        record = {"W_used": a(0x3f80), "gate": a(0x3f80), "residual_input": a(0), "after_attention_residual": a(0x3f80)}
        out = Q.analyze_rounding_candidates({"none": record})
        self.assertEqual(out["records"][0]["status"], "BOTH_MATCH_PATH_NOT_IDENTIFIED")
        self.assertFalse(out["compiled_path_uniquely_identified"])

    def test_full_support_model_mismatches_and_signed_zero_are_kept(self):
        a = lambda value: np.full((10,1024), value, np.uint16)
        record = {"W_used": a(0x3f81), "gate": a(0x3f81), "residual_input": a(0xbf82), "after_attention_residual": a(0x3880)}
        out = Q.analyze_rounding_candidates({"old_persistent": record})
        row = out["records"][0]
        self.assertEqual(row["counts"]["coordinates"], 10240)
        self.assertEqual(row["counts"]["fused_bitwise_match"], 10240)
        self.assertEqual(row["counts"]["split_bitwise_match"], 0)
        self.assertEqual(row["status"], "ONLY_FUSED_CANDIDATE_MATCHES")
        record["after_attention_residual"][:] = 0x8000
        out = Q.analyze_rounding_candidates({"none": record})
        self.assertEqual(out["records"][0]["counts"]["split_zero_sign_only_mismatch"], 10240)

    def test_combined_error_identity_not_product_attribution(self):
        a = lambda value: np.full((10,1024), value, np.uint16)
        records = {}
        for role, w in (("none",0x3f80),("natural_d",0x3f81),("common_c",0x3f80)):
            y = Q.rounding_candidates(w,0x3f81,0xbf82)["fused"]
            records[role] = {"W_before":a(0x3f80),"W_used":a(w),"gate":a(0x3f81),
                "residual_input":a(0xbf82),"after_attention_residual":a(y)}
        out = Q.analyze_native_triplet(records)
        self.assertEqual(len(out["rows"]),39)
        self.assertEqual(out["attribution"],"combined_only_product_not_captured")
        self.assertTrue(all(r["exact_closure"] for r in out["rows"]))
        self.assertTrue(out["ideal_arithmetic_is_not_GPU_emulation"])
        records["common_c"]["gate"][0,0] = 0
        with self.assertRaisesRegex(Q.QuantizationError,"same_B_pre_site_gate"):
            Q.analyze_native_triplet(records)


if __name__ == "__main__":
    unittest.main()
