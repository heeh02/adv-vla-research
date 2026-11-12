"""Synthetic exact-integer checks. No scientific arrays or model imports."""
import json
import unittest
import core as K


class ExactTests(unittest.TestCase):
    def test_all_finite_patterns_roundtrip_including_signed_zero(self):
        count = 0
        for bits in range(65536):
            if ((bits >> 7) & 255) == 255:
                with self.assertRaises(K.PrecisionError):
                    K.bits_to_units(bits)
                continue
            value, sign = K.bits_to_units(bits)
            self.assertEqual(K.units_to_bits(value, zero_sign=sign)[0], bits)
            count += 1
        self.assertEqual(count, 65280)

    def test_exact_subnormal_normal_boundary_and_sign(self):
        self.assertEqual(K.bits_to_units(1), (1, 0))
        self.assertEqual(K.bits_to_units(0x8001), (-1, 1))
        self.assertEqual(K.bits_to_units(0x007f)[0], 127)
        self.assertEqual(K.units_to_bits(128)[0], 0x0080)
        self.assertEqual(K.bits_to_units(0x3f80)[0], 1 << 133)

    def test_exponent_scale_gap_cannot_be_hidden_by_float_rounding(self):
        one = K.bits_to_units(0x3f80)[0]
        self.assertIsNone(K.units_to_bits(one + 1)[0])
        out = K.analyze_triplet([[0x3f80]], [[0]], [[1]], expected_shape=(1, 1))
        self.assertEqual(out["status"], "NO_GO_NATURAL_SAME_DELTA_UNREPRESENTABLE")
        self.assertEqual(out["counts"]["A_target_nonrepresentable"], 1)

    def test_same_requested_delta_can_be_unrepresentable_at_other_baseline(self):
        # B: 1 -> 1 + 2^-7, A: 2. Exact A+delta is between BF16 values.
        out = K.analyze_triplet([[0x4000, 0x3f80]], [[0x3f80, 0x3f80]],
                                [[0x3f81, 0x3f81]], expected_shape=(1, 2))
        self.assertEqual(out["counts"]["reference_delta_nonzero"], 2)
        self.assertEqual(out["counts"]["A_target_representable"], 1)
        self.assertEqual(out["per_position"][0]["nonrepresentable_channels"], [0])

    def test_signed_zero_transitions_are_not_a_nonzero_probe(self):
        out = K.analyze_triplet([[0x8000, 0]], [[0x8000, 0]], [[0, 0x8000]], expected_shape=(1, 2))
        self.assertEqual(out["status"], "NO_GO_ZERO_REFERENCE_DELTA")
        self.assertEqual(out["counts"]["B_zero_sign_transitions"], 2)
        self.assertEqual(out["counts"]["reference_delta_zero"], 2)
        self.assertEqual(out["counts"]["A_negative_zero"], 1)

    def test_nonzero_delta_can_reach_exact_zero_without_choosing_sign(self):
        out = K.analyze_triplet([[0x3f80]], [[0x3f80]], [[0]], expected_shape=(1, 1))
        self.assertEqual(out["status"], "PASS_EXACT_REPRESENTABILITY_ONLY")
        self.assertEqual(out["counts"]["A_target_zero_from_nonzero_delta"], 1)
        self.assertFalse(out["new_probe_generated"])

    def test_overflow_not_saturated_and_nonfinite_rejected(self):
        out = K.analyze_triplet([[0x7f7f]], [[0]], [[0x7f7f]], expected_shape=(1, 1))
        self.assertEqual(out["counts"]["A_target_overflow"], 1)
        for pattern in (0x7f80, 0xff80, 0x7fc1):
            with self.assertRaisesRegex(K.PrecisionError, "nonfinite"):
                K.analyze_triplet([[pattern]], [[0]], [[0]], expected_shape=(1, 1))

    def test_fixed_10240_denominator_locations_and_bounded_result(self):
        a = [[0x4000] * 1024 for _ in range(10)]
        b = [[0x3f80] * 1024 for _ in range(10)]
        e = [[0x3f81] * 1024 for _ in range(10)]
        out = K.analyze_triplet(a, b, e)
        self.assertEqual(out["counts"]["A_target_nonrepresentable"], 10240)
        self.assertTrue(all(r["coordinates"] == r["A_target_nonrepresentable"] == 1024 for r in out["per_position"]))
        self.assertLess(len(json.dumps(out).encode()), 128 * 1024)
        self.assertEqual(a[0][0], 0x4000)


if __name__ == "__main__":
    unittest.main()
