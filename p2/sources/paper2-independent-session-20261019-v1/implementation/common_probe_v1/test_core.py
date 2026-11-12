"""Independent exhaustive finite-BF16 oracle; synthetic values only."""
import bisect
import json
import math
import random
import unittest

import core as C
from exact_bf16 import bits_to_units, units_to_bits


def enumeration():
    # Independent direct format enumeration, never calls the projector or its bins.
    values = set()
    for code in range(65536):
        sign = -1 if code & 32768 else 1
        exponent, fraction = (code // 128) % 256, code % 128
        if exponent == 255:
            continue
        magnitude = fraction if exponent == 0 else (128+fraction)*2**(exponent-1)
        values.add(sign*magnitude)
    return tuple(sorted(values))


VALUES = enumeration()
VALUE_SET = set(VALUES)


def oracle(a, b, d):
    """Enumerate every eligible B target; exact A membership decides commonness."""
    if d >= 0:
        lo, hi = bisect.bisect_left(VALUES, b), bisect.bisect_right(VALUES, b+d)
        targets = reversed(VALUES[lo:hi])
    else:
        lo, hi = bisect.bisect_left(VALUES, b+d), bisect.bisect_right(VALUES, b)
        targets = VALUES[lo:hi]
    for target in targets:
        delta = target-b
        if a+delta in VALUE_SET:
            return delta
    raise AssertionError("zero must be feasible")


def bits(integer, negative_zero=False):
    value = units_to_bits(integer, zero_sign=int(negative_zero))[0]
    assert value is not None
    return value


class Tests(unittest.TestCase):
    def test_all_finite_BF16_values_match_507_segment_grid_union(self):
        self.assertEqual(len(VALUES), 65279)
        self.assertEqual(len(C.FINITE_BINS), 507)
        union = set()
        previous = None
        for low, high, stride in C.FINITE_BINS:
            if previous is not None:
                self.assertLess(previous, low)
            self.assertEqual(low % stride, 0)
            self.assertEqual(high % stride, 0)
            union.update(range(low, high+1, stride))
            previous = high
        self.assertEqual(union, VALUE_SET)

    def test_independent_oracle_random_entire_exponent_range(self):
        rng = random.Random(20261019)
        cases = [(rng.choice(VALUES), rng.choice(VALUES), rng.choice(VALUES)) for _ in range(160)]
        # More near-bin cases so failures are not swamped by unchanged coordinates.
        for _ in range(160):
            k = rng.randrange(254)
            scale = 1 << k
            a = rng.choice((128, 129, 254, 255))*scale
            b = rng.randrange(-255, 256)
            after = rng.randrange(-255, 256)
            cases.append((a if rng.randrange(2) else -a, b, after))
        changed = rescued = 0
        for a, b, after in cases:
            d = after-b
            got = C.project_common_units(a, b, d)
            self.assertEqual(got, oracle(a, b, d), (a, b, d))
            self.assertIn(a+got, VALUE_SET); self.assertIn(b+got, VALUE_SET)
            self.assertLessEqual(abs(got), abs(d)); self.assertGreaterEqual(got*d, 0)
            changed += got != d
            rescued += got != 0 and got != d
        self.assertGreater(changed, 50)
        self.assertGreater(rescued, 5)

    def test_every_exponent_boundary_both_signs_against_oracle(self):
        for k in range(254):
            stride = 1 << k
            for sign in (-1, 1):
                a = sign*(128*stride)
                b = sign*127
                after = sign*131
                d = after-b
                self.assertEqual(C.project_common_units(a, b, d), oracle(a, b, d))

    def test_joint_congruence_not_rounding_A_target_or_one_ULP(self):
        # A wants even delta after256; B=255 wants odd delta after its transition.
        self.assertIn(256+2, VALUE_SET)
        self.assertNotIn(255+2, VALUE_SET)
        self.assertEqual(C.project_common_units(256, 255, 3), 0)
        self.assertEqual(C.project_common_units(256, 0, 3), 2)
        self.assertEqual(C.project_common_units(256, 0, 1), 0)

    def test_zero_drop_is_not_generally_optimal_and_can_be_optimal(self):
        for d, expected in ((3, 2), (1, 0)):
            c = C.project_common_units(256, 0, d)
            self.assertEqual(c, expected)
            self.assertLessEqual((c-d)**2, d*d)
        self.assertLess((2-3)**2, 3**2)

    def test_preserves_all_naturally_common_and_zero_signs(self):
        for a, b, after in ((-256, 256, 258), (0, 0, 255), (256, 0, 4), (-128, 127, 127)):
            d = after-b
            if a+d in VALUE_SET:
                self.assertEqual(C.project_common_units(a, b, d), d)
        self.assertEqual(C.diagnostic_target_bits(0x8000, 0), 0x8000)
        self.assertEqual(C.diagnostic_target_bits(0, 0), 0)
        self.assertEqual(C.diagnostic_target_bits(bits(-1), 1), 0)

    def test_forbidden_nonfinite_invalid_reference_and_input_types(self):
        with self.assertRaisesRegex(ValueError, "baselines"):
            C.project_common_units(257, 0, 1)
        with self.assertRaisesRegex(ValueError, "reference"):
            C.project_common_units(0, 256, 1)
        with self.assertRaisesRegex(ValueError, "integer"):
            C.project_common_units(0., 0, 1)
        with self.assertRaises(ValueError):
            C.analyze_triplet([[0x7f80]], [[0]], [[0]], expected_shape=(1, 1))

    def test_full_signed_geometry_and_zero_drop_closure(self):
        a = [[bits(256), bits(0), bits(256), bits(0)]]
        b = [[bits(0), bits(0), bits(0), bits(0)]]
        after = [[bits(3), bits(-3), bits(1), bits(0)]]
        r = C.analyze_triplet(a, b, after, expected_shape=(1, 4))
        f = r["full"]
        exact = {k: int(v) for k, v in f["energy_exact_integer_units_squared"].items()}
        self.assertEqual(exact["reference_energy"], 19)
        self.assertEqual(exact["candidate_energy"], 13)
        self.assertEqual(exact["error_energy"], 2)
        self.assertEqual(exact["candidate_dot_reference"], 15)
        self.assertEqual(exact["initial_failed_reference_energy"], 10)
        self.assertEqual(exact["zero_drop_error_energy"], 10)
        self.assertEqual(f["counts"]["strictly_shrunk_nonzero"], 1)
        self.assertEqual(f["counts"]["zeroed_reference_nonzero"], 1)
        self.assertEqual(f["counts"]["sign_reversals"], 0)
        self.assertEqual(f["counts"]["magnitude_increases"], 0)
        self.assertAlmostEqual(f["relative_error_energy"], 2/19)
        self.assertGreater(f["angle_to_reference_degrees"], 0)
        self.assertFalse(r["native_target_arrays_returned"])
        json.dumps(r, allow_nan=False)

    def test_cancellation_can_be_removed_or_reversed_without_sign_flip(self):
        # d=(3,-3) sums to zero, c=(2,-3) sums to -1 despite no coordinate sign flip.
        r = C.analyze_triplet([[bits(256), bits(0)]], [[0, 0]], [[bits(3), bits(-3)]], expected_shape=(1, 2))
        sums = r["full"]["signed_all_coordinate_sum"]
        self.assertEqual(sums["reference_sum_exact_units"], ["0"])
        self.assertEqual(sums["candidate_sum_exact_units"], ["-1"])
        # d=(3,-2) positive sum; force first coordinate to zero with A=1024 stride8.
        r = C.analyze_triplet([[bits(1024), bits(0)]], [[0, 0]], [[bits(3), bits(-2)]], expected_shape=(1, 2))
        sums = r["full"]["signed_all_coordinate_sum"]
        self.assertEqual(sums["reference_sum_exact_units"], ["1"])
        self.assertEqual(sums["candidate_sum_exact_units"], ["-2"])
        self.assertEqual(r["full"]["counts"]["sign_reversals"], 0)

    def test_cross_position_signed_Gram_and_full_concat_energy_differ(self):
        r = C.analyze_triplet([[bits(256)], [bits(0)]], [[0], [0]], [[bits(3)], [bits(-3)]], expected_shape=(2, 1))
        cross = r["cross_position"]
        self.assertEqual(cross["reference_Gram_exact_units_squared"], [["9", "-9"], ["-9", "9"]])
        self.assertEqual(cross["candidate_Gram_exact_units_squared"], [["4", "-6"], ["-6", "9"]])
        sums = cross["summed_position_vector"]["exact_sum_energy_units_squared"]
        self.assertEqual(sums["reference"], "0")
        self.assertEqual(sums["candidate"], "1")
        self.assertEqual(int(r["full"]["energy_exact_integer_units_squared"]["reference_energy"]), 18)

    def test_zero_vectors_and_extreme_finite_values_report_without_float_overflow(self):
        r = C.analyze_triplet([[0]], [[0]], [[0]], expected_shape=(1, 1))
        self.assertIsNone(r["full"]["relative_error_energy"])
        self.assertIsNone(r["full"]["cosine_to_reference"])
        largest, smaller = 0x7f7f, 0x7f7e
        r = C.analyze_triplet([[largest, 0]], [[largest, 0]], [[smaller, largest]], expected_shape=(1, 2))
        self.assertTrue(math.isfinite(r["full"]["energy"]["reference_energy"]))
        self.assertTrue(math.isfinite(r["full"]["orthogonal_error_energy"]))
        json.dumps(r, allow_nan=False)

    def test_full_shape_bounded_JSON_and_no_input_mutation(self):
        a = [[bits(256)]*1024 for _ in range(10)]
        b = [[0]*1024 for _ in range(10)]
        after = [[bits(3 if ch % 2 else -3) for ch in range(1024)] for _ in range(10)]
        before = json.dumps([a, b, after])
        r = C.analyze_triplet(a, b, after)
        self.assertEqual(json.dumps([a, b, after]), before)
        self.assertEqual(r["full"]["counts"]["coordinates"], 10240)
        self.assertEqual(sum(x["counts"]["coordinates"] for x in r["per_position"]), 10240)
        self.assertLess(len(json.dumps(r, separators=(",", ":"), allow_nan=False).encode()), 220*1024)


if __name__ == "__main__":
    unittest.main()
