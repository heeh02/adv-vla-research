"""Only explicitly constructed synthetic vectors; no real caches are opened."""
import math
import json
from pathlib import Path
import unittest
from unittest.mock import patch
import adapter
import audit as A
import shape


def rms(v):
    return math.sqrt(math.fsum(x*x for x in v)/len(v))


def from_vectors(s, p):
    return shape.project(rms(s), rms(p), rms([b-a for a, b in zip(s, p)]), len(s))


def fixture():
    rows = []
    for si, snap in enumerate(A.SNAPSHOTS):
        for w in A.WINDOWS:
            n = 5 if w == 'first5' else 10
            for di, dim in enumerate(A.DIMENSIONS):
                s = [(.1+si+di)*(i-2) for i in range(n)]
                p = [2.*v for v in s] if (si+di)%2 else list(reversed(s))
                for c, vec in zip(A.CONTRASTS, (s, p, [b-a for a, b in zip(s, p)])):
                    rows.append({**dict(zip(A.IDENTITY, snap)), 'contrast': c,
                                 'policy_env_step': snap[3]*5, 'absolute_env_step': snap[3]*5+10,
                                 'graph': 'Gstar', 'window': w, 'command_dimension': dim,
                                 'positions': n, 'RMS': rms(vec),
                                 'mean_signed': math.fsum(vec)/n,
                                 'exactzero_slots': sum(v == 0 for v in vec),
                                 'unit': 'command_adapter_units', 'proposed_not_issued': True})
    return {'WINDOWS': rows}


class ShapeTests(unittest.TestCase):
    def test_positive_scaling(self):
        r = from_vectors([1, -2, 3, 0, 1], [3, -6, 9, 0, 3])
        self.assertAlmostEqual(r['signed_gain'], 3.)
        self.assertAlmostEqual(r['signed_residual_fraction'], 0.)
        self.assertEqual(r['signed_fraction_lower'], 0.)

    def test_negative_scaling_not_shape_change(self):
        r = from_vectors([1, -2, 3, 0, 1], [-2, 4, -6, 0, -2])
        self.assertAlmostEqual(r['signed_gain'], -2.)
        self.assertAlmostEqual(r['signed_residual_fraction'], 0.)
        self.assertEqual(r['nonnegative_residual_fraction'], 1.)

    def test_orthogonal(self):
        r = from_vectors([1, 0, 0, 0, 0], [0, 1, 0, 0, 0])
        self.assertAlmostEqual(r['cosine'], 0.)
        self.assertAlmostEqual(r['signed_residual_fraction'], 1.)
        self.assertTrue(r['gain_sign_uncertain'])

    def test_equal_RMS_can_hide_deformation(self):
        r = from_vectors([1, -1, 0, 0, 0], [0, 1, -1, 0, 0])
        self.assertAlmostEqual(r['cosine'], -.5)
        self.assertAlmostEqual(r['signed_residual_fraction'], .75)
        self.assertLessEqual(r['signed_fraction_lower'], .75)
        self.assertGreaterEqual(r['signed_fraction_upper'], .75)

    def test_true_zero_cases(self):
        self.assertEqual(shape.project(0., 0., 0., 5)['status'], 'both_zero')
        r = shape.project(0., 2., 2., 5)
        self.assertEqual(r['status'], 'pulse_zero')
        self.assertIsNone(r['signed_gain'])
        self.assertEqual(r['signed_residual_fraction'], 1.)
        r = shape.project(2., 0., 2., 5)
        self.assertEqual(r['signed_gain'], 0.)
        self.assertIsNone(r['signed_residual_fraction'])

    def test_invalid_geometry_is_not_silent_clipping(self):
        r = shape.project(1., 1., 4., 5)
        self.assertEqual(r['status'], 'invalid_moment_geometry')
        self.assertIsNone(r['cosine'])

    def test_common_units_scaling_invariance(self):
        base = shape.project(1., 2., 2., 5)
        for scale in (1e-120, 1e120):
            r = shape.project(scale, 2*scale, 2*scale, 5)
            self.assertEqual(r['cosine'], base['cosine'])
            self.assertEqual(r['signed_residual_fraction'], base['signed_residual_fraction'])

    def test_nonzero_below_guard_is_not_zero(self):
        r = shape.project(1e-200, 1e200, 1e200, 5)
        self.assertEqual(r['status'], 'nonzero_below_guard_scale')
        self.assertIsNone(r['signed_gain'])
        self.assertTrue(r['rounding_sensitive'])

    def test_condition_overflow_keeps_finite_JSON(self):
        r = shape.project(1e-160, 1., 1., 5)
        self.assertEqual(r['status'], 'nonzero_below_guard_scale')
        self.assertTrue(r['condition_overflow'])
        self.assertIsNone(r['tolerance_over_A'])
        json.dumps(r, allow_nan=False)

    def test_all336_complete_and_summary14(self):
        rows = shape.analyze(fixture())
        self.assertEqual(len(rows), 336)
        self.assertEqual(len(shape.summarize(rows)), 14)
        self.assertTrue(all(r['status'] == 'defined' for r in rows))

    def test_duplicate_and_missing_source_rows_rejected(self):
        data = fixture()
        data['WINDOWS'][-1] = data['WINDOWS'][0]
        with self.assertRaisesRegex(ValueError, 'no_duplicate'):
            shape.analyze(data)
        with self.assertRaisesRegex(ValueError, '1008_rows'):
            shape.analyze({'WINDOWS': []})

    def test_zero_summary_inconsistency_rejected(self):
        data = fixture()
        data['WINDOWS'][0]['exactzero_slots'] = 5
        with self.assertRaisesRegex(ValueError, 'RMS_zero_matches'):
            shape.analyze(data)

    def test_Mac_gate_precedes_any_real_file_read(self):
        with patch.object(adapter.sys, 'platform', 'darwin'):
            with self.assertRaisesRegex(ValueError, 'only_on_cloud'):
                adapter.load(Path('/path/does/not/exist'), '0'*64, lambda: None)


if __name__ == '__main__':
    unittest.main()
