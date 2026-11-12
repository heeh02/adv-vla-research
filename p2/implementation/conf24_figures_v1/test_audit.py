"""Synthetic-only semantic regression checks, including all invalid inputs."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import audit as A
import adapter
from synthetic_fixture import fixture, roundtrip


class CacheAuditTests(unittest.TestCase):
    def setUp(self):
        self.tables, self.producer = fixture()

    def verify(self):
        return A.verify(self.tables, self.producer, synthetic=True)

    def test_complete_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            decoded = roundtrip(self.tables, Path(d)/'synthetic')
            result = A.verify(decoded, self.producer, synthetic=True)
        self.assertEqual(result['qa']['rows'], {'FLOW-SUMMARY': 1512,
            'MECHANISM-SUMMARY': 2880, 'READING_EXACT': 1080, 'WINDOWS': 1008, 'COVERAGE': 24})
        self.assertEqual(len(result['command_cells']), 336)
        self.assertEqual(len(result['separation_counts']), 15)

    def test_zero_and_opposite_are_separate(self):
        data = self.verify()
        first = [r for r in data['command_cells'] if r['window'] == 'first5']
        self.assertEqual(sum(not r['ratio_defined'] for r in first), 2)
        self.assertTrue(any(r['RMS_ratio'] == 0 for r in first))
        self.assertTrue(any(r['pulse_mean_exactzero'] and not r['pulse_window_exactzero'] for r in first))
        self.assertTrue(any(r['signed_window_means_opposite'] for r in first))
        self.assertTrue(any(r['RMS_order'] == 'equal' for r in first))
        self.assertTrue(all(not r['signed_window_means_opposite'] for r in first
                            if r['pulse_mean_exactzero'] or r['persistent_mean_exactzero']))

    def test_malformed_support_rejected(self):
        for kind in self.tables:
            with self.subTest(kind=kind):
                broken = copy.deepcopy(self.tables)
                broken[kind].pop()
                with self.assertRaises(ValueError):
                    A.verify(broken, self.producer, synthetic=True)
                broken = copy.deepcopy(self.tables)
                broken[kind].append(copy.deepcopy(broken[kind][0]))
                with self.assertRaisesRegex(ValueError, 'no_duplicate_rows'):
                    A.verify(broken, self.producer, synthetic=True)

    def test_wrong_identity_or_clock(self):
        for key, val in [('root_id', 'wrong'), ('absolute_env_step', 55), ('graph', 'R')]:
            with self.subTest(key=key):
                broken = copy.deepcopy(self.tables)
                broken['WINDOWS'][0][key] = val
                with self.assertRaises(ValueError):
                    A.verify(broken, self.producer, synthetic=True)

    def test_scalar_corruption(self):
        for kind, key, value in [
            ('MECHANISM-SUMMARY', 'RMS_difference', 1.),
            ('FLOW-SUMMARY', 'coordinates_in_window', 10),
            ('WINDOWS', 'positions', 4),
            ('WINDOWS', 'proposed_not_issued', False),
            ('READING_EXACT', A.EXACT[0], False),
        ]:
            with self.subTest(kind=kind, key=key):
                broken = copy.deepcopy(self.tables)
                row = next((r for r in broken[kind] if r.get('contrast') == 'persistent_minus_pulse'),
                           broken[kind][0])
                row[key] = value
                with self.assertRaises(ValueError):
                    A.verify(broken, self.producer, synthetic=True)

    def test_real_mode_rejects_synthetic_receipt(self):
        with self.assertRaisesRegex(ValueError, 'completed_fixed_producer'):
            A.verify(self.tables, self.producer, synthetic=False)

    def test_darwin_gate_before_file_read(self):
        with patch.object(adapter.sys, 'platform', 'darwin'), patch.object(
                Path, 'open', side_effect=AssertionError('must not read')):
            with self.assertRaisesRegex(ValueError, 'cloud_Linux_only'):
                adapter.run('/nonexistent', '/nonexistent', '', 0)


if __name__ == '__main__':
    unittest.main()
