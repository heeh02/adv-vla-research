import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
import hashlib
import numpy as np
from inventory import inventory, vector_identity
from extract import extract
import archive_reader as A


class InventoryTests(unittest.TestCase):
    def setUp(self):
        self.ref = {'path': 'official/libero/libero/init_files/libero_10/test.pruned_init', 'bytes': 1, 'sha256': '0' * 64}

    def test_duplicate_rows_retained_and_not_declared_unseen(self):
        r, _ = inventory([np.array([1., 2.]), np.array([1., 2.]), np.array([3., 4.])], self.ref, [])
        self.assertEqual((r['row_count'], r['typed_unique_rows']), (3, 2))
        self.assertEqual(r['rows'][1]['first_typed_equal_index'], 0)
        self.assertTrue(all(x['prospective_unseen_eligibility'] == 'NOT_ESTABLISHED' for x in r['rows']))

    def test_signed_zero_is_not_two_numeric_states(self):
        r, _ = inventory([np.array([1., 0.]), np.array([1., -0.])], self.ref, [])
        self.assertEqual((r['typed_unique_rows'], r['exact_numeric_unique_rows']), (2, 1))

    def test_nonfinite_rejected(self):
        with self.assertRaises(ValueError):
            inventory([np.array([1., np.nan])], self.ref, [])

    def test_wider_than_float64_rejected(self):
        if np.dtype(np.longdouble).itemsize <= 8:
            self.skipTest('Host has no wider native floating dtype; explicit width guard reviewed.')
        with self.assertRaisesRegex(ValueError, 'lossless_float64'):
            inventory([np.array([1., 2.], dtype=np.longdouble)], self.ref, [])

    def test_wrong_reference_identity_rejected(self):
        root = {'root_id': 'known', 'selection': {'source_library': {**self.ref, 'path': self.ref['path'][9:]},
                'native_initial_vector': {'sha256': 'f' * 64}, 'original_library_index': 0,
                'original_init0_proof': {'cached_states_native_typed_sha256': '0' * 64}}}
        with self.assertRaises(ValueError):
            inventory([np.array([1., 2.])], self.ref, [root])

    def test_exact_reference_match_and_whole_library(self):
        values = [np.array([1., 2.]), np.array([3., 4.])]
        inv, _ = inventory(values, self.ref, [])
        root = {'root_id': 'known', 'selection': {'source_library': {**self.ref, 'path': self.ref['path'][9:]},
                'native_initial_vector': vector_identity(values[1]), 'original_library_index': 1,
                'original_init0_proof': {'cached_states_native_typed_sha256': inv['cached_states_native_typed_sha256']}}}
        result, matches = inventory(values, self.ref, [root])
        self.assertEqual(result['rows'][1]['matches_reviewed_DEV_roots'], ['known'])
        self.assertEqual(len(matches), 1)


def archive_case(folder):
    raw = io.BytesIO()
    bodies = {'official/libero/libero/init_files/libero_10/test.pruned_init': b'fake library', 'other.txt': b'unchanged'}
    rows = []
    with tarfile.open(fileobj=raw, mode='w') as t:
        for name, body in bodies.items():
            info = tarfile.TarInfo(name); info.size = len(body); info.mode = 0o644
            t.addfile(info, io.BytesIO(body)); rows.append({'path': name, **A.fid(body), 'mode': info.mode})
    archive = raw.getvalue(); manifest = b''.join(A.encoded(r) for r in rows)
    (folder / 'part.bin').write_bytes(archive); (folder / 'files.jsonl').write_bytes(manifest)
    refs = [{'path': n, 'uri': 'redacted-8a33e961f58f' + n, **A.fid(r)} for n, r in [('part.bin', archive), ('files.jsonl', manifest)]]
    inputs = {'input_objects': refs, 'input_object_count': 2, 'input_body_bytes': sum(r['bytes'] for r in refs),
              'archives': [{'name': 'base', 'archive': A.fid(archive), 'manifest_paths': ['files.jsonl'],
                            'members': 2, 'logical_file_bytes': sum(len(b) for b in bodies.values()),
                            'parts': [{'stage_path': 'part.bin', 'offset': 0, **A.fid(archive)}]}]}
    return inputs, [rows[0]], bodies


class ArchiveTests(unittest.TestCase):
    def test_only_pinned_library_extracted_all_members_verified(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d); ins, libs, bodies = archive_case(p)
            blobs, proofs = extract(p, ins, libs, lambda: None)
            self.assertEqual(blobs, {libs[0]['path']: bodies[libs[0]['path']]})
            self.assertEqual(proofs[0]['members'], 2)

    def test_unselected_member_corruption_is_not_ignored(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d); ins, libs, _ = archive_case(p)
            raw = (p / 'part.bin').read_bytes().replace(b'unchanged', b'CORRUPTED')
            (p / 'part.bin').write_bytes(raw)
            with self.assertRaises(A.ExtractionError):
                extract(p, ins, libs, lambda: None)

    def test_missing_library_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d); ins, libs, _ = archive_case(p); libs[0]['path'] += '.missing'
            with self.assertRaises(A.ExtractionError):
                extract(p, ins, libs, lambda: None)

    def test_deadline_propagates(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d); ins, libs, _ = archive_case(p)
            def stop():
                raise TimeoutError('fixed deadline')
            with self.assertRaises(TimeoutError):
                extract(p, ins, libs, stop)


if __name__ == '__main__':
    unittest.main()
