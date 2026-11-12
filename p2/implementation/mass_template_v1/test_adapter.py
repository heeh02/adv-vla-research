"""Synthetic arrays and existing small metadata only; no scientific NPZ reads."""
from pathlib import Path
import copy
import hashlib
import io
import json
import tempfile
import time
import unittest
from unittest import mock

import numpy as np

import adapter as A
import kernel as K


def fixture():
    rng = np.random.default_rng(19)
    corners = rng.normal(size=(4, 8, 10, 6)).astype(np.float64)
    # Orthogonal channel split makes projection closure exact by construction.
    def packed(values):
        gram = np.einsum('ahpd,bhpd->hpab', values, values)
        i, j = np.triu_indices(4)
        return gram[..., i, j]
    gram = packed(corners)
    parts = np.stack((packed(corners[..., :3]), packed(corners[..., 3:])))
    masses = rng.uniform(.01, .09, (2, 8, 10, 3)).astype(np.float64)
    one = {'raw_gram_upper': gram, 'masses': masses,
           'corner_defined': np.ones((4, 8, 10), bool),
           'group_corner_defined': np.ones((4, 8, 10, 3), bool),
           'projection_gram_upper': np.stack((parts, parts))}
    arrays = {k: np.broadcast_to(v, A.ABI[k][0]).copy() for k, v in one.items()}
    descriptions = {k: {'shape': list(v.shape), 'dtype': str(v.dtype), 'bytes': v.nbytes,
                         'payload_sha256': hashlib.sha256(v.tobytes()).hexdigest()} for k, v in arrays.items()}
    root = A.TASKS['CCONF-T06'][0]
    meta = {'task_id': 'CCONF-T06', 'root_id': root, 'query_index': 11,
            'source_identity': 'synthetic-only', 'arrays': descriptions}
    return arrays, meta, corners, masses


class AdapterTests(unittest.TestCase):
    def test_exact_stage_read_and_symlink_refusal(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            raw = b'owned-small-fixture'
            (base / 'x').write_bytes(raw)
            ref = A.identity(raw)
            self.assertEqual(A.read_exact(base, 'x', ref, cap=100), raw)
            (base / 'alias').symlink_to(base / 'x')
            with self.assertRaisesRegex(ValueError, 'no_stage_symlink'):
                A.read_exact(base, 'alias', ref, cap=100)
            with self.assertRaisesRegex(ValueError, 'canonical_relative_path'):
                A.read_exact(base, '../x', ref, cap=100)
            with self.assertRaisesRegex(ValueError, 'wholefile_SHA'):
                A.read_exact(base, 'x', {**ref, 'sha256': '0' * 64}, cap=100)

    def test_typed_field_mutation_missing_and_wrong_shape_fail_closed(self):
        arrays, meta, _, _ = fixture()
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            def write(values):
                buf = io.BytesIO()
                np.savez_compressed(buf, **values)
                raw = buf.getvalue()
                (base / 'QUERY-00.npz').write_bytes(raw)
                return {'path': 'QUERY-00.npz', **A.identity(raw)}
            ref = write(arrays)
            receipts = []
            with mock.patch.object(np, 'load', wraps=np.load) as loaded:
                actual = A.load_query(base, ref, meta, on_verified=receipts.append)
            self.assertEqual(loaded.call_count, 1)
            self.assertEqual(sum(v.nbytes for v in actual.values()), 668160)
            self.assertEqual(len(receipts), 1)
            self.assertTrue(all(not v.flags.writeable for v in actual.values()))
            changed = {**arrays, 'masses': arrays['masses'].copy()}
            changed['masses'][0, 0, 0, 0, 0, 0, 0] += .001
            with self.assertRaisesRegex(ValueError, 'typed_payload_SHA_masses'):
                A.load_query(base, write(changed), meta)
            incomplete = {k: v for k, v in arrays.items() if k != 'masses'}
            with self.assertRaisesRegex(ValueError, 'original_unique_NPZ_members'):
                A.load_query(base, write(incomplete), meta)
            wrong = {**arrays, 'masses': arrays['masses'][..., :2]}
            with self.assertRaisesRegex(ValueError, 'typed_array_shape_dtype_bytes_masses'):
                A.load_query(base, write(wrong), meta)

    def test_parent_metadata_chain_and_relabeling(self):
        inventory = json.loads((A.HERE / 'SOURCE-INPUT-INVENTORY.json').read_bytes())
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            for name, ref in inventory['metadata'].items():
                (base / name).write_bytes(Path(ref['local_source_path']).read_bytes())
            _, indexed = A.qualify_context(inventory, base)
            self.assertEqual(set(indexed), A.EXPECTED)
            forged = copy.deepcopy(inventory)
            forged['queries'][0]['source_identity'] = 'wrong-source'
            with self.assertRaisesRegex(ValueError, 'original_task_query_source_identity'):
                A.qualify_context(forged, base)
            forged = copy.deepcopy(inventory)
            forged['queries'][0]['query_index'] = 999
            with self.assertRaisesRegex(ValueError, 'all24_original_queries'):
                A.qualify_context(forged, base)
            forged = copy.deepcopy(inventory)
            forged['metadata']['PARENT-SOURCE.json']['sha256'] = '0' * 64
            with self.assertRaisesRegex(ValueError, 'historical_parent_identity_pinned'):
                A.qualify_context(forged, base)

    def test_pinned_new_source_manifest_rejects_changed_kernel(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            files = {'adapter.py': b'pass\n', 'kernel.py': b'pass\n', 'SOURCE-INPUT-INVENTORY.json': b'{}\n'}
            for name, raw in files.items():
                (base / name).write_bytes(raw)
            manifest = {'schema': 'PAPER2-mass-template-source-manifest-v1',
                        'files': [{'path': name, **A.identity(raw)} for name, raw in files.items()]}
            raw = A.encode(manifest)
            path = base / 'SOURCE-MANIFEST.json'
            path.write_bytes(raw)
            with mock.patch.object(A, 'HERE', base.resolve()):
                A.verify_source_manifest(path, A.identity(raw)['sha256'])
                (base / 'kernel.py').write_bytes(b'fail\n')
                with self.assertRaisesRegex(ValueError, 'wholefile_SHA'):
                    A.verify_source_manifest(path, A.identity(raw)['sha256'])

    def test_end_to_end_synthetic_18views_direct_vector_sse_and_24support(self):
        arrays, meta, corners, masses = fixture()
        result = A.analyze_query(arrays, meta, K)
        self.assertEqual(result['comparisons'], 18)
        self.assertEqual(len(result['rows']), 270)
        self.assertEqual(sum(r['comparison_role'] == 'primary' for r in result['rows']), 1)
        self.assertEqual(sum(r['comparison_role'] == 'support_window' for r in result['rows']), 2)
        self.assertTrue(all('__baseline_SSE_raw' not in key for key in result['arrays']))
        primary = next(r for r in result['rows'] if r['condition'] == 'C' and r['layer_index'] == 17
                       and r['flow_from'] == 0 and r['flow_to'] == 9 and r['window'] == 'full10' and r['view'] == 'full')
        ratio = masses[1].sum(-1) / masses[0].sum(-1)
        expected_sse = np.square(corners[3] - ratio[..., None] * corners[0]).sum()
        self.assertAlmostEqual(primary['baselines']['per_head_position_prefix_mass']['SSE'], expected_sse, places=10)
        all_rows = []
        for task, roots in A.TASKS.items():
            for root in roots:
                for query in (11, 16, 22):
                    all_rows.extend({**r, 'task_id': task, 'root_id': root, 'query_index': query} for r in result['rows'])
        full = A.aggregate_rows(all_rows, K)
        population = next(r for r in full['population'] if r['condition'] == 'C' and r['layer_index'] == 17
                          and r['flow_from'] == 0 and r['flow_to'] == 9 and r['window'] == 'full10' and r['view'] == 'full')
        self.assertEqual(population['planned_queries'], 24)
        self.assertEqual(population['observed_queries'], 24)
        self.assertAlmostEqual(population['baselines']['per_head_position_prefix_mass']['SSE'], expected_sse, places=10)
        missing_root = A.TASKS['CCONF-T06'][0]
        missing = A.aggregate_rows([r for r in all_rows if (r['root_id'], r['query_index']) != (missing_root, 11)], K)
        broken = next(r for r in missing['population'] if all(r[k] == population[k] for k in
                      ('condition', 'layer_index', 'flow_from', 'flow_to', 'view', 'window')))
        self.assertEqual(broken['planned_queries'], 24)
        self.assertEqual(broken['observed_queries'], 23)
        self.assertIsNone(broken['target_delta_energy'])
        self.assertIsNone(broken['baselines']['per_head_position_prefix_mass']['SSE'])
        with self.assertRaisesRegex(ValueError, 'unique_original_query_aggregate_row'):
            A.aggregate_rows(all_rows + all_rows[:1], K)
        raw_size = sum(a.nbytes for a in result['arrays'].values())
        self.assertLessEqual(raw_size * 24, A.OUTPUT_LIMITS['output_raw_array_bytes'])

    def test_absolute_deadline_output_exclusivity_and_failure_receipt(self):
        deadline = A.Deadline(time.time() + 60, 10)
        deadline.check()
        with mock.patch.object(time, 'time', return_value=deadline.epoch + 1):
            with self.assertRaisesRegex(ValueError, 'absolute_or_runtime'):
                deadline.check()
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            with mock.patch.object(sys_module(), 'platform', 'linux'):
                result = A.run(base / 'stage', base / 'failed', base / 'missing-manifest.json',
                               '0' * 64, deadline_epoch=time.time() + 60)
            self.assertFalse(result['analysis_complete'])
            self.assertTrue((base / 'failed/FAILURE.json').is_file())
            self.assertTrue((base / 'failed/MANIFEST.json').is_file())
            with self.assertRaises(FileExistsError):
                A.Output(base / 'failed', A.OUTPUT_LIMITS)

    def test_failure_reserve_is_spendable_only_by_receipts(self):
        with tempfile.TemporaryDirectory() as directory:
            out = A.Output(Path(directory) / 'new', {'output_bytes': 66000,
                           'single_output_bytes': 65536, 'output_raw_array_bytes': 65536})
            out.put('science.bin', b'x' * 464)
            with self.assertRaisesRegex(ValueError, 'total_output_budget'):
                out.put('too-big.bin', b'x')
            out.put('FAILURE.json', b'x' * 1000, finalizing=True)
            with self.assertRaisesRegex(ValueError, 'failure_reserve_only'):
                out.put('AGGREGATES.json.gz', b'x', finalizing=True)

    def test_cloud_runner_synthetic_staging_24queries_and_failure_finalization(self):
        arrays, meta, _, _ = fixture()
        buf = io.BytesIO()
        np.savez_compressed(buf, **arrays)
        payload = buf.getvalue()
        inventory = {'limits': A.OUTPUT_LIMITS.copy(), 'metadata': {}, 'queries': []}
        indexed = {}
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            stage = base / 'stage'
            stage.mkdir()
            for i, (root, query) in enumerate(sorted(A.EXPECTED)):
                task = next(t for t, rr in A.TASKS.items() if root in rr)
                qmeta = {**meta, 'root_id': root, 'task_id': task, 'query_index': query}
                indexed[root, query] = qmeta
                ref = {k: qmeta[k] for k in ('root_id', 'task_id', 'query_index', 'source_identity')}
                ref.update(path=f'QUERY-{i:02d}.npz', uri='synthetic:no-network', **A.identity(payload))
                inventory['queries'].append(ref)
                (stage / ref['path']).write_bytes(payload)
            loaded = {'SOURCE-INPUT-INVENTORY.json': A.encode(inventory),
                      'kernel.py': (A.HERE / 'kernel.py').read_bytes()}
            source_ref = {'bytes': 1, 'sha256': 'a' * 64}
            # Test-only dependency substitution. Production has no synthetic CLI
            # switch and cannot bypass the pinned parent metadata chain.
            with mock.patch.object(A.sys, 'platform', 'linux'), \
                 mock.patch.object(A, 'verify_source_manifest', return_value=({}, loaded, source_ref)), \
                 mock.patch.object(A, 'qualify_context', return_value=({}, indexed)):
                result = A.run(stage, base / 'complete', base / 'source.json', 'a' * 64,
                               deadline_epoch=time.time() + 60)
                self.assertTrue(result['analysis_complete'])
                self.assertEqual(result['completed_queries'], 24)
                self.assertEqual(result['completed_comparisons'], 432)
                self.assertEqual(len(result['verified_query_reads']), 24)
                self.assertTrue((base / 'complete/MANIFEST.json').is_file())
                self.assertEqual(result['derived_raw_array_bytes'], 39074832)
                with mock.patch.object(A, 'aggregate_rows', side_effect=ValueError('synthetic_aggregate_failure')):
                    failed = A.run(stage, base / 'aggregate-failed', base / 'source.json', 'a' * 64,
                                   deadline_epoch=time.time() + 60)
                self.assertFalse(failed['analysis_complete'])
                self.assertEqual(failed['completed_queries'], 24)
                self.assertIsNone(failed['aggregates'])
                self.assertTrue((base / 'aggregate-failed/FAILURE.json').is_file())
                self.assertTrue((base / 'aggregate-failed/MANIFEST.json').is_file())


def sys_module():
    return A.sys


if __name__ == '__main__':
    unittest.main()
