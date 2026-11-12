"""Synthetic producer artifacts only; no actual NPZ or result readback."""
from pathlib import Path
import copy
import gzip
import hashlib
import importlib.util
import json
import sys
import tempfile
import time
import unittest
from unittest import mock

import review as R

OLD = Path(__file__).resolve().parent.parent / 'mass_template_v1'


def old_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fixture():
    # Reuse the frozen implementation's synthetic generator, not its real inputs.
    saved_path = sys.path.copy()
    sys.path.insert(0, str(OLD))
    try:
        import adapter as A
        import kernel as K
        import test_adapter as T
        arrays, meta, _, _ = T.fixture()
        one = A.analyze_query(arrays, meta, K)
        descriptors = {k: {'shape': list(a.shape), 'dtype': str(a.dtype), 'bytes': a.nbytes,
                           'payload_sha256': hashlib.sha256(a.tobytes()).hexdigest()} for k, a in one['arrays'].items()}
        details, allrows, inputs, completed, files = {}, [], [], [], {}
        for i, (root, query) in enumerate(sorted(R.EXPECTED_QUERIES)):
            task = next(t for t, rr in R.TASKS.items() if root in rr)
            identity = {'root_id': root, 'task_id': task, 'query_index': query, 'source_identity': 'synthetic-only-' + str(i)}
            ref = {**identity, 'path': f'QUERY-{i:02d}.npz', 'bytes': 1, 'sha256': '1' * 64, 'uri': 'synthetic:never-network'}
            inputs.append(ref)
            rows = [{**r, **identity} for r in one['rows']]
            checks = [{**c, 'tags': {**c['tags'], **identity}} for c in one['checks']]
            allrows.extend(rows)
            dname = f'DERIVED-{i:02d}.npz'
            draw = b'synthetic opaque payload; reviewer must not NPZ-decode this'
            dref = {'path': dname, **R.ident(draw)}
            files[dname] = draw
            name = f'QUERY-{i:02d}.json.gz'
            detail = {'input': ref, 'source_manifest': R.PRODUCER_MANIFEST, 'derived': dref,
                      'rows': rows, 'checks': checks, 'array_descriptors': descriptors,
                      'posthoc_reuse': True, 'signed_vector_reconstructed': False, 'native_W_inference': False}
            details[name] = detail
            raw = gzip.compress(R.encoded(detail), mtime=0)
            files[name] = raw
            completed.append({'root_id': root, 'query_index': query, 'comparisons': 18,
                              'detail': {'path': name, **R.ident(raw)}, 'derived': dref})
        aggregate = A.aggregate_rows(allrows, K)
    finally:
        sys.path[:] = saved_path
    inventory = {'queries': inputs}
    ar = gzip.compress(R.encoded(aggregate), mtime=0)
    files['AGGREGATES.json.gz'] = ar
    summary = {'schema': 'PAPER2-saved-Gram-run-v1', 'analysis_complete': True, 'synthetic': False,
               'source_manifest': R.PRODUCER_MANIFEST, 'input_inventory': R.INVENTORY, 'failures': [], 'missing_queries': [],
               'parent_run_id': R.PARENT_RUN, 'planned_queries': 24, 'completed_queries': 24,
               'planned_comparisons': 432, 'completed_comparisons': 432, 'new_model_calls': 0,
               'new_env_steps': 0, 'cloud_calls': 0, 'input_deletions': 0, 'new_fit': False,
               'posthoc_CONF_reuse': True, 'completed': completed, 'verified_query_reads': inputs,
               'verified_local_input_body_bytes': 26236769, 'planned_input_body_bytes': 26236769,
               'failed_unverified_read_bytes': 0, 'derived_raw_array_bytes': 39074832,
               'aggregates': {'path': 'AGGREGATES.json.gz', **R.ident(ar)}}
    files['SUMMARY.json'] = R.encoded(summary)
    outputs = {name: {'path': name, **R.ident(raw)} for name, raw in files.items()}
    manifest = {'schema': 'PAPER2-saved-Gram-output-manifest-v1', 'analysis_complete': True,
                'source_manifest': R.PRODUCER_MANIFEST, 'files': list(outputs.values())}
    files['MANIFEST.json'] = R.encoded(manifest)
    parent = {'analysis_complete': True, 'synthetic': False, 'qualified_queries': 24, 'qualified_roots': 8,
              'qualified_comparisons': 432, 'missing_queries': 0, 'numerical_failures': 0, 'main_aggregates': {}}
    mapping = {'modality_mass': 'mass', 'within_modality': 'within', 'signed_scalar_oracle': 'oracle', 'zero_change': 'nochange'}
    for level in ('root', 'task', 'population'):
        old = []
        for row in aggregate[level]:
            if (row['condition'], row['layer_index'], row['flow_from'], row['flow_to'], row['view']) != ('C', 17, 0, 9, 'full'):
                continue
            value = {k: row[k] for k in ('aggregation_id', 'window', 'planned_queries', 'observed_queries')}
            value['target_delta_energy_mean'] = row['target_delta_energy']
            for new, previous in mapping.items():
                value[previous + '_error_energy_mean'] = row['baselines'][new]['SSE']
                value[previous + '_error_ratio_of_mean_energy'] = row['baselines'][new]['relative_SSE']
            old.append(value)
        parent['main_aggregates'][level] = old
    refs = {'producer_source_manifest': R.PRODUCER_MANIFEST, 'input_inventory': R.INVENTORY,
            'parent_summary': R.PARENT_SUMMARY, 'outputs': outputs}
    return summary, details, aggregate, inventory, parent, refs, files


class ReviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = fixture()

    def test_all_support_and_plot_schema_synthetic_export(self):
        summary, details, aggregate, inventory, parent, refs, _ = self.data
        result = R.validate_and_extract(summary, details, aggregate, inventory, parent, refs)
        self.assertEqual(result['support']['comparisons'], 432)
        self.assertEqual(result['support']['query_view_window_rows'], 6480)
        f = result['figure_data']
        self.assertEqual([len(f[k]) for k in ('root_rows', 'task_rows', 'population_rows', 'query_rows')], [120, 60, 15, 24])
        self.assertEqual(result['checks']['old_baseline_consistency']['matched_aggregate_panels'], 39)
        self.assertLess(len(R.encoded(result)), R.MAX_EXPORT)

    def test_duplicate_missing_and_changed_case_fail_closed(self):
        summary, details, aggregate, inventory, parent, refs, _ = self.data
        broken = details.copy()
        broken.pop('QUERY-23.json.gz')
        with self.assertRaisesRegex(R.Rejected, 'all24_query_details'):
            R.validate_and_extract(summary, broken, aggregate, inventory, parent, refs)
        broken = copy.deepcopy(details)
        broken['QUERY-00.json.gz']['rows'][0]['query_index'] = 999
        with self.assertRaisesRegex(R.Rejected, 'row_identity_query_index'):
            R.validate_and_extract(summary, broken, aggregate, inventory, parent, refs)
        broken = copy.deepcopy(details)
        broken['QUERY-00.json.gz']['rows'].append(broken['QUERY-00.json.gz']['rows'][0])
        with self.assertRaisesRegex(R.Rejected, 'duplicate_view_window_row'):
            R.validate_and_extract(summary, broken, aggregate, inventory, parent, refs)

    def test_full_denominator_role_and_old_baseline_tampering(self):
        summary, details, aggregate, inventory, parent, refs, _ = self.data
        broken = copy.deepcopy(details)
        broken['QUERY-00.json.gz']['rows'][0]['full_target_delta_energy'] += 1
        with self.assertRaisesRegex(R.Rejected, 'scalar_mismatch_full_target_ratio'):
            R.validate_and_extract(summary, broken, aggregate, inventory, parent, refs)
        broken = copy.deepcopy(details)
        broken['QUERY-00.json.gz']['rows'][0]['comparison_role'] = 'primary'
        with self.assertRaisesRegex(R.Rejected, 'fixed_row_tags_and_role'):
            R.validate_and_extract(summary, broken, aggregate, inventory, parent, refs)
        old = copy.deepcopy(parent)
        old['main_aggregates']['root'][0]['mass_error_energy_mean'] += 1
        with self.assertRaisesRegex(R.Rejected, 'scalar_mismatch_old_mass_SSE'):
            R.validate_and_extract(summary, details, aggregate, inventory, old, refs)

    def test_marker_precedence_even_malformed_or_stale_complete(self):
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(R.sys, 'platform', 'linux'):
            base = Path(directory)
            (base / 'SUMMARY.json').write_text('{"analysis_complete":true}')
            (base / 'FINALIZATION-FAILURE.json').write_bytes(b'bad-json')
            result = R.review_result(base, 'unused', 'unused', 'unused', 'unused', 'a' * 64, 'b' * 64,
                                     deadline_epoch=time.time() + 60)
            self.assertFalse(result['accepted'])
            self.assertIsNone(result['figure_data'])
            self.assertEqual(result['failures'][0]['stage'], 'producer_failure_precedence')
            self.assertIn('unreadable_marker', result['failures'][0])

    def test_cloud_file_validation_without_NPZ_decode_and_mutation(self):
        summary, details, aggregate, inventory, parent, refs, files = self.data
        docs = {'input_inventory': inventory, 'parent_summary': parent}
        source_refs = {k: v for k, v in refs.items() if k != 'outputs'}
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            for name, raw in files.items():
                (base / name).write_bytes(raw)
            with mock.patch.object(R.sys, 'platform', 'linux'), \
                 mock.patch.object(R, 'validate_sources', return_value=(source_refs, docs)):
                result = R.review_result(base, 'fixture', 'fixture', 'fixture', 'fixture', 'a' * 64,
                                         R.ident(files['MANIFEST.json'])['sha256'], deadline_epoch=time.time() + 60)
                self.assertTrue(result['accepted'], result['failures'])
                self.assertEqual(result['checks']['NPZ_objects_hashed'], 24)
                self.assertEqual(result['checks']['NPZ_decodes'], 0)
                receipt = R.write_export(base / 'review-export', result)
                self.assertLessEqual(receipt['file']['bytes'], R.MAX_EXPORT)
                # Use a separate result folder: unknown result-side output is a failure.
                (base / 'DERIVED-00.npz').write_bytes(b'x' * len(files['DERIVED-00.npz']))
                # Export must be outside producer's owned result directory.
                import shutil
                shutil.rmtree(base / 'review-export')
                bad = R.review_result(base, 'fixture', 'fixture', 'fixture', 'fixture', 'a' * 64,
                                      R.ident(files['MANIFEST.json'])['sha256'], deadline_epoch=time.time() + 60)
                self.assertFalse(bad['accepted'])
                self.assertIn('input_file_SHA_DERIVED-00.npz', bad['failures'][0]['error'])

    def test_no_survivor_reweighting_and_bounded_export(self):
        summary, details, aggregate, inventory, parent, refs, _ = self.data
        changed = copy.deepcopy(aggregate)
        changed['root'][0]['observed_queries'] = 2
        with self.assertRaisesRegex(R.Rejected, 'root_fixed3_support'):
            R.validate_and_extract(summary, details, changed, inventory, parent, refs)
        with tempfile.TemporaryDirectory() as directory:
            result = {'schema': R.SCHEMA, 'status': 'ACCEPTED_COMPLETE', 'accepted': True,
                      'figure_data': 'x' * R.MAX_EXPORT, 'failures': []}
            receipt = R.write_export(Path(directory) / 'new', result)
            self.assertFalse(receipt['accepted'])
            self.assertLess(receipt['file']['bytes'], R.MAX_EXPORT)
            with self.assertRaises(FileExistsError):
                R.write_export(Path(directory) / 'new', result)

    def test_legitimate_zero_mass_noop_null_and_oracle_zero_source(self):
        saved_path = sys.path.copy()
        sys.path.insert(0, str(OLD))
        try:
            import kernel as K
            import test_kernel as T
            p, z = T.fixture(heads=8)
            p[0, 0, :, :6] = 0.
            def validate(values):
                analyzed = K.analyze_saved_grams(**T.saved(values, z)[0])
                for row in analyzed['rows']:
                    row.update(condition='C', layer_index=17, flow_from=0, flow_to=9)
                    row['comparison_role'] = R.role(row)
                    R.validate_scalar_row(row, R.Checks(), aggregate=False)
                return T.row(analyzed)
            unavailable = validate(p)
            self.assertIsNone(unavailable['baselines']['per_head_position_prefix_mass']['SSE'])
            p[1, 0, :, :6] = 0.
            p[1, 1, :, :6] *= .5
            noop = validate(p)
            self.assertEqual(noop['baselines']['per_head_position_prefix_mass']['status'], 'DEFINED_WITH_ZERO_MASS_NOOP')
            exported = R.compact(noop)
            self.assertEqual(exported['baselines']['per_head_position_prefix_mass']['zero_mass_noop_head_positions'], 10)
            p[0, ..., :6] = 0.
            oracle = validate(p)
            self.assertEqual(oracle['baselines']['signed_scalar_oracle']['status'], 'SOURCE_ZERO_NOOP')
            self.assertIsNone(oracle['oracle_full_window_alpha'])
        finally:
            sys.path[:] = saved_path

    def test_pinned_source_payload_identity(self):
        # Existing small source/metadata files, never a scientific array.
        inventory_path = OLD / 'SOURCE-INPUT-INVENTORY.json'
        inventory = json.loads(inventory_path.read_bytes())
        parent = Path(inventory['metadata']['PARENT-SUMMARY.json']['local_source_path'])
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory).resolve()
            script = (R.HERE / 'review.py').read_bytes()
            (base / 'review.py').write_bytes(script)
            manifest = {'schema': 'PAPER2-result-review-source-manifest-v1',
                        'files': [{'path': 'review.py', **R.ident(script)}]}
            raw = R.encoded(manifest)
            path = base / 'SOURCE-MANIFEST.json'
            path.write_bytes(raw)
            with mock.patch.object(R, 'HERE', base):
                refs, _ = R.validate_sources(inventory_path, OLD / 'SOURCE-MANIFEST.json', parent,
                                             path, R.ident(raw)['sha256'], R.Deadline(time.time() + 60))
                self.assertEqual(refs['input_inventory'], R.INVENTORY)
                (base / 'review.py').write_bytes(script[:-1] + b'x')
                with self.assertRaisesRegex(R.Rejected, 'input_file_SHA_review.py'):
                    R.validate_sources(inventory_path, OLD / 'SOURCE-MANIFEST.json', parent,
                                       path, R.ident(raw)['sha256'], R.Deadline(time.time() + 60))


if __name__ == '__main__':
    unittest.main()
