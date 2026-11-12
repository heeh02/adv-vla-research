"""Metadata-only synthetic tests and actual frozen four-table identity checks."""
import copy
import gzip
import io
import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
import zipfile
import inventory_math as M
import worker as W
from metadata_reader import MetadataReader


def outcome(n=7, success=True):
    return {'status': 'accepted_success' if success else 'accepted_failure', 'success': success,
            'policy_steps': n, 'queries': (n + 4) // 5,
            'terminal_reason': 'official_success' if success else 'policy_budget_exhausted',
            'terminal_observation_physics_alignment': 'verified_call_order',
            'policy_trace': [{'query': (i - 1) // 5, 'slot': (i - 1) % 5, 'policy_step': i,
                              'done': False, 'official_success': success and i == n} for i in range(1, n + 1)]}


def expected(y): return {k: y[k] for k in ('status', 'success', 'policy_steps', 'queries')}


def fields(kind):
    spec = {'robot0_eef_pos': ([3], 24), 'robot0_eef_quat': ([4], 32), 'robot0_gripper_qpos': ([2], 16),
            'physical_sim_state': ([110], 880), 'noise': ([1, 10, 32], 1280),
            'raw_actions': ([1, 10, 32], 1280), 'actions': ([1, 10, 7], 560)}
    names = M.INPUT_FIELDS if kind == 'query_input' else M.ACTION_FIELDS if kind == 'query_actions' else M.OBS_FIELDS
    return {k: {'shape': spec[k][0], 'bytes': spec[k][1], 'sha256': 'a' * 64,
                'logical_dtype': 'float32' if k in ('noise', 'raw_actions') else 'float64',
                'stored_dtype': '<f4' if k in ('noise', 'raw_actions') else '<f8', 'finite': True} for k in names}


class FakeReader:
    instances = []
    def __init__(self, signer, row, deadline, cache_root):
        self.mapped, self.native, self.metadata_raw, self.read_members = {}, {}, {}, []
        self.index = {'schema': 'synthetic_metadata_only'}
        self.range = SimpleNamespace(range_requests=[])
        self.ys = {arm: outcome() for arm in ('reference', 'treatment')}
        for arm in self.ys:
            for name, kind in M.expected_members(arm, [0, 1]).items():
                raw = json.dumps(self.ys[arm]).encode() if kind == 'Y_json' else b''
                arc = {'bytes': len(raw) if kind == 'Y_json' else 400000, 'sha256': M.sha(raw) if raw else 'b' * 64}
                self.mapped[name] = {'row': arc, 'header_offset': 0, 'payload_offset': 512, 'header': b'0' * 512}
                if kind != 'Y_json':
                    role = {'query_input': 'actual_Y_wrapper_input', 'query_actions': 'canonical_native_sampler',
                            'terminal': 'actual_last_returned_wrapper_terminal'}[kind]
                    self.native[name] = {**arc, 'role': role, 'fields': fields(kind)}
        self.instances.append(self)
    def member(self, name, limit):
        assert name.endswith('/Y/Y-v1.json'), 'test must never request arrays'
        raw = json.dumps(self.ys[name.split('/')[0]]).encode()
        self.metadata_raw[name] = raw
        self.read_members.append({'member': name, 'bytes': len(raw), 'sha256': M.sha(raw)})
        return raw


class Writer:
    def __init__(self): self.raw = {}
    def put(self, path, relative):
        raw = Path(path).read_bytes(); self.raw[relative] = raw
        return {'uri': 'synthetic://inventory/' + relative, 'bytes': len(raw), 'sha256': M.sha(raw)}


class Tests(unittest.TestCase):
    def test_actual_four_table_join_has_exact404_and_28_tasks(self):
        here = Path(__file__).resolve().parent
        inputs = {name: (here / name).read_bytes() for name in M.INPUT_FILES}
        request = json.loads((here / 'REQUEST-v1.json').read_bytes())
        for name, raw in inputs.items(): self.assertEqual(request['inputs'][name], {'sha256': M.sha(raw), 'bytes': len(raw)})
        plan = M.build_plan(inputs)
        self.assertEqual(len(plan), 404); self.assertEqual(len({r['task'] for r in plan}), 28)
        self.assertEqual(sum(r['native_bytes'] is None for r in plan), 180)
        self.assertEqual(sum(r['expected_Y']['reference']['queries'] for r in plan), 14753)
        self.assertEqual(sum(r['expected_Y']['treatment']['queries'] for r in plan), 16308)

    def test_last_partial_query_uses_actual_terminal_step(self):
        y = outcome(); r = M.validate_trace(y, expected(y))
        self.assertEqual(r['query_policy_steps'], [0, 5])
        self.assertEqual(r['terminal_policy_step'], 7)
        self.assertEqual(r['queries'][-1]['executed_slots'], [0, 1])
        self.assertEqual(r['controls'][-1]['s_before'], 6)
        self.assertEqual(len(r['controls']), 7)

    def test_trace_rejects_off_by_one_slots_and_premature_terminal(self):
        for key, index, value in [('policy_step', 0, 0), ('slot', 5, 3), ('official_success', 3, True)]:
            y = outcome(); y['policy_trace'][index][key] = value
            with self.assertRaises(M.InventoryError): M.validate_trace(y, expected(y))

    def test_trace_overview_disagreement_not_recomputed_as_new_outcome(self):
        y = outcome(); e = expected(y); e['policy_steps'] = 8
        with self.assertRaisesRegex(M.InventoryError, 'Y_overview_policy_steps_mismatch'): M.validate_trace(y, e)

    def test_member_absence_keeps_unknown_size(self):
        row, issues = M.member_record('reference/Y/query-0001-input.npz', 'query_input', {}, {})
        self.assertIsNone(row['bytes']); self.assertEqual(issues, ['source_member_absent'])
        cost = M.costs([row])['query_input']; self.assertEqual(cost['missing_member_bytes_unknown'], 1)

    def test_archive_native_disagreement_and_missing_field_explicit(self):
        name = 'reference/Y/terminal.npz'; arc = {'bytes': 100, 'sha256': 'b' * 64}
        mapped = {name: {'row': arc, 'header': b'0' * 512, 'header_offset': 0, 'payload_offset': 512}}
        native = {name: {**arc, 'role': 'actual_last_returned_wrapper_terminal', 'fields': fields('terminal')}}
        native[name]['sha256'] = 'c' * 64
        rec, issues = M.member_record(name, 'terminal', mapped, native)
        self.assertEqual(issues, ['archive_native_identity_mismatch'])
        native[name]['sha256'] = arc['sha256']; del native[name]['fields']['robot0_eef_quat']
        rec, issues = M.member_record(name, 'terminal', mapped, native)
        self.assertIn('field_absent', issues); self.assertFalse(rec['payload_read_and_verified'])

    def test_metadata_reader_refuses_NPZ_even_if_caller_asks(self):
        reader = MetadataReader.__new__(MetadataReader)
        with self.assertRaisesRegex(M.InventoryError, 'inventory_forbids_array_payload_read'):
            reader.member('reference/Y/query-0000-input.npz')
        with self.assertRaisesRegex(ValueError, 'inventory_forbids_array_payload_read'):
            reader.fields('reference/Y/query-0000-input.npz', ('robot0_eef_pos',))

    def test_synthetic_inventory_to_cloud_only_readset_and_metadata_cache(self):
        row = {'case_id': 'synthetic', 'task': 'task', 'suite': 'libero_10', 'family': 'Camera Viewpoints',
               'index_sha256': 'a' * 64, 'expected_Y': {arm: expected(outcome()) for arm in ('reference', 'treatment')}}
        with tempfile.TemporaryDirectory() as d:
            report, metadata = W.inventory_case(row, None, time.time() + 400, d, FakeReader)
            self.assertEqual(report['status'], 'complete_manifest_and_trace_inventory')
            self.assertEqual(len(report['members']), 12)
            self.assertEqual(report['NPZ_payload_reads'], 0)
            self.assertEqual(len(FakeReader.instances[-1].read_members), 2)
            writer = Writer(); receipt = W.publish_case(writer, Path(d), report, metadata)
            cache = next(raw for k, raw in writer.raw.items() if k.endswith('.zip'))
            with zipfile.ZipFile(io.BytesIO(cache)) as z:
                self.assertFalse(any(k.endswith('.npz') for k in z.namelist()))
            frozen = next(raw for k, raw in writer.raw.items() if k.endswith('.json.gz'))
            reread = json.loads(gzip.decompress(frozen))
            self.assertEqual(reread['arms']['reference']['terminal_policy_step'], 7)
            summary = W.summarize([receipt], [row])
            self.assertEqual(summary['whole_member_payload_costs']['query_input']['member_count'], 4)
            self.assertFalse(summary['success_rate_recomputed'])

    def test_foreign_error_secret_not_saved(self):
        e = RuntimeError('synthetic-secret-or-signed-url')
        record = W.safe_error(e)
        self.assertIsNone(record['code']); self.assertNotIn('secret', json.dumps(record))


if __name__ == '__main__': unittest.main(verbosity=2)
