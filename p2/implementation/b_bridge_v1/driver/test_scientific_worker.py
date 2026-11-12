"""Synthetic-only R orchestration/negative-path tests; no real arrays or JAX."""
from pathlib import Path
import copy
import hashlib
import importlib.util
import pickle
from types import SimpleNamespace
import tempfile
import unittest
import numpy as np
import ml_dtypes

HERE = Path(__file__).resolve().parent

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m

W = load('worker_test_target', HERE / 'scientific_worker.py')
N = load('codec_test_target', HERE / 'native_codec.py')
FIELDS = ('query_post_rope_scaled', 'probabilities_native', 'attention_output_post_WO_pre_gate',
          'attention_gate', 'receiver_residual_input', 'hidden_after_attention_residual',
          'hidden_post_block', 'prefix_key_used', 'prefix_value_used', 'suffix_key_post_rope', 'suffix_value')


def fixture():
    b = ml_dtypes.bfloat16
    saved = SimpleNamespace(noise=np.zeros((1, 10, 32), np.float32), modified_v=np.ones((1, 2, 1, 1), b))
    branches = {'C': np.zeros(1, np.float32), 'T': np.ones(1, np.float32)}
    I = SimpleNamespace(encode_loaded=lambda campaign, source: branches,
        identity=lambda x: hashlib.sha256(pickle.dumps(x, protocol=4)).hexdigest(), value=lambda x: x,
        current_prefix=lambda source, meta: True)
    def selectors(schedule):
        return (np.array([False, True]), np.arange(18) == 8,
                np.arange(10) == 5 if schedule == 'pulse' else np.arange(10) >= 5, np.array(True))
    Q = SimpleNamespace(CALLS=(5, 6, 7, 8, 9), LAYERS=(0, 8, 17), FIELDS=FIELDS,
                        selectors=selectors, check_result=lambda meta: True)
    ident = I.identity((branches, saved.noise, saved.modified_v))
    actual = {}; old = {}
    for _, schedule, role in W.CALLS:
        m = {'reads': {k: np.ones((5, 3, 1, 1), b) for k in FIELDS},
             'flow': {'x': np.zeros((11, 1, 10, 32), np.float32),
                      'velocity_native': np.ones((10, 1, 10, 32), b),
                      'time': np.arange(11, dtype=np.float32), 'time_weak_type': np.ones(11, bool),
                      'dt': np.ones(10, np.float32)},
             'Vmod_argument': saved.modified_v.copy(),
             'pg': {side: {'cache_k': np.ones((3, 1, 2, 1), b)} for side in ('C', 'T')},
             'selectors': dict(zip(('kv_mask', 'layer_mask', 'flow_mask', 'use_vmod'), selectors(schedule)))}
        m['reads'].update({k: np.ones((5, 1, 1), np.int32) for k in ('mask', 'suffix_positions', 'query_defined', 'adarms_condition')})
        if schedule == 'persistent':
            for k in ('attention_output_post_WO_pre_gate', 'prefix_value_used', 'hidden_post_block'):
                m['reads'][k][1, 1] = 2
        a = {'raw32': np.zeros((1, 10, 32), np.float32), 'adapter7': np.zeros((1, 10, 7), np.float64), 'metadata': m}
        actual[schedule] = a
        arrays = {k: copy.deepcopy(v) for k, v in m.items() if k not in ('pg', 'reads', 'Vmod_argument')}
        arrays['reads'] = {k: copy.deepcopy(v) for k, v in m['reads'].items() if k not in ('prefix_key_used', 'prefix_value_used')}
        arrays.update(raw32=a['raw32'].copy(), adapter7=a['adapter7'].copy())
        used = [{'flow_call': j, 'layer_index': li, 'field': name,
                 'actual_identity': W.array_identity(m['reads'][name][ji, si])}
                for ji, j in enumerate(Q.CALLS) for si, li in enumerate(Q.LAYERS)
                for name in ('prefix_key_used', 'prefix_value_used')]
        old[role] = SimpleNamespace(arrays=arrays, metadata={'input_identity': ident,
            'Vmod_actual_identity': W.array_identity(m['Vmod_argument']), 'actual_prefix_used': used})
    rt = {'common': lambda t, c, noise, vmod, schedule: (actual[schedule]['raw32'], actual[schedule]['metadata']),
          'decode': lambda raw, branch: np.zeros((1, 10, 7), np.float64), 'wait': lambda x: x}
    result = {'charged_entries': [], 'returned_entries': [], 'completed_entries': [], 'passed_entries': [],
              'prefix_prefills_reserved': 0, 'suffix_evaluations_reserved': 0}
    return I, Q, saved, old, actual, rt, result


class ReferenceTests(unittest.TestCase):
    def run_case(self, folder, fixture_data):
        I, Q, saved, old, actual, rt, result = fixture_data
        store = W.OutputStore(folder, N)
        W.run_reference(campaign=object(), saved=saved, old_calls=old, I=I, Q=Q, runtime=rt,
                        guard=lambda _: None, store=store, result=result, check=lambda: None)
        return store, result

    def test_two_full_calls_charge_persist_and_gate(self):
        data = fixture()
        with tempfile.TemporaryDirectory() as d:
            store, r = self.run_case(d, data)
            self.assertEqual(r['passed_entries'], ['REF-01', 'REF-02'])
            self.assertEqual((r['prefix_prefills_reserved'], r['suffix_evaluations_reserved']), (4, 20))
            self.assertFalse(r['historical_full_PG_payload_reloaded'])
            self.assertFalse(r['same_W_setter_qualified'])
            obj = N.load_tree(Path(d) / 'CALL-REF-02.npz', store.refs['CALL-REF-02.npz'])
            self.assertNotIn('pg', obj['metadata'])
            self.assertEqual(len(store.refs), 8)

    def test_parity_failure_saves_actual_first(self):
        data = fixture(); data[4]['pulse']['raw32'][0, 0, 0] = 1
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(W.WorkerFault, 'all_saved_old_CALL'):
                self.run_case(d, data)
            self.assertTrue((Path(d) / 'CALL-REF-01.npz').is_file())
            self.assertEqual(data[6]['completed_entries'], ['REF-01'])
            self.assertEqual(data[6]['passed_entries'], [])
            self.assertFalse((Path(d) / 'CHARGE-REF-02.json').exists())

    def test_failed_invoke_remains_charged_not_completed(self):
        data = fixture()
        def fail(*args): raise RuntimeError('synthetic failure')
        data[5]['common'] = fail
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(RuntimeError): self.run_case(d, data)
            self.assertTrue((Path(d) / 'CHARGE-REF-01.json').is_file())
            self.assertEqual(data[6]['charged_entries'], ['REF-01'])
            self.assertEqual(data[6]['returned_entries'], [])
            self.assertEqual(data[6]['completed_entries'], [])

    def test_changed_PG_preserved_before_rejection(self):
        data = fixture(); data[4]['persistent']['metadata']['pg']['T']['cache_k'][0, 0, 0, 0] = 2
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(W.WorkerFault, 'complete_fresh_PG'):
                self.run_case(d, data)
            self.assertTrue((Path(d) / 'UNEXPECTED-PG-T-REF-02.npz').is_file())
            self.assertTrue((Path(d) / 'UNEXPECTED-PG-C-REF-02.npz').is_file())
            self.assertTrue((Path(d) / 'CALL-REF-02.npz').is_file())

    def test_unchanged_old_payload_but_wrong_prefix_identity_rejected(self):
        data = fixture()
        data[3]['shared_pulse'].metadata['actual_prefix_used'][0]['actual_identity']['sha256'] = '0' * 64
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(W.WorkerFault, 'old_actual_used_prefix'):
                self.run_case(d, data)
            self.assertTrue((Path(d) / 'CALL-REF-01.npz').is_file())

    def test_changed_old_receiver_stops_second_reference_after_save(self):
        data = fixture()
        field = 'query_post_rope_scaled'
        data[4]['persistent']['metadata']['reads'][field][1, 1] = 2
        data[3]['shared_persistent'].arrays['reads'][field][1, 1] = 2
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(W.WorkerFault, 'same_j6_L1_L9_receiver'):
                self.run_case(d, data)
            self.assertTrue((Path(d) / 'CALL-REF-02.npz').is_file())
            self.assertEqual(data[6]['completed_entries'], ['REF-01', 'REF-02'])
            self.assertEqual(data[6]['passed_entries'], ['REF-01'])

    def test_nonfinite_bits_saved_before_qualification_failure(self):
        data = fixture(); data[4]['pulse']['raw32'][0, 0, 0] = np.nan
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(W.WorkerFault, 'all_saved_old_CALL'):
                self.run_case(d, data)
            self.assertTrue((Path(d) / 'CALL-REF-01.npz').is_file())
            self.assertEqual(data[6]['completed_entries'], ['REF-01'])


if __name__ == '__main__': unittest.main()
