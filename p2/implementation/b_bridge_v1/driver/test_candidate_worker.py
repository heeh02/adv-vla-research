"""Synthetic C source/ABI/orchestration tests. No real array, model, or cloud IO."""
from pathlib import Path
from types import SimpleNamespace
import copy
import hashlib
import importlib.util
import json
import pickle
import tempfile
import unittest
import numpy as np
import ml_dtypes

HERE = Path(__file__).resolve().parent

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec); spec.loader.exec_module(value); return value

C = load('candidate_test_target', HERE / 'candidate_worker.py')
A = load('acceptance_test_target', HERE.parent / 'analysis/acceptance.py')
Z = load('quantization_test_target', HERE.parent / 'analysis/quantization.py')
CODEC = load('candidate_codec_test_target', HERE / 'native_codec.py')
B = ml_dtypes.bfloat16


def fixture():
    bf = lambda shape, v=1: np.full(shape, v, B)
    pg = {side: {'cache_k': bf((3, 1, 2, 1, 256)), 'cache_v': bf((3, 1, 2, 1, 256))} for side in ('C', 'T')}
    abi = {'num_steps': np.asarray(10, np.int32), 'num_steps_is_trace_input': np.asarray(True),
           'num_steps_weak_type': np.asarray(True), 'actual_dt': np.asarray(-.1, np.float32), 'dt_weak_type': np.asarray(True)}
    def single(target_value=1, active=False):
        reads = {'query_post_rope_scaled': bf((3, 1, 10, 1, 8, 256)),
                 'probabilities_native': bf((3, 1, 1, 8, 10, 12)),
                 'attention_output_post_WO_pre_gate': bf((3, 1, 10, 1024)),
                 'attention_gate': bf((3, 1, 1, 1024)), 'receiver_residual_input': bf((3, 1, 10, 1024), 0),
                 'hidden_after_attention_residual': bf((3, 1, 10, 1024)),
                 'hidden_post_block': bf((3, 1, 10, 1024)),
                 'prefix_key_used': bf((3, 1, 2, 1, 256)), 'prefix_value_used': bf((3, 1, 2, 1, 256)),
                 'suffix_key_post_rope': bf((3, 1, 10, 1, 256)), 'suffix_value': bf((3, 1, 10, 1, 256)),
                 'bridge_W_before': bf((3, 1, 10, 1024)), 'bridge_W_used': bf((3, 1, 10, 1024)),
                 'bridge_active': np.asarray([False, active, False]), 'mask': np.ones((1, 10, 12), bool),
                 'suffix_positions': np.arange(2, 12, dtype=np.int32)[None], 'query_defined': np.ones((1, 10), bool),
                 'adarms_condition': np.zeros((1, 1024), np.float32), 'final_readout': bf((1, 10, 1024), target_value)}
        for name in ('bridge_W_used', 'hidden_after_attention_residual', 'hidden_post_block'):
            reads[name][1] = target_value
        m = {'reads': reads, 'pg': copy.deepcopy(pg), 'gates': {k: np.asarray(True) for k in A.SINGLE_GATES},
             'actual_x': np.zeros((1, 10, 32), np.float32), 'actual_time': np.asarray(.4, np.float32),
             'time_weak_type': np.asarray(False), 'prefix_mask': np.ones((1, 2), bool),
             'prefix_positions': np.asarray([[0, 1]], np.int32), 'W_target_argument': bf((1, 10, 1024), target_value),
             'W_selector': np.asarray(active), 'actual_prefix_selector_matches': np.ones((3, 2), bool),
             'capture_layers': np.array([0, 8, 17], np.int32), 'generation_call': np.asarray(6, np.int32),
             'sampling_abi': copy.deepcopy(abi), 'completed_updates': np.asarray(0, np.int32),
             'prefix_prefills': np.asarray(2), 'suffix_evaluations': np.asarray(1), 'product_is_consumed': np.asarray(False)}
        return bf((1, 10, 32), target_value), m
    v, s = single()
    full_m = {'reads': {k: np.stack([value] * 5) for k, value in s['reads'].items()}, 'pg': copy.deepcopy(pg),
              'prefix_mask': s['prefix_mask'], 'prefix_positions': s['prefix_positions'], 'sampling_abi': copy.deepcopy(abi),
              'capture_layers': s['capture_layers'], 'capture_calls': np.array([5, 6, 7, 8, 9], np.int32),
              'flow': {'x': np.zeros((11, 1, 10, 32), np.float32), 'velocity_native': bf((10, 1, 10, 32)),
                       'time': np.full(11, .4, np.float32), 'time_weak_type': np.zeros(11, bool), 'dt': np.full(10, -.1, np.float32)},
              'bridge_gates': {k: np.asarray(True) for k in ('full_W_selector_inactive', 'full_schedule_is_B_pulse', 'all_before_equals_used')},
              'Vmod_argument': bf((1, 2, 1, 256))}
    full = {'raw32': np.zeros((1, 10, 32), np.float32), 'adapter7': np.zeros((1, 10, 7), np.float64), 'metadata': full_m}
    legacy = copy.deepcopy(full)
    legacy['metadata'].pop('bridge_gates')
    legacy['metadata']['reads'] = {k: val for k, val in legacy['metadata']['reads'].items() if not k.startswith('bridge_')}
    persistent = copy.deepcopy(legacy)
    for name in ('attention_output_post_WO_pre_gate', 'hidden_after_attention_residual', 'hidden_post_block'):
        persistent['metadata']['reads'][name][1, 1] = 2
    persistent['metadata']['flow']['velocity_native'][6] = 2
    references = {'pulse': legacy, 'persistent': persistent}
    saved = SimpleNamespace(raw_observations={'synthetic': np.zeros(3, np.uint8)}, noise=np.zeros((1, 10, 32), np.float32),
                            modified_v=bf((1, 2, 1, 256)))
    branches = {side: SimpleNamespace(decode_states=np.zeros((1, 32), np.float32)) for side in ('C', 'T')}
    I = SimpleNamespace(encode_loaded=lambda campaign, sv: branches,
        identity=lambda x: hashlib.sha256(pickle.dumps(x, protocol=4)).hexdigest(), current_prefix=lambda sv, m: True)
    old_calls = {role: SimpleNamespace(field=lambda *args, value=value: SimpleNamespace(storage=bf((5, 3, 1, 10, 1024), value).view(np.uint16)))
                 for role, value in (('baseline_none_1', 3), ('shared_pulse', 1), ('shared_persistent', 2))}
    targets = {role: bf((1, 10, 1024), value).view(np.uint16) for role, value in (('none', 1), ('natural_d', 2), ('common_c', 1.5))}
    T = SimpleNamespace(qualify_and_build_targets=lambda *args, **kwargs: (targets, {'synthetic': True}),
                        decode_native_target=lambda x: x.view(B).copy())
    invocations = []
    def run_single(t, c, x, tau, target, active):
        invocations.append(('single', float(target[0, 0, 0]), active))
        return single(float(target[0, 0, 0]), active)
    def run_full(*args):
        invocations.append(('full',))
        return full['raw32'], full['metadata']
    runtime = {'full_inactive': run_full, 'single': run_single, 'independent_single': run_single,
               'restore_time': lambda x, weak: x, 'wait': lambda x: x}
    campaign = SimpleNamespace(decode=lambda raw, states: np.zeros((1, 10, 7), np.float64))
    result = {'charged_entries': [], 'returned_entries': [], 'completed_entries': [], 'passed_entries': [],
              'prefix_prefills_reserved': 0, 'suffix_evaluations_reserved': 0,
              'model_binding': {'loaded_parameter_tree_sha256': 'b' * 64}}
    return dict(campaign=campaign, saved=saved, old_calls=old_calls, I=I, Q=SimpleNamespace(Q=SimpleNamespace(check_result=lambda m: True)),
                A=A, T=T, Z=Z, runtime=runtime, guard=lambda model: None, result=result, references=references,
                reference_result={}, source_ref={'sha256': 'a' * 64}, audit=b'synthetic', check=lambda: None), invocations, full


class CandidateTests(unittest.TestCase):
    def test_source_overlay_is_exclusive_and_old_bytes_unchanged(self):
        with tempfile.TemporaryDirectory() as d:
            old = Path(d) / 'old'; gm = old / 'repo/src/openpi/models/gemma.py'
            gm.parent.mkdir(parents=True); (old / 'metadata').mkdir()
            gm.write_bytes((HERE.parent / 'native/frozen/gemma.py').read_bytes())
            (old / 'keep.txt').write_bytes(b'unchanged original source')
            manifest = {'files': [{'path': 'repo/src/openpi/models/gemma.py', **C.R.digest(gm.read_bytes()), 'origin': 'old'}]}
            original = C.R.json_bytes(manifest); (old / 'metadata/SOURCE_MANIFEST-v1.json').write_bytes(original)
            oldref = C.R.OLD_MANIFEST
            try:
                C.R.OLD_MANIFEST = C.R.digest(original)
                dest, receipt = C.isolate_candidate_package({'old_package_root': str(old),
                    'output_dir': str(Path(d) / 'science'), 'run_id': 'synthetic-C'}, {'sha256': 'a'*64})
                self.assertEqual((old / 'metadata/SOURCE_MANIFEST-v1.json').read_bytes(), original)
                self.assertEqual(C.R.digest(gm.read_bytes())['sha256'], C.OLD_GEMMA_SHA)
                self.assertEqual(C.R.digest((dest / 'repo/src/openpi/models/gemma.py').read_bytes())['sha256'], C.NEW_GEMMA_SHA)
                self.assertEqual((dest / 'keep.txt').read_bytes(), b'unchanged original source')
                self.assertTrue(receipt['original_package_unchanged'])
                with self.assertRaisesRegex(C.R.WorkerFault, 'new_exclusive_candidate_source_package'):
                    C.isolate_candidate_package({'old_package_root': str(old), 'output_dir': str(Path(d) / 'science'),
                                                 'run_id': 'synthetic-C'}, {'sha256': 'a'*64})
            finally:
                C.R.OLD_MANIFEST = oldref

    def test_reference_objects_require_root_admission_and_exact_result_binding(self):
        with tempfile.TemporaryDirectory() as d:
            base = Path(d); rows = []; published = []
            for role, name in C.REFERENCE_NAMES.items():
                if role == 'reference_result': continue
                obj = {'raw32': np.zeros(1, np.float32), 'adapter7': np.zeros(1, np.float64), 'metadata': {}} if 'CALL' in name else {'synthetic_pg': np.zeros(1, np.float32)}
                ref = CODEC.save_tree(base / name, obj)
                published.append({'path': name, **ref})
                rows.append({'role': role, 'path': str(base/name), 'ref': {'path': name, 'uri': 'redacted-0e1cd6dbcd1d'+name, **ref}})
            for name in ('INPUT-QUALIFICATION.json', 'MODEL-BINDING.json', 'CHARGE-REF-01.json', 'CHARGE-REF-02.json', 'CHECK-REF-01.json', 'CHECK-REF-02.json'):
                published.append({'path': name, 'bytes': 1, 'sha256': 'f'*64})
            result = {'schema': 'B-BRIDGE-reference-result-v1', 'status': 'REFERENCE_QUALIFIED', 'synthetic': False,
                'phase': 'reference', 'run_id': 'synthetic-R', 'root_id': C.R.ROOT, 'query_index': 11,
                'source_manifest': {'sha256': C.REFERENCE_SOURCE_SHA}, 'analysis_complete': True,
                'scientific_complete': True, 'qualification_complete': True, 'prefix_prefills_reserved': 4,
                'suffix_evaluations_reserved': 20, 'unexecuted_entries': [], 'charged_but_completion_unproven': [],
                'outputs': published, **{k:['REF-01','REF-02'] for k in ('charged_entries','returned_entries','completed_entries','passed_entries')}}
            raw = C.R.json_bytes(result); (base/'RESULT.json').write_bytes(raw); ref = C.R.digest(raw)
            rows.append({'role':'reference_result','path':str(base/'RESULT.json'),'ref':{'path':'RESULT.json','uri':'redacted-e068fd37efc2',**ref}})
            context={'reference_dependencies':rows,'reference_admission':{'archive_complete':True,'readback_complete':True,
                'reference_run_id':'synthetic-R','reference_result_sha256':ref['sha256'],'source_manifest_sha256':C.REFERENCE_SOURCE_SHA}}
            actual, _, proof = C.load_reference(context, CODEC)
            self.assertIn('pg', actual['pulse']['metadata']); self.assertTrue(proof['native_reloaded'])
            context['reference_admission']['readback_complete'] = False
            with self.assertRaisesRegex(C.R.WorkerFault, 'archive_readback_admission'): C.load_reference(context, CODEC)
            context['reference_admission']['readback_complete'] = True
            rows[0]['ref']['sha256'] = '0'*64
            with self.assertRaisesRegex(C.R.WorkerFault, 'dependency_in_actual_complete_result'): C.load_reference(context, CODEC)

    def test_complete_fixed_tree_exact_budget_and_signed_readout(self):
        args, calls, _ = fixture()
        with tempfile.TemporaryDirectory() as d:
            args['store'] = C.OutputStore(d, CODEC)
            C.run_candidate(**args)
            self.assertEqual(len(calls), 11)
            self.assertEqual(args['result']['passed_entries'], list(C.CALL_IDS))
            self.assertEqual((args['result']['prefix_prefills_reserved'], args['result']['suffix_evaluations_reserved']), (22, 20))
            self.assertTrue(args['result']['scientific_complete'])
            self.assertFalse(args['result']['cross_state_same_delta_scientific_GO'])
            self.assertEqual(len(args['store'].refs), 39)
            self.assertTrue((Path(d) / 'QUANTIZATION-READOUT.json').is_file())

    def test_inactive_full_failure_preserved_no_single(self):
        args, calls, full = fixture(); full['raw32'][0, 0, 0] = 1
        with tempfile.TemporaryDirectory() as d:
            args['store'] = C.OutputStore(d, CODEC)
            with self.assertRaisesRegex(C.R.WorkerFault, 'complete_R_native_parity'): C.run_candidate(**args)
            self.assertEqual(len(calls), 1)
            self.assertTrue((Path(d) / 'CALL-CAN-01.npz').exists())
            self.assertFalse((Path(d) / 'CHARGE-S01.json').exists())

    def test_fresh_persistent_mismatch_stops_after_saved_independent_d(self):
        args, calls, _ = fixture()
        args['references']['persistent']['metadata']['flow']['velocity_native'][6, 0, 0, 0] = 3
        with tempfile.TemporaryDirectory() as d:
            args['store'] = C.OutputStore(d, CODEC)
            with self.assertRaisesRegex(A.AcceptanceError, 'native_exact_velocity'): C.run_candidate(**args)
            self.assertEqual(args['result']['charged_entries'][-1], 'S04')
            self.assertTrue((Path(d) / 'CALL-S04.npz').exists())
            self.assertFalse((Path(d) / 'CHARGE-S05.json').exists())

    def test_main_d_disagreement_stops_main_d_after_save(self):
        args, calls, _ = fixture()
        original = args['runtime']['single']
        def wrong(*callargs):
            v, m = original(*callargs)
            if float(callargs[-2][0, 0, 0]) == 2: v[0, 0, 0] = 3
            return v, m
        args['runtime']['single'] = wrong
        with tempfile.TemporaryDirectory() as d:
            args['store'] = C.OutputStore(d, CODEC)
            with self.assertRaisesRegex(C.R.WorkerFault, 'independent_and_main_natural_d'): C.run_candidate(**args)
            self.assertTrue((Path(d) / 'CALL-S06.npz').exists())
            self.assertFalse((Path(d) / 'CHARGE-S07.json').exists())

    def test_delayed_corruption_preserved_and_no_last_call(self):
        args, calls, _ = fixture()
        original = args['runtime']['single']; counter = [0]
        def changed(*callargs):
            counter[0] += 1; v, m = original(*callargs)
            if counter[0] == 6: v[0, 0, 0] = 3  # S09 main call, independent leaves separate.
            return v, m
        args['runtime']['single'] = changed
        with tempfile.TemporaryDirectory() as d:
            args['store'] = C.OutputStore(d, CODEC)
            with self.assertRaisesRegex(C.R.WorkerFault, 'delayed_repeat'): C.run_candidate(**args)
            self.assertTrue((Path(d) / 'CALL-S09.npz').exists())
            self.assertFalse((Path(d) / 'CHARGE-S10.json').exists())

    def test_non_L9_used_write_change_rejected_after_snapshot(self):
        args, calls, _ = fixture(); original = args['runtime']['single']
        def changed(*callargs):
            v, m = original(*callargs); m['reads']['bridge_W_used'][2, 0, 0, 0] = 2; return v, m
        args['runtime']['single'] = changed
        with tempfile.TemporaryDirectory() as d:
            args['store'] = C.OutputStore(d, CODEC)
            with self.assertRaisesRegex(C.R.WorkerFault, 'non_L9_actual_W'): C.run_candidate(**args)
            self.assertTrue((Path(d) / 'CALL-S01.npz').exists())
            self.assertFalse((Path(d) / 'CHARGE-S02.json').exists())

    def test_native_state_change_rejected_after_snapshot(self):
        args, calls, _ = fixture(); original = args['runtime']['single']
        def changed(*callargs):
            v, m = original(*callargs); m['actual_x'][0, 0, 0] = 1; return v, m
        args['runtime']['single'] = changed
        with tempfile.TemporaryDirectory() as d:
            args['store'] = C.OutputStore(d, CODEC)
            with self.assertRaisesRegex(A.AcceptanceError, 'native_exact_actual_x'): C.run_candidate(**args)
            self.assertTrue((Path(d) / 'CALL-S01.npz').exists())
            self.assertFalse((Path(d) / 'CHARGE-S02.json').exists())


if __name__ == '__main__': unittest.main()
