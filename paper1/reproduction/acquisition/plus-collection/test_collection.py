import copy
import importlib.util
import json
from pathlib import Path
import random
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock
import numpy as np

HERE = Path(__file__).resolve().parent
Q = HERE.parent
ROOT = next(p for p in Q.parents if (p/'AGENTS.md').is_file())
N = ROOT/'docs/experiments/analysis-system-20260929/native-write-flow-20261006-v1'
for p in (N/'libero-plus-policy-adapter-20261009-v2', Q/'recorder-candidate-v1', Q/'render-candidate-v1', Q/'render-runtime-v1', Q/'parallel-analysis-v1'):
    sys.path.insert(0, str(p))
import orchestrator as O
from test_contract import Fake, pair
import production as P
import environments as E
import collect
import consume
from canonical_step_capture import rng_copy, rng_arrays, rng_id


class Tests(unittest.TestCase):
    def test_production_isolates_rng_and_preserves_raw_side_effect_states(self):
        fixture=pair();b=fixture['branches']['reference'];noise=fixture['noise'];before=rng_id(rng_copy())
        def helper():
            np.random.random();random.random();return 'unchanged_numeric_result'
        with tempfile.TemporaryDirectory() as raw:
            store=P.Store(Path(raw)/'new')
            self.assertEqual(P.invariant_call(helper,b,noise,_rng_audit=P.rng_auditor(store,'calls')),'unchanged_numeric_result')
            self.assertEqual(before,rng_id(rng_copy()))
            rec=json.loads((store.path/'calls/rng-call-000.json').read_text())
            self.assertTrue(rec['environment_rng_restored']);self.assertTrue(all(rec['rng_changed_during_call'].values()))
            self.assertTrue((store.path/rec['raw_side_effect_states']['path']).exists())

    def test_production_exception_or_input_mutation_restores_rng_but_still_fails(self):
        for mutation in (False,True):
            fixture=pair();b=fixture['branches']['reference'];noise=fixture['noise'];before=rng_id(rng_copy())
            def helper():
                random.random();np.random.random()
                if mutation:b['observation']['image'][0,0,0]=1
                else:raise RuntimeError('synthetic neural failure')
            with tempfile.TemporaryDirectory() as raw:
                store=P.Store(Path(raw)/'new')
                with self.assertRaises(O.ContractError if mutation else RuntimeError):
                    P.invariant_call(helper,b,noise,_rng_audit=P.rng_auditor(store,'calls'))
                self.assertEqual(before,rng_id(rng_copy()))

    def test_actual_official_parser_prompt_is_used_without_worker_normalization(self):
        with tempfile.TemporaryDirectory() as raw:
            root=Path(raw)/'official';root.mkdir()
            bddl=root/'actual.bddl';bddl.write_text('(synthetic BDDL)')
            parser_path=root/'actual_parser.py';parser_path.write_text('# synthetic parser identity\n')
            def ident(path):
                import hashlib
                return {'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
            parser=SimpleNamespace(__file__=str(parser_path),get_problem_info=lambda path:{'language_instruction':'pIcK Actual Official Output'})
            record={'matched_prompt':'Pick lexical candidate','reference_contract':{'base_bddl':{'relative_to_plus_root':'actual.bddl',**ident(bddl)}}}
            prompt,proof=E.matched_base_prompt(record,root,{'official/actual_parser.py':ident(parser_path)},parser_module=parser)
            self.assertEqual(prompt,'pIcK Actual Official Output')
            self.assertFalse(proof['frame_lexical_exact_equal'])
            self.assertFalse(proof['normalization_implemented_by_this_worker'])
            self.assertEqual(record['matched_prompt'],'Pick lexical candidate')

    def startup_state(self):
        obs = {'agentview_image':np.zeros((256,256,3),np.uint8),'robot0_eye_in_hand_image':np.zeros((256,256,3),np.uint8),
               'robot0_eef_pos':np.zeros(3),'robot0_eef_quat':np.array([0.,0.,0.,1.]),'robot0_gripper_qpos':np.zeros(2)}
        physical = np.array([1., 2.])
        env = SimpleNamespace(env=SimpleNamespace(get_sim_state=lambda: physical))
        return {'env':env, 'observation':obs}, physical

    def test_startup_random_consumption_is_recorded_and_environment_streams_restored(self):
        state, _ = self.startup_state(); before = rng_id(rng_copy())
        def callback(*args):
            random.random(); np.random.random()
            return {'trace_qualified':True}
        with tempfile.TemporaryDirectory() as raw:
            store=P.Store(Path(raw)/'new')
            result=collect.isolated_startup(state,{},callback,store)
            self.assertTrue(result['trace_qualified']);self.assertEqual(before,rng_id(rng_copy()))
            receipt=json.loads((store.path/'STARTUP-STATE-ISOLATION-v2.json').read_text())
            self.assertTrue(receipt['changed_during_callback']['numpy_rng'])
            self.assertTrue(receipt['changed_during_callback']['python_rng'])
            self.assertTrue(all(receipt['same_after_rng_restore'].values()))
            self.assertFalse(receipt['physics_or_observation_restored'])

    def test_startup_exception_still_restores_both_random_streams(self):
        state,_=self.startup_state();before=rng_id(rng_copy())
        def callback(*args):
            random.random();np.random.random();raise RuntimeError('synthetic startup failure')
        with tempfile.TemporaryDirectory() as raw:
            store=P.Store(Path(raw)/'new')
            with self.assertRaisesRegex(RuntimeError,'synthetic startup'):
                collect.isolated_startup(state,{},callback,store)
            self.assertEqual(before,rng_id(rng_copy()))
            receipt=json.loads((store.path/'STARTUP-STATE-ISOLATION-v2.json').read_text())
            self.assertEqual(receipt['callback_exception_type'],'RuntimeError')

    def test_startup_physics_or_observation_mutation_still_hard_fails(self):
        for kind in ('physics','observation'):
            state,physical=self.startup_state();before=rng_id(rng_copy())
            def callback(*args):
                random.random()
                if kind=='physics':physical[0]+=1
                else:state['observation']['agentview_image'][0,0,0]=1
                return {'trace_qualified':True}
            with tempfile.TemporaryDirectory() as raw:
                store=P.Store(Path(raw)/'new')
                with self.assertRaises(O.ContractError):collect.isolated_startup(state,{},callback,store)
                self.assertEqual(before,rng_id(rng_copy()))
                receipt=json.loads((store.path/'STARTUP-STATE-ISOLATION-v2.json').read_text())
                self.assertTrue(receipt['changed_during_callback'][kind])
                self.assertFalse(receipt['same_after_rng_restore'][kind])

    def test_production_uses_single_operators_and_no_source_rescan(self):
        backend = Fake()
        backend.verify_sources = mock.Mock(side_effect=AssertionError('production source rescan'))
        for name in ('baseline', 'trajectory', 'prepare', 'capture'):
            setattr(backend, name, mock.Mock(wraps=getattr(backend, name)))
        fixture = pair(); b = fixture['branches']['variant']
        with tempfile.TemporaryDirectory() as raw:
            store = P.Store(Path(raw)/'new')
            result = P.record_branch(b, fixture['noise'], backend, store, 'treatment')
            self.assertEqual(backend.baseline.call_count, 1)
            self.assertEqual(backend.trajectory.call_count, 1)
            self.assertEqual(backend.prepare.call_count, 1)
            self.assertEqual(backend.capture.call_count, 6)
            backend.verify_sources.assert_not_called()
            self.assertTrue(result['report']['C_T_final_exact'])
            self.assertIsNone(result['report']['T_prefix_cache_identity'])
            self.assertTrue((store.path/'treatment/prefix.npz').exists())

    def test_full_PP_is_one_explicit_prefix_call(self):
        backend = Fake(); fixture = pair()
        with tempfile.TemporaryDirectory() as raw:
            store = P.Store(Path(raw)/'new')
            P.record_branch(fixture['branches']['reference'], fixture['noise'], backend, store, 'ref', full_pp=True)
            self.assertEqual(backend.prepare_flags, [True])
            with np.load(store.path/'ref/prefix.npz', allow_pickle=False) as data:
                self.assertIn('pg_probabilities', data.files)
                self.assertIn('pg_hidden_input', data.files)
                self.assertIn('pg_hidden_pre_attention_norm', data.files)

    def test_failed_T_exact_is_preserved_and_stable_C_can_continue_Y(self):
        fixture = pair()
        with tempfile.TemporaryDirectory() as raw:
            store = P.Store(Path(raw)/'new')
            out = P.record_branch(fixture['branches']['reference'], fixture['noise'], Fake('changed_actions'), store, 'ref')
            self.assertTrue((store.path/'ref/T-actions.npz').exists())
            self.assertFalse(out['report']['trace_qualified'])
            self.assertTrue(out['report']['canonical_Y_permitted'])
            self.assertEqual(out['views'], {})

    def test_finite_shadow_difference_is_reported_without_exact_gate(self):
        fixture = pair()
        with tempfile.TemporaryDirectory() as raw:
            result = P.record_branch(fixture['branches']['reference'], fixture['noise'], Fake('T_finite_shadow_difference'), P.Store(Path(raw)/'new'), 'ref')
            diff = result['report']['calls']['0']['O_vs_T_finite_differences']['hidden']
            self.assertGreater(diff['max_abs_FP32'], 0)

    def test_complete_rng_native_arrays_include_algorithm_and_no_pickle(self):
        values = rng_arrays(rng_copy())
        with tempfile.TemporaryDirectory() as raw:
            store = P.Store(Path(raw)/'new'); store.arrays('rng.npz', values, 'SYNTHETIC_RNG')
            with np.load(store.path/'rng.npz', allow_pickle=False) as data:
                for key, value in values.items():
                    self.assertEqual(data[key].dtype, value.dtype)
                    np.testing.assert_array_equal(data[key], value)

    def test_noise_is_shared_explicit_and_does_not_consume_sensor_rng(self):
        r = {'noise': {'parent_family_id': 'SYNTHETIC:parent', 'noise_label': 0, 'protocol': 'plus-shared-policy-noise-v1'}}
        before = rng_id(rng_copy())
        a = P.noise_for(r, 0); b = P.noise_for(r, 0)
        self.assertEqual(a.tobytes(), b.tobytes())
        self.assertEqual(before, rng_id(rng_copy()))
        import hashlib
        seed = int.from_bytes(hashlib.sha256(b'plus-shared-policy-noise-v1|SYNTHETIC:parent|0|0').digest()[:8], 'little')
        expected = np.random.Generator(np.random.PCG64(seed)).standard_normal((1,10,32)).astype(np.float32)
        self.assertEqual(a.tobytes(), expected.tobytes())
        r['noise']['noise_label'] = 1
        self.assertNotEqual(a.tobytes(), P.noise_for(r, 0).tobytes())

    def test_metric_envelope_matches_incremental_consumer(self):
        fixture = pair()
        with tempfile.TemporaryDirectory() as raw:
            record = P.record_branch(fixture['branches']['reference'], fixture['noise'], Fake(), P.Store(Path(raw)/'data'), 'ref')
            doc = {'schema':'plus-analysis-case-v1','data_role':'SYNTHETIC_SOFTWARE_FIXTURE',
                'identity':{k:'SYNTHETIC:'+k for k in consume.IDENTITY},
                'binding':{k:('a'*64 if k.endswith('_sha256') else 'SYNTHETIC:'+k) for k in consume.BINDING},
                'revision':0,'previous_sha256':None,'analysis_status':'complete','M':{'status':'complete','reason':None},
                'Y':{a:{'status':'not_collected','success':None}for a in ('reference','treatment')},
                'metrics':P.metric_rows(record,'treatment')+P.metric_rows(record,'reference')+P.metric_rows(record,'paired_difference',reference=record)}
            consume.validate(doc)
            marker_store = P.Store(Path(raw)/'closed'); marker_store.close_case(doc)
            self.assertTrue((marker_store.path/'ANALYSIS-CASE-v1.json').exists())
            with self.assertRaises(FileExistsError): marker_store.close_case(doc)

    def test_physics_XML_ignores_render_fields_but_not_mass_or_geometry(self):
        a = '<mujoco model="a"><asset><texture name="t" file="x"/></asset><worldbody><body name="b"><geom size="1" mass="2" rgba="1 0 0 1"/><camera pos="1 2 3"/></body></worldbody></mujoco>'
        b = a.replace('model="a"','model="b"').replace('file="x"','file="y"').replace('1 0 0 1','0 0 0 1').replace('pos="1 2 3"','pos="4 5 6"')
        self.assertEqual(E.physics_xml(a), E.physics_xml(b))
        self.assertNotEqual(E.physics_xml(a), E.physics_xml(b.replace('mass="2"','mass="3"')))

    def test_canonical_Y_uses_cached_q0_then_real_step_and_terminal(self):
        obs = {'agentview_image':np.zeros((256,256,3),np.uint8),'robot0_eye_in_hand_image':np.zeros((256,256,3),np.uint8),
               'robot0_eef_pos':np.zeros(3),'robot0_eef_quat':np.array([0.,0.,0.,1.]),'robot0_gripper_qpos':np.zeros(2)}
        class Env:
            prompt = 'synthetic goal'
            def __init__(self): self.env=SimpleNamespace(get_sim_state=lambda:np.array([1.])); self.n=0; self.closed=False
            def step(self, action): self.n+=1; return copy.deepcopy(obs), self.n==2
            def check_success(self): return self.n==2
            def close(self): self.closed=True
        env=Env(); state={'env':env,'observation':obs,'rng':rng_copy(),'physical':np.array([1.])}
        rec={'noise':{'parent_family_id':'SYNTHETIC:parent','noise_label':0,'protocol':'plus-shared-policy-noise-v1'},'official_case':{'max_policy_steps':20},'goal_binding_id':'SYNTHETIC:goal'}
        b=P.branch(obs,env.prompt,state['physical'],rec['goal_binding_id'])
        initial={'input_identity':O.request_binding(b),'physical_identity':O.array_id(state['physical']),
                 'actions':{'raw_actions':np.zeros((1,10,32),np.float32),'actions':np.zeros((1,10,7),np.float64)}}
        backend=SimpleNamespace(baseline=mock.Mock(side_effect=AssertionError('cached q0 must not call model')))
        with tempfile.TemporaryDirectory() as raw:
            result=collect.run_episode(state,rec,backend,P.Store(Path(raw)/'new'),'Y',lambda:None,initial=initial)
            self.assertEqual(result['status'],'accepted_success');self.assertEqual(result['policy_steps'],2)
            self.assertEqual(result['queries'],1);self.assertTrue(env.closed)

    def test_startup_T_only_mismatch_downgrades_but_C_repeat_failure_stops(self):
        fixture=pair();b=fixture['branches']['variant']
        state={'env':SimpleNamespace(source={'parsed_problem_sha256':'SYNTHETIC:source'},prompt=b['prompt']),
               'observation':b['observation'],'physical':b['physical_state']}
        record={'noise':{'parent_family_id':'SYNTHETIC:parent','noise_label':0,'protocol':'plus-shared-policy-noise-v1'}}
        with tempfile.TemporaryDirectory() as raw:
            out=collect.startup(state,record,Fake('changed_actions'),Path(raw)/'T_bad',False,synthetic=True)
            self.assertFalse(out['trace_qualified']);self.assertTrue(out['canonical_Y_permitted'])
            self.assertTrue((Path(raw)/'T_bad/FAILED-v1.json').exists())
            with self.assertRaises(O.ContractError):
                collect.startup(state,record,Fake('baseline_repeat'),Path(raw)/'C_bad',False,synthetic=True)


if __name__=='__main__': unittest.main()
