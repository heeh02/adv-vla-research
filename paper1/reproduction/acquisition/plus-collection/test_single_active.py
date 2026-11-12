"""Offline checks for saved pre-init comparison and single-live-wrapper order."""
import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock
import numpy as np

from test_collection import E, O, P, collect


class Tests(unittest.TestCase):
    def test_preinit_uses_saved_typed_fields_names_counts_and_xml(self):
        model = SimpleNamespace()
        fields, dimensions, names = {}, {}, {}
        for entity in ('body', 'joint', 'dof', 'geom', 'actuator'):
            setattr(model, E.S.COUNTS[entity], 1)
            dimensions[entity] = 1
            names[entity] = [{'id': 0, 'name': entity + '-0'}]
            setattr(model, entity + '_id2name', lambda i, e=entity: e + '-' + str(i))
            for field in E.S.FIELDS[entity].split():
                if field == 'geom_matid': continue
                value = np.array([1], np.float64)
                setattr(model, field, value)
                fields[field] = {**O.array_id(value), 'stored_dtype': value.dtype.str, 'finite': True}
        xml = '<mujoco><worldbody><body name="b" mass="1"/></worldbody></mujoco>'
        saved = {'schema': 'plus-passive-runtime-model-snapshot-v1', 'dimensions': dimensions,
                 'named_compiled_entities': names, 'compiled_fields': {'fields': fields},
                 'physics_XML_projection': E.physics_xml(xml)}
        E.verify_preinit_compiled(model, xml, saved)
        model.body_mass = np.array([2], np.float64)
        with self.assertRaisesRegex(Exception, 'reference compiled physics before init'):
            E.verify_preinit_compiled(model, xml, saved)
        model.body_mass = np.array([1], np.float64)
        with self.assertRaisesRegex(Exception, 'reference physics XML before init'):
            E.verify_preinit_compiled(model, xml.replace('mass="1"', 'mass="2"'), saved)

    def test_variant_is_closed_before_base_constructed_and_pair_uses_saved_q0(self):
        events = []
        class Env:
            def __init__(self, role, *args, **kwargs):
                if role == 'reference':
                    self.assert_variant_closed()
                    self.saved = kwargs['variant_preinit_snapshot']
                self.role = role
                self.env = object()
                self.prompt = 'same prompt'
                self.source = {'resolved_BDDL': {'sha256': 'a'}, 'scene': {'sha256': 'b'}, 'native_init0': {'sha256': 'c'}}
                events.append('create-' + role)
            def assert_variant_closed(self):
                assert 'close-treatment' in events and 'create-reference' not in events
            def close(self):
                self.env = None
                events.append('close-' + self.role)
        class Store:
            def __init__(self, path): self.path = Path(path)
            def json(self, name, value): events.append('save-' + name)
            def close_case(self, value): events.append('close-case')
        def initialize(env, case, store, prefix, check, **kwargs):
            events.append('init-' + prefix)
            return {'env': env, 'observation': {'image': np.zeros((2,2,3), np.uint8)},
                    'physical': np.array([1.], np.float64), 'rng': {'x': 1}}
        def record(branch, noise, backend, store, prefix, **kwargs):
            events.append('M-' + prefix)
            return {'report': {'trace_qualified': True}}
        def episode(state, record, backend, store, prefix, check, initial=None):
            events.append('Y-' + prefix)
            state['env'].close()
            return {'status': 'accepted_failure', 'success': False}
        def qualify(ref, var, ref_snapshot, var_snapshot, record):
            self.assertNotIn('env', var)
            self.assertEqual(var['prompt'], 'same prompt')
            self.assertEqual(var['physical'].tolist(), [1.])
            self.assertEqual(var_snapshot['marker'], 'treatment/model')
            events.append('pair-saved-q0')
            return {'qualified': True}
        record_ = {'identity': {'case_id': 'one:n0', 'family': 'Background Textures', 'task': 'task'},
                   'reference_contract': {'base_bddl': {'sha256': 'a'}}, 'goal_binding_id': 'goal',
                   'noise': {'noise_label': 0}, 'full_pp': False, 'collect_reference_Y': True,
                   'official_case': {'max_policy_steps': 280}}
        backend = SimpleNamespace(runtime_identity={'checkpoint': {'sha256': 'model'}})
        with tempfile.TemporaryDirectory() as tmp, \
             mock.patch.object(P, 'Store', Store), \
             mock.patch.object(E, 'matched_base_prompt', return_value=('same prompt', {})), \
             mock.patch.object(E, 'OfficialPlusEnvironment', side_effect=lambda *a, **k: Env('treatment')), \
             mock.patch.object(E, 'BaseReferenceEnvironment', side_effect=lambda *a, **k: Env('reference', **k)), \
             mock.patch.object(E, 'initialize', side_effect=initialize), \
             mock.patch.object(E, 'snapshot', side_effect=lambda state, store, name: {'marker': name}), \
             mock.patch.object(E, 'qualify_pair', side_effect=qualify), \
             mock.patch.object(collect, 'isolated_startup', return_value={'trace_qualified': True}), \
             mock.patch.object(collect, 'observation_identity', return_value={'id': 'q0'}), \
             mock.patch.object(collect, 'run_episode', side_effect=episode), \
             mock.patch.object(P, 'branch', return_value={}), \
             mock.patch.object(P, 'noise_for', return_value=np.zeros((1,10,32),np.float32)), \
             mock.patch.object(P, 'record_branch', side_effect=record), \
             mock.patch.object(P, 'metric_rows', return_value=[]):
            result = collect.collect_case(record_, classification={}, init_reader=None, plus_root=tmp,
                runtime_manifest={}, backend=backend, out=Path(tmp)/'case', startup_callback=lambda *_: {},
                check=lambda: None, binding={}, run_id='new-run', failure_location=lambda e,p: {'phase': p})
        self.assertIsNone(result['fatal'])
        self.assertEqual(result['M']['status'], 'complete')
        self.assertLess(events.index('M-M/treatment'), events.index('Y-treatment/Y'))
        self.assertLess(events.index('Y-treatment/Y'), events.index('close-treatment'))
        self.assertLess(events.index('close-treatment'), events.index('create-reference'))
        self.assertLess(events.index('M-M/reference'), events.index('Y-reference/Y'))
        self.assertIn('pair-saved-q0', events)


if __name__ == '__main__':
    unittest.main(verbosity=2)
