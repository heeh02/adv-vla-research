"""Real pinned codec + synthetic payload tests; never read actual neural arrays.

The fixed695-object production gate is exercised on source metadata and never
mocked or weakened. Full-corpus completion is deliberately not claimed by these
small fixtures. Original source/gate failures are tested separately from row ABI.
"""
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest import mock
import copy
import hashlib
import importlib.util
import json
import shutil
import sys
import tempfile
import time
import unittest
import numpy as np
import dev_loader as DL

RESEARCH=Path('/Users/researcher/.codex/worktrees/adv-vla-research-20260924/Adv-VLA-π0.5')
CONT=RESEARCH/'docs/experiments/state-dependent-response-use-20261013-v1/continuous'

@contextmanager
def aliases(values):
    previous={k:sys.modules.get(k)for k in values};sys.modules.update(values)
    try:yield
    finally:
        for k,v in previous.items():
            if v is None:sys.modules.pop(k,None)
            else:sys.modules[k]=v


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);value=importlib.util.module_from_spec(spec)
    sys.modules[name]=value;spec.loader.exec_module(value);return value


class DevLoaderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cm=DL.modules();cls.D=cls.cm.__enter__();cls.S=sys.modules['sources']
        with aliases({'prediction':cls.D.P,'reader':cls.D.R}):
            fp=module('_rss_synthetic_prediction_fixture',CONT/'task-blind-predictor-v1/test_prediction.py')
            with aliases({'test_prediction':fp}):
                cls.fixture=module('_rss_synthetic_codec_fixture',CONT/'task-blind-predictor-v1/test_reader.py')
        cls.old_adapter=module('_rss_original_signed_adapter',DL.HERE.parent/'signed_transfer_v1/input_adapter/adapter.py')
        cls.old_driver=module('_rss_original_signed_driver',DL.HERE.parent/'signed_transfer_v1/adapter.py')

    @classmethod
    def tearDownClass(cls):cls.cm.__exit__(None,None,None)

    def codec_fixture(self,folder):
        with aliases({'prediction':self.D.P,'reader':self.D.R}):
            return self.fixture.fixture_group(folder)

    def test_real_metadata_readset_and_original_source_pins(self):
        plan=json.loads((DL.HERE/'inventory/DEV-READ-PLAN-original.json').read_bytes())
        refs=self.D.W.validate_plan(plan)
        self.assertEqual(len(refs),695);self.assertEqual(sum(r['bytes']for r in refs),224484300)
        self.assertEqual(max(r['bytes']for r in refs),3327532)
        self.assertEqual(tuple(plan['population']['arms']),('shared','random_matched','distance_matched','full_v_restore'))
        for k,pin in self.S.PINS.items():self.assertEqual(hashlib.sha256((DL.HERE/'legacy'/k).read_bytes()).hexdigest(),pin)
        with self.assertRaisesRegex(ValueError,'exact_unique_readset'):
            self.D.W.validate_plan({**plan,'members':refs[:-1]})
        with self.assertRaisesRegex(ValueError,'original_complete_native_run'):
            self.D.W.original_gates({'status':'COMPLETE','scientific_complete':True,'synthetic':True,'failures':[]},[],[])

    def test_linux_guard_and_staged_binding_fail_closed_without_real_arrays(self):
        if not sys.platform.startswith('linux'):
            with self.assertRaisesRegex(ValueError,'real_arrays_only_cloud_Linux'):
                DL.load('/nonexistent',{'input_objects':[]},time.time()+60)
        with tempfile.TemporaryDirectory()as td:
            with self.assertRaisesRegex(ValueError,'exact_frozen_input_ref'):
                DL.load(td,{'input_objects':[]},time.time()+60,synthetic=True)
            plan=json.loads((DL.HERE/'inventory/DEV-READ-PLAN-original.json').read_bytes());r=plan['members'][0]
            bound={'uri':r['uri'],'path':'dummy.json','bytes':r['bytes'],'sha256':r['sha256']}
            Path(td,'dummy.json').write_bytes(b'{}')
            with self.assertRaisesRegex(ValueError,'whole_input_SHA'):
                DL.load(td,{'input_objects':[bound]},time.time()+60,synthetic=True)
            with self.assertRaisesRegex(ValueError,'absolute_science_deadline'):
                DL.load(td,{'input_objects':[bound]},time.time()-1,synthetic=True)

    def test_legacy_source_tamper_rejected_without_gate_relaxation(self):
        with tempfile.TemporaryDirectory()as td:
            root=Path(td);shutil.copytree(DL.HERE/'legacy',root/'legacy',ignore=shutil.ignore_patterns('__pycache__'))
            shutil.copyfile(DL.HERE/'LEGACY-SOURCE-PINS.json',root/'LEGACY-SOURCE-PINS.json')
            p=root/'legacy/dev_reader.py';p.write_bytes(p.read_bytes()+b'\n# synthetic tamper\n')
            with mock.patch.object(DL,'HERE',root):
                with self.assertRaisesRegex(ValueError,'byte_identical_legacy_source'):
                    with DL.modules():pass

    def test_original_codec_query_all_four_arms_and_raw_ABI(self):
        with tempfile.TemporaryDirectory()as td:
            plan,expected,group,_=self.codec_fixture(td);q,targets=self.D.R.build_query(group,plan)
            self.assertEqual(q.fingerprint(),expected.fingerprint());self.assertNotEqual(q.source_identity,q.fingerprint())
            self.assertEqual({t.arm for t in targets},set(self.D.P.ARMS))
            self.assertEqual(q.state_vector().shape,(2592,));self.assertEqual(q.state_vector().dtype,np.float64)
            self.assertEqual(q.first_flow_vector().shape,(320,));self.assertEqual(q.first_flow_vector().dtype,np.float64)
            self.assertEqual(q.p_native.value()[0,0].shape,(8,10,13));self.assertEqual(q.mask.value()[0].dtype,np.bool_)
            self.assertEqual(q.gate.value()[0,0].shape,(1024,))
            self.assertTrue(np.all(q.p_native.value()[0,0,:,:,:3].sum(-1)<1))
            for t in targets:
                self.assertEqual(t.effect.shape,(10,7));self.assertEqual(t.effect.dtype,np.float64)
                self.assertEqual(q.edit(t.arm).dtype,np.float32);self.assertEqual(q.edit(t.arm).shape,(3,256))
                self.assertEqual(t.input_identity,q.fingerprint());self.assertEqual(t.edit_identity,q.edit_identity(t.arm))
                self.assertEqual(group['effect_'+t.arm]['metadata']['technical']['selectors']['flow_mask'],[False]*5+[True]*5)
            self.assertTrue(np.array_equal(q.edit('full_v_restore'),
                            (q.clean_v.value().astype(np.float32)-q.treated_v.value().astype(np.float32))[0,:,0]))

    def test_distance_and_fullV_cannot_be_relabelled_as_pulse(self):
        for arm in ('distance_matched','full_v_restore'):
            with self.subTest(arm=arm),tempfile.TemporaryDirectory()as td:
                _,_,group,files=self.codec_fixture(td);ref=group['effect_'+arm]['ref'];path=files[ref['path']]
                with np.load(path,allow_pickle=False)as z:arrays={k:z[k].copy()for k in z.files}
                tree=json.loads(arrays['__schema__'].tobytes())
                tree['metadata']['technical']['selectors']['flow_mask']['values']=[i==5 for i in range(10)]
                arrays['__schema__']=np.frombuffer(json.dumps(tree).encode(),np.uint8);np.savez_compressed(path,**arrays)
                badref={**ref,**DL.digest(path)}
                with self.assertRaisesRegex(ValueError,'same_L9_persistent_V_operation'):
                    self.D.R.extract(path,badref)

    def test_old_cache_and_row_identity_ABI_using_original_writers(self):
        with tempfile.TemporaryDirectory()as td:
            plan,_,group,_=self.codec_fixture(td);q,targets=self.D.R.build_query(group,plan)
            with aliases({'prediction':self.D.P,'reader':self.D.R}):_,wo,_,_=self.fixture.fixture()
            features={'current':np.zeros(13,np.float64),'pooled':np.zeros(13,np.float64),'norm':np.zeros(13,np.float64)}
            model=SimpleNamespace(wo=wo,pooled=q.p_native.value()[0,0].astype(np.float64),features=lambda *_:features)
            target=next(t for t in targets if t.arm=='shared')
            old=self.S.row(self.D,model,q,target.effect,split='DEV',background='KT',
                           label_identity=target.source_identity,proof={'source_run_id':'synthetic'})
            expanded=self.old_adapter.expand_row(self.D,model,q,old)
            cache=self.old_driver.cache_arrays([expanded],{})
            ids=self.old_driver.row_manifest([expanded],SimpleNamespace(identity=self.D.P.array_id))
            ids=json.loads(json.dumps(ids));old_map={tuple(x['key']):x for x in ids}
            key=tuple([str(cache['row_keys'][0,0]),int(cache['row_keys'][0,1]),str(cache['row_keys'][0,2]),str(cache['row_keys'][0,3])])
            self.assertEqual(key,(q.root_id,q.query_index,'shared','KT'))
            self.assertEqual(old_map[key]['provenance']['input_identity'],q.fingerprint())
            want=self.D.P.local_write(q.p_native.value()[0,0].astype(np.float64),q.edit('shared').astype(np.float64),wo,q.gate).reshape(10,1024)
            self.assertTrue(DL.same_bytes(cache['effect'][0],target.effect))
            self.assertTrue(DL.same_bytes(cache['current_gated_write'][0],want))
            self.assertEqual(cache['split'][0],'DEV')

    def test_bitwise_QA_rejects_dtype_and_signed_zero_changes(self):
        y=np.zeros((10,7),np.float64)
        self.assertTrue(DL.same_bytes(y,y.copy()))
        self.assertFalse(DL.same_bytes(y,y.astype(np.float32)))
        z=y.copy();z[0,0]=-0.0
        self.assertTrue(np.array_equal(y,z));self.assertFalse(DL.same_bytes(y,z))
        self.assertFalse(DL.same_bytes(y,y.reshape(7,10)))

if __name__=='__main__':unittest.main(verbosity=2)
