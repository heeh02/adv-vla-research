"""Synthetic corner cases are unit tests only; no fixture is a scientific case."""
import copy
import gzip
import importlib.util
import json
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('d1',ROOT/'pipeline.py');p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)

class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.payload=json.loads(gzip.decompress((ROOT/'data/MOTION-C-PLOT-v1.json.gz').read_bytes()))
        cls.plan=json.loads((ROOT/'data/PANEL-PLAN.json').read_text());cls.config=json.loads((ROOT/'CONFIG.json').read_text())
    def fixture(self):
        row=copy.deepcopy(self.payload['rows'][0]);row['outcomes']={a:True for a in p.ARMS};row['noise_pair_cells']={'0':'SS','1':'SS'}
        for q in row['arm_qualification'].values(): q.update(status='accepted_success',issues=[],available_pose_clocks=[5,60])
        return row
    def motion_values(self,row,tr,ref):
        for pair,v in [('TR0',tr),('TR1',tr),('R0_R1',ref),('R0_R0prime',0)]:
            row['motion'][pair]['5']['position_distance'].update(value=v,status='measured',reason=None)
        return row
    def test_real_schema(self): self.assertEqual(p.validate(self.payload,self.plan)['unknown_outcomes'],39)
    def test_duplicate_key(self):
        x=copy.deepcopy(self.payload);x['rows'][1]=x['rows'][0]
        with self.assertRaisesRegex(ValueError,'duplicate observed'): p.validate(x,self.plan)
    def test_pairing_conflict(self):
        x=copy.deepcopy(self.payload);x['rows'][0]['noise_pair_cells']['0']='SF'
        with self.assertRaisesRegex(ValueError,'paired outcome'): p.validate(x,self.plan)
    def test_plan_identity_conflict(self):
        x=copy.deepcopy(self.payload);x['rows'][0]['family']='Sensor Noise'
        with self.assertRaisesRegex(ValueError,'plan identity'): p.validate(x,self.plan)
    def test_illegal_unit(self):
        x=copy.deepcopy(self.payload);x['rows'][0]['motion']['TR0']['5']['position_distance']['unit']='mm'
        with self.assertRaisesRegex(ValueError,'illegal motion unit'): p.validate(x,self.plan)
    def test_negative_distance(self):
        x=copy.deepcopy(self.payload);x['rows'][0]['motion']['TR0']['5']['position_distance']['value']=-.1
        with self.assertRaisesRegex(ValueError,'negative'): p.validate(x,self.plan)
    def test_outcome_bool_strict(self):
        x=copy.deepcopy(self.payload);x['rows'][0]['outcomes']['R0']=1
        with self.assertRaisesRegex(ValueError,'nullable'): p.validate(x,self.plan)
    def test_single_unknown_fixed_noise_weight(self):
        v={a:False for a in p.ARMS};v['R0']=True;v['T0']=None
        b=p.variant_bounds(v)
        self.assertEqual((b['harm']['lower'],b['harm']['upper']),(0,.5))
        self.assertEqual((b['risk_difference']['lower'],b['risk_difference']['upper']),(-.5,0))
    def test_multiple_unknown_shared_constraint(self):
        v={a:False for a in p.ARMS};v['R0']=None
        b=p.variant_bounds(v)
        self.assertEqual((b['excess_flip_vs_policy']['lower'],b['excess_flip_vs_policy']['upper']),(-.5,0))
        # Naive independent interval subtraction would wrongly yield [-1,.5].
        naive=(b['tr_flip']['lower']-b['rr_policy_flip']['upper'],b['tr_flip']['upper']-b['rr_policy_flip']['lower'])
        self.assertEqual(naive,(-1,.5));self.assertIsNone(b['excess_flip_vs_policy']['point'])
        v['R1']=None;b=p.variant_bounds(v)
        self.assertEqual((b['excess_flip_vs_policy']['lower'],b['excess_flip_vs_policy']['upper']),(-.5,1))
    def test_all_unknown_marginal_bounds(self):
        b=p.variant_bounds({a:None for a in p.ARMS})
        self.assertEqual((b['risk_difference']['lower'],b['risk_difference']['upper']),(-1,1))
        self.assertEqual(b['ss']['upper'],1);self.assertEqual(b['ff']['upper'],1)
        # The two marginal upper endpoints cannot be attained jointly.
        for values in p.itertools.product((False,True),repeat=5):
            m=p.complete_metrics(dict(zip(p.ARMS,values)));self.assertLessEqual(m['ss']+m['ff'],1)
    def test_exact_identities_every_assignment(self):
        for vals in p.itertools.product((False,True),repeat=5):
            m=p.complete_metrics(dict(zip(p.ARMS,vals)))
            self.assertEqual(m['risk_difference'],m['recovery']-m['harm'])
            self.assertEqual(m['tr_flip'],m['recovery']+m['harm'])
            self.assertEqual(m['ss']+m['ff']+m['tr_flip'],1)
    def test_missing_fixed_clock_not_interpolated(self):
        row=self.fixture();row['arm_qualification']['T1']['available_pose_clocks']=[5]
        v,why=p.read_motion(row,'TR1',60,'position_distance');self.assertIsNone(v);self.assertIn('not_verified',why)
    def test_missing_metric_clock(self):
        row=self.fixture();del row['motion']['TR0']['5']
        v,why=p.read_motion(row,'TR0',5,'position_distance');self.assertIsNone(v);self.assertEqual(why,'fixed_clock_or_metric_absent')
    def test_zero_denominator(self):
        row=self.motion_values(self.fixture(),1,0)
        m,_=p.motion_panel([row],5,'position_distance','R0_R1',self.config)
        self.assertIsNone(m['point']['ratio_of_task_means']);self.assertIsNone(m['ci95']['ratio_of_task_means'])
    def test_ratio_of_means_not_mean_ratios(self):
        a=self.motion_values(self.fixture(),2,1);b=self.motion_values(self.fixture(),10,10)
        b['task_id']='test:other';b['variant_id']='synthetic-test-only-b'
        m,_=p.motion_panel([a,b],5,'position_distance','R0_R1',self.config)
        self.assertAlmostEqual(m['point']['ratio_of_task_means'],12/11);self.assertNotAlmostEqual(m['point']['ratio_of_task_means'],1.5)
    def test_noise_variant_task_hierarchy(self):
        a=self.motion_values(self.fixture(),1,1);b=self.motion_values(self.fixture(),3,1);c=self.motion_values(self.fixture(),10,1)
        b['variant_id']='synthetic-test-only-b';c['variant_id']='synthetic-test-only-c';c['task_id']='test:other'
        m,_=p.motion_panel([a,b,c],5,'position_distance','R0_R1',self.config)
        self.assertEqual(m['point']['TR_mean'],6)  # task A mean2; task B mean10
    def test_three_support_denominators(self):
        rows=[self.motion_values(self.fixture(),1,1) for _ in range(4)]
        for i,r in enumerate(rows):r['variant_id']='synthetic-test-only-'+str(i);r['task_id']='test:'+('a' if i<2 else 'bc'[i-2])
        rows[0]['outcomes']['T0']=None
        for i in (0,1):rows[i]['motion']['TR1']['5']['position_distance'].update(value=None,status='unavailable',reason='test-only')
        outcome=p.outcomes_panel(rows,['a','b','c'],self.config);motion,_=p.motion_panel(rows,5,'position_distance','R0_R1',self.config)
        self.assertEqual(outcome['planned_variants'],4);self.assertEqual(outcome['planned_TR_pairs'],8);self.assertEqual(outcome['raw_pair_counts']['TR']['unknown'],1)
        self.assertIsNone(outcome['metrics']['risk_difference']['point']);self.assertIsNone(outcome['metrics']['risk_difference']['ci95'])
        self.assertEqual(motion['joint_variants'],2);self.assertEqual(motion['joint_tasks'],2)
    def test_same_repeat_unknown_is_not_zero(self):
        row=self.fixture();row['outcomes']['R0prime']=None
        self.assertIsNone(p.variant_bounds(row['outcomes'])['rr_repeat_flip']['point'])
        v,why=p.read_motion(row,'R0_R0prime',5,'position_distance');self.assertIsNone(v);self.assertIn('repeat_eligibility',why)
    def test_bootstrap_multiplicity_preserved(self):
        ds=p.bootstrap_indices(['a','b'],10,1);self.assertTrue(any(d==['a','a'] or d==['b','b'] for d in ds))
    def test_quantile_noninterpolated(self):
        self.assertEqual(p.interval(list(range(2000)),.05,2000),[50,1949])
        self.assertEqual(p.interval(list(range(2000)),.0125,2000),[12,1986])
    def test_undefined_draw_keeps_ci_null(self): self.assertIsNone(p.interval([None]+[1]*1999,.05,2000))
if __name__=='__main__': unittest.main()
