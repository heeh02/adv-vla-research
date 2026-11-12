"""Synthetic technical checks; these fixtures are never scientific samples."""
import copy, hashlib, json, math, unittest
import run_e as m

class ETests(unittest.TestCase):
    def fixture(self):
        return [dict(model='pi0_libero',variant_id='v',noise='0',step=s,task_id='t',family='f',
                     Y='SF',q0_norm='1',eef_mm='2',included='True',q0_status='qualified',motion_status='measured')
                for s in ('5','60')]

    def test_midcdf_exact_ties(self):
        for actual,expected in zip(m.weighted_midcdf([1,1,3],[.1,.2,.7]),[.15,.15,.65]):
            self.assertAlmostEqual(actual,expected)

    def test_midcdf_zero_mass_excluded(self):
        self.assertEqual(m.weighted_midcdf([1,2,3],[.5,0,.5]),[.25,None,.75])

    def test_rho_monotone_and_inverse(self):
        self.assertAlmostEqual(m.weighted_rank_rho([1,2,3],[1,2,3],[.1,.2,.7])[0],1)
        self.assertAlmostEqual(m.weighted_rank_rho([1,2,3],[3,2,1],[.1,.2,.7])[0],-1)

    def test_zero_variance_undefined(self):
        self.assertEqual(m.weighted_rank_rho([1,1],[2,3],[.5,.5]),(None,'zero_global_rank_variance'))

    def test_nested_weights_unbalanced_and_singleton_task(self):
        rows=[dict(task_id='t0',variant_id='v0'),dict(task_id='t0',variant_id='v0'),
              dict(task_id='t0',variant_id='v1'),dict(task_id='t1',variant_id='v2')]
        self.assertEqual(m.nested_weights(rows),[.125,.125,.25,.5])
        self.assertEqual(m.nested_weights(rows,['t0','t0']),[.25,.25,.5,0])

    def test_bootstrap_ranks_recomputed(self):
        original=m.weighted_midcdf([1,2,3],[.2,.3,.5])
        resampled=m.weighted_midcdf([1,2,3],[0,.3,.7])
        self.assertNotEqual(original[1],resampled[1])
        self.assertEqual(resampled[1],.15)

    def test_duplicate_key_stops(self):
        rows=self.fixture();rows.append(copy.deepcopy(rows[0]))
        with self.assertRaisesRegex(ValueError,'duplicate_motion_key'):m.prepare_motion(rows)

    def test_q0_conflict_stops(self):
        rows=self.fixture();rows[1]['q0_norm']='2'
        with self.assertRaisesRegex(ValueError,'q0_conflict'):m.prepare_motion(rows)

    def test_identity_conflict_stops(self):
        rows=self.fixture();rows[1]['task_id']='other'
        with self.assertRaisesRegex(ValueError,'identity_conflict'):m.prepare_motion(rows)

    def test_missing_clock_kept_excluded(self):
        ledger,accepted=m.prepare_motion(self.fixture()[:1])
        self.assertEqual(len(ledger),1);self.assertEqual(accepted,[])
        self.assertIn('step60_missing',ledger[0]['exclusion_reasons'])

    def test_outcome_unknown_does_not_exclude(self):
        rows=self.fixture()
        for row in rows:row['Y']=''
        ledger,accepted=m.prepare_motion(rows);self.assertEqual(len(accepted),1)

    def test_negative_distance_stops(self):
        rows=self.fixture();rows[0]['eef_mm']='-1'
        with self.assertRaisesRegex(ValueError,'negative_eef_mm'):m.prepare_motion(rows)

    def test_nonfinite_excluded(self):
        rows=self.fixture();rows[0]['eef_mm']='nan'
        ledger,accepted=m.prepare_motion(rows);self.assertEqual(accepted,[])
        self.assertIn('nonfinite',ledger[0]['exclusion_reasons'])

    def test_seed_draws_repeat_exactly(self):
        a=m.bootstrap_tasks(['a','b','c'],10,20261014)
        b=m.bootstrap_tasks(['a','b','c'],10,20261014)
        self.assertEqual(a,b)
        self.assertEqual(hashlib.sha256(json.dumps(a).encode()).hexdigest(),hashlib.sha256(json.dumps(b).encode()).hexdigest())

    def test_percentile_linear(self):self.assertEqual(m.percentile([0,10],.025),.25)

    def test_s4_ties_half(self):
        s=m.s4_module();cases=[dict(row_id='a',Y_cell='SF',x=2),dict(row_id='b',Y_cell='SS',x=2)]
        self.assertEqual(s.weighted_auc(cases,'x'),.5)

    def test_s4_other_outcomes_affect_noise_weights(self):
        s=m.s4_module()
        cases=[dict(row_id='a',Y_cell='SF',x=3),dict(row_id='a',Y_cell='FF',x=4),
               dict(row_id='b',Y_cell='SF',x=1),dict(row_id='c',Y_cell='SS',x=2)]
        self.assertAlmostEqual(s.weighted_auc(cases,'x'),1/3)
        self.assertAlmostEqual(s.weighted_auc([r for r in cases if r['Y_cell'] in {'SF','SS'}],'x'),.5)

    def test_s4_ineligible_undefined(self):
        self.assertIsNone(m.s4_module().weighted_auc([dict(row_id='a',Y_cell='SS',x=1)],'x'))

    def test_s4_wrong_unit_stops(self):
        row=dict(layer_display='18',generation_call='0',case_id='c',cohort='old_plus536_pi05',joint404_derived='True',
                 family='Camera Viewpoints',p_full_key_tv_status='measured',w_relative_l2_status='measured',
                 ae_relative_l2_status='measured',issued_status='measured',p_full_key_tv='1',w_relative_l2='1',
                 ae_relative_l2='1',issued_translation_first5_delta_rms='1',issued_translation_first5_unit='mm')
        with self.assertRaisesRegex(ValueError,'wrong_issued_unit'):m.e2_selected([row])

if __name__=='__main__':unittest.main(verbosity=2)
