import copy
import unittest
from unittest import mock
import numpy as np
import audit as A
import adapter
import synthetic_fixture as F


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.bundle=F.make()

    def test_all_cache_scores_fit_and_merged_groups(self):
        result=A.verify(*self.bundle,synthetic=True)
        self.assertTrue(result['qa']['passed'])
        self.assertEqual(len(result['qa']['merged_task_root_pairs']),24)
        self.assertFalse(result['qa']['legacy_1680_independently_reexecuted'])
        self.assertGreater(result['qa']['checked_numeric_cells'],100000)

    def test_score_matches_independent_scalar_oracle(self):
        rng=np.random.default_rng(1);pred=rng.normal(size=(3,4,10,7));actual=rng.normal(size=(4,10,7));sigma=rng.uniform(.5,2,size=(10,7))
        got,target=A.score(pred,actual,sigma)
        for m in range(3):
            for i in range(4):
                for w,end in enumerate((5,10)):
                    for s in range(2):
                        for d in range(7):
                            expected=sum(((float(pred[m,i,p,d])-float(actual[i,p,d]))/(float(sigma[p,d])if s else 1))**2 for p in range(end))/end
                            self.assertAlmostEqual(got[m,i,w,s,d],expected,places=12)

    def test_zero_targets_remain_undefined_and_signed_errors_preserved(self):
        verified=A.verify(*self.bundle,synthetic=True)
        for group in verified['groups']:
            self.assertTrue(np.isnan(group['relative_to_zero'][:,:,:,6]).all())
        deltas=np.concatenate([p['MSE_reduction'].ravel()for p in verified['pairs']])
        self.assertTrue(np.any(deltas<0)and np.any(deltas>0))
        self.assertIsNone(adapter.serial(float('nan')))

    def test_corrupt_prediction_rejected(self):
        bundle=copy.deepcopy(self.bundle);bundle[2]['current_full@edf_match'][0,0,0]+=1
        with self.assertRaisesRegex(ValueError,'query_MSE'):A.verify(*bundle,synthetic=True)

    def test_corrupt_sigma_hash_and_EDF_rejected(self):
        bundle=copy.deepcopy(self.bundle);bundle[3]['target_scale'][0]*=2
        with self.assertRaisesRegex(ValueError,'target_scale'):A.verify(*bundle,synthetic=True)
        bundle=copy.deepcopy(self.bundle);bundle[3]['current_full_design_s'][0]*=2
        with self.assertRaisesRegex(ValueError,'EDF'):A.verify(*bundle,synthetic=True)

    def test_missing_group_wrong_key_and_bad_task_count_rejected(self):
        bundle=copy.deepcopy(self.bundle);bundle[1]['groups']=bundle[1]['groups'][:-1]
        with self.assertRaisesRegex(ValueError,'group_roster'):A.verify(*bundle,synthetic=True)
        bundle=copy.deepcopy(self.bundle);bundle[1]['keys'][0],bundle[1]['keys'][1]=bundle[1]['keys'][1],bundle[1]['keys'][0]
        with self.assertRaisesRegex(ValueError,'key_order'):A.verify(*bundle,synthetic=True)
        bundle=copy.deepcopy(self.bundle);bundle[1]['paired_comparisons'][0]['positive_task_count'][0,0,0]+=1
        with self.assertRaisesRegex(ValueError,'positive_task_count'):A.verify(*bundle,synthetic=True)

    def test_no_actual_array_loading_on_Mac_or_synthetic_as_actual(self):
        with mock.patch.object(adapter.sys,'platform','darwin'):
            with self.assertRaisesRegex(ValueError,'cloud_Linux'):
                adapter.run('/no-input','/no-output','0'*64,0)
        with self.assertRaisesRegex(ValueError,'qualified_producer'):
            A.verify(*self.bundle,synthetic=False)


if __name__=='__main__':unittest.main()
