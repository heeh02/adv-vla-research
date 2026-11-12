import unittest
from collections import Counter
import analyze as A
import analyze_v3 as V

class BandAndSupportTests(unittest.TestCase):
    def test_frozen_band_and_light_null_partition(self):
        cases,_=A.collect();chosen=V.selected_bands()
        for row in cases:row['band']=chosen[row['row_id']]['band']
        counts=Counter((r['family'],r['band'])for r in cases)
        self.assertEqual(counts[('Light Conditions','unlabeled')],18)
        self.assertEqual(counts[('Light Conditions','lower_label')]+
                         counts[('Light Conditions','higher_label')],134)
        self.assertEqual(sum(counts.values()),536)

    def test_common_tasks_are_same_ids_not_marginal_rank(self):
        cases,_=A.collect();chosen=V.selected_bands()
        for row in cases:row['band']=chosen[row['row_id']]['band']
        result=V.summarize(cases,V.group(cases))
        common=result['common_tasks']
        self.assertEqual(len(common),28)
        self.assertEqual(len(result['common_task_rows']),28*4)
        self.assertEqual({r['task']for r in result['common_task_rows']},set(common))
        self.assertEqual(len(result['common_pair_rows']),6)
        light=next(r for r in result['labelled_rows']if r['family']=='Light Conditions')
        self.assertEqual(light['cases'],134)
        self.assertEqual(light['tasks'],40)
        for row in result['common_rows']:
            self.assertEqual(row['tasks'],28)
            self.assertLessEqual(row['risk_difference_bonferroni_95_lo'],
                                 row['risk_difference_pointwise_95_lo'])
            self.assertGreaterEqual(row['risk_difference_bonferroni_95_hi'],
                                    row['risk_difference_pointwise_95_hi'])

if __name__=='__main__':unittest.main()
