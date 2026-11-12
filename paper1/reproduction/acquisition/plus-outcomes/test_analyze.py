import unittest
import analyze as A

class OutcomeTests(unittest.TestCase):
    def test_paired_cell_direction_and_hierarchical_weights(self):
        self.assertEqual(A.point('SF')['risk_difference'],-1)
        self.assertEqual(A.point('SF')['harm'],1)
        self.assertEqual(A.point('FS')['risk_difference'],1)
        self.assertEqual(A.point('FS')['recovery'],1)
        # Two variants in the first family, one in the second: the task
        # average gives each family half weight, not each variant one third.
        sf=A.point('SF');ss=A.point('SS')
        first=A.vector_mean([sf,sf]);second=A.vector_mean([ss])
        task=A.vector_mean([first,second])
        self.assertEqual(task['risk_difference'],-.5)
        self.assertEqual(task['harm'],.5)

    def test_exact_536_source_join_and_task_bootstrap(self):
        cases,_=A.collect()
        self.assertEqual(len({r['case_id']for r in cases}),536)
        self.assertEqual(sum(r['reference_policy_steps']!=''for r in cases),508)
        self.assertEqual(sum(r['treatment_policy_steps']!=''for r in cases),0)
        result=A.analyze(cases)
        self.assertEqual(len(result['task_rows']),40)
        self.assertEqual(sum(result['raw_four_cells'].values()),536)
        self.assertEqual(result['raw_four_cells']['SF'],56)
        self.assertEqual(result['raw_four_cells']['FS'],20)
        self.assertAlmostEqual(result['overall']['risk_difference'],
                               result['overall']['recovery']-result['overall']['harm'])
        for row in result['family_rows']:
            lo,hi=row['risk_difference_pointwise_95_lo'],row['risk_difference_pointwise_95_hi']
            blo,bhi=row['risk_difference_bonferroni_95_lo'],row['risk_difference_bonferroni_95_hi']
            self.assertLessEqual(blo,lo);self.assertGreaterEqual(bhi,hi)

if __name__=='__main__':unittest.main()
