import unittest
from replay import quantile,unique_groups,bootstrap_means,compare

class ReplayTests(unittest.TestCase):
    def test_linear_quantile(self):
        self.assertEqual(quantile([5.,1.,3.],.25),2.)
        self.assertEqual(quantile([7.,7.],.975),7.)
    def test_duplicate_task_rejected(self):
        r={'family':'f','task_component':'t'}
        with self.assertRaisesRegex(ValueError,'duplicate task'):
            unique_groups([r,r],('family',))
    def test_invalid_draw_rejected(self):
        with self.assertRaisesRegex(ValueError,'bootstrap indices'):
            bootstrap_means([1.,2.],[[0,2]]*2000)
    def test_modified_result_rejected(self):
        with self.assertRaisesRegex(ValueError,'numerical mismatch'):
            compare({'mse':.2},{'mse':.1})
    def test_signed_and_zero_values_preserved(self):
        self.assertEqual(bootstrap_means([-2.,2.],[[0,1]]*2000),[0.]*2000)

if __name__=='__main__':unittest.main()
