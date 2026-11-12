from pathlib import Path
import copy, importlib.util, json, sys, unittest
import numpy as np
P=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('signed_driver_test',P/'adapter.py');D=importlib.util.module_from_spec(spec);spec.loader.exec_module(D)
class Tests(unittest.TestCase):
 def fixture(self):
  ref=json.loads((P/'reference/legacy_context_numbers.json').read_bytes())
  actual={'old_fit_identity':ref['old_fit_identity'],'fit_identity':ref['fit_identity'],'complete':True}
  for field,source in [('equal_task_scores','equal_task_cells'),('root_scores','all_task_cells')]:
   rows=[]
   for r in ref[source]:
    base={k:v for k,v in r.items()if k in ('root_id','task_id','parent_condition','K_background','window','dimension')}
    rows.extend({**base,'method':method,**values}for method,values in r['values'].items())
   actual[field]=rows
  return actual,ref
 def test_all_legacy_cells_pass_and_numerical_failure_is_hard(self):
  actual,ref=self.fixture();self.assertEqual(D.legacy_check(actual,ref)['comparisons'],1680)
  bad=copy.deepcopy(actual);bad['root_scores'][0]['DEV_scaled_MSE']+=1e-4
  with self.assertRaisesRegex(RuntimeError,'legacy_actual_scores_reproduced'):D.legacy_check(bad,ref)
 def test_missing_extra_duplicate_legacy_keys_rejected(self):
  actual,ref=self.fixture()
  for mutation in ('missing','extra','duplicate'):
   bad=copy.deepcopy(actual)
   if mutation=='missing':bad['root_scores'].pop()
   elif mutation=='extra':
    row=copy.deepcopy(bad['root_scores'][0]);row['root_id']='unexpected';bad['root_scores'].append(row)
   else:bad['root_scores'].append(copy.deepcopy(bad['root_scores'][0]))
   with self.assertRaises(RuntimeError):D.legacy_check(bad,ref)
 def test_json_zero_denominator_and_infinity_not_hidden(self):
  self.assertEqual(D.serial({'ratio':np.array([np.nan,1.])}),{'ratio':[None,1.]})
  with self.assertRaises(RuntimeError):D.serial(float('inf'))
if __name__=='__main__':unittest.main()
