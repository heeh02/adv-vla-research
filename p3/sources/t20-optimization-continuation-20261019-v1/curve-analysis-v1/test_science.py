"""Tiny artificial fixtures only, never actual log/probe statistics on Mac."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import science as S

def probe(step=20000,offset=0.):
    rows=[]
    for task in range(40):
        for role in ('C_old','T_added'):
            for index in range(10):
                for view in ('online','EMA'):
                    row=dict(physical_anchor_id=f't{task}-{role}-{index}',task_id=f't{task}',
                        source_demo_id=f'd{task}-{index//2}',source_j=index,parameter_view=view,role=role,
                        key_seed=task*20+index,draw=0,flow_time=.5,condition='C0_Hbase',split='train',effective_update=step)
                    for h in (5,10):
                        row[f'H{h}_D7']=.02+offset;row[f'H{h}_D25']=.004+offset
                        row[f'H{h}_D32']=(7*row[f'H{h}_D7']+25*row[f'H{h}_D25'])/32
                    rows.append(row)
    return dict(schema='t20-fixed-train-probe-scores-v1',effective_update=step,is_success_rate=False,
                seed=0,branch='floor',planned_scores=1600,completed_scores=1600,missing=[],records=rows)

class Tests(unittest.TestCase):
    def setUp(self):self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
    def tearDown(self):self.tmp.cleanup()
    def test_actual_update_identity_preupdate_loss_step_and32rows(self):
        p=self.root/'updates.jsonl'
        rows=[dict(effective_update=s,actual_stock_step=s,epoch=100,row_ids=['artificial']*32,
            stock_info={'loss':.01,'grad_norm':.05},LR_before_update=2.5e-6,blocked_update_seconds=1.7) for s in (20001,20002)]
        p.write_text(''.join(json.dumps(r)+'\n' for r in rows))
        item=dict(local_path=str(p),run_label='floor-s0',branch='floor',seed=0)
        result=S.read_updates(item)
        self.assertEqual(result[0]['loss_parameter_step'],20000)
        rows[-1]['actual_stock_step']=20003;p.write_text(''.join(json.dumps(r)+'\n' for r in rows))
        with self.assertRaises(ValueError):S.read_updates(item)
    def test_partial_last_line_and_gap_rejected(self):
        p=self.root/'updates.jsonl';p.write_text('{}')
        with self.assertRaisesRegex(ValueError,'updates_incomplete_line'):
            S.read_updates(dict(local_path=str(p),run_label='floor-s0',branch='floor',seed=0))
    def test_rolling_fixed_window_and_dispersion(self):
        import numpy as np
        mid,lo,hi=S.rolling([1.,2.,3.,4.,100.],3)
        self.assertTrue(np.isnan(mid[:2]).all())
        self.assertEqual(mid[2:].tolist(),[2.,3.,4.])
        self.assertEqual(lo[-1],3.5);self.assertEqual(hi[-1],52.)
    def test_probe_complete_support_and_D32_identity(self):
        p=self.root/'probe.json';value=probe();p.write_text(json.dumps(value))
        S.probe_record(dict(local_path=str(p),seed=0,branch='floor'))
        value['records'][0]['H10_D32']=10;p.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError,'D32_weighted_identity'):S.probe_record(dict(local_path=str(p),seed=0,branch='floor'))
    def test_paired_probe_changes_report_all400states40tasks(self):
        probes={('floor-s0',20000):probe(),('floor-s0',30000):probe(30000,-.001)}
        rows,tasks,changes=S.probe_summaries(probes)
        self.assertEqual(len(rows),48);self.assertEqual(len(changes),24)
        self.assertTrue(all(r['paired_states']==400 and r['paired_tasks']==40 and r['tasks_improved']==40 for r in changes))
    def test_fixed_random_key_mismatch_fails_pairing(self):
        probes={('floor-s0',20000):probe(),('floor-s0',30000):probe(30000)}
        probes[('floor-s0',30000)]['records'][0]['key_seed']+=1
        with self.assertRaisesRegex(ValueError,'probe_paired_identity:key_seed'):S.probe_summaries(probes)
    def test_branch_comparison_matches_seed_population(self):
        a=probe();b=probe(offset=-.001)
        rows=S.branch_contrasts({('floor-s0',20000):a,('reheat-s0',20000):b})
        self.assertEqual(len(rows),24);self.assertTrue(all(r['tasks_reheat_lower']==40 for r in rows))
        b['records'][0]['physical_anchor_id']='other'
        with self.assertRaisesRegex(ValueError,'branch_matched_probe_population'):S.branch_contrasts({('floor-s0',20000):a,('reheat-s0',20000):b})
    def test_real_analyze_rejects_Mac_before_input_reads(self):
        with mock.patch.object(S.platform,'system',return_value='Darwin'):
            with self.assertRaisesRegex(ValueError,'real_statistics_and_rendering_Baidu_CPU_only'):S.analyze([],{},self.root)

if __name__=='__main__':unittest.main(verbosity=2)
