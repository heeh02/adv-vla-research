"""Synthetic physical states only; no recorded array or target runtime imports."""
import copy
from pathlib import Path
import tempfile
import unittest
import numpy as np
import readout as R
from render import render


def fixture():
    by={}
    for arm,factor in [('N',0.),('S',1.),('R',2.)]:
        steps=[]
        for i in range(5):
            angle=factor*(i+1)*1e-5
            steps.append({'before':{'sim_time':np.asarray(3+i*.05)},
                'after':{'sim_time':np.asarray(3+(i+1)*.05)},'done':False,'official_success':False,
                'physical_outputs':{'returned_nonimage_observations':{
                    'robot0_eef_pos':np.array([factor*(i+1)*1e-6,0.,0.]),
                    'robot0_eef_quat':np.array([0.,0.,np.sin(angle/2),np.cos(angle/2)]),
                    'robot0_gripper_qpos':np.array([-factor*i*1e-6,factor*i*1e-7])},
                    'gripper_actuator_ctrl':np.array([-1.,1.]),'contact_count':27+int(arm=='R'and i==4)}})
        by[arm]={'trajectory':steps}
    return by


class Readout(unittest.TestCase):
    def test_all15_rows_signed_S_R_and_equal_target_not_equal_state(self):
        rows=R.compute(fixture())
        self.assertEqual(len(rows),15)
        self.assertTrue(all(r['gripper_actuator_target_equal']for r in rows))
        last=rows[-1]
        self.assertEqual(last['contrast'],'S-R')
        self.assertAlmostEqual(last['eef_delta_xyz_m'][0],-5e-6)
        self.assertEqual(last['candidate_contact_count']-last['reference_contact_count'],-1)
        self.assertNotEqual(last['gripper_qpos_delta_original_units'],[0.,0.])

    def test_quaternion_antipodes_have_zero_distance(self):
        self.assertEqual(R.orientation(np.array([0.,0.,0.,1.]),np.array([0.,0.,0.,-1.])),0.)

    def test_original_ten_comparisons_recomputed(self):
        rows=R.compute(fixture());old=[]
        for row in rows[:10]:
            old.append({**row,'eef_position_delta_m':np.array(row['eef_delta_xyz_m']),
                'gripper_qpos_delta':np.array(row['gripper_qpos_delta_original_units']),
                'N_contact_count':row['reference_contact_count'],'arm_contact_count':row['candidate_contact_count']})
        R.compare_original(rows,{'comparisons':old})
        old[0]['eef_position_distance_m']+=1e-12
        with self.assertRaisesRegex(ValueError,'original_scalar_exact'):
            R.compare_original(rows,{'comparisons':old})

    def test_missing_timepoint_cannot_be_filled_or_omitted(self):
        by=fixture();by['S']['trajectory'].pop()
        with self.assertRaisesRegex(ValueError,'all3pairs_all5times'):
            R.compute(by)

    def test_native_time_mismatch_rejected(self):
        by=fixture();by['R']['trajectory'][3]['after']['sim_time']=np.asarray(9.)
        with self.assertRaisesRegex(ValueError,'same_native_clock'):
            R.compute(by)

    def test_synthetic_render_produces_six_files(self):
        with tempfile.TemporaryDirectory()as td:
            render(R.compute(fixture()),td)
            self.assertEqual(len(list(Path(td).iterdir())),6)
            self.assertTrue(all(p.stat().st_size>1000 for p in Path(td).iterdir()))


if __name__=='__main__':unittest.main()
