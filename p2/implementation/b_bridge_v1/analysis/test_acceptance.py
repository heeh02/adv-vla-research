"""CPU synthetic gates and signed model-velocity readout, no model calls."""
import copy
import unittest
import numpy as np
import acceptance as A


def fixture():
    bits = lambda shape, value=0: np.full(shape,value,np.uint16)
    trace = {"W_before":bits((10,1024),0x3f80),"W_used":bits((10,1024),0x3f80),"gate":bits((1,1024),0x3f80),
        "residual_input":bits((10,1024)),"after_attention_residual":bits((10,1024),0x3f80),
        "hidden_post_block":bits((10,1024)),"velocity":bits((10,32)),
        "query_post_rope_scaled":bits((10,1,8,256)),"probabilities_native":bits((1,8,10,26)),
        "suffix_key_post_rope":bits((10,1,256)),"suffix_value":bits((10,1,256)),
        "mask":np.ones((10,26),bool),"suffix_positions":np.arange(16,26,dtype=np.int32),
        "adarms_condition":np.zeros((1024,),np.float32),"actual_x":np.zeros((10,32),np.float32),
        "actual_time":np.asarray(.3999999165534973,np.float32),"time_weak_type":np.asarray(False),
        "dt":np.asarray(-.1,np.float32),"prefix_key_used":bits((16,1,256)),"prefix_value_used":bits((16,1,256)),
        "bridge_active":np.asarray(False)}
    identity={**A.IDENTITY,**{key:'a'*64 for key in('snapshot_identity','noise_identity','prefix_identity','model_identity','source_manifest_sha256')}}
    return trace,identity


class Tests(unittest.TestCase):
    def test_active_own_target_is_real_identity_path(self):
        before,identity=fixture();after=copy.deepcopy(before);after['bridge_active']=np.asarray(True)
        proof=A.validate_site(before,after,before['W_before'],actual_identity=identity,reference_identity=identity,active=True)
        self.assertTrue(proof['qualified'])
        self.assertTrue(A.require_exact_fields(before,after,A.OUT_FIELDS)['all_native_bytes_exact'])

    def test_identity_state_prefix_selector_and_target_mismatch_refused(self):
        before,identity=fixture()
        for field in ('actual_time','actual_x','prefix_value_used','W_before'):
            after=copy.deepcopy(before);after[field].flat[0]+=1
            with self.assertRaises(A.AcceptanceError):
                A.validate_site(before,after,before['W_before'],actual_identity=identity,reference_identity=identity,active=False)
        after=copy.deepcopy(before);bad={**identity,'receiver':'A_none'}
        with self.assertRaises(A.AcceptanceError):A.validate_site(before,after,before['W_before'],actual_identity=bad,reference_identity=identity,active=False)
        after['W_used'][0,0]=0x3f81
        with self.assertRaises(A.AcceptanceError):A.validate_site(before,after,before['W_before'],actual_identity=identity,reference_identity=identity,active=False)

    def test_signed_effects_and_padding_kept_without_adapter_claim(self):
        none=np.zeros((10,32),np.uint16);d=none.copy();c=none.copy()
        d[:,0]=0x3f80;c[:,0]=0xbf80
        d[:,7]=0x4000;c[:,7]=0x4000
        out=A.velocity_readout({'none':none,'natural_d':d,'common_c':c})
        translation=next(r for r in out['rows']if r['window']=='full10'and r['group']=='translation_index_0_to_2')
        self.assertEqual(translation['effect_sign_reversals'],10)
        self.assertEqual(translation['effect_cosine'],-1.)
        self.assertEqual(translation['relative_bridge_error_norm'],2.)
        self.assertTrue(out['adapter7_not_computed'])
        self.assertTrue(out['padding_is_not_control'])
        rotation=next(r for r in out['rows']if r['group']=='rotation_index_3_to_5')
        self.assertIsNone(rotation['effect_cosine'])
        self.assertIsNone(rotation['relative_bridge_error_norm'])

    def test_smallest_finite_values_not_filled_zero_and_inf_rejected(self):
        none=np.zeros((10,32),np.uint16);d=none.copy();d[0,0]=1;c=d.copy()
        out=A.velocity_readout({'none':none,'natural_d':d,'common_c':c})
        row=next(r for r in out['rows']if r['window']=='full10'and r['group']=='translation_index_0_to_2')
        self.assertGreater(row['d_effect_energy'],0.)
        self.assertEqual(row['effect_norm_ratio'],1.)
        c[0,0]=0x7f80
        with self.assertRaises(A.AcceptanceError):A.velocity_readout({'none':none,'natural_d':d,'common_c':c})

    def test_actual_single_trace_mapping_and_missing_gate_refused(self):
        site,_=fixture();reads={}
        names={'bridge_W_before':'W_before','bridge_W_used':'W_used','attention_gate':'gate',
            'receiver_residual_input':'residual_input','hidden_after_attention_residual':'after_attention_residual',
            'hidden_post_block':'hidden_post_block'}
        for name in ('query_post_rope_scaled','probabilities_native','suffix_key_post_rope','suffix_value','prefix_key_used','prefix_value_used'):
            names[name]=name
        for name,key in names.items():reads[name]=np.repeat(site[key][None,None],3,axis=0)
        for name in ('mask','suffix_positions','adarms_condition'):reads[name]=site[name][None]
        reads['bridge_active']=np.array([False,False,False])
        metadata={'reads':reads,'capture_layers':np.array([0,8,17],np.int32),'generation_call':np.asarray(6),
            'completed_updates':np.asarray(0),'actual_x':site['actual_x'][None],'actual_time':site['actual_time'],
            'time_weak_type':site['time_weak_type'],'sampling_abi':{'actual_dt':site['dt']},
            'gates':{k:np.asarray(True)for k in A.SINGLE_GATES}}
        mapped=A.extract_site(site['velocity'][None],metadata)
        A.require_exact_fields(site,mapped,A.PRE_SITE+A.OUT_FIELDS)
        metadata['gates'].pop('target_finite')
        with self.assertRaisesRegex(A.AcceptanceError,'actual_single_native_gates'):
            A.extract_site(site['velocity'][None],metadata)


if __name__=='__main__':unittest.main()
