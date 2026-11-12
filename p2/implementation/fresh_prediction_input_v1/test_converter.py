"""Synthetic arrays only; tests do not load old adapters, model, env or NPZ files."""
import copy
import unittest
from unittest.mock import patch
import numpy as np
import converter as C


def bf(value):
    a=np.asarray(value,dtype='<f4')
    # Fixture values are exactly representable. This is not a payload constructor.
    return C.leaf((a.view('<u4')>>16).astype('<u2'),'bfloat16')


def fixture():
    n=4;mask=np.array([[True,True,True,False]]);pos=np.cumsum(mask,axis=-1,dtype=np.int32)-1
    z=np.zeros((1,n,1,256),np.float32);s=z.copy();s[0,:3,0,0]=1;r=z.copy();r[0,:3,0,4]=1
    basis=np.stack([np.eye(256,4,dtype=np.float64)]*3)
    random=np.stack([np.eye(256,dtype=np.float64)[:,4:8]]*3)
    arrays={'noise':C.leaf(np.zeros((1,10,32),np.float32)),
            'decode_state_C':C.leaf(np.zeros((1,32),np.float32)),
            'decode_state_T':C.leaf(np.zeros((1,32),np.float32)),
            'query_clock':C.leaf(np.array(.55,np.float64)),
            'WO':bf(np.zeros((8,256,1024),np.float32)),
            'prefix_mask_C':C.leaf(mask),'prefix_mask_T':C.leaf(mask),
            'prefix_positions_C':C.leaf(pos),'prefix_positions_T':C.leaf(pos),
            'K_C':bf(z),'K_T':bf(z),'V_C':bf(z),'V_T':bf(z),
            'Vmod_S':bf(s),'Vmod_R':bf(r),'E_S':C.leaf(s),'E_R':C.leaf(r),
            'basis_response':C.leaf(basis),'basis_random':C.leaf(random)}
    rm={'token_target_norm':np.array([1,1,1,0],np.float64),
        'token_actual_norm':np.array([1,1,1,0],np.float64),'token_residual':np.zeros(n,np.float64),
        'token_scale':np.array([1,1,1,0],np.float64),'zero_target':np.zeros(n,bool),
        'zero_direction_nonzero_target':np.zeros(n,bool),'unbracketed':np.zeros(n,bool)}
    arrays.update({'random_match.'+k:C.leaf(v) for k,v in rm.items()})
    binding={k:'a'*64 for k in C.BINDING_FIELDS};binding.update(root_id='synthetic-new-root',query_index=7)
    metadata={'artifact_id':'b'*64,'array_file':{'bytes':1,'sha256':'c'*64},'rank':4,'layers':[0,8,17],
              'layer_index':8,'layer_slot':1,'refit':False,
              'matrix_payload_sha256':{'response':C.sha(basis.tobytes()),'random':C.sha(random.tobytes())}}
    packet={'schema':C.SCHEMA,'synthetic':True,'binding':binding,'basis':metadata,'payload_basis_identity':'b'*64,
            'source_refs':{'input':{'uri':'redacted-2002904698ae','bytes':1,'sha256':'d'*64}},
            'arrays':arrays,'encoded_observations':{'mode':'recorded_identity_only','identities':{'C':'e'*64,'T':'f'*64},'bodies':None}}
    expected={'schema':C.SCHEMA,'synthetic':True,'binding':copy.deepcopy(binding),'basis':copy.deepcopy(metadata),
              'source_refs':copy.deepcopy(packet['source_refs']),'array_identities':{k:C.identity(v) for k,v in arrays.items()},
              'encoded_identities':{'C':'e'*64,'T':'f'*64},'encoding_mode':'recorded_identity_only','encoded_leaf_identities':None}
    return packet,expected


class ConverterTests(unittest.TestCase):
    def test_no_history_and_no_native_claim(self):
        p,e=fixture();r=C.validate_packet(p,e,synthetic=True)
        self.assertTrue(r['proof']['payload_arrays_exact'])
        for k in ['execution_authorized','native_runtime_qualified','fresh_C_AE_qualified','encoded_bodies_verified',
                  'source_files_SHA_verified_by_this_function','operator_reconstructed']:
            self.assertFalse(r['proof'][k])
        self.assertNotIn('old_endpoints',r)
        self.assertTrue(all(not v['storage'].flags.writeable for v in r['arrays'].values()))
        self.assertEqual(C.identity(p['arrays']['Vmod_S']),C.identity(r['arrays']['Vmod_S']))

    def test_BF16_is_bits_not_uint16_numeric(self):
        v=C.leaf(np.array([0x3f80,0xbf80,0x0000],dtype=np.uint16),'bfloat16')
        self.assertTrue(np.array_equal(C.decoded(v),np.array([1,-1,0],np.float32)))
        with self.assertRaises(ValueError):C.decoded(C.leaf(np.array([0x7f80],np.uint16),'bfloat16'))

    def test_real_arrays_default_rejected_on_Mac(self):
        p,e=fixture();p['synthetic']=e['synthetic']=False
        with patch.object(C.sys,'platform','darwin'),self.assertRaisesRegex(ValueError,'real_arrays_cloud_Linux_only'):
            C.validate_packet(p,e)

    def test_hash_mismatch_and_swapped_root_fail(self):
        p,e=fixture();p['binding']['root_id']='other'
        with self.assertRaisesRegex(ValueError,'exact_prebound_parent_model_source'):C.validate_packet(p,e,synthetic=True)
        p,e=fixture();p['arrays']['noise']=C.leaf(np.ones((1,10,32),np.float32))
        with self.assertRaisesRegex(ValueError,'every_array_matches'):C.validate_packet(p,e,synthetic=True)

    def test_actual_E_must_equal_actual_BF16_difference(self):
        p,e=fixture();p['arrays']['E_S']=C.leaf(np.zeros((1,4,1,256),np.float32));e['array_identities']['E_S']=C.identity(p['arrays']['E_S'])
        with self.assertRaisesRegex(ValueError,'actual_E_exact_BF16_payload_difference'):C.validate_packet(p,e,synthetic=True)

    def test_padding_unchanged_even_if_E_consistent(self):
        p,e=fixture();a=C.decoded(p['arrays']['Vmod_S']).copy();a[0,3,0,0]=1
        p['arrays']['Vmod_S']=bf(a);p['arrays']['E_S']=C.leaf(a)
        for k in ['Vmod_S','E_S']:e['array_identities'][k]=C.identity(p['arrays'][k])
        with self.assertRaisesRegex(ValueError,'padding_unchanged'):C.validate_packet(p,e,synthetic=True)

    def test_basis_rank_or_matrix_change_not_permitted(self):
        p,e=fixture();p['basis']['rank']=e['basis']['rank']=16
        with self.assertRaisesRegex(ValueError,'original_L9_rank4'):C.validate_packet(p,e,synthetic=True)
        p,e=fixture();a=C.decoded(p['arrays']['basis_response']).copy();a[1,0,0]=.5
        p['arrays']['basis_response']=C.leaf(a);e['array_identities']['basis_response']=C.identity(p['arrays']['basis_response'])
        with self.assertRaisesRegex(ValueError,'basis_matrix_payload'):C.validate_packet(p,e,synthetic=True)

    def test_unequal_actual_norms_are_preserved_not_excluded(self):
        p,e=fixture();a=C.decoded(p['arrays']['Vmod_R']).copy()*2
        p['arrays']['Vmod_R']=bf(a);p['arrays']['E_R']=C.leaf(a)
        p['arrays']['random_match.token_actual_norm']=C.leaf(np.array([2,2,2,0],np.float64))
        p['arrays']['random_match.token_residual']=C.leaf(np.array([1,1,1,0],np.float64))
        p['arrays']['random_match.unbracketed']=C.leaf(np.array([True,False,False,False]))
        e['array_identities']={k:C.identity(v) for k,v in p['arrays'].items()}
        r=C.validate_packet(p,e,synthetic=True)['proof']
        self.assertFalse(r['exact_actual_norm_match']);self.assertEqual(r['random_norm_residual'],[1,1,1,0])
        self.assertTrue(r['random_match_flags']['unbracketed'][0])

    def test_missing_fields_and_old_effect_injection_rejected(self):
        p,e=fixture();del p['arrays']['K_T']
        with self.assertRaisesRegex(ValueError,'complete_label_free'):C.validate_packet(p,e,synthetic=True)
        p,e=fixture();p['arrays']['old_effect_shared']=C.leaf(np.zeros((1,10,7),np.float64))
        with self.assertRaisesRegex(ValueError,'complete_label_free'):C.validate_packet(p,e,synthetic=True)

    def test_typed_encoded_bodies_checked_separately(self):
        p,e=fixture();bodies={s:{'state':C.leaf(np.zeros((1,32),np.float32))} for s in ['C','T']}
        p['encoded_observations'].update(mode='typed_body',bodies=bodies);e['encoding_mode']='typed_body'
        e['encoded_leaf_identities']={s:{k:C.identity(v) for k,v in body.items()} for s,body in bodies.items()}
        self.assertTrue(C.validate_packet(p,e,synthetic=True)['proof']['encoded_bodies_verified'])

    def test_full_equivalence_reports_mismatch_not_selected_subset(self):
        p,e=fixture();out=C.validate_packet(p,e,synthetic=True)['arrays']
        self.assertTrue(C.compare_legacy_projection(out,p['arrays'])['all_fields_exact'])
        reference=copy.deepcopy(p['arrays']);reference['noise']=C.leaf(np.ones((1,10,32),np.float32))
        self.assertFalse(C.compare_legacy_projection(out,reference)['all_fields_exact'])
        del reference['noise']
        with self.assertRaisesRegex(ValueError,'complete_equivalence_field_roster'):C.compare_legacy_projection(out,reference)


if __name__=='__main__':unittest.main()
