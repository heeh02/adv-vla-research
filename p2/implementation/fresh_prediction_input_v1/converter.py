"""Pure NumPy qualification of supplied native payloads; no loader or constructor.

Real inputs are Linux-cloud-only. This module never opens source arrays, imports
the old worker/native bridge, constructs edits, calls a model, or creates an env.
Its new hash domain must not be relabelled as an old Query fingerprint.
"""
import hashlib
import json
import re
import sys
import numpy as np

SCHEMA = 'paper2-fresh-prediction-input-packet-v1'
BASIS_ID = '0c9e72b35d04f4ec222a67cee8c1fd282a0b4adcb065e0a9c8a0d33941634d1f'
BASIS_FILE = {'bytes': 6615126, 'sha256': '466bb97418cf4a9f03d7f5abbc0700f2542fd14cee598d738e7bae6aba7824bc'}
BASIS_HASHES = {'response': 'cec22806b5318dee9502512d9a91853a2787c44012e20e0782ef7efe0a297f6c',
                'random': '1a96a2b396eeb7a04d8542f09063aa512531e5c77267d923f3fd3c5c0f5d6a61'}
CORE = {'noise', 'decode_state_C', 'decode_state_T', 'query_clock', 'WO',
        'prefix_mask_C', 'prefix_mask_T', 'prefix_positions_C', 'prefix_positions_T',
        'K_C', 'K_T', 'V_C', 'V_T', 'Vmod_S', 'Vmod_R', 'E_S', 'E_R',
        'basis_response', 'basis_random'}
RM_FLOAT = {'token_target_norm', 'token_actual_norm', 'token_residual', 'token_scale'}
RM_BOOL = {'zero_target', 'zero_direction_nonzero_target', 'unbracketed'}
REQUIRED = CORE | {'random_match.'+k for k in RM_FLOAT | RM_BOOL}
BINDING_FIELDS = {'root_id', 'query_index', 'parent_source_identity', 'prompt_sha256',
                  'model_binding_sha256', 'normalization_sha256', 'native_graph_source_sha256',
                  'initialization_vector_sha256', 'noise_policy_sha256', 'camera_policy_sha256'}


def need(ok, code):
    if not ok:
        raise ValueError(code)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def leaf(array, logical_dtype=None):
    """Copy typed payload bytes. BF16 uint16 storage needs an explicit label."""
    a = np.asarray(array)
    logical = logical_dtype or a.dtype.name
    if a.dtype.name == 'bfloat16':
        a = a.view(np.uint16)
    a = np.array(a, copy=True, order='C')
    a.setflags(write=False)
    return {'logical_dtype': logical, 'storage': a}


def decoded(value):
    need(type(value) is dict and set(value) == {'logical_dtype', 'storage'}, 'explicit_typed_leaf')
    a = np.asarray(value['storage'])
    need(not a.dtype.hasobject and a.size > 0, 'nonempty_nonobject_array')
    if value['logical_dtype'] == 'bfloat16':
        need(a.dtype == np.dtype('<u2'), 'BF16_requires_explicit_uint16_payload_storage')
        out = (a.astype('<u4') << np.uint32(16)).view('<f4')
    else:
        need(value['logical_dtype'] == a.dtype.name and a.dtype.kind in 'biuf', 'logical_native_dtype')
        out = a
    need(np.isfinite(out).all(), 'finite_native_payload')
    return out


def identity(value):
    decoded(value)
    a = np.asarray(value['storage'])
    return {'shape': list(a.shape), 'logical_dtype': value['logical_dtype'],
            'storage_dtype': a.dtype.str, 'bytes': a.nbytes, 'payload_sha256': sha(a.tobytes(order='C'))}


def same(a, b):
    return identity(a) == identity(b)


def _array(arrays, name, shape=None, dtype=None):
    v = arrays[name]
    a = decoded(v)
    need(shape is None or a.shape == tuple(shape), 'shape_'+name)
    need(dtype is None or v['logical_dtype'] == dtype, 'dtype_'+name)
    return a


def validate_packet(packet, expected, *, synthetic=False):
    """Validate exact prebound in-memory inputs, returning a label-free view.

    `expected` must be independently frozen by the caller, after file SHA and
    source joins. This pure function does not claim to revalidate source files.
    Real basis payload SHA values are fixed in this module, never caller-chosen.
    """
    need(type(synthetic) is bool and (synthetic or sys.platform.startswith('linux')),
         'real_arrays_cloud_Linux_only')
    need(packet.get('schema') == expected.get('schema') == SCHEMA, 'packet_schema')
    need(packet.get('synthetic') is synthetic and expected.get('synthetic') is synthetic,
         'real_synthetic_scope')
    binding = packet['binding']
    need(set(binding) == BINDING_FIELDS and binding == expected['binding'], 'exact_prebound_parent_model_source')
    need(type(binding['root_id']) is str and bool(binding['root_id']) and
         type(binding['query_index']) is int and binding['query_index'] >= 0, 'root_query_identity')
    need(all(type(binding[k]) is str and re.fullmatch('[0-9a-f]{64}', binding[k])
             for k in BINDING_FIELDS-{'root_id', 'query_index'}), 'typed_source_identity_digests')
    need(packet['source_refs'] == expected['source_refs'] and bool(packet['source_refs']), 'source_refs_exact')
    need(type(packet['source_refs']) is dict, 'named_source_refs')
    for ref in packet['source_refs'].values():
        need(type(ref) is dict and type(ref.get('bytes')) is int and ref['bytes'] > 0 and
             type(ref.get('sha256')) is str and re.fullmatch('[0-9a-f]{64}',ref['sha256']) and
             type(ref.get('uri')) is str and ref['uri'].startswith('redacted-24fc043a6cae'),
             'exact_source_object_identity')
    need(packet['basis'] == expected['basis'], 'basis_source_exact')
    basis = packet['basis']
    need(basis['rank'] == 4 and basis['layers'] == [0,8,17] and basis['layer_slot'] == 1 and
         basis['layer_index'] == 8 and basis['refit'] is False, 'original_L9_rank4_basis_scope')
    need(packet['payload_basis_identity'] == basis['artifact_id'], 'payload_bound_to_same_basis')
    if not synthetic:
        need(basis['artifact_id'] == BASIS_ID and basis['array_file'] == BASIS_FILE and
             basis['matrix_payload_sha256'] == BASIS_HASHES, 'original_basis_identity_and_matrix_hashes')
    arrays = packet['arrays']
    need(set(arrays) == set(expected['array_identities']) == REQUIRED, 'complete_label_free_array_roster')
    ids = {k: identity(v) for k,v in arrays.items()}
    need(ids == expected['array_identities'], 'every_array_matches_prebound_typed_payload')
    mask = _array(arrays, 'prefix_mask_T', dtype='bool')
    need(mask.ndim == 2 and mask.shape[0] == 1 and bool(mask.any()), 'valid_prefix_support')
    n = mask.shape[1]
    need((synthetic and 1 <= n <= 968) or n == 968, 'native_prefix_capacity')
    positions = _array(arrays, 'prefix_positions_T', (1,n), 'int32')
    need(np.array_equal(positions, np.cumsum(mask, axis=-1, dtype=np.int32)-1), 'prefix_position_rule')
    need(same(arrays['prefix_mask_C'], arrays['prefix_mask_T']) and
         same(arrays['prefix_positions_C'], arrays['prefix_positions_T']), 'paired_C_T_prefix_layout')
    _array(arrays, 'noise', (1,10,32), 'float32')
    for side in ('C','T'):
        _array(arrays, 'decode_state_'+side, (1,32), 'float32')
        for kind in ('K','V'):
            _array(arrays, kind+'_'+side, (1,n,1,256), 'bfloat16')
    need(same(arrays['decode_state_C'],arrays['decode_state_T']), 'C_T_decode_state_equal')
    _array(arrays, 'WO', (8,256,1024), 'bfloat16')
    for name in ('response','random'):
        a = _array(arrays, 'basis_'+name, (3,256,4), 'float64')
        need(sha(a.tobytes()) == basis['matrix_payload_sha256'][name], 'basis_matrix_payload_'+name)
        need(np.allclose(a[1].T@a[1], np.eye(4), rtol=0, atol=1e-10), 'basis_L9_orthonormal_'+name)
    vt = _array(arrays, 'V_T')
    dose = {}
    for arm in ('S','R'):
        vm = _array(arrays, 'Vmod_'+arm, vt.shape, 'bfloat16')
        e = _array(arrays, 'E_'+arm, vt.shape, 'float32')
        actual = np.subtract(vm, vt, dtype=np.float32)
        need(actual.tobytes() == e.tobytes(), 'actual_E_exact_BF16_payload_difference_'+arm)
        need(np.array_equal(arrays['Vmod_'+arm]['storage'][:,~mask[0]], arrays['V_T']['storage'][:,~mask[0]]),
             'padding_unchanged_'+arm)
        token_norm = np.linalg.norm(actual[0,:,0].astype(np.float64), axis=-1)
        dose[arm] = {'token_norm': token_norm.tolist(), 'global_norm': float(np.linalg.norm(actual[0,mask[0],0].astype(np.float64))),
                     'zero_actual_E': not bool(np.any(actual)), 'payload_bitwise_own_null': same(arrays['Vmod_'+arm],arrays['V_T'])}
    rm = {k:_array(arrays,'random_match.'+k,(n,), 'float64' if k in RM_FLOAT else 'bool') for k in RM_FLOAT|RM_BOOL}
    need(np.allclose(rm['token_target_norm'],dose['S']['token_norm'],rtol=1e-12,atol=0) and
         np.allclose(rm['token_actual_norm'],dose['R']['token_norm'],rtol=1e-12,atol=0), 'original_dose_norms')
    need((rm['token_actual_norm']-rm['token_target_norm']).tobytes()==rm['token_residual'].tobytes(), 'dose_residual_definition')
    need(np.array_equal(rm['zero_target'],mask[0]&(rm['token_target_norm']==0)), 'zero_target_definition')
    encoded = packet['encoded_observations']
    need(set(encoded) == {'mode','identities','bodies'}, 'explicit_encoded_observation_level')
    need(set(encoded['identities'])=={'C','T'} and encoded['identities']==expected['encoded_identities'], 'recorded_encoded_identity_exact')
    for v in encoded['identities'].values():
        need(type(v) is str and re.fullmatch('[0-9a-f]{64}',v), 'encoded_identity_digest')
    mode = encoded['mode']
    need(mode == expected['encoding_mode'] and mode in ('recorded_identity_only','typed_body'), 'encoded_binding_mode')
    if mode == 'typed_body':
        need(set(encoded['bodies']) == {'C','T'}, 'both_encoded_bodies')
        encoded_ids = {s:{k:identity(v) for k,v in body.items()} for s,body in encoded['bodies'].items()}
        need(all(encoded_ids[s] for s in ('C','T')) and encoded_ids == expected['encoded_leaf_identities'], 'all_encoded_leaf_identities')
    else:
        need(encoded['bodies'] is None and expected.get('encoded_leaf_identities') is None, 'identity_only_not_body_validation')
    out = {k:leaf(v['storage'],v['logical_dtype']) for k,v in arrays.items()}
    proof = {'schema':'paper2-fresh-prediction-payload-qualification-v1', 'synthetic':synthetic,
             'payload_arrays_exact':True, 'basis_matrices_exact':True, 'operator_reconstructed':False,
             'old_natural_C_T_AE_or_persistent_endpoints_consumed':False,
             'encoded_binding_mode':mode, 'encoded_bodies_verified':mode=='typed_body',
             'source_files_SHA_verified_by_this_function':False,
             'caller_source_file_SHA_and_semantic_join_required':True,
             'native_runtime_qualified':False, 'fresh_C_AE_qualified':False,
             'prospective_independence_qualified':False, 'execution_authorized':False,
             'new_model_calls':0, 'new_environment_steps':0, 'new_fits':0,
             'binding':binding, 'array_identities':ids, 'basis':basis, 'dose':dose,
             'random_match_flags':{k:rm[k].tolist() for k in RM_BOOL},
             'random_norm_residual':rm['token_residual'].tolist(),
             'exact_actual_norm_match':bool(np.all(rm['token_residual'][mask[0]]==0))}
    proof['new_packet_identity'] = sha(canonical({'schema':SCHEMA,'binding':binding,'arrays':ids,'basis':basis,
                                                 'encoded_identities':encoded['identities'],'source_refs':packet['source_refs']}))
    return {'arrays':out, 'proof':proof}


def compare_legacy_projection(converted, legacy_projection):
    """Compare every declared field, never selecting a passing subset or label."""
    need(set(converted) == set(legacy_projection) == REQUIRED, 'complete_equivalence_field_roster')
    rows = {k:{'exact':same(converted[k],legacy_projection[k]),'new':identity(converted[k]),
               'reference':identity(legacy_projection[k])} for k in sorted(REQUIRED)}
    return {'schema':'paper2-legacy-free-payload-equivalence-v1', 'all_fields_exact':all(r['exact'] for r in rows.values()),
            'fields':rows, 'native_runtime_qualified':False, 'prediction_validated':False,
            'execution_authorized':False}
