"""One dynamic whole-query Vmod graph. No load, launcher or native qualification.

Reuses the unchanged scan-native suffix helper; never alters Gemma/Block/RMS.
Only L9/j5 operands are retained, plus all actual Euler states/velocities/times.
The added outputs and dynamic Vmod setter define a NEW graph requiring native
own-null, full-cache positive and stock comparisons in its own runtime.
"""
from dataclasses import dataclass
import hashlib,importlib.util,json,sys
from pathlib import Path
from types import MethodType
from typing import Callable

HERE=Path(__file__).resolve().parent
SCAN_SHA='16b145983c11168b133111b43864d8f55ae2b625de87105526e9dc6d9991b5c5'
CONTRACT_SHA='1952b5a4770d6e0fab5290575b09b2016d5987561dec683189b038c2c1e022a5'
def _load(name,path,expected):
    if hashlib.sha256(path.read_bytes()).hexdigest()!=expected:raise ValueError('retained_helper_SHA')
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
C=_load('_shared_response_retained_contract',HERE/'retained_contract.py',CONTRACT_SHA)
require=C.require
_previous=sys.modules.get('contract');sys.modules['contract']=C
try:J=_load('_shared_response_retained_scan',HERE/'retained_scan_candidate.py',SCAN_SHA)
finally:
    if _previous is None:sys.modules.pop('contract',None)
    else:sys.modules['contract']=_previous
LAYER=8
FIRST_CALL=5
FIRST_FIELDS=('query_post_rope_scaled','probabilities_native','attention_output_post_WO_pre_gate','attention_gate',
              'receiver_residual_input','prefix_key_used','prefix_value_used','suffix_key_post_rope','suffix_value')


def validate_selectors(kv_mask,layer_mask,flow_mask,use_vmod):
    """Host-only explicit scope check. No arm/mode enters a JIT static arg."""
    import numpy as np
    arrays=[np.asarray(x)for x in(kv_mask,layer_mask,flow_mask,use_vmod)]
    for a,shape in zip(arrays,((2,),(18,),(10,),()),strict=True):require(a.shape==shape and a.dtype==np.bool_,'dynamic_boolean_selector_ABI')
    kv,layer,flow,vm=arrays
    none=not(kv.any()or layer.any()or flow.any())
    main=np.array_equal(layer,np.arange(18)==LAYER)and any(np.array_equal(flow,f)for f in(np.arange(10)==FIRST_CALL,np.arange(10)>=FIRST_CALL))
    full=not bool(vm)and layer.all()and flow.all()and kv.all()
    require(none or main or full,'only_L9_j5_or_j5to9_or_fullcache_control')
    return {'scope':'no_replace'if none else'fullcache_wiring_control'if full else'L9_response_edit',
            'schedule':'none'if none else'all10'if full else'pulse_j5'if flow.sum()==1 else'persistent_j5_to_j9'}


def _abi(prepared,noise,vmod,kv_mask,layer_mask,flow_mask,use_vmod,num_steps):
    J._prepared_abi(prepared)
    _,jnp,_,_=J._deps()
    for name,value,shape in(('kv',kv_mask,(2,)),('layer',layer_mask,(18,)),('flow',flow_mask,(10,)),('vmod_mode',use_vmod,())):
        require(value.shape==shape and value.dtype==jnp.bool_,'dynamic_boolean_'+name)
    require(vmod.shape==prepared.cache[1][LAYER].shape and vmod.dtype==jnp.bfloat16,'external_Vmod_native_BF16_BN1x256')
    require(noise.shape==(prepared.prefix_mask.shape[0],10,32)and noise.dtype==jnp.float32,'original_explicit_noise_FP32')
    require(jnp.shape(num_steps)==()and jnp.issubdtype(jnp.asarray(num_steps).dtype,jnp.integer),'dynamic_integer_num_steps')


def _bf16_bits_equal(a,b):
    jax,jnp,_,_=J._deps()
    return jnp.array_equal(jax.lax.bitcast_convert_type(a,jnp.uint16),jax.lax.bitcast_convert_type(b,jnp.uint16))


def _selected_cache(prepared,vmod,kv_mask,layer_mask,flow_active,use_vmod):
    """Immutable full cache; Vmod replaces only L9 values at valid prefix keys.

    K level is an independent selector. Vmod never depends on it. All inactive
    addresses retain receiver values; unequal padding in an active Vmod payload
    is explicitly rejected by result gates, not silently qualified by masking.
    """
    _,jnp,_,_=J._deps()
    valid=prepared.prefix_mask[None,:,:,None,None]
    location=layer_mask[:,None,None,None,None]&valid&flow_active
    vsource=prepared.donor_cache[1].at[LAYER].set(jnp.where(use_vmod,vmod,prepared.donor_cache[1][LAYER]))
    k=jnp.where(location&kv_mask[0],prepared.donor_cache[0],prepared.cache[0])
    v=jnp.where(location&kv_mask[1],vsource,prepared.cache[1])
    return k,v


def _empty_first(prepared):
    """Shape-only buffer; no extra suffix/model evaluation for allocation."""
    _,jnp,_,_=J._deps();b,n=prepared.prefix_mask.shape
    z=lambda shape:jnp.zeros(shape,jnp.bfloat16)
    return {'query_post_rope_scaled':z((b,10,1,8,256)),
      'probabilities_native':z((b,1,8,10,n+10)),
      'attention_output_post_WO_pre_gate':z((b,10,1024)),
      'attention_gate':z((b,1,1024)),'receiver_residual_input':z((b,10,1024)),
      'prefix_key_used':z((b,n,1,256)),'prefix_value_used':z((b,n,1,256)),
      'suffix_key_post_rope':z((b,10,1,256)),'suffix_value':z((b,10,1,256)),
      'mask':jnp.zeros((b,10,n+10),jnp.bool_),
      'suffix_positions':jnp.zeros((b,10),prepared.prefix_positions.dtype)}


def _recording_velocity(model,prepared,x,time,cache):
    """Exact scan-native method; recorded P/Q/W are from this actual read."""
    suffix,mask,positions,condition=J._suffix(model,prepared,x,time)
    readout,operands=model.PaliGemma.llm([None,suffix],mask=mask,positions=positions,
        adarms_cond=[None,condition],kv_cache=cache,method='scan_native_suffix')
    velocity=model.action_out_proj(readout[:,-10:])
    # The unchanged helper returns selected layer indices (0,8,17); only L9 is
    # exported for the edited arm. No natural three-layer trace is substituted.
    first={name:operands[name][1]for name in FIRST_FIELDS}
    first.update(mask=mask,suffix_positions=positions)
    return velocity,first


def _integrate(model,prepared,noise,vmod,kv_mask,layer_mask,flow_mask,use_vmod,*,num_steps):
    _abi(prepared,noise,vmod,kv_mask,layer_mask,flow_mask,use_vmod,num_steps)
    jax,jnp,_,_=J._deps();dt=-1.0/num_steps
    xs=jnp.zeros((11,*noise.shape),noise.dtype).at[0].set(noise)
    velocities=jnp.zeros((10,*noise.shape),jnp.bfloat16)
    times=jnp.zeros((11,),jnp.asarray(1.0).dtype).at[0].set(1.0)
    weak_times=jnp.zeros(11,jnp.bool_).at[0].set(True)
    first=_empty_first(prepared)
    def body(carry):
        x,time,index,xs,velocities,times,weak_times,first,reached=carry
        cache=_selected_cache(prepared,vmod,kv_mask,layer_mask,flow_mask[index],use_vmod)
        velocity,operands=_recording_velocity(model,prepared,x,time,cache)
        require(velocity.shape==x.shape and velocity.dtype==jnp.bfloat16,'native_BF16_velocity')
        first=jax.lax.cond(index==FIRST_CALL,lambda:operands,lambda:first)
        next_x=x+dt*velocity
        next_time=time+dt
        return (next_x,next_time,index+1,xs.at[index+1].set(next_x),velocities.at[index].set(velocity),
            times.at[index+1].set(next_time),weak_times.at[index+1].set(bool(getattr(next_time,'weak_type',False))),first,reached|(index==FIRST_CALL))
    def cond(carry):
        _,time,index,*_=carry
        return (time>=-dt/2)&(index<10)
    final,time,index,xs,velocities,times,weak_times,first,reached=jax.lax.while_loop(cond,body,
        (noise,1.0,jnp.asarray(0,jnp.int32),xs,velocities,times,weak_times,first,jnp.asarray(False)))
    valid=prepared.prefix_mask[...,None,None]
    ignored_values_equal=jnp.all(jnp.where(valid,True,jax.lax.bitcast_convert_type(vmod,jnp.uint16)==jax.lax.bitcast_convert_type(prepared.cache[1][LAYER],jnp.uint16)))
    main_layer=jnp.array_equal(layer_mask,jnp.arange(18)==LAYER)
    main_flow=jnp.array_equal(flow_mask,jnp.arange(10)==FIRST_CALL)|jnp.array_equal(flow_mask,jnp.arange(10)>=FIRST_CALL)
    none=~(jnp.any(kv_mask)|jnp.any(layer_mask)|jnp.any(flow_mask))
    full=(~use_vmod)&jnp.all(layer_mask)&jnp.all(flow_mask)&jnp.all(kv_mask)
    expected=_selected_cache(prepared,vmod,kv_mask,layer_mask,flow_mask[FIRST_CALL],use_vmod)
    gates={'prefix_mask_identical':jnp.array_equal(prepared.prefix_mask,prepared.donor_prefix_mask),
      'prefix_positions_identical':jnp.array_equal(prepared.prefix_positions,prepared.donor_prefix_positions),
      'valid_token_support_identical':jnp.array_equal(prepared.token_mask,prepared.prefix_mask),
      'declared_selector_scope':none|(main_layer&main_flow)|full,
      'Vmod_unchanged_padding':(~use_vmod)|ignored_values_equal,
      'Vmod_finite':jnp.all(jnp.isfinite(vmod)),
      'actual_first_K_matches_selector':_bf16_bits_equal(first['prefix_key_used'],expected[0][LAYER]),
      'actual_first_V_matches_selector':_bf16_bits_equal(first['prefix_value_used'],expected[1][LAYER]),
      'schedule_finished_at_ten':index==10,'num_steps_is_ten':num_steps==10,'schedule_terminal_time':time<-dt/2,
      'first_j5_reached':reached,'original_noise_finite':jnp.all(jnp.isfinite(noise)),
      'cache_finite':jnp.all(jnp.stack([jnp.all(jnp.isfinite(a))for a in(*prepared.cache,*prepared.donor_cache)])),
      'flow_finite':jnp.all(jnp.isfinite(xs))&jnp.all(jnp.isfinite(velocities))&jnp.all(jnp.isfinite(times)),
      'endpoint_matches_final_state':jnp.array_equal(final,xs[-1])}
    return final,{'gates':gates,'flow':{'x':xs,'velocity_native':velocities,'time':times,'time_weak_type':weak_times,'dt':jnp.broadcast_to(jnp.asarray(dt),(10,))},
      'first_read':first,'first_read_layer_index':jnp.asarray(LAYER,jnp.int32),'first_read_call_index':jnp.asarray(FIRST_CALL,jnp.int32),
      'pre_j5_x':xs[FIRST_CALL],'pre_j5_time':times[FIRST_CALL],'pre_j5_time_weak_type':weak_times[FIRST_CALL],
      'receiver_L9_K':prepared.cache[0][LAYER],'receiver_L9_V':prepared.cache[1][LAYER],
      'donor_L9_K':prepared.donor_cache[0][LAYER],'donor_L9_V':prepared.donor_cache[1][LAYER],'Vmod_argument':vmod,
      'receiver_donor_cache_equal':jnp.stack([_bf16_bits_equal(a,b)for a,b in zip(prepared.cache,prepared.donor_cache,strict=True)]),
      'prefix_mask':prepared.prefix_mask,'prefix_positions':prepared.prefix_positions,
      'selectors':{'kv_mask':kv_mask,'layer_mask':layer_mask,'flow_mask':flow_mask,'use_vmod':use_vmod},
      'completed_updates':index,'final_time':time,
      'sampling_abi':{'num_steps':jnp.asarray(num_steps),'num_steps_is_trace_input':jnp.asarray(isinstance(num_steps,jax.core.Tracer)),
        'num_steps_weak_type':jnp.asarray(bool(getattr(jnp.asarray(num_steps),'weak_type',False))),
        'actual_dt':jnp.asarray(dt),'dt_weak_type':jnp.asarray(bool(getattr(jnp.asarray(dt),'weak_type',False)))}}


def shared_response_query(model,receiver_observation,donor_observation,noise,vmod,kv_mask,layer_mask,flow_mask,use_vmod,*,num_steps=10):
    require(receiver_observation is not None and donor_observation is not None,'two_nonempty_real_prefills_required')
    J._architecture(model)
    prepared=J.prepare_candidate(model,receiver_observation,donor_observation)
    return _integrate(model,prepared,noise,vmod,kv_mask,layer_mask,flow_mask,use_vmod,num_steps=num_steps)


def check_result(metadata):
    required={'prefix_mask_identical','prefix_positions_identical','valid_token_support_identical','declared_selector_scope',
      'Vmod_unchanged_padding','Vmod_finite','actual_first_K_matches_selector','actual_first_V_matches_selector',
      'schedule_finished_at_ten','num_steps_is_ten','schedule_terminal_time','first_j5_reached','original_noise_finite',
      'cache_finite','flow_finite','endpoint_matches_final_state'}
    require(set(metadata['gates'])==required,'all_new_graph_result_gates')
    for name in sorted(required):require(bool(metadata['gates'][name]),'shared_response_'+name)
    validate_selectors(**metadata['selectors'])
    return True  # source/shape/schedule only; never native qualification


def edit_identity(metadata):
    """Host identity uses ACTUAL traced values, not ideal projection coordinates."""
    import numpy as np
    check_result(metadata)
    def ident(value):
        a=np.asarray(value);require(not a.dtype.hasobject,'numeric_identity_only')
        header={'shape':list(a.shape),'logical_dtype':a.dtype.name,'stored_dtype':a.dtype.str,'bytes':a.nbytes}
        return {**header,'sha256':hashlib.sha256(json.dumps(header,sort_keys=True,separators=(',',':')).encode()+b'\0'+a.tobytes()).hexdigest()}
    base=np.asarray(metadata['receiver_L9_V']);used=np.asarray(metadata['first_read']['prefix_value_used'])
    edit=used.astype(np.float32)-base.astype(np.float32)
    return {'schema':'actual-native-BF16-L9-edit-identity-v1','receiver_V':ident(base),'Vmod_argument':ident(metadata['Vmod_argument']),
      'actually_used_V_at_j5':ident(used),'actual_E_used_minus_receiver_FP32':ident(edit),
      'actual_E_nonzero_elements':int(np.count_nonzero(edit)),'whole_E_is_zero':bool(not np.any(edit)),
      'P_kind':'native_bfloat16_full_prefix_plus_suffix','precast_FP32_P_available':False,'qualified':False}


def require_same_edit_across_K(left,right):
    """Post-call necessary pairing gate; never chooses E after observing Y."""
    import numpy as np
    check_result(left);check_result(right)
    ls,rs=left['selectors'],right['selectors']
    require(bool(ls['use_vmod'])and bool(rs['use_vmod'])and bool(ls['kv_mask'][1])and bool(rs['kv_mask'][1])
        and bool(ls['kv_mask'][0])!=bool(rs['kv_mask'][0]),'two_K_levels_same_active_Vmod')
    require(np.array_equal(ls['layer_mask'],rs['layer_mask'])and np.array_equal(ls['flow_mask'],rs['flow_mask']),'same_K_pair_edit_schedule')
    def same(a,b):
        a,b=np.asarray(a),np.asarray(b);return a.shape==b.shape and a.dtype==b.dtype and a.tobytes()==b.tobytes()
    for name in('Vmod_argument','receiver_L9_V','receiver_L9_K','donor_L9_K','prefix_mask','prefix_positions','pre_j5_x','pre_j5_time','pre_j5_time_weak_type'):
        require(same(left[name],right[name]),'same_E_K_pair_'+name)
    for name,count in(('x',6),('velocity_native',5),('time',6),('time_weak_type',6)):
        require(same(left['flow'][name][:count],right['flow'][name][:count]),'same_pre_j5_history_'+name)
    for name in('prefix_value_used','query_post_rope_scaled','suffix_key_post_rope','suffix_value','receiver_residual_input','attention_gate'):
        require(same(left['first_read'][name],right['first_read'][name]),'same_first_receiver_K_pair_'+name)
    return True


@dataclass(frozen=True)
class Runtime:
    common:Callable
    stock:Callable
    decode:Callable
    wait:Callable
    common_compiled:Callable


def production_runtime(campaign):
    """Create once; zero model calls. Caller first verifies GPU encoded audits."""
    require(sys.platform.startswith('linux'),'already_loaded_authorized_Linux_runtime')
    import jax
    from openpi.eval.campaign_policy import CampaignPolicy
    from openpi.models.pi0 import Pi0
    from openpi.shared.nnx_utils import module_jit
    require(isinstance(campaign,CampaignPolicy),'actual_loaded_CampaignPolicy')
    model=campaign.policy._model;require(isinstance(model,Pi0)and getattr(model.sample_actions,'__func__',None)is Pi0.sample_actions,'original_stock_sampler')
    wrapped=getattr(campaign.sample,'__wrapped__',None)
    require(getattr(wrapped,'__self__',None)is model and getattr(wrapped,'__func__',None)is Pi0.sample_actions,'original_stock_binding')
    require(not jax.config.jax_enable_x64,'native_x64_false')
    compiled=module_jit(MethodType(shared_response_query,model),static_argnums=(),static_argnames=())
    def common(receiver,donor,noise,vmod,kv_mask,layer_mask,flow_mask,use_vmod):
        validate_selectors(kv_mask,layer_mask,flow_mask,use_vmod)
        return compiled(receiver.observation,donor.observation,noise,vmod,kv_mask,layer_mask,flow_mask,use_vmod,num_steps=10)
    return Runtime(common,lambda branch,noise:campaign.sample(jax.random.key(0),branch.observation,noise=noise,num_steps=10),
        lambda raw,branch:campaign.decode(raw,branch.decode_states),jax.block_until_ready,compiled)
