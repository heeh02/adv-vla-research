"""B-only first-suffix bridge at original j6, L9 post-WO/pre-gate.

No model loading, random noise, cloud, environment or parameter mutation.
Every API call includes the frozen two-prefill boundary. Single returns v6,
not an integrated action, and never changes num_steps to1 or updates Euler.
"""
from pathlib import Path
from types import MethodType
import hashlib
import importlib.util
import sys

HERE = Path(__file__).resolve().parent
PVP_SHA = 'efbfbbe7214af7d97ff52df4b1a20e155617f0dfa5e7c53982914432f2f1c024'
path = HERE/'frozen/candidate_query.py'
if hashlib.sha256(path.read_bytes()).hexdigest() != PVP_SHA:
    raise ValueError('frozen_PVP_query_SHA')
spec = importlib.util.spec_from_file_location('_B_bridge_frozen_PVP_query', path)
Q = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = Q
spec.loader.exec_module(Q)
J, D, need = Q.J, Q.D, Q.need
FIELDS = Q.FIELDS + ('bridge_W_before', 'bridge_W_used', 'bridge_active')
LAYERS, CALLS = Q.LAYERS, Q.CALLS


def _empty(prepared):
    _, jnp, _, _ = J._deps()
    trace = Q._empty(prepared)
    shape = trace['attention_output_post_WO_pre_gate'].shape
    trace.update(bridge_W_before=jnp.zeros(shape, jnp.bfloat16),
                 bridge_W_used=jnp.zeros(shape, jnp.bfloat16),
                 bridge_active=jnp.zeros((5, 3), jnp.bool_))
    return trace


def _velocity(model, prepared, x, time, cache, target, active, *, independent=False):
    suffix, mask, pos, condition = J._suffix(model, prepared, x, time)
    method = 'scan_b_bridge_independent_suffix' if independent else 'scan_b_bridge_suffix'
    readout, ops = model.PaliGemma.llm(
        [None, suffix], mask=mask, positions=pos, adarms_cond=[None, condition], kv_cache=cache,
        w_target=target, w_active=active, method=method)
    velocity = model.action_out_proj(readout[:, -10:])
    trace = {name: ops[name] for name in FIELDS}
    trace.update(mask=mask, suffix_positions=pos, query_defined=ops['query_defined'],
                 adarms_condition=condition, final_readout=readout[:, -10:])
    return velocity, trace


def _integrate_inactive(model, prepared, noise, vmod, kv_mask, layer_mask, flow_mask, use_vmod,
                        w_target, w_active, *, num_steps):
    """Frozen PVP integrator expressions, plus a dynamically disabled W selector."""
    D._abi(prepared,noise,vmod,kv_mask,layer_mask,flow_mask,use_vmod,num_steps)
    jax,jnp,_,_=J._deps();dt=-1.0/num_steps
    xs=jnp.zeros((11,*noise.shape),noise.dtype).at[0].set(noise);velocities=jnp.zeros((10,*noise.shape),jnp.bfloat16)
    times=jnp.zeros((11,),jnp.asarray(1.0).dtype).at[0].set(1.0);weak_times=jnp.zeros(11,jnp.bool_).at[0].set(True);trace=_empty(prepared)
    def body(carry):
        x,time,index,xs,velocities,times,weak_times,trace,filled=carry
        cache=D._selected_cache(prepared,vmod,kv_mask,layer_mask,flow_mask[index],use_vmod)
        velocity,operands=_velocity(model,prepared,x,time,cache,w_target,w_active & (index == 6))
        need(velocity.shape==x.shape and velocity.dtype==jnp.bfloat16,'native_BF16_velocity')
        trace=jax.lax.cond(index>=5,lambda:jax.tree.map(lambda old,new:old.at[index-5].set(new),trace,operands),lambda:trace)
        filled=jax.lax.cond(index>=5,lambda:filled.at[index-5].set(True),lambda:filled)
        next_x=x+dt*velocity;next_time=time+dt
        return (next_x,next_time,index+1,xs.at[index+1].set(next_x),velocities.at[index].set(velocity),times.at[index+1].set(next_time),
            weak_times.at[index+1].set(bool(getattr(next_time,'weak_type',False))),trace,filled)
    def cond(carry):
        _,time,index,*_=carry
        return(time>=-dt/2)&(index<10)
    final,time,index,xs,velocities,times,weak_times,trace,filled=jax.lax.while_loop(cond,body,
        (noise,1.0,jnp.asarray(0,jnp.int32),xs,velocities,times,weak_times,trace,jnp.zeros(5,jnp.bool_)))
    valid=prepared.prefix_mask[...,None,None]
    ignored=jnp.all(jnp.where(valid,True,jax.lax.bitcast_convert_type(vmod,jnp.uint16)==jax.lax.bitcast_convert_type(prepared.cache[1][8],jnp.uint16)))
    none=~(jnp.any(kv_mask)|jnp.any(layer_mask)|jnp.any(flow_mask)|use_vmod)
    main=(~kv_mask[0])&kv_mask[1]&use_vmod&jnp.array_equal(layer_mask,jnp.arange(18)==8)&(jnp.array_equal(flow_mask,jnp.arange(10)==5)|jnp.array_equal(flow_mask,jnp.arange(10)>=5))
    matches=[]
    for ji,j in enumerate(CALLS):
        cache=D._selected_cache(prepared,vmod,kv_mask,layer_mask,flow_mask[j],use_vmod)
        matches.append(jnp.stack([jnp.stack([D._bf16_bits_equal(trace[name][ji,si],cache[kv][li])for kv,name in enumerate(('prefix_key_used','prefix_value_used'))])for si,li in enumerate(LAYERS)]))
    matches=jnp.stack(matches)
    gates={'prefix_mask_identical':jnp.array_equal(prepared.prefix_mask,prepared.donor_prefix_mask),
        'prefix_positions_identical':jnp.array_equal(prepared.prefix_positions,prepared.donor_prefix_positions),
        'valid_token_support_identical':jnp.array_equal(prepared.token_mask,prepared.prefix_mask),'declared_selector_scope':none|main,
        'Vmod_unchanged_padding':(~use_vmod)|ignored,'Vmod_finite':jnp.all(jnp.isfinite(vmod)),
        'actual_all_K_matches_selector':jnp.all(matches[...,0]),'actual_all_V_matches_selector':jnp.all(matches[...,1]),
        'schedule_finished_at_ten':index==10,'num_steps_is_ten':num_steps==10,'schedule_terminal_time':time<-dt/2,
        'all_j5_to_j9_reached':jnp.all(filled),'original_noise_finite':jnp.all(jnp.isfinite(noise)),
        'cache_finite':jnp.all(jnp.stack([jnp.all(jnp.isfinite(a))for a in(*prepared.cache,*prepared.donor_cache)])),
        'flow_finite':jnp.all(jnp.isfinite(xs))&jnp.all(jnp.isfinite(velocities))&jnp.all(jnp.isfinite(times)),
        'endpoint_matches_final_state':jnp.array_equal(final,xs[-1])}
    return final,{'gates':gates,'reads':trace,'capture_filled':filled,'actual_prefix_selector_matches':matches,
        'flow':{'x':xs,'velocity_native':velocities,'time':times,'time_weak_type':weak_times,'dt':jnp.broadcast_to(jnp.asarray(dt),(10,))},
        'prefix_mask':prepared.prefix_mask,'prefix_positions':prepared.prefix_positions,'Vmod_argument':vmod,
        'selectors':{'kv_mask':kv_mask,'layer_mask':layer_mask,'flow_mask':flow_mask,'use_vmod':use_vmod},
        'capture_layers':jnp.asarray(LAYERS,jnp.int32),'capture_calls':jnp.asarray(CALLS,jnp.int32),'completed_updates':index,'final_time':time,
        'sampling_abi':{'num_steps':jnp.asarray(num_steps),'num_steps_is_trace_input':jnp.asarray(isinstance(num_steps,jax.core.Tracer)),
        'num_steps_weak_type':jnp.asarray(bool(getattr(jnp.asarray(num_steps),'weak_type',False))),
        'actual_dt':jnp.asarray(dt),'dt_weak_type':jnp.asarray(bool(getattr(jnp.asarray(dt),'weak_type',False)))},
        'bridge_gates':{'full_W_selector_inactive':~w_active & ~jnp.any(trace['bridge_active']),
                        'full_schedule_is_B_pulse':jnp.array_equal(flow_mask,jnp.arange(10)==5),
                        'all_before_equals_used':D._bf16_bits_equal(trace['bridge_W_before'],trace['bridge_W_used'])}}


def full_inactive_query(model, receiver, donor, noise, vmod, kv_mask, layer_mask, flow_mask, use_vmod,
                        w_target, w_active, *, num_steps=10):
    prepared, pg = Q._prepare(model, receiver, donor)
    raw, meta = _integrate_inactive(model, prepared, noise, vmod, kv_mask, layer_mask, flow_mask, use_vmod,
                                    w_target, w_active, num_steps=num_steps)
    return raw, {**meta, 'pg': pg}


def _single(model, receiver, donor, x6, tau6, target, active, *, num_steps, independent):
    prepared, pg = Q._prepare(model, receiver, donor)
    jax, jnp, _, _ = J._deps()
    need(x6.shape == (1,10,32) and x6.dtype == jnp.float32, 'actual_B_x6_FP32_ABI')
    need(jnp.shape(tau6) == () and jnp.asarray(tau6).dtype == jnp.float32, 'actual_B_tau6_FP32_scalar')
    need(target.shape == (1,10,1024) and target.dtype == jnp.bfloat16 and active.shape == () and active.dtype == jnp.bool_,
         'native_BF16_full_W_target_dynamic_selector')
    need(jnp.shape(num_steps) == () and jnp.issubdtype(jnp.asarray(num_steps).dtype,jnp.integer), 'original_dynamic_num_steps')
    velocity, trace = _velocity(model, prepared, x6, tau6, prepared.cache, target, active, independent=independent)
    need(velocity.shape == x6.shape and velocity.dtype == jnp.bfloat16, 'single_native_BF16_velocity32')
    matches = jnp.stack([jnp.stack([D._bf16_bits_equal(trace[name][si], prepared.cache[kv][li])
                                   for kv,name in enumerate(('prefix_key_used','prefix_value_used'))]) for si,li in enumerate(LAYERS)])
    gates = {'original_num_steps_ten': num_steps == 10, 'all_future_and_current_V_edits_off': jnp.all(matches),
             'only_L9_W_selected': jnp.array_equal(trace['bridge_active'],jnp.asarray((False,True,False)) & active),
             'prefix_mask_identical':jnp.array_equal(prepared.prefix_mask,prepared.donor_prefix_mask),
             'prefix_positions_identical':jnp.array_equal(prepared.prefix_positions,prepared.donor_prefix_positions),
             'target_finite': jnp.all(jnp.isfinite(target)), 'velocity_finite': jnp.all(jnp.isfinite(velocity))}
    return velocity, {'reads':trace, 'pg':pg, 'gates':gates, 'actual_x':x6, 'actual_time':tau6,
                      'time_weak_type':jnp.asarray(bool(getattr(tau6,'weak_type',False))),
                      'prefix_mask':prepared.prefix_mask,'prefix_positions':prepared.prefix_positions,
                      'W_target_argument':target,'W_selector':active,'actual_prefix_selector_matches':matches,
                      'capture_layers':jnp.asarray(LAYERS,jnp.int32),'generation_call':jnp.asarray(6,jnp.int32),
                      'sampling_abi':{'num_steps':jnp.asarray(num_steps),'num_steps_is_trace_input':jnp.asarray(isinstance(num_steps,jax.core.Tracer)),
                      'num_steps_weak_type':jnp.asarray(bool(getattr(jnp.asarray(num_steps),'weak_type',False))),
                      'actual_dt':jnp.asarray(-1.0/num_steps),'dt_weak_type':jnp.asarray(bool(getattr(jnp.asarray(-1.0/num_steps),'weak_type',False)))},
                      'completed_updates':jnp.asarray(0,jnp.int32), 'prefix_prefills':jnp.asarray(2),
                      'suffix_evaluations':jnp.asarray(1),'product_is_consumed':jnp.asarray(False)}


def single_query(model,receiver,donor,x6,tau6,target,active,*,num_steps=10):
    return _single(model,receiver,donor,x6,tau6,target,active,num_steps=num_steps,independent=False)


def independent_single_query(model,receiver,donor,x6,tau6,target,active,*,num_steps=10):
    return _single(model,receiver,donor,x6,tau6,target,active,num_steps=num_steps,independent=True)


def restore_time(value, weak):
    import numpy as np
    import jax.numpy as jnp
    a = np.asarray(value)
    need(a.shape == () and a.dtype == np.float32 and np.isfinite(a), 'saved_tau6_FP32_scalar')
    restored = jnp.asarray(float(a)) if bool(weak) else jnp.asarray(a,dtype=jnp.float32)
    need(bool(getattr(restored,'weak_type',False)) == bool(weak) and np.asarray(restored).dtype == a.dtype and
         np.asarray(restored).tobytes() == a.tobytes(), 'restored_tau6_bits_and_weak_type')
    return restored


def production_runtime(campaign):
    need(sys.platform.startswith('linux'), 'B_bridge_loaded_Linux_runtime_only')
    import jax
    import jax.numpy as jnp
    import numpy as np
    from openpi.eval.campaign_policy import CampaignPolicy
    from openpi.models.pi0 import Pi0
    from openpi.shared.nnx_utils import module_jit
    model = campaign.policy._model
    need(isinstance(campaign,CampaignPolicy) and isinstance(model,Pi0), 'actual_loaded_policy')
    need(not jax.config.jax_enable_x64 and len(jax.devices()) == 1 and jax.devices()[0].platform == 'gpu',
         'one_native_GPU_x64_disabled')
    need(hasattr(model.PaliGemma.llm.module,'scan_b_bridge_suffix'), 'new_isolated_overlay_loaded')
    full = module_jit(MethodType(full_inactive_query,model),static_argnums=(),static_argnames=())
    one = module_jit(MethodType(single_query,model),static_argnums=(),static_argnames=())
    independent = module_jit(MethodType(independent_single_query,model),static_argnums=(),static_argnames=())
    def run_full(t,c,noise,vmod):
        selectors=Q.selectors('pulse')
        return full(t.observation,c.observation,noise,vmod,*selectors,jnp.zeros((1,10,1024),jnp.bfloat16),
                    np.asarray(False,bool),num_steps=10)
    def run_single(compiled,t,c,x6,tau6,target,active):
        need(np.asarray(active).shape == () and np.asarray(active).dtype == bool, 'host_dynamic_selector_bool')
        return compiled(t.observation,c.observation,x6,tau6,target,np.asarray(active,bool),num_steps=10)
    return {'full_inactive':run_full,
            'single':lambda *args:run_single(one,*args),
            'independent_single':lambda *args:run_single(independent,*args),
            'restore_time':restore_time,'wait':jax.block_until_ready,
            'compiled_full':full,'compiled_single':one,'compiled_independent_single':independent,
            'model_calls_on_factory_creation':0}
