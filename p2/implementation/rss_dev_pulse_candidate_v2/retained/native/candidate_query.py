"""G-star: same native Euler/setter math, real selected PG and j5..9 outputs.

No model restoration, environment, warmup, float promotion, or cloud API.
The independent Gemma overlay only exports an already computed residual leaf.
"""
from dataclasses import dataclass
from pathlib import Path
from types import MethodType
import hashlib,importlib.util,sys

RETAINED_SHA='82945974b6a50302226a262f8dfc4c35c855863febe861a510c08d58fd19d3e7'
p=Path(__file__).with_name('retained_shared_response.py')
if not p.is_file():p=Path(__file__).resolve().parents[4]/'execution/shared-response-edit-v1/shared_response_query.py'
if hashlib.sha256(p.read_bytes()).hexdigest()!=RETAINED_SHA:raise ValueError('PVP_original_setter_SHA')
s=importlib.util.spec_from_file_location('_PVP_retained_shared_response',p);D=importlib.util.module_from_spec(s);sys.modules[s.name]=D;s.loader.exec_module(D)
J=D.J;need=D.require
LAYERS=(0,8,17);CALLS=(5,6,7,8,9)
FIELDS=('query_post_rope_scaled','probabilities_native','attention_output_post_WO_pre_gate','attention_gate',
 'receiver_residual_input','hidden_after_attention_residual','hidden_post_block','prefix_key_used','prefix_value_used','suffix_key_post_rope','suffix_value')
GATES=('prefix_mask_identical','prefix_positions_identical','valid_token_support_identical','declared_selector_scope',
 'Vmod_unchanged_padding','Vmod_finite','actual_all_K_matches_selector','actual_all_V_matches_selector','schedule_finished_at_ten',
 'num_steps_is_ten','schedule_terminal_time','all_j5_to_j9_reached','original_noise_finite','cache_finite','flow_finite','endpoint_matches_final_state')

def selectors(schedule):
 import numpy as np
 need(schedule in('none','pulse','persistent'),'PVP_fixed_schedule')
 return (np.asarray([False,schedule!='none'],bool),np.asarray([i==8 and schedule!='none'for i in range(18)],bool),
  np.asarray([i==5 if schedule=='pulse'else i>=5 if schedule=='persistent'else False for i in range(10)],bool),np.asarray(schedule!='none',bool))

def validate_selectors(kv_mask,layer_mask,flow_mask,use_vmod):
 import numpy as np
 D.validate_selectors(kv_mask,layer_mask,flow_mask,use_vmod)
 got=[np.asarray(x)for x in(kv_mask,layer_mask,flow_mask,use_vmod)]
 need(any(all(np.array_equal(x,y)for x,y in zip(got,selectors(name)))for name in('none','pulse','persistent')),'PVP_KT_only_three_schedules')

def _prepare(model,receiver,donor):
 J._architecture(model);_,jnp,preprocess,make_mask=J._deps()
 need(receiver is not None and donor is not None,'PVP_two_original_observations')
 def prefill(obs):
  obs=preprocess(None,obs,train=False);prefix,valid,ar=model.embed_prefix(obs);pos=jnp.cumsum(valid,axis=1)-1
  _,cache,pg=model.PaliGemma.llm([prefix,None],mask=make_mask(valid,ar),positions=pos,method='scan_native_prefix')
  return obs,valid,pos,cache,pg
 obs,valid,pos,cache,tpg=prefill(receiver);_,dv,dp,dc,cpg=prefill(donor)
 prepared=J.Prepared(obs,cache,dc,valid,dv,pos,dp,valid);J._prepared_abi(prepared)
 return prepared,{'T':tpg,'C':cpg}

def _empty(prepared):
 _,jnp,_,_=J._deps();b,n=prepared.prefix_mask.shape;z=lambda shape:jnp.zeros((5,3,*shape),jnp.bfloat16)
 out={'query_post_rope_scaled':z((b,10,1,8,256)),'probabilities_native':z((b,1,8,10,n+10)),
 'attention_output_post_WO_pre_gate':z((b,10,1024)),'attention_gate':z((b,1,1024)),
 'receiver_residual_input':z((b,10,1024)),'hidden_after_attention_residual':z((b,10,1024)),
 'hidden_post_block':z((b,10,1024)),'prefix_key_used':z((b,n,1,256)),'prefix_value_used':z((b,n,1,256)),
 'suffix_key_post_rope':z((b,10,1,256)),'suffix_value':z((b,10,1,256))}
 out.update(mask=jnp.zeros((5,b,10,n+10),jnp.bool_),suffix_positions=jnp.zeros((5,b,10),prepared.prefix_positions.dtype),
 query_defined=jnp.zeros((5,b,10),jnp.bool_),adarms_condition=jnp.zeros((5,b,1024),jnp.float32),
 final_readout=jnp.zeros((5,b,10,1024),jnp.bfloat16))
 return out

def _velocity(model,prepared,x,time,cache):
 suffix,mask,pos,condition=J._suffix(model,prepared,x,time)
 readout,ops=model.PaliGemma.llm([None,suffix],mask=mask,positions=pos,adarms_cond=[None,condition],kv_cache=cache,method='scan_native_suffix')
 velocity=model.action_out_proj(readout[:,-10:])
 trace={name:ops[name]for name in FIELDS}
 trace.update(mask=mask,suffix_positions=pos,query_defined=ops['query_defined'],adarms_condition=condition,final_readout=readout[:,-10:])
 return velocity,trace

def _integrate(model,prepared,noise,vmod,kv_mask,layer_mask,flow_mask,use_vmod,*,num_steps):
 D._abi(prepared,noise,vmod,kv_mask,layer_mask,flow_mask,use_vmod,num_steps)
 jax,jnp,_,_=J._deps();dt=-1.0/num_steps
 xs=jnp.zeros((11,*noise.shape),noise.dtype).at[0].set(noise);velocities=jnp.zeros((10,*noise.shape),jnp.bfloat16)
 times=jnp.zeros((11,),jnp.asarray(1.0).dtype).at[0].set(1.0);weak_times=jnp.zeros(11,jnp.bool_).at[0].set(True);trace=_empty(prepared)
 def body(carry):
  x,time,index,xs,velocities,times,weak_times,trace,filled=carry
  cache=D._selected_cache(prepared,vmod,kv_mask,layer_mask,flow_mask[index],use_vmod)
  velocity,operands=_velocity(model,prepared,x,time,cache)
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
 'actual_dt':jnp.asarray(dt),'dt_weak_type':jnp.asarray(bool(getattr(jnp.asarray(dt),'weak_type',False)))}}

def query(model,receiver,donor,noise,vmod,kv_mask,layer_mask,flow_mask,use_vmod,*,num_steps=10):
 prepared,pg=_prepare(model,receiver,donor)
 raw,meta=_integrate(model,prepared,noise,vmod,kv_mask,layer_mask,flow_mask,use_vmod,num_steps=num_steps)
 return raw,{**meta,'pg':pg}

def check_result(m):
 import numpy as np
 need(set(m['gates'])==set(GATES),'PVP_exact_gate_set')
 for name in GATES:need(bool(m['gates'][name]),'PVP_'+name)
 validate_selectors(**m['selectors']);need(np.array_equal(m['capture_layers'],LAYERS)and np.array_equal(m['capture_calls'],CALLS),'PVP_real_capture_axes')
 need(np.asarray(m['capture_filled']).shape==(5,)and np.asarray(m['capture_filled']).all(),'PVP_all_reads_recorded')
 n=np.asarray(m['prefix_mask']).shape[1]
 for k,shape,dtype in [('x',(11,1,10,32),'float32'),('velocity_native',(10,1,10,32),'bfloat16'),('time',(11,),'float32'),('time_weak_type',(11,),'bool'),('dt',(10,),'float32')]:
  a=np.asarray(m['flow'][k]);need(a.shape==shape and a.dtype.name==dtype and np.isfinite(a).all(),'PVP_actual_flow_'+k)
 need(bool(m['sampling_abi']['num_steps_is_trace_input'])and np.asarray(m['sampling_abi']['actual_dt']).dtype==np.float32,'PVP_original_dynamic_num_steps_FP32_dt')
 for k,shape in [('query_post_rope_scaled',(5,3,1,10,1,8,256)),('probabilities_native',(5,3,1,1,8,10,n+10)),('attention_output_post_WO_pre_gate',(5,3,1,10,1024)),('attention_gate',(5,3,1,1,1024)),('receiver_residual_input',(5,3,1,10,1024)),('hidden_after_attention_residual',(5,3,1,10,1024)),('hidden_post_block',(5,3,1,10,1024)),('prefix_key_used',(5,3,1,n,1,256)),('prefix_value_used',(5,3,1,n,1,256)),('suffix_key_post_rope',(5,3,1,10,1,256)),('suffix_value',(5,3,1,10,1,256))]:
  a=np.asarray(m['reads'][k]);need(a.shape==shape and a.dtype.name=='bfloat16'and np.isfinite(a).all(),'PVP_actual_read_'+k)
 for side in('C','T'):
  p=m['pg'][side];n=np.asarray(m['prefix_mask']).shape[1]
  for k,shape in [('hidden_input',(3,1,n,2048)),('hidden_pre_attention_norm',(3,1,n,2048)),('cache_k',(3,1,n,1,256)),('cache_v',(3,1,n,1,256))]:
   a=np.asarray(p[k]);need(a.shape==shape and a.dtype.name=='bfloat16'and np.isfinite(a).all(),'PVP_actual_PG_'+side+'_'+k)
  need(np.array_equal(p['selected_layers'],LAYERS)and np.array_equal(p['positions'],m['prefix_positions']),'PVP_PG_axes_positions')
 return True

def production_runtime(campaign):
 need(sys.platform.startswith('linux'),'PVP_loaded_Linux_runtime_only')
 import jax
 from openpi.eval.campaign_policy import CampaignPolicy
 from openpi.models.pi0 import Pi0
 from openpi.shared.nnx_utils import module_jit
 need(isinstance(campaign,CampaignPolicy)and isinstance(campaign.policy._model,Pi0),'PVP_actual_loaded_policy')
 need(not jax.config.jax_enable_x64,'PVP_native_x64_false')
 need(len(jax.devices())==1 and jax.devices()[0].platform=='gpu','PVP_one_allocated_GPU')
 compiled=module_jit(MethodType(query,campaign.policy._model),static_argnums=(),static_argnames=())
 def common(receiver,donor,noise,vmod,schedule):
  args=selectors(schedule);validate_selectors(*args)
  return compiled(receiver.observation,donor.observation,noise,vmod,*args,num_steps=10)
 return {'common':common,'decode':lambda raw,branch:campaign.decode(raw,branch.decode_states),'wait':jax.block_until_ready,'compiled':compiled}
