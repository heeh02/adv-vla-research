"""Native G-star same-query contrasts. All neural reductions are descriptive."""
import numpy as np
import native_reader as N
ROLES=('baseline_none_1','shared_pulse','shared_persistent')
CONTRASTS=(('pulse_minus_none',1,0),('persistent_minus_none',2,0),('persistent_minus_pulse',2,1))
LAYERS=(0,8,17);FLOWS=(5,6,7,8,9)
COMMANDS=('translation_command_x','translation_command_y','translation_command_z','rotation_command_x','rotation_command_y','rotation_command_z','gripper_command')
GATES={'prefix_mask_identical','prefix_positions_identical','valid_token_support_identical','declared_selector_scope','Vmod_unchanged_padding','Vmod_finite','actual_all_K_matches_selector','actual_all_V_matches_selector','schedule_finished_at_ten','num_steps_is_ten','schedule_terminal_time','all_j5_to_j9_reached','original_noise_finite','cache_finite','flow_finite','endpoint_matches_final_state'}
FIELDS=('attention_output_post_WO_pre_gate','receiver_residual_input','hidden_after_attention_residual','hidden_post_block')

def exact(a,b):return a.shape==b.shape and a.dtype==b.dtype and a.tobytes()==b.tobytes()
def check(ok,code):N.need(ok,code)
def val(r,path,shape,dtype):return N.field(r,path,shape,dtype).numeric()
def bit(r,path):return r.field(*path).storage

def metrics(a,b):
 a=np.asarray(a,np.float64).ravel();b=np.asarray(b,np.float64).ravel();d=b-a
 aa=float(a@a);bb=float(b@b);dd=float(d@d);den=np.sqrt(aa*bb)
 return {'coordinate_count':len(d),'signed_mean':float(d.mean()),'signed_min':float(d.min()),'signed_max':float(d.max()),'rms_difference':float(np.sqrt(dd/len(d))),
  'maxabs_difference':float(np.max(np.abs(d))),'difference_energy':dd,'reference_energy':aa,'candidate_energy':bb,
  'cosine':None if den==0 else float(np.clip((a@b)/den,-1,1)),'relative_L2':None if aa==0 else float(np.sqrt(dd/aa)),
  'exactzero_difference':bool(not np.any(d)),'reference_exactzero':aa==0,'candidate_exactzero':bb==0,'different_coordinates':int(np.count_nonzero(d))}

def validate(inp,calls,entries,root,q):
 check(inp.event=='PVP_INPUT_VERIFIED'and inp.metadata['root_id']==root and inp.metadata['query_index']==q,'PVP_actual_input_root_query')
 check(len(calls)==3 and len(entries)==3 and len({r.synthetic for r in [inp,*calls]})==1,'PVP_three_roles_same_data_scope')
 noise=N.field(inp,('noise',),(1,10,32),'float32');N.field(inp,('consumed_decode_state',),(1,32),'float32')
 n=calls[0].field('prefix_mask').shape[-1];check(0<n<=968,'PVP_actual_prefix_capacity')
 vm=N.field(inp,('Vmod',),(1,n,1,256),'bfloat16');e=N.field(inp,('actual_E',),(1,n,1,256),'float32')
 check(len({r.metadata['input_identity']for r in calls})==1,'PVP_same_encoded_inputs_and_noise')
 for i,(r,entry)in enumerate(zip(calls,entries)):
  m=r.metadata;check(r.event=='PVP_CALL_OUTPUT'and m['graph']=='Gstar','PVP_same_Gstar_not_R')
  check(all(m.get(k)==v for k,v in entry.items()),'PVP_actual_entry_not_order_inference')
  check(m['role']==ROLES[i]and m['root_id']==root and m['query_index']==q and m['receiver']=='T'and m['donor']=='C','PVP_actual_scientific_role')
  check(N.same_ref(m['input_ref'],inp.ref)and m['PG_refs']==calls[0].metadata['PG_refs'],'PVP_same_original_input_and_PG_references')
  val(r,('raw32',),(1,10,32),'float32');val(r,('adapter7',),(1,10,7),'float64')
  val(r,('prefix_mask',),(1,n),'bool');val(r,('prefix_positions',),(1,n),'int32')
  check(np.array_equal(val(r,('capture_layers',),(3,),'int32'),LAYERS)and np.array_equal(val(r,('capture_calls',),(5,),'int32'),FLOWS),'PVP_recorded_axes')
  for name,shape,dtype in [('x',(11,1,10,32),'float32'),('velocity_native',(10,1,10,32),'bfloat16'),('time',(11,),'float32'),('time_weak_type',(11,),'bool'),('dt',(10,),'float32')]:val(r,('flow',name),shape,dtype)
  check(exact(bit(r,('flow','x'))[0],noise.storage),'PVP_same_actual_noise')
  for field in FIELDS:val(r,('reads',field),(5,3,1,10,1024),'bfloat16')
  for field,shape in [('query_post_rope_scaled',(5,3,1,10,1,8,256)),('probabilities_native',(5,3,1,1,8,10,n+10)),('attention_gate',(5,3,1,1,1024)),('suffix_key_post_rope',(5,3,1,10,1,256)),('suffix_value',(5,3,1,10,1,256))]:val(r,('reads',field),shape,'bfloat16')
  for k in('kv_mask','layer_mask','flow_mask','use_vmod'):
   check(np.array_equal(r.field('selectors',k).numeric(),np.asarray(entry[k],bool)),'PVP_actual_dynamic_selector')
  check(set(r.arrays['gates'])==GATES and all(isinstance(a,N.Native)and a.dtype=='bool'and a.shape==()and bool(a.numeric())for a in r.arrays['gates'].values()),'PVP_recorded_native_gates_pass')
  if i:
   a=m['Vmod_actual_identity'];check(a['shape']==list(vm.shape)and a['logical_dtype']=='bfloat16'and a['bytes']==vm.storage.nbytes and a['sha256']==N.sha(vm.storage.tobytes()),'PVP_actual_same_frozen_Vmod')
  for k in('prefix_mask','prefix_positions'):check(N.equal(calls[0].field(k),r.field(k)),'PVP_same_prefix_support_positions')
 p,s=calls[1:]
 for name,count in(('x',7),('velocity_native',6),('time',7),('time_weak_type',7),('dt',10)):
  check(exact(bit(p,('flow',name))[:count],bit(s,('flow',name))[:count]),'PVP_same_first_write_and_x6_'+name)
 for name in p.arrays['reads']:
  check(name in s.arrays['reads']and exact(bit(p,('reads',name))[0],bit(s,('reads',name))[0]),'PVP_same_j5_'+name)
 for name in('query_post_rope_scaled','probabilities_native','receiver_residual_input','suffix_key_post_rope','suffix_value','attention_gate'):
  check(exact(bit(p,('reads',name))[1,:2],bit(s,('reads',name))[1,:2]),'PVP_j6_L1_L9_same_receiver_'+name)
 for name in('mask','suffix_positions','query_defined','adarms_condition'):
  check(exact(bit(p,('reads',name))[:2],bit(s,('reads',name))[:2]),'PVP_j5_j6_same_'+name)
 for r in calls:
  check(not np.any(e.numeric()[:,~r.field('prefix_mask').numeric()[0]]),'PVP_no_actual_E_on_padding')
 return n

def analyze(inp,calls,entries,base):
 root,q=base['root_id'],base['query_index'];n=validate(inp,calls,entries,root,q)
 base={**base,'policy_env_step':5*q,'absolute_env_step':10+5*q,'graph':'Gstar'}
 endpoints=np.stack([r.field('adapter7').numeric()[0]for r in calls]);raw=np.stack([r.field('raw32').numeric()[0]for r in calls])
 xs=np.stack([r.field('flow','x').numeric()[:,0]for r in calls]);vs=np.stack([r.field('flow','velocity_native').numeric()[:,0]for r in calls])
 slot=[];windows=[];neural=[];flows=[];reads=[]
 for ci,(name,b,a)in enumerate(CONTRASTS):
  delta=endpoints[b]-endpoints[a]
  for p in range(10):
   for d,dim in enumerate(COMMANDS):slot.append({**base,'contrast':name,'action_slot':p,'command_dimension':dim,'reference':float(endpoints[a,p,d]),'candidate':float(endpoints[b,p,d]),'signed_effect':float(delta[p,d]),'exactzero':delta[p,d]==0,'unit':'command_adapter_units','proposed_not_issued':True})
  for count in(5,10):
   for d,dim in enumerate(COMMANDS):
    v=delta[:count,d];windows.append({**base,'contrast':name,'window':'first5'if count==5 else'full10','command_dimension':dim,'mean_signed':float(v.mean()),'meanabs':float(np.abs(v).mean()),'RMS':float(np.sqrt(np.mean(v*v))),'maxabs':float(np.max(np.abs(v))),'exactzero_slots':int(np.count_nonzero(v==0)),'positions':count,'unit':'command_adapter_units','proposed_not_issued':True})
  for field,array in(('x',xs),('velocity_native',vs)):
   for j in range(array.shape[1]):
    for p in range(10):flows.append({**base,'contrast':name,'field':field,'flow_index':j,'reference_flow_time':float(calls[a].field('flow','time').numeric()[j]),'candidate_flow_time':float(calls[b].field('flow','time').numeric()[j]),'action_slot':p,'coordinates':'native_32_not_physical_units',**metrics(array[a,j,p],array[b,j,p])})
  for ji,j in enumerate(FLOWS):
   for li,layer in enumerate(LAYERS):
    row={**base,'contrast':name,'flow_index':j,'reference_flow_time':float(calls[a].field('flow','time').numeric()[j]),'candidate_flow_time':float(calls[b].field('flow','time').numeric()[j]),'layer_index':layer,'layer_label':layer+1}
    for field in('query_post_rope_scaled','probabilities_native','attention_gate','suffix_key_post_rope','suffix_value'):
     aa=bit(calls[a],('reads',field))[ji,li];bb=bit(calls[b],('reads',field))[ji,li]
     row[field+'_typed_exact']=exact(aa,bb)
    reads.append(row)
    for field in FIELDS:
     aa=calls[a].field('reads',field).numeric()[ji,li,0];bb=calls[b].field('reads',field).numeric()[ji,li,0]
     for p in range(10):neural.append({**base,'contrast':name,'field':field,'flow_index':j,'reference_flow_time':float(calls[a].field('flow','time').numeric()[j]),'candidate_flow_time':float(calls[b].field('flow','time').numeric()[j]),'layer_index':layer,'layer_label':layer+1,'action_slot':p,'coordinates':'AE_1024_native',**metrics(aa[p],bb[p])})
 e=inp.field('actual_E').numeric();mask=calls[0].field('prefix_mask').numeric();ee=e[0,mask[0],0].astype(np.float64)
 proof={**base,'selected_calls':[{'role':r.metadata['role'],'call_id':r.metadata['call_id'],'ref':r.ref}for r in calls],
  'input_ref':inp.ref,'old_saved_sources':inp.metadata['sources'],'same_j5_and_x6_checked':True,'same_j6_L1_L9_receiver_checked':True,
  'exactzero_actual_E':bool(not np.any(e)),'actual_E_valid_energy':float(np.sum(ee*ee)),'valid_prefix_tokens':int(mask.sum()),
  'prefix_capacity':n,'same_E_all_three_calls_claim':False,'same_E_pulse_persistent':True,'all_readout_calls_source_qualified':True}
 compact={'adapter7':endpoints,'raw32':raw,'flow_x':xs,'flow_velocity_decoded':vs,'flow_time':np.stack([r.field('flow','time').numeric()for r in calls])}
 return {'slots':slot,'windows':windows,'neural':neural,'flow':flows,'reading_exact':reads,'proof':proof,'compact':compact}


def summary_rows(rows,*,neural):
 """Small root readback. Full action-position rows remain in original tables."""
 groups={}
 for row in rows:
  if neural and row['contrast']!='persistent_minus_pulse':continue
  for size in ((5,10)if neural else(10,)):
   if row['action_slot']>=size:continue
   keys=['task_id','root_id','parent_condition','query_index','policy_env_step','absolute_env_step','graph','contrast','field','flow_index','reference_flow_time','candidate_flow_time','coordinates']
   if neural:keys+=['layer_index','layer_label']
   key=tuple((k,row[k])for k in keys)+(('action_window','first5'if size==5 else'full10'),)
   groups.setdefault(key,[]).append(row)
 out=[]
 for key,values in groups.items():
  count=sum(r['coordinate_count']for r in values);energy=sum(r['difference_energy']for r in values)
  out.append({**dict(key),'action_positions':len(values),'coordinates_in_window':count,'signed_mean':sum(r['signed_mean']*r['coordinate_count']for r in values)/count,
   'RMS_difference':float(np.sqrt(energy/count)),'maxabs_difference':max(r['maxabs_difference']for r in values),
   'difference_energy':energy,'reference_energy':sum(r['reference_energy']for r in values),'candidate_energy':sum(r['candidate_energy']for r in values),
   'all_exactzero_difference':all(r['exactzero_difference']for r in values),'different_coordinates':sum(r['different_coordinates']for r in values),
   'summary_not_independent_samples':True})
 return out
