"""One six-query shard of the predeclared G-star experiment. No loader/cloud/env."""
import copy,hashlib,json,pickle,random,time
from pathlib import Path
import numpy as np
import candidate_query as Q
import input_contract as I
RESULT_SCHEMA='PVP-native-shard-result-v1'
SCOPE='posthoc_same_saved_input_KT_pulse_vs_persistent'

def need(ok,code):I.need(ok,code)
def host(v):
 if isinstance(v,dict):return {k:host(x)for k,x in v.items()}
 if isinstance(v,(list,tuple)):return type(v)(host(x)for x in v)
 return np.asarray(v)if hasattr(v,'dtype')else v

def tree_exact(a,b):
 if isinstance(a,dict):return isinstance(b,dict)and set(a)==set(b)and all(tree_exact(a[k],b[k])for k in a)
 if isinstance(a,(list,tuple)):return type(a)==type(b)and len(a)==len(b)and all(tree_exact(x,y)for x,y in zip(a,b))
 return I.exact(a,b)
def aid(v):
 a=np.asarray(v);return {'shape':list(a.shape),'dtype':a.dtype.str,'logical_dtype':a.dtype.name,'bytes':a.nbytes,'sha256':hashlib.sha256(a.tobytes()).hexdigest()}
def rng():return {'python':hashlib.sha256(pickle.dumps(random.getstate(),protocol=4)).hexdigest(),'numpy':hashlib.sha256(pickle.dumps(np.random.get_state(),protocol=4)).hexdigest()}

def result_exact(a,b):
 return all(tree_exact(a[k],b[k])for k in('raw32','adapter7'))and all(tree_exact(a['metadata'][k],b['metadata'][k])for k in('flow','reads','pg','prefix_mask','prefix_positions','sampling_abi'))

def pulse_pair(p,s):
 a,b=p['metadata'],s['metadata']
 for name,count in(('x',7),('velocity_native',6),('time',7),('time_weak_type',7),('dt',10)):
  need(I.exact(a['flow'][name][:count],b['flow'][name][:count]),'PVP_same_j5_and_x6_'+name)
 for name in Q.FIELDS:need(I.exact(a['reads'][name][0],b['reads'][name][0]),'PVP_same_j5_all3_'+name)
 # j6 before L9 attention is still upstream of the first differing setter.
 for name in('query_post_rope_scaled','probabilities_native','receiver_residual_input','suffix_key_post_rope','suffix_value','attention_gate'):
  need(I.exact(a['reads'][name][1,:2],b['reads'][name][1,:2]),'PVP_same_j6_receiver_L1_L9_'+name)
 for name in('mask','suffix_positions','query_defined','adarms_condition'):
  need(I.exact(a['reads'][name][:2],b['reads'][name][:2]),'PVP_same_j5_j6_'+name)
 need(I.exact(a['Vmod_argument'],b['Vmod_argument'])and tree_exact(a['pg'],b['pg']),'PVP_same_E_K_and_prefix')
 return True

class Runtime:
 synthetic=False
 def __init__(self,campaign):
  self.campaign=campaign;self.rt=Q.production_runtime(campaign);self.common_compiled=self.rt['compiled']
 def encode(self,saved):return I.encode_loaded(self.campaign,saved)
 def compute(self,branches,saved,vmod,schedule):return self.rt['common'](branches['T'],branches['C'],saved.noise,vmod,schedule)
 def wait(self,value):return self.rt['wait'](value)
 def decode(self,raw,branches):return self.rt['decode'](raw,branches['T'])

def validate_plan(plan):
 need(plan['schema']=='PVP-native-shard-plan-v1'and plan['scope']==SCOPE and plan['shard_id']in('CCONF-T06','CCONF-T07','CCONF-T08','CCONF-T09'),'PVP_declared_shard')
 population=plan['global_population'];need(hashlib.sha256(json.dumps(population,sort_keys=True,separators=(',',':')).encode()).hexdigest()=='7013c6a2d5229d499e755bfe59789ba71f9bdc0b58da029ff14fa4b481e3474f','PVP_all24_population_frozen_identity')
 need(plan['planned_queries']==[[x['root_id'],x['query_index']]for x in population if x['task']==int(plan['shard_id'][-2:])],'PVP_actual_task_population')
 keys=[tuple(x)for x in plan['planned_queries']];need(len(keys)==len(set(keys))==6 and {q for _,q in keys}=={11,16,22},'PVP_two_roots_three_fixed_anchors')
 expected=[]
 phases={'qualification':(('baseline_none_1','none'),('baseline_none_2','none'),('V_own_null_pulse','pulse'),('V_own_null_persistent','persistent')),
 'effect':(('shared_pulse','pulse'),('shared_persistent','persistent')),'delayed_repeat':(('shared_pulse_repeat','pulse'),('shared_persistent_repeat','persistent'))}
 for phase,roles in phases.items():
  for root,q in keys:
   for role,schedule in roles:expected.append((phase,root,q,role,schedule))
 entries=plan['entries'];need(len(entries)==48 and len({e['call_id']for e in entries})==48,'PVP48_unique_charged_calls')
 need([(e['phase'],e['root_id'],e['query_index'],e['role'],e['schedule'])for e in entries]==expected,'PVP_frozen_order_roles')
 need(all(e['receiver']=='T'and e['donor']=='C'and e['prefix_prefills']==2 and e['suffix_evaluations']==10 and e['full_query']is True for e in entries),'PVP_true_two_prefill_full10_cost')
 for e in entries:
  a=Q.selectors(e['schedule']);need(all(I.exact(np.asarray(e[k]),v)for k,v in zip(('kv_mask','layer_mask','flow_mask','use_vmod'),a))and e['num_steps']==10,'PVP_frozen_actual_selectors')
 need(plan['caps']=={'full_queries':48,'prefix_prefills':96,'suffix_evaluations':480,'environment_steps':0,'archive_bytes':1024**3},'PVP_exact_per_task_caps')
 return keys

def run_shard(campaign,inputs,plan,*,charge,persist,source_guard,deadline_epoch,runtime=None):
 keys=validate_plan(plan);by={x.key:x for x in inputs};need(len(by)==len(inputs)==6 and set(by)==set(keys),'PVP_all_six_bound_inputs_no_survivor_selection')
 rt=runtime if runtime is not None else Runtime(campaign)
 need(all(x.synthetic==rt.synthetic for x in inputs),'PVP_same_real_synthetic_scope')
 result={'schema':RESULT_SCHEMA,'scope':SCOPE,'shard_id':plan['shard_id'],'synthetic':bool(rt.synthetic),'complete':False,'qualification_complete':False,
 'status':'FAILED_PRESERVED','planned_queries':plan['planned_queries'],'charged_entries':[],'completed_entries':[],'passed_entries':[],
 'prefix_prefills_reserved':0,'suffix_evaluations_reserved':0,'environment_steps':0,'new_fit':False,'training_calls':0,'stock_equivalence_claim':False,
 'new_behavior_or_independent_holdout_claim':False,'qualification_scope':'local_noedit_repeat_and_both_schedule_own_nulls_before_effects','failures':[]}
 contexts={};snapshots={};phase='source_guard';active=None
 def check():need(time.time()<deadline_epoch,'PVP_absolute_science_deadline')
 def save(event,metadata,arrays=None):
  check();return persist(event,metadata,{}if arrays is None else arrays)
 try:
  check();proof=source_guard(campaign,inputs,plan,rt);need(proof.get('model_bound')is True and proof.get('source_bound')is True and proof.get('inputs_bound')is True,'PVP_actual_model_source_inputs_guard')
  save('PVP_PLANNED',{'plan':plan,'proof':proof,'synthetic':rt.synthetic})
  for saved in inputs:
   check();branches=rt.encode(saved);contexts[saved.key]={'saved':saved,'branches':branches,'source_identity':I.identity((branches,saved.noise,saved.modified_v)),'refs':{}}
   contexts[saved.key]['input_ref']=save('PVP_INPUT_VERIFIED',{'root_id':saved.root_id,'query_index':saved.query_index,'sources':saved.sources,'original_source_identity':saved.source_identity,'encoded_identities':saved.encoded_identities},
    {'noise':saved.noise,'consumed_decode_state':saved.decode_state,'Vmod':saved.modified_v,'actual_E':saved.actual_E})
  for entry in plan['entries']:
   check();active=entry;phase=entry['phase'];key=(entry['root_id'],entry['query_index']);ctx=contexts[key];saved=ctx['saved'];role=entry['role']
   vmod=saved.old_VT if role.startswith('V_own_null')else saved.modified_v if role.startswith('shared')else np.zeros_like(saved.old_VT)
   checkpoints={'before_compute':rng()};old=I.identity((ctx['branches'],saved.noise,vmod));charge(copy.deepcopy(entry));result['charged_entries'].append(entry['call_id']);result['prefix_prefills_reserved']+=2;result['suffix_evaluations_reserved']+=10
   raw,metadata=rt.compute(ctx['branches'],saved,vmod,entry['schedule']);rt.wait((raw,metadata));checkpoints['after_compute']=rng()
   adapter=rt.decode(raw,ctx['branches']);rt.wait(adapter);checkpoints['after_decode']=rng();actual={'raw32':np.asarray(raw),'adapter7':np.asarray(adapter),'metadata':host(metadata)};m=actual['metadata']
   # Preserve exact generated output before native/identity qualification.
   if 'pg_refs'not in ctx:
    ctx['pg_refs']={side:save('PVP_QUERY_PG',{**entry,'condition':side,'graph':'Gstar','noise_independent_prefix':True},{'pg':m['pg'][side]})for side in('C','T')};ctx['pg']=m['pg']
   pg_equal=tree_exact(ctx['pg'],m['pg']);call_pg_refs=ctx['pg_refs']
   if not pg_equal:
    call_pg_refs={side:save('PVP_UNEXPECTED_PG',{**entry,'condition':side,'original_PG_ref':ctx['pg_refs'][side],'qualified':False},{'pg':m['pg'][side]})for side in('C','T')}
   used=[]
   for ji,j in enumerate(Q.CALLS):
    for si,li in enumerate(Q.LAYERS):
     for name in('prefix_key_used','prefix_value_used'):
      used.append({'flow_call':j,'layer_index':li,'field':name,'actual_identity':aid(m['reads'][name][ji,si]),'expected_source':'archived_shared_Vmod'if name=='prefix_value_used'and li==8 and entry['flow_mask'][j]and role.startswith('shared')else'PG_T_cache','PG_ref':call_pg_refs['T'],'input_ref':ctx['input_ref']})
   arrays={k:v for k,v in m.items()if k not in('pg','reads','Vmod_argument')};arrays['reads']={k:v for k,v in m['reads'].items()if k not in('prefix_key_used','prefix_value_used')};arrays.update(raw32=actual['raw32'],adapter7=actual['adapter7'])
   ctx['refs'][role]=save('PVP_CALL_OUTPUT',{**entry,'graph':'Gstar','PG_refs':call_pg_refs,'input_ref':ctx['input_ref'],'actual_prefix_used':used,'Vmod_actual_identity':aid(m['Vmod_argument']),'input_identity':ctx['source_identity']},arrays)
   save('PVP_PREFIX_IDENTITY_COMPARISON',{**entry,'PG_refs':call_pg_refs,**I.prefix_comparison(saved,m)})
   checkpoints['after_persist']=rng();result['completed_entries'].append(entry['call_id'])
   Q.check_result(m);I.current_prefix(saved,m)
   need(actual['raw32'].shape==(1,10,32)and actual['raw32'].dtype==np.float32 and actual['adapter7'].shape==(1,10,7)and actual['adapter7'].dtype==np.float64 and np.isfinite(actual['raw32']).all()and np.isfinite(actual['adapter7']).all(),'PVP_original_endpoint_ABI')
   need(pg_equal,'PVP_same_actual_PG_before_reference_reuse')
   need(I.exact(m['Vmod_argument'],vmod),'PVP_actual_Vmod_argument')
   need(I.exact(m['flow']['x'][0],saved.noise),'PVP_actual_original_initial_noise')
   need(old==I.identity((ctx['branches'],saved.noise,vmod)),'PVP_no_input_mutation')
   need(all(x['numpy']==checkpoints['before_compute']['numpy']for x in checkpoints.values()),'PVP_NumPy_RNG_mutation')
   own=Q.selectors(entry['schedule']);need(all(I.exact(m['selectors'][name],v)for name,v in zip(('kv_mask','layer_mask','flow_mask','use_vmod'),own)),'PVP_actual_dynamic_selector')
   if role=='baseline_none_1':ctx['baseline']=actual
   elif role in('baseline_none_2','V_own_null_pulse','V_own_null_persistent'):need(result_exact(ctx['baseline'],actual),'PVP_actual_repeat_or_schedule_null_exact')
   elif role in('shared_pulse','shared_persistent'):
    ctx[role]=actual
    if role=='shared_persistent':pulse_pair(ctx['shared_pulse'],actual)
   else:need(result_exact(ctx[role.removesuffix('_repeat')],actual),'PVP_effect_delayed_repeat_exact')
   checkpoints['checks']=rng();save('PVP_CALL_CHECKS',{**entry,'passed':True,'input_preserved':True,'source_same_old_VT_KT_E':True,'full_or_resume':True})
   save('PVP_CALL_RNG',{**entry,'state_hashes':checkpoints,'Python_policy':'segmented_diagnostic_only','NumPy_policy':'strict_unchanged'})
   need(rng()['numpy']==checkpoints['before_compute']['numpy'],'PVP_NumPy_RNG_mutation_after_final_persistence')
   result['passed_entries'].append(entry['call_id'])
   if phase=='qualification'and len(result['passed_entries'])==24:result['qualification_complete']=True
  result.update(complete=True,qualification_complete=True,status='synthetic_PVP_COMPLETE'if rt.synthetic else'PVP_native_qualification_PASS')
 except Exception as e:
  msg=str(e);code=msg if len(msg)<=200 and all(c.isalnum()or c in'_:.-'for c in msg)else None
  frames=[];tb=e.__traceback__
  while tb is not None:
   frames.append({'file':Path(tb.tb_frame.f_code.co_filename).name,'function':tb.tb_frame.f_code.co_name,'line':tb.tb_lineno});tb=tb.tb_next
  result['failures'].append({'phase':phase,'call_id':None if active is None else active['call_id'],'exception_type':type(e).__name__,'code':code,'frames':frames[-10:]})
 finally:
  result['qualification_barrier']='all24_local_qualification_calls_before_local_effects_no_cross_GPU_GO'
  result['query_status']=[]
  passed=set(result['passed_entries']);charged=set(result['charged_entries'])
  for root,q in keys:
   es=[e for e in plan['entries']if(e['root_id'],e['query_index'])==(root,q)]
   result['query_status'].append({'root_id':root,'query_index':q,'planned_calls':8,'charged_calls':sum(e['call_id']in charged for e in es),'passed_calls':sum(e['call_id']in passed for e in es),'status':'COMPLETE'if all(e['call_id']in passed for e in es)else'FAILED_OR_UNEXECUTED','unexecuted_call_ids':[e['call_id']for e in es if e['call_id']not in charged]})
  # Parent must provide a bounded terminal reserve; never extend scientific time.
  persist('PVP_RESULT',result,{})
 return result
