"""Saved-input replay ABI: no renderer, simulator, random sampling or model IO."""
from dataclasses import dataclass,fields,is_dataclass
from pathlib import Path
import copy,hashlib,importlib.util,sys
import numpy as np
HASH_SHA='7b943d9edf46fcc2ea53f34d2e65cbeba6167eb1f03b4c712b956cfbd0c16c52'
def need(ok,code):
 if not ok:raise ValueError(code)
def sha(b):return hashlib.sha256(b).hexdigest()
def exact(a,b):
 a,b=np.asarray(a),np.asarray(b);return a.shape==b.shape and a.dtype==b.dtype and a.tobytes()==b.tobytes()
def explicit(x):
 if is_dataclass(x):return {f.name:explicit(getattr(x,f.name))for f in fields(x)}
 if isinstance(x,dict):return {k:explicit(v)for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [explicit(v)for v in x]
 return np.asarray(x)if hasattr(x,'dtype')else x

def hasher():
 p=Path(__file__).with_name('retained_input_hash.py')
 if not p.is_file():p=Path(__file__).resolve().parents[4]/'continuous/l9-intervention-worker-v1/retained_input_hash.py'
 need(sha(p.read_bytes())==HASH_SHA,'PVP_original_input_hash_source')
 name='_PVP_original_input_identity'
 if name not in sys.modules:
  s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m)
 return sys.modules[name]
def identity(x):return hasher().tree_id(explicit(x))
def value(x):
 if hasattr(x,'storage')and hasattr(x,'descriptor'):
  a=x.storage
  if x.dtype=='bfloat16':
   import ml_dtypes
   return np.array(a.view(ml_dtypes.bfloat16),copy=True)
  return np.array(a,copy=True)
 if isinstance(x,dict):return {k:value(v)for k,v in x.items()}
 if isinstance(x,list):return [value(v)for v in x]
 return x

def ref(record):return {k:record.ref[k]for k in('path','bytes','sha256')}
@dataclass(frozen=True)
class Branch:
 observation:object
 decode_states:object
@dataclass(frozen=True)
class SavedInput:
 root_id:str
 query_index:int
 parent_condition:str
 raw_observations:dict
 prompt:str
 noise:np.ndarray
 decode_state:np.ndarray
 encoded_identities:dict
 old_KT:np.ndarray
 old_VT:np.ndarray
 prefix_mask:np.ndarray
 prefix_positions:np.ndarray
 modified_v:np.ndarray
 actual_E:np.ndarray
 sources:dict
 source_identity:str
 synthetic:bool=False
 @property
 def key(self):return self.root_id,self.query_index


def from_records(*,root_runtime,root_input,prompt_binding,pair,clock,binding,edits,old_natural_T,synthetic=False):
 records=(root_runtime,root_input,prompt_binding,pair,clock,binding,edits,old_natural_T)
 need(all(r.synthetic==synthetic for r in records),'PVP_same_real_or_synthetic_records')
 for r,event in zip(records,('ROOT_RUNTIME_BINDING','ROOT_INPUT_BINDING','ACTUAL_PROMPT_BINDING','PAIRED_INPUT','QUERY_REACHED','CURRENT_GPU_INPUT_BINDING','FROZEN_QUERY_EDITS','CALL_OUTPUT')):need(r.event==event,'PVP_actual_event_'+event)
 bm=binding.metadata;root,q=bm['root_id'],bm['query_index'];source=bm['source'];parent=source['parent_condition']
 need(q in(11,16,22)and parent in('C','T'),'PVP_fixed_original_anchor')
 for r in records:need(r.metadata['root_id']==root,'PVP_original_root_binding')
 for r in(pair,clock,binding,edits,old_natural_T):need(r.metadata['query_index']==q,'PVP_original_query_binding')
 need(source['same_state_pair']is True and source['restored_parent_condition']==parent and source['env_step_index']==5*q,'PVP_original_same_state_pair')
 for r in(pair,clock):need(r.metadata['env_step_index']==5*q and r.metadata['absolute_env_step_index']==10+5*q,'PVP_original_physical_clock')
 need(exact(value(pair.field('query_time_actual')),value(clock.field('query_time_actual'))),'PVP_original_clock_payload')
 proof=source['actual_pair_proof'];need(all(k in pair.metadata and pair.metadata[k]==v for k,v in proof.items()),'PVP_original_pair_proof_not_filename')
 prompt=root_runtime.metadata['actual_prompt'];need(type(prompt)is str and prompt==prompt_binding.metadata['actual_prompt'],'PVP_original_prompt_text')
 ps=sha(prompt.encode());need(prompt_binding.metadata['actual_prompt_utf8_sha256']==ps==root_input.metadata['actual_prompt_sha256'],'PVP_original_prompt_SHA')
 need(root_input.metadata['parent_condition']==parent,'PVP_original_parent_condition')
 views=pair.arrays['views'];need(set(views)=={'clean','nativeperturb'},'PVP_scientific_C_T_roles')
 raw={side:value(views[name]['observation'])for side,name in(('C','clean'),('T','nativeperturb'))}
 for side in('C','T'):need({'agentview_image','robot0_eye_in_hand_image','robot0_eef_pos','robot0_eef_quat','robot0_gripper_qpos'}<=set(raw[side]),'PVP_complete_original_raw_observation')
 m=old_natural_T.metadata;need(m['kind']=='common'and m['role']=='natural_T'and m['receiver']=='T'and m['input_source']==source['source_identity'],'PVP_original_natural_T_common')
 noise=value(binding.field('noise'));state=value(binding.field('consumed_decode_state'));need(noise.shape==(1,10,32)and noise.dtype==np.float32 and state.shape==(1,32)and state.dtype==np.float32,'PVP_original_noise_and_decode_ABI')
 need(exact(noise,value(old_natural_T.field('flow','x'))[0]),'PVP_original_diagnostic_noise_payload')
 kt,vt=(value(old_natural_T.field(k))for k in('receiver_L9_K','receiver_L9_V'));vm=value(edits.field('Vmod','shared'));e=value(edits.field('actual_E','shared'))
 mask=value(old_natural_T.field('prefix_mask'));pos=value(old_natural_T.field('prefix_positions'))
 need(kt.shape==vt.shape==vm.shape and vt.shape==(1,mask.shape[1],1,256)and all(x.dtype.name=='bfloat16'for x in(kt,vt,vm)),'PVP_original_native_K_V_Vmod_ABI')
 need(e.shape==vt.shape and e.dtype==np.float32 and exact(vm.astype(np.float32)-vt.astype(np.float32),e),'PVP_original_actual_E_not_ideal_projection')
 need(mask.dtype==bool and pos.dtype==np.int32 and exact(pos,np.cumsum(mask,axis=-1,dtype=np.int32)-1),'PVP_original_prefix_layout')
 need(exact(vm[:,~mask[0]],vt[:,~mask[0]]),'PVP_original_padding_unchanged')
 sources={r.event if r is not old_natural_T else'natural_T_common':ref(r)for r in records}
 result=SavedInput(root,q,parent,raw,prompt,noise,state,{s:bm[s]for s in('C','T')},kt,vt,mask,pos,vm,e,sources,source['source_identity'],synthetic)
 for a in(noise,state,kt,vt,mask,pos,vm,e):a.setflags(write=False)
 return result

def encode_loaded(campaign,saved):
 before=identity(saved.raw_observations);rng=identity(np.random.get_state());branches={}
 for side in('C','T'):
  obs,batch=campaign.encode([copy.deepcopy(saved.raw_observations[side])],[saved.prompt]);branches[side]=Branch(obs,batch['state'])
  need(identity(branches[side])==saved.encoded_identities[side],'PVP_original_GPU_encoded_'+side)
  need(exact(np.asarray(batch['state']),saved.decode_state),'PVP_original_consumed_decode_state_'+side)
 need(before==identity(saved.raw_observations)and rng==identity(np.random.get_state()),'PVP_encode_no_input_or_NumPy_mutation')
 return branches

def current_prefix(saved,metadata):
 pg=metadata['pg']['T']
 need(exact(np.asarray(pg['cache_k'])[1],saved.old_KT)and exact(np.asarray(pg['cache_v'])[1],saved.old_VT),'PVP_same_original_KT_VT_current_runtime')
 need(exact(metadata['prefix_mask'],saved.prefix_mask)and exact(metadata['prefix_positions'],saved.prefix_positions),'PVP_same_original_addresses')
 need(exact(saved.modified_v.astype(np.float32)-np.asarray(pg['cache_v'])[1].astype(np.float32),saved.actual_E),'PVP_same_archived_actual_E')
 return True

def prefix_comparison(saved,metadata):
 """Small evidence before the unchanged gate; never diagnoses bad source data."""
 def ident(a):
  a=np.asarray(a);return {'shape':list(a.shape),'logical_dtype':a.dtype.name,'storage_dtype':a.dtype.str,'bytes':a.nbytes,'payload_sha256':sha(a.tobytes())}
 def compare(old,new):
  a,b=np.asarray(old),np.asarray(new);same_shape=a.shape==b.shape
  da=a.astype(np.float32);db=b.astype(np.float32)
  return {'old':ident(a),'new':ident(b),'exact':exact(a,b),'different_elements':int(np.count_nonzero(a!=b))if same_shape else None,
   'max_abs_difference_FP32':float(np.max(np.abs(da-db)))if same_shape else None}
 pg=metadata['pg']['T'];vt=np.asarray(pg['cache_v'])[1];kt=np.asarray(pg['cache_k'])[1]
 e=saved.modified_v.astype(np.float32)-vt.astype(np.float32)if vt.shape==saved.modified_v.shape else np.empty(0,np.float32)
 return {'encoded_inputs_already_verified':True,'original_common_ref':saved.sources['natural_T_common'],
 'KT':compare(saved.old_KT,kt),'VT':compare(saved.old_VT,vt),'actual_E':compare(saved.actual_E,e),
 'classification':'current_compiled_prefix_numeric_identity_check','mismatch_is_not_proof_of_corrupt_original_input':True,
 'graph_or_device_cause_identified':False,'old_and_new_arrays_retained_by_original_ref_and_PG_ref':True}
