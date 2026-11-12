"""Bind original frozen predictor to CONF common records; no models or IO calls."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import hashlib,importlib.util,json,sys
import numpy as np
PREDICTOR_SHA='f1ffaf7204d0ecb114397efd0fe9fc35a03ed967da8fe588d29f3539bc258436'
FIT_ID='31ab045c8a96d4a1e5cdae7d8d41a62ca07056d6892a09ecc4ea65bf178a04e8'

def predictor():
 p=Path(__file__).with_name('taskblind_prediction.py')
 if not p.is_file():p=Path(__file__).resolve().parents[2]/'task-blind-predictor-v1/prediction.py'
 if hashlib.sha256(p.read_bytes()).hexdigest()!=PREDICTOR_SHA:raise ValueError('original_frozen_predictor_source')
 name='_CONF_diagnostic_original_predictor'
 if name not in sys.modules:
  s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m)
 return sys.modules[name]
P=predictor();need=P.need

def ref(r):return {k:r[k]for k in('path','bytes','sha256')}
def same_ref(a,b):return ref(a)==ref(b)
def native(leaf):return P.Native(leaf.storage,leaf.dtype,leaf.descriptor['payload_sha256'])
def part(leaf,index):
 a=np.asarray(leaf.storage[index]);return P.Native(a,leaf.dtype,P.sha(a.tobytes()))
def same(a,b):return a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()
def same_leaf(a,b):return a.dtype==b.dtype and same(a.storage,b.storage)
def actual_id(a):
 h={'shape':list(a.storage.shape),'logical_dtype':a.logical_dtype,'stored_dtype':'<V2'if a.logical_dtype=='bfloat16'else a.storage.dtype.str,'bytes':a.storage.nbytes}
 return {**h,'sha256':P.sha(json.dumps(h,sort_keys=True,separators=(',',':')).encode()+b'\0'+a.storage.tobytes())}

def common_roles(records,root,q,source):
 out={}
 for r in records:
  m=r.metadata
  if r.event!='CALL_OUTPUT'or m.get('kind')!='common':continue
  need(m['root_id']==root and m['query_index']==q and m['input_source']==source,'common_actual_root_query_source')
  role=m['role'];need(role not in out,'unique_actual_common_role');out[role]=r
 return out

def rebuild_query(binding,common,edits):
 need(binding.event=='CURRENT_GPU_INPUT_BINDING'and edits.event=='FROZEN_QUERY_EDITS','actual_input_and_edit_records')
 m=binding.metadata;root,q=m['root_id'],m['query_index'];source=m['source']
 need(11<=q<=22 and source['root_id']==root and source['query_index']==q and source['env_step_index']==5*q and source['same_state_pair']is True,'original_query_clock')
 for role,side in(('natural_C','C'),('natural_T','T')):
  r=common[role];x=r.metadata
  need(x['role']==role and x['receiver']==side and x['kind']=='common'and x['root_id']==root and x['query_index']==q and x['input_source']==source['source_identity'],'natural_common_role_input')
 need(edits.metadata['root_id']==root and edits.metadata['query_index']==q,'edit_actual_root_query')
 arms=P.ARMS if q in(11,16,22)else('shared',);need(edits.metadata['selected_arms']==list(arms)and set(edits.arrays['Vmod'])==set(arms)and set(edits.arrays['actual_E'])==set(arms),'sealed_full_modified_V_dictionary')
 c,t=common['natural_C'],common['natural_T']
 for name in('prefix_mask','prefix_positions'):need(same_leaf(c.field(name),t.field(name)),'same_natural_prefix_layout')
 for name in('time','time_weak_type','dt'):need(same_leaf(c.field('flow',name),t.field('flow',name)),'same_C_T_sampling_clock')
 noise=native(binding.field('noise'))
 for r in(c,t):need(part(r.field('flow','x'),0).manifest()==noise.manifest(),'actual_same_C_T_initial_noise')
 flow={side+'_'+site:part(r.field('flow',name),j)for side,r in(('C',c),('T',t))for site,name,j in(('x0','x',0),('v0','velocity_native',0),('x5','x',5),('v5','velocity_native',5))}
 identity=P.sha(P.encoded({'input_source_identity':source['source_identity'],'natural_C_receipt':ref(c.ref),'natural_T_receipt':ref(t.ref),'layer_index':8,'first_read_call':5,'flow_sites':[0,5]}))
 query=P.Query(root,q,5*q,identity,native(t.field('first_read','probabilities_native')),native(t.field('first_read','mask')),native(t.field('prefix_mask')),native(t.field('prefix_positions')),native(t.field('first_read','suffix_positions')),native(c.field('receiver_L9_V')),native(t.field('receiver_L9_V')),native(t.field('first_read','attention_gate')),flow,native(binding.field('consumed_decode_state')),part(t.field('flow','time'),[0,5]),part(t.field('flow','time_weak_type'),[0,5]),{arm:native(edits.field('Vmod',arm))for arm in arms})
 query.validate(SimpleNamespace(roots={root:None}))
 for arm in arms:
  e=native(edits.field('actual_E',arm));need(e.logical_dtype=='float32'and same(e.storage,query.edit(arm)[None,:,None,:]),'frozen_actual_E_is_BF16_payload_difference')
 return query

SITE=('query_post_rope_scaled','suffix_key_post_rope','suffix_value','receiver_residual_input','attention_gate','probabilities_native','mask','suffix_positions')
def same_first(base,other,*,compare_P):
 for name,n in(('x',6),('velocity_native',5),('time',6),('time_weak_type',6),('dt',5)):
  a,b=base.field('flow',name),other.field('flow',name);need(a.dtype==b.dtype and same(a.storage[:n],b.storage[:n]),'same_pre_j5_'+name)
 for name in SITE:
  if name=='probabilities_native'and not compare_P:continue
  need(same_leaf(base.field('first_read',name),other.field('first_read',name)),'same_first_receiver_'+name)

def bind_target(query,common,edits,label,seal):
 m=label.metadata;arm,bg=m['arm'],m['reading_context'];root,q=query.root_id,query.query_index
 need(label.event=='ACTUAL_EFFECT'and m['root_id']==root and m['query_index']==q and arm=='shared'and bg in('KT','KC'),'only_predeclared_shared_target')
 need(seal['key']==[root,q,arm,bg]and seal['available']is True and same_ref(m['edit_ref'],edits.ref),'same_sealed_key_and_original_edit_ref')
 base_role,edit_role=('natural_T','effect_shared')if bg=='KT'else('K_only','K_shared')
 base,edit=common[base_role],common[edit_role]
 need(same_ref(m['baseline'],base.ref)and same_ref(m['source'],edit.ref),'ACTUAL_EFFECT_exact_source_and_baseline')
 natural=common['natural_T'];same_first(natural,base,compare_P=bg=='KT');same_first(base,edit,compare_P=True)
 qvalue=query if bg=='KT'else replace(query,p_native=native(base.field('first_read','probabilities_native')))
 need(qvalue.fingerprint()==seal['input_identity']==m['input_identity']and qvalue.edit_identity(arm)==seal['edit_identity']==m['edit_identity'],'reconstructed_Query_matches_seal_and_actual_label')
 need(seal['natural_context_identity']==query.fingerprint(),'KC_only_P_changes_natural_context')
 need(same_ref(edit.metadata['edit_ref'],edits.ref),'effect_frozen_edit_reference')
 e=native(edits.field('actual_E',arm));vm=qvalue.modified_v[arm];actual=edit.metadata['actual_edit']
 for key,v in(('receiver_V',qvalue.treated_v),('Vmod_argument',vm),('actually_used_V_at_j5',vm),('actual_E_used_minus_receiver_FP32',e)):
  need(actual[key]==actual_id(v),'native_actual_edit_identity_'+key)
 need(bool(seal['zero_actual_E'])==(not np.any(e.storage)),'sealed_exactzero_matches_actual_E')
 return qvalue,base,edit

def load_model(artifact,arrays,read_only_wo,*,synthetic=False):
 need(artifact['schema']=='task-blind-frozen-artifact-v1'and(synthetic or(artifact['model_identity']==FIT_ID and artifact['synthetic']is False)),'original_frozen_fit_artifact')
 model=P.load_frozen(artifact['model'],arrays,expected_identity=artifact['model_identity'])
 need(read_only_wo.event=='READ_ONLY_WO','actual_WO_record');wo=native(read_only_wo.field('native_L9_WO'))
 m=read_only_wo.metadata;need(m['path']=='PaliGemma/llm/layers/attn/attn_vec_einsum_1/w'and m['layer_index']==8 and m['stacked_shape']==[18,8,256,1024]and m['model_forward_calls']==0 and m['actual_layer_identity']==actual_id(wo),'actual_readonly_WO_parameter_identity')
 need(wo.manifest()==model.wo.manifest(),'actual_common_WO_matches_frozen_predictor')
 return model
