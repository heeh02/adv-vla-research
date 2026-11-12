"""Isolated Plus q0 replay contract; no model construction, env step or cloud API."""
import argparse,hashlib,json,math,sys
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
LAYERS=(0,8,17);CALLS=(0,5,9)
ROLES={'baseline':'canonical_native_sampler','trajectory':'independent_native_trajectory_candidate',
       'prefix_false':'instrumented_prefix_replay_false','prefix_true':'instrumented_prefix_replay_true',
       'O':'instrumented_shadow_O','C':'independent_canonical_suffix_shadow_on_O_prefix_cache'}
class ContractError(ValueError):pass
def need(ok,code):
    if not ok:raise ContractError(code)
def array_id(value):
    a=np.asarray(value);need(not a.dtype.hasobject,'object_array_forbidden')
    h={'shape':list(a.shape),'dtype':a.dtype.str,'logical_dtype':a.dtype.name,'bytes':a.nbytes}
    h['sha256']=hashlib.sha256(json.dumps(h,sort_keys=True).encode()+b'\0'+a.tobytes(order='C')).hexdigest();return h
def tree_id(value):
    def convert(v):
        if isinstance(v,dict):return {str(k):convert(x)for k,x in sorted(v.items())}
        if isinstance(v,(tuple,list)):return [convert(x)for x in v]
        if isinstance(v,(str,int,float,bool,type(None))):return v
        return array_id(v)
    return hashlib.sha256(json.dumps(convert(value),sort_keys=True,allow_nan=False).encode()).hexdigest()
def exact(a,b):return array_id(a)==array_id(b)
def measure(reference,candidate):
    r,c=np.asarray(reference),np.asarray(candidate);need(r.shape==c.shape and r.dtype==c.dtype,'comparison_ABI')
    need(r.size>0 and np.isfinite(r).all()and np.isfinite(c).all(),'nonfinite_or_empty')
    # High-dimensional diagnostic arrays stay FP32; only scalar sums are wider.
    d=c.astype(np.float32)-r.astype(np.float32)
    energy=math.fsum(float(x)**2 for x in d.flat);base=math.fsum(float(x)**2 for x in r.flat)
    return {'typed_exact':exact(r,c),'max_abs_FP32':float(np.max(np.abs(d))),
            'rms_FP32_reference':math.sqrt(energy/r.size),'relative_l2':math.sqrt(energy/base)if base else None,
            'zero_reference_energy':base==0,'reference':array_id(r),'candidate':array_id(c)}
def hard_exact(a,b,code):need(exact(a,b),code)
def verify_pins(manifest=None,root=None):
    doc=json.loads(Path(manifest or HERE/'SOURCE_PINS-v2.json').read_text())
    root=Path(root)if root is not None else next(p for p in HERE.parents if(p/'AGENTS.md').is_file())
    for r in doc['upstream']+doc.get('candidate',[]):
        p=root/r['path'];b=p.read_bytes();need(len(b)==r['bytes']and hashlib.sha256(b).hexdigest()==r['sha256'],'source_pin:'+r['path'])
    return doc
def finite(values,code):
    for value in values.values():
        if isinstance(value,dict):finite(value,code)
        else:
            a=np.asarray(value);need(a.size>0 and not a.dtype.hasobject and np.isfinite(a).all(),code)
def action_check(r):
    for field,shape,dtype in(('raw_actions',(1,10,32),np.float32),('actions',(1,10,7),np.float64)):
        a=np.asarray(r[field]);need(a.shape==shape and a.dtype==dtype,'action_ABI_'+field)
    finite({k:r[k]for k in('raw_actions','actions')},'nonfinite_actions')
class Writer:
    """One private local output; native arrays plus identities, no inferred fields."""
    def __init__(self,path):self.path=Path(path);self.path.mkdir(parents=True,exist_ok=False);self.files=[]
    def json(self,name,value):
        p=self.path/name;p.parent.mkdir(parents=True,exist_ok=True)
        with p.open('x')as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')
    def arrays(self,name,values,role):
        need(values and all('/'not in k and '..'not in k for k in values),'flat_explicit_field_names')
        data={};schema={}
        for k,v in values.items():
            a=np.asarray(v);proof=array_id(a)
            data[k]=a.view(np.uint16)if a.dtype.name=='bfloat16'else a
            schema[k]={**proof,'stored_dtype':data[k].dtype.str,'finite':bool(np.isfinite(a).all())}
        p=self.path/name;p.parent.mkdir(parents=True,exist_ok=True)
        with p.open('xb')as f:np.savez(f,**data,__schema__=np.asarray(json.dumps({'role':role,'fields':schema},sort_keys=True)))
        b=p.read_bytes();rec={'path':name,'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest(),'role':role,'fields':schema}
        self.files.append(rec);return rec

def prefix_arrays(p,full_pp):
    values={k:p[k]for k in('prefix_valid_mask','prefix_ar_mask')}
    # The unchanged False prepare already returns native-dtype H/Z from this
    # replay graph. Preserve them independently of the quadratic PP opt-in.
    values.update({'pg_'+k:p['pg'][k]for k in('hidden_input','hidden_pre_attention_norm','cache_k','cache_v','positions','mask','query_defined','key_visible','selected_layers')if k in p['pg']})
    values.update({'span_'+k:v for k,v in p['prefix_spans'].items()})
    if full_pp:values.update({'pg_probabilities':p['pg']['probabilities']}if 'probabilities'in p['pg']else{})
    return values

def capture_arrays(r):
    values={k:r[k]for k in('actual_x','actual_time','adarms_condition','suffix_valid_mask','suffix_ar_mask','suffix_tokens','suffix_positions','suffix_to_all_key_mask','final_readout','velocity','canonical_readout','canonical_velocity')if k in r}
    values.update({'O_'+k:v for k,v in r.get('native_trace',{}).items()})
    return values

def prepare_check(p,layout,full_pp):
    need(p['role']==ROLES['prefix_true'if full_pp else'prefix_false'],'prefix_role')
    cache=p['cache'];need(isinstance(cache,tuple)and len(cache)==2,'actual_KV_tuple')
    n=np.asarray(p['prefix_valid_mask']).shape[-1];a=layout['A'];d=layout['D'];h=layout['H'];kd=layout['dh']
    valid=np.asarray(p['prefix_valid_mask']);ar=np.asarray(p['prefix_ar_mask'])
    need(n>0 and valid.shape==(1,n)and valid.dtype==np.bool_ and ar.shape==(n,)and ar.dtype==np.bool_,'dynamic_prefix_mask_ABI')
    for x in cache:need(np.asarray(x).shape==(18,1,n,layout['K'],kd),'dynamic_full_cache_ABI')
    pg=p['pg'];hard_exact(pg['selected_layers'],np.array(LAYERS,np.int32),'selected_layers')
    need(all(k in pg for k in('hidden_input','hidden_pre_attention_norm')),'actual_PG_H_Z_missing')
    native_dtype=np.asarray(cache[0]).dtype
    need(native_dtype.name in('bfloat16','float16','float32')and np.asarray(cache[1]).dtype==native_dtype,'native_prefix_dtype')
    for field in('hidden_input','hidden_pre_attention_norm'):
        value=np.asarray(pg[field])
        need(value.shape==(len(LAYERS),1,n,layout['PG_D']),'actual_PG_H_Z_shape_'+field)
        need(value.dtype==native_dtype,'actual_PG_H_Z_dtype_'+field)
    hard_exact(pg['cache_k'],np.asarray(cache[0])[list(LAYERS)],'selected_K_not_used_cache')
    hard_exact(pg['cache_v'],np.asarray(cache[1])[list(LAYERS)],'selected_V_not_used_cache')
    need(np.asarray(pg['mask']).shape==(1,n,n)and np.asarray(pg['mask']).dtype==np.bool_,'actual_prefix_mask')
    need(np.asarray(pg['positions']).shape==(1,n),'actual_prefix_positions')
    hard_exact(pg['query_defined'],np.asarray(pg['mask']).any(-1),'prefix_query_defined')
    hard_exact(pg['key_visible'],np.asarray(pg['mask']).any(-2),'prefix_key_visible')
    need(set(p['prefix_spans']) and 'all_prefix_tokens'in p['prefix_spans'],'actual_prefix_spans')
    hard_exact(p['prefix_spans']['all_prefix_tokens'],np.array([0,n],np.int32),'actual_N_span')
    for span in p['prefix_spans'].values():
        s=np.asarray(span);need(s.shape==(2,)and s.dtype.kind in 'iu'and 0<=int(s[0])<=int(s[1])<=n,'span_bounds')
    # These are definitions from the pinned make_attn_mask/prefix producer,
    # not inferred lengths or a fixed camera/token whitelist.
    groups=np.cumsum(ar)
    expected=(groups[None,:]<=groups[:,None])[None]&valid[:,None,:]&valid[:,:,None]
    hard_exact(pg['mask'],expected,'prefix_mask_not_actual_valid_ar')
    hard_exact(pg['positions'],np.cumsum(valid,axis=1,dtype=np.int32)-1,'prefix_positions_not_actual_valid')
    if full_pp:
        need('probabilities'in pg and np.asarray(pg['probabilities']).shape==(3,1,layout['K'],h//layout['K'],n,n),'actual_PP_missing_or_shape')
        need(np.asarray(pg['probabilities']).dtype==native_dtype,'actual_PP_dtype')
    finite(prefix_arrays(p,full_pp),'nonfinite_prefix')
    for value in cache:finite({'cache':value},'nonfinite_full_cache')
    return n,tree_id(cache)

def trajectory_check(t,noise,layout):
    need(t['role']==ROLES['trajectory']and t['T_prefix_cache_identity']is None,'trajectory_role_or_unexposed_cache')
    s=t['snapshot'];a,d=layout['A'],layout['action_dim']
    need(set(('call_indices','x_t','velocity','readout','hidden','time','final_actions'))<=set(s),'missing_T_core_field')
    hard_exact(s['call_indices'],np.arange(10,dtype=np.int32),'all10_original_indices')
    need(t['index_basis']=='pinned_native_trajectory_scan_axis_length10','native_scan_index_provenance')
    need(np.asarray(s['x_t']).shape==(10,1,a,d)and np.asarray(s['velocity']).shape==(10,1,a,d),'fullstep_x_v')
    need(np.asarray(s['readout']).shape==(10,1,a,layout['D'])and np.asarray(s['hidden']).shape==(10,3,1,a,layout['D']),'native_fullstep_H_R')
    ts=np.asarray(s['time'])
    need(ts.shape==(10,)and np.isfinite(ts).all()and ts[0]==1 and np.all(np.diff(ts)<0),'actual_native_time')
    hard_exact(s['x_t'][0],noise,'initial_noise_changed')
    hard_exact(s['final_actions'],t['raw_actions'],'final_boundary_output')
    need(t['native_H_status']=='candidate_native_H_requires_exact_final'and t['T_loop_time_weak_type']is None,'native_H_and_time_provenance')
    finite(s,'nonfinite_T')

def capture_check(c,o,p,x,time,layout,cache_id):
    need(c['role']==ROLES['C']and o['role']==ROLES['O'],'capture_role')
    need(c['cache_id']==o['cache_id']==cache_id,'cross_cache_capture')
    for r in(c,o):
        need(r['shadow_time_weak_type']is False,'shadow_time_must_be_strong_typed')
        hard_exact(r['actual_x'],x,'capture_actual_x');hard_exact(r['actual_time'],time,'capture_actual_time')
        finite(capture_arrays(r),'nonfinite_capture')
    tr=o['native_trace'];a,d,h,n=layout['A'],layout['D'],layout['H'],np.asarray(p['prefix_valid_mask']).shape[-1]
    need(set(('hidden_post_block','probabilities_AP_AA','attention_output_post_WO_pre_gate'))<=set(tr),'missing_O_core_field')
    for k in('hidden_post_block','attention_output_post_WO_pre_gate'):need(np.asarray(tr[k]).shape==(3,1,a,d),'O_shape_'+k)
    need(np.asarray(tr['probabilities_AP_AA']).shape==(3,1,layout['K'],h//layout['K'],a,n+a),'dynamic_P_shape')
    hard_exact(tr['selected_layers'],np.array(LAYERS,np.int32),'O_selected_layers')
    for k in('adarms_condition','suffix_tokens','suffix_valid_mask','suffix_ar_mask'):hard_exact(c[k],o[k],'CO_input_'+k)
    for k in('suffix_positions','suffix_to_all_key_mask'):hard_exact(c[k],tr[k],'CO_input_'+k)
    mask=np.asarray(tr['suffix_to_all_key_mask']);need(mask.shape==(1,a,n+a)and mask.dtype==np.bool_,'full_AP_AA_mask')
    valid,ar=np.asarray(o['suffix_valid_mask']),np.asarray(o['suffix_ar_mask'])
    need(valid.shape==(1,a)and valid.dtype==np.bool_ and ar.shape==(a,)and ar.dtype==np.bool_,'suffix_mask_ABI')
    groups=np.cumsum(ar)
    suffix_mask=(groups[None,:]<=groups[:,None])[None]&valid[:,None,:]&valid[:,:,None]
    expected=np.concatenate([np.repeat(np.asarray(p['prefix_valid_mask'])[:,None,:],a,axis=1),suffix_mask],axis=-1)
    hard_exact(mask,expected,'suffix_mask_not_prepared_prefix')
    positions=np.asarray(p['prefix_valid_mask']).sum(axis=-1,dtype=np.int32)[:,None]+np.cumsum(valid,axis=-1,dtype=np.int32)-1
    hard_exact(tr['suffix_positions'],positions,'suffix_positions_not_prepared_prefix')
    hard_exact(tr['query_defined'],mask.any(-1),'query_defined');hard_exact(tr['key_visible'],mask.any(-2),'key_visible')
    hard_exact(o['final_readout'],o['canonical_readout'],'O_vs_own_control_R')
    hard_exact(o['velocity'],o['canonical_velocity'],'O_vs_own_control_v')
    # Independent C/O is measured, never silently set to zero or gated exact.
    return {k:measure(c['canonical_'+k],o[v])for k,v in(('readout','final_readout'),('velocity','velocity'))}

def request_binding(branch):
    return tree_id({'observation':branch['observation'],'prompt':branch['prompt']})

def prepared_binding(p):
    return tree_id({k:p[k]for k in('cache','pg','prefix_valid_mask','prefix_ar_mask','prefix_spans','model_inputs','model_input_absent','request_binding')})

def run_pair(pair,backend,out,*,synthetic=False,full_pp=False):
    need(synthetic or sys.platform.startswith('linux'),'real_neural_recording_Linux_only')
    need(type(full_pp)is bool,'explicit_full_PP_flag')
    need(type(pair['query'])is int and pair['query']==0 and pair['environment_pair_status']=='externally_qualified'and pair['pair_binding_id'],'external_q0_pair_contract')
    need(set(pair['branches'])=={'reference','variant'},'two_branches_only')
    branches=pair['branches'];ref,var=branches['reference'],branches['variant']
    need(type(ref['query'])is int and type(var['query'])is int and ref['query']==var['query']==0,'same_query_clock')
    hard_exact(ref['physical_state'],var['physical_state'],'same_q_different_physical_state')
    need(ref['goal_binding_id']==var['goal_binding_id']==pair['goal_binding_id'],'external_goal_binding')
    noise=np.asarray(pair['noise']);need(noise.shape==(1,10,32)and noise.dtype==np.float32,'explicit_FP32_10x32_noise')
    finite({'noise':noise},'nonfinite_noise')
    layout=backend.layout
    if not synthetic:need(layout=={'A':10,'action_dim':32,'D':1024,'PG_D':2048,'H':8,'K':1,'dh':256},'real_pi05_layout')
    initial=tree_id(pair);source=backend.verify_sources();runtime_binding=tree_id(backend.runtime_identity);writer=Writer(out)
    report={'pair_id':pair['pair_id'],'query':0,'synthetic':synthetic,'real_numerical_qualification':False,
            'input_binding':initial,'source_binding':source,'runtime_identity':backend.runtime_identity,
            'branches':{},'segmentation':{'status':'unmeasured_optional','source_mask':None},'controls_issued':0,
            'role':'isolated_M_replay_not_closed_loop_Y','canonical_Y_action_source':ROLES['baseline'],
            'unavailable_fields':['attention_gate','suffix_V','actual_gated_residual_write'],
            'W_scope':'actual_O_graph_post_WO_pre_gate_branch;not_total_W_exhaustiveness',
            'PG_recording':{'enabled_every_q0':True,'selected_layers_zero_based':list(LAYERS),'width':layout['PG_D'],
                'hidden_input_site':'pre_block_residual_input',
                'hidden_pre_attention_norm_site':'post_pre_attention_RMSNorm_actual_input_to_Attention',
                'recording_role':'instrumented_prefix_replay_not_native_executed',
                'dtype':'actual_prefix_cache_native_dtype_no_cast','post_block_H':'not_requested',
                'full_PP_requested':full_pp,'extra_prefix_or_model_calls_for_H_Z':0},
            'full_generation_order':['C0','C1','T0','T1','C2'],'source_call_count':0}
    writer.json('INPUT-RECEIPT-v1.json',{'pair_id':pair['pair_id'],'pair_binding_id':pair['pair_binding_id'],
        'goal_binding_id':pair['goal_binding_id'],'input_binding':initial,'source_binding':source,
        'prompts':{name:b['prompt']for name,b in branches.items()},'external_pair_qualification_not_reimplemented':True})
    def call(method,*args,**kwargs):
        need(backend.verify_sources()==source,'source_identity_changed')
        need(tree_id(backend.runtime_identity)==runtime_binding,'runtime_identity_changed')
        rng=tree_id(np.random.get_state())
        before=tree_id(pair);value=method(*args,**kwargs);report['source_call_count']+=1
        need(tree_id(pair)==before==initial,'input_or_noise_mutated')
        need(tree_id(np.random.get_state())==rng,'sensor_numpy_rng_mutated')
        need(tree_id(backend.runtime_identity)==runtime_binding,'runtime_identity_changed')
        need(backend.verify_sources()==source,'source_identity_changed')
        return value
    def save_actions(name,label,r):
        writer.arrays(name+'/'+label+'.npz',{k:r[k]for k in('raw_actions','actions')if k in r},r.get('role','unknown'))
    measured={}
    try:
        for name,b in branches.items():
            writer.arrays(name+'/input.npz',{**{'observation_'+k:v for k,v in b['observation'].items()},'physical_state':b['physical_state'],'noise':noise},'passive_input_or_explicit_counterfactual')
            need(type(b['prompt'])is str and bool(b['prompt']),'explicit_prompt_required')
            finite({'observation':b['observation'],'physical_state':b['physical_state']},'nonfinite_input')
            base=call(backend.baseline,b,noise);save_actions(name,'C0',base)
            repeat=call(backend.baseline,b,noise);save_actions(name,'C1',repeat)
            need(base['role']==repeat['role']==ROLES['baseline'],'baseline_role')
            for r in(base,repeat):action_check(r)
            for field in('raw_actions','actions'):hard_exact(base[field],repeat[field],'baseline_repeat_'+field)
            if b['passive_output']is not None:
                need(b['passive_output_status']=='recorded_Y','passive_output_role')
                save_actions(name,'passive-Y',dict(b['passive_output'],role=ROLES['baseline']))
                for field in('raw_actions','actions'):hard_exact(base[field],b['passive_output'][field],'passive_Y_replay_'+field)
            else:need(b['passive_output_status']=='counterfactual_not_previously_executed','missing_passive_output_status')
            trajectories=[]
            for k in range(2):
                t=call(backend.trajectory,b,noise);save_actions(name,'T%d-actions'%k,t)
                writer.arrays(name+'/T%d-trajectory.npz'%k,t['snapshot'],t.get('role','unknown'))
                trajectories.append(t)
            finalbase=call(backend.baseline,b,noise);save_actions(name,'C2',finalbase)
            need(finalbase['role']==ROLES['baseline'],'baseline_role');action_check(finalbase)
            for field in('raw_actions','actions'):hard_exact(base[field],finalbase[field],'baseline_after_recorder_'+field)
            for t in trajectories:
                action_check(t);trajectory_check(t,noise,layout)
                backend.require_native_actions(name,base['raw_actions'],t['raw_actions'])
                for field in('raw_actions','actions'):hard_exact(base[field],t[field],'baseline_recorder_'+field)
            t,t2=trajectories
            # T did not expose its own cache. O/C-shadow use a separately
            # prepared same-request prefix, never a fabricated T cache ID.
            prepared=[]
            for pp in ((False,True)if full_pp else(False,)):
                for k in range(2):
                    p=call(backend.prepare,b,full_pp=pp)
                    writer.arrays(name+'/prefix-%s-%d.npz'%(pp,k),prefix_arrays(p,pp),p.get('role','unknown'))
                    writer.arrays(name+'/model-inputs-%s-%d.npz'%(pp,k),p['model_inputs'],'O_prefix_actual_consumed_inputs')
                    n,cid=prepare_check(p,layout,pp)
                    need(p['request_binding']==request_binding(b),'prefix_request_binding')
                    finite(p['model_inputs'],'nonfinite_model_inputs')
                    if prepared:
                        prior=prepared[0]
                        hard_exact(prior['prefix_valid_mask'],p['prefix_valid_mask'],'prefix_repeat_mask')
                        need(tree_id(prior['model_inputs'])==tree_id(p['model_inputs']),'prefix_repeat_model_inputs')
                        need(tree_id(prior['cache'])==cid,'prefix_repeat_or_PP_full_cache')
                        need(tree_id(prior['prefix_spans'])==tree_id(p['prefix_spans']),'prefix_repeat_spans')
                    prepared.append(p)
            p=prepared[-1];pid=prepared_binding(p)
            def prepared_call(method,*args):
                need(prepared_binding(p)==pid,'prepared_input_mutated')
                v=call(method,p,*args)
                need(prepared_binding(p)==pid,'prepared_input_mutated')
                need(v['request_binding']==p['request_binding'],'capture_request_binding')
                return v
            report['branches'][name]={'N':n,'O_used_cache_identity':cid,'O_request_binding':p['request_binding'],
                'native_H_status':'T_native_H_raw32_and_adapter7_exact_gate_passed',
                'H_primary_role':'T_native_H_after_exact_raw32_and_adapter7_gate','T_prefix_cache_identity':None,
                'O_prefix_relationship_to_T':'same_request_separate_prefill;T_cache_identity_unexposed',
                'T_loop_time_weak_type':None,'model_input_absent':p['model_input_absent'],'captures':{},'passive_output_status':b['passive_output_status'],
                'T_repeat_differences':{field:measure(t['snapshot'][field],t2['snapshot'][field])for field in('x_t','time','hidden','readout','velocity')}}
            for j in CALLS:
                s=t['snapshot'];captures=[]
                for k in range(2):
                    c=prepared_call(backend.capture,'C',s,j)
                    writer.arrays(name+'/call-%02d-C%d.npz'%(j,k),capture_arrays(c),c.get('role','unknown'))
                    o=prepared_call(backend.capture,'O',s,j)
                    writer.arrays(name+'/call-%02d-O%d.npz'%(j,k),capture_arrays(o),o.get('role','unknown'))
                    diff=capture_check(c,o,p,s['x_t'][j],s['time'][j],layout,cid)
                    captures.append((c,o,diff))
                c,o,diff=captures[0];c2,o2,_=captures[1];measured[name,j]=(c,o)
                report['branches'][name]['captures'][str(j)]={
                    'C_vs_O':diff,'C_vs_O_first7_velocity':measure(c['canonical_velocity'][...,:7],o['velocity'][...,:7]),
                    'primary_call':j==0,'O_role':ROLES['O'],'canonical_intermediate_H_claim':False,
                    'C_shadow_vs_T':{k:measure(s[field][j],c['canonical_'+k])for k,field in (('readout','readout'),('velocity','velocity'))},
                    'O_shadow_vs_T':{'hidden':measure(s['hidden'][j],o['native_trace']['hidden_post_block']),'readout':measure(s['readout'][j],o['final_readout']),'velocity':measure(s['velocity'][j],o['velocity'])},
                    'shadow_repeats':{'C_R':measure(c['canonical_readout'],c2['canonical_readout']),'C_v':measure(c['canonical_velocity'],c2['canonical_velocity']),
                        'O_R':measure(o['final_readout'],o2['final_readout']),'O_v':measure(o['velocity'],o2['velocity']),
                        'O_H':measure(o['native_trace']['hidden_post_block'],o2['native_trace']['hidden_post_block'])},
                    'time_argument_role':'stored_T_time_value_strong_typed_for_shadow;T_loop_weak_type_unmeasured'}
        report['paired_effect_errors']={}
        for j in CALLS:
            r,v=measured['reference',j],measured['variant',j];report['paired_effect_errors'][str(j)]={}
            for field in('readout','velocity'):
                cd=np.asarray(v[0]['canonical_'+field]).astype(np.float32)-np.asarray(r[0]['canonical_'+field]).astype(np.float32)
                ofield='final_readout'if field=='readout'else'velocity'
                od=np.asarray(v[1][ofield]).astype(np.float32)-np.asarray(r[1][ofield]).astype(np.float32)
                report['paired_effect_errors'][str(j)][field]=measure(cd,od)
        need(tree_id(pair)==initial,'input_or_noise_mutated')
        report.update(status='synthetic_contract_PASS'if synthetic else'candidate_replay_contract_complete_requires_root_review',files=writer.files)
        writer.json('RESULT-v1.json',report);return report
    except BaseException as error:
        writer.json('FAILED-v1.json',{'status':'failed_preserved','exception_type':type(error).__name__,
            'contract_code':str(error)if isinstance(error,ContractError)else None,'files':writer.files,
            'synthetic':synthetic,'no_retry':True,'partial_report':report});raise
if __name__=='__main__':
    argparse.ArgumentParser(description=__doc__).parse_args();p=verify_pins()
    print(json.dumps({'status':'dry_source_pins_verified','upstream_files':len(p['upstream']),'model_loads':0,'cloud_calls':0,'real_numerical_qualification':False}))
