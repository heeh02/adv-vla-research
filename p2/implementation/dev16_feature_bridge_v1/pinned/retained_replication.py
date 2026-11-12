"""Bounded cloud-local DEV replication analysis. No credentials, fit or model.

An owner supplies original files through get_local(ref); this module checks
file and typed leaf identity. Only small tables leave the CPU worker.
"""
from pathlib import Path
import csv,hashlib,io,json,re,zipfile
import numpy as np

ARMS=('shared','random_matched','distance_matched','full_v_restore')
ROOTS=('SDRU-CDEV-T00-R00','SDRU-CDEV-T01-R00')
DIMS=tuple('translation_command_'+s for s in 'xyz')+tuple('rotation_command_'+s for s in 'xyz')+('gripper_command',)
SITE_FIELDS=('query_post_rope_scaled','suffix_key_post_rope','suffix_value','receiver_residual_input','attention_gate')

def need(ok,code):
    if not ok:raise ValueError(code)
def sha(b):return hashlib.sha256(b).hexdigest()
def canonical(x):return json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def decode(x):
    if isinstance(x,dict):
        if '__sequence__'in x:return [decode(v)for v in x['values']]
        need('__array__'not in x,'metadata_contains_array');return {k:decode(v)for k,v in x.items()}
    return x
def aid(a,logical=None):
    a=np.asarray(a);return {'shape':list(a.shape),'logical_dtype':logical or a.dtype.name,
        'storage_dtype':str(a.dtype),'bytes':a.nbytes,'payload_sha256':sha(a.tobytes())}
def eq(a,b):return a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()
def native_fp32(a,logical):
    return (a.astype(np.uint32)<<16).view(np.float32)if logical=='bfloat16'else a.astype(np.float32)
def refsame(a,b):return all(a.get(k)==b.get(k)for k in('path','bytes','sha256'))


class Record:
    def __init__(self,path,ref):
        self.path=Path(path);self.ref=ref;self.z=None
        need(self.path.is_file()and not self.path.is_symlink()and self.path.stat().st_size==ref['bytes']<=64*1024**2,'bounded_exact_file')
        h=hashlib.sha256()
        with self.path.open('rb')as f:
            for b in iter(lambda:f.read(1024**2),b''):h.update(b)
        need(h.hexdigest()==ref['sha256'],'original_file_SHA')
        if self.path.suffix=='.npz':
            with zipfile.ZipFile(self.path)as z:
                need(sum(i.file_size for i in z.infolist())<=16*1024**2,'uncompressed_member_cap')
                need(z.getinfo('__schema__.npy').file_size<=2*1024**2,'schema_cap')
            self.z=np.load(self.path,allow_pickle=False)
            raw=self.z['__schema__'];need(raw.dtype==np.uint8 and raw.ndim==1,'schema_uint8');self.tree=json.loads(raw.tobytes())
        else:self.tree=json.loads(self.path.read_bytes())
        self.meta=decode(self.tree['metadata'])
        for k,v in ref.get('expected',{}).items():
            if k in('root_id','query_index','role','call_id','arm','code'):need(self.meta.get(k)==v,'original_metadata_'+k)
    def close(self):
        if self.z is not None:self.z.close()
    def leaf(self,*path):
        d=self.tree['arrays']
        for k in path:d=d[k]
        a=np.asarray(self.z[d['__array__']]);need(not a.dtype.hasobject,'no_object_dtype')
        expected={k:d[k]for k in('shape','logical_dtype','storage_dtype','bytes','payload_sha256')}
        need(aid(a,d['logical_dtype'])==expected,'typed_leaf_ABI_SHA')
        need(d['logical_dtype']==a.dtype.name or(d['logical_dtype']=='bfloat16'and a.dtype==np.uint16),'native_logical_dtype')
        values=native_fp32(a,d['logical_dtype'])if d['logical_dtype']=='bfloat16'else a
        need(np.isfinite(values).all(),'finite_native_leaf')
        return a.copy(),expected


def extract(path,ref):
    r=Record(path,ref)
    try:
        ev=ref['expected']['event'];out={'metadata':r.meta,'ref':ref,'ids':{},'values':{}}
        def take(name,*p,retain=False):
            a,d=r.leaf(*p);out['ids'][name]=d
            if retain:out['values'][name]=a
            return a,d
        if ev=='GPU_INPUT_AUDIT':
            a,d=take('noise','explicit_noise');need(a.shape==(1,10,32)and a.dtype==np.float32,'original_FP32_noise')
            take('decode_state','consumed_decode_state')
            for s in('C','T'):
                c=r.meta['checks'][s];need(c['exact']is True and len(c['checks'])==2,'actual_GPU_audit')
                need(all(v['exact']and v['noise_exact']and v['input_binding_before']==v['input_binding_after']==v['GPU_encoded_input_binding']for v in c['checks']),'actual_encoded_input_hash')
            need(r.meta['checks']['decode_states_equal']is True,'decode_state_C_T')
        elif ev=='CALL_OUTPUT':
            for name,shape,dtype in [('raw32',(1,10,32),'float32'),('adapter7',(1,10,7),'float64')]:
                a,d=take(name,name,retain=True);need(a.shape==shape and d['logical_dtype']==dtype,'original_endpoint_ABI')
            for key in('x','velocity_native','time','time_weak_type','dt'):
                a,d=take('flow_'+key,'flow',key)
                subset=a[:6]if key!='velocity_native'else a[:5]
                out['ids']['pre_j5_'+key]=aid(subset,d['logical_dtype'])
            for k in SITE_FIELDS+('probabilities_native',):take(k,'first_read',k)
            if r.meta['role']in('natural_C','natural_T'):
                for k in('receiver_L9_K','receiver_L9_V','prefix_mask','prefix_positions'):
                    take(k,k,retain=k=='receiver_L9_V')
            t=r.meta['technical'];need(t and all(t['gates'].values())and t['sampling_abi']['num_steps']==10,'native_site_gates')
            if r.meta['role'].startswith('effect_'):
                s=t['selectors'];need(s['kv_mask']==[False,True]and s['layer_mask']==[i==8 for i in range(18)]and s['flow_mask']==[i>=5 for i in range(10)]and s['use_vmod']is True,'same_L9_persistent_V_site')
        elif ev=='FROZEN_QUERY_EDITS':
            for arm in ARMS:
                for k in('Vmod','actual_E'):take(k+'/'+arm,k,arm,retain=True)
        elif ev=='ACTUAL_EFFECT':take('effect','signed_adapter7_effect',retain=True)
        elif ev=='EXACT_COMPARISON':need(r.meta['checks']and all(r.meta['checks'].values()),'original_null_not_exact')
        else:raise ValueError('unselected_event')
        return out
    finally:r.close()


def actual_edit_identity(a,logical):
    # Exact original S.edit_identity hash domain; BF16 storage is uint16 here.
    header={'shape':list(a.shape),'logical_dtype':logical,'stored_dtype':'<V2'if logical=='bfloat16'else a.dtype.str,'bytes':a.nbytes}
    return {**header,'sha256':sha(canonical(header)+b'\0'+a.tobytes())}


def query_data(run,qp,get_local,read_exact=None):
    refs={m['path']:m for m in run['members']}
    def read(name):
        ref=refs[name]
        return read_exact(ref,lambda p:extract(p,ref))if read_exact is not None else extract(get_local(ref),ref)
    audit=read(qp['input_audit']);ed=read(qp['edits'])
    calls={name:read(path)for name,path in qp['calls'].items()}
    null=read(qp['null_comparison']);c,t=calls['natural_C'],calls['natural_T']
    need(refsame(null['metadata']['reference'],t['ref'])and refsame(null['metadata']['candidate'],calls['V_own_null']['ref']),'actual_null_ref_binding')
    need(all(eq(t['values'][k],calls['V_own_null']['values'][k])for k in('raw32','adapter7')),'original_null_endpoint_bytes')
    effects={}
    vt=t['values']['receiver_L9_V'];vtd=t['ids']['receiver_L9_V']
    need(vtd['logical_dtype']=='bfloat16'and vt.shape==(1,968,1,256),'native_prefix_V256')
    for arm in ARMS:
        output=calls['effect_'+arm];label=read(qp['effects'][arm]);m=label['metadata']
        need(refsame(m['actual_effect_source'],output['ref'])and refsame(m['edit_ref'],ed['ref']),'effect_endpoint_edit_ref')
        need(refsame(output['metadata']['baseline_refs']['T'],t['ref'])and refsame(output['metadata']['edit_ref'],ed['ref']),'endpoint_actual_baseline_and_edit')
        need(output['metadata']['source_identity']==audit['metadata']['source']['source_identity'],'call_source_audit')
        vm=ed['values']['Vmod/'+arm];e=ed['values']['actual_E/'+arm]
        need(ed['ids']['Vmod/'+arm]['logical_dtype']=='bfloat16'and e.dtype==np.float32 and vm.shape==vt.shape==e.shape,'actual_E_Vmod_ABI')
        need(eq(e,native_fp32(vm,'bfloat16')-native_fp32(vt,'bfloat16')),'actual_E_is_native_Vmod_minus_receiver')
        actual=output['metadata']['actual_edit']
        for k,a,dtype in [('receiver_V',vt,'bfloat16'),('Vmod_argument',vm,'bfloat16'),('actually_used_V_at_j5',vm,'bfloat16'),('actual_E_used_minus_receiver_FP32',e,'float32')]:
            need(actual[k]==actual_edit_identity(a,dtype),'actually_used_'+k)
        y=label['values']['effect'];need(y.dtype==np.float64 and y.shape==(10,7),'actual_effect_target_ABI')
        need(eq(y,output['values']['adapter7'][0]-t['values']['adapter7'][0]),'actual_effect_endpoint_subtraction')
        effects[arm]={'label':y,'label_ref':label['ref'],'output':output,'E_id':ed['ids']['actual_E/'+arm],
                      'Vmod_id':ed['ids']['Vmod/'+arm],'whole_E_zero':bool(not np.any(e))}
    # Drop large Vmod/E/V caches before returning. No neural payload in outputs.
    for cc in calls.values():cc['values'].pop('receiver_L9_V',None)
    return {'audit':audit,'calls':calls,'effects':effects,'edit_ref':ed['ref']}


def input_identity(d):
    a=d['audit'];m=a['metadata'];s=m['source']
    return {'source_identity':s['source_identity'],'original_pair_receipt':s['pair_receipt'],
        'original_query_clock':s['query_clock_receipt'],'actual_prompt_sha256':s['actual_prompt_sha256'],
        'initial_selection':s['actual_initial_selection'],'noise':a['ids']['noise'],'decode_state':a['ids']['decode_state'],
        'C':m['checks']['C'],'T':m['checks']['T']}


def compare_query(left,right,root,q,graph_same=True):
    common=input_identity(left)==input_identity(right)
    natural_left,natural_right=left['calls']['natural_T'],right['calls']['natural_T']
    cache_keys=('receiver_L9_K','receiver_L9_V','prefix_mask','prefix_positions')
    cache_same=all(natural_left['ids'][k]==natural_right['ids'][k]for k in cache_keys)
    site_keys=(*SITE_FIELDS,*(f'pre_j5_{k}'for k in('x','velocity_native','time','time_weak_type','dt')))
    site_same=all(natural_left['ids'][k]==natural_right['ids'][k]for k in site_keys)
    eligibility=[];positions=[];windows=[];baseline=[]
    for side in('C','T'):
        a,b=left['calls']['natural_'+side]['values']['adapter7'][0],right['calls']['natural_'+side]['values']['adapter7'][0]
        for p in range(10):
            for j,dim in enumerate(DIMS):baseline.append({'root_id':root,'query_index':q,'condition':side,'action_slot':p,'dimension':dim,
                'r2':float(a[p,j]),'r3':float(b[p,j]),'r3_minus_r2':float(b[p,j]-a[p,j])})
    for arm in ARMS:
        a,b=left['effects'][arm],right['effects'][arm];ca,cb=a['output'],b['output']
        checks={'same_actual_inputs_and_source':common,'same_graph_source_and_model':graph_same,'same_prefix_receiver_cache':cache_same,
                'same_E':a['E_id']==b['E_id'],'same_Vmod':a['Vmod_id']==b['Vmod_id'],
                'same_site_and_sampling_ABI':ca['metadata']['technical']['selectors']==cb['metadata']['technical']['selectors']and ca['metadata']['technical']['sampling_abi']==cb['metadata']['technical']['sampling_abi']}
        eligible=all(checks.values());reason=[k for k,v in checks.items()if not v]
        key={'root_id':root,'query_index':q,'arm':arm}
        eligibility.append({**key,**checks,'eligible_full_sampling_replication':eligible,'same_j5_receiver_operands':site_same,
            'eligible_same_local_receiver_replication':eligible and site_same,
            'same_natural_native_P':natural_left['ids']['probabilities_native']==natural_right['ids']['probabilities_native'],
            'ineligible_reasons':reason,'actual_E_zero_r2':a['whole_E_zero'],'actual_E_zero_r3':b['whole_E_zero'],
            'effect_endpoint_raw32_exact':eq(ca['values']['raw32'],cb['values']['raw32']),
            'effect_endpoint_adapter7_exact':eq(ca['values']['adapter7'],cb['values']['adapter7']),
            'signed_effect_typed_exact':eq(a['label'],b['label']),'comparison_role':'replication'if eligible else'diagnostic_only_ineligible'})
        ya,yb=a['label'],b['label']
        for p in range(10):
            for j,dim in enumerate(DIMS):
                u,v=float(ya[p,j]),float(yb[p,j]);positions.append({**key,'eligible':eligible,'action_slot':p,'dimension':dim,
                    'r2_signed_effect':u,'r3_signed_effect':v,'r3_minus_r2':v-u,
                    'r2_sign':int(np.sign(u)),'r3_sign':int(np.sign(v)),'both_exact_zero':u==0 and v==0})
        for window,n in [('proposed_first5',5),('full10',10)]:
            for j,dim in enumerate(DIMS):
                u,v=ya[:n,j],yb[:n,j];norm=float(np.linalg.norm(u)*np.linalg.norm(v));amp=max(float(np.max(abs(u))),float(np.max(abs(v))))
                windows.append({**key,'eligible':eligible,'window':window,'dimension':dim,
                    'r2_maxabs':float(np.max(abs(u))),'r3_maxabs':float(np.max(abs(v))),
                    'effect_difference_maxabs':float(np.max(abs(v-u))),
                    'relative_to_max_effect':float(np.max(abs(v-u)))/amp if amp else None,
                    'cosine_same_dimension_across_slots':float(np.dot(u,v))/norm if norm else None,
                    'exact_equal_positions':int(np.count_nonzero(u==v)),'same_sign_positions':int(np.count_nonzero(np.sign(u)==np.sign(v))),
                    'opposite_nonzero_sign_positions':int(np.count_nonzero(u*v<0)),'positions':n})
    return eligibility,positions,windows,baseline


def run(plan,get_local,output_writer,*,read_exact=None,synthetic=False):
    """No network API. Owner must authenticate frozen plan/READY/ACK before call.

    Prefer read_exact(ref,consume_path), the retained one-member/finally-delete
    callback. get_local is the compatible alternative owning its cache lifetime.
    output_writer(name,bytes) meters total output.
    A failed restore/shape is preserved per query, never a numerical zero.
    """
    need(plan['schema']=='L9-DEV-replication-plan-v1'and plan['roots']==list(ROOTS)and plan['queries']==list(range(11,23))and plan['arms']==list(ARMS),'exact_DEV_scope')
    need([r['label']for r in plan['runs']]==['r2','r3'],'two_original_runs')
    need(callable(read_exact)or callable(get_local),'explicit_local_restore_owner')
    gp=plan['graph_provenance']
    need(sha(canonical(gp['common_source_files']))==gp['common_source_digest']and gp['different_files']==['metadata/REQUEST-v1.json','study/integration/science_dispatch.py'],'frozen_common_graph_definition')
    rows={k:[]for k in('ELIGIBILITY','POSITIONS','WINDOWS','BASELINES','MISSING')}
    model_ids=[]
    for r in plan['runs']:
        ref=r['model_binding']
        b=read_exact(ref,lambda p:Path(p).read_bytes())if read_exact is not None else Path(get_local(ref)).read_bytes()
        need(len(b)==ref['bytes']and sha(b)==ref['sha256'],'actual_model_binding_SHA')
        model=json.loads(b);need(model['model_bound']is True and model['adapter_bound']is True,'actual_same_model_adapter_bound');model_ids.append(sha(b))
    graph_same=(model_ids[0]==model_ids[1]==plan['graph_provenance']['same_model_binding_file_sha256'])
    for root in ROOTS:
        for q in range(11,23):
            try:
                data=[]
                for r in plan['runs']:
                    qp=next(v for v in r['queries']if(v['root_id'],v['query_index'])==(root,q))
                    data.append(query_data(r,qp,get_local,read_exact))
                for k,v in zip(('ELIGIBILITY','POSITIONS','WINDOWS','BASELINES'),compare_query(*data,root,q,graph_same)):
                    rows[k].extend(v)
                del data
            except Exception as e:
                code=str(e)if re.fullmatch('[A-Za-z0-9_]{1,120}',str(e))else'callback_or_decode_failure'
                rows['MISSING'].append({'root_id':root,'query_index':q,'arms':list(ARMS),'status':'UNAVAILABLE','error_type':type(e).__name__,'reason':code})
    summary=[]
    for root in ROOTS:
        for arm in ARMS:
            rr=[v for v in rows['ELIGIBILITY']if(v['root_id'],v['arm'])==(root,arm)];elig=[v for v in rr if v['eligible_full_sampling_replication']]
            summary.append({'root_id':root,'arm':arm,'planned_queries':12,'observed_queries':len(rr),'eligible_queries':len(elig),
                'ineligible_queries':len(rr)-len(elig),'missing_queries':12-len(rr),
                'typed_exact_signed_effects':sum(v['signed_effect_typed_exact']for v in elig),
                'exact_repetition_fraction_among_eligible':sum(v['signed_effect_typed_exact']for v in elig)/len(elig)if elig else None,
                'all12_eligible':len(elig)==12,'same_local_receiver_eligible':sum(v['eligible_same_local_receiver_replication']for v in rr)})
    result={'schema':'L9-DEV-replication-readout-v1','status':'SOFTWARE_SYNTHETIC'if synthetic else'COMPLETE'if not rows['MISSING']else'PARTIAL_PRESERVED',
        'planned_query_arms':96,'comparison_summary':summary,'missing_queries':rows['MISSING'],
        'scope':'same existing DEV only; not independent samples or heldout score; no CI, new threshold, fit or hardware attribution',
        'eligibility_meaning':'full-sampling eligibility fixes original input+noise+native edit+graph; j5 receiver equality is separately required for same-local-state wording',
        'no_neural_array_export':True,'new_model_or_env_calls':0,'input_once_bound':plan['bounds']['original_member_bytes_once']}
    written=0
    for name,values in rows.items():
        if not values:continue
        buf=io.StringIO();w=csv.DictWriter(buf,fieldnames=list(values[0]));w.writeheader();w.writerows(values);b=buf.getvalue().encode()
        written+=len(b);need(written<plan['bounds']['output_candidate_cap_bytes']-65536,'small_output_cap');output_writer('DEV-'+name+'-v1.csv',b)
    b=(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode();need(written+len(b)<=plan['bounds']['output_candidate_cap_bytes'],'small_receipt_cap');output_writer('RESULT-v1.json',b)
    return result
