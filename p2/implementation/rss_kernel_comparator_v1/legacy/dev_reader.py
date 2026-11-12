"""Actual r3 typed event -> original Query/Target identities, CPU-only.

One exact object is consumed at a time; never keep complete effect flow tensors.
Transport/READY/ACK verification is the execution owner's responsibility.
"""
from pathlib import Path
import importlib.util, sys
import numpy as np
import prediction as P

def retained():
    p=Path(__file__).with_name('retained_replication.py')
    if not p.is_file():p=Path(__file__).resolve().parent.parent/'l9-replication-readout-v1/replication.py'
    P.need(P.sha(p.read_bytes())=='d3e3204863c20c160022f9ae2282e2f47567e41dda0b6b8b7454e962a034292c','retained_reader_source_SHA')
    name='_taskblind_native_record_v1'
    if name in sys.modules:return sys.modules[name]
    s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
R=retained()
SITE=('query_post_rope_scaled','suffix_key_post_rope','suffix_value','receiver_residual_input','attention_gate','probabilities_native','mask','suffix_positions')

def native(a,logical):return P.Native(a,logical,P.sha(a.tobytes()))
def receipt(ref):return {k:ref[k] for k in ('path','bytes','sha256')}
def extract(path,ref):
    r=R.Record(path,ref)
    try:
        event=ref['event'].rsplit('.',1)[0];m=r.meta;out={'event':event,'metadata':m,'ref':ref,'values':{},'ids':{}}
        def take(name,*path,keep=False):
            a,d=r.leaf(*path);out['ids'][name]=d
            if keep:out['values'][name]=native(a,d['logical_dtype'])
            return a,d
        if event=='GPU_INPUT_AUDIT':
            for name in ('explicit_noise','consumed_decode_state'):take(name,name,keep=True)
            for side in ('C','T'):
                c=m['checks'][side];P.need(c['exact']is True and len(c['checks'])==2,'actual_GPU_input_audit')
                P.need(all(x['exact']is True and x['noise_exact']is True and x['input_binding_before']==x['input_binding_after']==x['GPU_encoded_input_binding'] for x in c['checks']),'original_input_and_noise_exact')
            P.need(m['checks']['decode_states_equal']is True,'actual_decode_state_equal')
        elif event=='CALL_OUTPUT':
            natural=m['role'] in ('natural_C','natural_T')
            P.need(natural or m['role'] in tuple('effect_'+a for a in P.ARMS),'only_natural_or_declared_effect')
            for name,shape,dtype in (('raw32',(1,10,32),'float32'),('adapter7',(1,10,7),'float64')):
                a,d=take(name,name,keep=True);P.need(a.shape==shape and d['logical_dtype']==dtype,'native_endpoint_ABI')
            for name in ('x','velocity_native','time','time_weak_type','dt'):
                a,d=take('flow_'+name,'flow',name)
                count=5 if name in ('velocity_native','dt') else 6
                out['ids']['pre_j5_'+name]=R.aid(a[:count],d['logical_dtype'])
                if natural and name!='dt':
                    for j in (0,5):out['values'][name+str(j)]=native(a[j],d['logical_dtype'])
            for name in SITE:take(name,'first_read',name,keep=natural and name in ('probabilities_native','mask','attention_gate','suffix_positions'))
            if natural:
                for name in ('receiver_L9_V','prefix_mask','prefix_positions'):take(name,name,keep=True)
            t=m['technical'];P.need(t and t['gates'] and all(v is True for v in t['gates'].values()) and t['sampling_abi']['num_steps']==10,'actual_native_site_gates')
            if not natural:
                s=t['selectors'];P.need(s['kv_mask']==[False,True] and s['layer_mask']==[i==8 for i in range(18)] and s['flow_mask']==[i>=5 for i in range(10)] and s['use_vmod']is True,'same_L9_persistent_V_operation')
            P.need(m['synthetic']is False or ref.get('_synthetic_test')is True,'native_event_not_synthetic')
        elif event=='FROZEN_QUERY_EDITS':
            for arm in P.ARMS:
                for name in ('Vmod','actual_E'):take(name+'/'+arm,name,arm,keep=True)
        elif event=='ACTUAL_EFFECT':take('effect','signed_adapter7_effect',keep=True)
        elif event=='READ_ONLY_WO':
            a,d=take('WO','native_L9_WO',keep=True)
            P.need(a.shape==(8,256,1024) and d['logical_dtype']=='bfloat16','original_native_WO_BF16_shape')
            P.need(m['path']=='PaliGemma/llm/layers/attn/attn_vec_einsum_1/w' and m['layer_index']==8 and m['stacked_shape']==[18,8,256,1024],'actual_WO_parameter_path')
            P.need(m['actual_layer_identity']==R.actual_edit_identity(a,d['logical_dtype']),'actual_WO_parameter_identity')
        elif event=='EXACT_COMPARISON':P.need(m['checks'] and all(v is True for v in m['checks'].values()),'original_exact_comparison_failed')
        elif event in ('CAMPAIGN_PLANNED','RESULT'):pass
        else:raise ValueError('unselected_event')
        return out
    finally:r.close()

def build_query(group,plan):
    a=group['GPU_INPUT_AUDIT'];c=group['natural_C'];t=group['natural_T'];ed=group['FROZEN_QUERY_EDITS']
    root,q=a['metadata']['root_id'],a['metadata']['query_index'];proof=a['metadata']['source'];cv,tv=c['values'],t['values']
    P.need(proof['root_id']==root and proof['query_index']==q and proof['env_step_index']==5*q,'actual_source_query_clock')
    for rec in (c,t,ed):P.need((rec['metadata']['root_id'],rec['metadata']['query_index'])==(root,q),'query_event_join')
    for rec in (c,t):P.need(rec['metadata']['source_identity']==proof['source_identity'],'natural_source_identity')
    for name in ('prefix_mask','prefix_positions'):P.need(cv[name].manifest()==tv[name].manifest(),'C_T_same_prefix_layout')
    for rec in (c,t):P.need(rec['values']['x0'].manifest()==a['values']['explicit_noise'].manifest(),'natural_noise_equals_original_GPU_audit')
    flow={side+'_'+site:rec['values'][kind+str(j)] for side,rec in (('C',c),('T',t)) for site,kind,j in (('x0','x',0),('v0','velocity_native',0),('x5','x',5),('v5','velocity_native',5))}
    source=P.sha(P.encoded({'input_source_identity':proof['source_identity'],'natural_C_receipt':receipt(c['ref']),
        'natural_T_receipt':receipt(t['ref']),'layer_index':8,'first_read_call':5,'flow_sites':[0,5]}))
    query=P.Query(root,q,proof['env_step_index'],source,tv['probabilities_native'],tv['mask'],tv['prefix_mask'],tv['prefix_positions'],tv['suffix_positions'],
        cv['receiver_L9_V'],tv['receiver_L9_V'],tv['attention_gate'],flow,a['values']['consumed_decode_state'],
        native(np.stack([tv['time0'].storage,tv['time5'].storage]),'float32'),
        native(np.stack([tv['time_weak_type0'].storage,tv['time_weak_type5'].storage]),'bool'),
        {arm:ed['values']['Vmod/'+arm] for arm in P.ARMS})
    query.validate(plan);targets=[]
    for arm in P.ARMS:
        eff=group['effect_'+arm];label=group['label_'+arm];m=eff['metadata'];lm=label['metadata']
        for ref,want in ((m['baseline_refs']['C'],c['ref']),(m['baseline_refs']['T'],t['ref']),(m['edit_ref'],ed['ref']),
                         (lm['actual_effect_source'],eff['ref']),(lm['edit_ref'],ed['ref'])):P.need(R.refsame(ref,want),'actual_baseline_edit_label_receipts')
        P.need(m['source_identity']==proof['source_identity'],'effect_source_audit')
        e=ed['values']['actual_E/'+arm];vm=query.modified_v[arm];vt=query.treated_v
        P.need(e.logical_dtype=='float32' and e.storage.shape==vt.storage.shape and R.eq(e.storage,query.edit(arm)[None,:,None,:]),'actual_E_from_native_BF16_payload')
        for name,v in (('receiver_V',vt),('Vmod_argument',vm),('actually_used_V_at_j5',vm),('actual_E_used_minus_receiver_FP32',e)):
            P.need(m['actual_edit'][name]==R.actual_edit_identity(v.storage,v.logical_dtype),'actually_used_native_edit_'+name)
        for name in (*SITE,*(f'pre_j5_{s}' for s in ('x','velocity_native','time','time_weak_type','dt'))):
            P.need(eff['ids'][name]==t['ids'][name],'V_only_same_native_reading_and_receiver_'+name)
        target=P.target_from_endpoints(query,arm,tv['adapter7'],eff['values']['adapter7'],source_identity=eff['ref']['sha256'])
        P.need(label['values']['effect'].logical_dtype=='float64' and R.eq(label['values']['effect'].storage,target.effect),'original_ACTUAL_EFFECT_recomputed')
        P.need(lm['input_identity']==query.fingerprint() and lm['edit_identity']==query.edit_identity(arm),'original_predictor_input_edit_identity')
        targets.append(target)
    return query,targets
