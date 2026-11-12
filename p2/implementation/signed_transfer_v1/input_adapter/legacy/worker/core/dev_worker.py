"""Exact r3 CPU refit. Caller owns IO/transport and a bounded output writer.

No credential, JAX/model/environment import; no old predictor/score is consumed.
"""
import csv,io,json,re,time
from collections import Counter
import numpy as np
import prediction as P
import reader as R

EXPECTED={'CAMPAIGN_PLANNED':1,'READ_ONLY_WO':1,'GPU_INPUT_AUDIT':48,'CALL_OUTPUT':288,
          'FROZEN_QUERY_EDITS':48,'ACTUAL_EFFECT':192,'EXACT_COMPARISON':116,'RESULT':1}

def validate_plan(plan):
    P.need(plan['schema']=='sru-task-blind-refit-exact-readset-candidate-v1','readset_schema')
    pop=plan['population'];P.need(pop['roots']==list(P.ROOTS) and pop['query_indices']==list(range(11,23)) and pop['arms']==list(P.ARMS),'fixed_DEV_population')
    refs=plan['members'];P.need(len(refs)==695 and len({r['uri'] for r in refs})==len({r['path'] for r in refs})==695,'exact_unique_readset')
    P.need(Counter(r['event'].rsplit('.',1)[0] for r in refs)==EXPECTED,'exact_event_counts')
    P.need(sum(r['bytes'] for r in refs)==plan['selected_body_bytes']==224484300 and max(r['bytes'] for r in refs)==3327532,'exact_source_input_budget')
    for r in refs:P.need(r['uri'].startswith('bos://') and r['uri'].endswith('/'+r['object_key']) and P.is_sha(r['sha256']) and 0<r['bytes']<=3327532,'exact_source_ref')
    return refs

def original_gates(original,comparisons,calls):
    P.need(original['status']=='COMPLETE' and original['scientific_complete']is True and original['synthetic']is False and not original['failures'],'original_complete_native_run')
    P.need(original['used']=={'full':432,'prefill':860,'suffix':4320} and original['checked_queries']==48 and original['dev_targets']==original['held_targets']==96,'original_full_population')
    records=original['call_records'];by={r['receipt']['path']:r for r in records}
    P.need(len(by)==432 and all(r['status']=='recorded' for r in records),'original_call_ledger_complete')
    for rec in calls:
        actual=by[rec['ref']['path']]
        P.need(R.R.refsame(actual['receipt'],rec['ref']) and all(actual[k]==rec['metadata'][k] for k in ('root_id','query_index','role','call_id')),'original_call_receipt_identity')
    vnull=set();knull=set();repeat=set();initial=[]
    for rec in comparisons:
        m=rec['metadata'];left=by[m['reference']['path']];right=by[m['candidate']['path']]
        P.need(R.R.refsame(m['reference'],left['receipt']) and R.R.refsame(m['candidate'],right['receipt']),'original_comparison_receipts')
        P.need((left['root_id'],left['query_index'])==(right['root_id'],right['query_index']),'original_comparison_same_query')
        key=(left['root_id'],left['query_index']);code=m['code']
        if code=='V_same_mode_own_null_exact':
            P.need((left['role'],right['role'])==('natural_T','V_own_null'),'original_V_null_roles');vnull.add(key)
        elif code=='K_same_mode_own_null_exact':
            P.need((left['role'],right['role'])==('natural_T','K_own_null'),'original_K_null_roles');knull.add(key)
        elif code=='delayed_same_runtime_shared_repeat_exact':
            P.need((left['role'],right['role'])==('effect_shared','repeat_shared'),'original_repeat_roles');repeat.add(key)
        else:initial.append((key,code))
    expected={(r,q) for r in P.ROOTS for q in range(11,23)}
    P.need(vnull==repeat==expected and knull=={(r,q) for r in P.ROOTS for q in (11,16,22)},'every_DEV_query_original_null_repeat')
    P.need(set(initial)=={((r,11),code) for r in P.ROOTS if r.endswith('R00') for code in ('current_common_C_repeat_exact','current_common_T_repeat_exact','fullcache_T_to_C_exact','fullcache_C_to_T_exact')},'original_initial_repeat_fullcache_gates')
    return {'original_V_null_queries':len(vnull),'original_K_null_queries':len(knull),'original_shared_repeat_queries':len(repeat),'original_initial_gates':len(initial)}

def csv_bytes(rows):
    o=io.StringIO();w=csv.DictWriter(o,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows);return o.getvalue().encode()

def run(plan,read_exact,output_writer,*,deadline_epoch,synthetic=False):
    """read_exact(ref, consume(Path)): one verified object then immediate cleanup.

    output_writer(name, bytes) includes all science files in caller's 32 MiB cap.
    Successful software fixture is SYNTHETIC_COMPLETE, never native fit evidence.
    """
    count=0;read_bytes=0;groups={};calls=[];comparisons=[];extras={};failures=[];targets=[];model=None;products=[];current_ref=None;current_query=None
    def check():P.need(time.time()<deadline_epoch,'science_deadline')
    def put(name,data):
        receipt=output_writer(name,data);products.append({'path':name,'bytes':len(data),'sha256':P.sha(data)});return receipt
    try:
        refs=validate_plan(plan)
        for ref in refs:
            check();current_ref=R.receipt(ref);rec=read_exact(ref,lambda p:R.extract(p,ref));count+=1;read_bytes+=ref['bytes'];event=rec['event'];m=rec['metadata']
            if event in ('CAMPAIGN_PLANNED','READ_ONLY_WO','RESULT'):extras[event]=rec;continue
            if event=='EXACT_COMPARISON':comparisons.append(rec);continue
            key=(m['root_id'],m['query_index']);P.need(key in {(r,q) for r in P.ROOTS for q in range(11,23)},'only_DEV_query_events')
            group=groups.setdefault(key,{})
            slot=m['role'] if event=='CALL_OUTPUT' else 'label_'+m['arm'] if event=='ACTUAL_EFFECT' else event
            P.need(slot not in group,'no_duplicate_event_role');group[slot]=rec
            if event=='CALL_OUTPUT':calls.append(rec)
        check();planned=extras['CAMPAIGN_PLANNED']['metadata'];P.need(planned['synthetic']is False,'original_plan_not_synthetic')
        gates=original_gates(extras['RESULT']['metadata'],comparisons,calls)
        wo=extras['READ_ONLY_WO']['values']['WO']
        p=P.Plan(readset_sha256=P.sha(P.encoded(plan)),native_source_sha256=plan['source_binding']['source_manifest_sha256'],
                 intervention_contract_sha256=P.sha(P.encoded(planned['plan'])))
        queries=[];checks=[]
        for key in p.keys():
            check();current_query=list(key);q,ts=R.build_query(groups[key],p);queries.append(q);targets.extend(ts)
            checks.append({'root_id':key[0],'query_index':key[1],'input_identity':q.fingerprint(),'targets':len(ts),
                'actual_E_and_endpoint_exact':True,'original_input_audit_exact':True,'same_native_V_only_reading_receiver':True,
                'zero_actual_E_arms':[a for a in P.ARMS if not np.any(q.edit(a))]})
        # Parsed role objects are no longer needed; immutable Query owns only
        # pre-intervention features/Vmod, Target owns adapter7 differences.
        groups.clear();calls.clear();comparisons.clear();extras.clear()
        put('SOURCE-CHECKS-v1.json',P.encoded({'schema':'task-blind-source-checks-v1','synthetic':synthetic,
            'source_run_id':plan['source_run_id'],'source_binding':plan['source_binding'],'gates':gates,'queries':checks,
            'original_camera_basis_proof':planned['source_basis'],'read_objects':count,'read_bytes':read_bytes,
            'archive_control_checks_owned_by_execution':True}))
        check();model=P.fit(p,queries,targets,wo);check()
        P.need(isinstance(model,P.Fitted),'complete_DEV_fit_required')
        b=io.BytesIO();np.savez_compressed(b,**model.arrays());arraybytes=b.getvalue()
        arrayref={'path':'FROZEN-MODEL-v1.npz','bytes':len(arraybytes),'sha256':P.sha(arraybytes)}
        put(arrayref['path'],arraybytes)
        put('FROZEN-MODEL-v1.json',P.encoded({'schema':'task-blind-frozen-artifact-v1','model_identity':model.identity,
            'model':model.manifest(),'arrays_file':arrayref,'role':'all_public_data_DEV_refit','synthetic':synthetic,
            'remote_archive_required_before_future_use':True}))
        rows=[];by={t.key:t for t in targets}
        for q in queries:
            check()
            for arm in P.ARMS:
                preds=model.predict(q,arm);y=by[q.root_id,q.query_index,arm].effect
                for method,pred in preds.items():
                    P.need(pred is not None,'DEV_prediction_defined')
                    for window,end in (('first5',5),('full10',10)):
                        for d,unit in enumerate(P.UNITS):
                            diff=pred[:end,d]-y[:end,d]
                            rows.append({'root_id':q.root_id,'query_index':q.query_index,'arm':arm,'method':method,'window':window,
                                'command_dimension':unit,'MSE':float(np.mean(diff*diff)),'MAE':float(np.mean(abs(diff))),
                                'zero_actual_edit':not np.any(q.edit(arm)),'role':'DEV_IN_SAMPLE_NOT_VALIDATION'})
        put('DEV-ERRORS-v1.csv',csv_bytes(rows));check()
    except Exception as e:
        code=str(e) if re.fullmatch('[A-Za-z0-9_]{1,160}',str(e)) else 'read_decode_fit_or_output_error'
        failures.append({'type':type(e).__name__,'code':code,'last_source_ref':current_ref,'last_query':current_query})
    complete=not failures and isinstance(model,P.Fitted)
    result={'schema':'task-blind-refit-result-v1','status':('SYNTHETIC_COMPLETE' if synthetic else 'COMPLETE') if complete else 'PARTIAL',
        'fit_complete':complete and not synthetic,'software_fit_complete':complete,'synthetic':synthetic,
        'source_run_id':plan.get('source_run_id'),'read_object_count':count,'read_body_bytes':read_bytes,'target_count':len(targets),
        'expected_targets':192,'expected_queries':48,'model_identity':model.identity if isinstance(model,P.Fitted) else None,
        'outputs':products,'errors':failures,'model_calls':0,'environment_calls':0,'camera_basis_refit':False,
        'new_task_evaluation':False,'DEV_errors_are_in_sample':True,'behavior_claim':False}
    # Small terminal receipt uses wrapper's reserved output/archival deadline.
    put('TASK-BLIND-RESULT-v1.json',P.encoded(result));return result
