"""Actual GPU0 RSS preparation evidence: exact data audit and grouped CPU fits."""
from pathlib import Path
import argparse, hashlib, importlib.util, json, math, sys, time, traceback
import numpy as np

HERE=Path(__file__).resolve().parent
RESULT_NAME='rss-cpu-result.json'

def need(ok,code):
    if not ok:raise RuntimeError(code)

def fid(path):
    raw=Path(path).read_bytes();return {'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec)
    sys.modules[name]=m;spec.loader.exec_module(m);return m

def serial(x):
    if isinstance(x,np.ndarray):return serial(x.tolist())
    if isinstance(x,np.generic):return serial(x.item())
    if isinstance(x,float):
        need(not math.isinf(x),'no_infinite_report_value')
        return None if math.isnan(x)else x
    if isinstance(x,dict):return {str(k):serial(v)for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [serial(v)for v in x]
    return x

def write(path,value):
    raw=(json.dumps(serial(value),sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
    with Path(path).open('xb')as f:f.write(raw)
    return fid(path)

def missing_and_cost(dataset,proof,inventory):
    rows=[]
    for q in dataset['queries']:
        for arm in ('shared','random_matched'):
            rows.append({'root_id':q['root_id'],'cluster_id':q['cluster_id'],'task_id':q['task_id'],
                         'query_index':q['query_index'],'env_step_index':5*q['query_index'],
                         'arm':arm,'required_schedule':[5],'existing_schedule':[5,6,7,8,9],
                         'layer_index':8,'status':'NO_COMPATIBLE_PULSE_LABEL_IN_VERIFIED_DEV_CORPUS',
                         'input_identity':q['source_identity'],'edit_identity':q['provenance']['edit_identities'][arm],
                         'native_T_baseline':q['provenance']['natural_T_source'],
                         'frozen_edit_source':q['provenance']['frozen_edit_source'],
                         'input_audit_source':q['provenance']['input_audit_source'],
                         'parent_source':q['provenance']['parent_source'],
                         'existing_persistent_target':q['provenance']['target_identities'][arm],
                         'first5_only_still_requires_pulse_generation':True})
    need(len(rows)==96,'exact96_missing_dual_pulse_labels')
    subset=[r for r in rows if r['query_index']in(11,22)]
    need(len(subset)==16,'exact16_q11_q22_minimal_labels')
    hist=json.loads((HERE/'inventory/COST-EVIDENCE-v1.json').read_text())
    call_bytes=[x['source']['bytes']for x in proof['actual_persistent_selectors']]
    labels=[x['bytes']for x in proof['input_receipts']if x['path'].endswith('ACTUAL_EFFECT.npz')]
    scenarios=[]
    for n in (2,4,8,16,32):
        clusters=6*n;parents=2*clusters;snapshots=2*parents
        scenarios.append({'roots_per_task':n,'fixed_source_plus_held_tasks':6,'new_init_clusters':clusters,
                          'paired_parent_trajectories':parents,'snapshots_q11_q22':snapshots,
                          'four_arm_full_generations':4*snapshots,'conservative_prefill_bound':8*snapshots,
                          'nominal_suffix_calls':40*snapshots,'proposed_snapshot_output_cap_bytes':16*1024**2*snapshots,
                          'optional_execution_main_q11_anchors':parents,
                          'optional_N_S_R_20_step_env_steps':parents*3*20,
                          'optional_post_first5_replan_full_generations':parents*3*3,
                          'not_a_selected_or_authorized_sample_size':True,
                          'init_library_unique_unused_capacity_not_verified_here':True})
    costs={'schema':'rss-minimal-label-and-cost-ledger-v1','executed_new_model_calls':0,
           'scope':'future candidate budgets; no GPU authorization or execution',
           'verified_DEV_reuse':{'none_endpoints':48,'shared_persistent':48,'random_matched_persistent':48,
                                'other_persistent_labels':96,'shared_pulse':0,'random_matched_pulse':0,
                                'pulse_coverage_scope':'exact DEV r3 corpus; other unindexed sources stay unknown'},
           'minimum_existing_support_label_debt':{'queries':48,'dual_pulse_labels':96,
               'full_generation_conservative':96,'prefill_conservative_bound':192,'suffix_nominal':960,
               'resume_j5_suffix_conditional_lower_count':480,
               'resume_not_qualified':True,'fresh_native_qualification_and_any_required_baseline_replay_not_included':True,
               'snapshot_output_planning_cap_bytes':48*16*1024**2},
           'q11_q22_subset_separate_proposal':{'snapshots':8,'dual_pulse_labels':16,
               'full_generation_conservative':16,'prefill_conservative_bound':32,'suffix_nominal':160,
               'resume_j5_suffix_conditional_lower_count':80,'snapshot_output_planning_cap_bytes':8*16*1024**2,
               'not_replacement_for_all48_coverage':True,'only4_clusters_not_statistical_sufficiency':True},
           'qualification_unit':{'one_snapshot_four_full_generations':4,'nominal_suffix':40,
               'repeat_own_null_recorder_checks_extra_to_freeze':True,'not_yet_executed':True},
           'future_new_root_scenarios':scenarios,
           'historical_effect_record_bytes':{'count':len(call_bytes),'total':sum(call_bytes),'maximum':max(call_bytes)},
           'historical_label_record_bytes':{'count':len(labels),'total':sum(labels),'maximum':max(labels)},
           'output_cap_semantics':'16MiB/snapshot is a proposed conservative record budget, not observed future output or final ABI acceptance',
           'walltime_and_gpu_hours':{'measured_isolated_model_time_available':False,
               'formula':'GPU_hours = (N_full*tau_full + restoration + compilation + qualification + archive + release)/3600, separately counting queue occupancy',
               'tau_full':'not measured for the proposed exact four-arm runner; do not use queue/restore as model throughput',
               'historical_evidence':hist,'currency_quote':None,
               'reason_no_currency_quote':'No verified account charge rate; compute units and historical wall segments reported without inventing prices'},
           'missing_not_repaired_by_CPU':['96 actual DEV pulse labels under full old support','independent new within/held roots','N/S/R execution correspondence'],
           'unknowns_before_GPU':['exact fresh runtime qualification and throughput','chosen common Camera operator','N from pulse variance rather than persistent proxy alone','unique unused initialization roster','allocation/call/storage hard caps']}
    return {'schema':'rss-exact-missing-pulse-labels-v1','labels':rows,'q11_q22_subset':[{'root_id':r['root_id'],'query_index':r['query_index'],'arm':r['arm']}for r in subset],
            'other_unindexed_sources':'UNKNOWN_NOT_PROVEN_ABSENT','CONF_pulse_metadata':inventory['pulse_inventory']},costs

def main():
    p=argparse.ArgumentParser()
    for k in('stage','out','source-manifest-sha256','deadline-epoch'):p.add_argument('--'+k,required=True)
    a=p.parse_args();out=Path(a.out);out.mkdir(exist_ok=True);need(not any(out.iterdir()),'empty_owned_science_output')
    deadline=float(a.deadline_epoch);start=time.monotonic();phase='verify_source';times={}
    def check():need(time.time()<deadline,'absolute_science_deadline')
    protocol=json.loads((HERE/'PROTOCOL.json').read_text())
    result={'schema':'paper2-rss-cpu-result-v1','status':'INCOMPLETE','analysis_complete':False,
            'input_qualified':False,'synthetic':False,'scope':protocol['scope'],
            'new_model_calls':0,'new_env_steps':0,'training_updates':0,'JPFS_new_bytes':0,'GPU':0}
    try:
        need(sys.platform.startswith('linux'),'actual_arrays_only_cloud_Linux');check()
        source=HERE/'SOURCE-MANIFEST.json';source_id=fid(source)
        need(source_id['sha256']==a.source_manifest_sha256,'externally_bound_source_manifest')
        manifest=json.loads(source.read_text());need(manifest['schema']=='paper2-rss-cpu-source-v1','source_schema')
        for r in manifest['files']:
            check();f=HERE/r['path'];need(f.is_file()and not f.is_symlink()and f.resolve().is_relative_to(HERE),'regular_bound_source')
            need(fid(f)=={k:r[k]for k in('bytes','sha256')},'source_member_SHA')
        inputs=json.loads((HERE/'INPUTS.json').read_text());stage=Path(a.stage).resolve()
        need(len(inputs['input_objects'])==len({r['uri']for r in inputs['input_objects']})==inputs['input_object_count'],'unique_input_roster')
        for r in inputs['input_objects']:
            check();f=stage/r['path'];need(f.is_file()and not f.is_symlink()and f.parent==stage,'exact_staged_input_path')
            need(fid(f)=={k:r[k]for k in('bytes','sha256')},'exact_whole_staged_input')
        result.update(source_manifest=source_id,inputs_manifest=fid(HERE/'INPUTS.json'),
                      input_objects_verified=inputs['input_object_count'],input_body_bytes=inputs['input_body_bytes'])
        phase='inventory_and_initialization';t=time.monotonic()
        inv=load('_rss_inventory',HERE/'inventory/audit.py').audit(stage,inputs,check,synthetic=False)
        write(out/'inventory-audit.json',inv);times[phase]=time.monotonic()-t
        phase='native_DEV_reconstruction';t=time.monotonic()
        dataset,proof=load('_rss_dev_loader',HERE/'dev_loader.py').load(stage,inputs,deadline)
        groups={r['root_id']:r for r in inv['groups']}
        need(len(groups)==4,'verified_four_initialization_groups')
        for q in dataset['queries']:
            g=groups[q['root_id']];need(q['task_id']==g['task_id'],'source_task_cluster_join')
            q['cluster_id']=g['cluster_id']
        dataset.update(groups=inv['groups'],independence=inv['independence'])
        for row in proof['all_rows']:row['cluster_id']=groups[row['root_id']]['cluster_id']
        proof['verified_initialization_groups']=inv['groups']
        consumed=proof['input_receipts']+inv['receipts']
        need(len(consumed)==len({r['uri']for r in consumed})==inputs['input_object_count'],'all_inputs_semantically_consumed_once')
        need({(r['uri'],r['bytes'],r['sha256'])for r in consumed}==
             {(r['uri'],r['bytes'],r['sha256'])for r in inputs['input_objects']},'complete_exact_semantic_input_identity_set')
        write(out/'DEV-source-proof.json',proof);times[phase]=time.monotonic()-t
        result.update(input_qualified=True,pulse_reuse={'DEV_shared_pulse':0,'DEV_random_pulse':0,
            'verified_native_persistent_targets':192,'scope':'exact DEV source only','CONF_shared_pulse_metadata':inv['pulse_inventory'].get('CONF_shared_pulse_anchors'),
            'other_unindexed_sources':'UNKNOWN_NOT_PROVEN_ABSENT'},cluster_evidence=inv['independence'])
        phase='grouped_CPU_baselines';t=time.monotonic()
        math_module=load('_rss_math',HERE/'math/core.py')
        analysis=math_module.run(dataset,check=check,synthetic=False);check()
        need(analysis['summary']['analysis_complete']is True and analysis['summary']['synthetic']is False,'actual_grouped_math_complete')
        write(out/'grouped-baselines.json',analysis['summary'])
        np.savez_compressed(out/'grouped-predictions.npz',**analysis['arrays'])
        times[phase]=time.monotonic()-t
        phase='minimal_missing_and_cost';t=time.monotonic()
        missing,cost=missing_and_cost(dataset,proof,inv)
        write(out/'missing-pulse-labels.json',missing);write(out/'minimal-cost-ledger.json',cost)
        write(out/'precision-budget.json',analysis['summary']['precision_sensitivity']);times[phase]=time.monotonic()-t
        result.update(status='RSS_CPU_COMPLETE',analysis_complete=True,new_fits=analysis['summary']['new_fits'],
            fold_count=len(analysis['summary']['folds']),fold_details_output='grouped-baselines.json',
            mathematical_scope='retrospective persistent labels only; no pulse fits or new blind CONF',
            exact_minimal_missing_DEV_pulse_labels=96,old_frozen_results_replaced=False,GPU_auto_GO=False,
            raw_context_recovered_and_preprocessing_trainfold_only=True,
            orthogonal_QA=analysis['summary']['orthogonal_QA'])
        result['outputs']=[{'path':f.name,**fid(f)}for f in sorted(out.iterdir())]
        need(all(r['bytes']<=64*1024**2 for r in result['outputs']),'per_output_64MiB_limit')
    except BaseException as error:
        result['failure']={'phase':phase,'type':type(error).__name__,
            'code':str(error)if isinstance(error,(ValueError,RuntimeError))and len(str(error))<180 and all(c.isalnum()or c=='_'for c in str(error))else None,
            'frames':[{'file':Path(f.filename).name,'line':f.lineno,'function':f.name}for f in traceback.extract_tb(error.__traceback__)[-12:]]}
    result.update(elapsed_seconds=time.monotonic()-start,phase_seconds=times)
    write(out/RESULT_NAME,result);need((out/RESULT_NAME).stat().st_size<=2*1024**2,'main_result_2MiB')
    print(json.dumps({'status':result['status'],'analysis_complete':result['analysis_complete'],'failure':result.get('failure')}),flush=True)
    return 0 if result['analysis_complete']else 2

if __name__=='__main__':raise SystemExit(main())
