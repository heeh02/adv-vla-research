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

def reproduction_qa(current, reference, tolerance=1e-10):
    """Compare all old endpoint/contrast score trees, preserving undefined cells."""
    tracks=[]
    for family in ('current4','current_full','rbf'):
        tracks += [f'fixed__{family}__{v}' for v in ('0.1','1','10')]
        tracks += ['nested__'+family,'contrast_nested__'+family]
    numbers=0; maximum=0.0
    def compare(a,b):
        nonlocal numbers, maximum
        if isinstance(a,dict):
            need(isinstance(b,dict) and set(a)==set(b),'reproduction_exact_score_fields')
            for k in a: compare(a[k],b[k])
        elif isinstance(a,list):
            need(isinstance(b,list) and len(a)==len(b),'reproduction_exact_score_support')
            for av,bv in zip(a,b):compare(av,bv)
        elif isinstance(a,(int,float)) and not isinstance(a,bool):
            need(isinstance(b,(int,float)) and not isinstance(b,bool),'reproduction_numeric_type')
            need(math.isfinite(a) and math.isfinite(b),'reproduction_finite')
            numbers+=1; maximum=max(maximum,abs(a-b))
        else:need(a==b,'reproduction_exact_identity_or_undefined')
    for field in ('OOF_root_arm_scores','paired_persistent_contrast'):
        old=reference[field];new=current[field]
        if field=='paired_persistent_contrast':old=old['root_scores'];new=new['root_scores']
        for track in tracks:
            need(track in old and track in new,'reproduction_all15_old_tracks')
            compare(old[track],new[track])
    need(maximum<=tolerance,'old_v2_reproduction_tolerance')
    return {'schema':'rss-kernel-old-v2-score-reproduction-v1','status':'PASS',
        'tracks':tracks,'score_sections':2,'numeric_cells':numbers,
        'maximum_absolute_difference':maximum,'absolute_tolerance':tolerance,
        'reference':fid(HERE/'REFERENCE-v2.json'),
        'preserves_undefined_support_and_root_arm_identity':True,
        'prediction_array_bitwise_reproduction_claim':False}

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
        phase='old_v2_reproduction';t=time.monotonic()
        qa=reproduction_qa(analysis['summary'],json.loads((HERE/'REFERENCE-v2.json').read_text()))
        write(out/'old-v2-reproduction-QA.json',qa);times[phase]=time.monotonic()-t
        result.update(status='RSS_CPU_COMPLETE',analysis_complete=True,new_fits=analysis['summary']['new_fits'],
            fold_count=len(analysis['summary']['folds']),fold_details_output='grouped-baselines.json',
            mathematical_scope='retrospective persistent PCA4/full nonlinear comparison; no pulse fits or new blind CONF',
            exact_minimal_missing_DEV_pulse_labels=96,old_frozen_results_replaced=False,GPU_auto_GO=False,
            raw_context_recovered_and_preprocessing_trainfold_only=True,old_v2_reproduction=qa,
            old_replay_and_incremental_counts=protocol['expected_statistical_fits'])
        result['outputs']=[{'path':f.name,**fid(f)}for f in sorted(out.iterdir())]
        need(all(r['bytes']<=64*1024**2 for r in result['outputs']),'per_output_64MiB_limit')
    except BaseException as error:
        result.update(status='INCOMPLETE',analysis_complete=False)
        result['failure']={'phase':phase,'type':type(error).__name__,
            'code':str(error)if isinstance(error,(ValueError,RuntimeError))and len(str(error))<180 and all(c.isalnum()or c=='_'for c in str(error))else None,
            'frames':[{'file':Path(f.filename).name,'line':f.lineno,'function':f.name}for f in traceback.extract_tb(error.__traceback__)[-12:]]}
    result.update(elapsed_seconds=time.monotonic()-start,phase_seconds=times)
    write(out/RESULT_NAME,result);need((out/RESULT_NAME).stat().st_size<=2*1024**2,'main_result_2MiB')
    print(json.dumps({'status':result['status'],'analysis_complete':result['analysis_complete'],'failure':result.get('failure')}),flush=True)
    return 0 if result['analysis_complete']else 2

if __name__=='__main__':raise SystemExit(main())
