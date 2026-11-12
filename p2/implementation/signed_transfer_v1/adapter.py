"""Actual existing-data CPU analysis; never loads a policy or calls a model."""
from pathlib import Path
import argparse, gzip, hashlib, importlib.util, json, math, sys, time, traceback
import numpy as np

HERE=Path(__file__).resolve().parent
RESULT_NAME='signed-transfer-result.json'
SCOPE={'DEV_shared_KT':48,'CONF_shared_KT':96,'CONF_shared_KC_support':24,
       'target_shape':[10,7],'site':'G L9 first edited j5','new_model_calls':0,
       'new_env_steps':0,'training_updates':0,'GPU':0,'JPFS_new_bytes':0}

def need(ok,code):
    if not ok:raise RuntimeError(code)

def file_id(path):
    raw=Path(path).read_bytes();return {'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module;spec.loader.exec_module(module);return module

def serial(x):
    if isinstance(x,np.ndarray):return serial(x.tolist())
    if isinstance(x,np.generic):return serial(x.item())
    if isinstance(x,float):
        need(not math.isinf(x),'no_infinite_report_value')
        return None if math.isnan(x) else x
    if isinstance(x,dict):return {str(k):serial(v)for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [serial(v)for v in x]
    return x

def write_json(path,value):
    raw=(json.dumps(serial(value),sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
    with Path(path).open('xb')as f:f.write(raw)
    return file_id(path)

def legacy_check(actual,reference):
    need(actual['old_fit_identity']==reference['old_fit_identity']and actual['complete'] is True,
         'legacy_same_original_fit_and_complete_population')
    comparisons=0;max_scaled=0.0
    for field,oldfield,keyfields in [
      ('equal_task_scores','equal_task_cells',('parent_condition','K_background','window','dimension')),
      ('root_scores','all_task_cells',('root_id','parent_condition','K_background','window','dimension'))]:
        lookup={tuple(r[k]for k in keyfields)+(r['method'],):r for r in actual[field]}
        need(len(lookup)==len(actual[field]),'unique_legacy_score_keys')
        expected={tuple(r[k]for k in keyfields)+(m,)for r in reference[oldfield]for m in r['values']}
        need(set(lookup)==expected,'exact_complete_legacy_score_key_set')
        for row in reference[oldfield]:
            key=tuple(row[k]for k in keyfields)
            for method,values in row['values'].items():
                found=lookup[key+(method,)]
                for metric in ('MSE','DEV_scaled_MSE'):
                    ref=values[metric];got=found[metric]
                    need(type(ref)in(int,float)and type(got)in(int,float)and math.isfinite(got),'finite_legacy_score')
                    delta=abs(got-ref)/max(1.0,abs(ref));max_scaled=max(max_scaled,delta);comparisons+=1
                    need(delta<=1e-10,'legacy_actual_scores_reproduced')
    need(comparisons==1680,'all1680_legacy_numeric_comparisons')
    return {'passed':True,'comparisons':comparisons,'maximum_scaled_absolute_error':max_scaled,
            'tolerance':1e-10,'old_fit_identity':actual['old_fit_identity'],
            'refit_identity_actual':actual['fit_identity'],'refit_identity_reference':reference['fit_identity'],
            'refit_bitwise_identity_matches':actual['fit_identity']==reference['fit_identity'],
            'old_scores_replaced':False,'posthoc_not_new_blind_test':True}

def row_manifest(rows,core):
    return [{'key':list(r['key']),'split':r['split'],'zero_actual_E':r['zero_actual_E'],
             'provenance':r['provenance'],'arrays':{k:core.identity(v)for k,v in r.items()if isinstance(v,np.ndarray)}}for r in rows]

def cache_arrays(rows,frozen):
    fields=('context9','current4','pooled4','originalnorm4','current_gated_write','pooled_gated_write',
            'current_headpos_read','current_headpos_norms','effect')
    out={k:np.stack([r[k]for r in rows])for k in fields}
    out.update({'row_keys':np.array([[str(v)for v in r['key']]for r in rows]),
                'split':np.array([r['split']for r in rows]),'zero_actual_E':np.array([r['zero_actual_E']for r in rows],bool)})
    out.update({'frozen_'+k:v for k,v in frozen.items()});return out

def comparison_summary(evaluation):
    names=evaluation['model_names'];index={n:i for i,n in enumerate(names)}
    definitions=[('current_full','current4'),('current_full','pooled_full'),('current_full','c9'),
                 ('current_full','zero'),('headpos_signed','headpos_norms'),
                 ('headpos_signed','headpos_signed_sum'),('headpos_signed','current_full')]
    rows=[]
    for control in ('lambda1','edf_match'):
        for stronger,weaker in definitions:
            a=stronger+'@'+control;b='zero'if weaker=='zero'else weaker+'@'+control
            if a not in index or b not in index:
                rows.append({'stronger':a,'weaker':b,'status':'CAPACITY_MATCH_UNAVAILABLE'});continue
            for group in evaluation['groups']:
                if group['level']!='population':continue
                ai,bi=index[a],index[b];m=group['mse'];difference=m[bi]-m[ai]
                task=[g for g in evaluation['groups']if g['level']=='task'and g['parent']==group['parent']and g['background']==group['background']]
                need(len(task)==4,'four_tasks_per_comparison')
                task_delta=np.stack([g['mse'][bi]-g['mse'][ai]for g in task])
                rows.append({'stronger':a,'weaker':b,'status':'DEFINED','parent':group['parent'],'background':group['background'],
                             'positive_means_lower_error':True,'MSE_reduction':difference,
                             'task_order':[g['label']for g in task],'task_MSE_reduction':task_delta,
                             'positive_task_count':(task_delta>0).sum(0),'negative_task_count':(task_delta<0).sum(0),
                             'ties':(task_delta==0).sum(0),'task_count':4,'not_independent_coordinate_tests':True})
    return rows

def main():
    p=argparse.ArgumentParser()
    for k in ('stage','out','source-manifest-sha256','deadline-epoch'):p.add_argument('--'+k,required=True)
    a=p.parse_args();out=Path(a.out);out.mkdir(exist_ok=True);need(not any(out.iterdir()),'empty_owned_science_output')
    deadline=float(a.deadline_epoch);started=time.monotonic();phase='verify_source';times={}
    def check():need(time.time()<deadline,'absolute_science_deadline')
    result={'schema':'paper2-signed-transfer-result-v1','status':'INCOMPLETE','analysis_complete':False,
            'input_qualified':False,'synthetic':False,'scope':SCOPE,'new_model_calls':0,'new_env_steps':0,
            'training_updates':0,'JPFS_new_bytes':0,'posthoc_CONF_not_new_blind_test':True}
    try:
        check();need(sys.platform.startswith('linux'),'actual_existing_arrays_only_cloud_Linux')
        source=HERE/'SOURCE-MANIFEST.json';source_id=file_id(source);need(source_id['sha256']==a.source_manifest_sha256,'externally_bound_source_manifest')
        manifest=json.loads(source.read_bytes());need(manifest['schema']=='paper2-signed-transfer-source-v1','exact_science_source_schema')
        for ref in manifest['files']:
            check();f=HERE/ref['path'];need(f.is_file()and not f.is_symlink()and f.resolve().is_relative_to(HERE),'regular_bound_source')
            need(file_id(f)=={k:ref[k]for k in('bytes','sha256')},'source_member_SHA')
        inputs_id=file_id(HERE/'INPUTS.json');inputs=json.loads((HERE/'INPUTS.json').read_bytes())
        protocol=json.loads((HERE/'PROTOCOL.json').read_bytes());need(protocol['scope']==SCOPE,'frozen_scope')
        result.update(source_manifest=source_id,inputs_manifest=inputs_id,protocol=file_id(HERE/'PROTOCOL.json'))
        phase='load_qualified_existing_data';t=time.monotonic();adapter=load('_signed_transfer_input',HERE/'input_adapter/adapter.py')
        payload=adapter.load_rows(Path(a.stage),out.parent/'qualified-cache',deadline)
        times['qualified_data_decode_seconds']=time.monotonic()-t;check()
        receipt=payload['input_receipt'];need(receipt['unique_objects']==1500 and receipt['unique_body_bytes']==769677897,'all1500_inputs_consumed')
        result.update(input_qualified=True,input_objects_verified=1500,input_body_bytes=769677897,input_receipt=receipt)
        phase='reproduce_old_shared48_reference';t=time.monotonic()
        qa=legacy_check(payload['original_qa'],json.loads((HERE/'reference/legacy_context_numbers.json').read_bytes()))
        write_json(out/'legacy-reproduction.json',qa);result['legacy_reproduction']=qa;times['legacy_QA_seconds']=time.monotonic()-t
        phase='fit_DEV_only';t=time.monotonic();core=load('_signed_transfer_math',HERE/'math/core.py')
        rows=payload['rows'];dev=[r for r in rows if r['split']=='DEV'];conf=[r for r in rows if r['split']=='CONF']
        suite=core.fit_rows(dev,payload['frozen'],check=check);fit=suite.manifest()
        write_json(out/'fit-manifest.json',fit);result['fit_identity']=file_id(out/'fit-manifest.json')
        times['DEV_fit_seconds']=time.monotonic()-t;check()
        phase='evaluate_frozen_models';t=time.monotonic();evaluation=core.evaluate_rows(suite,conf);check()
        train_predictions=suite.predict_rows(dev);train_scores=core.score_predictions(train_predictions,np.stack([r['effect']for r in dev]),suite.target_scale)
        comparisons=comparison_summary(evaluation);times['evaluation_seconds']=time.monotonic()-t
        phase='save_reusable_cloud_outputs';t=time.monotonic()
        np.savez_compressed(out/'qualified-feature-cache.npz',**cache_arrays(rows,payload['frozen']));check()
        np.savez_compressed(out/'fit-arrays.npz',**suite.arrays());check()
        prediction_arrays={k:v for k,v in evaluation['predictions'].items()}
        prediction_arrays.update({'DEV__'+k:v for k,v in train_predictions.items()})
        prediction_arrays.update({'INTERACTION__'+k:v for k,v in evaluation['paired_interaction']['predictions'].items()})
        prediction_arrays['INTERACTION_actual']=evaluation['paired_interaction']['actual']
        prediction_arrays['CONF_actual']=np.stack([r['effect']for r in conf]);prediction_arrays['DEV_actual']=np.stack([r['effect']for r in dev])
        np.savez_compressed(out/'predictions.npz',**prediction_arrays);check()
        write_json(out/'row-identities.json',row_manifest(rows,core))
        report={k:v for k,v in evaluation.items()if k not in('predictions','paired_interaction')}
        report['paired_interaction']={k:v for k,v in evaluation['paired_interaction'].items()if k!='predictions'}
        report['DEV_training_score']=train_scores;report['paired_comparisons']=comparisons
        with gzip.open(out/'evaluation.json.gz','xb')as f:f.write((json.dumps(serial(report),sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode())
        check();times['output_seconds']=time.monotonic()-t
        result.update(status='SIGNED_TRANSFER_COMPLETE',analysis_complete=True,
            support={'DEV_shared_KT':len(dev),'CONF_shared_KT':sum(r['key'][3]=='KT'for r in conf),'CONF_shared_KC':sum(r['key'][3]=='KC'for r in conf),
                     'CONF_tasks':4,'CONF_roots':8,'heads_positions_not_independent_samples':True},
            model_names=evaluation['model_names'],windows=evaluation['windows'],scales=evaluation['scales'],
            model_diagnostics=fit['models'],unavailable_models=fit['unavailable'],edf_anchor=fit['edf_anchor'],
            population_scores=[g for g in evaluation['groups']if g['level']=='population'],paired_comparisons=comparisons,
            DEV_training_mean_MSE=np.mean(train_scores['mse'],axis=1),
            same_query_interaction={'population_scores':[g for g in evaluation['paired_interaction']['groups']if g['level']=='population'],
               'scale_scope':evaluation['paired_interaction']['scale_scope'],'auxiliary_only':True,'paired_queries':24},
            array_semantics={'full_write':'FP64 gated proxy same as original feature algebra; not native edited W',
               'cutoff':fit['information_cutoff'],'new_result_scope':'DEV-only refits evaluated on already-seen CONF; no new blind or causal mechanism PASS'},
            old192_results_replaced=False,old_shared48_QA_separate_from_new_family_scaling=True,
            conditional_interpretation_only=True,GPU_auto_GO=False)
        outputs=[]
        for f in sorted(out.iterdir()):
            need(f.is_file()and not f.is_symlink()and f.stat().st_size<=64*1024**2,'bounded_regular_cloud_output')
            outputs.append({'path':f.name,**file_id(f)})
        result['outputs']=outputs
    except BaseException as error:
        result['failure']={'phase':phase,'type':type(error).__name__,
            'code':str(error)if isinstance(error,(ValueError,RuntimeError))and len(str(error))<180 and all(c.isalnum()or c=='_'for c in str(error))else None,
            'frames':[{'file':Path(f.filename).name,'line':f.lineno,'function':f.name}for f in traceback.extract_tb(error.__traceback__)[-12:]]}
    result['elapsed_seconds']=time.monotonic()-started;result['phase_seconds']=times
    write_json(out/RESULT_NAME,result);need((out/RESULT_NAME).stat().st_size<=2*1024**2,'small_result_2MiB')
    print(json.dumps({'status':result['status'],'analysis_complete':result['analysis_complete'],'failure':result.get('failure')}),flush=True)
    return 0 if result['analysis_complete']else 2

if __name__=='__main__':raise SystemExit(main())
