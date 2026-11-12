"""Constructed scalar-only fixture; never discovers or opens actual inputs."""
from pathlib import Path
from itertools import product
import argparse
import copy
import hashlib
import json
import analysis as A
import adapter


def comparison(delta, reference=1., candidate=None):
    reference=0. if reference==0 else reference
    candidate=(reference+delta/2 if reference else delta) if candidate is None else candidate
    exact=delta==0
    desc={'shape':[10], 'logical_dtype':'float64', 'storage_dtype':'float64',
          'bytes':80, 'payload_sha256':'a'*64}
    candidate_desc=copy.deepcopy(desc)
    if not exact:candidate_desc['payload_sha256']='b'*64
    return {'reference':desc,'candidate':candidate_desc,'typed_exact':exact,
            'raw_payload_exact':exact,'numeric_equal':exact,
            'numeric_nonzero_differences':0 if exact else 10,'max_abs':delta*2,
            'RMS_difference':delta,'RMS_reference':reference,'RMS_candidate':candidate,
            'relative_RMS_to_reference':delta/reference if reference else None,
            'relative_RMS_to_candidate':delta/candidate if candidate else None,
            'reference_zero':reference==0,'candidate_zero':candidate==0}


def fixture():
    fields=[];features=[];endpoints=[]
    for si, key in enumerate(A.KEYS):
        base={'root_id':key[0],'query_index':key[1]}
        for fi, name in enumerate(A.FIELDS):
            delta=.001*(si+1)*(fi+1) if name in ('T_v0','T_x5','T_v5','T_P_j5_L9') else 0.
            fields.append({**base,'field':name,**comparison(delta)})
        for fi,name in enumerate(A.FEATURES):
            delta=.002*(si+1)*(fi+1) if name in ('current_headpos','current_full_write') else 0.
            features.append({**base,'feature':name,
                'scope':'explicitly_synthetic_fixture','mixed_units_physical_magnitude':False,
                'magnitude_scope':'numeric_algebra_or_cache_QA_only',**comparison(delta)})
        for c,w,(di,d) in product(A.COMPARISONS,A.WINDOWS,enumerate(A.DIMS)):
            delta=.001*(si+1)*(di+1)
            row={**base,'comparison':c,'window':w,'dimension':d,**comparison(delta)}
            if c=='persistent_effect_own_baselines':
                s=delta/(.5+si*.3);sr=delta/(.8+si*.2)
                if si==0 and di==0:s=0.
                if si==1 and di==1:sr=0.
                if si==2 and di==2:
                    row.update(comparison(0.));delta=0.
                row.update(new_S_RMS=s,new_S_minus_R_RMS=sr,
                    residual_RMS_over_new_S_RMS=delta/s if s else None,
                    residual_RMS_over_new_S_minus_R_RMS=delta/sr if sr else None,
                    new_S_zero=s==0,new_S_minus_R_zero=sr==0)
                if si==3 and di==3:
                    row['new_S_RMS']=delta/(1-4e-14)
                    row['residual_RMS_over_new_S_RMS']=1-4e-14
            endpoints.append(row)
        for c,d in product(('none_raw32','persistent_raw32'),range(32)):
            endpoints.append({**base,'comparison':c,'window':'full10',
                'dimension':f'normalized_raw_coordinate_{d}',**comparison(.001*(si+1)*(d+1))})
    counts={k:A.COUNTS[k] for k in ('snapshots','native_field_comparisons','feature_comparisons','endpoint_comparisons')}
    producer={'schema':'paper2-dev16-feature-bridge-result-v1','status':'DEV16_FEATURE_BRIDGE_COMPLETE',
        'synthetic':True,'analysis_complete':True,'input_qualified':True,'snapshots':8,
        'new_model_calls':0,'new_env_steps':0,'new_fits':0,'comparison_counts':counts,
        'whole_graph_equivalence':'UNMEASURED','prediction_validated':False}
    qa={'schema':'paper2-dev16-feature-bridge-QA-v1','synthetic':True,'pass':True,
        'all8_retained':True,'native72_final_lists_checked':True,'new_N_P_checks_events_bound':16,
        'counts':counts,'new_model_calls':0,'new_env_steps':0,'new_fits':0,
        'fresh_C_context':'UNMEASURED','whole_graph_equivalence':'UNMEASURED'}
    protocol={'statistical_or_physical_acceptance_threshold':None}
    return fields,features,endpoints,producer,qa,protocol


def stage_fixture(path):
    stage=Path(path);stage.mkdir(parents=True,exist_ok=False)
    fields,features,endpoints,producer,qa,protocol=fixture()
    refs=[]
    for name,value in zip(adapter.INPUT_NAMES[:3]+('QA.json','PROTOCOL.json'),
                           (fields,features,endpoints,qa,protocol)):
        pin=adapter.write(stage/name,value)
        refs.append({'path':name,'uri':'redacted-0ec71650eb2d'+name,**pin})
    producer['outputs']=[{k:r[k] for k in ('path','bytes','sha256')} for r in refs]
    name='dev16-feature-bridge-result.json';pin=adapter.write(stage/name,producer)
    refs.append({'path':name,'uri':'redacted-0ec71650eb2d'+name,**pin})
    archive={'run_id':'jinwen-paper2-dev16-feature-bridge-20261020-v2',
             'archive_complete':True,'dev16_feature_bridge_analysis_complete':True,
             'science_archive':{'complete_for_observed_files':True,'errors':[],
                               'objects':[{'name':r['path'],**r} for r in refs]}}
    name='ARCHIVE-RECEIPT.json';pin=adapter.write(stage/name,archive)
    refs.append({'path':name,'uri':'redacted-0ec71650eb2d'+name,**pin})
    return {'input_objects':refs,'input_object_count':7,'input_body_bytes':sum(r['bytes'] for r in refs)}


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);args=p.parse_args()
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    data=A.aggregate(*fixture(),synthetic=True)
    import renderer
    result=renderer.render(data,out)
    adapter.write(out/'SYNTHETIC-SUMMARY.json',{**data['summary'],'figure':result})
    adapter.support_csv(data['support'],out/'FULL-SUPPORT.csv.gz')
    print(json.dumps({'synthetic':True,'layout':result['layout'],'support_rows':len(data['support'])}))


if __name__=='__main__':main()
