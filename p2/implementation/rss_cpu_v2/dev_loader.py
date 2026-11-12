"""Read only the exact historical DEV corpus; no learned feature transform.

The byte-pinned native reader verifies original inputs, actual edit selectors,
own-null/repeat gates, native edit identity and signed endpoint labels. All
real neural-array handling is Linux-cloud-only. The old cache is a QA oracle,
never a source of fold-fitted PCA, context or pooled reading.
"""
from pathlib import Path
from contextlib import contextmanager
import hashlib, importlib.util, json, sys, time
import numpy as np

HERE=Path(__file__).resolve().parent

def need(ok,code):
    if not ok: raise ValueError(code)

def digest(p):
    raw=Path(p).read_bytes()
    return {'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}

def same_bytes(a,b):
    return a.shape==b.shape and a.dtype==b.dtype and a.tobytes(order='C')==b.tobytes(order='C')

@contextmanager
def modules():
    names=('ablation','sources','prediction','reader')
    previous={n:sys.modules.get(n)for n in names};oldpath=list(sys.path)
    folder=HERE/'legacy'
    try:
        pins=json.loads((HERE/'LEGACY-SOURCE-PINS.json').read_text())
        for p in pins['files']:
            need(digest(folder/p['name'])=={k:p[k]for k in('bytes','sha256')},'byte_identical_legacy_source')
        sys.path.insert(0,str(folder))
        for name in ('ablation','sources'):
            spec=importlib.util.spec_from_file_location(name,folder/(name+'.py'))
            m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m)
        S=sys.modules['sources']
        D=S.dependencies({name:folder/name for name in S.PINS})
        yield D
    finally:
        sys.path[:]=oldpath
        for n,v in previous.items():
            if v is None:sys.modules.pop(n,None)
            else:sys.modules[n]=v

def load(stage,inputs,deadline_epoch,*,synthetic=False):
    need(synthetic or sys.platform.startswith('linux'),'real_arrays_only_cloud_Linux')
    stage=Path(stage).resolve();refs={x['uri']:x for x in inputs['input_objects']}
    plan=json.loads((HERE/'inventory/DEV-READ-PLAN-original.json').read_text())
    read_receipts=[]
    def check():need(time.time()<deadline_epoch,'absolute_science_deadline')
    def path(ref):
        check();bound=refs.get(ref['uri'])
        need(bound and all(bound[k]==ref[k]for k in('bytes','sha256')),'exact_frozen_input_ref')
        p=stage/bound['path'];need(p.parent==stage and p.is_file()and not p.is_symlink(),'flat_regular_staged_file')
        need(digest(p)=={k:bound[k]for k in('bytes','sha256')},'whole_input_SHA')
        read_receipts.append({k:bound[k]for k in('path','uri','bytes','sha256')})
        return p
    with modules()as D:
        P,R,W=D.P,D.R,D.W
        original=W.validate_plan(plan);groups={};extras={};calls=[];comparisons=[]
        actual_selectors=[]
        for ref in original:
            check();rec=R.extract(path(ref),ref);ev,m=rec['event'],rec['metadata']
            if ev in('CAMPAIGN_PLANNED','READ_ONLY_WO','RESULT'):
                need(ev not in extras,'unique_global_record');extras[ev]=rec
            elif ev=='EXACT_COMPARISON':comparisons.append(rec)
            else:
                key=(m['root_id'],m['query_index'])
                need(key in {(r,q)for r in P.ROOTS for q in range(11,23)},'exact_DEV_query_population')
                slot=m['role']if ev=='CALL_OUTPUT'else 'label_'+m['arm']if ev=='ACTUAL_EFFECT'else ev
                group=groups.setdefault(key,{})
                need(slot not in group,'unique_query_slot');group[slot]=rec
                if ev=='CALL_OUTPUT':
                    calls.append(rec)
                    if m['role'].startswith('effect_'):
                        sel=m['technical']['selectors']
                        need(sel['flow_mask']==[False]*5+[True]*5,'actual_persistent_selector')
                        actual_selectors.append({'root_id':key[0],'query_index':key[1],'arm':m['role'][7:],
                                                 'source':R.receipt(ref),'selectors':sel})
        gates=W.original_gates(extras['RESULT']['metadata'],comparisons,calls)
        planned=extras['CAMPAIGN_PLANNED']['metadata']
        need(planned['synthetic']is False or synthetic,'real_original_plan')
        p=P.Plan(readset_sha256=P.sha(P.encoded(plan)),native_source_sha256=plan['source_binding']['source_manifest_sha256'],
                 intervention_contract_sha256=P.sha(P.encoded(planned['plan'])))
        wo=extras['READ_ONLY_WO']['values']['WO'];byquery=[];identities=[];native_queries=[]
        for key in p.keys():
            check();q,targets=R.build_query(groups[key],p);byarm={t.arm:t for t in targets}
            need(set(byarm)==set(P.ARMS),'complete_four_arm_labels')
            qid=q.fingerprint();edits={a:q.edit(a).astype(np.float64)for a in P.ARMS}
            row={'root_id':q.root_id,'cluster_id':q.root_id,'task_id':q.root_id.split('-')[2],
                 'query_index':q.query_index,'source_identity':qid,
                 'state_vector':q.state_vector(),'first_flow_delta':q.first_flow_vector(),
                 'time_scalar':q.env_step_index/520.,'p':q.p_native.value()[0,0].astype(np.float64),
                 'mask':q.mask.value()[0].astype(bool),'gate':q.gate.value()[0,0].astype(np.float64),
                 'edits':edits,'targets':{a:byarm[a].effect.copy()for a in P.ARMS},
                 'zero_actual_E':{a:not np.any(edits[a])for a in P.ARMS},
                 'edit_magnitude':{a:float(np.linalg.norm(edits[a]))for a in P.ARMS},
                 'provenance':{'query_fingerprint':qid,'input_source_identity':q.source_identity,
                     'natural_C_source':{k:groups[key]['natural_C']['ref'][k]for k in('uri','path','bytes','sha256')},
                     'natural_T_source':{k:groups[key]['natural_T']['ref'][k]for k in('uri','path','bytes','sha256')},
                     'frozen_edit_source':{k:groups[key]['FROZEN_QUERY_EDITS']['ref'][k]for k in('uri','path','bytes','sha256')},
                     'input_audit_source':{k:groups[key]['GPU_INPUT_AUDIT']['ref'][k]for k in('uri','path','bytes','sha256')},
                     'parent_source':groups[key]['GPU_INPUT_AUDIT']['metadata']['source'],
                     'edit_identities':{a:q.edit_identity(a)for a in P.ARMS},
                     'target_identities':{a:{'source_identity':byarm[a].source_identity,'array':P.array_id(byarm[a].effect)}for a in P.ARMS}}}
            byquery.append(row);native_queries.append(q)
            identities.append({'root_id':q.root_id,'query_index':q.query_index,'cluster_id':q.root_id,
                               'task_id':row['task_id'],'schedule':'persistent_j5_to_j9','arms':list(P.ARMS),
                               **row['provenance']})
        need(len(byquery)==48 and len(actual_selectors)==192,'complete48_queries192_persistent_labels')
        cache_ref=next(x for x in inputs['input_objects']if x['uri'].endswith('/qualified-feature-cache.npz'))
        identity_ref=next(x for x in inputs['input_objects']if x['uri'].endswith('/row-identities.json'))
        cache_path=path(cache_ref);old_ids=json.loads(path(identity_ref).read_text())
        old_map={tuple(x['key']):x for x in old_ids};qa=[]
        with np.load(cache_path,allow_pickle=False)as cache:
            cache_keys=[tuple([str(k[0]),int(k[1]),str(k[2]),str(k[3])])for k in cache['row_keys']]
            need(len(cache_keys)==len(set(cache_keys))==168,'original_cache_shared_population168')
            idx={k:i for i,k in enumerate(cache_keys)}
            for q,row in zip(native_queries,byquery):
                check();key=(q.root_id,q.query_index,'shared','KT');i=idx[key]
                need(cache['split'][i]=='DEV','cache_QA_DEV_only')
                need(same_bytes(cache['effect'][i],row['targets']['shared']),'old_shared_target_bitwise_same')
                want=P.local_write(row['p'],row['edits']['shared'],wo,q.gate).reshape(10,1024)
                need(same_bytes(cache['current_gated_write'][i],want),'old_current_write_bitwise_same')
                need(old_map[key]['provenance']['input_identity']==q.fingerprint(),'original_query_fingerprint_matches_cache')
                qa.append({'key':list(key),'target_bitwise_equal':True,'current_write_bitwise_equal':True,'input_identity_equal':True})
        dataset={'schema':'rss-raw-DEV-v1','schedule':'persistent_j5_to_j9','arms':list(P.ARMS),
                 'wo':wo.value().astype(np.float64),'queries':byquery,
                 'root_independence':'distinct historical root identities; init-vector verification supplied separately; deterministic roots, not iid sampling'}
        proof={'schema':'rss-DEV-loader-proof-v1','real_source':not synthetic,'source_run_id':plan['source_run_id'],
               'source_binding':plan['source_binding'],'original_gates':gates,'original_readset_identity':digest(HERE/'inventory/DEV-READ-PLAN-original.json'),
               'actual_query_count':48,'actual_arm_labels':192,'actual_persistent_selectors':actual_selectors,
               'dual_pulse_labels_verified_in_this_DEV_corpus':0,'shared_pulse':0,'random_matched_pulse':0,
               'pulse_status':'NO_PULSE_IN_EXACT_REVIEWED_DEV_CORPUS; not a claim of project-wide nonexistence',
               'all_rows':identities,'cache_QA':qa,'old_learned_transforms_used_for_new_folds':False,
               'input_receipts':read_receipts,'old_frozen_results_changed':False,
               'raw_context_and_P_recovered_for_train_fold_reconstruction':True,
               'model_calls':0,'env_steps':0,'GPU':0}
        return dataset,proof
