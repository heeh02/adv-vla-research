"""Baidu-only bounded P08 geometry and source-bound sparse-time visualization."""
import argparse, collections, gzip, hashlib, importlib.util, json, os, re, sys, time, traceback
from pathlib import Path
import numpy as np
import p08_reader as P
from cloud_read import ExactDownload

PHASE='start'
def need(ok,code):
    if not ok:raise ValueError(code)
def sha(raw):return hashlib.sha256(raw).hexdigest()
def load(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
def mark(s):
    global PHASE
    PHASE=s;print(json.dumps({'phase':s}),flush=True)
def scalar(key,label,value,unit=''):
    return {'key':key,'label':label,'value':value,'unit':unit}
def serial(obj):return json.dumps(obj,ensure_ascii=False,sort_keys=True,allow_nan=False,separators=(',',':')).encode()

def run(root,deadline):
    root=Path(root).resolve(strict=True);meta=root/'package/metadata';own=root/'package/worker'
    req=json.loads((meta/'REQUEST-v1.json').read_bytes())
    need(sys.platform=='linux' and root==Path('/tmp/Jinwen/adv-vla')/req['run_id'],'owned_cloud_root')
    need(req['resources']=={'CPU':16,'GPU':0,'memory_GiB':64} and 180<deadline-time.time()<=1800,'CPU_runtime_scope')
    for name,ref in req['metadata'].items():
        raw=(meta/name).read_bytes();need(len(raw)==ref['bytes'] and sha(raw)==ref['sha256'],'frozen_metadata_SHA')
    cfg,planned=P.load_inputs(meta/'INPUTS-v2.json')
    allow={}
    for part in cfg['parts_to_read']:
        r=cfg['parts'][part];prefix=r['artifact_uri']+'payload/'
        allow[prefix+'MEMBER_MANIFEST.json']={'sha256':r['manifest_sha256'],'max_bytes':2_000_000}
        allow[prefix+'members.tar']={'sha256':r['tar_sha256'],'bytes':r['tar_bytes'],'max_bytes':r['tar_bytes']}
    need(len(allow)==80,'exact_selected_archive_readset')
    S=load('_p08_math',own/'statistics.py')
    split=json.loads((meta/'SPLIT-v1.json').read_bytes());examples=json.loads((meta/'HEATMAP-CASES-v1.json').read_bytes())
    fresh=S.freeze_metadata(planned);expected_examples=fresh.pop('heatmap_examples')
    need(fresh=={k:v for k,v in split.items() if k not in ('source_ledger','selection_basis')},'frozen_split_recomputed_from_ledger')
    selected={r['episode_id'] for r in examples['examples']};need(len(selected)==10 and expected_examples==examples['examples'],'frozen_heatmap_cases')
    mark('restore_verified_selected_shards_on_Baidu')
    downloader=ExactDownload(allow,deadline-240)
    try:
        records=[]
        for i,v in enumerate(P.iter_owners(meta/'INPUTS-v2.json',downloader,root/'selected-read',deadline-240)):
            records.append(v)
            if i%500==0:print(json.dumps({'owners_read':i+1,'planned':6720}),flush=True)
        proofs=dict(downloader.proofs)
    finally:downloader.close()
    need(len(records)==6720,'planned_anchors_retained')
    available=[r for r in records if r['status']=='selected_owner_verified']
    need(len(available)==6554 and len(proofs)==80,'selected_arrays_complete')
    mark('fixed_train_slot1_channel_bases')
    need(time.time()<deadline-300,'fit_time_remaining')
    fit=S.fit_bases([r['ledger'] for r in available],np.stack([r['Hdelta'] for r in available]),np.stack([r['Hclean'] for r in available]),split)
    out=root/'output';out.mkdir();files={}
    runtime=load('_p08_base_runtime',own/'v5_runtime.py');base=runtime.assemble(root,deadline)
    v5=base/'worker/repo/docs/experiments/first-paper-evidence-upgrade-20261008-v1/paired-evaluation-protocol-20261011-v2/submission-v1/cloud-readout-v5'
    sys.path.insert(0,str(v5));import bindings
    writer=bindings.load('bos_writer').BOSWriter(req['output_prefix'],deadline-15)
    def publish(name,value,compress=False,max_bytes=None):
        raw=serial(value)
        if max_bytes:need(len(raw)<=max_bytes,'bounded_viewer_payload')
        if compress:raw=gzip.compress(raw,mtime=0)
        p=out/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw);ref=writer.put(p,name);files[name]=ref;return ref
    arrays={};fit_meta={k:v for k,v in fit.items() if k!='layers'};fit_meta['layers']={}
    for layer,model in fit['layers'].items():
        arrays['L'+layer+'_center']=model['center'];arrays['L'+layer+'_clean_fit_center']=model['clean_fit_center'];fit_meta['layers'][layer]=model['metadata']
        for name,b in model['bases'].items():
            if b is not None:arrays['L'+layer+'_'+name]=b
    p=out/'FIXED-BASES-v1.npz';np.savez_compressed(p,**arrays);files[p.name]=writer.put(p,p.name)
    publish('FIT-METADATA-v1.json',fit_meta)
    mark('same_owner_dimensions_attention_and_sparse_geometry')
    readouts=[];by_anchor={};flat=[];attn_rows=[]
    for i,r in enumerate(available):
        need(time.time()<deadline-160,'describe_time_remaining')
        d=S.describe_owner(r['ledger'],r['Hdelta'],r['Hclean'],fit)
        a=S.attention_summary(r['Aclean'],r['Aactual'],r['validity']['clean_source_mass'],r['validity']['actual_source_mass'],r['validity']['source_mass_delta'])
        d['attention']=a['fixed8'];d['attention_heads']=a['heads'];by_anchor[d['anchor_id']]=d;readouts.append(d)
        for layer,v in d['layers'].items():
            row={**r['ledger'],'layer':layer,**{k:v['metrics'][k] for k in ('response_rms','relative_rms','coordinate_PR_channel_energy_pooled_positions','mean_only_one_minus_raw_normalized_error')}}
            for name,c in v['controls'].items():
                row[name+'_retention']=c['retention_centered'];row[name+'_raw_score']=c['one_minus_raw_normalized_error']
            E=v['metrics']['raw_response_energy'];cs=v['controls']
            for ref in ('clean','random'):
                row['gain_over_'+ref]=(cs[ref]['sse']-cs['response']['sse'])/E if E and cs[ref]['sse'] is not None and cs['response']['sse'] is not None else None
            flat.append(row)
        for arow in a['fixed8']:attn_rows.append({**r['ledger'],**arow})
    for r in records:
        if r['status']!='selected_owner_verified':
            for layer in map(str,S.LAYERS):flat.append({**r['ledger'],'layer':layer})
            for layer in S.LAYERS:
                for camera in ('main','wrist'):attn_rows.append({**r['ledger'],'layer':layer,'camera':camera})
    limits=S.frozen_axis_limits(readouts)
    groups=collections.defaultdict(list)
    for r in records:groups[r['owner']['episode_id']].append(r)
    # An independently collected clean rollout is explicitly distinct from
    # the clean rerender at a condition-rollout's current visited state.
    clean_outcomes={}
    for eid,rs in groups.items():
        r=rs[0]
        if r['owner']['history_condition']=='clean':
            key=(r['owner']['root'],r['owner']['noise_repeat']);need(key not in clean_outcomes,'unique_clean_outcome')
            clean_outcomes[key]={'episode_id':eid,'success':r['owner']['original_success'],'result_sha256':r['ledger']['original_result_sha256']}
    spaces=[];domains={str(l):{'hidden_delta':0.,'attention_probability':0.,'attention_delta':0.} for l in S.LAYERS}
    for r in available:
        if r['owner']['episode_id'] in selected:
            for li,l in enumerate(S.LAYERS):
                z=domains[str(l)];z['hidden_delta']=max(z['hidden_delta'],float(np.max(np.abs(r['Hdelta'][li]))));z['attention_probability']=max(z['attention_probability'],float(max(np.max(r['Aclean'][li]),np.max(r['Aactual'][li]))));z['attention_delta']=max(z['attention_delta'],float(np.max(np.abs(r['Aactual'][li]-r['Aclean'][li]))))
    for layer,v in fit['layers'].items():
        d=domains[layer];m=v['metadata']
        spaces.append({'id':'AE-L'+layer+'-position-centered-response','layer':int(layer),'object':'hidden_delta','basis_id':'response-L'+layer,
          'basis_sha256':m['basis']['response']['sha256'],'fit_population':'D3 old task folds0/1/2; slot1; persistent10; task-condition-root-noise-position equal',
          'center':'training slot1 mean per action position; SHA '+m['position_center_sha256'],'scale':'none','sign_convention':m['sign_rule'],
          'source_dimension':1024,'analysis_rank':16,'display_rank':3,'axis_labels':['Response PC1','Response PC2','Response PC3'],
          'axis_limits':limits.get(layer),'basis_status':m['basis']['response']['status'],
          'color_domains':{'hidden_delta':[-(d['hidden_delta']or 1.),d['hidden_delta']or 1.],'attention_probability':[0,d['attention_probability']or 1.],'attention_delta':[-(d['attention_delta']or 1.),d['attention_delta']or 1.]}})
    cases=[];heat_bytes=0;pair_rows=[]
    labels={'response_rms':'原始响应 RMS','relative_rms':'相对 clean RMS','coordinate_PR_channel_energy_pooled_positions':'通道能量参与率'}
    for eid,rs in sorted(groups.items()):
        rs=sorted(rs,key=lambda r:r['owner']['slot']);o=rs[0]['owner'];reference=clean_outcomes.get((o['root'],o['noise_repeat']));actual=o['original_success'];cy=reference['success'] if reference else None
        outcome={'actual':{'value':bool(actual) if actual is not None else None,'episode_id':eid,'result_sha256':rs[0]['ledger']['original_result_sha256']},
          'reference':{'value':bool(cy) if cy is not None else None,'source':reference},
          'paired_status':'qualified' if cy is not None and actual is not None else 'missing',
          'pairing_rule':'original D3 root(initial state and task)+noise-repeat; reference is the independent clean rollout',
          'neural_pair_role':'clean rerender at same visited state, not independent clean rollout'}
        case={'id':eid,'episode_id':eid,'task_id':o['task_id'],'task_label':o['task_id'],'condition':o['history_condition'],'fold':o['task_fold'],
              'model':'pi05_libero','checkpoint':None,'cohort':'D3/P08','prompt':None,'outcome':outcome,'heatmap_available':eid in selected,
              'heatmap_href':'cases/'+sha(eid.encode())+'.json.gz' if eid in selected else None,'slots':[],'pair_metrics':{}}
        heat={'schema':'p08-heatmap-v1','data_class':'observed','case_id':eid,'slots':[]}
        for r in rs:
            o=r['owner'];d=by_anchor.get(o['anchor_id']);slot={'slot':o['slot'],'status':'observed' if d else 'missing','reason':'' if d else r['status'],'owner_id':o['anchor_id'],
              'query_index':o['query_index'],'control_step':o['control_step'],'sensor_time':o['sensor_time'],'j':0,'t':1,
              'identities':{'selected_npz_sha256':r['ledger']['selected_file_sha256'],'source_cache_sha256':r['ledger']['cache_sha256'],'trace_sha256':r['ledger']['trace_sha256'],'noise_label':o['noise_repeat']},'layers':{},'controls':[]}
            hs={'slot':o['slot'],'owner_id':o['anchor_id'],'status':slot['status'],'layers':{}}
            if d:
                for li,layer in enumerate(map(str,S.LAYERS)):
                    v=d['layers'][layer];metrics=[scalar(k,label,v['metrics'][k])for k,label in labels.items()]
                    metrics.extend(scalar(k+'_retention16',{'response':'响应','clean':'clean','random':'随机'}[k]+'基底保留率',c['retention_centered'])for k,c in v['controls'].items())
                    metrics.extend(scalar('attention_'+a['camera'],'相机 '+a['camera']+' 无条件半L1',a['unconditional_camera_half_L1'])for a in d['attention'] if a['layer']==int(layer))
                    metrics.extend(scalar('plan_'+c,'计划前缀 '+c+' 差异RMS',o['physical_prefix_'+c+'_rms'])for c in ('translation','rotation','gripper'))
                    projected=[{k:p[k] for k in ('position','xyz','retention3','retention16')} for p in v['projection']['points']]
                    slot['layers'][layer]={'space_id':'AE-L'+layer+'-position-centered-response','projection':{'points':projected},'metrics':metrics if eid in selected else metrics[:3],
                      'position_metrics':[]}
                    if eid in selected:
                        hs['layers'][layer]={'space_id':slot['layers'][layer]['space_id'],'hidden_delta':r['Hdelta'][li].ravel().tolist(),
                           'attention':{'clean':r['Aclean'][li].ravel().tolist(),'actual':r['Aactual'][li].ravel().tolist(),
                           'valid_clean':r['validity']['clean_source_mass'][li,:,:2].ravel().tolist(),'valid_actual':r['validity']['actual_source_mass'][li,:,:2].ravel().tolist(),
                           'metrics':[{'head':a['head'],'camera':a['camera'],'clean_mass':a['camera_mass_clean'],'actual_mass':a['camera_mass_actual'],'delta_mass':a['delta_camera_mass']}for a in d['attention_heads'] if a['layer']==int(layer)]}}
                if eid in selected:slot['controls'].append({'kind':'raw_flow_delta','owner_id':o['anchor_id'],'shape':[5,32],'values':r['flow_delta'].ravel().tolist(),'source_sha256':r['source']['selected_file_sha256'],'source':'selected native flow; not robot velocity'})
            case['slots'].append(slot)
            if eid in selected:heat['slots'].append(hs)
        if all(r['status']=='selected_owner_verified' for r in rs):
            pair=S.describe_pair(rs[0]['Hdelta'],rs[1]['Hdelta']);pair_rows.append({'episode_id':eid,**pair})
            case['pair_metrics']={l:[scalar('full_response_cosine','两观测时点完整响应余弦',v['full_position_channel_cosine']['value'])]for l,v in pair.items()}
        if eid in selected:
            heat_bytes+=len(serial(heat));need(heat_bytes<=16*1024**2,'all_heatmap_bytes_bound')
            publish('viewer/'+case['heatmap_href'],heat,compress=True)
        cases.append(case)
    mark('heldout_condition_layer_summaries')
    heldtasks=[r['task_key']for r in split['tasks'] if r['role']=='heldout'];summaries=[]
    metrics=['response_rms','relative_rms','coordinate_PR_channel_energy_pooled_positions','mean_only_one_minus_raw_normalized_error',
             'response_retention','clean_retention','random_retention','response_raw_score','clean_raw_score','random_raw_score','gain_over_clean','gain_over_random']
    for condition in split['history_conditions']:
        for layer in map(str,S.LAYERS):
            for slot in (1,2):
                rr=[r for r in flat if r['history_condition']==condition and r['layer']==layer and int(r['slot'])==slot and S.task_key(r)in heldtasks]
                summaries.append({'condition':condition,'layer':int(layer),'slot':slot,'population':'heldout16tasks','planned_rows':sum(r['history_condition']==condition and int(r['slot'])==slot and S.task_key(r)in heldtasks for r in planned),
                  'metrics':{f:S.summarize_fixed_metric(rr,f,heldtasks)for f in metrics}})
    bypair=collections.defaultdict(dict)
    for r in flat:bypair[(r['episode_id'],r['layer'])][int(r['slot'])]=r
    paired_differences=[]
    for (eid,layer),pair in bypair.items():
        need(set(pair)=={1,2},'paired_slot_scalar_support')
        r={**pair[1],'slot':'2minus1'}
        for f in metrics:r[f]=pair[2].get(f)-pair[1].get(f) if pair[2].get(f)is not None and pair[1].get(f)is not None else None
        paired_differences.append(r)
    slot_summaries=[];attention_summaries=[]
    for condition in split['history_conditions']:
        for layer in map(str,S.LAYERS):
            rr=[r for r in paired_differences if r['history_condition']==condition and r['layer']==layer and S.task_key(r)in heldtasks]
            slot_summaries.append({'condition':condition,'layer':int(layer),'contrast':'slot2-minus-slot1','population':'heldout16tasks','metrics':{f:S.summarize_fixed_metric(rr,f,heldtasks)for f in metrics}})
            for slot in (1,2):
                for camera in ('main','wrist'):
                    rr=[r for r in attn_rows if r['history_condition']==condition and r['layer']==int(layer) and int(r['slot'])==slot and r['camera']==camera and S.task_key(r)in heldtasks]
                    attention_summaries.append({'condition':condition,'layer':int(layer),'slot':slot,'camera':camera,'population':'heldout16tasks','metrics':{f:S.summarize_fixed_metric(rr,f,heldtasks)for f in ('camera_mass_clean','camera_mass_actual','delta_camera_mass','unconditional_camera_half_L1')}})
    publish('cloud-only/OWNER-READOUT-v1.json.gz',readouts,compress=True)
    publish('cloud-only/ATTENTION-SCALARS-v1.json.gz',attn_rows,compress=True)
    publish('cloud-only/PAIR-READOUT-v1.json.gz',pair_rows,compress=True)
    publish('HELDOUT-SUMMARY-v1.json',summaries)
    publish('PAIRED-SLOT-SUMMARY-v1.json',slot_summaries)
    publish('ATTENTION-SUMMARY-v1.json',attention_summaries)
    viewer={'schema':'p08-viewer-v1','data_class':'observed','dataset_id':'D3-P08-two-observation-anchors','spaces':spaces,'cases':cases,
            'support':{'planned_cases':3360,'planned_anchors':6720,'available_anchors':6554,'missing_anchors':166,'heatmap_cases':10},
            'scope':'native AE response plus shadow unconditional camera attention; j0 only; two physical observations; geometry is reconstruction, not outcome prediction'}
    publish('viewer/manifest.json.gz',viewer,compress=True,max_bytes=32*1024**2)
    publish('INPUT-READ-RECEIPT-v1.json',{'objects':proofs,'raw_arrays_to_Mac':False,'selected_source_archives_only':True})
    result={'schema':'p08-dynamic-result-v1','run_id':req['run_id'],'planned_case_count':3360,'planned_anchors':6720,'available_anchors':6554,'missing_anchors':166,
       'GPU':0,'model_calls':0,'new_episodes':0,'new_training':0,'raw_arrays_to_Mac':False,'methods_frozen_before_arrays':True,
       'fit_metadata':fit_meta,'heldout_summaries':summaries,'paired_slot_summaries':slot_summaries,'attention_summaries':attention_summaries,'output_files':dict(files),'heatmap_uncompressed_bytes':heat_bytes,
       'three_dimensional_view_does_not_establish_manifold':True}
    ref=publish('RESULT-v1.json',result)
    publish('DELIVERY-v1.json',{'schema':'p08-dynamic-delivery-v1','run_id':req['run_id'],'result':ref,'GPU':0})
    print(json.dumps({'status':'complete','run_id':req['run_id'],'result':ref}),flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--execute-cloud',action='store_true');p.add_argument('--root');p.add_argument('--deadline',type=float);a=p.parse_args()
    if not a.execute_cloud:print(json.dumps({'status':'dry_no_cloud_calls'}));return 0
    try:run(a.root,a.deadline);return 0
    except BaseException as e:
        code=e.args[0] if len(e.args)==1 and isinstance(e.args[0],str) and re.fullmatch(r'[A-Za-z0-9_]{1,100}',e.args[0]) else 'unclassified_safe'
        frames=[{'file':Path(x.filename).name,'line':x.lineno,'function':x.name} for x in traceback.extract_tb(e.__traceback__)]
        print(json.dumps({'status':'failed','phase':PHASE,'error_type':type(e).__name__,'code':code,'frames':frames[-8:]}),flush=True);return 2
if __name__=='__main__':raise SystemExit(main())
