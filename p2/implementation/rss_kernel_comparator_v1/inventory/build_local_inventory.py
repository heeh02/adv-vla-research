"""Local metadata/source audit only. No array decode, model, credential, network, or cloud call."""
from pathlib import Path
import collections, hashlib, json, subprocess

MAIN=Path('/Users/researcher/Adv-VLA-π0.5/docs/experiments/paper2-independent-session-20261019-v1')
OLD=Path('/Users/researcher/.codex/worktrees/adv-vla-research-20260924/Adv-VLA-π0.5/docs/experiments/state-dependent-response-use-20261013-v1')
OUT=MAIN/'implementation/rss_cpu_v1/inventory'
LEG=MAIN/'implementation/signed_transfer_v1/input_adapter/legacy'
BASE='redacted-2ac5a54c7015'
footprint={}
def ident(raw): return {'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
def read(path):
 path=Path(path);raw=path.read_bytes();footprint[str(path)]={'path':str(path),**ident(raw),'type':'JSON_metadata'}
 return json.loads(raw)
def source(path):
 path=Path(path);raw=path.read_bytes();footprint[str(path)]={'path':str(path),**ident(raw),'type':'source_text'}
 return raw

def put(name,value):
 p=OUT/name;raw=(json.dumps(value,ensure_ascii=False,sort_keys=True,indent=2)+'\n').encode();p.write_bytes(raw)
 print(name, len(raw),hashlib.sha256(raw).hexdigest())

def minimal_ref(r):return {k:r[k]for k in ('path','uri','bytes','sha256')if k in r}

dev=read(LEG/'metadata/dev/READ-PLAN-v1.json');cloud=read(LEG/'metadata/dev/source-cloud.json');index=read(LEG/'metadata/dev/source-index.json')
request=read(OLD/'continuous/l9-intervention-root-v3/REQUEST-v1.json')
source_manifest=read(OLD/'continuous/l9-intervention-root-v3/prepared-gpu-v1/SOURCE_MANIFEST-v1.json')
source_rows={r['path']:r for r in source_manifest['files']}
worker_path=OLD/'continuous/l9-intervention-worker-v1/worker.py';worker=source(worker_path)
worker_pin=source_rows['study/continuous/l9-intervention-worker-v1/worker.py']
assert ident(worker)=={k:worker_pin[k]for k in ('bytes','sha256')}
assert request['flow_indices']==list(range(5,10))
assert 'np.arange(10)>=5' in worker.decode()
for n in ['dev_reader.py','dev_worker.py','taskblind_prediction.py','retained_prediction.py','retained_replication.py','sources.py','helper.py']:
 source(LEG/'worker/core'/n)
for n in ['continuous/l9-intervention-control-v1/contract.py','continuous/l9-effect-readout-v1/readout.py','continuous/cloud-v1/native_runtime.py','execution/shared-response-edit-v1/shared_response_query.py','protocol/native-effect-v1/PROTOCOL-v1.md']:
 source(OLD/n)

member_by={(r['path'],r['bytes'],r['sha256']):r for a in index['archives']for r in a['members']}
role_count=collections.Counter(r['role']for r in cloud['worker_result']['call_records'])
call_refs=[]
for r in cloud['worker_result']['call_records']:
 if r['role'] in ['natural_T','effect_shared','effect_random_matched']:
  ref=r['receipt'];full=member_by[(ref['path'],ref['bytes'],ref['sha256'])]
  call_refs.append({'root_id':r['root_id'],'query_index':r['query_index'],'call_id':r['call_id'],'role':r['role'],'status':r['status'],'schedule':'none'if r['role']=='natural_T'else'persistent_j5_to_j9','schedule_evidence':'source-proven locally; actual selectors verified by retained dev_reader on cloud full read','object':minimal_ref(full),'member_path':full['path']})
put('DEV-CALL-IDENTITY-MAP-v1.json',{'schema':'rss-DEV-call-identity-v1','source_run_id':dev['source_run_id'],'source_binding':dev['source_binding'],'calls':call_refs,'role_counts':dict(role_count),'N_endpoint_shape':[1,10,7],'effect_shape':[10,7],'first5':'slice positions0..4 only after full signed target validation','local_neural_array_reads':0})

# Actual parent-root initial vectors are independently identifiable from native receipts.
candidates=read(OLD/'continuous/protocol/TASK-ROOT-CANDIDATES-v1.json')
candidate_by={r['root_trajectory_slot_id']:r for r in candidates['candidate_parent_rollouts']}
cluster_rows=[];cluster_inputs=[]
for tag in ['t00r00','t00r01','t01r00','t01r01']:
 folder=OLD/f'continuous/actual-{tag}-v1';summary=read(folder/'CLOUD-RESULT-original-v1.json');idx=read(folder/'EXACT-OBJECT-INDEX-v2.json')
 sel=summary['worker_result']['actual_initial_selection'];root=summary['root_id'];assert sel['root_trajectory_id']==root
 members=[r for q in idx['queries']for r in q['members']]
 choose=[r for r in members if r['path'].endswith('ACTUAL_ROOT_INITIAL_VECTOR.npz')or r['path'].endswith('ROOT_RUNTIME_BINDING.json')]
 assert len(choose)==2
 entry={'root_id':root,'task_component_id':candidate_by[root]['component_id'],'parent_task_id':candidate_by[root]['parent_task_id'],
   'source_run_id':summary['run_id'],'parent_observation_role':'clean_parent','parent_noise_protocol':summary['worker_result']['parent'].get('noise_protocol'),
   'source_library':sel['source_library'],'original_library_index':sel['original_library_index'],'unique_vector_rank':sel['unique_vector_rank'],
   'native_initial_vector':sel['native_initial_vector'],'selection_rule':sel['rule'],
   'cluster_key':[candidate_by[root]['component_id'],sel['source_library']['sha256'],sel['original_library_index'],sel['native_initial_vector']['sha256']],
   'actual_summary':footprint[str(folder/'CLOUD-RESULT-original-v1.json')], 'exact_index':footprint[str(folder/'EXACT-OBJECT-INDEX-v2.json')],
   'small_objects':[minimal_ref(r)for r in choose], 'status':'ACTUAL_METADATA_DISTINCT_INITIALIZATION; PAYLOAD_CPU_RECHECK_PENDING'}
 cluster_rows.append(entry)
 for ref in choose:cluster_inputs.append({'root_id':root,'stage_name':root+'__'+ref['path'],**minimal_ref(ref)})
for task in {r['task_component_id']for r in cluster_rows}:
 rr=[r for r in cluster_rows if r['task_component_id']==task];assert {r['original_library_index']for r in rr}=={0,1};assert len({r['native_initial_vector']['sha256']for r in rr})==2
put('ROOT-CLUSTER-MAP-v1.json',{'schema':'rss-CDEV-root-initialization-identity-v1','roots':cluster_rows,'known_distinct_initialization_clusters':4,'tasks':2,
 'inference_limit':'Two deterministically selected distinct vectors per task, not proof of IID sampling or reliable four-root power estimation. Same-task temporal/arm/noise records remain grouped. Missing/failed payload checks must preserve unknown/conflict rather than force four clusters.',
 'hash_domain':'plus-policy-native-array-v1; SHA256(compact sorted JSON(shape,dtype,order=C)+NUL+C-order payload); raw payload SHA separate',
 'arrays_decoded_locally':False})
put('ROOT-CLUSTER-VERIFY-READ-PLAN-v1.json',{'schema':'rss-CDEV-root-small-verification-plan-v1','input_objects':cluster_inputs,'count':len(cluster_inputs),'input_body_bytes':sum(x['bytes']for x in cluster_inputs),
 'initial_vector_objects':4,'initial_vector_body_bytes':sum(x['bytes']for x in cluster_inputs if x['path'].endswith('.npz')),
 'decode':'Original event __schema__ tree; initial_vector float64 payload under root/arrays/initial_vector. Verify wholefile and descriptor SHA, root_trajectory_id, original_library_index, source_library and both typed/raw vector SHA against ROOT-CLUSTER-MAP. ROOT_RUNTIME_BINDING for source/root identity, not substitute for vector bytes.',
 'new_model_calls':0,'GPU':0,'read_arrays_on_Mac':False})

# CONF shared-pulse is real but is not DEV and contains no matched-random pulse arm.
pvp_plan=read(OLD/'continuous/task-heldout-confirmation-v1/pvp-readout-v1/READ-PLAN-CANDIDATE-v1.json')
pvp_result=read(OLD/'continuous/task-heldout-confirmation-v1/root/pvp-readout-cpu-v1/run-v1/readback/science/RESULT-v1.json')
pvp_cpu=read(OLD/'continuous/task-heldout-confirmation-v1/root/pvp-readout-cpu-v1/run-v1/readback/summary/RESULT-v1.json')
conf=[];small=[]
for task in pvp_plan['tasks']:
 calls=[]
 for c in task['calls']:
  e=c['expected_entry'];calls.append({'call_id':e['call_id'],'role':e['role'],'root_id':e['root_id'],'query_index':e['query_index'],'schedule':e['schedule'],'flow_mask':e['flow_mask'],'actual_passed':e['call_id']in task['source_result']['worker_result']['passed_entries'],'object':minimal_ref(c['ref'])})
 assert all(c['actual_passed']for c in calls)
 conf.append({'shard_id':task['shard_id'],'run_id':task['binding']['run_id'],'binding':task['binding'],'source_result_ref':task['source_result_ref'],'planned_ref':minimal_ref(task['planned_ref']),'calls':calls,
  'interpretation':'CONF_ONLY_SHARED_PULSE_AVAILABLE; no DEV or random_matched-pulse in this exact campaign. Planned actual_status fields are historical and are not current completion fields.'})
 small.append({'purpose':'PVP_call_roster_and_schedule','shard_id':task['shard_id'],**minimal_ref(task['planned_ref'])})
for r in pvp_cpu['cloud_objects']:
 if r['relative'].endswith(('PVP-COMPACT-AXES-v1.json','PVP-ENDPOINT-FLOW-v1.npz','PVP-SLOTS-v1.csv')):
  small.append({'purpose':'CONF_ONLY_native_derived_endpoint_or_signed_effect','path':r['relative'].split('/')[-1],**{k:r[k]for k in ('uri','bytes','sha256')}})
put('CONF-PVP-REUSE-INDEX-v1.json',{'schema':'rss-CONF-pvp-exclusion-and-reuse-v1','tasks':conf,'completed_queries':pvp_result['completed_queries'],'coverage':pvp_result['coverage'],'graph_reporting_label':'Gstar; exact source bindings retained, not asserted identical to DEV G',
 'known_shared_pulse_anchors':24,'known_random_matched_pulse_anchors_in_this_campaign':0,'no_global_nonexistence_claim':True,'small_cloud_objects':small,'small_object_bytes':sum(r['bytes']for r in small),'allow_as_DEV_training':False,
 'label_scope':'derived endpoint/slot artifacts contain none/shared_pulse/shared_persistent for original CONF anchors; CPU verify axes/shape before extracting10x7, first5. Do not confuse window RMS summaries with full signed labels.'})

compact=read(OLD/'continuous/l9-effect-cpu-root-v1/run-v1/readback/RESULT-v1.json')
compact_ref=next(r for r in compact['cloud_objects']if r['relative'].endswith('L9-COMPACT-BUNDLE-v1.json.gz'))
put('OPTIONAL-DEV-COMPACT-READ-PLAN-v1.json',{'schema':'rss-DEV-compact-endpoint-read-plan-v1','input_objects':[{'path':'L9-COMPACT-BUNDLE-v1.json.gz',**{k:compact_ref[k]for k in ('uri','bytes','sha256')}}],
 'count':1,'input_body_bytes':compact_ref['bytes'],'reuse_role':'Already produced CPU compact includes original 432 CALL_OUTPUT raw32/adapter7 + selected metadata and192ACTUAL_EFFECT. Useful N endpoint and independent signed-effect verification without full neural records; graph/schedule remains persistent.',
 'provenance_limit':'compact extraction cannot independently re-prove every original NPZ membership; bind original-ref and actual extractor receipt. DEV695 full read supersedes need to GET this optional duplicate cache.'})

searches=[]
for pattern in ['shared_pulse|random_pulse|random_matched_pulse','random_matched_pulse|random_pulse|random.matched.*pulse|pulse.*random.matched']:
 cp=subprocess.run(['rg','-l',pattern,'continuous','protocol','execution','--glob','*.json','--glob','*.md','--glob','*.py'],cwd=OLD,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,check=False)
 paths=cp.stdout.splitlines();searches.append({'tool':'rg -l','pattern':pattern,'roots':['continuous','protocol','execution'],'extensions':['json','md','py'],'matched_paths':paths,'exit_code':cp.returncode,'negative_result_is_not_global_absence_proof':True})
for n in ['continuous/l9-intervention-root-v2/run-v2/readback/CLOUD-RESULT-v1.json','continuous/l9-intervention-root-v1/run-v1/failure-readback-v1/INDEX-v1.json','continuous/l9-replication-readout-v1/INPUT-CANDIDATE-v1.json']:
 read(OLD/n)
put('LOCAL-AUDIT-FOOTPRINT-v1.json',{'schema':'rss-pulse-local-audit-footprint-v1','files':list(footprint.values()),'searches':searches,'scope_limits':['No bucket/prefix listing','No remote GET','No neural NPZ payload read','No scan outside named study and retained adapter','Unindexed historical runs remain unknown'],
 'local_graph_source_pin':{'worker':footprint[str(worker_path)],'frozen_expected':worker_pin,'matches':True},'arrays_read':0,'cloud_calls':0,'models_or_simulators':0})

report={'schema':'rss-DEV-pulse-reuse-audit-v1','status':'LOCAL_METADATA_AND_SOURCE_AUDIT_COMPLETE_CLOUD_VALUE_CHECK_PENDING',
 'classifications':[
 {'population':'DEV48 CDEV T00/T01 R00/R01 q11..22','source_run':dev['source_run_id'],'class':'SOURCE_AND_PRIOR_CLOUD_READER_PROVEN_PERSISTENT','shared_targets':48,'random_matched_targets':48,'all_four_arm_targets':192,'none_endpoints':48,
 'reason':'Frozen worker uses np.arange(10)>=5 for every V effect. Retained dev_reader explicitly rejects non-persistent actual selector; prior signed-transfer true CPU accepted this reader. Local small audit has not redecoded any payload.',
 'full_signed':'192 ACTUAL_EFFECT float64 10x7; N raw endpoint1x10x7 exists in natural_T calls; first5 is deterministic slice, not another observation.','pulse_training_eligible':False},
 {'population':'DEV r2 partial R00/R01 campaign','class':'KNOWN_PROTOCOL_PERSISTENT_ONLY_ACTUAL_SUPPORT_PARTIAL','reason':'Same requested persistent operation; r2 actual call records contain24shared+24random targets and48natural_T endpoints; no pulse schedule is authorized in frozen worker. Preserve failed/uncompleted remainder.'},
 {'population':'DEV r1 failure-readback','class':'FAILED_OR_UNKNOWN_NOT_USABLE_PULSE','reason':'Only failure index reviewed; do not assume unexecuted scientifically or count as completed pulse labels.'},
 {'population':'CONF T06..T09 PC/PT q11/16/22','class':'CONF_ONLY_SHARED_PULSE_REAL_NOT_DEV','shared_pulse':24,'random_matched_pulse_in_exact_campaign':0,'full_signed':'PVP endpoint-flow NPZ and slots CSV available by exact derived refs; raw current graph labels must stay CONF/Gstar.'},
 {'population':'Early SDRU-Q01/Q02 q0 native K/V intervention','class':'REAL_SINGLE_CALL_DIFFERENT_INTERVENTION_AND_SUPPORT','reason':'One-call clean donor K/V/KV replacement at3layers, no matching frozen shared/rank4 and random_matched pair; cannot substitute for requested DEV48 pulse labels.'},
 {'population':'Other unindexed sources','class':'UNKNOWN_NOT_PROVEN_ABSENT','reason':'Bounded local search and named exact indexes are not a whole-cloud census.'}],
 'root_cluster_evidence':'ROOT-CLUSTER-MAP-v1.json: actual metadata supports4distinct typed initial vectors across2tasks, not4IID draws. ROOT-CLUSTER-VERIFY-READ-PLAN gives8smallobjects for payload/source verification.',
 'immediate_CPU_options':{'minimal_schedule_label_check':'194objects/572954bytes PULSE-SMALL-READ-PLAN.json','full_DEV_strict_grouped_analysis':'695objects/224484300bytes DEV-RAW-READ-PLAN.json','root_init_checks':'8objects, see ROOT-CLUSTER-VERIFY-READ-PLAN-v1.json','optional_N_compact':'1object/1994019bytes, omit if695 already read'},
 'classification_logic':[
 'Whole object bytes/SHA, event schema, run/source binding, root/query/arm and metadata must agree before use.',
 'Join ACTUAL_EFFECT.actual_effect_source and edit_ref to exact CALL_OUTPUT/FROZEN_QUERY_EDITS; full10x7 finite signed target minus own natural_T baseline must hold.',
 'Actual V selector flow true only atj5 classifies pulse; true atj5..9 classifies persistent; missing/unsupported shape orclock remains UNKNOWN_OR_INPUT_FAILURE, never fills pulse.',
 'No-edit identity is actual native_T / own-baseline; zero effect label does not itself prove a baseline call.',
 'CONF lineage is retained even if pulse exists; cannot silently relabel DEV or fresh blind.',
 'Underlying runtime supports pulse andpersistent but capability is not evidence a DEV pulse call was executed.',
 'Source-only schedule proof and payload-verified schedule use separate status fields.'
 ],'root_cpu_should_emit':['per_query_arm schedule/selector/clock/source tree proof','N/shared/random full signed10x7 availability','pulse reuse count separatedDEV/CONF','explicit unknowns and mismatches','init cluster mapping + payload hash verification'],
 'new_GPU_or_model_calls_authorized_by_inventory':False,'actions':{'cloud_calls':0,'model_calls':0,'simulator_steps':0,'neural_array_decodes':0,'old_files_modified':False}}
put('INVENTORY-REPORT-v1.json',report)
put('DEV-RAW-EXTRACTION-CONTRACT-v1.json',{'schema':'rss-DEV-fold-safe-raw-extraction-contract-v1','input_plan':'DEV-RAW-READ-PLAN.json',
 'reusable_modules':{n:footprint[str(LEG/'worker/core'/n)]for n in ['dev_reader.py','dev_worker.py','taskblind_prediction.py','retained_replication.py','sources.py']},
 'module_loading':'Use sources.dependencies with original PINS and retained readers. Do not edit legacy.py or import a policy model; all array decoding is onroot-controlledLinuxCPU.',
 'objects_and_fields':{
 'GPU_INPUT_AUDIT':'explicit_noise; consumed_decode_state; original C/T GPU encoded binding and source_identity; actual query clock',
 'natural_C_and_T_CALL':'x0,v0,x5,v5; first_read.P, fullmask, suffixpositions, gate; C/T receiver_L9_V and valid prefix; raw32/adapter7',
 'READ_ONLY_WO':'native_L9_WO [8,256,1024] BF16 exact parameter identity',
 'FROZEN_QUERY_EDITS':'actual_E/Vmod for shared/random_matched/distance_matched/full_v_restore with fullpadding and zero cases',
 'effect_CALL_and_ACTUAL_EFFECT':'actual persistent selector, sameprej5/currentreceiver proof, signed10x7 endpoint subtraction andinput/edit receipts'},
 'raw_feature_functions':{'context':'Query.state_vector(): concatenated C/T x0,v0,x5,v5 plus32state; do not use allDEV PCA coords','firstflow':'Query.first_flow_vector(): original FP32 T_v0-C_v0, then castFP64','P':'Query.p_native actualBF16 fullsoftmax; preserve mask/order, no prefix renormalization','E':'Query.edit(arm): FP32 difference ofactualBF16 targetandbaseline, then analysisFP64'},
 'fold_rules':['Entire initcluster allqueries/arms in onefold','Fitstate/firstflow/write PCA, scales and Pbar only trainingroots','Training targetmean andscale fittrainingrows; declared evaluation scales separately labeled','All four persistent arms may supportold-data diagnostic, but it is not the newdual-pulse training experiment','Use exact sourceoriginal_gates before any statisticalfit'],
 'memory':'Original dev_reader extracts oneobjectatatime and retainsselectedleaves. Release full effect flow aftergating; querycache can containrequiredrawfeatures andP/E once, not4copies ofeachneuralrecord.',
 'root_only_CPU':True,'new_model_calls':0,'native_precision_unchanged':True})
