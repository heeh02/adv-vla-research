"""Exact per-task unified input/archive/source admission, no cloud calls."""
import gzip,importlib.util,json,re
from pathlib import Path
import contract as C

def document(package,ref):
 raw=(Path(package)/C.relative(ref['package_path'])).read_bytes()
 if ref.get('compression')=='gzip':C.need(C.ident(raw)==ref['stored'],'compressed_metadata_SHA');raw=gzip.decompress(raw)
 C.need(C.ident(raw)=={k:ref[k]for k in('bytes','sha256')}and len(raw)<=8*1024**2,'original_small_metadata_SHA');return json.loads(raw),raw

def verify_conf(package,plan):
 package=Path(package)
 request={'shard_id':plan['shard_id'],'source_run_ids':[plan['source_run_id']],'read_prefixes':[C.BASE+plan['source_run_id']+'/',C.BASE+C.FIT_RUN+'/'],'planned_input_objects':len(plan['read_objects']),'planned_input_body_bytes':plan['planned_input_body_bytes']}
 def module(name,path):
  s=importlib.util.spec_from_file_location(name,path);v=importlib.util.module_from_spec(s);s.loader.exec_module(v);return v
 Q=module('_unified_original_broker',package/'worker/retained/query_broker.py');G=module('_unified_source_guard',package/'worker/source_gate.py')
 source=plan['source'];contract=plan['source_contract'];manifest,_=document(package,source['source_manifest']);G.verify(manifest,source,contract)
 for variant in contract['approved_source_variants']:
  for ref in variant['repair_proof'].values():C.need(C.ident(package/C.relative(ref['package_path']))=={k:ref[k]for k in('bytes','sha256')},'root_approved_source_proof_actual_bytes')
 objects={r['uri']:r for r in plan['read_objects']};C.need(len(objects)==len(plan['read_objects'])==request['planned_input_objects']and sum(r['bytes']for r in objects.values())==request['planned_input_body_bytes'],'physical_object_count_bytes')
 for uri,r in objects.items():C.need(r['object_key']==uri.split('/',3)[3]and 0<r['bytes']<=32*1024**2 and sum(uri.startswith(p)for p in request['read_prefixes'])==1,'exact_small_whole_objects_only')
 remote={};logical={};packs={};prefix=request['read_prefixes'][0]
 for ar in plan['archives']:
  ready,rr=document(package,ar['ready']);ack,aa=document(package,ar['ack']);body,bb=document(package,ar['ack_body'])
  C.need(ar['output_prefix']==prefix and ready['binding']==ar['binding']and ready['binding']['run_id']==plan['source_run_id'],'same_actual_task_archive')
  C.need(rr==Q.encoded(ready)and aa==Q.encoded(ack)and bb==Q.encoded(body)and max(len(rr),len(aa),len(bb))<=Q.MAX_MANIFEST,'original_bounded_READY_ACK_domains')
  C.need(body=={k:v for k,v in ack.items()if k not in('durable_ack_object_key','durable_ack_verification')}and ack['durable_ack_verification']=={'method':'fresh_GET','bytes':len(bb),'sha256':C.sha(bb)},'original_published_ACK_BODY')
  Q.validate_members(ready['members'],64*1024**2)
  for r in Q.verify_ack(ack,ready,C.sha(rr)):
   uri=prefix+r['object_key'];C.need(uri not in remote,'unique_archive_object');remote[uri]=r
   if not r['path'].startswith('archive-packs/'):logical[r['path']]={k:r[k]for k in('path','bytes','sha256')}
 for pack in plan['packs']:
  idx,raw=document(package,pack['index']);obj=pack['object']
  for ref in(obj,pack['index']):C.need(ref['uri']in remote and all(remote[ref['uri']][k]==ref[k]for k in('bytes','sha256')),'original_pack_index_ACK_refs')
  C.need(idx['binding']==pack['binding']and idx['compression']=='none'and len(raw)<=512*1024 and len(idx['original_members'])<=2048,'bounded_original_pack_index')
  for row in idx['original_members']:C.need(row['path']not in logical,'unique_original_member');logical[row['path']]=row
  packs[obj['uri']]={'pack':obj,'index_payload':raw,'original_members':idx['original_members']}
 allowed={}
 for r in [source['result'],source['source_manifest'],source['model_binding'],source['predictions'],source['publication'],*source['members'],plan['fit_manifest'],plan['fit_arrays'],*plan['basis'].values()]:
  C.need(r['key']not in allowed,'unique_callback_member');allowed[r['key']]=r
  if 'package_path'in r:document(package,r);continue
  uri=r['transport_uri'];C.need(uri in objects,'declared_physical_URI')
  if uri.startswith(prefix):
   actual=logical.get(r['path']);C.need(actual is not None and all(actual[k]==r[k]for k in('bytes','sha256')),'source_selected_original_SHA')
   C.need(r['event']!='OBSERVABLE_UPDATE'and not r['event'].startswith('PARENT_'),'no_physics_substep_or_parent_history')
  else:C.need(all(objects[uri][k]==r[k]for k in('bytes','sha256')),'exact_frozen_small_artifact')
 C.need(len(source['members'])==301 and plan['planned_queries']==[[r['root_id'],q]for r in plan['roots']for q in range(11,23)],'fixed24_original_query_keys')
 return {'science_plan':plan,'objects':objects,'packs':packs,'allowed':allowed,'input_body_bytes':request['planned_input_body_bytes']}

def verify_dev(package,config):
 package=Path(package)
 def doc(r):
  p=package/C.relative(r['path']);C.need(C.ident(p)=={k:r[k]for k in('bytes','sha256')},'exact_metadata_ref');raw=p.read_bytes();return json.loads(raw),raw
 plan,_=doc(config['read_plan']);index,indexraw=doc(config['source_docs']['index']);cloud,cloudraw=doc(config['source_docs']['cloud'])
 C.need(C.sha(indexraw)==plan['source_index']['sha256']and len(indexraw)==plan['source_index']['bytes'],'old_index_exact_source')
 C.need(C.sha(cloudraw)==plan['local_cloud_attestation']['sha256']and len(cloudraw)==plan['local_cloud_attestation']['bytes'],'local_cloud_attestation_not_remote_bytes')
 C.need(plan['schema']=='sru-task-blind-refit-exact-readset-candidate-v1'and plan['source_run_id']==index['run_id']==cloud['run_id']==C.SOURCE_RUNS[0],'one_original_r3')
 C.need(index['scientific_complete']is True and cloud['scientific_complete']is True and cloud['archive_complete']is True and not cloud['failure'],'actual_completed_r3_source')
 pop=plan['population'];C.need(pop['roots']==['SDRU-CDEV-T00-R00','SDRU-CDEV-T00-R01','SDRU-CDEV-T01-R00','SDRU-CDEV-T01-R01']and pop['query_indices']==list(range(11,23))and pop['confirmation_task_rows']==0,'DEV_only_no_CONF')
 C.need(pop['root_role']=='all_development_refit_including_previously_revealed_R01'and pop['unique_queries']==48 and pop['query_arm_targets']==192 and pop['arms']==['shared','random_matched','distance_matched','full_v_restore'],'all_revealed_DEV_fixed_population')
 spec=importlib.util.spec_from_file_location('_taskblind_original_broker',package/'worker/retained/query_broker.py');Q=importlib.util.module_from_spec(spec);spec.loader.exec_module(Q)
 allmembers={};prefix=C.BASE+C.SOURCE_RUNS[0]+'/'
 C.need(len(index['archives'])==len(config['archive_docs'])==2,'two_original_archives')
 for ar,refs in zip(index['archives'],config['archive_docs']):
  ready,rr=doc(refs['ready']);ack,aa=doc(refs['ack']);body,bb=doc(refs['ack_body'])
  C.need({k:ar['ready'][k]for k in('bytes','sha256')}=={'bytes':len(rr),'sha256':C.sha(rr)}and {k:ar['ack'][k]for k in('bytes','sha256')}=={'bytes':len(aa),'sha256':C.sha(aa)},'original_READY_complete_IPC_ACK_bytes')
  C.need(ready['binding']==ack['binding']==ar['binding']==plan['source_binding'],'source_archive_binding')
  C.need(rr==Q.encoded(ready)and aa==Q.encoded(ack)and bb==Q.encoded(body),'canonical_original_controls')
  C.need(body=={k:v for k,v in ack.items()if k not in('durable_ack_object_key','durable_ack_verification')},'ACK_body_separate_byte_domain')
  C.need(ack['durable_ack_verification']=={'method':'fresh_GET','bytes':len(bb),'sha256':C.sha(bb)},'remote_ACK_body_publication_proof')
  members=Q.verify_ack(ack,ready,C.sha(rr));expanded=[{**r,'uri':prefix+r['object_key']}for r in members]
  C.need(expanded==ar['members'],'index_equals_actual_ACK_inventory')
  for r in expanded:C.need(r['uri']not in allmembers,'original_member_unique');allmembers[r['uri']]=r
 objects={}
 for r in plan['members']:
  expected=allmembers.get(r['uri']);C.need(expected is not None and all(r[k]==expected[k]for k in('path','object_key','bytes','sha256')),'selected_from_verified_archive')
  C.need(re.fullmatch(r'\d+-'+re.escape(r['event']),r['path'])and r['uri']==prefix+r['object_key']and r['uri']not in objects,'exact_event_full_URI')
  C.need(0<r['bytes']<=3327532,'bounded_original_member')
  C.need(not any(x in r['path']for x in('FROZEN_PREDICTOR','ALL24_PREDICTIONS')),'no_old_fitted_parameters_or_predictions')
  objects[r['uri']]={k:r[k]for k in('uri','object_key','bytes','sha256')}
 C.need(len(objects)==plan['selected_member_count']==C.DEV_INPUT_OBJECTS and sum(r['bytes']for r in objects.values())==plan['selected_body_bytes']==C.DEV_INPUT_BYTES,'exact695_and_original_bytes')
 return {'science_plan':plan,'objects':objects,'input_body_bytes':C.DEV_INPUT_BYTES}
