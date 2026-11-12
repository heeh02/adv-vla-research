"""Existing-source pipeline only. The frozen statistical core is imported unchanged."""
import csv,io,json,time
from pathlib import Path
import numpy as np
import contract as C

def csv_bytes(rows):
 if not rows:return b'status\n'
 fields=list(dict.fromkeys(k for r in rows for k in r));s=io.StringIO();w=csv.DictWriter(s,fields);w.writeheader()
 for r in rows:w.writerow({k:json.dumps(v,sort_keys=True,separators=(',',':'))if isinstance(v,(list,dict))else v for k,v in r.items()})
 return s.getvalue().encode()
def npz_bytes(arrays):
 b=io.BytesIO();np.savez_compressed(b,**arrays);return b.getvalue()
def read_npz(path):
 with np.load(path,allow_pickle=False)as z:return{k:np.array(z[k],copy=True)for k in z.files}
def safe(error):
 s=str(error);return{'type':type(error).__name__,'code':s if len(s)<=160 and s.replace('_','').isalnum()else None}

def conf_task(D,old,artifact,arrays,phase,read,qualified,check,*,synthetic=False):
 plan=phase['science_plan'];source=plan['source'];errors=[];rows=[];missing={};buffers={tuple(k):[]for k in plan['planned_queries']};wo=None
 def rawdoc(ref):
  raw=read(ref,lambda p:p.read_bytes());return json.loads(raw),raw
 def doc(ref):return rawdoc(ref)[0]
 # Admission already replayed source/READY/ACK proofs; keep original scientific
 # model and prediction-publication joins here, exactly as DIFFERENCE did.
 manifest=doc(source['source_manifest']);model_binding=doc(source['model_binding']);result=doc(source['result']);seal,raw_seal=rawdoc(source['predictions']);publication=doc(source['publication'])
 from source_gate import verify
 verify(manifest,source,plan['source_contract'])
 C.need(model_binding['model_bound']is True and model_binding['adapter_bound']is True and model_binding['loaded_parameter_tree_sha256']==plan['source_contract']['loaded_parameter_tree_sha256'],'same_original_loaded_model')
 freeze=result.get('worker_result',result)['freeze'];C.need(C.ident(raw_seal)=={k:freeze['prediction_receipt'][k]for k in('bytes','sha256')}and seal['planned_keys']==plan['planned_queries'],'original_complete_prediction_seal')
 C.need(publication['prediction_payload_sha256']==C.sha(raw_seal)and publication['fit_identity']==plan['fit_identity'],'original_publication_frozen_fit_identity')
 for ref in source['members']:
  check()
  try:r=read(ref,lambda p:D.N.load(p,ref,synthetic=synthetic))
  except Exception as error:errors.append({'source':{k:ref[k]for k in('path','bytes','sha256')},**safe(error)});continue
  m=r.metadata
  if r.event=='READ_ONLY_WO':C.need(wo is None,'unique_original_WO');wo=r;continue
  if r.event=='CALL_OUTPUT'and(m.get('kind')!='common'or m.get('role')not in('natural_C','natural_T','effect_shared','K_only','K_shared')):continue
  if r.event=='ACTUAL_EFFECT'and m.get('arm')!='shared':continue
  key=m.get('root_id'),m.get('query_index');C.need(key in buffers,'only_original_CONF_query');buffers[key].append(r)
  C.need(sum(v.native_bytes for group in buffers.values()for v in group)<=1024**3,'bounded_task_native_RAM1GiB')
 bound=D.H.load_model(artifact,arrays,wo,synthetic=synthetic);C.need(bound.identity==old.identity,'same_original_preprocessing_each_task')
 import sources as S,ablation as A
 for key,records in buffers.items():
  check()
  def one(event):
   found=[r for r in records if r.event==event];C.need(len(found)==1,'one_required_'+event);return found[0]
  try:rows.extend(S.conf_rows(D,old,binding=one('CURRENT_GPU_INPUT_BINDING'),common_records=[r for r in records if r.event=='CALL_OUTPUT'],edits=one('FROZEN_QUERY_EDITS'),labels=[r for r in records if r.event=='ACTUAL_EFFECT'],seal_rows=[r for r in seal['rows']if tuple(r['key'][:2])==key],qualified=qualified,synthetic=synthetic))
  except Exception as error:
   reason=safe(error);errors.append({'query':list(key),**reason})
   for target in A.CONF_KEYS:
    if target[:2]==key:missing[target]=reason['code']or reason['type']
  records.clear()
 return rows,missing,errors

def run(planned,reads,output,*,package,deadline_epoch,synthetic=False):
 import sources as S,ablation as A
 def check():C.need(time.time()<deadline_epoch,'context_ablation_science_deadline')
 paths={n:package/'worker/core'/n for n in S.PINS};D=S.dependencies(paths);first=planned['phases'][C.SHARDS[0]];r0=reads.get(C.SHARDS[0]);p0=first['science_plan']
 artifact=r0(p0['fit_manifest'],lambda p:json.loads(p.read_bytes()));arrays=r0(p0['fit_arrays'],read_npz)
 C.need(synthetic or artifact['synthetic']is False and artifact['model_identity']==C.FIT_ID,'original_actual_fit_artifact')
 old=D.P.load_frozen(artifact['model'],arrays,expected_identity=artifact['model_identity']);scales=A.FrozenScales(old,synthetic=synthetic)
 devphase=planned['phases']['DEV'];rd=reads.get('DEV')
 def devread(ref,fn):return rd(devphase['by_original_uri'][ref['uri']],fn)
 dev,devproof=S.dev_rows(D,old,devphase['science_plan'],devread,check=check,synthetic=synthetic);rd.cleanup();check()
 fitted=A.fit(scales,dev);output.put_a('CONTEXT-FIT-v1.json',A.encoded({'identity':fitted.identity,'manifest':fitted.manifest()}));output.put_a('CONTEXT-FIT-v1.npz',npz_bytes(fitted.arrays()))
 output.put_b('CONTEXT-DEV-FIT-SUMMARY-v1.json',C.encoded({'schema':'context-shared-DEV-fit-summary-v1','new_fit_identity':fitted.identity,'old_fit_identity':old.identity,'DEV_shared_targets':48,'methods':list(A.METHODS),'feature_count':[9,13,13],'lambda':1,'original_preprocessing_fixed':True,'posthoc_not_blind':True,'old_scores_replaced':False,'source':devproof,'synthetic':synthetic}))
 rq=reads.get('QUAL');qualified=S.qualification(rq,planned['qualification'],synthetic=synthetic);rq.cleanup();conf=[];missing={};errors=[]
 for sid in C.SHARDS:
  check();phase=planned['phases'][sid];read=reads.get(sid);plan=phase['science_plan']
  if sid!=C.SHARDS[0]:
   af=read(plan['fit_manifest'],lambda p:json.loads(p.read_bytes()));aa=read(plan['fit_arrays'],read_npz)
   same=D.P.load_frozen(af['model'],aa,expected_identity=af['model_identity']);C.need(same.identity==old.identity,'same_frozen_fit_each_original_task');del aa
  try:
   values,lost,issues=conf_task(D,old,artifact,arrays,phase,read,qualified,check,synthetic=synthetic);conf.extend(values);missing.update(lost);errors.extend({'shard':sid,**e}for e in issues)
  except Exception as error:
   info={'shard':sid,**safe(error)};errors.append(info)
   roots={r['root_id']for r in plan['roots']}
   for key in A.CONF_KEYS:
    if key[0]in roots:missing[key]=info['code']or info['type']
  finally:read.cleanup()
 report=A.evaluate(fitted,conf,unavailable=missing);check()
 for name,key in(('COVERAGE','coverage'),('ROOT-SCORES','root_scores'),('TASK-SCORES','equal_task_scores'),('INCREMENTS','paired_increments')):output.put_b('CONTEXT-'+name+'-v1.csv',csv_bytes(report[key]))
 output.put_a('CONTEXT-PREDICTIONS-v1.json',C.encoded(report['predictions']))
 output.put_a('CONTEXT-INPUT-IDENTITIES-v1.json',C.encoded([r.manifest()for r in dev+conf]))
 for split,rows in(('DEV',dev),('CONF',conf)):
  if rows:output.put_a('CONTEXT-'+split+'-MATRICES-v1.npz',npz_bytes({'current13':np.stack([r.current for r in rows]),'pooled13':np.stack([r.pooled for r in rows]),'actual_effect':np.stack([r.effect for r in rows]),'zero_actual_E':np.array([r.zero_actual_E for r in rows],bool)}))
 result={'schema':'context-ablation-result-v1','complete':report['complete']and not errors,'synthetic':synthetic,'old_fit_identity':old.identity,'fit_identity':fitted.identity,'new_fit':True,'old_scores_replaced':False,'posthoc_not_new_blind_test':True,'DEV_shared_targets':len(dev),'CONF_KT_targets':sum(r.key[3]=='KT'for r in conf),'CONF_KC_support_targets':sum(r.key[3]=='KC'for r in conf),'planned_DEV':48,'planned_CONF_KT':96,'planned_CONF_KC_support':24,'missing_keys':[list(k)for k in A.CONF_KEYS if k not in{r.key for r in conf}],'errors':errors[:16],'error_count':len(errors),'model_calls':0,'env_steps':0,'original_arrays_to_Mac':False}
 output.put_b('CONTEXT-RESULT-v1.json',C.encoded(result));return result
