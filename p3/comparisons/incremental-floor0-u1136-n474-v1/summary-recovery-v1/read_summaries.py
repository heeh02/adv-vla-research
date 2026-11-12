"""Root-only exact403 summary cache recovery; no index/OUTCOME/array or mutation."""
import argparse,ast,fcntl,hashlib,json,re,sys,time,threading
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
from urllib.parse import urlsplit
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent;C=HERE.parents[2]
HOST=C.parents[1]/'t20-optimization-continuation-20261019-v1/infra/eval-host-allowed-v1'
PLAN=HERE/'READ-PLAN-v1.json'
PREFIX='redacted-43794b26f503'

def need(ok,reason):
 if not ok:raise ValueError(reason)
def identity(raw):return dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
def ref(p):
 p=Path(p).resolve();return dict(path=str(p),**identity(p.read_bytes()))
def exact(r):
 p=Path(r['path']);need(identity(p.read_bytes())=={k:r[k]for k in ('bytes','sha256')},'frozen_local_source');return p
def canonical(v):return json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()

def verify():
 for r in json.loads((HERE/'SOURCE-MANIFEST-v1.json').read_bytes())['files']:exact(r)
 plan=json.loads(PLAN.read_bytes());docs={k:json.loads(exact(r).read_bytes())for k,r in plan['sources'].items()}
 need(plan['schema']=='floor0-Plus403-summary-only-GET-plan-v1' and plan['GET_count']==403 and plan['GET_bytes']==1270348,'exact403_read_budget')
 need(docs['ledger']['generation']==7 and docs['adoption']['applied'] is True and docs['adoption']['after_closed']==474 and docs['preservation']['status']=='ALL474_AND_UNRESOLVED_SNAPSHOT_METADATA_PRESERVED','actual_gen7_adoption_and_preservation')
 need(plan['sources']['ledger']['sha256']==docs['adoption']['after_identity']['sha256'],'actual_adopted_ledger_bytes')
 allrows=plan['missing403']+plan['reuse71']
 need([x['ordinal']for x in allrows]==list(range(1136,1610)) and sum(x['success']for x in allrows)==421,'all474_not_71_slice')
 need(sum(x['summary']['bytes']for x in plan['missing403'])==1270348 and len(plan['reuse71'])==71,'exact_missing_and_existing_split')
 for item,ledger,read in zip(allrows,docs['ledger']['rows'],docs['final_readback']['cases']):
  need(item['ordinal']==ledger['ordinal']==read['population_ordinal'] and item['id']==ledger['id']==read['id'] and item['success'] is ledger['success'] is read['success'],'same_committed_case')
  need(ledger['state']=='closed' and ledger['scientific_attempt']==0 and ledger['original_evidence_authenticated'] is True and item['status']==ledger['scientific_status']=='complete','unchanged_original_outcome')
  s=item['summary'];need(s=={k:ledger['case_summary_ref'][k]for k in ('uri','bytes','sha256')} and s['uri']==PREFIX+f"cases/p{item['ordinal']:05d}/CASE_SUMMARY-v1.json" and 0<s['bytes']<=4096,'exact_summary_only_URI_size_SHA')
 for item in plan['reuse71']:exact(item['local'])
 return plan,docs

def original_exact_GET(plan):
 """Only replace redundant HEAD with already pinned expected size."""
 source=exact(plan['existing_fetch_source'])
 node=next(n for n in ast.parse(source.read_bytes()).body if isinstance(n,ast.FunctionDef)and n.name=='fetch')
 heads=[n for n in node.body if isinstance(n,ast.Try)and 'get_object_meta_data' in ast.unparse(n)]
 need(len(heads)==1,'one_original_HEAD_block')
 index=node.body.index(heads[0])
 node.body[index:index+1]=ast.parse("need(expected is not None, 'prebound_size_required'); size=expected['bytes']").body
 ns=dict(need=need,identity=identity,urlsplit=urlsplit,PREFIX=PREFIX)
 exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(source)+'::exact_size_no_HEAD','exec'),ns)
 return ns['fetch']

def validate_summary(raw,item,plan,model):
 need(identity(raw)=={k:item['summary'][k]for k in ('bytes','sha256')},'whole_summary_SHA')
 s=json.loads(raw)
 need(s['run_id']==plan['run_id'] and s['population_ordinal']==item['ordinal'] and s['id']==item['id'] and s['attempt']==0 and s['status']==item['status'] and s['success'] is item['success'],'summary_committed_identity')
 need(s['source_package_sha256']==plan['source_package_sha256'] and s['model_identity_sha256']==hashlib.sha256(canonical(model)).hexdigest(),'original_model_and_source')
 o=s['outcome'];need(o['uri']==PREFIX+f"cases/p{item['ordinal']:05d}/OUTCOME-v1.json" and type(o['bytes'])is int and o['bytes']>0 and re.fullmatch('[0-9a-f]{64}',o['sha256']),'exact_original_OUTCOME_reference_only')
 return s

def cached(out,item,plan,model):
 path=out/(item['summary']['sha256']+'-CASE_SUMMARY-v1.json')
 if not path.exists():return None
 s=validate_summary(path.read_bytes(),item,plan,model)
 return dict(ordinal=item['ordinal'],summary=item['summary'],local=ref(path),outcome=s['outcome'],reused_local=True)

def run_locked(out):
 plan,docs=verify()
 marker=out/'CACHE-IDENTITY-v1.json'
 expected=dict(plan=ref(PLAN),source_manifest=ref(HERE/'SOURCE-MANIFEST-v1.json'),run_id=plan['run_id'],job_id=plan['job_id'])
 if marker.exists():need(json.loads(marker.read_bytes())==expected,'resume_same_exact_cache_namespace')
 else:
  with marker.open('x')as f:json.dump(expected,f,sort_keys=True);f.write('\n')
 fetch=original_exact_GET(plan);auth={};result=dict(schema='Plus474-summary-cache-recovery-readback-v1',run_id=plan['run_id'],source=ref(__file__),plan=ref(PLAN),started_epoch=time.time(),GETs=[],summaries=[],errors=[],GET_bytes_verified=0,GET_bytes_charged=0,HEAD_calls=0,index_GETs=0,OUTCOME_GETs=0,array_GETs=0,parent_verification_GETs=0,Create_Stop_STS_PUT_calls=0,shared_CAS=0,new_scientific_attempts=0,max_workers=8,automatic_GET_retry=False,repeated_successful_GETs=0)
 try:
  # Exact original local71 are reused; they consume no cloud GET.
  for item in plan['reuse71']:
   raw=exact(item['local']).read_bytes();s=validate_summary(raw,item,plan,docs['final_readback']['model'])
   result['summaries'].append(dict(ordinal=item['ordinal'],summary=item['summary'],local=item['local'],outcome=s['outcome'],reused_local=True))
  pending=[]
  for item in plan['missing403']:
   existing=cached(out,item,plan,docs['final_readback']['model'])
   if existing:result['summaries'].append(existing)
   else:pending.append(item)
  result['prior_successful_cache_reused']=403-len(pending)
  sys.path.insert(0,str(HOST));import eval_host as H
  with H.quiet():
   reader,_,_,proxy,_,_,_,transport=H.load_helpers()
   with proxy():
    auth=reader.read_RAM_auth()
    from baidubce.services.bos.bos_client import BosClient
    from baidubce.retry.retry_policy import NoRetryPolicy
    local=threading.local()
    def one(item):
     if not hasattr(local,'client'):
      cfg=transport.sdk_configuration(auth,'https://bj.bcebos.com');cfg.retry_policy=NoRetryPolicy();cfg.connection_timeout_in_mills=15000;local.client=BosClient(cfg)
     sref=item['summary'];raw=fetch(local.client,sref['uri'],sref['bytes'],expected=sref)
     summary=validate_summary(raw,item,plan,docs['final_readback']['model']);path=out/(sref['sha256']+'-CASE_SUMMARY-v1.json')
     with path.open('xb')as f:f.write(raw)
     return dict(ordinal=item['ordinal'],summary=sref,local=ref(path),outcome=summary['outcome'],reused_local=False)
    result['GET_bytes_charged']=sum(item['summary']['bytes']for item in pending)
    need(result['GET_bytes_charged']<=1270348,'exact_total_summary_budget')
    with ThreadPoolExecutor(max_workers=8)as pool:
     futures={pool.submit(one,item):item for item in pending}
     for future in as_completed(futures):
      item=futures[future];event=dict(ordinal=item['ordinal'],reference=item['summary'],verified=False)
      try:
       value=future.result();event['verified']=True;result['GET_bytes_verified']+=item['summary']['bytes'];result['summaries'].append(value)
      except Exception as error:
       detail=dict(ordinal=item['ordinal'],error_type=type(error).__name__)
       if re.fullmatch('[A-Za-z0-9_:.]{1,160}',str(error)):detail['code']=str(error)
       result['errors'].append(detail)
      result['GETs'].append(event)
  need(not result['errors'] and len(result['GETs'])==len(pending) and result['GET_bytes_verified']==result['GET_bytes_charged'] and len(result['summaries'])==474,'complete403_recovery_plus71_reuse')
  result['summaries'].sort(key=lambda r:r['ordinal'])
  result.update(status='ALL474_SUMMARIES_VERIFIED403_GET71_REUSED',exact474_OUTCOME_GET_bytes=sum(r['outcome']['bytes']for r in result['summaries']),exact474_OUTCOME_GET_count=474)
 except Exception as error:
  result.update(status='PARTIAL_CACHE_RECOVERY_NO_OUTCOME_OR_ADOPTION_CHANGE',error_type=type(error).__name__)
  if re.fullmatch('[A-Za-z0-9_:.]{1,160}',str(error)):result['code']=str(error)
 finally:auth.clear();result['finished_epoch']=time.time()
 result['GETs'].sort(key=lambda r:r['ordinal'])
 path=out/f"READBACK-{len(list(out.glob('READBACK-*.json')))+1:03d}.json"
 with path.open('x')as f:json.dump(result,f,sort_keys=True,indent=2);f.write('\n')
 return dict(status=result['status'],receipt=ref(path),GET_bytes_verified=result['GET_bytes_verified'])

def run(out):
 out.mkdir(parents=True,exist_ok=True)
 need(not out.is_symlink(),'owned_cache_directory')
 with (out/'ROOT.lock').open('a')as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  try:return run_locked(out)
  finally:fcntl.flock(lock,fcntl.LOCK_UN)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--execute-read-only',action='store_true');p.add_argument('--out',type=Path);a=p.parse_args()
 if a.execute_read_only:
  need(a.out is not None,'explicit_new_output_directory');print(json.dumps(run(a.out)))
 else:
  plan,_=verify();original_exact_GET(plan);print(json.dumps(dict(status='OFFLINE_VERIFIED_NO_CLOUD',planned_GETs=403,planned_GET_bytes=1270348,reuse_local=71,cloud_calls=0)))
