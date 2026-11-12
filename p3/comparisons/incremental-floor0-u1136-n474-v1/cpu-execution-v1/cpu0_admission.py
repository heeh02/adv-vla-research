"""Private admission only for the pinned plus474 CPU0 job; no GPU gate edits."""
import ast
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

HERE=Path(__file__).resolve().parent
RESOURCES={'GPU':0,'CPU':2,'memory_GiB':8}
QUEUE='redacted-68f1c47a0c15';PARENT='redacted-6e77221ca60e'


def need(ok,reason):
 if not ok:raise ValueError(reason)


def ref(path):
 p=Path(path).resolve();need(p.is_file() and not p.is_symlink(),'regular_reference')
 h=hashlib.sha256();n=0
 with p.open('rb')as f:
  for b in iter(lambda:f.read(1024**2),b''):n+=len(b);h.update(b)
 return dict(path=str(p),bytes=n,sha256=h.hexdigest())


def exact(r):need(ref(r['path'])==r,'frozen_CPU0_source_changed')
def read(p):return json.loads(Path(p).read_bytes())
def load(name,p):
 s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m


def selected(key):
 for r in read(HERE/'CPU0-SOURCE-MANIFEST-v1.json')['files']:exact(r)
 targets=read(HERE/'TARGETS-v1.json');need(key in targets['targets'],'only_frozen_plus474_CPU0_target')
 target=targets['targets'][key]
 for r in target['pins']:exact(r)
 for k in ('base_controller','fast_account','thread_authority'):exact(targets[k])
 need(target['resources']==RESOURCES,'pinned_GPU0_CPU2_memory8')
 directory=Path(target['directory'])
 old={n:sys.modules.get(n)for n in ('contract','build','root')}
 try:
  load('contract',directory/'contract.py');load('build',directory/'build.py')
  module=load('root',directory/'root.py');R=module.prepare()
 finally:
  for n,v in old.items():
   if v is None:sys.modules.pop(n,None)
   else:sys.modules[n]=v
 R.verify_sources();d,q=R.build.verify(directory/'prepared-v1')
 need(R.C.RESOURCES==q['resources']==RESOURCES and R.C.RUN==q['run_id']==target['run_id'],'exact_CPU0_contract')
 need(d['source_package']==target['source_package'] and d['request']==target['request'],'exact_CPU0_prepared_source_request')
 need(R.C.QUEUE==QUEUE and R.C.PARENT==PARENT,'allowed_Baidu_route')
 return R,target,targets


def counts(account,ledger,now):
 a=account.get('account',account)
 need(a['owner_id']=='3a59a0c408234b37a90a806c3ddc7a5a' and a['visible_managed_coverage_complete'] is True and not a['needs'],'complete_actual_Baidu_account')
 need(type(a['epoch'])in(int,float) and 0<=now-a['epoch']<=120,'fresh_complete_Baidu120')
 g=a['GPU_known_active_plus_reserved'];need(type(g)is int and g>=0,'actual_Baidu_GPU_observation')
 need(type(ledger['generation'])is int and isinstance(ledger['reservations'],list),'actual_registry_generation')
 return dict(schema='exact-Baidu-CPU0-admission-counts-v1',scope='one_pinned_plus474_CPU0_job_only',
  CPU_added_GPU=0,new_GPU=0,global_Baidu_observed_GPU=g,total_GPU=None,thread_total_GPU=None,
  JD_current_GPU=None,JD_current_observed_epoch=None,
  external_GPU_applicability='not_required_for_fixed_Baidu_GPU0_increment',
  GPU_cap_decision='not_applicable_to_this_CPU0_job',GPU_admission_authorized=False,
  registry_generation=ledger['generation'],rows_deleted=0,observed_epoch=a['epoch'])


def wire(body,R,target):
 need(R.C.RESOURCES==target['resources']==RESOURCES,'CPU0_contract_not_mutated')
 need(set(body)=={'name','queue','command','jobType','priority','enableBccl','faultTolerance','retentionPeriod','datasources','labels','jobSpec'},'exact_CPU0_wire_top_fields')
 need(body['name']==target['run_id']==R.C.RUN and body['queue']==QUEUE and body['jobType']=='PyTorchJob','exact_CPU0_name_queue_type')
 need(body['enableBccl'] is False and body['faultTolerance'] is False and body['datasources']==[],'no_Bccl_fault_tolerance_or_dataset')
 spec=body['jobSpec'];need(set(spec)=={'image','replicas','resources','envs','imageConfig','enableRDMA'},'exact_CPU0_wire_spec_fields')
 need(type(spec['replicas'])is int and spec['replicas']==1 and spec['image']==R.C.IMAGE and spec['enableRDMA'] is False,'one_original_image_CPU_replica')
 need(spec['resources']==[{'name':'cpu','quantity':2},{'name':'memory','quantity':8}],'no_GPU_or_extra_resource_wire')
 env={r['name']:r['value']for r in spec['envs']};need(len(env)==len(spec['envs']),'unique_env_names')
 public={'PYTHONUNBUFFERED':'1','PYTHONDONTWRITEBYTECODE':'1','CUDA_VISIBLE_DEVICES':'',
         'NVIDIA_VISIBLE_DEVICES':'void','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1',
         'OPENBLAS_NUM_THREADS':'1','JAX_PLATFORMS':'cpu'}
 need(set(env)==set(public)|R.AUTH and all(env[k]==v for k,v in public.items()),'exact_CPU0_public_environment')
 labels={r['key']:r['value']for r in body['labels']};need(labels.get('one_vla_run_id')==target['run_id'] and labels.get('source_sha')==target['source_package']['sha256'][:32],'exact_run_source_labels')
 return dict(schema='CPU0-final-Create-wire-proof-v1',run_id=target['run_id'],resources=RESOURCES,
  replicas=1,GPU_resource_entries=0,CPU_only_resource_fields=['cpu','memory'],queue_id=QUEUE,
  image=spec['image'],enableRDMA=False,enableBccl=False,faultTolerance=False,
  GPU_visibility_disabled=True,source_package=target['source_package'],
  command_sha256=hashlib.sha256(body['command'].encode()).hexdigest(),
  provider_GPU0_still_required_by_original_exact_job=True,credentials_or_command_saved=False)


def install(R,target,targets):
 """No file/registry/cloud mutations. All changes affect this private instance."""
 need(R.C.RESOURCES==RESOURCES and R.C.RUN==target['run_id'],'CPU0_only_install')
 original_helpers=R.helpers
 def helpers():
  H,values=original_helpers();P=values[-1]
  fast=load('_CPU0_unchanged_fast_Baidu_account',targets['fast_account']['path'])
  P.build_live_backend_class=fast.make_factory(P)
  return H,values
 R.C.THREAD_SCOPE_INSTALLED=True;R.helpers=helpers
 R.C.account_gate=lambda a,l,e,n:counts(a,l,n)
 def fresh_account(auth,P,registry_path,external_path):
  need(R.C.RESOURCES==RESOURCES,'CPU0_only_fresh_account')
  a=P.build_live_backend_class(P.load_common())(auth).account();ledger=R.C.read(registry_path)
  historical=ref(external_path);epoch=None
  if historical['bytes']<=256*1024:
   document=read(external_path);epoch=document.get('observed_epoch')
  return dict(counts=counts(a,ledger,time.time()),complete_Baidu_account=a,
   Baidu_observed_epoch=a['epoch'],external_observed_epoch=None,
   external=historical,external_role='historical_reference_only_not_current_GPU_evidence',
   external_original_observed_epoch=epoch,JD_current_GPU=None,
   registry=ref(registry_path),CPU0_admission_adapter=ref(HERE/'CPU0-SOURCE-MANIFEST-v1.json'))
 R.fresh_account=fresh_account
 # Preserve the original lock, generation/SHA CAS, historical reference check,
 # and intent write. Remove only the non-applicable JD timestamp conjunct.
 source=Path(targets['base_controller']['path']).read_text()
 node=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef)and n.name=='commit_submit_intent')
 matches=[n for n in ast.walk(node)if isinstance(n,ast.BoolOp)and ast.unparse(n)=="0 <= now - resource['Baidu_observed_epoch'] <= 120 and 0 <= now - resource['external_observed_epoch'] <= 300"]
 need(len(matches)==1 and len(matches[0].values)==2,'one_original_JD_freshness_conjunct')
 class RemoveOnlyJD(ast.NodeTransformer):
  def visit_BoolOp(self,n):
   return n.values[0] if n is matches[0] else self.generic_visit(n)
 node=RemoveOnlyJD().visit(node)
 exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),__file__+'::CPU0_commit_no_JD_timestamp','exec'),R.__dict__)
 original_commit=R.commit_submit_intent
 def commit(registry,state,intent,resource):
  need(intent.get('GPU')==0 and intent.get('CPU')==2 and intent.get('memory_GiB')==8 and intent.get('run_id')==target['run_id'],'CPU0_only_intent')
  need(resource['counts']['schema']=='exact-Baidu-CPU0-admission-counts-v1' and resource['external_observed_epoch'] is None,'explicit_CPU0_not_applicable_evidence')
  return original_commit(registry,state,intent,resource)
 R.commit_submit_intent=commit
 original_create=R.explicit_create
 def create(backend,body):
  proof=wire(body,R,target)
  intent=R.logged(R.HERE/'run-v1','SUBMIT-INTENT-v1.json')
  need(intent and intent['run_id']==target['run_id'] and intent['source_package_sha256']==target['source_package']['sha256'] and intent['GPU']==0,'original_CPU0_submit_intent_required')
  need(not R.logged(R.HERE/'run-v1','CPU0-FINAL-WIRE-v1.json'),'CPU0_wire_already_attempted_reconcile_only')
  R.journal(R.HERE/'run-v1','CPU0-FINAL-WIRE-v1.json',proof)
  return original_create(backend,body)
 R.explicit_create=create
 return R
