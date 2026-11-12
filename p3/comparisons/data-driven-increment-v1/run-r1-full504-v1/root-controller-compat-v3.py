from pathlib import Path
import sys,types,json,hashlib
B=Path('/Users/researcher/Adv-VLA-π0.5/docs/experiments/paper3-training-study-20261019-v1/eval-closeout-v1/comparison-v1/data-driven-increment-v1');P=B/'run-r1-full504-v1/cpu0-package-v1';sys.path.insert(0,str(B));import root_adapter
R,d,q=root_adapter.prepare(P);R.review_gate(R.C.read(P/'ROOT-REVIEW-v1.json'),d,q)
original_stage=R.stage_source
def exact_stage(backend,writer,prepared,state,descriptor,retry_review=None):
 R.C.exact(P/'SOURCE_PACKAGE-v1.tar.gz',descriptor['source_package'])
 return original_stage(backend,writer,P,state,descriptor,retry_review)
R.stage_source=exact_stage
mapping={'bridge_ready':'increment_input_spec','repair_ready':'frozen_dependencies'}
for key in mapping.values():assert key in q
assert not (P/'run-v1/SUBMIT-INTENT-v1.json').exists(),'existing_Create_intent_requires_reconcile'
f=R.create_once
assert all(k in f.__code__.co_consts for k in mapping)
def replace(v):
 if isinstance(v,str):return mapping.get(v,v)
 if isinstance(v,tuple):return tuple(replace(x) for x in v)
 return v
R.create_once=types.FunctionType(f.__code__.replace(co_consts=tuple(replace(v) for v in f.__code__.co_consts)),f.__globals__,f.__name__,f.__defaults__,f.__closure__)
receipt=B/'run-r1-full504-v1/ROOT-CONTROLLER-COMPAT-v3.json'
value={'root_reviewed':True,'run_id':q['run_id'],'science_source_sha256':d['source_package']['sha256'],'changes':{'stage_source_location':'frozen package root, not prepared controls directory','intent_dependency_fields':mapping},'cloud_Create_calls_before':0,'old_unknown_source_PUTs_preserved':True,'source_now_staged_whole_SHA':R.logged(P/'run-v1','STAGED-v1.json'),'scientific_bytes_modified':False,'original_deadline_resource_and_once_Create_checks_preserved':True}
assert not receipt.exists();receipt.write_text(json.dumps(value,indent=2)+'\n')
sp=root_adapter.module('_thread_scope_sciencev3',root_adapter.CLOSEOUT/'resource-thread-cap-v1/thread_scope.py');sp.authority()
sys.argv=[str(P/'root.py'),'execute','--prepared',str(P/'prepared-v1'),'--state',str(P/'run-v1'),'--review',str(P/'ROOT-REVIEW-v1.json'),'--registry',str(sp.ACCOUNT),'--external-inventory',str(root_adapter.CLOSEOUT/'comparison-v1/incremental-floor0-u1136-n474-v1/cpu-execution-v1/EXTERNAL-NOT-APPLICABLE-v1.json')]
raise SystemExit(R.main())
