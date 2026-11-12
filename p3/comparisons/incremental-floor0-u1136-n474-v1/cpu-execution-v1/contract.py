"""Independent zero-GPU transport for the exact474 addition to the3152 snapshot."""
import hashlib,json,re
from pathlib import Path,PurePosixPath
RUN='jinwen-paper3-floor0-plus-u1136-added474-20261020-v1'
OUTPUT='redacted-9dff0b6ceaff'+RUN+'/'
QUEUE='redacted-68f1c47a0c15';PARENT='redacted-6e77221ca60e';POOL='aihc-serverless'
OWNER='3a59a0c408234b37a90a806c3ddc7a5a'
IMAGE='onevla-model-cn-north-1.jcr.service.jdcloud.com/one-vla/adv-vla-light@sha256:8a0b1fb7132dcef3b799acb883cc1b6e6e08487eb220a580b7051cb5db4b2fb4'
RESOURCES={'CPU':2,'GPU':0,'memory_GiB':8}
PHASES={'science_runtime_seconds':3300,'archive_target_seconds':3480,'max_runtime_seconds':3600}
LIMITS={'input_limit_bytes':400*1024**2,'transfer_limit_bytes':512*1024**2,'output_limit_bytes':32*1024**2,'private_tmp_limit_bytes':1024**3,'mac_readback_limit_bytes':8*1024**2}
SMALL=('DELIVERY-v1.json','RESULT-v1.json','SCIENTIFIC-MANIFEST-v1.json','INPUT-RECEIPT-v1.json','QUALIFIED-COVERAGE-CHANGE-v1.json','CACHE-MANIFEST-v1.json','FAILURE-v1.json')
THREAD_SCOPE_INSTALLED=False

def need(ok,code):
    if not ok:raise ValueError(code)
def encoded(v):return json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()+b'\n'
def identity(raw):return {'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
def ident(p):return identity(Path(p).read_bytes())
def exact(p,r):need(ident(p)=={k:r[k] for k in ('bytes','sha256')},'source_identity')
def read(p):return json.loads(Path(p).read_bytes())
def safe_error(e):return {'type':type(e).__name__,'code':str(e) if re.fullmatch('[A-Za-z0-9_.:-]{1,160}',str(e)) else 'redacted'}
def relative(v):
    p=PurePosixPath(v);need(str(p)==v and not p.is_absolute() and '\\' not in v and all(x not in ('','.','..') for x in v.split('/')),'safe_relative');return v
def account_gate(*a,**k):raise ValueError('mandatory_thread_scope_account_adapter_not_installed')

def read_acl(refs):
    need(len(refs)==478 and sum(r['bytes'] for r in refs)==322946637,'exact_increment_readset')
    scope=set();new_prefix='redacted-a604e543d689'
    for r in refs:
        uri=r['uri'];need(uri.startswith('redacted-24fc043a6cae'),'Jinwen_only')
        if uri.startswith(new_prefix):
            tail=uri[len(new_prefix):];need(re.fullmatch(r'p[0-9]{5}/OUTCOME-v1\.json',tail),'only_new474_OUTCOMEs')
            need(1136<=int(tail.split('/')[0][1:])<=1609,'only_new474')
            scope.add(new_prefix.removeprefix('bos://')+'*/'+tail.split('/')[-1])
        else:
            need(uri.endswith(('ENDPOINTS-v1.jsonl.gz','JD-ENDPOINT-DELTA-v1.jsonl.gz','SINGLE-GROUPS-v1.csv.gz','PAIRED-GROUPS-v1.csv.gz')),'only_exact_old_cache_or_table')
            scope.add(uri.removeprefix('bos://'))
    need(len(scope)==5,'four_exact_objects_one_run_OUTCOME_pattern')
    return sorted(scope)

def request_gate(q):
    need(q['schema']=='paper3-Plus474-increment-CPU-request-v1' and q['run_id']==RUN and q['output_prefix']==OUTPUT,'exact_new_transport_identity')
    need(q['queue_id']==QUEUE and q['physical_queue_id']==PARENT and q['resources']==RESOURCES,'allowed_queue_CPU0')
    need(q['phases']==PHASES and q['limits']==LIMITS,'fixed_one_hour_byte_envelope')
    need(q['input_objects']==478 and q['input_bytes']==322946637 and q['old3152_OUTCOME_GETs']==0,'unchanged_exact_increment_readset')
    need(q['new_model_calls']==q['new_episodes']==q['JPFS_new_bytes']==0,'existing_metadata_only')
    need(q['credential_scope_wider_than_runtime_readset'] is True and q['input_permission']==['READ'],'honest_compact_READ_only_scope')
    need(q['full_population_evaluation_complete'] is False and q['snapshot_T30_count']==3626,'snapshot_not_population_completion')
    return q
