"""Bounded four-root recorded-command CPU replay, never a reaction policy."""
import hashlib,json,re
from pathlib import Path,PurePosixPath
RESOURCES={'CPU':8,'GPU':0,'memory_GiB':32}
LIMITS={'input_limit_bytes':2*1024**3,'transfer_limit_bytes':4*1024**3,'output_limit_bytes':32*1024**2,
        'cache_limit_bytes':1024**3,'private_tmp_limit_bytes':4*1024**3,'mac_readback_limit_bytes':8*1024**2}
PHASES={'science_runtime_seconds':900,'archive_target_seconds':1050,'max_runtime_seconds':1200}
BASE='redacted-2ac5a54c7015'
IMAGE='onevla-model-cn-north-1.jcr.service.jdcloud.com/one-vla/adv-vla-light@sha256:8a0b1fb7132dcef3b799acb883cc1b6e6e08487eb220a580b7051cb5db4b2fb4'
ROOTS=['SDRU-CDEV-T00-R00','SDRU-CDEV-T01-R00','SDRU-CDEV-T00-R01','SDRU-CDEV-T01-R01']
def need(ok,code):
    if not ok:raise ValueError(code)
def sha(raw):return hashlib.sha256(raw).hexdigest()
def encoded(v):return json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()+b'\n'
def ident(path):
    p=Path(path);need(p.is_file()and not p.is_symlink(),'regular_source_or_result');h=hashlib.sha256();before=p.stat()
    with p.open('rb')as f:
        for raw in iter(lambda:f.read(1024**2),b''):h.update(raw)
    after=p.stat();need((before.st_ino,before.st_size,before.st_mtime_ns)==(after.st_ino,after.st_size,after.st_mtime_ns),'file_changed_during_hash')
    return {'bytes':after.st_size,'sha256':h.hexdigest()}
def relative(value):
    need(type(value)is str and value and '\\'not in value,'canonical_relative');p=PurePosixPath(value)
    need(not p.is_absolute()and str(p)==value and all(v not in('','.','..')for v in value.split('/')),'canonical_relative');return value
def request_gate(r):
    need(r.get('schema')=='recorded-replay-CPU-request-v1'and re.fullmatch(r'jinwen-state-response-replay-cpu-20261013-[a-z0-9-]+',r.get('run_id','')),'explicit_new_replay_run')
    need(r.get('output_prefix')==BASE+r['run_id']+'/'and r.get('resources')==RESOURCES and r.get('image')==IMAGE,'same_image_GPU0_CPU8_RAM32')
    need(all(r.get(k)==v for k,v in{**LIMITS,**PHASES}.items()),'fixed_replay_byte_time_caps')
    need(r.get('model_calls')==r.get('training_steps')==0 and r.get('env_steps_max')==480 and r.get('root_ids')==ROOTS,'fixed_four_root_model_free_steps')
    need(r.get('rng_mode')=='PRIMARY_NO_RNG_INJECTION'and r.get('rng_restores')==0,'no_rng_injection')
    objects=r.get('read_objects');scopes=r.get('read_prefixes')
    need(type(objects)is list and len(objects)>0 and type(scopes)is list and len(scopes)==6 and len(set(scopes))==6,'four_source_two_runtime_prefixes')
    need(len({o['uri']for o in objects})==len(objects),'unique_exact_input_objects')
    for o in objects:
        need(type(o['bytes'])is int and 0<o['bytes']<=64*1024**2 and re.fullmatch('[0-9a-f]{64}',o['sha256']),'exact_input_size_SHA')
        matches=[p for p in scopes if o['uri'].startswith(p)];need(len(matches)==1 and o['object_key']==o['uri'][len(matches[0]):],'exact_object_within_bound_scope')
        relative(o['object_key'])
    need(sum(o['bytes']for o in objects)==r['planned_input_body_bytes']<LIMITS['cache_limit_bytes'],'exact_once_readset_within_input_cache')
    return r
