"""One post-hoc shared-context CPU job; no model/environment or auth imports."""
import hashlib,json,re
from pathlib import Path,PurePosixPath
RESOURCES={'CPU':8,'GPU':0,'memory_GiB':32}
LIMITS={'input_limit_bytes':1024**3,'transfer_limit_bytes':2*1024**3,'output_limit_bytes':8*1024**2,'cache_limit_bytes':64*1024**2,'private_tmp_limit_bytes':512*1024**2,'mac_readback_limit_bytes':4*1024**2}
PHASES={'science_runtime_seconds':1800,'archive_target_seconds':2100,'max_runtime_seconds':2400}
BASE='redacted-2ac5a54c7015'
ROOT_ID='CONTEXT-SHARED-ABLATION'
DEV_RUN='jinwen-state-response-cont-20261013-l9-r3'
SHARDS=['CCONF-T06','CCONF-T07','CCONF-T08','CCONF-T09']
SOURCE_BY_TASK={s:'jinwen-state-response-cont-20261013-conf-20261014-'+('v4'if s=='CCONF-T09'else'v3')+'-'+s[-3:].lower()for s in SHARDS}
SOURCE_RUNS=[DEV_RUN,*SOURCE_BY_TASK.values()]
FIT_RUN='jinwen-state-response-taskblind-cpu-20261014-dev-v1'
QUAL_RUN='jinwen-state-response-conf-cpu-20261014-v3'
FIT_ID='31ab045c8a96d4a1e5cdae7d8d41a62ca07056d6892a09ecc4ea65bf178a04e8'
DEV_INPUT_BYTES=224484300;DEV_INPUT_OBJECTS=695
INPUT_BYTES=781960761;INPUT_OBJECTS=1506

def need(ok,code):
 if not ok:raise ValueError(code)
def sha(raw):return hashlib.sha256(raw).hexdigest()
def encoded(v):return json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()+b'\n'
def relative(v):
 need(type(v)is str and v and '\\'not in v,'relative_path');p=PurePosixPath(v)
 need(not p.is_absolute()and str(p)==v and all(k not in('','.','..')for k in v.split('/')),'safe_relative_path');return v
def ident(path):
 if type(path)is bytes:return {'bytes':len(path),'sha256':sha(path)}
 p=Path(path);need(p.is_file()and not p.is_symlink(),'regular_exact_file');before=p.stat();h=hashlib.sha256()
 with p.open('rb')as f:
  for b in iter(lambda:f.read(1024**2),b''):h.update(b)
 after=p.stat();need((before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns)==(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns),'immutable_file_during_hash')
 return {'bytes':after.st_size,'sha256':h.hexdigest()}
def config_gate(c):
 need(c['schema']=='context-ablation-CPU-config-v1'and c['source_run_ids']==SOURCE_RUNS,'fixed_original_source_runs')
 need(c['resources']==RESOURCES and c['limits']==LIMITS and c['phases']==PHASES,'fixed_CPU_limits')
 need(c['model_calls']==c['env_steps']==0 and c['new_fit']is True,'one_shared_only_CPU_refit');return c
