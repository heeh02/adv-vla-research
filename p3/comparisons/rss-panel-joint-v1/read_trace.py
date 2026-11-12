"""Fetch one derived1.3MB qualification table; no raw arrays/model/OUTCOME GET."""
from pathlib import Path
import json,hashlib,sys,types,gzip,io
from urllib.parse import urlsplit
from collections import defaultdict,Counter
H=Path(__file__).resolve().parent;C=H.parents[1];P=C.parent/'rss-common-panel-20261021-v1'
import root_adapter
receipt=json.loads((H/'ACTUAL-ACCEPTANCE-v1.json').read_text());r=receipt['publications']['NEW-PAIR-TRACE-v1.jsonl.gz'];assert r['bytes']<2*1024**2 and r['whole_GET_SHA_verified'];dest=H/'derived-trace.jsonl.gz';meter={'consumed':0}
if dest.exists():data=dest.read_bytes()
else:
 R,_,_=root_adapter.prepare(H/'cpu0-package-v1');f=R.bounded_get_function;R.bounded_get_function=types.FunctionType(f.__code__,{**f.__globals__,'HERE':root_adapter.BASE.parent},f.__name__,f.__defaults__,f.__closure__);get=R.bounded_get_function();host,(L,M,B,proxy,life,stop,G,Q)=R.helpers();backend=None;auth={};env={}
 try:
  with host.quiet(),proxy():
   auth=L.read_RAM_auth();backend=B(auth,None);env=backend.sts([{'service':'bce:bos','region':'bj','effect':'Allow','resource':[r['uri'].removeprefix('bos://')],'permission':['READ']}],900);client=backend.old._bos_client(env);u=urlsplit(r['uri']);data=get(client,u.netloc,u.path.lstrip('/'),meter=meter,cap=2*1024**2,expect=r);dest.write_bytes(data)
 finally:
  if backend:backend.close()
  env.clear();auth.clear()
assert len(data)==r['bytes']and hashlib.sha256(data).hexdigest()==r['sha256']
with gzip.GzipFile(fileobj=io.BytesIO(data))as stream:decoded=stream.read(32*1024**2+1)
assert len(decoded)<=32*1024**2;rows=[json.loads(x)for x in decoded.splitlines()if x];assert len(rows)==10530
panel=json.loads((P/'PANEL-FROZEN-v1.json').read_text());cases=dict(zip(panel['ordinals'],panel['case_definitions']));groups=defaultdict(Counter)
for row in rows:
 c=cases[row['ordinal']];diag=row['diagnostic'];reason=diag['reason']
 for kind,key in [('suite',c['suite']),('category',c['category']),('suite_category',c['suite']+' | '+c['category'])]:
  v=groups[(row['contrast'],kind,key)];v['planned']+=1;v[reason]+=1
  if row['control_source']and row['treatment_source']:v['both_endpoints']+=1
  if reason=='q0_input_difference':
   for f in ['q0','runtime','shared_noise']:v['rejected_'+f+':'+diag[f]]+=1
  if row['cell']:v[row['cell']]+=1
out={'schema':'RSS-stage1-existing-derived-trace-strata-v1','trace_publication':r,'trace_GET_bytes_this_call':meter['consumed'],'raw_ARRAY_OUTCOME_model_GETs':0,'full_fixed_panel_claimed':False,'groups':[{'contrast':x[0],'group_kind':x[1],'group':x[2],**dict(v)}for x,v in sorted(groups.items())]}
(H/'QUALIFICATION-STRATA-v1.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps([g for g in out['groups']if g['group_kind']=='suite'and 'minus_T20'in g['contrast']],indent=2))
