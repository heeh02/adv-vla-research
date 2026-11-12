"""Fetch only approved small published derived artifacts using exact references."""
from pathlib import Path
import argparse,json,hashlib,sys,types
from urllib.parse import urlsplit
ap=argparse.ArgumentParser();ap.add_argument('--analysis',type=Path,required=True);ap.add_argument('--name',action='append',required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();h=a.analysis.resolve();assert h.parent==Path(__file__).resolve().parent
acc=json.loads((h/'ACTUAL-ACCEPTANCE-v1.json').read_bytes());assert acc['CPU_released'];refs=acc['publications'];chosen=[refs[n]for n in a.name];assert len(chosen)==len(set(a.name))and sum(x['bytes']for x in chosen)<=8*1024**2
assert all(x['whole_GET_SHA_verified']and x['uri'].startswith('redacted-24fc043a6cae')for x in chosen)
a.out.mkdir(parents=True,exist_ok=True);sys.path.insert(0,str(h));import root_adapter
R,_,_=root_adapter.prepare(h/'cpu0-package-v1');f=R.bounded_get_function;R.bounded_get_function=types.FunctionType(f.__code__,{**f.__globals__,'HERE':root_adapter.BASE.parent},f.__name__,f.__defaults__,f.__closure__);get=R.bounded_get_function();H,(L,M,B,proxy,life,stop,G,P)=R.helpers();backend=None;auth={};env={};meter={'consumed':0};receipts=[]
try:
 with H.quiet(),proxy():
  auth=L.read_RAM_auth();backend=B(auth,None);env=backend.sts([{'service':'bce:bos','region':'bj','effect':'Allow','resource':[x['uri'].removeprefix('bos://')for x in chosen],'permission':['READ']}],900);client=backend.old._bos_client(env)
  for name,r in zip(a.name,chosen):
   dest=a.out/name;assert dest.parent==a.out
   if dest.exists():data=dest.read_bytes()
   else:
    u=urlsplit(r['uri']);data=get(client,u.netloc,u.path.lstrip('/'),meter=meter,cap=8*1024**2,expect=r);dest.write_bytes(data)
   assert len(data)==r['bytes']and hashlib.sha256(data).hexdigest()==r['sha256'];receipts.append({'path':str(dest.resolve()),**r})
finally:
 if backend:backend.close()
 env.clear();auth.clear()
report={'GET_bytes':meter['consumed'],'files':receipts,'raw_array_or_OUTCOME_GETs':0,'cloud_mutations':0};(a.out/'READBACK-v1.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'files':len(receipts),'GET_bytes':meter['consumed']}))
