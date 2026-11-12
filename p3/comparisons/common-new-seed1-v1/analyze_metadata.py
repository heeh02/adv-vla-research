"""Bounded derived metadata join only: no raw model/state/action arrays or fitting."""
from pathlib import Path
import sys,json,hashlib,gzip,io,ast,types,time
from collections import Counter,defaultdict
from urllib.parse import urlsplit
HERE=Path(__file__).resolve().parent;C=HERE.parents[1];B=C/'comparison-v1/data-driven-increment-v1';sys.path.insert(0,str(B));import root_adapter

def need(v,s):
 if not v:raise ValueError(s)
def canonical(v):return json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def ref(p):b=p.read_bytes();return {'path':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
def exact(r):
 p=Path(r['path']);b=p.read_bytes();need(len(b)==r['bytes']and hashlib.sha256(b).hexdigest()==r['sha256'],'exact_local_metadata');return json.loads(b)

def main():
 need(not (HERE/'RESULT-v1.json').exists(),'new_analysis_only')
 dirs={'floor':B/'run-floor1-full503-v1','reheat':B/'run-r1-full504-v1'};receipts={k:json.loads((d/'ROOT-SCIENCE-ADOPTION-v1.json').read_text())for k,d in dirs.items()};refs={};results={}
 for label,r in receipts.items():
  need(r['root_reviewed']and r['whole_GET_SHA_verified']and r['CPU_released'],'actual_published_science');results[label]=exact(r['source_result']);refs[label]=r['normalized_delta_publication'];need(refs[label]['whole_GET_SHA_verified']and refs[label]['bytes']<1024**2,'small_published_metadata_only')
 R,_,_=root_adapter.prepare(dirs['reheat']/'cpu0-package-v1');f=R.bounded_get_function;R.bounded_get_function=types.FunctionType(f.__code__,{**f.__globals__,'HERE':root_adapter.BASE.parent},f.__name__,f.__defaults__,f.__closure__);get=R.bounded_get_function();H,(L,M,Backend,proxy,life,stop,G,P)=R.helpers();backend=None;auth={};env={};meter={'consumed':0};objects={}
 try:
  with H.quiet(),proxy():
   auth=L.read_RAM_auth();backend=Backend(auth,None);env=backend.sts([{'service':'bce:bos','region':'bj','effect':'Allow','resource':[v['uri'].removeprefix('bos://') for v in refs.values()],'permission':['READ']}],900);client=backend.old._bos_client(env)
   for label,r in refs.items():
    dest=HERE/(label+'-metadata.jsonl.gz')
    if dest.exists():b=dest.read_bytes();need(len(b)==r['bytes']and hashlib.sha256(b).hexdigest()==r['sha256'],'existing_derived_metadata_same_SHA')
    else:
     u=urlsplit(r['uri']);b=get(client,u.netloc,u.path.lstrip('/'),meter=meter,cap=4*1024**2,expect=r);dest.write_bytes(b)
    with gzip.GzipFile(fileobj=io.BytesIO(b))as s:raw=s.read(32*1024**2+1)
    need(len(raw)<=32*1024**2,'bounded_derived_rows');rows=[json.loads(l)for l in raw.splitlines()if l.strip()];expected={x['ordinal']:x['outcome_canonical_sha256'] for x in results[label]['new_endpoint_identity']}
    need(len(rows)==len(expected)and {x['ordinal']:x['outcome_canonical_sha256'] for x in rows}==expected,'exact_cloud_result_endpoint_identities')
    for row in rows:need(row['model_id']==results[label]['model_id'] and row['attempt']==0 and row['adopted']is True and row['status']=='complete' and type(row['success'])is bool,'only_actual_adopted_metadata')
    objects[label]={x['ordinal']:x for x in rows}
 finally:
  if backend:backend.close()
  env.clear();auth.clear()
 deps=json.loads((dirs['reheat']/'prepared-v1/DEPENDENCIES-v1.json').read_text());pin=next(r for r in deps['files']if r['package_path']=='frozen-bridge/vendor/behavior_tables.py');source=Path(pin['source_path']).read_bytes();need(len(source)==pin['bytes']and hashlib.sha256(source).hexdigest()==pin['sha256'],'same_frozen_original_pair_gate')
 nodes=[n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef)and n.name in ('restore_noise_keys','pair_gate')];need(len(nodes)==2,'two_original_functions');ns={'need':need,'canonical':canonical};exec(compile(ast.Module(body=nodes,type_ignores=[]),pin['source_path'],'exec'),ns)
 for bucket in objects.values():
  for row in bucket.values():ns['restore_noise_keys'](row)
 q=json.loads((C/'plus-proven-tail503-v1/floor1/REQUEST-v1.json').read_text());cases=dict(zip(q['ordinals'],q['cases']));ords=sorted(objects['floor'].keys()&objects['reheat'].keys());need(ords==list(range(1009,1512)),'complete_original_common503_support')
 traces=[];groups=defaultdict(Counter);reason=Counter()
 for o in ords:
  a,b=objects['floor'][o],objects['reheat'][o];need(a['case_definition_sha256']==b['case_definition_sha256']==hashlib.sha256(canonical(cases[o])).hexdigest() and a['population_identity']==b['population_identity'],'same_actual_case_and_population');why=ns['pair_gate'](a,b);reason[why]+=1;cell=('S'if a['success']else'F')+('S'if b['success']else'F')if why=='qualified_recorded_pair'else None
  traces.append({'ordinal':o,'category':cases[o]['category'],'suite':cases[o]['suite'],'reason':why,'cell':cell,'floor_outcome_SHA':a['outcome_canonical_sha256'],'reheat_outcome_SHA':b['outcome_canonical_sha256']})
  for g in ['ALL',cases[o]['category']]:
   groups[g]['planned']+=1;groups[g]['reason:'+why]+=1
   if cell:groups[g][cell]+=1
 output={}
 for name,v in groups.items():
  n=sum(v[k]for k in ['SS','SF','FS','FF']);output[name]={'planned':v['planned'],'qualified':n,'SS':v['SS'],'SF':v['SF'],'FS':v['FS'],'FF':v['FF'],'gains':v['FS'],'harms':v['SF'],'net':v['FS']-v['SF'],'qualified_delta':(v['FS']-v['SF'])/n if n else None,'reasons':{k[7:]:v for k,v in v.items()if k.startswith('reason:')}}
 (HERE/'PAIRED-TRACE-v1.json').write_text(json.dumps(traces,indent=2)+'\n');r={'status':'COMPLETE_DERIVED_METADATA_DIRECT_COMPARISON','schema':'paper3-new-seed1-direct-floor-reheat-v1','control':results['floor']['model_id'],'treatment':results['reheat']['model_id'],'common_case_count':503,'excluded_only_reheat_ordinal':[1008],'groups':output,'original_gate':pin,'input_publications':refs,'input_cloud_science_receipts':{k:ref(d/'ROOT-SCIENCE-ADOPTION-v1.json')for k,d in dirs.items()},'trace':ref(HERE/'PAIRED-TRACE-v1.json'),'metadata_GET_bytes':meter['consumed'],'raw_OUTCOME_GETs':0,'model_or_action_array_GETs':0,'model_calls':0,'new_episodes':0,'computation':'local bounded published metadata index join; no raw arrays/fitting/rendering','population_SR_or_causal_effect_claimed':False,'training_seeds_pooled':False,'CI':None,'timestamp':time.time()};(HERE/'RESULT-v1.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(output))
if __name__=='__main__':main()
