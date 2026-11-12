from pathlib import Path
import json,gzip,hashlib
H=Path(__file__).resolve().parent;C=H.parents[1];P=C.parent/'rss-common-panel-20261021-v1'
def ident(b):return dict(bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def canon(x):return json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def ref(p):return {'path':str(p.resolve()),**ident(p.read_bytes())}
rows=[];summaries={};readset={};specs=[]
for label in ['floor-s0','floor-s1','reheat-s0','reheat-s1']:
 d=H/'inputs-by-model'/label;s=json.loads((d/'INPUT-SPEC-v1.json').read_bytes());specs.append(s);rows+=json.loads(gzip.decompress((d/'NEW-ENDPOINTS-v1.json.gz').read_bytes()));summaries.update(json.loads(gzip.decompress((d/'NEW-SUMMARIES-v1.json.gz').read_bytes())))
 for r in s['readset']['objects']:
  if r['uri']in readset:assert readset[r['uri']]==r
  readset[r['uri']]=r
assert len(rows)==423==len(summaries)and len(readset)==425
out=H/'prepared-v1';out.mkdir();s=dict(specs[0]);panel=json.loads((P/'PANEL-FROZEN-v1.json').read_bytes());weights=json.loads((P/'WEIGHTS-v1.json').read_bytes());s.update(analysis_kind='fixed_RSS_panel_partial_joint',model_id='four-fixed-T30-EMA-panel',run_id='jinwen-paper3-rss-panel-stage1-joint-20261021-v1',source_models={r['model_id']:specs[i]['model_source']for i,r in enumerate([next(x for x in rows if x['model_id']==ss['model_id'])for ss in specs])},source_runs={ss['model_id']:ss['run_id']for ss in specs},panel=panel,panel_ref=ref(P/'PANEL-FROZEN-v1.json'),weights=weights,weights_ref=ref(P/'WEIGHTS-v1.json'),selected_case_definition_SHA=hashlib.sha256(canon(panel['case_definitions'])).hexdigest(),new_count=423,adoption_total=423,already_processed=0,new_endpoint_identity=[{'model_id':r['model_id'],'ordinal':r['ordinal'],'outcome_canonical_sha256':r['outcome_canonical_sha256']}for r in sorted(rows,key=lambda r:(r['model_id'],r['ordinal']))],adoption_sources=[r for ss in specs for r in ss['adoption_sources']],output_prefix='redacted-4839b54e79b8',scope='CPU-only fixed1053 partial joint + reusable subset baseline cache; no new science',readset={'objects':sorted(readset.values(),key=lambda r:r['uri']),'object_count':425,'exact_total_bytes':sum(r['bytes']for r in readset.values()),'old_OUTCOME_GETs':0,'summary_cloud_GETs':0,'new_outcome_GETs':423})
s['contrasts']=[r for r in json.loads((C/'comparison-v1/INPUT-PLAN-v1.json').read_text())['contrasts']if not r['id'].startswith('T20_')];assert len(s['contrasts'])==10
for name in ['worker.py','cloud_adapter.py']:s[name.removesuffix('.py')+'_source']=ident((H/name).read_bytes())
(out/'DEPENDENCIES-v1.json').write_bytes((H/'inputs-by-model/floor-s0/DEPENDENCIES-v1.json').read_bytes())
(out/'NEW-ENDPOINTS-v1.json.gz').write_bytes(gzip.compress(canon(rows),mtime=0));(out/'NEW-SUMMARIES-v1.json.gz').write_bytes(gzip.compress(canon(summaries),mtime=0))
for n in ['DEPENDENCIES-v1.json','NEW-ENDPOINTS-v1.json.gz','NEW-SUMMARIES-v1.json.gz']:s[n]=ident((out/n).read_bytes())
s['frozen_dependencies']=s['DEPENDENCIES-v1.json'];(out/'INPUT-SPEC-v1.json').write_bytes(canon(s)+b'\n');print({'new_endpoints':423,'objects':425,'bytes':s['readset']['exact_total_bytes'],'baseline_GETs':2,'new_GPU':0})
