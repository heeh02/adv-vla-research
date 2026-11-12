from pathlib import Path
import json,gzip,hashlib
H=Path(__file__).resolve().parent;B=H.parent/'rss-panel-joint-v1'
def canon(x):return json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def ident(b):return dict(bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def ref(p):return {'path':str(p.resolve()),**ident(p.read_bytes())}
s=json.loads((B/'prepared-v1/INPUT-SPEC-v1.json').read_bytes());tracebytes=(B/'derived-trace.jsonl.gz').read_bytes();publication=json.loads((B/'ACTUAL-ACCEPTANCE-v1.json').read_text())['publications']['NEW-PAIR-TRACE-v1.jsonl.gz'];assert ident(tracebytes)=={k:publication[k]for k in ['bytes','sha256']};traces=[json.loads(x)for x in gzip.decompress(tracebytes).splitlines()];selected=[r for r in traces if r['contrast']in ['floor_s0_T30_minus_T20','floor_s1_T30_minus_T20']and r['diagnostic']['reason']=='q0_input_difference'];assert len(selected)==107
pairs=[];keys=set()
for r in selected:
 a=r['control_source'];seed=0 if r['contrast'].startswith('floor_s0')else 1;model=f'T-floor-s{seed}-u30000-EMA';keys.add((model,r['ordinal']));assert 'JD-ENDPOINT-DELTA-v1.jsonl.gz'in a['cache']['object']['uri'];pairs.append({'control_model':f'T-s{seed}-u20000-EMA','treatment_model':model,'ordinal':r['ordinal'],'diagnostic':r['diagnostic'],'control_outcome_SHA':a['cache']['row_identity']['outcome_canonical_sha256']})
rawrows=json.loads(gzip.decompress((B/'prepared-v1/NEW-ENDPOINTS-v1.json.gz').read_bytes()));rows=[r for r in rawrows if(r['model_id'],r['ordinal'])in keys];by={(r['model_id'],r['ordinal']):r for r in rows};assert len(rows)==107
for p in pairs:p['treatment_outcome_SHA']=by[(p['treatment_model'],p['ordinal'])]['outcome_canonical_sha256']
fullsummaries=json.loads(gzip.decompress((B/'prepared-v1/NEW-SUMMARIES-v1.json.gz').read_bytes()));summaries={r['summary']['uri']:fullsummaries[r['summary']['uri']]for r in rows};base=selected[0]['control_source']['cache']['object'];assert all(r['control_source']['cache']['object']==base for r in selected)
refs=[base]+[r['outcome']for r in rows];s.update(analysis_kind='fixed_RSS_q0_source_audit',model_id='two-floor-T30-parent-q0-audit',run_id='jinwen-paper3-rss-q0-source-audit-20261021-v1',output_prefix='redacted-4080ad4d72e5',new_count=107,adoption_total=107,selected_pairs=pairs,baseline_cache=base,new_endpoint_identity=[{'model_id':r['model_id'],'ordinal':r['ordinal'],'outcome_canonical_sha256':r['outcome_canonical_sha256']}for r in rows],readset={'objects':refs,'object_count':108,'exact_total_bytes':sum(r['bytes']for r in refs),'old_OUTCOME_GETs':0,'summary_cloud_GETs':0},diagnostic_selection_rule='all107 floor-parent q0-rejected pairs in already verified first-stage trace; no outcome ranking, no rerun',source_trace=ref(B/'derived-trace.jsonl.gz'))
for name in ['worker.py','cloud_adapter.py']:s[name.removesuffix('.py')+'_source']=ident((H/name).read_bytes())
out=H/'prepared-v1';out.mkdir();(out/'DEPENDENCIES-v1.json').write_bytes((B/'prepared-v1/DEPENDENCIES-v1.json').read_bytes());(out/'NEW-ENDPOINTS-v1.json.gz').write_bytes(gzip.compress(canon(rows),mtime=0));(out/'NEW-SUMMARIES-v1.json.gz').write_bytes(gzip.compress(canon(summaries),mtime=0))
for name in ['DEPENDENCIES-v1.json','NEW-ENDPOINTS-v1.json.gz','NEW-SUMMARIES-v1.json.gz']:s[name]=ident((out/name).read_bytes())
s['frozen_dependencies']=s['DEPENDENCIES-v1.json'];(out/'INPUT-SPEC-v1.json').write_bytes(canon(s)+b'\n');print({'q0_pairs':107,'input_bytes':s['readset']['exact_total_bytes'],'baseline_GETs':1,'new_GPU_or_env_calls':0})
