"""Read-only metadata census and deterministic outcome-blind sampling. No cloud mutation."""
from pathlib import Path
import json,hashlib,gzip,csv,importlib.util,argparse
from collections import defaultdict,Counter
from datetime import datetime,timezone
HERE=Path(__file__).resolve().parent; C=HERE.parent/'eval-closeout-v1'
def load(p):return json.loads(Path(p).read_bytes())
def canonical(x):return json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()
def sha(x):return hashlib.sha256(x).hexdigest()
def ref(p):p=Path(p);b=p.read_bytes();return dict(path=str(p),bytes=len(b),sha256=sha(b))
def module(name,p):s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
SOURCES={}
OUT=HERE
COUNTS=None
def exact(r):
 p=Path(r['path']);b=p.read_bytes();assert len(b)==r['bytes'] and sha(b)==r['sha256'],p;SOURCES[str(p)]=r;return json.loads(b)
def save(n,x):(OUT/n).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def main():
 route=module('route',C/'reheat-floor-commoncase-v1/analyze.py')
 routeproof=C/'plus-init-family-evidence-v1/ROUTING-EVIDENCE-v1.json';cases,routes=route.metadata(load(routeproof));SOURCES[str(routeproof)]=ref(routeproof)
 # Only immutable case descriptors enter this selector. Neither availability nor outcomes enter.
 groups=defaultdict(list)
 for o,(case,r)in enumerate(zip(cases,routes)):
  family=r['init_family'] or 'UNMAPPED_LAYOUT:'+case['suite']
  key=(case['suite'],family,case['category'],str(case['difficulty_level']))
  groups[key].append(o)
 seed='RSS-paper3-common-panel-v1-20261021'
 selected=sorted(min(v,key=lambda o:sha(seed.encode()+b'\0'+canonical(cases[o]))) for v in groups.values())
 selection={'rule':'One fixed-hash minimum per suite x mapped original init-family x category x official difficulty; unmapped layout uses suite/category/difficulty and remains unmapped. No outcomes or availability used.', 'salt':seed,'n':len(selected),'ordinals':selected,'strata_count':len(groups),'population_sha256':route.POP,'case_definitions':[cases[o]for o in selected], 'not_population_proportional':True,'not_power_guarantee':True}
 if (HERE/'PANEL-FROZEN-v1.json').exists():assert load(HERE/'PANEL-FROZEN-v1.json')==selection, 'frozen panel changed; do not resample'
 else:save('PANEL-FROZEN-v1.json',selection)
 plan=load(C/'comparison-v1/INPUT-PLAN-v1.json');t30=[m for m in plan['models']if 'u30000' in m];base=['T-s0-u20000-EMA','T-s1-u20000-EMA','original'];models=t30+base
 coverage={p:{m:set()for m in models}for p in ['Plus','L10']};ident={};assigned=defaultdict(set);inflight=defaultdict(set)
 compare=module('compare',C/'comparison-v1/compare_endpoints.py')
 ads=list((C/'root/results').glob('*/ADOPTION-v1.json'))
 ads.append(C/'control-recovery-v1/PLUS-ADOPTION-APPLIED-v1.json')
 ads+=list((HERE.parents[1]/'t20-optimization-continuation-20261019-v1/evaluation/results').glob('*/ADOPTION-v1.json'))
 for p in sorted(ads):
  ad=load(p)
  if ad.get('panel','Plus')!='Plus' or ad.get('applied')is False:continue
  if ad.get('model_id')not in t30 and 'model_id'in ad:continue
  SOURCES[str(p)]=ref(p)
  if ad.get('schema')=='T30-Plus-recovery-case-adoption-applied-v1':
   assert ad['prior403_rows_unchanged'] and ad['new_retry_authorized']is False
   ledger=exact(ad['after']);closed=[r for r in ledger['rows']if r['state']=='closed'];assert len(closed)==474
   coverage['Plus']['T-floor-s0-u30000-EMA'].update(r['ordinal']for r in closed);continue
  assert ad.get('old_results_overwritten')is False and ad.get('scientific_attempt_changed')is False
  m=ad.get('model_id','T-floor-s0-u30000-EMA');assert m in t30
  for o in ad['adopted_ordinals']:coverage['Plus'][m].add(o)
 activepath=C/'root/ACTIVE-BATCH-v1.json';active=load(activepath);SOURCES[str(activepath)]=ref(activepath)
 l10cases={}; ledgers=set()
 for u in active['units']:
  p=Path(u['case_ledger_path'])
  if p.exists():d=load(p);SOURCES[str(p)]=ref(p)
  else:
   assert u['panel']=='Plus' and u.get('GPU_release_proven'), 'missing current ledger'
   lo,hi=u['ordinals'];assert hi-lo+1==u['case_count']
   d={'rows':[{'ordinal':o,'state':'archived_assignment'} for o in range(lo,hi+1)]}
  panel=u['panel'];m=u['model_id'];assert m in t30
  for row in d['rows']:
   o=row['ordinal']
   if panel=='Plus' and 'case_definition_sha256'in row:assert row['case_definition_sha256']==compare.digest(cases[o]), 'full case identity'
   if 'checkpoint_sha256'in row:assert row['checkpoint_sha256']==plan['models'][m]['checkpoint_sha256'], 'fixed endpoint identity'
   assigned[(panel,m)].add(o)
   if not u.get('GPU_release_proven'):inflight[(panel,m)].add(o)
   if panel=='L10'and row['state']=='closed':
    assert row['checkpoint_sha256']==plan['models'][m]['checkpoint_sha256']
    assert row['native_terminal']['outcome']['headers_and_member_SHA_verified'];SOURCES[row['independent_readback']['path']]=row['independent_readback']
    coverage[panel][m].add(o)
  if panel=='L10' and not l10cases:
   req=exact(u['request']);l10cases=dict(zip(req['ordinals'],req['rows']));assert len(l10cases)==300
 # Cloud-verified derived singleton table: metadata only; ignore success/normal_failure columns.
 table=C/'comparison-cpu-execution-v1/small-preview-v1/compressed/SINGLE-GROUPS-v1.csv.gz'
 pin=next(r for r in load(C/'comparison-cpu-execution-v1/repair-v1/INPUT-SPEC-v1.json')['tables']if r.get('original_name')=='SINGLE-GROUPS-v1.csv')
 assert table.stat().st_size==pin['bytes']and sha(table.read_bytes())==pin['sha256'];SOURCES[str(table)]=ref(table)
 lookup={json.dumps([c[k]for k in ('suite','task_id','task_name')],separators=(',',':')):o for o,c in enumerate(cases)}
 with gzip.open(table,'rt')as f:
  for r in csv.DictReader(f):
   if r['model']in base and r['panel']=='Plus'and r['group_kind']=='task':
    assert r['planned']=='1'and r['closed']in ['0','1']
    if r['closed']=='1':coverage['Plus'][r['model']].add(lookup[json.dumps(json.loads(r['group']),separators=(',',':'))])
 # Audit of historical L10 exact 44 closures; original300 is separately verified by the existing CPU coverage.
 old=C/'l10-t20-baseline-gap-audit-v1/EXISTING-44-v1.jsonl.gz';SOURCES[str(old)]=ref(old)
 with gzip.open(old,'rt')as f:
  rows=[json.loads(l)for l in f if l.strip()]
 for r in rows:coverage['L10'][r['model_id']].add(r['ordinal'])
 original_result=load(C/'l10-endpoint-strata527-v1/cpu-execution-v2/run-v1/readback/science/RESULT-v1.json');assert original_result['Original_closed']==300 and original_result['T20_closed_each_seed']==22
 coverage['L10']['original']=set(range(300))
 oldreport=C/'l10-endpoint-strata527-v1/cpu-execution-v2/run-v1/readback/science/RESULT-v1.json';SOURCES[str(oldreport)]=ref(oldreport)
 # Proven T30 counts must match latest root count receipt; failures are retained in coverage.
 expected={'T-floor-s0-u30000-EMA':2113,'T-floor-s1-u30000-EMA':1511,'T-reheat-s0-u30000-EMA':504,'T-reheat-s1-u30000-EMA':1008}
 if COUNTS:
  snapshot=load(COUNTS);expected={f'T-{k.replace("_","-")}-u30000-EMA':v['closed']for k,v in snapshot['Plus'].items()}
 assert {m:len(coverage['Plus'][m])for m in t30}==expected, {m:len(coverage['Plus'][m])for m in t30}
 assert sum(len(coverage['L10'][m])for m in t30)==605
 clean=[o for o,c in l10cases.items()if c.get('condition_id')=='C0_Hbase']
 if not clean:clean=[o for o,c in l10cases.items()if 'C0_Hbase'in json.dumps(c)]
 assert len(clean)==50,('clean count',len(clean),l10cases[0])
 save('COVERAGE-IDENTITIES-v1.json',{'coverage':{p:{m:sorted(v)for m,v in d.items()}for p,d in coverage.items()},'assigned':{p:{m:sorted(assigned[(p,m)])for m in t30}for p in coverage},'potential_inflight':{p:{m:sorted(inflight[(p,m)])for m in t30}for p in coverage}})
 out={'timestamp_utc':datetime.now(timezone.utc).isoformat(),'scope':'Endpoint availability only, not strict q0/runtime/noise eligibility; no outcomes used for selection.', 'panels':{},'selected_Plus':{},'selected_L10_clean':{},'missing_is_not_dispatch_authority':True}
 for p in coverage:
  d=coverage[p];common4=set.intersection(*(d[m]for m in t30));common7=set.intersection(*(d[m]for m in models))
  out['panels'][p]={'counts':{m:len(v)for m,v in d.items()},'common_four_T30':len(common4),'common_seven_endpoints':len(common7),'common_four_ordinals':sorted(common4),'common_seven_ordinals':sorted(common7)}
 gaps=[]
 for panel,sel,label in [('Plus',selected,'selected_Plus'),('L10',clean,'selected_L10_clean')]:
  sel=set(sel);out[label]['cases']=len(sel);out[label]['models']={}
  for m in models:
   missing=sel-coverage[panel][m];in_flight=missing&inflight[(panel,m)] if m in t30 else set();historical=missing&assigned[(panel,m)]-in_flight if m in t30 else missing
   never=missing-assigned[(panel,m)] if m in t30 else set()
   out[label]['models'][m]={'existing':len(sel&coverage[panel][m]),'missing_endpoint':len(missing),'current_registered_unclosed':len(in_flight),'historically_assigned_unclosed_needs_evidence':len(historical),'no_registered_assignment_in_active_manifest':len(never)}
   for o in sorted(missing):gaps.append(dict(panel=panel,model_id=m,ordinal=o,case=cases[o]if panel=='Plus'else l10cases[o],classification='registered_unclosed'if o in in_flight else'historical_unclosed_not_rerunnable'if o in historical else'no_registered_assignment_requires_full_namespace_check',automatic_rerun=False))
  out[label]['common_four_existing']=len(sel&set.intersection(*(coverage[panel][m]for m in t30)))
  out[label]['common_seven_existing']=len(sel&set.intersection(*(coverage[panel][m]for m in models)))
 out['full_existing_Plus_common_strata']=dict(Counter(cases[o]['category']for o in out['panels']['Plus']['common_four_ordinals']))
 out['panel_categories']=dict(Counter(cases[o]['category']for o in selected));out['panel_suites']=dict(Counter(cases[o]['suite']for o in selected))
 save('CENSUS-v1.json',out);save('GAPS-NOT-DISPATCH-v1.json',gaps);save('SOURCES-v1.json',list(SOURCES.values()));print(json.dumps(out,ensure_ascii=False)[:1000]);print('TOTAL MISSING',len(gaps))
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,default=HERE);ap.add_argument('--counts',type=Path);args=ap.parse_args();OUT=args.out;OUT.mkdir(parents=True,exist_ok=True);COUNTS=args.counts;main()
