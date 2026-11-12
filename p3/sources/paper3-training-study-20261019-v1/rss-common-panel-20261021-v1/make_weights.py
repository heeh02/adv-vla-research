"""Freeze metadata-only target weights; no outcome/coverage file is read."""
from pathlib import Path
import json,hashlib,importlib.util
from collections import defaultdict
from fractions import Fraction
H=Path(__file__).resolve().parent;C=H.parent/'eval-closeout-v1'
panel=json.loads((H/'PANEL-FROZEN-v1.json').read_text())
s=importlib.util.spec_from_file_location('official_route',C/'reheat-floor-commoncase-v1/analyze.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
proof=json.loads((C/'plus-init-family-evidence-v1/ROUTING-EVIDENCE-v1.json').read_text());cases,routes=m.metadata(proof)
groups=defaultdict(lambda:defaultdict(list));layout=[]
for o,c in zip(panel['ordinals'],panel['case_definitions']):
 assert cases[o]==c
 family=routes[o]['init_family']
 if family:groups[family][c['category']].append(o)
 else:assert c['category']=='Objects Layout';layout.append(o)
assert len(groups)==40 and sum(len(v)for g in groups.values()for v in g.values())==1033 and len(layout)==20
rows=[]
for family,categories in sorted(groups.items()):
 for category,ords in sorted(categories.items()):
  assert len({str(cases[o]['difficulty_level'])for o in ords})==len(ords)
  w=Fraction(1,len(groups)*len(categories)*len(ords))
  for o in sorted(ords):rows.append(dict(panel='Plus',target='mapped_task_macro',ordinal=o,family=family,category=category,difficulty=cases[o]['difficulty_level'],weight_numerator=w.numerator,weight_denominator=w.denominator))
for o in layout:rows.append(dict(panel='Plus',target='unmapped_layout_descriptive',ordinal=o,family=None,category='Objects Layout',difficulty=cases[o]['difficulty_level'],weight_numerator=1,weight_denominator=len(layout)))
q=json.loads((C/'l10-candidate/bindings-full300-v1/floor-s0/REQUEST-v1.json').read_text());clean=[(o,c)for o,c in zip(q['ordinals'],q['rows'])if c['condition_id']=='C0_Hbase'];assert len(clean)==50 and len({c['task']for o,c in clean})==10
for o,c in clean:rows.append(dict(panel='L10',target='clean50',ordinal=o,task=c['task'],init=c['init'],weight_numerator=1,weight_denominator=50))
for target in ['mapped_task_macro','unmapped_layout_descriptive','clean50']:assert sum(Fraction(r['weight_numerator'],r['weight_denominator'])for r in rows if r['target']==target)==1
v={'schema':'paper3-fixed-panel-target-weights-v1','panel_sha256':hashlib.sha256((H/'PANEL-FROZEN-v1.json').read_bytes()).hexdigest(),'outcome_files_read':0,'coverage_files_read':0,'post_historical_exploration_not_blind_preregistration':True,'three_targets_not_pooled':True,'rows':rows}
out=H/'WEIGHTS-v1.json'
if out.exists():assert json.loads(out.read_text())==v
else:out.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
print('1033 mapped +20 layout +50 clean weights checked; each target sums exactly1; no outcomes read')
