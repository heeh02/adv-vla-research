"""Availability/identifiability audit only; reads no endpoint outcomes or effect sizes."""
from pathlib import Path
import json,hashlib
from fractions import Fraction
from collections import Counter
H=Path(__file__).resolve().parent
coverage=json.loads((H/'after-stage1-closeout-v6/COVERAGE-IDENTITIES-v1.json').read_text())['coverage']['Plus'];weights=json.loads((H/'WEIGHTS-v1.json').read_text());rows=[r for r in weights['rows']if r['panel']=='Plus'];models=list(coverage)
sets={m:set(v)for m,v in coverage.items()};t30=[m for m in models if'u30000'in m]
contrasts=[]
for seed in [0,1]:
 parent=f'T-s{seed}-u20000-EMA';floor=f'T-floor-s{seed}-u30000-EMA';reheat=f'T-reheat-s{seed}-u30000-EMA'
 contrasts += [(f'floor_s{seed}_T30_minus_T20',parent,floor),(f'reheat_s{seed}_T30_minus_T20',parent,reheat),(f'reheat_minus_floor_s{seed}_T30',floor,reheat)]
out=[]
for name,a,b in contrasts:
 for target in ['mapped_task_macro','unmapped_layout_descriptive']:
  group=[r for r in rows if r['target']==target];common=sets[a]&sets[b];available=sum((Fraction(r['weight_numerator'],r['weight_denominator'])for r in group if r['ordinal']in common),Fraction(0));parentmissing=sum((Fraction(r['weight_numerator'],r['weight_denominator'])for r in group if r['ordinal']not in sets[a]),Fraction(0))
  out.append(dict(contrast=name,target=target,planned=len(group),both_endpoint_count=sum(r['ordinal']in common for r in group),both_endpoint_weight=float(available),unidentified_weight_at_least=float(1-available),minimum_full_target_bound_width_before_q0=2*float(1-available),control_missing_weight=float(parentmissing),actual_qualified_weight_not_asserted=True))
common4=set.intersection(*(sets[m]for m in t30));common7=set.intersection(*sets.values());main=[r for r in rows if r['target']=='mapped_task_macro'];breadth={'four_models':len({r['family']for r in main if r['ordinal']in common4}),'seven_models':len({r['family']for r in main if r['ordinal']in common7})}
value={'schema':'fixed_panel_availability_weight_audit-v1','outcomes_read':0,'actual_pair_gate_not_substituted':True,'coverage_source':'after-stage1-closeout-v6/COVERAGE-IDENTITIES-v1.json','weights_sha256':hashlib.sha256((H/'WEIGHTS-v1.json').read_bytes()).hexdigest(),'contrasts':out,'covered_task_families_out_of40':breadth,'bound_is_logical_identification_not_sampling_CI':True}
p=H/'WEIGHTED-COVERAGE-v1.json';p.write_text(json.dumps(value,indent=2)+'\n');print(json.dumps(value,indent=2))
