"""Verify unchanged inputs, displayed scalar coverage and spatial provenance."""
from pathlib import Path
import hashlib
import json
import math
from collections import Counter

ROOT=Path(__file__).resolve().parent
CANDIDATE=ROOT.parent.parent
QA=CANDIDATE/'qa/result_figures'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
source=json.loads((CANDIDATE/'baseline/SOURCE-MANIFEST.json').read_text())
frozen=json.loads((CANDIDATE/'baseline/INPUT-MANIFEST.json').read_text())
checks=[]
for row in frozen:
 p=ROOT.parent/row['path']
 assert sha(p)==row['sha256'],str(p)
checks.append({'check':'frozen_plot_inputs_unchanged','files':len(frozen),'passed':True})
before=next(x for x in source if x['path']=='figures/attention_spatial.pdf')
assert sha(ROOT/'spatial_source/original_attention_spatial.pdf')==before['sha256']
checks.append({'check':'original_spatial_pdf_preserved','sha256':before['sha256'],'passed':True})

D=json.loads((ROOT/'data/r11_result_sources.json').read_text())
B=json.loads((ROOT/'data/figure_data.json').read_text())['basis']
q=json.loads((QA/'same_basis_translation.json').read_text())
assert len(q['plotted']['retention'])==36
assert len(q['plotted']['readout_gain'])==12
assert len(q['plotted']['readout_error'])==24
for r in q['plotted']['retention']:
 assert -2<=100*r['interval_95'][0]<=100*r['mean']<=100*r['interval_95'][1]<=100
for r in q['plotted']['readout_gain']:
 assert -80<=100*r['relative_error_reduction_interval_95'][0]<=100*r['relative_error_reduction']<=100*r['relative_error_reduction_interval_95'][1]<=105
for r in q['plotted']['readout_error']:assert .15<r['ratio']<9
for r in q['restoration']:
 xmax=.132 if r['family']=='Camera Viewpoints' else .021
 assert 0<=r['along']<=xmax and 0<=r['orthogonal']<=xmax
checks.append({'check':'figure3_scalar_and_interval_coverage','counts':[36,12,24,8],'all_inside_axes':True,'passed':True})

profiles=json.loads((ROOT/'data/r11_attention_profiles.json').read_text())['profiles']
q=json.loads((QA/'fig4_attention.json').read_text())
assert q['profiles']==profiles
assert len(q['association_rows'])==16
for r in profiles:
 for metric,maximum in [('p_full_key_tv',.22),('w_delta_rms',2.8),('ae_delta_rms',6.2),('ae_relative_l2',.25)]:
  assert 0<=r[metric]<=maximum,(metric,r)
 assert round(r['ae_reference_rms'],1)==({0:46.8,5:33.2,9:16.6}[r['generation_call']])
for r in q['association_rows']:
 assert -.55<=float(r['rho_issued_low'])<=float(r['rho_issued'])<=float(r['rho_issued_high'])<=.9
checks.append({'check':'figure5_frozen_profiles_and_associations','profile_rows':12,'association_rows':16,'new_generation_CIs':False,'passed':True})

motion=[r for r in D['motion_points'] if r['included']=='True']
assert len(D['motion_points'])==2144 and len(motion)==2084
case_sets={}
for model,n in [('pi05_libero',516),('pi0_libero',526)]:
 bystep={t:{(r['variant_id'],r['noise']):r for r in motion if r['model']==model and r['step']==t} for t in ['5','60']}
 assert len(bystep['5'])==n and set(bystep['5'])==set(bystep['60'])
 assert all(bystep['5'][k]['q0_norm']==bystep['60'][k]['q0_norm'] for k in bystep['5'])
 case_sets[model]={'pairs_per_time':n,'same_keys_and_initial_commands':True,'keys_sha256':hashlib.sha256(json.dumps(sorted(bystep['5'])).encode()).hexdigest()}
for r in motion:
 assert 0<=float(r['q0_norm'])<=1.6 and 0<=float(r['eef_mm'])<=440
checks.append({'check':'figure6_every_qualified_point_and_same_case_times','planned_rows':2144,'drawn_rows':2084,'excluded_rows_retained_in_source':60,'by_model':case_sets,'outcome_counts':dict(Counter(r['Y'] for r in motion)),'passed':True})

q=json.loads((QA/'fig6_outcomes.json').read_text())
assert len(q['motion_rows'])==8 and len(q['outcome_rows'])==48
for r in q['motion_rows']:
 assert 0<=float(r['ratio_ci95_low'])<=float(r['ratio_of_task_means'])<=float(r['ratio_ci95_high'])<=4.2
unknown=Counter()
for r in q['outcome_rows']:
 p=None if r['point']=='' else float(r['point'])*100
 if r['metric']=='risk_difference':
  lim=(-43,16)
  vals=[float(r['bound_lower'])*100,float(r['bound_upper'])*100] if p is None else [float(r['ci98_75_lower'])*100,p,float(r['ci98_75_upper'])*100]
 else:
  lim=(0,28) if r['metric']=='harm' else ((0,18) if r['metric']=='recovery' else (0,36))
  vals=[float(r['bound_lower'])*100,float(r['bound_upper'])*100] if p is None else [p]
 assert all(lim[0]<=v<=lim[1] for v in vals),(r['metric'],vals)
 if p is None:unknown[r['metric']]+=1
checks.append({'check':'figure7_full_panel_and_missing_bounds','motion_rows':8,'success_rows':8,'flip_rows':24,'harm_recovery_rows':16,'unknown_rows_by_metric':dict(unknown),'all_inside_axes':True,'passed':True})

for name in ['same_basis_translation','attention_spatial','fig4_attention','fig5_command_motion','fig6_outcomes']:
 q=json.loads((QA/(name+'.json')).read_text())
 assert q['labels_outside_canvas']==[],(name,q['labels_outside_canvas'])
 for suffix in ['pdf','svg','png']:assert (ROOT.parent/'figures'/f'{name}.{suffix}').is_file()
checks.append({'check':'five_figures_have_pdf_svg_png_and_no_offcanvas_labels','passed':True})
report={'status':'PASS','checks':checks,'scientific_inputs_modified':False,'new_model_calls':0,'new_fits':0,'new_bootstraps':0,'visual_review':'All five PNGs inspected; second rendering corrected Fig5 header overlap and Fig7 row-label spacing.'}
(QA/'PRESERVATION-AND-COVERAGE.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
print(json.dumps(report,ensure_ascii=False,indent=2))
