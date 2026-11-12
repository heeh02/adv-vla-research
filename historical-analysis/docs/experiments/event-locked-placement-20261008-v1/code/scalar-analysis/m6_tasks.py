import pandas as pd, numpy as np, json
from scipy.stats import spearmanr, pearsonr
R='/Users/researcher/.codex/worktrees/adv-vla-research-20260924/Adv-VLA-π0.5/docs/experiments/libero-light-v1/overnight-20260929/native-camera-severity-confirm-v2/SEVERITY_FULL2160_COMPACT_RESULTS-v1.json'
d=json.load(open(R))['contrasts']
sev={k:pd.Series({t['task_id']:t['value'] for t in v['task_values']}) for k,v in d.items()}
ep=pd.read_pickle('ep.pkl'); anc=pd.read_pickle('anc.pkl')
ep['task_id']=ep.suite+'-t'+ep.task.astype(str).str.zfill(2)
ep['dY']=ep.succ-ep.clean_succ
def d3(conds): 
    return ep[ep.condition.isin(conds)].dropna(subset=['dY']).groupby('task_id').dY.mean()
D3_30=d3(['camera_minus30','camera_plus30']); D3_15=d3(['camera_minus15','camera_plus15']); D3_hl=d3(['headlight_off'])
D3_all=d3(['camera_minus30','camera_plus30','camera_minus15','camera_plus15','camera_minus15_surface_cool','camera_plus15_surface_warm'])
D3_clean=ep[ep.condition=='clean'].groupby('task_id').succ.mean()
S30=sev['supp_pooled_30_minus_own_clean']; F15=sev['fresh_pooled_15_minus_own_clean']; Fhl=sev['fresh_headlight_minus_own_clean']
def rep(a,b,name):
    j=pd.concat([a.rename('a'),b.rename('b')],axis=1).dropna()
    r=spearmanr(j.a,j.b); p=pearsonr(j.a,j.b)
    # top-decile overlap
    print(f'{name:45s} n={len(j)} spearman={r.correlation:+.2f} (p={r.pvalue:.3f}) pearson={p[0]:+.2f} (p={p[1]:.3f})')
    return j
print('== cross-cohort replication of task susceptibility (dSR per task) ==')
j=rep(D3_30,S30,'D3 cam±30  vs  Severity cam±30 (indep. states)')
rep(D3_15,F15,'D3 cam±15  vs  Fresh cam±15')
rep(D3_all,S30,'D3 all-camera  vs  Severity cam±30')
rep(D3_all,F15,'D3 all-camera  vs  Fresh cam±15')
rep(F15,S30,'Fresh cam±15  vs  Severity cam±30 (same states)')
rep(D3_hl,Fhl,'D3 headlight vs Fresh headlight')
print('\nworst tasks D3 cam±30:', D3_30.sort_values().head(8).round(2).to_dict())
print('worst tasks Sev cam±30:', S30.sort_values().head(8).round(2).to_dict())
# task features
a0=anc[(anc.anchor_slot==0)&anc.condition.isin(['camera_minus30','camera_plus30'])].copy()
a0['task_id']=a0.suite+'-t'+a0.task.astype(str).str.zfill(2)
feat=a0.groupby('task_id').agg(T=('physical_prefix_translation_rms','mean'),H=('hidden_free_relative_clean_rms_prefix','mean'),Tclean=('native_clean_update_group_rms_prefix_translation','mean'))
feat['Trel']=feat['T']/feat.Tclean
# episode length proxy: last reached anchor slot in clean SS episodes
sup=pd.read_csv('/Users/researcher/.codex/worktrees/adv-vla-research-20260924/Adv-VLA-π0.5/docs/experiments/libero-light-v1/integrated-analysis-20260928/atlas-v1/results-v1/support.csv',low_memory=False)
sup['task_id']=sup.suite+'-t'+sup.task.astype(str).str.zfill(2)
rs=sup[(sup.condition=='clean')&(sup.anchor_status=='reached_metadata_eligible')].groupby(['episode_id','task_id']).control_step.max().groupby('task_id').mean()
feat['len']=rs
# independent clean margin: Severity/Fresh clean? use D3 clean SR (different states) as margin proxy
feat['cleanSR_D3']=D3_clean
target=S30  # independent outcome
print('\n== predict Severity cam±30 task dSR from D3-derived task features ==')
for c in ['T','H','Trel','len','cleanSR_D3']:
    jj=pd.concat([feat[c],target.rename('y')],axis=1).dropna(); r=spearmanr(jj[c],jj.y)
    print(f'  {c:12s} spearman={r.correlation:+.2f} p={r.pvalue:.3f}')
jj=pd.concat([feat,D3_30.rename('y')],axis=1).dropna()
print('== same with D3 cam±30 dSR ==')
for c in ['T','H','Trel','len','cleanSR_D3']:
    r=spearmanr(jj[c],jj.y); print(f'  {c:12s} spearman={r.correlation:+.2f} p={r.pvalue:.3f}')
feat.join([D3_30.rename('D3_30'),S30.rename('Sev_30'),F15.rename('Fresh_15'),D3_15.rename('D3_15')]).to_csv('M6_task_features.csv')
