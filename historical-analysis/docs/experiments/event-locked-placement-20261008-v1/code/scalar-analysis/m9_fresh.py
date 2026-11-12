import pandas as pd, numpy as np
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr
D='/Users/researcher/.codex/worktrees/adv-vla-research-20260924/Adv-VLA-π0.5/docs/experiments/analysis-system-20260929/fresh-within-run-analysis-20261006-v1'
b=pd.read_csv(D+'/readout-subspace-v1/population-readout-v1/BEHAVIOR_LINK-v1.csv')
print(b.outcome_pair.value_counts().to_dict(), b.neural_support.value_counts().to_dict())
lu=pd.read_csv(D+'/native-path-readout-v1/execution-v1/run-v1/small-readback/analysis/last_update-v1.csv',low_memory=False)
lu=lu[(lu.action_scope=='first5')&(lu.coordinate_domain=='adapter_retained_0_7')]
p=lu.pivot_table(index='episode_id',columns='quantity',values='delta_raw_rms')
p['s']=p['delta_E9_recorded_endpoint_update']/p['delta_x9']; p['r']=p['delta_final_recorded_raw_actions']/p['delta_x9']
ang=pd.read_csv(D+'/native-path-readout-v1/execution-v1/readback-recovery-v1/files/analysis/last_update_angles-v1.csv',low_memory=False)
print('angles cols',list(ang.columns)[:30])
b=b.merge(p[['s','r','delta_x9','delta_final_recorded_raw_actions']],left_on='episode_id',right_index=True,how='left')
b['task_id']=b.suite+'-t'+b.task.astype(str)
b['harm']=((b.clean_success==1)&(b.condition_success==0)).astype(int)
e=b[(b.clean_success==1)].copy()
V=['R_response_RMS','native_flow7_difference_RMS','visible_fraction_W7','s','r','delta_x9','delta_final_recorded_raw_actions']
print('median by harm:'); print(e.groupby('harm')[V].median().round(4).to_string())
rng=np.random.default_rng(7)
ok=e.groupby(['task_id','condition']).harm.transform(lambda x:x.any() and (x==0).any()); s=e[ok].copy()
print('strata',s.groupby(['task_id','condition']).ngroups,'harm',s.harm.sum())
for v in V:
    s['rk']=s.groupby(['task_id','condition'])[v].rank(pct=True); m=s.rk.notna()
    auc=roc_auc_score(s.harm[m],s.rk[m]); bs=[]
    tasks=s.task_id.unique()
    for _ in range(500):
        tt=rng.choice(tasks,len(tasks)); bb=pd.concat([s[s.task_id==t] for t in tt]); mm=bb.rk.notna()
        if bb.harm[mm].nunique()>1: bs.append(roc_auc_score(bb.harm[mm],bb.rk[mm]))
    # unstratified within-condition
    au2=[];w=[]
    for c,g in e.groupby('condition'):
        g=g.dropna(subset=[v])
        if g.harm.sum()>=5: au2.append(roc_auc_score(g.harm,g[v])); w.append(g.harm.sum())
    print(f'{v:35s} stratified AUC={auc:.3f} [{np.percentile(bs,2.5):.2f},{np.percentile(bs,97.5):.2f}]  within-cond AUC={np.average(au2,weights=w):.3f}')
# condition level
c=b.groupby('condition').agg(dSR=('condition_success','mean'),clean=('clean_success','mean'),**{v:(v,'mean') for v in ['R_response_RMS','native_flow7_difference_RMS','visible_fraction_W7','s','r']})
c['dSR']=(c.dSR-c.clean)*100; print(c.round(4).to_string())
# task-level vs D3/Severity
f=pd.read_csv('M6_task_features.csv',index_col=0); f.index=[i.replace('-t0','-t') if i.split('-t')[1].startswith('0') else i for i in f.index]
tl=b[b.condition.isin(['camera_minus15','camera_plus15'])].groupby('task_id')[['R_response_RMS','native_flow7_difference_RMS','visible_fraction_W7','s']].mean()
tl=tl.join(f[['Sev_30','Fresh_15','D3_30']])
for v in ['R_response_RMS','native_flow7_difference_RMS','visible_fraction_W7','s']:
    for y in ['Sev_30','Fresh_15','D3_30']:
        r=spearmanr(tl[v],tl[y],nan_policy='omit'); print(f'  task {v:28s} vs {y}: rho={r.correlation:+.2f} p={r.pvalue:.3f}')
