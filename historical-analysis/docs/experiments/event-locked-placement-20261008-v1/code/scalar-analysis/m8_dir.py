import pandas as pd, numpy as np
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr
el=pd.read_pickle('el.pkl'); el['task_id']=el.suite+'-t'+el.task.astype(str).str.zfill(2)
f=pd.read_csv('M6_task_features.csv',index_col=0)
CONT=['camera_minus30','camera_plus30','camera_minus15','camera_plus15','camera_minus15_surface_cool','camera_plus15_surface_warm','headlight_off','matched_global','surface_warm','surface_cool']
e=el[el.history_condition.isin(CONT)&el.outcome_cell_condition_clean.isin(['11','01'])].copy()
for c in ['full5120_cosine','stored_querymean_cosine','change_relative_rms','change_action_translation_rms','slot1_relative_rms','slot2_relative_rms']: e[c]=pd.to_numeric(e[c],errors='coerce')
e['harm']=(e.outcome_cell_condition_clean=='01').astype(int)
print(e.groupby(['layer','harm'])[['full5120_cosine','stored_querymean_cosine','change_relative_rms']].median().round(3).to_string())
rng=np.random.default_rng(5)
for L in [1,9,18]:
    b=e[e.layer==L].copy()
    ok=b.groupby(['task_id','history_condition']).harm.transform(lambda x:x.any() and (x==0).any()); b=b[ok]
    for v in ['full5120_cosine','stored_querymean_cosine','change_relative_rms']:
        b['r']=b.groupby(['task_id','history_condition'])[v].rank(pct=True); m=b.r.notna()
        auc=roc_auc_score(b.harm[m],b.r[m])
        tasks=b.task_id.unique(); bs=[]
        for _ in range(500):
            tt=rng.choice(tasks,len(tasks)); bb=pd.concat([b[b.task_id==t] for t in tt]); mm=bb.r.notna()
            bs.append(roc_auc_score(bb.harm[mm],bb.r[mm]))
        print(f'L{L} {v:25s} stratified AUC={auc:.3f} [{np.percentile(bs,2.5):.2f},{np.percentile(bs,97.5):.2f}] n_harm={b.harm.sum()}')
    tl=e[(e.layer==L)&e.history_condition.isin(['camera_minus30','camera_plus30'])].groupby('task_id')[['full5120_cosine','change_relative_rms']].mean().join(f[['Sev_30']])
    for v in ['full5120_cosine','change_relative_rms']:
        r=spearmanr(tl[v],tl.Sev_30,nan_policy='omit'); print(f'   task-level L{L} {v} vs Severity±30: rho={r.correlation:+.2f} p={r.pvalue:.3f}')
