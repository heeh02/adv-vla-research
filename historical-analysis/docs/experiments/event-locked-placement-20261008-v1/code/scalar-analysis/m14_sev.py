import pandas as pd, numpy as np, itertools
from scipy.stats import spearmanr, mannwhitneyu
from sklearn.metrics import roc_auc_score
d=pd.read_csv('/Users/researcher/.codex/worktrees/adv-vla-research-20260924/Adv-VLA-π0.5/docs/experiments/analysis-system-20260929/control-transfer-full-v2/SCALAR-v2.csv')
d['task_id']=d.suite+'-t'+d.task.astype(str).str.zfill(2)
def vec(r,pref,ax): return np.array([r[f'{pref}_delta_signed_mean_axis_{a}_native'] for a in ax])
T=[0,1,2]
for w in ['first5','first20']:
    d[w+'_t']=d.apply(lambda r: vec(r,w,T),axis=1)
d['eef4']=d.apply(lambda r: np.array([r.q004_eef_signed_x_native,r.q004_eef_signed_y_native,r.q004_eef_signed_z_native]),axis=1)
d['eef1']=d.apply(lambda r: np.array([r.q001_eef_signed_x_native,r.q001_eef_signed_y_native,r.q001_eef_signed_z_native]),axis=1)
def cos(a,b):
    na,nb=np.linalg.norm(a),np.linalg.norm(b); return np.nan if na==0 or nb==0 else a@b/na/nb
# 1. consistency across noise (same root & sign)
for col in ['first5_t','first20_t','eef1','eef4']:
    cs=[];
    for (root,s),g in d.groupby(['root','sign']):
        v=list(g[col]); cs+= [cos(a,b) for a,b in itertools.combinations(v,2)]
    anti=[]
    for (root,n),g in d.groupby(['root','noise_repeat']):
        if len(g)==2: anti.append(cos(g[col].iloc[0],g[col].iloc[1]))
    # null: different roots same sign
    rng=np.random.default_rng(0); nl=[]
    v=list(d[col])
    for _ in range(3000):
        i,j=rng.choice(len(v),2,replace=False)
        if d.root.iloc[i]!=d.root.iloc[j]: nl.append(cos(v[i],v[j]))
    print(f'{col:10s} cos across noises (same root,sign) median={np.nanmedian(cs):+.3f} mean={np.nanmean(cs):+.3f} | cos(-30,+30) same root/noise median={np.nanmedian(anti):+.3f} | random-pair null median={np.nanmedian(nl):+.3f}')
# 2. systematic vs total per root x sign
rows=[]
for (root,s),g in d.groupby(['root','sign']):
    for col in ['first20_t','eef4']:
        V=np.stack(g[col]); m=V.mean(0); tot=(V**2).sum(1).mean(); sys=(m**2).sum()
        rows.append(dict(root=root,sign=s,col=col,task_id=g.task_id.iloc[0],sys_norm=np.sqrt(sys),tot_rms=np.sqrt(tot),sys_frac=sys/tot,
           harm=((g.clean_success==1)&(g.angle_success==0)).sum(),recov=((g.clean_success==0)&(g.angle_success==1)).sum(),n=len(g),
           dSR=(g.angle_success-g.clean_success).mean()))
S=pd.DataFrame(rows)
for col in ['first20_t','eef4']:
    s=S[S.col==col].copy(); s['any_harm']=(s.harm>0).astype(int)
    print(f'\n[{col}] root×sign units={len(s)}, with harm={s.any_harm.sum()}')
    print(s.groupby('any_harm')[['sys_norm','tot_rms','sys_frac']].median().round(4).to_string())
    for v in ['sys_norm','tot_rms','sys_frac']:
        print(f'   AUC(any harm) {v}: {roc_auc_score(s.any_harm,s[v]):.3f}   spearman with dSR: {spearmanr(s[v],s.dSR).correlation:+.3f}')
    # within-task: units in tasks that have both harm and no-harm units
    ok=s.groupby('task_id').any_harm.transform(lambda x:x.any() and (x==0).any()); q=s[ok].copy()
    for v in ['sys_norm','tot_rms','sys_frac']:
        q['rk']=q.groupby('task_id')[v].rank(pct=True); print(f'   within-task AUC {v}: {roc_auc_score(q.any_harm,q.rk):.3f} (tasks={q.task_id.nunique()}, harm units={q.any_harm.sum()})')
S.to_pickle('sev_sys.pkl'); d.to_pickle('sev480.pkl')
