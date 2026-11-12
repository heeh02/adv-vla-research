import pandas as pd,numpy as np,pickle,glob
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr
A='/private/tmp/claude-501/-Users-geminihe-Adv-VLA--0-5--claude-worktrees-project-experiment-cvpr-analysis-b14247/c734d103-08cf-4d61-81a1-391708909fc0/scratchpad/ana/'
anc=pd.read_pickle(A+'anc.pkl')
sup=pd.read_csv('/Users/researcher/.codex/worktrees/adv-vla-research-20260924/Adv-VLA-π0.5/docs/experiments/libero-light-v1/integrated-analysis-20260928/atlas-v1/results-v1/support.csv',low_memory=False)
anc=anc.merge(sup[['anchor_id','control_step']],on='anchor_id',how='left')
# release step per episode from executed actions (last close->open), plus first close
rel={}
for f in sorted(glob.glob('traj/shard-*.pkl')):
    o=pickle.load(open(f,'rb'))
    for ep,a in o['exec'].items():
        g=np.sign(a[:,6]); g[g==0]=-1
        opens=np.where((g[1:]<0)&(g[:-1]>0))[0]+1; closes=np.where((g[1:]>0)&(g[:-1]<0))[0]+1
        rel[(ep[0],ep[1],ep[2])]=dict(release=opens[-1] if len(opens) else np.nan, grasp=closes[0] if len(closes) else np.nan, steps=len(a))
R=pd.DataFrame([dict(root=k[0],noise_repeat=k[1],condition=k[2],**v) for k,v in rel.items()])
X=anc.merge(R,on=['root','noise_repeat','condition'])
CONT=['camera_minus30','camera_plus30','camera_minus15','camera_plus15','camera_minus15_surface_cool','camera_plus15_surface_warm','headlight_off','matched_global','surface_warm','surface_cool']
X=X[X.condition.isin(CONT)].dropna(subset=['succ','clean_succ'])
X['cell']=np.select([(X.succ==1)&(X.clean_succ==1),(X.succ==0)&(X.clean_succ==1)],['SS','harm'],'other')
X['task_id']=X.suite+'-t'+X.task.astype(str)
# event-locked anchor: latest anchor with control_step <= grasp (pre-grasp) and <= release (pre-place) ; plus q0
V={'H18':'hidden_free_relative_clean_rms_prefix','Trans':'physical_prefix_translation_rms','JS':'attention_free_js_raw_prefix','AttMain':'attention_free_source_mass_delta_prefix_main'}
def pick(g,event):
    t=g[event].iloc[0]
    if np.isnan(t): return None
    h=g[g.control_step<=t]
    return h.sort_values('control_step').iloc[-1] if len(h) else None
rows=[]
for ep,g in X.groupby('episode_id'):
    base=dict(episode_id=ep,cell=g.cell.iloc[0],task_id=g.task_id.iloc[0],condition=g.condition.iloc[0])
    for ev in ['grasp','release']:
        r=pick(g,ev)
        if r is not None:
            for k,v in V.items(): base[f'{ev}_{k}']=r[v]
            base[f'{ev}_gap']=g[ev].iloc[0]-r.control_step
    q0=g[g.anchor_slot==0]
    for k,v in V.items(): base[f'q0_{k}']=q0[v].iloc[0] if len(q0) else np.nan
    rows.append(base)
E=pd.DataFrame(rows); E=E[E.cell.isin(['SS','harm'])]; E['y']=(E.cell=='harm').astype(int)
print('events located: grasp',E.grasp_H18.notna().sum(),'release',E.release_H18.notna().sum(),'of',len(E),'; median gap to release anchor', E.release_gap.median())
ok=E.groupby(['task_id','condition']).y.transform(lambda s:s.any() and (s==0).any()); S=E[ok].copy()
rng=np.random.default_rng(3)
for ev in ['q0','grasp','release']:
    for k in V:
        v=f'{ev}_{k}'; S['r']=S.groupby(['task_id','condition'])[v].rank(pct=True); m=S.r.notna()
        if S.y[m].nunique()<2: continue
        auc=roc_auc_score(S.y[m],S.r[m]); bs=[]; tasks=S.task_id.unique()
        for _ in range(400):
            tt=rng.choice(tasks,len(tasks)); bb=pd.concat([S[S.task_id==t] for t in tt]); mm=bb.r.notna()
            if bb.y[mm].nunique()>1: bs.append(roc_auc_score(bb.y[mm],bb.r[mm]))
        print(f'{ev:8s} {k:8s} stratified AUC={auc:.3f} [{np.percentile(bs,2.5):.2f},{np.percentile(bs,97.5):.2f}] n_harm={S.y[m].sum()}')
E.to_pickle('fa_eventlock.pkl')
