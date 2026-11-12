import pandas as pd,numpy as np
from sklearn.metrics import roc_auc_score
O=pd.read_pickle('fa_events.pkl'); O['success']=O.success.astype(float)
# tolerance per root/object: 95th pct of final xy error among successful episodes (any condition)
tol=O[O.success==1].groupby(['root','obj']).final_err_xy.quantile(.95).rename('tol')
O=O.join(tol,on=['root','obj'])
# episode-level summary over manipulated objects
def ep_mode(g):
    pk=g[g.picked_type]
    never=bool(len(pk)) and (~pk.ever_lift).any()
    mis=((g.final_err_xy>g.tol*1.0)&(g.final_err_xy>0.03)).any()
    return pd.Series(dict(success=g.success.iloc[0],never_pick=never,misplaced=mis,att=pk.attempts_to_lift.max() if len(pk) else np.nan,
        max_err=g.final_err_xy.max(),n_close=g.n_close.iloc[0],task_name=g.task_name.iloc[0],task_id=g.task_id.iloc[0]))
E=O.groupby(['root','noise','cond']).apply(ep_mode).reset_index()
C=E[E.cond=='clean'].set_index(['root','noise'])[['success','never_pick','misplaced','att','max_err']].add_prefix('c_')
P=E[E.cond!='clean'].join(C,on=['root','noise']).dropna(subset=['success','c_success'])
P['cell']=np.select([(P.success==1)&(P.c_success==1),(P.success==0)&(P.c_success==1),(P.success==1)&(P.c_success==0)],['SS','harm','recov'],'FF')
def mode(r):
    if r.never_pick: return '没拿起物体'
    if r.misplaced: return '拿起了但放错/掉落'
    return '物体到位但任务未完成'
for cellname in ['harm','FF']:
    s=P[P.cell==cellname]; print(f'\n{cellname} (n={len(s)}) modes:'); print(s.apply(mode,axis=1).value_counts().to_string())
cf=E[(E.cond=='clean')&(E.success==0)]; print(f'\nclean failures (n={len(cf)}) modes:'); print(cf.apply(mode,axis=1).value_counts().to_string())
fam=lambda c: 'cam±30' if '30' in c else ('cam±15+材质' if 'surface' in c and 'camera' in c else ('cam±15' if 'camera' in c and 'q0' not in c else ('仅首帧' if 'q0' in c else '光度')))
P['fam']=P.cond.map(fam)
print('\nharm modes by family:'); print(pd.crosstab(P[P.cell=='harm'].fam,P[P.cell=='harm'].apply(mode,axis=1)).to_string())
# grasp reliability across conditions in all episodes (attempts to lift for picked objects)
print('\nattempts-to-first-lift (picked objects), by family (all episodes):')
print(P.groupby('fam').att.describe()[['count','mean','50%']].round(2).to_string())
print('clean:', E[E.cond=='clean'].att.describe()[['count','mean','50%']].round(2).to_dict())
print('\nP(never pick) and P(misplaced) by family:'); print(P.groupby('fam')[['never_pick','misplaced']].mean().round(3).to_string()); print('clean',E[E.cond=='clean'][['never_pick','misplaced']].mean().round(3).to_dict())
X=P[P.cell.isin(['SS','harm'])].copy(); X['y']=(X.cell=='harm').astype(int)
ok=X.groupby(['task_id','cond']).y.transform(lambda s:s.any() and (s==0).any()); X=X[ok]
for v in ['att','max_err','n_close']:
    X['r']=X.groupby(['task_id','cond'])[v].rank(pct=True); m=X.r.notna()
    print(f'stratified AUC {v}: {roc_auc_score(X.y[m],X.r[m]):.3f} (harm {X.y[m].sum()})')
P.to_pickle('fa_ep_pairs.pkl'); E.to_pickle('fa_ep.pkl')
print('\nper-task harm modes (cam±30):')
h=P[(P.cell=='harm')]; t=h.assign(m=h.apply(mode,axis=1)).groupby(['task_id','task_name','m']).size().unstack(fill_value=0)
t['total']=t.sum(1); print(t.sort_values('total',ascending=False).head(14).to_string())
