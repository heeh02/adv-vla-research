import pandas as pd,numpy as np
from scipy.stats import spearmanr,pearsonr
O=pd.read_pickle('fa_events.pkl'); O['success']=O.success.astype(float)
fam=lambda c: 'cam±30' if '30' in c else ('cam±15+材质' if 'surface' in c and 'camera' in c else ('cam±15' if 'camera' in c and 'q0' not in c else ('仅首帧' if 'q0' in c else ('clean' if c=='clean' else '光度'))))
O['fam']=O.cond.map(fam)
# use the object with the largest clean displacement per root as primary target
prim=O[O.cond=='clean'].groupby(['root','obj']).moved.median().reset_index().sort_values('moved').groupby('root').tail(1)[['root','obj']]
Q=O.merge(prim,on=['root','obj'])
# 1. placement offset in SUCCESSFUL episodes by family, relative to clean reference (median final pos over clean successes)
print('final placement xy error of primary object, SUCCESSFUL episodes (m):')
print(Q[Q.success==1].groupby('fam').final_err_xy.describe(percentiles=[.5,.75,.9])[['count','50%','75%','90%']].round(4).to_string())
# clean-vs-clean noise floor = clean successes' own spread (already relative to clean median)
# 2. systematic offset direction per root under cam±30: mean (dx,dy) across noises & signs
s=Q[(Q.success==1)]
sys=s.groupby(['root','fam'])[['dx','dy']].mean(); sys['sys_off']=np.hypot(sys.dx,sys.dy)
print('\nsystematic (mean) placement offset per root, median over roots:'); print(sys.groupby('fam').sys_off.median().round(4).to_string())
# 3. harm anatomy refined
C=O[O.cond=='clean'][['root','noise','obj','success']].rename(columns={'success':'c_success'})
Z=Q.merge(C,on=['root','noise','obj'])
Z['cell']=np.select([(Z.success==1)&(Z.c_success==1),(Z.success==0)&(Z.c_success==1)],['SS','harm'],'other')
tol=Q[Q.success==1].groupby('root').final_err_xy.quantile(.95).rename('tol')
Z=Z.join(tol,on='root')
def m(r):
    if r.picked_type and not r.ever_lift: return 'A 没拿起'
    if r.grip_end>0 and r.eef_obj_end<0.06: return 'B 仍抓着没放'
    if r.final_err_xy>r.tol: return 'C 放偏(超出成功范围)'
    return 'D 放在成功范围内仍失败'
H=Z[Z.cell=='harm'].copy(); H['mode']=H.apply(m,axis=1)
print('\nharm anatomy (primary object):'); print(H['mode'].value_counts().to_string())
print(pd.crosstab(H.fam,H['mode']).to_string())
cf=Q[(Q.cond=='clean')&(Q.success==0)].join(tol,on='root'); cf['mode']=cf.apply(m,axis=1); print('clean failures:',cf['mode'].value_counts().to_dict())
# 4. task-level law: offset/tolerance ratio vs vulnerability (D3 own and independent severity)
Q['task_id']=Q.root.str.rsplit('-',n=1).str[0]
tt=Q[Q.fam=='cam±30'].groupby('task_id').final_err_xy.median().rename('off30')
tclean=Q[(Q.fam=='clean')&(Q.success==1)].groupby('task_id').final_err_xy.quantile(.9).rename('spread_clean')
tolt=Q[Q.success==1].groupby('task_id').final_err_xy.quantile(.95).rename('tol_task')
T=pd.concat([tt,tclean,tolt],axis=1)
T['ratio']=T.off30/T.tol_task
f=pd.read_csv('/private/tmp/claude-501/-Users-geminihe-Adv-VLA--0-5--claude-worktrees-project-experiment-cvpr-analysis-b14247/c734d103-08cf-4d61-81a1-391708909fc0/scratchpad/ana/M6_task_features.csv',index_col=0)
T=T.join(f[['D3_30','Sev_30','Fresh_15']])
for x in ['off30','tol_task','ratio','spread_clean']:
    for y in ['D3_30','Sev_30']:
        j=T[[x,y]].dropna(); r=spearmanr(j[x],j[y]); p=pearsonr(j[x],j[y])
        print(f'task {x:12s} vs {y}: spearman {r.correlation:+.2f} (p={r.pvalue:.3f})  pearson {p[0]:+.2f} (p={p[1]:.3f}) n={len(j)}')
T.to_csv('fa_task_law.csv'); Z.to_pickle('fa_Z.pkl')
print(T.sort_values('Sev_30').head(10).round(3).to_string())
