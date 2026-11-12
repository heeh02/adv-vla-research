import pandas as pd,numpy as np
from scipy.stats import spearmanr,pearsonr
O=pd.read_pickle('fa_events.pkl'); O['success']=O.success.astype(float)
prim=O[O.cond=='clean'].groupby(['root','obj']).moved.median().reset_index().sort_values('moved').groupby('root').tail(1)[['root','obj']]
Q=O.merge(prim,on=['root','obj']); Q['task_id']=Q.root.str.rsplit('-',n=1).str[0]
f=pd.read_csv('/private/tmp/claude-501/-Users-geminihe-Adv-VLA--0-5--claude-worktrees-project-experiment-cvpr-analysis-b14247/c734d103-08cf-4d61-81a1-391708909fc0/scratchpad/ana/M6_task_features.csv',index_col=0)
S=Q[Q.success==1]
# (a) non-circular: systematic offset from SUCCESSFUL cam±30 episodes only
def sysoff(g): return np.hypot(g.dx.mean(),g.dy.mean())
rows=[]
for t,g in S.groupby('task_id'):
    r={'task_id':t}
    for c in ['camera_minus30','camera_plus30']:
        h=g[g.cond==c]; r[c]=np.mean([sysoff(x) for _,x in h.groupby('root')]) if len(h) else np.nan
    cl=g[g.cond=='clean']; r['clean_spread']=cl.final_err_xy.median() if len(cl) else np.nan
    r['succ_err30']=g[g.cond.isin(['camera_minus30','camera_plus30'])].final_err_xy.median()
    rows.append(r)
T=pd.DataFrame(rows).set_index('task_id'); T['sys30']=T[['camera_minus30','camera_plus30']].mean(1)
T=T.join(f[['D3_30','Sev_30','Fresh_15','D3_15','H','T']])
print('NON-CIRCULAR: offsets measured only on SUCCESSFUL episodes')
for x in ['sys30','succ_err30','clean_spread','H','T']:
    for y in ['Sev_30','D3_30','Fresh_15']:
        j=T[[x,y]].dropna(); r=spearmanr(j[x],j[y]); p=pearsonr(j[x],j[y])
        print(f'  {x:12s} vs {y:8s}: spearman {r.correlation:+.2f} (p={r.pvalue:.3f}) pearson {p[0]:+.2f} (p={p[1]:.3f}) n={len(j)}')
# (b) direction: +30 vs -30 mean offset vectors per root
cs=[]
for r,g in S.groupby('root'):
    a=g[g.cond=='camera_plus30'][['dx','dy']].mean().values; b=g[g.cond=='camera_minus30'][['dx','dy']].mean().values
    if np.all(np.isfinite(a)) and np.all(np.isfinite(b)) and np.linalg.norm(a)>0.003 and np.linalg.norm(b)>0.003: cs.append(a@b/np.linalg.norm(a)/np.linalg.norm(b))
print(f'\n(b) cos(+30 offset, -30 offset) per root: median {np.median(cs):+.2f}, frac<0: {np.mean(np.array(cs)<0):.2f}, n={len(cs)}')
# (c) condition level: systematic offset (successful eps) vs condition dSR (from D3 paired outcomes on the same 36 tasks)
ep=pd.read_pickle('/private/tmp/claude-501/-Users-geminihe-Adv-VLA--0-5--claude-worktrees-project-experiment-cvpr-analysis-b14247/c734d103-08cf-4d61-81a1-391708909fc0/scratchpad/ana/ep.pkl')
ep['dY']=ep.succ-ep.clean_succ
cc=[]
for c,g in S[S.cond!='clean'].groupby('cond'):
    so=np.mean([sysoff(x) for _,x in g.groupby('root')]); me=g.final_err_xy.median()
    d=ep[ep.condition==c].dY.mean()*100
    cc.append(dict(cond=c,sys_off=so,succ_err=me,dSR=d))
CC=pd.DataFrame(cc).sort_values('sys_off'); print('\n(c) condition level'); print(CC.round(4).to_string())
print('  spearman sys_off vs dSR: %+.2f, pearson %+.2f'%(spearmanr(CC.sys_off,CC.dSR).correlation,pearsonr(CC.sys_off,CC.dSR)[0]))
# (d) near-miss: harm final error relative to task tolerance
Z=pd.read_pickle('fa_Z.pkl'); H=Z[Z.cell=='harm']
rr=(H.final_err_xy/H.tol).dropna(); print('\n(d) harm final error / success tolerance: quantiles', rr.quantile([.1,.25,.5,.75,.9]).round(2).to_dict())
T.to_csv('fa_task_law2.csv'); CC.to_csv('fa_cond_law.csv')
