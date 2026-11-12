import pandas as pd,numpy as np
from scipy.stats import wilcoxon
O=pd.read_pickle('fa_events.pkl'); O['success']=O.success.astype(float)
prim=O[O.cond=='clean'].groupby(['root','obj']).moved.median().reset_index().sort_values('moved').groupby('root').tail(1)[['root','obj']]
Q=O.merge(prim,on=['root','obj'])
ep=pd.read_pickle('/private/tmp/claude-501/-Users-geminihe-Adv-VLA--0-5--claude-worktrees-project-experiment-cvpr-analysis-b14247/c734d103-08cf-4d61-81a1-391708909fc0/scratchpad/ana/ep.pkl')
roots=Q.root.unique()
rng=np.random.default_rng(0)
def test_err(rs,cond):
    a=[];b=[]
    for r in rs:
        g=Q[(Q.root==r)]
        x=g[(g.cond==cond)].final_err_xy; y=g[(g.cond=='clean')].final_err_xy
        if len(x) and len(y): a.append(x.mean()); b.append(y.mean())
    return wilcoxon(np.array(a)-np.array(b),alternative='greater').pvalue if len(a)>5 else 1
def test_sr(rs,cond):
    e=ep[ep.root.isin(rs)&(ep.condition==cond)].dropna(subset=['succ','clean_succ'])
    h=((e.succ==0)&(e.clean_succ==1)).sum(); rc=((e.succ==1)&(e.clean_succ==0)).sum()
    from scipy.stats import binomtest
    return binomtest(int(h),int(h+rc),0.5,alternative='greater').pvalue if h+rc>0 else 1
for cond in ['camera_minus30','camera_minus15','headlight_off']:
    out=[]
    for n in [6,10,16,24,32,46]:
        pe=[];ps=[]
        for _ in range(200):
            rs=rng.choice(roots,min(n,len(roots)),replace=False); pe.append(test_err(rs,cond)<0.05); ps.append(test_sr(rs,cond)<0.05)
        out.append((n,np.mean(pe),np.mean(ps)))
    print(cond,' roots(×3 noise): power[placement-error, SR] =',[(n,round(a,2),round(b,2)) for n,a,b in out])
# LOCO at condition level: sys_off vs H0
C=pd.read_csv('fa_cond_law.csv'); C=C[~C.cond.str.contains('q0')]
M=pd.read_csv('/private/tmp/claude-501/-Users-geminihe-Adv-VLA--0-5--claude-worktrees-project-experiment-cvpr-analysis-b14247/c734d103-08cf-4d61-81a1-391708909fc0/scratchpad/ana/M4_condition_table.csv',index_col=0)
C=C.join(M[['H0','V0','T0']],on='cond')
def loco(x,y):
    e=[]
    for i in range(len(x)):
        m=np.ones(len(x),bool);m[i]=False; b=np.polyfit(x[m],y[m],1); e.append((np.polyval(b,x[i])-y[i])**2)
    return np.sqrt(np.mean(e))
y=C.dSR.values
for k in ['sys_off','succ_err','H0','V0','T0']: print(f'LOCO RMSE (10 continuous conds) using {k}: {loco(C[k].values,y):.2f} pp; r={np.corrcoef(C[k],y)[0,1]:+.2f}')
print('mean-only:',np.sqrt(np.mean([(y[np.arange(len(y))!=i].mean()-y[i])**2 for i in range(len(y))])).round(2))
