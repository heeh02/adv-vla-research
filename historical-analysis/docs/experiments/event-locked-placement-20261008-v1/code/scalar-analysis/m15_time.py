import pandas as pd, numpy as np
from scipy.stats import spearmanr, wilcoxon
d=pd.read_pickle('sev480.pkl')
LIM={'libero_spatial':220,'libero_object':280,'libero_goal':300,'libero_10':520}
print('steps at failure (max per suite):'); print(d[d.angle_success==0].groupby('suite').angle_steps.describe()[['count','min','50%','max']].to_string())
print(d[d.clean_success==0].groupby('suite').clean_steps.describe()[['count','min','50%','max']].to_string())
ss=d[(d.clean_success==1)&(d.angle_success==1)].copy()
ss['ratio']=ss.angle_steps/ss.clean_steps
print('\nSS pairs: angle/clean step ratio median %.3f mean %.3f, frac slower %.2f, wilcoxon p=%.2g'%(ss.ratio.median(),ss.ratio.mean(),(ss.ratio>1).mean(),wilcoxon(ss.angle_steps-ss.clean_steps).pvalue))
print(ss.groupby('suite').ratio.describe()[['count','25%','50%','75%']].round(3).to_string())
# mean world-direction of early deviation
V=np.stack(d.first5_t); print('\nmean first5 translation delta', V.mean(0).round(3), ' frac of pairs with cos>0.5 to mean:', np.mean([ (v@V.mean(0))/np.linalg.norm(v)/np.linalg.norm(V.mean(0))>0.5 for v in V]).round(2))
# task-level: clean duration margin vs vulnerability
d['lim']=d.suite.map(LIM)
cl=d[d.clean_success==1].groupby('task_id').agg(clean_steps=('clean_steps','median'),lim=('lim','first'))
cl['margin']=1-cl.clean_steps/cl.lim
sl=ss.groupby('task_id').ratio.median().rename('slowdown')
dsr=d.groupby('task_id').apply(lambda g:(g.angle_success-g.clean_success).mean()).rename('dSR')
T=cl.join(sl).join(dsr)
T['pred_timeout']=(T.clean_steps*T.slowdown>T.lim).astype(int)
f=pd.read_csv('M6_task_features.csv',index_col=0)
T=T.join(f[['D3_30','D3_15','Fresh_15']])
for x in ['margin','clean_steps','slowdown']:
    for y in ['dSR','D3_30']:
        r=spearmanr(T[x],T[y],nan_policy='omit'); print(f'task {x:12s} vs {y:6s}: rho={r.correlation:+.2f} p={r.pvalue:.3f}')
print(T.sort_values('dSR').head(10).round(3).to_string())
T.to_csv('M15_task_time.csv')
