import pandas as pd, numpy as np
from scipy.optimize import minimize_scalar
anc=pd.read_pickle('anc.pkl'); ep=pd.read_pickle('ep.pkl')
CONT=['camera_minus30','camera_plus30','camera_minus15','camera_plus15','camera_minus15_surface_cool','camera_plus15_surface_warm','headlight_off','matched_global','surface_warm','surface_cool']
a=anc[(anc.condition.isin(CONT))&(anc.anchor_slot.isin([0,1]))].copy()
a['task_id']=a.suite+'-t'+a.task.astype(str)
ep['task_id']=ep.suite+'-t'+ep.task.astype(str)
ep['dY']=ep.succ-ep.clean_succ
X={'T0':('physical_prefix_translation_rms',0),'H0':('hidden_free_relative_clean_rms_prefix',0),'V0':('visual_relative_clean_rms_main',0),'Att0':('attention_free_source_mass_delta_prefix_main',0),'T1':('physical_prefix_translation_rms',1),'H1':('hidden_free_relative_clean_rms_prefix',1)}
def cond_table(tasks=None):
    e=ep[ep.condition.isin(CONT)].dropna(subset=['dY'])
    aa=a
    if tasks is not None:
        e=pd.concat([e[e.task_id==t] for t in tasks]); aa=pd.concat([a[a.task_id==t] for t in tasks])
    tab=e.groupby('condition').dY.mean().rename('dSR').to_frame()
    for k,(v,s) in X.items(): tab[k]=aa[aa.anchor_slot==s].groupby('condition')[v].mean()
    return tab
T=cond_table(); print((T.assign(dSR=T.dSR*100)).round(4).sort_values('dSR').to_string())
def fit_lin(x,y): b=np.polyfit(x,y,1); return lambda z: np.polyval(b,z)
def fit_hinge(x,y):
    best=None
    for t0 in np.linspace(np.min(x),np.max(x),60):
        z=np.maximum(0,x-t0); A=np.c_[np.ones_like(z),z]; coef,res,*_=np.linalg.lstsq(A,y,rcond=None)
        sse=((A@coef-y)**2).sum()
        if best is None or sse<best[0]: best=(sse,t0,coef)
    _,t0,coef=best; return (lambda z: coef[0]+coef[1]*np.maximum(0,z-t0)), t0, coef
def loco(x,y,kind):
    err=[]
    for i in range(len(x)):
        m=np.ones(len(x),bool); m[i]=False
        f=fit_lin(x[m],y[m]) if kind=='lin' else (fit_hinge(x[m],y[m])[0] if kind=='hinge' else (lambda z,yy=y[m]: np.full_like(z,yy.mean(),dtype=float)))
        err.append((f(np.array([x[i]]))[0]-y[i])**2)
    return np.sqrt(np.mean(err))*100
y=T.dSR.values
print('\nLOCO RMSE (pp) of predicting condition dSR:  [mean-only baseline=%.2f]'%loco(T.T0.values,y,'mean'))
for k in X:
    x=T[k].values
    f,t0,coef=fit_hinge(x,y)
    print(f'  {k:5s} linear={loco(x,y,"lin"):.2f}  hinge={loco(x,y,"hinge"):.2f}  (hinge knot={t0:.4f}, slope={coef[1]*100:.1f}pp/unit)')
# family indicator model: camera30 vs rest
fam=np.array([1 if '30' in c else 0 for c in T.index]); 
err=[]
for i in range(len(y)):
    m=np.ones(len(y),bool); m[i]=False
    pred=y[m][fam[m]==fam[i]].mean(); err.append((pred-y[i])**2)
print('  family(±30 vs rest) LOCO RMSE=%.2f'%(np.sqrt(np.mean(err))*100))
# bootstrap over tasks: correlation of T0 and dSR across conditions
rng=np.random.default_rng(3); tasks=a.task_id.unique(); rs={k:[] for k in X}
for _ in range(500):
    tt=rng.choice(tasks,len(tasks)); tb=cond_table(tt)
    for k in X: rs[k].append(np.corrcoef(tb[k],tb.dSR)[0,1])
print('\ntask-bootstrap Pearson r(condition mean x, dSR) 95% CI:')
for k in X: print(f'  {k}: {np.mean(rs[k]):+.2f} [{np.percentile(rs[k],2.5):+.2f},{np.percentile(rs[k],97.5):+.2f}]')
T.to_csv('M4_condition_table.csv')
