import pandas as pd, numpy as np
from sklearn.metrics import roc_auc_score
x=pd.read_pickle('decomp_full.pkl')
def cos(a,b):
    if a is None or b is None: return np.nan
    na,nb=np.linalg.norm(a),np.linalg.norm(b); return np.nan if na==0 or nb==0 else float(a@b/na/nb)
rows=[]
for (root,n,c,br),g in x.groupby(['root','noise_repeat','condition','branch_role']):
    g=g.set_index('anchor_slot')
    if not {0,1,2}<=set(g.index): continue
    cv={s:g.loc[s,'current_v'] for s in [0,1,2]}; sv={s:g.loc[s,'state_v'] for s in [0,1,2]}
    rows.append(dict(root=root,noise=n,condition=c,branch=br,cell=g.loc[0,'cell'],suite=g.loc[0,'suite'],task=g.loc[0,'task'],
      cur0=g.loc[0,'current_rms'],cur1=g.loc[1,'current_rms'],cur2=g.loc[2,'current_rms'],st1=g.loc[1,'state_rms'],st2=g.loc[2,'state_rms'],
      pers01=cos(cv[0],cv[1]),pers12=cos(cv[1],cv[2]),pers02=cos(cv[0],cv[2]),
      align_state2_cur=cos(sv[2],cv[0]+cv[1]) if cv[1] is not None else np.nan,
      align_state2_cur0=cos(sv[2],cv[0]),
      bias_mean=np.linalg.norm((cv[0]+cv[1]+cv[2])/3) if all(v is not None for v in cv.values()) else np.nan))
E=pd.DataFrame(rows); E['task_id']=E.suite+'-t'+E.task.astype(str)
E['bias_ratio']=E.bias_mean/((E.cur0**2+E.cur1**2+E.cur2**2)/3)**.5
E.to_pickle('bias_feats.pkl')
F=['cur0','cur1','cur2','st1','st2','pers01','pers12','pers02','align_state2_cur','align_state2_cur0','bias_mean','bias_ratio']
print('medians by branch x cell'); print(E[E.cell.isin(['SS','harm','recov'])].groupby(['branch','cell'])[F].median().round(3).T.to_string())
c=E[(E.branch=='continuous')&E.cell.isin(['SS','harm'])].copy()
c['harm']=(c.cell=='harm').astype(int)
print('\ncontinuous: n harm',c.harm.sum(),'SS',(c.harm==0).sum())
ok=c.groupby(['task_id','condition']).harm.transform(lambda s:s.any() and (s==0).any()); s=c[ok].copy()
print('strata',s.groupby(['task_id','condition']).ngroups,'harms',s.harm.sum())
for v in F:
    s['rk']=s.groupby(['task_id','condition'])[v].rank(pct=True); m=s.rk.notna()
    a1=roc_auc_score(s.harm[m],s.rk[m]) if s.harm[m].nunique()>1 else np.nan
    m2=c[v].notna(); a2=roc_auc_score(c.harm[m2],c[v][m2])
    print(f'  {v:20s} stratified AUC={a1:.3f}   pooled AUC={a2:.3f}')
