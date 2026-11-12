import pandas as pd, numpy as np, itertools
from scipy.stats import binomtest
import statsmodels.formula.api as smf
ep=pd.read_pickle('ep.pkl'); anc=pd.read_pickle('anc.pkl')
CONT=['camera_minus30','camera_plus30','camera_minus15','camera_plus15','camera_minus15_surface_cool','camera_plus15_surface_warm','headlight_off','matched_global','surface_warm','surface_cool']
PULSE=['camera_plus15_q0_only','headlight_off_q0_only','surface_warm_q0_only']
rng=np.random.default_rng(0)
print('=== M1: outcome noise floor ===')
cl=ep[ep.condition=='clean'].dropna(subset=['succ'])
# clean-vs-clean discordance across noise repeats within root
d=[];
for r,g in cl.groupby('root'):
    s=g.succ.values
    for a,b in itertools.combinations(range(len(s)),2): d.append(s[a]!=s[b])
print('clean-vs-clean (diff noise) pairs',len(d),'discordance %.3f'%np.mean(d))
print('clean SR %.3f'%cl.succ.mean(), ' roots with any clean failure:', (cl.groupby('root').succ.min()==0).sum(),'/',cl.root.nunique())
rows=[]
for c in CONT+PULSE:
    g=ep[(ep.condition==c)].dropna(subset=['succ','clean_succ'])
    h=((g.succ==0)&(g.clean_succ==1)).sum(); rcv=((g.succ==1)&(g.clean_succ==0)).sum(); n=len(g)
    p=binomtest(int(h),int(h+rcv),0.5).pvalue if h+rcv>0 else np.nan
    # task-cluster bootstrap of dSR
    t=g.groupby(['suite','task']).apply(lambda x:(x.succ-x.clean_succ).mean())
    bs=[t.sample(len(t),replace=True,random_state=int(rng.integers(1e9))).mean() for _ in range(2000)]
    rows.append(dict(cond=c,n=n,harm=h,recov=rcv,flip_rate=(h+rcv)/n,net_pp=100*(rcv-h)/n,ci_lo=100*np.percentile(bs,2.5),ci_hi=100*np.percentile(bs,97.5),mcnemar_p=p))
M1=pd.DataFrame(rows); print(M1.round(3).to_string())
null=M1[M1.cond.isin(['matched_global','surface_warm','surface_cool'])]
print('near-null (photometric weak) flip rate %.3f; harm share %.2f'%(null.flip_rate.mean(), null.harm.sum()/(null.harm.sum()+null.recov.sum())))
M1.to_csv('M1_outcomes.csv',index=False)
print('\n=== M2: variance decomposition (slot0=q0, continuous conds) ===')
a=anc[(anc.anchor_slot==0)&anc.condition.isin(CONT)].copy()
a['task_id']=a.suite+'-t'+a.task.astype(str)
vars={'V_main':'visual_relative_clean_rms_main','V_wrist':'visual_relative_clean_rms_wrist','H18':'hidden_free_relative_clean_rms_prefix','Att_main':'attention_free_source_mass_delta_prefix_main','Att_wrist':'attention_free_source_mass_delta_prefix_wrist','JS':'attention_free_js_raw_prefix','Trans':'physical_prefix_translation_rms','Rot':'physical_prefix_rotation_rms','Grip':'physical_prefix_gripper_rms'}
out=[]
for k,v in vars.items():
    y=a[v].astype(float)
    if k in ['V_main','V_wrist','H18','Trans','Rot','Grip','JS']: y=np.log(y.clip(lower=1e-6))
    df=a.assign(y=y).dropna(subset=['y'])
    tot=((df.y-df.y.mean())**2).sum()
    def r2(f): return 1-smf.ols(f,df).fit().ssr/tot
    r_c=r2('y~C(condition)'); r_ct=r2('y~C(condition)+C(task_id)'); r_cxt=r2('y~C(condition)*C(task_id)'); r_root=r2('y~C(condition)*C(root)')
    out.append(dict(var=k,condition=r_c,task=r_ct-r_c,cond_x_task=r_cxt-r_ct,init_state=r_root-r_cxt,noise_resid=1-r_root))
M2=pd.DataFrame(out); print(M2.round(3).to_string()); M2.to_csv('M2_vardecomp.csv',index=False)
