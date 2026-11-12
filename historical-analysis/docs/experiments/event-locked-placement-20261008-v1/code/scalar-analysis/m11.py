import pandas as pd, numpy as np
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
ep=pd.read_pickle('ep.pkl'); anc=pd.read_pickle('anc.pkl')
CONT=['camera_minus30','camera_plus30','camera_minus15','camera_plus15','camera_minus15_surface_cool','camera_plus15_surface_warm','headlight_off','matched_global','surface_warm','surface_cool']
e=ep[ep.condition.isin(CONT)].dropna(subset=['succ','clean_succ']).copy()
e['dY']=e.succ-e.clean_succ; e['task_id']=e.suite+'-t'+e.task.astype(str)
fit=smf.ols('dY~C(condition)+C(task_id)+C(condition):C(task_id)',e).fit()
a=anova_lm(fit,typ=1); a['share']=a.sum_sq/a.sum_sq.sum(); print('=== outcome dY ANOVA (episode level) ==='); print(a[['df','sum_sq','share','F','PR(>F)']].round(4).to_string())
# also condition success (not paired) to avoid shared-clean artifacts
fit2=smf.ols('succ~C(condition)+C(task_id)+C(condition):C(task_id)+C(root)',e).fit(); a2=anova_lm(fit2,typ=1); a2['share']=a2.sum_sq/a2.sum_sq.sum()
print('\n=== perturbed-episode success ANOVA ==='); print(a2[['df','share','F','PR(>F)']].round(4).to_string())
print('\n=== condition share of variance over time (slots 0-2) ===')
V={'H18':'hidden_free_relative_clean_rms_prefix','Att':'attention_free_source_mass_delta_prefix_main','JS':'attention_free_js_raw_prefix','Trans':'physical_prefix_translation_rms','Rot':'physical_prefix_rotation_rms','Grip':'physical_prefix_gripper_rms','Vis':'visual_relative_clean_rms_main'}
rows=[]
for s in [0,1,2]:
    d=anc[(anc.anchor_slot==s)&anc.condition.isin(CONT)].copy(); d['task_id']=d.suite+'-t'+d.task.astype(str)
    for k,v in V.items():
        y=d[v].astype(float); y=np.log(y.clip(lower=1e-6)) if k!='Att' else y
        dd=d.assign(y=y).dropna(subset=['y']); tot=((dd.y-dd.y.mean())**2).sum()
        r_c=1-smf.ols('y~C(condition)',dd).fit().ssr/tot; r_ct=1-smf.ols('y~C(condition)+C(task_id)',dd).fit().ssr/tot
        r_root=1-smf.ols('y~C(condition)*C(root)',dd).fit().ssr/tot
        rows.append(dict(slot=s,var=k,condition=r_c,task=r_ct-r_c,state_x=r_root-r_ct,noise=1-r_root))
R=pd.DataFrame(rows); print(R.pivot(index='var',columns='slot',values='condition').round(2).to_string())
print('noise share:'); print(R.pivot(index='var',columns='slot',values='noise').round(2).to_string())
