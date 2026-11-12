import pandas as pd, numpy as np
from scipy.stats import spearmanr
el=pd.read_pickle('el.pkl'); el['task_id']=el.suite+'-t'+el.task.astype(str).str.zfill(2)
f=pd.read_csv('M6_task_features.csv',index_col=0)
num=['slot1_main_mass_clean','slot1_wrist_mass_clean','slot1_main_mass_delta','slot1_wrist_mass_delta','slot1_main_TV','slot1_wrist_TV','slot1_relative_rms','slot2_main_mass_clean','slot2_wrist_mass_clean']
for c in num: el[c]=pd.to_numeric(el[c],errors='coerce')
rows=[]
for L in [1,9,18]:
    e=el[el.layer==L]
    cl=e[e.history_condition=='clean'].groupby('task_id')[['slot1_main_mass_clean','slot1_wrist_mass_clean','slot2_main_mass_clean','slot2_wrist_mass_clean']].mean()
    cl['main_share']=cl.slot1_main_mass_clean/(cl.slot1_main_mass_clean+cl.slot1_wrist_mass_clean)
    cl['main_share2']=cl.slot2_main_mass_clean/(cl.slot2_main_mass_clean+cl.slot2_wrist_mass_clean)
    c30=e[e.history_condition.isin(['camera_minus30','camera_plus30'])].groupby('task_id')[['slot1_main_mass_delta','slot1_wrist_mass_delta','slot1_main_TV','slot1_relative_rms']].mean()
    j=cl.join(c30).join(f[['Sev_30','D3_30','Fresh_15']])
    for x in ['slot1_main_mass_clean','slot1_wrist_mass_clean','main_share','main_share2','slot1_main_mass_delta','slot1_wrist_mass_delta','slot1_main_TV','slot1_relative_rms']:
        for y in ['Sev_30','D3_30','Fresh_15']:
            jj=j[[x,y]].dropna(); r=spearmanr(jj[x],jj[y])
            rows.append(dict(layer=L,x=x,y=y,rho=r.correlation,p=r.pvalue))
R=pd.DataFrame(rows)
print(R.pivot_table(index=['x'],columns=['layer','y'],values='rho').round(2).to_string())
print('\nsignificant (p<0.05):'); print(R[R.p<0.05].round(3).to_string())
