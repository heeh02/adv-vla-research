import pandas as pd, numpy as np
from sklearn.metrics import roc_auc_score
anc=pd.read_pickle('anc.pkl')
CONT=['camera_minus30','camera_plus30','camera_minus15','camera_plus15','camera_minus15_surface_cool','camera_plus15_surface_warm','headlight_off','matched_global','surface_warm','surface_cool']
a=anc[anc.condition.isin(CONT)].dropna(subset=['succ','clean_succ']).copy()
a['cell']=np.select([(a.succ==1)&(a.clean_succ==1),(a.succ==0)&(a.clean_succ==1),(a.succ==1)&(a.clean_succ==0)],['SS','harm','recov'],'FF')
V={'H18':'hidden_free_relative_clean_rms_prefix','Trans':'physical_prefix_translation_rms','Rot':'physical_prefix_rotation_rms','Grip':'physical_prefix_gripper_rms','JS':'attention_free_js_raw_prefix','Vis':'visual_relative_clean_rms_main'}
print('median response by slot and outcome cell (camera ±30 only) [ratio harm/SS]')
cam30=a[a.condition.isin(['camera_minus30','camera_plus30'])]
for k,v in V.items():
    t=cam30.groupby(['anchor_slot','cell'])[v].median().unstack('cell')
    t['harm/SS']=t['harm']/t['SS']
    print(k); print(t[['SS','harm','recov','harm/SS']].round(4).to_string())
print('\nwithin-condition AUC (harm vs SS), conditions with >=5 harms, by slot')
rows=[]
for s in range(0,4):
    b=a[(a.anchor_slot==s)&a.cell.isin(['SS','harm'])]
    for k,v in V.items():
        au=[];w=[]
        for c,g in b.groupby('condition'):
            g=g.dropna(subset=[v])
            if (g.cell=='harm').sum()>=5 and (g.cell=='SS').sum()>=5:
                au.append(roc_auc_score(g.cell=='harm',g[v])); w.append((g.cell=='harm').sum())
        rows.append(dict(slot=s,var=k,auc=np.average(au,weights=w) if au else np.nan,k=len(au),harms=sum(w)))
print(pd.DataFrame(rows).pivot(index='var',columns='slot',values='auc').round(3).to_string())
# reached counts by slot x cell (survivorship)
print(a.groupby(['anchor_slot','cell']).size().unstack().to_string())
