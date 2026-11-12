import pandas as pd, numpy as np
from sklearn.metrics import roc_auc_score
anc=pd.read_pickle('anc.pkl')
CONT=['camera_minus30','camera_plus30','camera_minus15','camera_plus15','camera_minus15_surface_cool','camera_plus15_surface_warm','headlight_off','matched_global','surface_warm','surface_cool']
a=anc[anc.condition.isin(CONT)].dropna(subset=['succ','clean_succ']).copy()
a['cell']=np.select([(a.succ==1)&(a.clean_succ==1),(a.succ==0)&(a.clean_succ==1),(a.succ==1)&(a.clean_succ==0)],['SS','harm','recov'],'FF')
a['task_id']=a.suite+'-t'+a.task.astype(str)
a['Trel']=a.physical_prefix_translation_rms/ a.native_clean_update_group_rms_prefix_translation
V={'H18':'hidden_free_relative_clean_rms_prefix','Trans':'physical_prefix_translation_rms','Trel':'Trel','Rot':'physical_prefix_rotation_rms','JS':'attention_free_js_raw_prefix'}
rng=np.random.default_rng(1)
for s in [0,1,2]:
    b=a[(a.anchor_slot==s)&a.cell.isin(['SS','harm'])].copy()
    # strata: task x condition with >=1 harm and >=1 SS
    ok=b.groupby(['task_id','condition']).cell.transform(lambda x:(x=='harm').any() and (x=='SS').any())
    b=b[ok]
    out=[]
    for k,v in V.items():
        b['r']=b.groupby(['task_id','condition'])[v].rank(pct=True)
        auc=roc_auc_score(b.cell=='harm',b.r)
        # task-cluster bootstrap
        tasks=b.task_id.unique(); bs=[]
        for _ in range(1000):
            tt=rng.choice(tasks,len(tasks)); bb=pd.concat([b[b.task_id==t] for t in tt])
            if (bb.cell=='harm').sum()>0: bs.append(roc_auc_score(bb.cell=='harm',bb.r))
        out.append(f'{k}:{auc:.3f}[{np.percentile(bs,2.5):.2f},{np.percentile(bs,97.5):.2f}]')
    print(f'slot{s} strata={b.groupby(["task_id","condition"]).ngroups} harms={(b.cell=="harm").sum()} SS={(b.cell=="SS").sum()} :: '+'  '.join(out))
# which tasks carry the harm? concentration
h=a[(a.anchor_slot==0)&(a.cell=='harm')]
tc=h.groupby('task_id').size().sort_values(ascending=False)
print('harm events per task (top 12):',tc.head(12).to_dict(), ' tasks with any harm:',len(tc),'/40')
r=a[(a.anchor_slot==0)&(a.cell=='recov')].groupby('task_id').size()
both=pd.concat([tc.rename('harm'),r.rename('recov')],axis=1).fillna(0)
print('corr task harm vs recov counts: %.2f'%both.harm.corr(both.recov))
