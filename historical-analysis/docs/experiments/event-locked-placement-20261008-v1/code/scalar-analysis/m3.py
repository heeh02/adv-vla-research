import pandas as pd, numpy as np
from scipy.stats import spearmanr, rankdata
anc=pd.read_pickle('anc.pkl'); ep=pd.read_pickle('ep.pkl')
CONT=['camera_minus30','camera_plus30','camera_minus15','camera_plus15','camera_minus15_surface_cool','camera_plus15_surface_warm','headlight_off','matched_global','surface_warm','surface_cool']
V={'V':'visual_relative_clean_rms_main','H':'hidden_free_relative_clean_rms_prefix','Att':'attention_free_source_mass_delta_prefix_main','JS':'attention_free_js_raw_prefix','T':'physical_prefix_translation_rms','R':'physical_prefix_rotation_rms','G':'physical_prefix_gripper_rms'}
a=anc[(anc.anchor_slot==0)&anc.condition.isin(CONT)].copy()
a=a.dropna(subset=['succ','clean_succ'])
a['dY']=a.succ-a.clean_succ
a['task_id']=a.suite+'-t'+a.task.astype(str)
for k,v in V.items(): a[k]=a[v].astype(float)
links=[('V','H'),('H','Att'),('H','JS'),('H','T'),('Att','T'),('JS','T'),('T','R'),('H','dY'),('Att','dY'),('T','dY'),('R','dY'),('G','dY')]
def within(df,x,y,grp):
    # rank-correlation after removing group (condition) means of ranks
    rx=df.groupby(grp)[x].rank(pct=True); ry=df.groupby(grp)[y].rank(pct=True)
    return np.corrcoef(rx-rx.groupby(df[grp]).transform('mean'), ry-ry.groupby(df[grp]).transform('mean'))[0,1]
res=[]
levels={'episode':['episode_id','condition'],'root×cond':['root','condition'],'task×cond':['task_id','condition'],'condition':['condition']}
for lv,keys in levels.items():
    g=a.groupby(keys)[list(V)+['dY']].mean().reset_index()
    for x,y in links:
        r=spearmanr(g[x],g[y],nan_policy='omit').correlation
        w=within(g,x,y,'condition') if lv!='condition' else np.nan
        res.append(dict(level=lv,n=len(g),link=f'{x}->{y}',rho_total=r,rho_within_cond=w))
R=pd.DataFrame(res); P=R.pivot_table(index='link',columns='level',values=['rho_total','rho_within_cond'])
print(P.round(2).to_string())
# reliability of task×cond dY and T via init split-half
s=a.groupby(['task_id','condition','init'])[['dY','T','H']].mean().unstack('init')
inits=sorted(a.init.unique())
for v in ['dY','T','H']:
    r=s[v][inits[0]].corr(s[v][inits[1]]); sb=2*r/(1+r)
    print(f'task×cond split-half reliability {v}: r={r:.3f}, Spearman-Brown={sb:.3f}')
# within-condition split-half (remove condition mean)
sc=s.copy()
for v in ['dY','T']:
    for i in inits:
        col=sc[(v,i)]; sc[(v,i)]=col-col.groupby(level='condition').transform('mean')
    r=sc[v][inits[0]].corr(sc[v][inits[1]]); print(f'within-cond split-half {v}: r={r:.3f}')
R.to_csv('M3_links.csv',index=False)
