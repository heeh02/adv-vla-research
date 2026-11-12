import pandas as pd, numpy as np
from sklearn.metrics import roc_auc_score
R='/Users/researcher/.codex/worktrees/adv-vla-research-20260924/Adv-VLA-π0.5/docs/experiments/libero-light-v1/integrated-analysis-20260928/unified-generation-v1/results-v1/ten-step-transport.csv'
t=pd.read_csv(R,low_memory=False,usecols=['source_case_id','suite','task','root','noise_repeat','condition','window','channel','step_index','qualified','step_l2','cumulative_l2','endpoint_l2','step_cosine_to_endpoint','adjacent_cosine'])
t=t[(t.window=='planned_first5')&(t.qualified==True)]
w=t.pivot_table(index=['source_case_id','condition','root','noise_repeat','suite','task'],columns=['channel','step_index'],values=['step_l2','cumulative_l2'])
def comb(q,step,chs=('translation','rotation','gripper')): return np.sqrt(sum(w[(q,c,step)]**2 for c in chs))
out=pd.DataFrame(index=w.index)
for name,chs in {'all7':('translation','rotation','gripper'),'trans':('translation',),'rot':('rotation',)}.items():
    out[f's_{name}']=comb('step_l2',9,chs)/comb('cumulative_l2',8,chs)
    out[f'r_{name}']=comb('cumulative_l2',9,chs)/comb('cumulative_l2',8,chs)
    out[f'end_{name}']=comb('cumulative_l2',9,chs)
out=out.reset_index()
ep=pd.read_pickle('ep.pkl'); ep['source_case_id']='D3:'+ep.root+':n'+ep.noise_repeat.astype(str)+':'+ep.condition
m=out.merge(ep[['source_case_id','succ','clean_succ']],on='source_case_id')
m=m[m.clean_succ==1].dropna(subset=['succ']); m['harm']=(m.succ==0).astype(int); m['task_id']=m.suite+'-t'+m.task.astype(str)
print('cases',len(m),'harm',m.harm.sum()); print(m.groupby('harm')[[c for c in out.columns if c[:2] in ('s_','r_','en')]].median().round(4).to_string())
rng=np.random.default_rng(11)
ok=m.groupby(['task_id','condition']).harm.transform(lambda x:x.any() and (x==0).any()); s=m[ok].copy()
print('strata',s.groupby(['task_id','condition']).ngroups,'harms',s.harm.sum())
for v in [c for c in out.columns if c[:2] in ('s_','r_','en')]:
    s['rk']=s.groupby(['task_id','condition'])[v].rank(pct=True); mm=s.rk.notna()
    auc=roc_auc_score(s.harm[mm],s.rk[mm]); bs=[]; tasks=s.task_id.unique()
    for _ in range(500):
        tt=rng.choice(tasks,len(tasks)); bb=pd.concat([s[s.task_id==x] for x in tt]); k=bb.rk.notna()
        if bb.harm[k].nunique()>1: bs.append(roc_auc_score(bb.harm[k],bb.rk[k]))
    print(f'  {v:12s} stratified AUC={auc:.3f} [{np.percentile(bs,2.5):.2f},{np.percentile(bs,97.5):.2f}]')
m.to_pickle('d3_s.pkl')
