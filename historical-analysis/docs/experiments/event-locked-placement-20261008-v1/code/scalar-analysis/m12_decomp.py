import pandas as pd, numpy as np, json
R='/Users/researcher/.codex/worktrees/adv-vla-research-20260924/Adv-VLA-π0.5/docs/experiments/libero-light-v1/integrated-analysis-20260928/time-pulse-v1/results-v1/anchor-decomposition.csv'
d=pd.read_csv(R,low_memory=False)
x=d[(d.window=='planned_first5')&(d.channel=='full_adapter')].copy()
for k in ['total','current','state']:
    x[k+'_v']=x[k+'_signed_by_coordinate_json'].map(lambda s: np.array(json.loads(s)) if isinstance(s,str) else None)
ep=pd.read_pickle('ep.pkl')
x=x.merge(ep[['root','noise_repeat','condition','succ','clean_succ']],on=['root','noise_repeat','condition'],how='left')
x['cell']=np.select([(x.succ==1)&(x.clean_succ==1),(x.succ==0)&(x.clean_succ==1),(x.succ==1)&(x.clean_succ==0)],['SS','harm','recov'],'FF')
x.to_pickle('decomp_full.pkl')
g=x.groupby(['branch_role','anchor_slot'])[['total_rms','current_rms','state_rms']].median()
print('median RMS of action difference (7-d adapter, planned first5)'); print(g.round(4).to_string())
g2=x.groupby(['branch_role','anchor_slot']).apply(lambda t: pd.Series({'state_energy_share':(t.energy_state/t.energy_total).median(),'cross_term_share':(t.cross_term_2dot_mean/t.energy_total).median()}))
print(g2.round(3).to_string())
print(x.groupby(['branch_role','cell']).root.nunique())
