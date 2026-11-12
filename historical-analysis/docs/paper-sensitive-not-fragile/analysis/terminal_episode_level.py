"""Does using the last recorded policy query (1-5 control steps before termination) instead of the true
terminal state change the placement-bias result?

Existing data only (event-locked package shard-*.pkl, 3,024 D3 episodes). For each episode we take the primary
object's planar position at the last two recorded queries (5 control steps apart), estimate its velocity, and
linearly extrapolate to the termination step (result.json `steps`). We then (1) quantify the unobserved
displacement, and (2) recompute the systematic placement offset, its re-seeding floor, the spread, and the
correlation with paired success change using extrapolated terminal positions.
Linear extrapolation is a first-order approximation; it does not observe contact events in the final steps.
"""
import glob
import json
import os
import pickle

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.join(HERE, '..', '..', 'experiments', 'event-locked-placement-20261008-v1')
rng = np.random.default_rng(20261008)

ob = pd.read_csv(os.path.join(PKG, 'data', 'review', 'object_episode.csv'))
ob = ob[ob.primary].copy()
need = {(r.root, int(r.noise), r.cond): int(r.physics_flat_offset) for r in ob.itertuples()}

last = {}
for f in sorted(glob.glob(os.path.join(PKG, 'data', 'derived-trajectories', 'shard-*.pkl'))):
    o = pickle.load(open(f, 'rb'))
    for ep, st in o['state'].items():
        key = (ep[0], int(ep[1]), ep[2])
        if key not in need or len(st) < 2:
            continue
        b = need[key]
        pL = st[-1]['phys'][b:b + 2].astype(float)
        pP = st[-2]['phys'][b:b + 2].astype(float)
        dstep = st[-1]['step'] - st[-2]['step']
        T = o['result'].get(ep, {}).get('steps')
        gap = (T - st[-1]['step']) if T is not None else np.nan
        last[key] = dict(xL=pL[0], yL=pL[1], vx=(pL[0] - pP[0]) / dstep, vy=(pL[1] - pP[1]) / dstep, gap=gap)

L = pd.DataFrame([dict(root=k[0], noise=k[1], cond=k[2], **v) for k, v in last.items()])
m = ob.merge(L, on=['root', 'noise', 'cond'])
assert np.allclose(m.xL, m.final_x) and np.allclose(m.yL, m.final_y)
m['xT'] = m.xL + m.vx * m.gap
m['yT'] = m.yL + m.vy * m.gap
m['unobs'] = np.hypot(m.xT - m.xL, m.yT - m.yL)


ref = m[(m.cond=='clean')&(m.success==True)].groupby('root')[['xT','yT']].median()
m2 = m.join(ref.rename(columns={'xT':'rx','yT':'ry'}), on='root'); m2['errT']=np.hypot(m2.xT-m2.rx, m2.yT-m2.ry)
R=os.path.expanduser('~/.codex/worktrees/adv-vla-research-20260924/Adv-VLA-π0.5/docs/experiments/analysis-system-20260929/comprehensive-review-20260930/late-evidence-synthesis-v1/joint-results-v2/episode-layer.csv.gz')
e=pd.read_csv(R,low_memory=False); e=e[(e.layer==18)&(e.original_present==True)].copy(); e['task_id']=e.root.str.rsplit('-',n=1).str[0]
P=['camera_minus15','camera_minus15_surface_cool','camera_minus30','camera_plus15','camera_plus15_surface_warm','camera_plus30','headlight_off','matched_global','surface_cool','surface_warm']
e=e[e.history_condition.isin(P)].copy(); e['h']=e[['slot1_relative_rms','slot2_relative_rms']].mean(1); e['a']=e[['slot1_action_translation_rms','slot2_action_translation_rms']].mean(1)
j=e.merge(m2[(m2.success==True)][['root','noise','cond','errT','final_err_xy']],left_on=['root','noise_repeat','history_condition'],right_on=['root','noise','cond'])
def adj(d,x,y,groups):
    d=d[[x,y]+groups].dropna().copy(); X=pd.get_dummies(d[groups].astype(str),drop_first=True).astype(float); X.insert(0,'c',1.0); X=X.values
    rx=d[x].rank().values; ry=d[y].rank().values
    rx=rx-X@np.linalg.lstsq(X,rx,rcond=None)[0]; ry=ry-X@np.linalg.lstsq(X,ry,rcond=None)[0]; return float(np.corrcoef(rx,ry)[0,1])
tasks=j.task_id.unique(); res={}
for x in ['h','a']:
    st=[]
    for _ in range(500):
        s=rng.choice(tasks,len(tasks)); jj=pd.concat([j[j.task_id==t].assign(task_id=f'{t}_{k}') for k,t in enumerate(s)])
        st.append(adj(jj,x,'errT',['history_condition','task_id']))
    res[x]=dict(n=len(j),adj_terminal=adj(j,x,'errT',['history_condition','task_id']),ci=np.percentile(st,[2.5,97.5]).round(3).tolist(),adj_lastquery=adj(j,x,'final_err_xy',['history_condition','task_id']))
print(res)
json.dump(res,open(os.path.join(HERE,'terminal_episode_level.json'),'w'),indent=1)
