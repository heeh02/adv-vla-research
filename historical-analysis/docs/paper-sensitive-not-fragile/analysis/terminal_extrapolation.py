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

out = {}
s = m[m.success == True]
out['gap_steps'] = s.gap.value_counts().sort_index().to_dict()
out['unobserved_disp_mm_success'] = dict(median=float(s.unobs.median() * 1e3), p90=float(s.unobs.quantile(.9) * 1e3),
                                         p99=float(s.unobs.quantile(.99) * 1e3), frac_gt_2mm=float((s.unobs > .002).mean()),
                                         frac_gt_5mm=float((s.unobs > .005).mean()))
fam = lambda c: 'clean' if c == 'clean' else ('camera' if 'camera' in c and 'q0' not in c else 'other')
out['unobserved_median_mm_by_family'] = (s.groupby(s.cond.map(fam)).unobs.median() * 1e3).round(2).to_dict()

# recompute offsets with extrapolated terminal positions
ref = m[(m.cond == 'clean') & (m.success == True)].groupby('root')[['xT', 'yT']].median()
m = m.join(ref.rename(columns={'xT': 'rx', 'yT': 'ry'}), on='root')
m['dxT'] = m.xT - m.rx
m['dyT'] = m.yT - m.ry
law = pd.read_csv(os.path.join(PKG, 'data', 'scalar', 'fa_cond_law.csv')).set_index('cond')
rows = []
for c, g in m[(m.success == True) & (m.cond != 'clean')].groupby('cond'):
    sysv, spr = [], []
    for r, h in g.groupby('root'):
        v = h[['dxT', 'dyT']].values
        mu = v.mean(0)
        sysv.append(np.hypot(*mu))
        if len(v) > 1:
            spr.append(np.sqrt(((v - mu) ** 2).sum(1).mean()))
    rows.append(dict(cond=c, sys_T=np.mean(sysv), spread_T=np.mean(spr), sys_orig=law.loc[c, 'sys_off'], dSR=law.loc[c, 'dSR']))
R = pd.DataFrame(rows)
out['offset_terminal_vs_lastquery_r'] = float(pearsonr(R.sys_T, R.sys_orig)[0])
out['corr_sysT_dSR'] = dict(pearson=float(pearsonr(R.sys_T, R.dSR)[0]), spearman=float(spearmanr(R.sys_T, R.dSR).correlation))
out['corr_spreadT_dSR'] = float(pearsonr(R.spread_T, R.dSR)[0])
out['per_condition_mm'] = {r.cond: dict(terminal=round(r.sys_T * 1e3, 1), last_query=round(r.sys_orig * 1e3, 1),
                                        spread=round(r.spread_T * 1e3, 1)) for r in R.itertuples()}
# re-seeding floor with terminal positions
cl = m[(m.cond == 'clean') & (m.success == True)]
res = {r: g[['dxT', 'dyT']].values for r, g in cl.groupby('root')}
fl = []
for _ in range(2000):
    offs = []
    for r, v in res.items():
        if len(v) < 2:
            continue
        c = v[rng.integers(0, len(v), 3)].mean(0)
        rf = np.median(v[rng.integers(0, len(v), 3)], 0)
        offs.append(np.hypot(*(c - rf)))
    fl.append(np.mean(offs))
out['reseed_floor_terminal_mm'] = dict(mean=float(np.mean(fl) * 1e3), ci=(np.percentile(fl, [2.5, 97.5]) * 1e3).tolist())
json.dump(out, open(os.path.join(HERE, 'terminal_extrapolation.json'), 'w'), indent=1)
print(json.dumps(out, indent=1))
