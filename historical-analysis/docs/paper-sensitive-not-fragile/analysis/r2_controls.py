"""Round-2 reviewer-requested checks on EXISTING data only (no new rollouts, no model calls).
(a) cell-level association after removing condition (and task) effects
(b) within-condition AUROC for harm among clean-success pairs, camera conditions only (flip rate above placebo)
(c) placement offset vs dSR: without +-30, leave-one-condition-out range, and a survivor-bias check using only
    initial states whose episodes in that condition all succeeded
"""
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.expanduser('~/.codex/worktrees/adv-vla-research-20260924/Adv-VLA-π0.5/docs/experiments')
EL = R + '/analysis-system-20260929/comprehensive-review-20260930/late-evidence-synthesis-v1/joint-results-v2/episode-layer.csv.gz'
PKG = os.path.join(HERE, '..', '..', 'experiments', 'event-locked-placement-20261008-v1', 'data')
rng = np.random.default_rng(20261008)
PERSIST = ['camera_minus15', 'camera_minus15_surface_cool', 'camera_minus30', 'camera_plus15', 'camera_plus15_surface_warm',
           'camera_plus30', 'headlight_off', 'matched_global', 'surface_cool', 'surface_warm']
CAMERA = [c for c in PERSIST if 'camera' in c]
out = {}

e = pd.read_csv(EL, low_memory=False)
e = e[(e.layer == 18) & (e.original_present == True)].copy()
e['cond'] = e.history_condition
e['task_id'] = e.root.str.rsplit('-', n=1).str[0]
cl = e[e.cond == 'clean'][['root', 'noise_repeat', 'original_success']].rename(columns={'original_success': 'cs'})
p = e[e.cond.isin(PERSIST)].merge(cl, on=['root', 'noise_repeat'])
p['s'] = p.original_success.astype(float)
p['cs'] = p.cs.astype(float)
p['dY'] = p.s - p.cs
p['h'] = p.slot1_relative_rms
p['tv'] = p.slot1_main_TV
p['a'] = p.slot1_action_translation_rms


def resid_rank_corr(d, x, y, groups):
    d = d[[x, y] + groups].dropna().copy()
    X = pd.get_dummies(d[groups].astype(str), drop_first=True).astype(float)
    X.insert(0, 'c', 1.0)
    X = X.values
    rx = d[x].rank().values
    ry = d[y].rank().values
    rx = rx - X @ np.linalg.lstsq(X, rx, rcond=None)[0]
    ry = ry - X @ np.linalg.lstsq(X, ry, rcond=None)[0]
    return float(np.corrcoef(rx, ry)[0, 1])


# (a) cell level
cells = p.groupby(['cond', 'task_id'])[['h', 'tv', 'a', 'dY']].mean().reset_index()
A = {}
for x in ['h', 'tv', 'a']:
    st = []
    tasks = cells.task_id.unique()
    for _ in range(1000):
        s = rng.choice(tasks, len(tasks))
        cc = pd.concat([cells[cells.task_id == t].assign(task_id=f'{t}_{k}') for k, t in enumerate(s)])
        st.append(resid_rank_corr(cc, x, 'dY', ['cond']))
    A[x] = dict(cell_cond_adjusted=resid_rank_corr(cells, x, 'dY', ['cond']),
                cell_cond_task_adjusted=resid_rank_corr(cells, x, 'dY', ['cond', 'task_id']),
                ci_cond_adjusted=np.nanpercentile(st, [2.5, 97.5]).tolist())
out['a_cell_adjusted'] = A

# (b) AUROC within camera conditions, clean-success pairs, harm vs both-success
B = {}
q = p[(p.cs == 1) & p.cond.isin(CAMERA)].copy()
q['harm'] = (q.s == 0).astype(int)
for x in ['h', 'tv', 'a']:
    # pair-weighted within-condition AUC: pool within-condition comparisons only
    num = den = 0.0
    for c, g in q.groupby('cond'):
        g = g.dropna(subset=[x])
        if g.harm.nunique() < 2:
            continue
        auc = roc_auc_score(g.harm, g[x])
        npair = g.harm.sum() * (len(g) - g.harm.sum())
        num += auc * npair
        den += npair
    st = []
    tasks = q.task_id.unique()
    for _ in range(1000):
        s = rng.choice(tasks, len(tasks))
        qq = pd.concat([q[q.task_id == t] for t in s])
        n2 = d2 = 0.0
        for c, g in qq.groupby('cond'):
            g = g.dropna(subset=[x])
            if g.harm.nunique() < 2:
                continue
            npair = g.harm.sum() * (len(g) - g.harm.sum())
            n2 += roc_auc_score(g.harm, g[x]) * npair
            d2 += npair
        st.append(n2 / d2 if d2 else np.nan)
    B[x] = dict(auc=num / den, ci=np.nanpercentile(st, [2.5, 97.5]).tolist(), n=int(len(q)), harms=int(q.harm.sum()))
out['b_camera_auc'] = B

# (c) placement offset
law = pd.read_csv(os.path.join(PKG, 'scalar', 'fa_cond_law.csv'))
C = {}
no30 = law[~law.cond.isin(['camera_plus30', 'camera_minus30'])]
C['without_pm30'] = dict(n=len(no30), pearson=pearsonr(no30.sys_off, no30.dSR)[0], spearman=spearmanr(no30.sys_off, no30.dSR).correlation)
loo = [pearsonr(law.drop(i).sys_off, law.drop(i).dSR)[0] for i in law.index]
C['leave_one_out_pearson_range'] = [min(loo), max(loo)]
# survivor-bias check: offset using only roots where every episode of that condition succeeded
ob = pd.read_csv(os.path.join(PKG, 'review', 'object_episode.csv'))
ob = ob[ob.primary]
rows = []
for (c, r), g in ob[ob.cond != 'clean'].groupby(['cond', 'root']):
    if (g.success == True).all() and len(g) >= 2:
        rows.append(dict(cond=c, root=r, off=float(np.hypot(g.dx.mean(), g.dy.mean()))))
sb = pd.DataFrame(rows).groupby('cond').off.mean()
j = law.set_index('cond').join(sb.rename('off_allsucc')).dropna()
C['survivor_check'] = dict(n=len(j), pearson=pearsonr(j.off_allsucc, j.dSR)[0], spearman=spearmanr(j.off_allsucc, j.dSR).correlation,
                           corr_with_original_offset=pearsonr(j.off_allsucc, j.sys_off)[0],
                           offsets_mm=(j.off_allsucc * 1000).round(1).to_dict())
out['c_placement'] = C


def cj(o):
    if isinstance(o, dict):
        return {k: cj(v) for k, v in o.items()}
    if isinstance(o, list):
        return [cj(v) for v in o]
    if isinstance(o, (float, np.floating)):
        return None if np.isnan(o) else round(float(o), 4)
    if isinstance(o, np.integer):
        return int(o)
    return o


json.dump(cj(out), open(os.path.join(HERE, 'r2_controls.json'), 'w'), indent=1)
print(json.dumps(cj(out), indent=1))
