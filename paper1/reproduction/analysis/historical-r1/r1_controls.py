"""Post-hoc controls requested by round-1 review, on EXISTING data only (no new rollouts, no model calls).

A. Re-seeding placebo: outcome discordance between clean episodes of the same initial state that differ only
   in generation-noise seed, versus perturbed/clean discordance (same seed) per condition.
B. Reliability of task-level outcomes: split-half (initial state 12 vs 13) for cells; cross-cohort for
   camera +-30 (multi-condition study vs independent large-angle study); and whether response magnitude
   predicts the reliable task-level vulnerability.
C. Episode-level continuous outcome: within-condition rank association between response magnitude and the
   primary object's final placement error in successful episodes.
D. Attention: |signed main-camera mass change| vs translation response, within condition and task+condition adjusted.
Sources: D3 episode-layer (layer 18), severity compact results, event-locked object_episode.csv.
"""
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import binomtest, spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.expanduser('~/.codex/worktrees/adv-vla-research-20260924/Adv-VLA-π0.5/docs/experiments')
EL = R + '/analysis-system-20260929/comprehensive-review-20260930/late-evidence-synthesis-v1/joint-results-v2/episode-layer.csv.gz'
SEV = R + '/libero-light-v1/overnight-20260929/native-camera-severity-confirm-v2/SEVERITY_FULL2160_COMPACT_RESULTS-v1.json'
OBJ = os.path.join(HERE, '..', '..', 'experiments', 'event-locked-placement-20261008-v1', 'data', 'review', 'object_episode.csv')
rng = np.random.default_rng(20261008)
out = {}

e = pd.read_csv(EL, low_memory=False)
e = e[(e.layer == 18) & (e.original_present == True)].copy()
e['cond'] = e.history_condition
e['task_id'] = e.root.str.rsplit('-', n=1).str[0]
e['init'] = e.root.str.rsplit('-i', n=1).str[1]
e['succ'] = e.original_success.astype(float)
PERSIST = ['camera_minus15', 'camera_minus15_surface_cool', 'camera_minus30', 'camera_plus15', 'camera_plus15_surface_warm',
           'camera_plus30', 'headlight_off', 'matched_global', 'surface_cool', 'surface_warm']

# ---------- A. placebo ----------
clean = e[e.cond == 'clean'][['root', 'task_id', 'noise_repeat', 'succ']]
pl = []
for root, g in clean.groupby('root'):
    v = g.sort_values('noise_repeat').succ.values
    for i in range(len(v)):
        for j in range(i + 1, len(v)):
            pl.append(dict(root=root, task_id=g.task_id.iloc[0], disc=float(v[i] != v[j])))
pl = pd.DataFrame(pl)
cm = clean.rename(columns={'succ': 'clean_succ'})
pairs = e[e.cond.isin(PERSIST)].merge(cm[['root', 'noise_repeat', 'clean_succ']], on=['root', 'noise_repeat'])
pairs['disc'] = (pairs.succ != pairs.clean_succ).astype(float)
pairs['harm'] = ((pairs.clean_succ == 1) & (pairs.succ == 0)).astype(float)
pairs['recov'] = ((pairs.clean_succ == 0) & (pairs.succ == 1)).astype(float)


def task_boot(df, col, B=2000):
    tasks = df.task_id.unique()
    by = {t: df[df.task_id == t][col].values for t in tasks}
    st = []
    for _ in range(B):
        s = rng.choice(tasks, len(tasks))
        st.append(np.concatenate([by[t] for t in s]).mean())
    return np.percentile(st, [2.5, 97.5])


A = {'placebo_clean_reseed': dict(n_pairs=len(pl), disc=pl.disc.mean(), ci=task_boot(pl, 'disc').tolist())}
for c in PERSIST:
    g = pairs[pairs.cond == c]
    A[c] = dict(n=len(g), disc=g.disc.mean(), ci=task_boot(g, 'disc').tolist(), harm=int(g.harm.sum()), recov=int(g.recov.sum()),
                sign_p=binomtest(int(g.harm.sum()), int(g.harm.sum() + g.recov.sum())).pvalue if g.harm.sum() + g.recov.sum() > 0 else np.nan)
g = pairs
A['all_persistent'] = dict(n=len(g), disc=g.disc.mean(), harm=int(g.harm.sum()), recov=int(g.recov.sum()))
out['A_placebo'] = A

# ---------- B. reliability ----------
pairs['dY'] = pairs.succ - pairs.clean_succ
cell_init = pairs.groupby(['cond', 'task_id', 'init']).dY.mean().unstack('init')
cell_init = cell_init.dropna()
rB = {}
rB['split_half_cells_400'] = dict(n=len(cell_init), spearman=spearmanr(cell_init.iloc[:, 0], cell_init.iloc[:, 1]).correlation,
                                  pearson=float(np.corrcoef(cell_init.iloc[:, 0], cell_init.iloc[:, 1])[0, 1]))
cam30 = pairs[pairs.cond.isin(['camera_minus30', 'camera_plus30'])]
d3_task30 = cam30.groupby('task_id').dY.mean()
sev = json.load(open(SEV))['contrasts']['supp_pooled_30_minus_own_clean']['task_values']
sev30 = pd.Series({t['task_id']: t['value'] for t in sev})
j = pd.concat([d3_task30.rename('d3'), sev30.rename('sev')], axis=1).dropna()
rB['cross_cohort_task_cam30'] = dict(n=len(j), spearman=spearmanr(j.d3, j.sev).correlation, p=spearmanr(j.d3, j.sev).pvalue,
                                     pearson=float(np.corrcoef(j.d3, j.sev)[0, 1]))
# does response magnitude (multi-condition study) predict the independent large-angle task vulnerability?
cam30['resp_h'] = cam30[['slot1_relative_rms', 'slot2_relative_rms']].mean(1)
cam30['resp_a'] = cam30[['slot1_action_translation_rms', 'slot2_action_translation_rms']].mean(1)
cam30['resp_tv'] = cam30[['slot1_main_TV', 'slot2_main_TV']].mean(1)
tf = cam30.groupby('task_id')[['resp_h', 'resp_a', 'resp_tv']].mean()
jj = tf.join(sev30.rename('sev')).dropna()
for c in ['resp_h', 'resp_a', 'resp_tv']:
    r = spearmanr(jj[c], jj.sev)
    rB[f'task_{c}_vs_independent_sev30'] = dict(n=len(jj), spearman=r.correlation, p=r.pvalue)
# bootstrap CI of the response-vs-vulnerability task correlation
for c in ['resp_h', 'resp_a']:
    st = []
    for _ in range(2000):
        idx = rng.choice(len(jj), len(jj))
        st.append(spearmanr(jj[c].values[idx], jj.sev.values[idx]).correlation)
    rB[f'task_{c}_vs_independent_sev30']['ci'] = np.nanpercentile(st, [2.5, 97.5]).tolist()
out['B_reliability'] = rB

# ---------- C. episode-level placement ----------
ob = pd.read_csv(OBJ)
ob = ob[ob.primary & (ob.success == True) & ob.cond.isin(PERSIST)][['root', 'noise', 'cond', 'final_err_xy']]
m = pairs.merge(ob, left_on=['root', 'noise_repeat', 'cond'], right_on=['root', 'noise', 'cond'])
m['resp_h'] = m[['slot1_relative_rms', 'slot2_relative_rms']].mean(1)
m['resp_a'] = m[['slot1_action_translation_rms', 'slot2_action_translation_rms']].mean(1)


def adj_rank_corr(df, x, y, groups):
    d = df[[x, y] + groups].dropna().copy()
    for v in [x, y]:
        d[v] = d[v].rank()
        for gcol in groups:  # additive removal by iterative demeaning
            pass
    # residualize ranks on additive group means via least squares
    X = pd.get_dummies(d[groups].astype(str), drop_first=True).astype(float)
    X.insert(0, 'c', 1.0)
    X = X.values
    rx = d[x].values - X @ np.linalg.lstsq(X, d[x].values, rcond=None)[0]
    ry = d[y].values - X @ np.linalg.lstsq(X, d[y].values, rcond=None)[0]
    return float(np.corrcoef(rx, ry)[0, 1]), len(d)


C = {}
for x in ['resp_h', 'resp_a']:
    r_c, n = adj_rank_corr(m, x, 'final_err_xy', ['cond'])
    r_tc, _ = adj_rank_corr(m, x, 'final_err_xy', ['cond', 'task_id'])
    st = []
    tasks = m.task_id.unique()
    for _ in range(1000):
        s = rng.choice(tasks, len(tasks))
        mm = pd.concat([m[m.task_id == t].assign(task_id=f'{t}_{k}') for k, t in enumerate(s)])
        st.append(adj_rank_corr(mm, x, 'final_err_xy', ['cond', 'task_id'])[0])
    C[x] = dict(n=n, within_condition=r_c, task_condition_adjusted=r_tc, ci=np.nanpercentile(st, [2.5, 97.5]).tolist())
out['C_episode_placement'] = C

# ---------- D. attention |mass| ----------
pairs['abs_m'] = pairs[['slot1_main_mass_delta', 'slot2_main_mass_delta']].abs().mean(1)
pairs['tv'] = pairs[['slot1_main_TV', 'slot2_main_TV']].mean(1)
pairs['act'] = pairs[['slot1_action_translation_rms', 'slot2_action_translation_rms']].mean(1)
pairs['hid'] = pairs[['slot1_relative_rms', 'slot2_relative_rms']].mean(1)
D = {}
for x in ['abs_m', 'tv', 'hid']:
    D[x] = dict(within_condition=adj_rank_corr(pairs, x, 'act', ['cond'])[0],
                task_condition_adjusted=adj_rank_corr(pairs, x, 'act', ['cond', 'task_id'])[0], n=int(pairs[[x, 'act']].dropna().shape[0]))
out['D_attention_abs_mass'] = D


def clean_json(o):
    if isinstance(o, dict):
        return {k: clean_json(v) for k, v in o.items()}
    if isinstance(o, list):
        return [clean_json(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if np.isnan(o) else round(float(o), 4)
    if isinstance(o, (np.integer,)):
        return int(o)
    return o


json.dump(clean_json(out), open(os.path.join(HERE, 'r1_controls.json'), 'w'), indent=1)
print(json.dumps(clean_json(out), indent=1))
