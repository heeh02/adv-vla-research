"""P4 prediction models (PREREG-P4.md sections 4 and 5). numpy only.

All fitting, standardization, PCA and hyperparameter selection use training rows only. Inner CV is task-grouped
(3 folds, seed 20261009 + outer fold) and selects the configuration with the largest pooled inner-validation EEF
(ties: first in the listed order, which runs from most to least regularized and from small to large m).
"""
import numpy as np

from p4lib import PCA, cell_means, eef, group_folds, ridge_fit, standardize_fit

LAM_M2L = list(np.logspace(4, -3, 24))                       # 24 configurations, most regularized first
LAM_M3 = list(np.logspace(3, -2, 6))
M_GRID = [8, 16, 32, 64]
CFG_M3 = [(lam, m) for m in M_GRID for lam in LAM_M3]       # 24 configurations
KMAX = 64


# ------------------------------------------------------------------ feature blocks
class Blocks:
    """Training-fitted feature construction for one (train, eval) split."""

    def __init__(self, data, tr, use_pca_full=False, pca_key='n0', z_keys=('u0', 'a0'), seed=20261009):
        self.data, self.z_keys, self.pca_key = data, z_keys, pca_key
        self.z = {k: standardize_fit(data[k][tr]) for k in z_keys}
        self.use_full = use_pca_full
        self.pca = None
        if pca_key is not None:
            src = data['dr0full'] if use_pca_full else data[pca_key]
            self.pca = PCA(src[tr], KMAX, seed=seed)
            s = self.pca.scores(src[tr], KMAX)
            self.smu, self.ssd = standardize_fit(s)

    def X(self, idx, keys, m):
        parts = [(self.data[k][idx] - self.z[k][0]) / self.z[k][1] for k in keys]
        if m:
            src = self.data['dr0full'] if self.use_full else self.data[self.pca_key]
            s = self.pca.scores(src[idx], KMAX)
            parts.append(((s - self.smu) / self.ssd)[:, :m])
        return np.hstack(parts) if parts else np.zeros((len(idx), 0))


def m0_pred(data, tr, ev, use_cond):
    Y = data['Y']
    if use_cond:
        return cell_means(Y[tr], data['cell'][tr], data['cell'][ev], data['cond'][tr], data['cond'][ev])
    return cell_means(Y[tr], data['anchor'][tr], data['anchor'][ev], data['anchor'][tr], data['anchor'][ev])


def ridge_family(data, tr, ev, keys, configs, use_cond, pca_key='n0', use_full=False, base='m0', target='Y'):
    """Generic: base + ridge(features) fitted on residual target - base. configs: list of (lam, m)."""
    Y = data[target]

    def base_of(a, b):
        if base == 'm0':
            return m0_pred(data, a, b, use_cond) if target == 'Y' else _cond_mean(data, target, a, b, use_cond)
        if base == 'C1':
            return data['C1'][b]
        if base == 'M2':                                  # post-hoc (DEVIATIONS-P4 PH-1): no-fit persistence 9*u0
            return 9.0 * data['u0'][b]
        raise ValueError(base)

    def fit_eval(a, b, cfgs):
        mmax = max(m for _, m in cfgs)
        blk = Blocks(data, a, use_pca_full=use_full, pca_key=pca_key if mmax else None, z_keys=keys)
        ra = Y[a] - base_of(a, a)
        bb = base_of(a, b)
        Xa_all, Xb_all = blk.X(a, keys, mmax), blk.X(b, keys, mmax)   # features computed once, sliced per config
        nz = Xa_all.shape[1] - mmax
        outs = []
        for lam, m in cfgs:
            Xa, Xb = Xa_all[:, :nz + m], Xb_all[:, :nz + m]
            if Xa.shape[1] == 0:
                outs.append(bb)
                continue
            B = ridge_fit(Xa, ra, lam)
            outs.append(bb + Xb @ B)
        return outs

    # inner CV
    inner = group_folds(data['task'][tr], 3, 20261009 + int(data.get('_outer_fold', 0)))
    sse = np.zeros(len(configs))
    for f in range(3):
        a, b = tr[inner != f], tr[inner == f]
        preds = fit_eval(a, b, configs)
        sse += np.array([((Y[b] - p) ** 2).sum() for p in preds])
    best = int(np.argmin(sse))      # argmin returns the first minimum -> listed-order tie break
    pred = fit_eval(tr, ev, [configs[best]])[0]
    return pred, configs[best]


def _cond_mean(data, target, a, b, use_cond):
    key = 'cond' if use_cond else 'anchor'
    return cell_means(data[target][a], data[key][a], data[key][b], data[key][a], data[key][b], min_n=1)


# ------------------------------------------------------------------ part B: remaining update
def predict_remaining(data, tr, ev, use_cond=True):
    """Returns dict model -> (len(ev), 35) predictions and dict model -> chosen hyperparameters."""
    Y, u0 = data['Y'], data['u0']
    P, H = {}, {}
    P['Z'] = np.zeros((len(ev), Y.shape[1]))
    P['M0'] = m0_pred(data, tr, ev, use_cond)
    # M1 magnitude baseline
    yn = (Y[tr] ** 2).sum(1)
    ok = (yn > 0) & (data['R0'][tr] > 0)
    def design(idx):
        cols = [np.ones(len(idx)), np.log(data['R0'][idx])]
        levels = data['cond_levels'] if use_cond else []
        cols += [(data['cond'][idx] == c).astype(float) for c in levels[1:]]
        cols += [(data['anchor'][idx] == a).astype(float) for a in data['anchor_levels'][1:]]
        return np.column_stack(cols)
    beta = np.linalg.lstsq(design(tr[ok]), np.log(yn[ok]), rcond=None)[0]
    s_hat = np.exp(design(ev) @ beta / 2)
    e = P['M0']
    en = np.linalg.norm(e, axis=1, keepdims=True)
    P['M1'] = np.where(en > 0, e / np.where(en > 0, en, 1), 0) * s_hat[:, None]
    # M2 persistence and M2g gain persistence
    P['M2'] = 9.0 * u0[ev]
    if use_cond:
        bc = {c: (Y[tr][data['cond'][tr] == c] * u0[tr][data['cond'][tr] == c]).sum() /
                 max((u0[tr][data['cond'][tr] == c] ** 2).sum(), 1e-300) for c in np.unique(data['cond'][tr])}
        P['M2g'] = np.array([bc.get(c, 9.0) for c in data['cond'][ev]])[:, None] * u0[ev]
        H['M2g'] = {str(k): float(v) for k, v in bc.items()}
    else:
        b = (Y[tr] * u0[tr]).sum() / (u0[tr] ** 2).sum()
        P['M2g'] = b * u0[ev]
        H['M2g'] = float(b)
    # ridge models
    P['M2L'], H['M2L'] = ridge_family(data, tr, ev, ('u0',), [(l, 0) for l in LAM_M2L], use_cond)
    P['M3'], H['M3'] = ridge_family(data, tr, ev, ('u0', 'a0'), CFG_M3, use_cond)
    P['M3n'], H['M3n'] = ridge_family(data, tr, ev, ('a0',), CFG_M3, use_cond)
    P['M3f'], H['M3f'] = ridge_family(data, tr, ev, ('u0',), CFG_M3, use_cond, use_full=True)
    return P, H


def magnitude_r2_preds(data, tr, ev):
    """OOF predictions of log||Y||^2 from [log R0] / [log||u0||^2] / both, each plus cell dummies (training OLS)."""
    yl = np.log((data['Y'] ** 2).sum(1))
    lr, lu = np.log(data['R0']), np.log((data['u0'] ** 2).sum(1))
    cells = data['cell_levels']
    def D(idx, extra):
        return np.column_stack([np.ones(len(idx))] + extra + [(data['cell'][idx] == c).astype(float) for c in cells[1:]])
    out = {}
    for name, ex in dict(logR0=[lr], logU0=[lu], both=[lr, lu]).items():
        okt = np.isfinite(yl[tr]) & np.all([np.isfinite(e[tr]) for e in ex], 0)
        b = np.linalg.lstsq(D(tr[okt], [e[tr[okt]] for e in ex]), yl[tr[okt]], rcond=None)[0]
        out[name] = D(ev, [e[ev] for e in ex]) @ b
    return out


# ------------------------------------------------------------------ part C: cross-seed transfer
def predict_crossseed(cs, tr, ev):
    """cs: dict with D (target), C1, Dnorm_o, abar_o, nbar_o, cond, task, anchor ... Returns preds and hyperparameters."""
    P, H = {}, {}
    P['C1'] = cs['C1'][ev]
    mean_c = {c: cs['D'][tr][cs['cond'][tr] == c].mean(0) for c in np.unique(cs['cond'][tr])}
    e = np.array([mean_c[c] for c in cs['cond'][ev]])
    en = np.linalg.norm(e, axis=1, keepdims=True)
    P['C0'] = np.where(en > 0, e / np.where(en > 0, en, 1), 0) * cs['Dnorm_o'][ev][:, None]
    P['C2'], H['C2'] = ridge_family(cs, tr, ev, ('abar_o',), CFG_M3, True, pca_key='nbar_o', target='D')
    P['C3'], H['C3'] = ridge_family(cs, tr, ev, (), CFG_M3, True, pca_key='nbar_o', base='C1', target='D')
    return P, H


# ------------------------------------------------------------------ evaluation helpers
def task_sums(Y, P, task, cond=None):
    """Per-task (and optional per-condition) sums of squared error and target energy."""
    err = ((Y - P) ** 2).sum(1)
    tot = (Y ** 2).sum(1)
    keys = list(zip(task, cond)) if cond is not None else list(task)
    out = {}
    for k, e, t in zip(keys, err, tot):
        a = out.setdefault(k, [0.0, 0.0])
        a[0] += e
        a[1] += t
    return out


def eef_weighted(sums, weights, by_cond=False):
    if not by_cond:
        se = sum(weights.get(t, 0) * v[0] for t, v in sums.items())
        st = sum(weights.get(t, 0) * v[1] for t, v in sums.items())
        return 1 - se / st if st > 0 else np.nan
    conds = sorted({c for _, c in sums})
    vals = []
    for c in conds:
        se = sum(weights.get(t, 0) * v[0] for (t, cc), v in sums.items() if cc == c)
        st = sum(weights.get(t, 0) * v[1] for (t, cc), v in sums.items() if cc == c)
        if st > 0:
            vals.append(1 - se / st)
    return float(np.mean(vals))


def posthoc_M2R(data, tr, ev, use_cond=True):
    """POST-HOC (not preregistered; DEVIATIONS-P4 PH-1): readout null-space correction on top of the strongest
    preregistered baseline M2 = 9*u0. Same grid/budget and inner CV as M3."""
    return ridge_family(data, tr, ev, (), CFG_M3, use_cond, base='M2')


def posthoc_C1s(cs, tr, ev):
    """POST-HOC (DEVIATIONS-P4 PH-2): fitted command average M0_cond + ridge[z(C1)], same lambda grid as M2L."""
    return ridge_family(cs, tr, ev, ('C1',), [(l, 0) for l in LAM_M2L], True, target='D')
