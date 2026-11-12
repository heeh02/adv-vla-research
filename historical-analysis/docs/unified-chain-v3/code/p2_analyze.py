"""P2 analysis exactly as frozen in reports/PREREG-P2.md (sha256 028a4c2a...).

Primary (DEV-1): u = u_xt (exact f32 x_t increments), space E = positions 0..4 x dims 0..6.
Secondary/sensitivity: space F (positions 0..9), and u_vel (stored bf16 velocity x dt; the frozen prereg's original primary).
Outputs: data/p2_pairs.csv.gz, reports/P2-RESULTS.json
"""
import glob
import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, '..', 'data')
R = os.path.join(HERE, '..', 'reports')
rng = np.random.default_rng(20261008)
PERSIST = ['camera_minus15', 'camera_minus15_surface_cool', 'camera_minus30', 'camera_plus15', 'camera_plus15_surface_warm',
           'camera_plus30', 'headlight_off', 'matched_global', 'surface_cool', 'surface_warm']
PULSE = ['camera_plus15_q0_only', 'headlight_off_q0_only', 'surface_warm_q0_only']


def load():
    metas, arrs = [], {}
    for f in sorted(glob.glob(os.path.join(D, 'p2', 'shard-*.npz'))):
        z = np.load(f)
        m = pd.DataFrame(z['meta'])
        m['shard'] = os.path.basename(f)[:-4]
        metas.append(m)
        for k in ['u_vel', 'u_xt', 'dcand', 'dphys', 'shared', 'free3', 'verr']:
            arrs.setdefault(k, []).append(z[k])
    meta = pd.concat(metas, ignore_index=True)
    arrs = {k: np.concatenate(v) for k, v in arrs.items()}
    return meta, arrs


def metrics(u, pos):
    """u: (N,10 steps,10 pos,7). Returns dict of per-pair metrics in the given position slice."""
    v = u[:, :, pos, :].reshape(u.shape[0], u.shape[1], -1).astype(np.float64)
    step_norm = np.linalg.norm(v, axis=2)                      # (N,10)
    S = (step_norm ** 2).sum(1)
    sumnorm = step_norm.sum(1)
    dx = v.sum(1)
    F = (dx ** 2).sum(1)
    with np.errstate(divide='ignore', invalid='ignore'):
        Keff = sumnorm ** 2 / S
        c = np.sqrt(F) / sumnorm
        share = step_norm ** 2 / S[:, None]
        cosk = (v * dx[:, None, :]).sum(2) / (step_norm * np.linalg.norm(dx, axis=1)[:, None])
    return dict(S=S, F=F, Keff=Keff, c=c, share=share, cosk=cosk, dx=dx)


def level_components(df, y):
    """Sequential projections: condition, task|condition, condition x task | main effects, within."""
    X0 = np.ones((len(df), 1))
    C = pd.get_dummies(df.cond, drop_first=True).astype(float).values
    T = pd.get_dummies(df.task_id, drop_first=True).astype(float).values
    cell = pd.get_dummies(df.cond + '|' + df.task_id, drop_first=True).astype(float).values

    def proj(X, yv):
        b = np.linalg.lstsq(X, yv, rcond=None)[0]
        return X @ b
    yc = y - y.mean()
    p1 = proj(np.hstack([X0, C]), y) - y.mean()
    p2 = proj(np.hstack([X0, C, T]), y) - y.mean()
    p3 = proj(np.hstack([X0, cell]), y) - y.mean()
    return {'condition': p1, 'task': p2 - p1, 'cond_x_task': p3 - p2, 'within': yc - p3}


def decompose(df):
    out = {}
    comps = {k: level_components(df, df[k].values) for k in ['logS', 'logK', 'logc2']}
    for L in ['condition', 'task', 'cond_x_task', 'within']:
        a, b, c = comps['logS'][L], comps['logK'][L], comps['logc2'][L]
        f = a + b + c
        va, vb, vc = a.var(), b.var(), c.var()
        out[L] = dict(var_logF=float(f.var()), var_logS=float(va), var_logK=float(vb), var_logc2=float(vc),
                      cov2_S_K=float(2 * np.cov(a, b, bias=True)[0, 1]), cov2_S_c=float(2 * np.cov(a, c, bias=True)[0, 1]),
                      cov2_K_c=float(2 * np.cov(b, c, bias=True)[0, 1]),
                      D2_share=float((vb + vc) / (va + vb + vc)) if (va + vb + vc) > 0 else None)
    return out


def main():
    meta, A = load()
    meta['task_id'] = meta.root.str.rsplit('-', n=1).str[0]
    meta['suite'] = meta.root.str.split('-').str[0]
    meta['slot'] = meta.groupby(['root', 'noise', 'cond']).q.rank(method='dense').astype(int) - 1
    res = {'n_traces': int(len(meta)), 'n_episodes': int(meta.groupby(['root', 'noise', 'cond']).ngroups)}
    E, Fsp = slice(0, 5), slice(0, 10)
    PRIMARY = os.environ.get('P2_PRIMARY', 'u_xt')  # DEV-1: u_xt primary, u_vel sensitivity
    SENS = 'u_vel' if PRIMARY == 'u_xt' else 'u_xt'
    SUF = '' if PRIMARY == 'u_xt' else '-' + PRIMARY   # velocity-version rerun writes to separate files
    mE = metrics(A[PRIMARY], E)
    mF = metrics(A[PRIMARY], Fsp)
    mX = metrics(A[SENS], E)
    res['primary_source'] = PRIMARY
    df = meta.copy()
    for k in ['S', 'F', 'Keff', 'c']:
        df[k] = mE[k]
        df[k + '_F'] = mF[k]
        df[k + '_xt'] = mX[k]
    dcE = A['dcand'][:, E, :].reshape(len(df), -1).astype(np.float64)
    with np.errstate(divide='ignore', invalid='ignore'):
        mV = metrics(A['u_vel'], E); mXT = metrics(A['u_xt'], E)
        df['id_resid'] = np.linalg.norm(mV['dx'] - dcE, axis=1) / np.linalg.norm(dcE, axis=1)
        df['id_resid_xt'] = np.linalg.norm(mXT['dx'] - dcE, axis=1) / np.linalg.norm(dcE, axis=1)
    # counterfactual ratio natural/shared at taus 0,5,9 (E space)
    fr = np.linalg.norm(A['free3'][:, :, E, :].reshape(len(df), 3, -1), axis=2)
    sh = np.linalg.norm(A['shared'][:, :, E, :].reshape(len(df), 3, -1), axis=2)
    with np.errstate(divide='ignore', invalid='ignore'):
        for j, t in enumerate([0, 5, 9]):
            df[f'ratio_nat_shared_t{t}'] = fr[:, j] / sh[:, j]
    df['verr_actual_mean'] = A['verr'][:, 0].mean(1)
    df['dphys_E_norm'] = np.linalg.norm(A['dphys'][:, E, :].reshape(len(df), -1), axis=1)
    for k in range(10):
        df[f'share_{k}'] = mE['share'][:, k]
        df[f'cos_{k}'] = mE['cosk'][:, k]
    df.to_csv(os.path.join(D, f'p2_pairs{SUF.replace("-", "_")}.csv.gz'), index=False)

    # ---------- QC ----------
    pers = df[df.cond.isin(PERSIST)]
    qc = {}
    qc['QC1_x0_equal_frac'] = float(df.x0eq.mean())
    qc['QC1_same_time_frac'] = float(df.same_time.mean())
    good = pers[np.isfinite(pers.id_resid)]
    qc['QC2_identity_resid_median'] = float(good.id_resid.median())
    qc['QC2_identity_resid_p99'] = float(good.id_resid.quantile(.99))
    qc['QC2_xt_resid_median'] = float(good.id_resid_xt.median())
    qc['QC2_pass'] = bool(qc['QC2_identity_resid_median'] < 0.01 and qc['QC2_identity_resid_p99'] < 0.05)
    lm = df[df.slot > 0]
    pulse_lm = lm[lm.cond.isin(PULSE)].F
    pers_lm = lm[lm.cond.isin(PERSIST)].F
    qc['QC3_pulse_landmark_rel'] = float(np.sqrt(pulse_lm.median()) / np.sqrt(pers_lm.median())) if len(pulse_lm) else None
    qc['QC3_pass'] = bool(qc['QC3_pulse_landmark_rel'] is not None and qc['QC3_pulse_landmark_rel'] < 0.05)
    qc['QC4_velocity_shadow_rms_error_median'] = float(df.verr_actual_mean.median())
    clean_rows = df[df.cond == 'clean']
    qc['clean_rows'] = int(len(clean_rows))
    qc['clean_rows_F_max'] = float(clean_rows.F.max()) if len(clean_rows) else None
    res['QC'] = qc

    # ---------- primary analysis ----------
    pa = pers[(pers.F > 0) & (pers.S > 0)].copy()
    pa['logF'] = np.log(pa.F)
    pa['logS'] = np.log(pa.S)
    pa['logK'] = np.log(pa.Keff)
    pa['logc2'] = 2 * np.log(pa.c)
    res['n_primary_pairs'] = int(len(pa))
    res['n_excluded_zero'] = int(len(pers) - len(pa))
    res['decomposition'] = {}
    for name, sub in [('pooled', pa)] + [(f'slot{s}', pa[pa.slot == s]) for s in range(3)]:
        dec = decompose(sub)
        loso = {}
        for su in sorted(sub.suite.unique()):
            d2 = decompose(sub[sub.suite != su])
            loso[su] = {L: d2[L]['D2_share'] for L in d2}
        crit = {}
        for L in dec:
            full = dec[L]['D2_share']
            dev = max(abs(loso[s][L] - full) for s in loso)
            crit[L] = dict(D2_share=full, max_loso_dev=float(dev), D2_independent_factor=bool(full >= 0.25 and dev <= 0.10))
        res['decomposition'][name] = dict(levels=dec, leave_one_suite_out=loso, D2=crit)
    # D3: systematic cancellation via task-cluster bootstrap of median c
    tasks = pa.task_id.unique()
    by = {t: pa[pa.task_id == t].c.values for t in tasks}
    meds = [np.median(np.concatenate([by[t] for t in rng.choice(tasks, len(tasks))])) for _ in range(2000)]
    res['D3'] = dict(c_median=float(pa.c.median()), ci=np.percentile(meds, [2.5, 97.5]).tolist(),
                     systematic_cancellation=bool(np.percentile(meds, 97.5) < 0.9),
                     c_quantiles=pa.c.quantile([.05, .25, .5, .75, .95]).round(4).to_dict(),
                     by_condition=pa.groupby('cond').c.median().round(4).to_dict(),
                     by_slot=pa.groupby('slot').c.median().round(4).to_dict())
    # descriptive
    desc = pa.groupby('cond').agg(n=('F', 'size'), F_med=('F', 'median'), S_med=('S', 'median'), Keff_med=('Keff', 'median'),
                                  c_med=('c', 'median')).round(6)
    res['descriptive_by_condition'] = desc.to_dict(orient='index')
    res['sensitivity'] = dict(
        c_median_spaceF=float(pa.c_F.median()), c_median_xt=float(pa.c_xt.median()),
        corr_c_E_vs_xt=float(np.corrcoef(pa.c, pa.c_xt)[0, 1]))
    # descriptive (added after review): per-step energy share, local cancellation, covariance-inclusive shares
    shp = pa[[f'share_{k}' for k in range(10)]].values
    res['step_share'] = {q: np.percentile(shp, q, 0).round(4).tolist() for q in (25, 50, 75)}
    res['frac_c_below'] = {t: float((pa.c < t).mean()) for t in (0.9, 0.7, 0.5)}
    res['frac_c_below_0.9_by_condition'] = (pa.c < 0.9).groupby(pa.cond).mean().round(4).to_dict()
    res['nonS_share_incl_cov'] = {L: float(1 - v['var_logS'] / v['var_logF']) for L, v in res['decomposition']['pooled']['levels'].items()}
    # split of that remainder (review round 2): A = logK + 2logc. Var(A)/Var(logF) is the accumulation term alone;
    # 2Cov(logS, A)/Var(logF) is co-variation of magnitude and accumulation (not an independent explained share).
    res['identity_remainder_terms'] = {L: dict(varA_over_varF=float((v['var_logK'] + v['var_logc2'] + v['cov2_K_c']) / v['var_logF']),
                                               cov2_S_A_over_varF=float((v['cov2_S_K'] + v['cov2_S_c']) / v['var_logF']))
                                       for L, v in res['decomposition']['pooled']['levels'].items()}
    rcols = [f'ratio_nat_shared_t{t}' for t in [0, 5, 9]]
    res['counterfactual_ratio_median'] = {c: float(pa[c].replace([np.inf, -np.inf], np.nan).median()) for c in rcols}
    json.dump(res, open(os.path.join(R, f'P2-RESULTS{SUF}.json'), 'w'), indent=1, default=float)
    print(json.dumps(res, indent=1, default=float)[:6000])


if __name__ == '__main__':
    main()
