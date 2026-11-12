"""P1 analysis exactly as frozen in reports/PREREG-P1.md (sha256 a084cde3...).
Requires data/p2/*.npz, data/p1/*.npz and data/p2_pairs.csv.gz (from p2_analyze.py).
Outputs data/p1_pairs.csv.gz, reports/P1-RESULTS.json
"""
import glob
import json
import os

import numpy as np
import pandas as pd

from p2_analyze import PERSIST, level_components

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, '..', 'data')
R = os.path.join(HERE, '..', 'reports')
rng = np.random.default_rng(20261008)
SRC = os.environ.get('P1_SOURCE', 'u_xt')   # DEV-1 primary u_xt; 'u_vel' = stored-velocity sensitivity rerun
SUF = '' if SRC == 'u_xt' else '-' + SRC
FAMILY = {c: ('camera' if 'camera' in c else 'lighting' if c in ('headlight_off', 'matched_global') else 'surface') for c in PERSIST}


def load():
    rows = []
    for f in sorted(glob.glob(os.path.join(D, 'p2', 'shard-*.npz'))):
        z = np.load(f)
        m = pd.DataFrame(z['meta'])[['root', 'noise', 'cond', 'q']]
        V = 100.0 * (z[SRC][:, :, :5, :].astype(np.float64) ** 2).sum((2, 3))    # (N,10) velocity energy, E space (DEV-1: exact x_t increments)
        m[[f'V_{k}' for k in range(10)]] = V
        rows.append(m)
    v = pd.concat(rows, ignore_index=True)
    rr = []
    for f in sorted(glob.glob(os.path.join(D, 'p1', 'shard-*.npz'))):
        z = np.load(f)
        m = pd.DataFrame(z['meta'])
        m[[f'R_{k}' for k in range(10)]] = z['R'].astype(np.float64)
        rr.append(m)
    r = pd.concat(rr, ignore_index=True)
    pairs = pd.read_csv(os.path.join(D, f'p2_pairs{SUF.replace("-", "_")}.csv.gz'))
    df = pairs.merge(v, on=['root', 'noise', 'cond', 'q']).merge(r, on=['root', 'noise', 'cond', 'q'])
    return df


def decomp_factors(df, cols):
    comps = {c: level_components(df, df[c].values) for c in cols}
    out = {}
    for L in ['condition', 'task', 'cond_x_task', 'within']:
        M = np.stack([comps[c][L] for c in cols])
        cov = np.cov(M, bias=True)
        out[L] = dict(total_var=float(M.sum(0).var()), var={c: float(cov[i, i]) for i, c in enumerate(cols)},
                      cov2={f'{cols[i]}|{cols[j]}': float(2 * cov[i, j]) for i in range(len(cols)) for j in range(i + 1, len(cols))})
    return out


def ols_predict(Xtr, ytr, Xte):
    Xtr = np.column_stack([np.ones(len(Xtr)), Xtr])
    Xte = np.column_stack([np.ones(len(Xte)), Xte])
    b = np.linalg.lstsq(Xtr, ytr, rcond=None)[0]
    return Xte @ b


def main():
    df = load()
    df = df[df.cond.isin(PERSIST) & (df.F > 0) & (df.S > 0)].copy()
    Rk = df[[f'R_{k}' for k in range(10)]].values
    Vk = df[[f'V_{k}' for k in range(10)]].values
    df['Rtot'] = Rk.sum(1)
    df['Teff'] = Vk.sum(1) / Rk.sum(1)
    df['P'] = df.dphys_E_norm ** 2
    df['alpha'] = df.P / df.F
    df = df[(df.Rtot > 0) & (df.P > 0)].copy()
    for c in ['Rtot', 'Teff', 'Keff', 'alpha', 'S', 'P']:
        df['log' + c] = np.log(df[c])
    df['logc2'] = 2 * np.log(df.c)
    df['family'] = df.cond.map(FAMILY)
    res = {'n_pairs': int(len(df))}
    S_recon = 0.01 * df.Rtot * df.Teff
    res['QC_a_rel_diff_median'] = float((np.abs(S_recon - df.S) / df.S).median())
    res['alpha_quantiles'] = df.alpha.quantile([.05, .5, .95]).round(4).to_dict()
    res['Teff_quantiles'] = df.Teff.quantile([.05, .25, .5, .75, .95]).round(4).to_dict()
    res['Teff_IQR_by_condition'] = df.groupby('cond').Teff.quantile([.25, .5, .75]).unstack().round(4).to_dict(orient='index')
    # D1: two-factor decomposition of log S
    res['D1'] = {}
    for name, sub in [('pooled', df)] + [(f'slot{s}', df[df.slot == s]) for s in range(3)]:
        dec = decomp_factors(sub, ['logRtot', 'logTeff'])
        loso = {su: decomp_factors(sub[sub.suite != su], ['logRtot', 'logTeff']) for su in sorted(sub.suite.unique())}
        crit = {}
        for L in dec:
            share = lambda d: d[L]['var']['logTeff'] / (d[L]['var']['logRtot'] + d[L]['var']['logTeff'])
            full = share(dec)
            dev = max(abs(share(loso[s]) - full) for s in loso)
            crit[L] = dict(Teff_share=float(full), max_loso_dev=float(dev), D1_independent=bool(full >= 0.25 and dev <= 0.10))
        res['D1'][name] = dict(levels=dec, criterion=crit)
    # full chain decomposition (descriptive)
    res['chain_decomposition_pooled'] = decomp_factors(df, ['logRtot', 'logTeff', 'logKeff', 'logc2', 'logalpha'])
    # D2: task-grouped 5-fold CV
    tasks = np.array(sorted(df.task_id.unique()))
    perm = rng.permutation(len(tasks))
    fold_of = {tasks[i]: k % 5 for k, i in enumerate(perm)}
    df['fold'] = df.task_id.map(fold_of)
    y = df.logP.values
    pred = {m: np.zeros(len(df)) for m in ['A', 'B', 'C']}
    for k in range(5):
        tr, te = df.fold != k, df.fold == k
        dtr, dte = df[tr], df[te]
        pred['A'][te.values] = ols_predict(dtr[['logRtot']].values, dtr.logP.values, dte[['logRtot']].values)
        prof = dtr.groupby('cond')[['logTeff', 'logc2', 'logKeff']].mean()
        Xtr = np.column_stack([dtr.logRtot, prof.loc[dtr.cond].values])
        Xte = np.column_stack([dte.logRtot, prof.loc[dte.cond].values])
        pred['B'][te.values] = ols_predict(Xtr, dtr.logP.values, Xte)
        Ctr = pd.get_dummies(dtr.cond).reindex(columns=PERSIST, fill_value=0).astype(float).values[:, 1:]
        Cte = pd.get_dummies(dte.cond).reindex(columns=PERSIST, fill_value=0).astype(float).values[:, 1:]
        pred['C'][te.values] = ols_predict(np.column_stack([dtr.logRtot, Ctr]), dtr.logP.values, np.column_stack([dte.logRtot, Cte]))
    r2 = lambda p, yy: 1 - ((yy - p) ** 2).sum() / ((yy - yy.mean()) ** 2).sum()
    res['D2'] = {m: float(r2(pred[m], y)) for m in pred}
    by = {t: np.where(df.task_id.values == t)[0] for t in tasks}
    diffs = []
    for _ in range(1000):
        idx = np.concatenate([by[t] for t in rng.choice(tasks, len(tasks))])
        diffs.append(r2(pred['B'][idx], y[idx]) - r2(pred['A'][idx], y[idx]))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    res['D2']['B_minus_A'] = float(res['D2']['B'] - res['D2']['A'])
    res['D2']['B_minus_A_ci'] = [float(lo), float(hi)]
    res['D2']['transferable'] = bool(res['D2']['B_minus_A'] >= 0.05 and lo > 0)
    # D3 descriptive: leave-one-family-out; profile of unseen family predicted from condition-mean logRtot
    lofo = {}
    for fam in ['camera', 'lighting', 'surface']:
        tr, te = df.family != fam, df.family == fam
        dtr, dte = df[tr], df[te]
        pa = ols_predict(dtr[['logRtot']].values, dtr.logP.values, dte[['logRtot']].values)
        cm = dtr.groupby('cond').agg(mR=('logRtot', 'mean'), mT=('logTeff', 'mean'), mc=('logc2', 'mean'), mK=('logKeff', 'mean'))
        cm_te = dte.groupby('cond').logRtot.mean()
        prof_te = pd.DataFrame({c: ols_predict(cm[['mR']].values, cm[c].values, cm_te.values[:, None]) for c in ['mT', 'mc', 'mK']},
                               index=cm_te.index)
        prof_tr = cm[['mT', 'mc', 'mK']]
        pb = ols_predict(np.column_stack([dtr.logRtot, prof_tr.loc[dtr.cond].values]), dtr.logP.values,
                         np.column_stack([dte.logRtot, prof_te.loc[dte.cond].values]))
        lofo[fam] = dict(n=int(te.sum()), R2_A=float(r2(pa, dte.logP.values)), R2_Bprime=float(r2(pb, dte.logP.values)))
    res['D3_leave_family_out'] = lofo
    # descriptive
    tk = (df[[f'V_{k}' for k in range(10)]].values / df[[f'R_{k}' for k in range(10)]].values)
    res['Tk_median_by_step'] = np.nanmedian(tk, 0).round(6).tolist()
    res['by_condition'] = df.groupby('cond').agg(Rtot=('Rtot', 'median'), Teff=('Teff', 'median'), Keff=('Keff', 'median'),
                                                 c=('c', 'median'), alpha=('alpha', 'median'), P=('P', 'median')).round(6).to_dict(orient='index')
    df.to_csv(os.path.join(D, f'p1_pairs{SUF.replace("-", "_")}.csv.gz'), index=False)
    json.dump(res, open(os.path.join(R, f'P1-RESULTS{SUF}.json'), 'w'), indent=1, default=float)
    print(json.dumps(res, indent=1, default=float)[:8000])


if __name__ == '__main__':
    main()
