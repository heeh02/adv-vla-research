"""P3 analysis exactly as frozen in reports/PREREG-P3.md (sha256 11733a3a...).
Inputs: data/p2/*.npz (dphys at 3 queries), event-locked package trajectories (eef per query, executed actions,
results) and data/review/object_episode.csv (primary object last-query offsets).
Outputs: data/p3_units.csv.gz, reports/P3-RESULTS.json

Revision after review (DEV-3/DEV-4, see reports/DEVIATIONS.md):
- D_sys sums the three preregistered anchors {q0, lm1, lm2} (P3_ANCHORS=prereg3, default);
  P3_ANCHORS=all reproduces the first-pass all-anchor sum as a sensitivity (outputs suffixed -allanchors).
- Mirror test primary on the 46 preregistered (object-covered) roots; all 72 roots reported as extension.
- D3 AUROCs on one common population, with task-cluster bootstrap CIs and seed split-half reliability.
- Pulse conditions have no q0 trace in D3, so the pulse descriptive is not computable and is dropped.
"""
import glob
import json
import os
import pickle

import numpy as np
import pandas as pd

from p2_analyze import PERSIST

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, '..', 'data')
R = os.path.join(HERE, '..', 'reports')
PKG = os.path.join(HERE, '..', '..', 'experiments', 'event-locked-placement-20261008-v1', 'data')
rng = np.random.default_rng(20261008)
LM2 = {'libero_spatial': 40, 'libero_object': 55, 'libero_goal': 60, 'libero_10': 100}
LM1Q = {'libero_spatial': 'q004', 'libero_object': 'q005', 'libero_goal': 'q006', 'libero_10': 'q010'}
LM2Q = {'libero_spatial': 'q008', 'libero_object': 'q011', 'libero_goal': 'q012', 'libero_10': 'q020'}
CAMERA = [c for c in PERSIST if 'camera' in c]
ANCHORS = os.environ.get('P3_ANCHORS', 'prereg3')
SUF = '' if ANCHORS == 'prereg3' else '-allanchors'


def cos(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return np.nan if na == 0 or nb == 0 else float(a @ b / na / nb)


def boot_median(vals, groups, B=2000):
    vals, groups = np.asarray(vals), np.asarray(groups)
    ok = np.isfinite(vals)
    vals, groups = vals[ok], groups[ok]
    if len(vals) == 0:
        return float('nan'), [float('nan'), float('nan')], 0
    ug = np.unique(groups)
    by = {g: vals[groups == g] for g in ug}
    meds = [np.median(np.concatenate([by[g] for g in rng.choice(ug, len(ug))])) for _ in range(B)]
    return float(np.median(vals)), np.percentile(meds, [2.5, 97.5]).tolist(), int(len(vals))


def main():
    # ---- per-query same-state command deviations (physical translation, sum of first 5 executed steps)
    rows = []
    for f in sorted(glob.glob(os.path.join(D, 'p2', 'shard-*.npz'))):
        z = np.load(f)
        m = pd.DataFrame(z['meta'])[['root', 'noise', 'cond', 'q']]
        d = z['dphys'][:, :5, :3].astype(np.float64).sum(1)
        m[['dx_cmd', 'dy_cmd', 'dz_cmd']] = d
        rows.append(m)
    q = pd.concat(rows, ignore_index=True)
    suite_q = q.root.str.split('-').str[0]
    q['anchor'] = np.select([q.q == 'q000', q.q == suite_q.map(LM1Q), q.q == suite_q.map(LM2Q)], ['q0', 'lm1', 'lm2'], 'later')
    if ANCHORS == 'prereg3':
        q = q[q.anchor != 'later']
    # ---- trajectories
    eef, execs, results = {}, {}, {}
    for f in sorted(glob.glob(os.path.join(PKG, 'derived-trajectories', 'shard-*.pkl'))):
        o = pickle.load(open(f, 'rb'))
        for ep, st in o['state'].items():
            eef[(ep[0], int(ep[1]), ep[2])] = {d['step']: d['eef'] for d in st}
        for ep, a in o['exec'].items():
            execs[(ep[0], int(ep[1]), ep[2])] = a
        for ep, r in o['result'].items():
            results[(ep[0], int(ep[1]), ep[2])] = r
    res = {}
    # ---- QC-P3a: clean episodes, cos(sum executed translation between queries, eef displacement)
    cs = []
    for k, st in eef.items():
        if k[2] != 'clean' or k not in execs:
            continue
        steps = sorted(st)
        a = execs[k]
        for s0, s1 in zip(steps[:-1], steps[1:]):
            if s1 <= len(a):
                cs.append(cos(a[s0:s1, :3].sum(0), st[s1] - st[s0]))
    cs = np.array([c for c in cs if np.isfinite(c)])
    res['QC_P3a_cos_median'] = float(np.median(cs))
    res['QC_P3a_pass'] = bool(np.median(cs) > 0.8)
    # ---- objects
    ob = pd.read_csv(os.path.join(PKG, 'review', 'object_episode.csv'))
    ob = ob[ob.primary]
    # floor: unit-level clean re-sampling of last-query offsets
    cl = ob[(ob.cond == 'clean') & (ob.success == True)]
    resid = {r: g[['final_x', 'final_y']].values for r, g in cl.groupby('root')}
    floor_draws = []
    for _ in range(400):
        for r, v in resid.items():
            if len(v) < 2:
                continue
            c3 = v[rng.integers(0, len(v), 3)].mean(0)
            ref = np.median(v[rng.integers(0, len(v), 3)], 0)
            floor_draws.append(np.hypot(*(c3 - ref)))
    floor95 = float(np.percentile(floor_draws, 95))
    res['floor_unit_lastquery_mm'] = dict(median=float(np.median(floor_draws) * 1e3), p95=floor95 * 1e3)
    # ---- units root x condition
    units = []
    for (root, cond), g in q[q.cond.isin(PERSIST)].groupby(['root', 'cond']):
        suite = root.split('-')[0]
        per_seed = g.groupby('noise')[['dx_cmd', 'dy_cmd']].sum()
        Dsys = per_seed.mean().values
        slot_sys = {s: g[g.anchor == a].groupby('noise')[['dx_cmd', 'dy_cmd']].sum().mean().values for s, a in enumerate(['q0', 'lm1', 'lm2'])}
        o = ob[(ob.root == root) & (ob.cond == cond) & (ob.success == True)]
        Osys = o[['dx', 'dy']].mean().values if len(o) >= 2 else np.array([np.nan, np.nan])
        # EEF closed-loop deviation after landmark 2
        es = []
        for n in range(3):
            kp, kc = (root, n, cond), (root, n, 'clean')
            if kp in eef and kc in eef:
                t = LM2[suite] + 5
                if t in eef[kp] and t in eef[kc]:
                    es.append((eef[kp][t] - eef[kc][t])[:2])
        Esys = np.mean(es, 0) if len(es) >= 2 else np.array([np.nan, np.nan])
        units.append(dict(root=root, task_id=root.rsplit('-', 1)[0], cond=cond, n_seed_cmd=len(per_seed), n_succ=len(o),
                          n_anchor_per_seed=float(g.groupby('noise').size().mean()),
                          **{f'Dseed{n}_{ax}': (per_seed.loc[n, f'd{ax}_cmd'] if n in per_seed.index else np.nan) for n in range(3) for ax in 'xy'},
                          D_x=Dsys[0], D_y=Dsys[1], O_x=Osys[0], O_y=Osys[1], E_x=Esys[0], E_y=Esys[1],
                          D_norm=float(np.hypot(*Dsys)), O_norm=float(np.hypot(*Osys)) if np.all(np.isfinite(Osys)) else np.nan,
                          cos_DO=cos(Dsys, Osys) if np.all(np.isfinite(Osys)) else np.nan,
                          cos_DE=cos(Dsys, Esys) if np.all(np.isfinite(Esys)) else np.nan,
                          cos_EO=cos(Esys, Osys) if np.all(np.isfinite(Esys)) and np.all(np.isfinite(Osys)) else np.nan,
                          **{f'cos_slot{s}_O': (cos(slot_sys[s], Osys) if np.all(np.isfinite(Osys)) and np.all(np.isfinite(slot_sys[s])) else np.nan)
                             for s in range(3)}))
    U = pd.DataFrame(units)
    U.to_csv(os.path.join(D, f'p3_units{SUF.replace("-", "_")}.csv.gz'), index=False)
    res['anchors'] = ANCHORS
    res['anchors_per_seed_mean'] = float(U.n_anchor_per_seed.mean())
    res['QC_P3b'] = dict(units=int(len(U)), units_with_O=int(U.O_norm.notna().sum()),
                         camera_units_above_floor=int(((U.cond.isin(CAMERA)) & (U.O_norm > floor95)).sum()))
    # ---- D1 direction chain (camera, above floor)
    sel = U[U.cond.isin(CAMERA) & (U.O_norm > floor95)]
    med, ci, n = boot_median(sel.cos_DO, sel.task_id)
    res['D1'] = dict(cos_DO_median=med, ci=ci, n_units=n, direction_predicts_offset=bool(med > 0 and ci[0] > 0),
                     n_roots=int(sel.root.nunique()), n_tasks=int(sel.task_id.nunique()),
                     units_by_condition=sel.cond.value_counts().to_dict(),
                     n_camera_units_with_O=int(U[U.cond.isin(CAMERA)].O_norm.notna().sum()),
                     note='O_sys uses successful seeds only (>=2); association within selected successful episodes, not a failure explanation',
                     cos_DE=boot_median(sel.cos_DE, sel.task_id), cos_EO=boot_median(sel.cos_EO, sel.task_id))
    s106 = U[U.cond.isin(CAMERA) & (U.O_norm > 0.0106)]
    m106, ci106, n106 = boot_median(s106.cos_DO, s106.task_id)
    res['D1_sensitivity_floor_10.6mm'] = dict(cos_DO_median=m106, ci=ci106, n_units=n106, n_tasks=int(s106.task_id.nunique()))
    # ---- D2 mirror test
    mir = {}
    obj_roots = set(ob.root.unique())
    for pop, (a, b) in [(p, ab) for p in ('prereg46', 'all72') for ab in [('camera_plus15', 'camera_minus15'), ('camera_plus30', 'camera_minus30')]]:
        A = U[U.cond == a].set_index('root')
        Bm = U[U.cond == b].set_index('root')
        common = A.index.intersection(Bm.index)
        if pop == 'prereg46':
            common = common[common.isin(obj_roots)]
        cs_ = [cos(A.loc[r, ['D_x', 'D_y']].values.astype(float), Bm.loc[r, ['D_x', 'D_y']].values.astype(float)) for r in common]
        tids = [r.rsplit('-', 1)[0] for r in common]
        m_, ci_, n_ = boot_median(cs_, tids)
        mir[pop + ':' + a + '_vs_' + b] = dict(cos_median=m_, ci=ci_, n=n_,
                                   verdict=('mirror(opposite)' if ci_[1] < 0 else 'same-direction' if ci_[0] > 0 else 'unresolved'))
    res['D2_mirror'] = mir
    # ---- D3 outcome (secondary): clean-success pairs, harm vs predictors, within-condition AUROC
    from sklearn.metrics import roc_auc_score
    rows = []
    for (root, cond), g in q[q.cond.isin(PERSIST)].groupby(['root', 'cond']):
        for n in range(3):
            kp, kc = (root, n, cond), (root, n, 'clean')
            if kp not in results or kc not in results or not results[kc].get('success'):
                continue
            o_other = ob[(ob.root == root) & (ob.cond == cond) & (ob.success == True) & (ob.noise != n)]
            Oloo = np.hypot(o_other.dx.mean(), o_other.dy.mean()) if len(o_other) >= 1 else np.nan
            u = U[(U.root == root) & (U.cond == cond)]
            # post-hoc (review round 2): leave-one-seed-out command magnitude, same information condition as O_loo
            dl = np.array([[u[f'Dseed{m}_x'].iloc[0], u[f'Dseed{m}_y'].iloc[0]] for m in range(3) if m != n]) if len(u) else np.empty((0, 2))
            dl = dl[np.all(np.isfinite(dl), axis=1)]
            rows.append(dict(root=root, task_id=root.rsplit('-', 1)[0], cond=cond, noise=n, harm=int(not results[kp].get('success')),
                             D_norm=float(u.D_norm.iloc[0]) if len(u) else np.nan, O_loo=Oloo,
                             D_loo=float(np.hypot(*dl.mean(0))) if len(dl) else np.nan, n_O_other=int(len(o_other))))
    H = pd.DataFrame(rows)
    H.to_csv(os.path.join(D, f'p3_harm{SUF.replace("-", "_")}.csv.gz'), index=False)

    def auc_wc(Hs, x):
        num = den = 0.0
        for c, g in Hs.groupby('cond'):
            if g.harm.nunique() < 2:
                continue
            npair = g.harm.sum() * (len(g) - g.harm.sum())
            num += roc_auc_score(g.harm, g[x]) * npair
            den += npair
        return float(num / den) if den else np.nan

    out = {}
    for popname, Hs in [('own_population', None), ('common_population', H.dropna(subset=['D_norm', 'O_loo']))]:
        for x in ['D_norm', 'O_loo']:
            Hx = H.dropna(subset=[x]) if Hs is None else Hs
            tids = Hx.task_id.unique()
            byt = {t: Hx[Hx.task_id == t] for t in tids}
            bs = [auc_wc(pd.concat([byt[t] for t in rng.choice(tids, len(tids))]), x) for _ in range(1000)]
            out[f'{popname}:{x}'] = dict(auc_within_condition=auc_wc(Hx, x), ci=np.nanpercentile(bs, [2.5, 97.5]).tolist(),
                                         n=int(len(Hx)), harms=int(Hx.harm.sum()), n_tasks=int(len(tids)))
    # cross-seed rank correlation of SINGLE-seed magnitudes (within condition, Spearman). This is NOT the reliability of the
    # actual predictors ||mean_i D_i|| and O_loo (vector averaging can cancel; O has success filtering and a shared clean
    # reference), so no Spearman-Brown step-up is reported (review round 2).
    rel = {}
    Dn = {n: np.hypot(U[f'Dseed{n}_x'], U[f'Dseed{n}_y']) for n in range(3)}
    rs = []
    for i, j in [(0, 1), (0, 2), (1, 2)]:
        for c in PERSIST:
            m = (U.cond == c) & Dn[i].notna() & Dn[j].notna()
            rs.append(pd.Series(Dn[i][m].values).corr(pd.Series(Dn[j][m].values), method='spearman'))
    r1 = float(np.nanmean(rs))
    rel['D_single_seed_norm'] = dict(mean_cross_seed_spearman=r1)
    rs = []
    so = ob[ob.success == True].assign(on=lambda d: np.hypot(d.dx, d.dy)).pivot_table(index=['root', 'cond'], columns='noise', values='on')
    for i, j in [(0, 1), (0, 2), (1, 2)]:
        for c in PERSIST:
            sub = so.xs(c, level='cond').dropna(subset=[i, j]) if c in so.index.get_level_values('cond') else None
            if sub is not None and len(sub) >= 5:
                rs.append(sub[i].corr(sub[j], method='spearman'))
    r1 = float(np.nanmean(rs)) if rs else np.nan
    rel['O_single_seed_norm'] = dict(mean_cross_seed_spearman=r1)
    out['cross_seed_rank_corr_single_seed_magnitudes'] = rel
    res['D3_outcome'] = out
    # ---- descriptive
    res['phase_cos_with_O_camera_above_floor'] = {a: boot_median(sel[f'cos_slot{s}_O'], sel.task_id) for s, a in enumerate(['q0', 'lm1', 'lm2'])}
    nonc = U[U.cond.isin(['headlight_off', 'matched_global', 'surface_cool', 'surface_warm'])]
    res['noncamera_cos_DO'] = boot_median(nonc.cos_DO, nonc.task_id)
    res['pulse_cos_DO'] = 'not computable: pulse conditions have no q0 trace in D3'
    json.dump(res, open(os.path.join(R, f'P3-RESULTS{SUF}.json'), 'w'), indent=1, default=float)
    print(json.dumps(res, indent=1, default=float))


if __name__ == '__main__':
    main()
