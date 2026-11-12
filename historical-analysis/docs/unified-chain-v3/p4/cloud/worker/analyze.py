"""P4 analysis worker (Baidu CPU), exactly as frozen in reports/PREREG-P4.md (sha256 dbab0f6d...).

Reads the per-shard feature npz written by the extraction jobs (BOS), runs
  A  the exact per-pair gain decomposition T = rho * g * nu (first/last log changes, condition-balanced aggregation),
  B  the cutoff-after-first-update prediction of the remaining nine-step update (task-held-out + leave-family-out),
  C  the q0 cross-seed transfer test (and later anchors reported separately),
and writes small result tables (JSON / csv.gz) plus full prediction vectors (npz, BOS only).
"""
import argparse
import csv
import gzip
import io
import json
import os
import sys
import time
import traceback

import numpy as np

from cloudio import BOS, SafeError, https_get
from extract import load_head
from p4lib import (cos_rows, decomposition_logs, eef, group_folds, head_geometry, hier_from_parts, hier_mean,
                   hier_mean_parts, task_bootstrap)
import p4models as pm

PERSIST = ['camera_minus15', 'camera_minus15_surface_cool', 'camera_minus30', 'camera_plus15', 'camera_plus15_surface_warm',
           'camera_plus30', 'headlight_off', 'matched_global', 'surface_cool', 'surface_warm']
FAMILY = {c: ('camera' if 'camera' in c else 'lighting' if c in ('headlight_off', 'matched_global') else 'surface') for c in PERSIST}
LM1Q = {'libero_spatial': 'q004', 'libero_object': 'q005', 'libero_goal': 'q006', 'libero_10': 'q010'}
LM2Q = {'libero_spatial': 'q008', 'libero_object': 'q011', 'libero_goal': 'q012', 'libero_10': 'q020'}
B_BOOT = 2000
TERMS = ['T', 'rho', 'g', 'nu']
MODELS_B = ['Z', 'M0', 'M1', 'M2', 'M2g', 'M2L', 'M3', 'M3n', 'M3f']


def ci(a):
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    return [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))] if len(a) else [None, None]


def load_features(bos, prefix_keys):
    parts = []
    for key in prefix_keys:
        raw = bos.get(key)
        if raw is None:
            raise SafeError('missing feature object ' + key.rsplit('/', 1)[-1])
        z = np.load(io.BytesIO(raw), allow_pickle=False)
        parts.append({k: z[k] for k in z.files})
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


# ============================================================ part A
def part_A(F, task, cond, anchor, suite, out):
    R, A, L, V, M, Rc, Ac, Vv = (F[f'step_{k}'].astype(np.float64) for k in ('R', 'A', 'L', 'V', 'M', 'Rc', 'Ac', 'Vv'))
    d, ok, logs = decomposition_logs(R, A, L, V)
    res = {}
    # exclusions
    reasons = {}
    for nm, X in dict(R=R, A=A, L=L, V=V).items():
        bad = ~np.all(np.isfinite(X[:, [0, 9]]) & (X[:, [0, 9]] > 0), 1)
        reasons[nm] = {c: int((bad & (cond == c)).sum()) for c in PERSIST}
    res['n_pairs'] = int(len(R))
    res['n_included'] = int(ok.sum())
    res['excluded_by_reason_and_condition'] = reasons
    vals = np.column_stack([d[t] for t in TERMS])[ok]
    tk, ck = task[ok], cond[ok]

    parts = hier_mean_parts(vals, tk, ck)
    point = hier_from_parts(parts)
    reps = task_bootstrap(lambda w: hier_from_parts(parts, w), tk, B=B_BOOT)
    shares = reps[:, 1:] / reps[:, [0]]
    res['primary'] = {
        'aggregate_dlog': dict(zip(TERMS, point.tolist())),
        'aggregate_dlog_ci': {t: ci(reps[:, i]) for i, t in enumerate(TERMS)},
        'share': {t: float(point[i + 1] / point[0]) for i, t in enumerate(TERMS[1:])},
        'share_ci': {t: ci(shares[:, i]) for i, t in enumerate(TERMS[1:])},
        'identity_max_abs_residual': float(np.abs(vals[:, 0] - vals[:, 1:].sum(1)).max()),
    }
    # per-condition (task-equal) and sign consistency
    bycond = {}
    for c in PERSIST:
        m = ck == c
        bycond[c] = dict(zip(TERMS, hier_mean(vals[m], tk[m], ck[m]).tolist()))
    res['by_condition'] = bycond
    res['sign_consistency'] = {t: int(sum(np.sign(bycond[c][t]) == np.sign(point[i]) for c in PERSIST)) for i, t in enumerate(TERMS)}
    # dominant-term rule and nu rule (prereg section 3)
    rule = {}
    for i, t in enumerate(TERMS[1:]):
        sh, (lo, hi) = res['primary']['share'][t], res['primary']['aggregate_dlog_ci'][t]
        rule[t] = dict(share=sh, ci_excludes_0=bool(lo is not None and (lo > 0 or hi < 0)),
                       sign_consistent_conditions=res['sign_consistency'][t],
                       dominant=bool(sh >= 0.5 and (lo > 0 or hi < 0) and res['sign_consistency'][t] >= 8))
    nu_sh_hi = max(abs(x) for x in res['primary']['share_ci']['nu'])
    rule['nu_negligible'] = bool(abs(res['primary']['share']['nu']) < 0.1 and nu_sh_hi < 0.2)
    res['decision'] = rule
    # strata
    strata = {}
    for name, lab in dict(family=np.array([FAMILY[c] for c in ck]), anchor=anchor[ok], suite=suite[ok]).items():
        strata[name] = {}
        for g in np.unique(lab):
            m = lab == g
            strata[name][str(g)] = dict(zip(TERMS, hier_mean(vals[m], tk[m], ck[m]).tolist())) | {'n': int(m.sum())}
    res['strata'] = strata
    # leave-one-suite-out shares
    s_ok = suite[ok]
    res['leave_one_suite_out_share'] = {}
    for s in np.unique(s_ok):
        m = s_ok != s
        if m.sum() == 0:
            continue
        p = hier_mean(vals[m], tk[m], ck[m])
        res['leave_one_suite_out_share'][str(s)] = {t: float(p[i + 1] / p[0]) for i, t in enumerate(TERMS[1:])}
    # sensitivities
    sens = {'pair_pooled_mean': dict(zip(TERMS, vals.mean(0).tolist()))}
    m3 = np.isin(anchor[ok], ['q0', 'lm1', 'lm2'])
    sens['anchors_q0_lm1_lm2'] = dict(zip(TERMS, hier_mean(vals[m3], tk[m3], ck[m3]).tolist()))
    dv, okv, _ = decomposition_logs(R, A, L, Vv)
    valsv = np.column_stack([dv[t] for t in TERMS])[okv]
    sens['stored_bf16_velocity_V'] = dict(zip(TERMS, hier_mean(valsv, task[okv], cond[okv]).tolist())) | {'n': int(okv.sum())}
    unstable = np.abs(vals[:, 3]) > np.log(2)
    sens['n_unstable_abs_dlognu_gt_ln2'] = int(unstable.sum())
    sens['unstable_by_condition'] = {c: int((unstable & (ck == c)).sum()) for c in PERSIST}
    sens['excluding_unstable'] = dict(zip(TERMS, hier_mean(vals[~unstable], tk[~unstable], ck[~unstable]).tolist()))
    v0 = V[ok, 0]
    small = v0 <= np.quantile(v0, 0.10)
    sens['small_response_bottom10pct_V0'] = dict(zip(TERMS, hier_mean(vals[small], tk[small], ck[small]).tolist())) | {
        'n': int(small.sum()), 'by_condition': {c: int((small & (ck == c)).sum()) for c in PERSIST}}
    res['sensitivity'] = sens
    res['counterexamples_dlogrho_ge_0'] = {c: int(((vals[:, 1] >= 0) & (ck == c)).sum()) for c in PERSIST} | {
        'total': int((vals[:, 1] >= 0).sum())}
    res['counterexamples_dominant_not_rho'] = int((np.abs(vals[:, 1]) < np.maximum(np.abs(vals[:, 2]), np.abs(vals[:, 3]))).sum())
    # profiles (descriptive): condition-balanced mean of log X_k per step, overall and per family
    with np.errstate(divide='ignore', invalid='ignore'):
        prof_src = dict(logT=logs['T'], logrho=logs['rho'], logg=logs['g'], lognu=logs['nu'], logR=np.log(R), logA=np.log(A),
                        logN=np.log(np.maximum(R - A, 0)), logV=np.log(V), logrho_clean=np.log(Ac / Rc),
                        maprel=np.sqrt(M / V))
    prof_rows = []
    fam = np.array([FAMILY[c] for c in cond])
    for scope, mask in [('all', ok)] + [(f, ok & (fam == f)) for f in ('camera', 'lighting', 'surface')] + \
                       [(c, ok & (cond == c)) for c in PERSIST]:
        for name, X in prof_src.items():
            Xm = X[mask]
            good = np.all(np.isfinite(Xm), 1)
            if good.sum() == 0:
                continue
            p = hier_mean(Xm[good], task[mask][good], cond[mask][good])
            prof_rows.append([scope, name] + [f'{v:.6g}' for v in p])
    out['profiles'] = prof_rows
    # rho profile CI (overall)
    lr = logs['rho'][ok]
    lr_parts = hier_mean_parts(lr, tk, ck)
    rr = task_bootstrap(lambda w: hier_from_parts(lr_parts, w), tk, B=500)
    res['logrho_profile_ci'] = [ci(rr[:, k]) for k in range(10)]
    out['pair_A'] = dict(ok=ok, **{f'd{t}': d[t] for t in TERMS})
    return res


# ============================================================ part B
def build_B(F, Ur, task, cond, anchor):
    n = len(task)
    u = F['u'].astype(np.float64)
    a0 = F['a0'].astype(np.float64)
    n0 = F['n0'].astype(np.float32)
    rk = Ur.shape[1]
    dr0full = (n0.reshape(n, 5, 1024) + (a0.reshape(n, 5, rk) @ Ur.T).astype(np.float32)).reshape(n, -1)
    cell = np.char.add(np.char.add(cond.astype(str), '|'), anchor.astype(str))
    return dict(Y=u[:, 1:].sum(1).reshape(n, -1), u0=u[:, 0].reshape(n, -1), a0=a0, n0=n0, dr0full=dr0full,
                R0=F['step_R'][:, 0].astype(np.float64), task=task, cond=cond, anchor=anchor, cell=cell,
                cond_levels=sorted(set(cond)), anchor_levels=sorted(set(anchor)), cell_levels=sorted(set(cell)))


def eval_block(Y, P, task, cond, models):
    res = {m: dict(EEF=eef(Y, P[m]), cos_median=float(np.nanmedian(cos_rows(P[m], Y)))) for m in models}
    sums = {m: pm.task_sums(Y, P[m], task) for m in models}
    sums_c = {m: pm.task_sums(Y, P[m], task, cond) for m in models}
    for m in models:
        res[m]['EEF_condition_balanced'] = pm.eef_weighted(sums_c[m], {t: 1.0 for t in set(task)}, by_cond=True)
    return res, sums, sums_c


def boot_diff(sums, sums_c, a, b, tasks, seed=20261009):
    f = lambda w: [pm.eef_weighted(sums[a], w) - pm.eef_weighted(sums[b], w),
                   pm.eef_weighted(sums_c[a], w, True) - pm.eef_weighted(sums_c[b], w, True)]
    reps = task_bootstrap(f, tasks, B=B_BOOT, seed=seed)
    return {'pooled_ci': ci(reps[:, 0]), 'condition_balanced_ci': ci(reps[:, 1])}


def part_B(D, out):
    Y, task, cond, anchor = D['Y'], D['task'], D['cond'], D['anchor']
    n = len(Y)
    folds = group_folds(task, 5, 20261008)
    P = {m: np.zeros_like(Y) for m in MODELS_B}
    H = {}
    mag = {k: np.zeros(n) for k in ('logR0', 'logU0', 'both')}
    for f in range(5):
        tr, ev = np.where(folds != f)[0], np.where(folds == f)[0]
        D['_outer_fold'] = f
        p, h = pm.predict_remaining(D, tr, ev, use_cond=True)
        for m in MODELS_B:
            P[m][ev] = p[m]
        H[f] = {k: (list(map(float, v)) if isinstance(v, tuple) else v) for k, v in h.items()}
        mp = pm.magnitude_r2_preds(D, tr, ev)
        for k in mag:
            mag[k][ev] = mp[k]
        print(json.dumps({'B_fold': f, 'seconds': round(time.time() - T0)}), flush=True)
    res = {'n': int(n), 'hyperparameters_by_fold': H}
    met, sums, sums_c = eval_block(Y, P, task, cond, MODELS_B)
    res['metrics'] = met
    tasks = sorted(set(task))
    res['primary_contrast_M3_minus_M2L'] = {'pooled': met['M3']['EEF'] - met['M2L']['EEF'],
                                            'condition_balanced': met['M3']['EEF_condition_balanced'] - met['M2L']['EEF_condition_balanced'],
                                            **boot_diff(sums, sums_c, 'M3', 'M2L', tasks)}
    res['M3_minus_each_baseline'] = {b: {'pooled': met['M3']['EEF'] - met[b]['EEF'], **boot_diff(sums, sums_c, 'M3', b, tasks)}
                                     for b in ('M0', 'M1', 'M2', 'M2g', 'M2L')}
    res['strongest_baseline'] = max(('M0', 'M1', 'M2', 'M2g', 'M2L'), key=lambda b: met[b]['EEF'])
    # by condition / anchor / family
    for name, lab in dict(condition=cond, anchor=anchor, family=np.array([FAMILY[c] for c in cond])).items():
        res['by_' + name] = {str(g): {m: eef(Y[lab == g], P[m][lab == g]) for m in MODELS_B} for g in np.unique(lab)}
    # magnitude R2 (centered) on log||Y||^2
    yl = np.log((Y ** 2).sum(1))
    res['magnitude_R2'] = {k: float(1 - ((yl - v) ** 2).sum() / ((yl - yl.mean()) ** 2).sum()) for k, v in mag.items()}
    # final planned command D = u0 + Y: observed vs future
    Dfin = D['u0'] + Y
    res['final_command'] = {m: eef(Dfin, D['u0'] + P[m]) for m in MODELS_B}
    res['final_command_note'] = 'Z here equals the observed first update alone (u0); other rows add the predicted future part'
    obs_share = (D['u0'] ** 2).sum(1) / np.maximum((Dfin ** 2).sum(1), 1e-300)
    res['observed_share_of_final_energy_quantiles'] = np.quantile(obs_share, [.1, .25, .5, .75, .9]).tolist()
    # leave-family-out
    fam = np.array([FAMILY[c] for c in cond])
    lofo = {}
    for i, held in enumerate(('camera', 'lighting', 'surface')):
        tr, ev = np.where(fam != held)[0], np.where(fam == held)[0]
        D['_outer_fold'] = 10 + i
        p, h = pm.predict_remaining(D, tr, ev, use_cond=False)
        Ye = Y[ev]
        mt = {m: eef(Ye, p[m]) for m in MODELS_B}
        s = {m: pm.task_sums(Ye, p[m], task[ev]) for m in ('M3', 'M2L')}
        reps = task_bootstrap(lambda w: pm.eef_weighted(s['M3'], w) - pm.eef_weighted(s['M2L'], w), sorted(set(task[ev])), B=B_BOOT)
        lofo[held] = {'n': int(len(ev)), 'EEF': mt, 'M3_minus_M2L': mt['M3'] - mt['M2L'], 'ci': ci(reps),
                      'hyperparameters': {k: (list(map(float, v)) if isinstance(v, tuple) else v) for k, v in h.items()}}
        print(json.dumps({'LOFO': held, 'seconds': round(time.time() - T0)}), flush=True)
    res['leave_family_out'] = lofo
    pc = res['primary_contrast_M3_minus_M2L']
    pos_lofo = sum(lofo[f]['M3_minus_M2L'] > 0 for f in lofo)
    res['decision'] = {'readout_adds_knowledge': bool(pc['pooled'] >= 0.02 and pc['pooled_ci'][0] > 0 and pos_lofo >= 2),
                       'delta_pooled': pc['pooled'], 'ci': pc['pooled_ci'], 'lofo_positive': int(pos_lofo)}
    out['pred_B'] = P
    out['pair_B'] = {m: ((Y - P[m]) ** 2).sum(1) for m in MODELS_B} | {f'cos_{m}': cos_rows(P[m], Y) for m in MODELS_B}
    return res


# ============================================================ part C
def build_C(F, D, Ur, task, cond, anchor, q, root, noise, require_same_state=True):
    keys = {}
    for i, k in enumerate(zip(root, cond, q)):
        keys.setdefault(k, []).append(i)
    rows = []
    qc = {'units': 0, 'units_3seeds': 0, 'vis_identical': 0, 'rc0_distinct': 0}
    for k, idx in keys.items():
        qc['units'] += 1
        if sorted(noise[idx].tolist()) != [0, 1, 2]:
            continue
        qc['units_3seeds'] += 1
        vis_same = len(set(F['vis_hash'][idx].tolist())) == 1
        rc_dist = len(set(F['rc0_hash'][idx].tolist())) == 3
        qc['vis_identical'] += int(vis_same)
        qc['rc0_distinct'] += int(rc_dist)
        # v2 fix: the same-observation check (QC-S) applies to q0 only (prereg section 5); later anchors are different
        # visited states by design and only need three seeds with distinct noise.
        if not rc_dist or (require_same_state and not vis_same):
            continue
        rows.append(idx)
    n = len(rows) * 3
    Dm = F['u'].astype(np.float64).sum(1).reshape(len(task), -1)
    phys = F['dphys'].astype(np.float64)[:, :, 0:3].sum(1)
    cs = {k: [] for k in ('D', 'C1', 'Dnorm_o', 'abar_o', 'nbar_o', 'task', 'cond', 'anchor', 'root', 'noise', 'phys', 'phys_C1')}
    for idx in rows:
        for t in range(3):
            i, o = idx[t], [idx[j] for j in range(3) if j != t]
            cs['D'].append(Dm[i]); cs['C1'].append(Dm[o].mean(0)); cs['Dnorm_o'].append(np.linalg.norm(Dm[o], axis=1).mean())
            cs['abar_o'].append(F['abar'][o].astype(np.float64).mean(0)); cs['nbar_o'].append(F['nbar'][o].mean(0))
            cs['task'].append(task[i]); cs['cond'].append(cond[i]); cs['anchor'].append(anchor[i]); cs['root'].append(root[i])
            cs['noise'].append(noise[i]); cs['phys'].append(phys[i]); cs['phys_C1'].append(phys[o].mean(0))
    cs = {k: np.asarray(v) for k, v in cs.items()}
    cs['nbar_o'] = cs['nbar_o'].astype(np.float32)
    return cs, qc


def part_C(F, Dset, Ur, task, cond, anchor, q, root, noise, out):
    res = {}
    for cls in ('q0', 'lm1', 'lm2', 'later'):
        m = anchor == cls
        cs, qc = build_C({k: v[m] for k, v in F.items()}, None, Ur, task[m], cond[m], anchor[m], q[m], root[m], noise[m],
                         require_same_state=(cls == 'q0'))
        r = {'qc': qc, 'n_targets': int(len(cs['D']))}
        if len(cs['D']) < 50:
            res[cls] = r
            continue
        folds = group_folds(cs['task'], 5, 20261008)
        P = {k: np.zeros_like(cs['D']) for k in ('C0', 'C1', 'C2', 'C3')}
        H = {}
        for f in range(5):
            tr, ev = np.where(folds != f)[0], np.where(folds == f)[0]
            cs['_outer_fold'] = 20 + f
            p, h = pm.predict_crossseed(cs, tr, ev)
            for k in P:
                P[k][ev] = p[k]
            H[f] = {k: list(map(float, v)) for k, v in h.items()}
        met, sums, sums_c = eval_block(cs['D'], P, cs['task'], cs['cond'], list(P))
        tasks = sorted(set(cs['task']))
        r.update(metrics=met, hyperparameters_by_fold=H,
                 C2_minus_C1={'pooled': met['C2']['EEF'] - met['C1']['EEF'], **boot_diff(sums, sums_c, 'C2', 'C1', tasks)},
                 C3_minus_C1={'pooled': met['C3']['EEF'] - met['C1']['EEF'], **boot_diff(sums, sums_c, 'C3', 'C1', tasks)},
                 C1_minus_C0={'pooled': met['C1']['EEF'] - met['C0']['EEF'], **boot_diff(sums, sums_c, 'C1', 'C0', tasks)},
                 phys_translation_EEF_C1=eef(cs['phys'], cs['phys_C1']),
                 by_condition={c: {k: eef(cs['D'][cs['cond'] == c], P[k][cs['cond'] == c]) for k in P} for c in np.unique(cs['cond'])},
                 same_visited_state=bool(cls == 'q0'))
        d2, d3 = r['C2_minus_C1'], r['C3_minus_C1']
        r['decision'] = {'readout_transfers_better_than_command_average': bool(d2['pooled'] >= 0.02 and d2['pooled_ci'][0] > 0),
                         'readout_adds_beyond_command_average': bool(d3['pooled'] >= 0.02 and d3['pooled_ci'][0] > 0)}
        res[cls] = r
        out.setdefault('pair_C', {})[cls] = dict(task=cs['task'], cond=cs['cond'], root=cs['root'], noise=cs['noise'],
                                                 **{f'sse_{k}': ((cs['D'] - P[k]) ** 2).sum(1) for k in P},
                                                 **{f'cos_{k}': cos_rows(P[k], cs['D']) for k in P}, Dnorm2=(cs['D'] ** 2).sum(1))
        print(json.dumps({'C': cls, 'seconds': round(time.time() - T0)}), flush=True)
    return res


# ============================================================ main
def write_csv_gz(header, rows):
    sio = io.StringIO()
    w = csv.writer(sio)
    w.writerow(header)
    w.writerows(rows)
    return gzip.compress(sio.getvalue().encode())


T0 = time.time()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--request', required=True)
    ap.add_argument('--workdir', required=True)
    args = ap.parse_args()
    q = json.load(open(args.request))
    deadline = float(os.environ['P4_DEADLINE_EPOCH'])
    bos = BOS(q['output_uri'], deadline - 30)
    res = {'run_id': q['run_id'], 'prereg_sha256': q['prereg_sha256'], 'numpy': np.__version__, 'status': 'running'}
    out = {}
    try:
        W, b, hinfo = load_head(os.environ.pop('P4_HEAD_URL'), q['head_sha256'])
        Ur, s, r, erank = head_geometry(W[:, :7])
        res['head'] = {**hinfo, 'W7_singular_values': s.tolist(), 'rank_1e-3': r, 'entropy_rank': erank}
        F = load_features(bos, q['feature_keys'])
        res['n_loaded'] = int(len(F['cond']))
        cond, root, q_, noise = F['cond'].astype(str), F['root'].astype(str), F['q'].astype(str), F['noise'].astype(int)
        u = F['u'].astype(np.float64)
        S = (u ** 2).sum((2, 3)).sum(1)
        Fe = (u.sum(1) ** 2).sum((1, 2))
        keep = np.isin(cond, PERSIST) & (S > 0) & (Fe > 0)
        res['population'] = {'loaded': int(len(cond)), 'kept_S_F_positive': int(keep.sum()), 'expected': 8673}
        F = {k: v[keep] for k, v in F.items()}
        cond, root, q_, noise = cond[keep], root[keep], q_[keep], noise[keep]
        task = np.array([x.rsplit('-', 1)[0] for x in root])
        suite = np.array([x.split('-')[0] for x in root])
        anchor = np.array(['q0' if qq == 'q000' else 'lm1' if qq == LM1Q[s_] else 'lm2' if qq == LM2Q[s_] else 'later'
                           for qq, s_ in zip(q_, suite)])
        qh = F['qh']
        e_all = np.concatenate([qh[:, 0], qh[:, 2]])
        res['QC'] = {'QC_H_rel_err_E_median': float(np.median(e_all)), 'QC_H_rel_err_E_p99': float(np.quantile(e_all, .99)),
                     'QC_H_rel_err_32_median': float(np.median(np.concatenate([qh[:, 1], qh[:, 3]]))),
                     'QC_H_pass': bool(np.median(e_all) < 0.01 and np.quantile(e_all, .99) < 0.05),
                     'QC_X_x0_equal_frac': float(F['x0eq'].mean()), 'QC_X_closure_max_abs': float(F['closure'].max()),
                     'dt_max_abs_dev_from_-0.1': float(F['dt_dev'].max()),
                     'QC_N_all_finite': bool(all(np.isfinite(F[f'step_{k}']).all() for k in ('R', 'A', 'L', 'V', 'Rc', 'Ac')))}
        print(json.dumps({'QC': res['QC'], 'n': int(keep.sum())}), flush=True)
        if res['QC']['QC_H_pass']:
            try:
                res['A'] = part_A(F, task, cond, anchor, suite, out)
            except Exception as e:  # noqa: BLE001
                res['A'] = {'failed': type(e).__name__ + ': ' + str(e)[:300], 'trace': traceback.format_exc()[-1200:]}
            print(json.dumps({'A_done': round(time.time() - T0)}), flush=True)
            Dset = build_B(F, Ur, task, cond, anchor)
            try:
                res['B'] = part_B(Dset, out)
            except Exception as e:  # noqa: BLE001
                res['B'] = {'failed': type(e).__name__ + ': ' + str(e)[:300], 'trace': traceback.format_exc()[-1200:]}
            try:
                res['C'] = part_C(F, Dset, Ur, task, cond, anchor, q_, root, noise, out)
            except Exception as e:  # noqa: BLE001
                res['C'] = {'failed': type(e).__name__ + ': ' + str(e)[:300], 'trace': traceback.format_exc()[-1200:]}
        else:
            res['halt'] = 'QC-H failed: head identity not established; parts A/B/C (readable coordinates) not run'
        # per-pair table for the Mac (scalars only)
        head = ['root', 'noise', 'cond', 'q', 'anchor', 'task']
        cols = {}
        if 'pair_A' in out:
            cols.update({k: v for k, v in out['pair_A'].items()})
        if 'pair_B' in out:
            cols.update({f'B_{k}': v for k, v in out['pair_B'].items()})
        names = list(cols)
        rows = []
        for i in range(len(cond)):
            rows.append([root[i], noise[i], cond[i], q_[i], anchor[i], task[i]] +
                        [(f'{cols[k][i]:.6g}' if not isinstance(cols[k][i], (bool, np.bool_)) else int(cols[k][i])) for k in names])
        bos.put_bytes(write_csv_gz(head + names, rows), 'tables/p4_pairs.csv.gz')
        if 'profiles' in out:
            bos.put_bytes(write_csv_gz(['scope', 'quantity'] + [f'k{j}' for j in range(10)], out['profiles']), 'tables/p4_profiles.csv.gz')
        for cls, d in out.get('pair_C', {}).items():
            names = [k for k in d if k not in ('task', 'cond', 'root', 'noise')]
            rows = [[d['root'][i], d['noise'][i], d['cond'][i], d['task'][i]] + [f'{d[k][i]:.6g}' for k in names] for i in range(len(d['task']))]
            bos.put_bytes(write_csv_gz(['root', 'noise', 'cond', 'task'] + names, rows), f'tables/p4_crossseed_{cls}.csv.gz')
        if 'pred_B' in out:
            buf = io.BytesIO()
            np.savez_compressed(buf, **{k: v.astype(np.float32) for k, v in out['pred_B'].items()})
            bos.put_bytes(buf.getvalue(), 'vectors/predictions_B.npz')
        res['status'] = 'complete'
    except Exception as e:  # noqa: BLE001
        res['status'] = 'failed'
        res['fatal'] = type(e).__name__ + ': ' + str(e)[:300]
        res['trace'] = traceback.format_exc()[-1500:]
    res['seconds'] = round(time.time() - T0, 1)
    try:
        bos.put_bytes(json.dumps(res, indent=1, default=lambda x: x.tolist() if hasattr(x, 'tolist') else str(x)).encode(),
                      'P4-RESULTS.json')
    except Exception as e:  # noqa: BLE001
        print(json.dumps({'results_upload_failed': type(e).__name__}), flush=True)
    print(json.dumps({'status': res['status'], 'seconds': res['seconds']}), flush=True)
    sys.exit(0 if res['status'] == 'complete' else 3)


if __name__ == '__main__':
    main()
