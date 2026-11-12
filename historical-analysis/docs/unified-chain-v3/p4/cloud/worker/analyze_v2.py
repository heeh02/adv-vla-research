"""P4 analysis v2 (Baidu CPU): (1) part C later anchors with the q0-only same-state check (implementation fix of v1),
(2) post-hoc PH-1: readout null-space correction on top of the no-fit persistence baseline M2 (labelled post-hoc).
v1 results stay unchanged (analysis/); v2 writes to analysis-v2/."""
import argparse, io, json, os, sys, time, traceback
import numpy as np
from cloudio import BOS
from extract import load_head
from p4lib import eef, group_folds, head_geometry, task_bootstrap
import p4models as pm
import analyze as an


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--request', required=True); ap.add_argument('--workdir', required=True)
    args = ap.parse_args()
    q = json.load(open(args.request))
    bos = BOS(q['output_uri'], float(os.environ['P4_DEADLINE_EPOCH']) - 30)
    res = {'run_id': q['run_id'], 'status': 'running', 'note': 'v2: part C later-anchor fix + post-hoc PH-1'}
    out = {}
    try:
        W, b, hinfo = load_head(os.environ.pop('P4_HEAD_URL'), q['head_sha256'])
        Ur, *_ = head_geometry(W[:, :7])
        F = an.load_features(bos, q['feature_keys'])
        cond, root, q_, noise = F['cond'].astype(str), F['root'].astype(str), F['q'].astype(str), F['noise'].astype(int)
        task = np.array([x.rsplit('-', 1)[0] for x in root]); suite = np.array([x.split('-')[0] for x in root])
        anchor = np.array(['q0' if qq == 'q000' else 'lm1' if qq == an.LM1Q[s_] else 'lm2' if qq == an.LM2Q[s_] else 'later'
                           for qq, s_ in zip(q_, suite)])
        res['n'] = int(len(cond))
        res['C'] = an.part_C(F, None, Ur, task, cond, anchor, q_, root, noise, out)
        D = an.build_B(F, Ur, task, cond, anchor)
        Y = D['Y']
        folds = group_folds(task, 5, 20261008)
        P = np.zeros_like(Y); H = {}
        for f in range(5):
            tr, ev = np.where(folds != f)[0], np.where(folds == f)[0]
            D['_outer_fold'] = f
            P[ev], H[f] = pm.posthoc_M2R(D, tr, ev)
        M2 = 9.0 * D['u0']
        s_r, s_2 = pm.task_sums(Y, P, task), pm.task_sums(Y, M2, task)
        reps = task_bootstrap(lambda w: pm.eef_weighted(s_r, w) - pm.eef_weighted(s_2, w), sorted(set(task)), B=2000)
        ph = {'EEF_M2R': eef(Y, P), 'EEF_M2': eef(Y, M2), 'delta': eef(Y, P) - eef(Y, M2), 'ci': an.ci(reps),
              'hyperparameters_by_fold': {k: list(map(float, v)) for k, v in H.items()},
              'by_anchor': {a: eef(Y[anchor == a], P[anchor == a]) - eef(Y[anchor == a], M2[anchor == a]) for a in np.unique(anchor)}}
        fam = np.array([an.FAMILY[c] for c in cond]); lofo = {}
        for i, held in enumerate(('camera', 'lighting', 'surface')):
            tr, ev = np.where(fam != held)[0], np.where(fam == held)[0]
            D['_outer_fold'] = 10 + i
            p, h = pm.posthoc_M2R(D, tr, ev, use_cond=False)
            s1, s2 = pm.task_sums(Y[ev], p, task[ev]), pm.task_sums(Y[ev], M2[ev], task[ev])
            r2 = task_bootstrap(lambda w: pm.eef_weighted(s1, w) - pm.eef_weighted(s2, w), sorted(set(task[ev])), B=2000)
            lofo[held] = {'delta': eef(Y[ev], p) - eef(Y[ev], M2[ev]), 'ci': an.ci(r2), 'hyper': list(map(float, h))}
        ph['leave_family_out'] = lofo
        res['PH1_M2_plus_readout'] = ph
        for cls, d in out.get('pair_C', {}).items():
            names = [k for k in d if k not in ('task', 'cond', 'root', 'noise')]
            rows = [[d['root'][i], d['noise'][i], d['cond'][i], d['task'][i]] + [f'{d[k][i]:.6g}' for k in names] for i in range(len(d['task']))]
            bos.put_bytes(an.write_csv_gz(['root', 'noise', 'cond', 'task'] + names, rows), f'tables/p4_crossseed_{cls}.csv.gz')
        res['status'] = 'complete'
    except Exception as e:  # noqa: BLE001
        res['status'] = 'failed'; res['fatal'] = type(e).__name__ + ': ' + str(e)[:300]; res['trace'] = traceback.format_exc()[-1500:]
    bos.put_bytes(json.dumps(res, indent=1, default=lambda x: x.tolist() if hasattr(x, 'tolist') else str(x)).encode(), 'P4-RESULTS-v2.json')
    print(json.dumps({'status': res['status']}), flush=True)
    sys.exit(0 if res['status'] == 'complete' else 3)


if __name__ == '__main__':
    main()
