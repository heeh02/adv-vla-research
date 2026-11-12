"""P4 analysis v3 (Baidu CPU): post-hoc PH-2 only - fitted command-average control C1s for the cross-seed test,
per anchor class, compared with C1/C2/C3 recomputed identically to v2 (same folds, same seeds)."""
import argparse, json, os, sys, traceback
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
    res = {'run_id': q['run_id'], 'status': 'running', 'note': 'v3: post-hoc PH-2 fitted command-average control'}
    try:
        W, b, _ = load_head(os.environ.pop('P4_HEAD_URL'), q['head_sha256'])
        Ur, *_ = head_geometry(W[:, :7])
        F = an.load_features(bos, q['feature_keys'])
        cond, root, q_, noise = F['cond'].astype(str), F['root'].astype(str), F['q'].astype(str), F['noise'].astype(int)
        task = np.array([x.rsplit('-', 1)[0] for x in root]); suite = np.array([x.split('-')[0] for x in root])
        anchor = np.array(['q0' if qq == 'q000' else 'lm1' if qq == an.LM1Q[s_] else 'lm2' if qq == an.LM2Q[s_] else 'later'
                           for qq, s_ in zip(q_, suite)])
        out = {}
        for cls in ('q0', 'lm1', 'lm2', 'later'):
            m = anchor == cls
            cs, qc = an.build_C({k: v[m] for k, v in F.items()}, None, Ur, task[m], cond[m], anchor[m], q_[m], root[m], noise[m],
                                require_same_state=(cls == 'q0'))
            folds = group_folds(cs['task'], 5, 20261008)
            P = {k: np.zeros_like(cs['D']) for k in ('C1', 'C1s', 'C2', 'C3')}
            H = {}
            for f in range(5):
                tr, ev = np.where(folds != f)[0], np.where(folds == f)[0]
                cs['_outer_fold'] = 20 + f
                p, h = pm.predict_crossseed(cs, tr, ev)
                P['C1'][ev], P['C2'][ev], P['C3'][ev] = p['C1'], p['C2'], p['C3']
                P['C1s'][ev], H[f] = pm.posthoc_C1s(cs, tr, ev)
            Y, t = cs['D'], cs['task']
            sums = {k: pm.task_sums(Y, P[k], t) for k in P}
            def d(a, b_):
                reps = task_bootstrap(lambda w: pm.eef_weighted(sums[a], w) - pm.eef_weighted(sums[b_], w), sorted(set(t)), B=2000)
                return {'delta': eef(Y, P[a]) - eef(Y, P[b_]), 'ci': an.ci(reps)}
            out[cls] = {'n': int(len(Y)), 'EEF': {k: eef(Y, P[k]) for k in P}, 'C1s_minus_C1': d('C1s', 'C1'),
                        'C2_minus_C1s': d('C2', 'C1s'), 'C3_minus_C1s': d('C3', 'C1s'),
                        'C1s_lambda_by_fold': {k: float(v[0]) for k, v in H.items()}}
            print(json.dumps({'cls': cls, 'EEF': out[cls]['EEF']}), flush=True)
        res['PH2'] = out
        res['status'] = 'complete'
    except Exception as e:  # noqa: BLE001
        res['status'] = 'failed'; res['fatal'] = type(e).__name__ + ': ' + str(e)[:300]; res['trace'] = traceback.format_exc()[-1500:]
    bos.put_bytes(json.dumps(res, indent=1, default=lambda x: x.tolist() if hasattr(x, 'tolist') else str(x)).encode(), 'P4-RESULTS-v3.json')
    print(json.dumps({'status': res['status']}), flush=True)
    sys.exit(0 if res['status'] == 'complete' else 3)


if __name__ == '__main__':
    main()
