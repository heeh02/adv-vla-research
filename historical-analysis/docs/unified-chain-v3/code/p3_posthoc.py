"""POST-HOC checks requested in review round 2 (not preregistered; reported separately from P3-RESULTS.json).
Paired task-cluster bootstrap of within-condition AUROC differences on one common population, each replicate computing
all predictors on the same resample. Predictors: D_norm (prereg; includes the evaluated seed), D_loo (command magnitude
from the other seeds only, same information condition as O_loo), O_loo (prereg).
Reads data/p3_harm.csv.gz (written by p3_analyze.py); writes reports/P3-POSTHOC-auroc-paired.json.
"""
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import rankdata

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, '..', 'data')
R = os.path.join(HERE, '..', 'reports')
B = 10000
PRED = ['D_norm', 'D_loo', 'O_loo']


def auc_parts(y, x):
    """Mann-Whitney AUROC numerator (ties averaged) and pair count for one condition."""
    n1 = y.sum()
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return 0.0, 0.0
    r = rankdata(x)
    return float(r[y == 1].sum() - n1 * (n1 + 1) / 2), float(n1 * n0)


def main():
    H = pd.read_csv(os.path.join(D, 'p3_harm.csv.gz')).dropna(subset=PRED).reset_index(drop=True)
    cells = {}
    for (t, c), g in H.groupby(['task_id', 'cond']):
        cells.setdefault(t, []).append((c, g.harm.values, {p: g[p].values for p in PRED}))
    tasks = np.array(sorted(cells))

    def auc_all(task_sample):
        by_c = {}
        for t in task_sample:
            for c, y, xs in cells[t]:
                by_c.setdefault(c, []).append((y, xs))
        num = dict.fromkeys(PRED, 0.0)
        den = 0.0
        for c, lst in by_c.items():
            y = np.concatenate([a for a, _ in lst])
            for p in PRED:
                nu, de = auc_parts(y, np.concatenate([b[p] for _, b in lst]))
                num[p] += nu
            den += de
        return {p: num[p] / den if den else np.nan for p in PRED}

    point = auc_all(tasks)
    rng = np.random.default_rng(20261009)
    reps = [auc_all(rng.choice(tasks, len(tasks))) for _ in range(B)]
    A = {p: np.array([r[p] for r in reps]) for p in PRED}
    out = dict(note='post-hoc; paired task-cluster bootstrap; common population; not a preregistered test',
               n=int(len(H)), harms=int(H.harm.sum()), n_tasks=int(len(tasks)), B=B, seed=20261009,
               auc={p: dict(point=point[p], ci=np.nanpercentile(A[p], [2.5, 97.5]).tolist()) for p in PRED})
    for a, b in [('O_loo', 'D_norm'), ('O_loo', 'D_loo'), ('D_loo', 'D_norm')]:
        d = A[a] - A[b]
        out[f'diff_{a}_minus_{b}'] = dict(point=point[a] - point[b], ci=np.nanpercentile(d, [2.5, 97.5]).tolist(),
                                          frac_reps_le_0=float(np.nanmean(d <= 0)))
    json.dump(out, open(os.path.join(R, 'P3-POSTHOC-auroc-paired.json'), 'w'), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
