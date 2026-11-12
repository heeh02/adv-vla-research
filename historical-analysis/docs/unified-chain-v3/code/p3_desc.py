"""Descriptive (NOT preregistered) direction comparisons between conditions, same-scene.
Reads data/p2/*.npz (dphys) with the preregistered anchors {q0, lm1, lm2}; unit = root x condition, D_sys = seed mean of the
anchor-summed first-5-step translation of the adapter-planned command difference (same visited state).
Outputs reports/P3-DESC-direction-similarity.json (2D x/y cosines, task-cluster bootstrap) and
reports/P3-DESC-direction-global.json (3D mean unit direction and resultant length per condition / suite).
"""
import glob
import json
import os

import numpy as np
import pandas as pd

from p2_analyze import PERSIST
from p3_analyze import LM1Q, LM2Q, boot_median

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, '..', 'data')
R = os.path.join(HERE, '..', 'reports')
PAIRS = [('headlight_off', 'camera_plus30'), ('headlight_off', 'camera_minus30'), ('headlight_off', 'camera_plus15'),
         ('headlight_off', 'camera_minus15'), ('matched_global', 'headlight_off'), ('matched_global', 'camera_plus30'),
         ('surface_warm', 'surface_cool'), ('surface_warm', 'camera_plus30'), ('camera_plus15', 'camera_plus30'),
         ('camera_minus15', 'camera_minus30')]


def units():
    rows = []
    for f in sorted(glob.glob(os.path.join(D, 'p2', 'shard-*.npz'))):
        z = np.load(f)
        m = pd.DataFrame(z['meta'])[['root', 'noise', 'cond', 'q']]
        m[['dx', 'dy', 'dz']] = z['dphys'][:, :5, :3].astype(np.float64).sum(1)
        rows.append(m)
    q = pd.concat(rows, ignore_index=True)
    s = q.root.str.split('-').str[0]
    q = q[(q.q == 'q000') | (q.q == s.map(LM1Q)) | (q.q == s.map(LM2Q))]
    q = q[q.cond.isin(PERSIST)]
    return q.groupby(['root', 'cond', 'noise'])[['dx', 'dy', 'dz']].sum().groupby(['root', 'cond']).mean()


def cos_rows(A, B):
    na, nb = np.linalg.norm(A, axis=1), np.linalg.norm(B, axis=1)
    ok = (na > 0) & (nb > 0)
    out = np.full(len(A), np.nan)
    out[ok] = (A[ok] * B[ok]).sum(1) / na[ok] / nb[ok]
    return out


def main():
    U = units()
    sim = {}
    for a, b in PAIRS:
        A, B = U.xs(a, level='cond')[['dx', 'dy']], U.xs(b, level='cond')[['dx', 'dy']]
        com = A.index.intersection(B.index)
        c = cos_rows(A.loc[com].values, B.loc[com].values)
        m, ci, n = boot_median(c, [r.rsplit('-', 1)[0] for r in com])
        sim[f'{a}~{b}'] = [round(m, 3), [round(x, 3) for x in ci], n]
    json.dump(sim, open(os.path.join(R, 'P3-DESC-direction-similarity.json'), 'w'), indent=1)
    glob_ = {}
    for c in PERSIST:
        X = U.xs(c, level='cond')[['dx', 'dy', 'dz']]
        uv = X.values / np.linalg.norm(X.values, axis=1, keepdims=True)
        mu = uv.mean(0)
        suite = X.index.str.split('-').str[0]
        glob_[c] = dict(mean_unit=mu.round(3).tolist(), resultant=round(float(np.linalg.norm(mu)), 3),
                        per_suite={s: uv[suite == s].mean(0).round(2).tolist() for s in sorted(set(suite))})
    json.dump(glob_, open(os.path.join(R, 'P3-DESC-direction-global.json'), 'w'), indent=1)
    print(json.dumps(sim, indent=0))


if __name__ == '__main__':
    main()
