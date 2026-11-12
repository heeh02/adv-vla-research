"""DEV-2 sensitivity: P2-D2 shares and P1-D2 R^2 restricted to the three preregistered anchors {q0, lm1, lm2} (slot <= 2;
for persistent conditions slots 0-2 are exactly q0/lm1/lm2). Reads data/p1_pairs.csv.gz; writes reports/SENS-first3-anchors.json."""
import json
import os

import numpy as np
import pandas as pd

from p1_analyze import ols_predict
from p2_analyze import decompose

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, '..', 'data')
R = os.path.join(HERE, '..', 'reports')


def main():
    df = pd.read_csv(os.path.join(D, 'p1_pairs.csv.gz'))
    df = df[df.slot <= 2].copy()
    df['logK'] = df.logKeff
    dec = decompose(df)
    rng = np.random.default_rng(20261008)
    tasks = np.array(sorted(df.task_id.unique()))
    perm = rng.permutation(len(tasks))
    fold = df.task_id.map({tasks[i]: k % 5 for k, i in enumerate(perm)})
    y = df.logP.values
    pa, pb = np.zeros(len(df)), np.zeros(len(df))
    for k in range(5):
        tr, te = (fold != k).values, (fold == k).values
        dtr, dte = df[tr], df[te]
        pa[te] = ols_predict(dtr[['logRtot']].values, dtr.logP.values, dte[['logRtot']].values)
        prof = dtr.groupby('cond')[['logTeff', 'logc2', 'logKeff']].mean()
        pb[te] = ols_predict(np.column_stack([dtr.logRtot, prof.loc[dtr.cond].values]), dtr.logP.values,
                             np.column_stack([dte.logRtot, prof.loc[dte.cond].values]))
    r2 = lambda p: 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    out = dict(n=int(len(df)), P2_D2_share={L: round(v['D2_share'], 3) for L, v in dec.items()},
               P1_D2=dict(A=round(r2(pa), 3), B=round(r2(pb), 3)))
    json.dump(out, open(os.path.join(R, 'SENS-first3-anchors.json'), 'w'), indent=1)
    print(out)


if __name__ == '__main__':
    main()
