"""Original published scalar estimator functions, unchanged AST. No I/O."""
import numpy as np
from scipy.stats import rankdata
from itertools import combinations
NAMES=["hidden","main_TV","action","delta_sr"]
PAIRS=list(combinations(range(4),2))

def corr(a):
    a = np.asarray(a, dtype=float)
    a = a - a.mean(axis=0)
    den = np.sqrt((a * a).sum(axis=0))
    with np.errstate(divide='ignore', invalid='ignore'):
        return a.T @ a / np.outer(den, den)

def group_center(a, codes):
    counts = np.bincount(codes)
    means = np.stack([np.bincount(codes, weights=a[:, j], minlength=len(counts)) / np.maximum(counts, 1) for j in range(a.shape[1])], axis=1)
    return a - means[codes]

def rank_residuals(a, groups):
    """Rank all rows, then remove specified additive categorical group effects.

    Alternating demeaning solves the two-way least-squares projection. This is
    a descriptive partial rank correlation, not a within-group ranked Spearman.
    """
    z = rankdata(a, axis=0, method='average').astype(float)
    if not groups:
        return z
    for _ in range(1000):
        previous = z.copy()
        for codes in groups:
            z = group_center(z, codes)
        if np.max(np.abs(z - previous)) < 1e-10:
            return z
    raise RuntimeError('two-way demeaning did not converge')

def correlations(a, groups=()):
    c = corr(rank_residuals(a, groups))
    return {f'{NAMES[i]}__{NAMES[j]}': float(c[i, j]) for (i, j) in PAIRS}

def auc(y, x):
    (y, x) = (np.asarray(y, dtype=int), np.asarray(x, dtype=float))
    (n1, n0) = (int(y.sum()), int((1 - y).sum()))
    if not n1 or not n0:
        return (np.nan, 0)
    return (float((rankdata(x)[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)), n1 * n0)
