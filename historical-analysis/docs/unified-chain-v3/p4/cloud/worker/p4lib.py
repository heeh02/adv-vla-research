"""P4 numerical core (numpy only), as frozen in reports/PREREG-P4.md (sha256 dbab0f6d...).

Used unchanged by the cloud extraction/analysis workers and by local synthetic tests.
Conventions: E window = action positions 0..4 x output dims 0..6; dt = -0.1 (x_{k+1} = x_k + dt v_k);
energies are sums of squares over the window (no averaging). Ridge uses per-sample scaling:
B = (X'X + lam * n * I)^-1 X'Y on training-standardized features (hyperparameter convention, see README).
"""
import numpy as np

DT = -0.1
NSTEP = 10
POS = slice(0, 5)
DIMS = slice(0, 7)


def bf16_bits_to_f32(a):
    a = np.ascontiguousarray(a)
    if a.dtype.kind == 'V' or a.dtype == np.uint16 or a.dtype.itemsize == 2 and a.dtype.kind != 'f':
        return (a.view(np.uint16).astype(np.uint32) << 16).view(np.float32)
    return a.astype(np.float32)


# ---------------------------------------------------------------- head geometry
def head_geometry(W7):
    """W7: (1024, 7) float64. Returns readable basis U_r (1024, r), singular values, numerical rank, entropy rank."""
    U, s, _ = np.linalg.svd(np.asarray(W7, dtype=np.float64), full_matrices=False)
    r = int((s > 1e-3 * s[0]).sum())
    p = s ** 2 / (s ** 2).sum()
    erank = float(np.exp(-(p[p > 0] * np.log(p[p > 0])).sum()))
    return U[:, :r], s, r, erank


# ---------------------------------------------------------------- per-trace step energies
def step_terms(dr, rc, u, W7, Ur, vdiff_stored=None):
    """dr, rc: (10, 5, 1024) readout delta / clean readout (E positions); u: (10, 5, 7) x_t-increment difference.
    Returns dict of (10,) float64 arrays: R, A, L, V, M (mapping residual energy), Rc, Ac, and Vv if vdiff given."""
    d = dr.astype(np.float64)
    c = rc.astype(np.float64)
    W = np.asarray(W7, dtype=np.float64)
    lin = d @ W                                    # (10,5,7) linear recompute of the velocity difference
    veff = u.astype(np.float64) / DT               # effective velocity difference
    out = dict(R=(d ** 2).sum((1, 2)), A=((d @ Ur) ** 2).sum((1, 2)), L=(lin ** 2).sum((1, 2)),
               V=(veff ** 2).sum((1, 2)), M=((veff - lin) ** 2).sum((1, 2)),
               Rc=(c ** 2).sum((1, 2)), Ac=((c @ Ur) ** 2).sum((1, 2)))
    if vdiff_stored is not None:
        out['Vv'] = (vdiff_stored.astype(np.float64) ** 2).sum((1, 2))
    return out


def decomposition_logs(R, A, L, V):
    """Per-pair first/last log changes. Inputs (n, 10). Returns dict of (n,) and a validity mask."""
    ends = [0, NSTEP - 1]
    ok = np.ones(len(R), bool)
    for X in (R, A, L, V):
        ok &= np.all(np.isfinite(X[:, ends]) & (X[:, ends] > 0), axis=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        lT = np.log(V / R)
        lrho = np.log(A / R)
        lg = np.log(L / A)
        lnu = np.log(V / L)
    d = {k: x[:, -1] - x[:, 0] for k, x in dict(T=lT, rho=lrho, g=lg, nu=lnu).items()}
    return d, ok, dict(T=lT, rho=lrho, g=lg, nu=lnu)


# ---------------------------------------------------------------- hierarchical aggregation + task bootstrap
def hier_mean_parts(values, task, cond):
    """values: (n,) or (n, k). Returns {cond: {task: cell mean}} as nested dicts of arrays."""
    v = np.asarray(values, dtype=np.float64)
    out = {}
    for c in np.unique(cond):
        mc = cond == c
        out[c] = {t: v[mc & (task == t)].mean(0) for t in np.unique(task[mc])}
    return out


def hier_mean(values, task, cond, task_weights=None):
    """Condition-balanced mean: pairs -> (task, condition) cell mean -> tasks equal within condition -> conditions equal."""
    return hier_from_parts(hier_mean_parts(values, task, cond), task_weights)


def hier_from_parts(parts, task_weights=None):
    """Same aggregation from precomputed cell means (fast task-bootstrap reweighting)."""
    cm = []
    for c, cells in parts.items():
        ts = list(cells)
        w = np.array([1.0 if task_weights is None else task_weights.get(t, 0.0) for t in ts])
        if w.sum() == 0:
            continue
        cm.append(np.tensordot(w / w.sum(), np.stack([cells[t] for t in ts]), axes=1))
    return np.mean(np.stack(cm), 0)


def task_bootstrap(stat_fn, tasks, B=2000, seed=20261009):
    """stat_fn(weights: dict task->multiplicity) -> array. Returns (B, ...) replicate array."""
    rng = np.random.default_rng(seed)
    tasks = np.asarray(sorted(set(tasks)))
    reps = []
    for _ in range(B):
        draw = rng.choice(tasks, len(tasks))
        u, cnt = np.unique(draw, return_counts=True)
        reps.append(stat_fn(dict(zip(u, cnt.astype(float)))))
    return np.asarray(reps)


# ---------------------------------------------------------------- prediction utilities
def standardize_fit(X):
    mu = X.mean(0)
    sd = X.std(0)
    sd = np.where(sd > 1e-12, sd, 1.0)
    return mu, sd


def ridge_fit(X, Y, lam):
    n, p = X.shape
    G = X.T @ X + lam * n * np.eye(p)
    return np.linalg.solve(G, X.T @ Y)


def rsvd_basis(Xc, k, seed=20261009, oversample=16, n_iter=4):
    """Top-k right singular vectors (k, p) of centered Xc via randomized SVD (Halko et al.)."""
    rng = np.random.default_rng(seed)
    n, p = Xc.shape
    kk = min(k + oversample, min(n, p))
    Q = np.linalg.qr(Xc @ rng.standard_normal((p, kk)).astype(Xc.dtype))[0]
    for _ in range(n_iter):
        Q = np.linalg.qr(Xc.T @ Q)[0]
        Q = np.linalg.qr(Xc @ Q)[0]
    _, _, Vt = np.linalg.svd(Q.T @ Xc, full_matrices=False)
    return Vt[:min(k, Vt.shape[0])]


class PCA:
    def __init__(self, X, k, seed=20261009):
        self.mu = X.mean(0)
        self.V = rsvd_basis((X - self.mu).astype(np.float32), k, seed=seed).astype(np.float32)

    def scores(self, X, m):
        return (X - self.mu).astype(np.float32) @ self.V[:m].T


def eef(Y, P):
    den = (Y ** 2).sum()
    return float(1 - ((Y - P) ** 2).sum() / den) if den > 0 else np.nan


def cos_rows(P, Y):
    a, b = np.linalg.norm(P, axis=1), np.linalg.norm(Y, axis=1)
    out = np.full(len(Y), np.nan)
    ok = (a > 0) & (b > 0)
    out[ok] = (P[ok] * Y[ok]).sum(1) / a[ok] / b[ok]
    return out


def group_folds(groups, k, seed):
    """Deterministic task-grouped fold labels: permute sorted unique groups, assign i % k."""
    g = np.asarray(sorted(set(groups)))
    perm = np.random.default_rng(seed).permutation(len(g))
    fold_of = {g[i]: j % k for j, i in enumerate(perm)}
    return np.array([fold_of[x] for x in groups])


def cell_means(Y, cells_train, cells_eval, fallback_train, fallback_eval, min_n=20):
    """Mean of Y per cell on training; cells with < min_n training rows fall back to the fallback cell mean."""
    out = np.zeros((len(cells_eval), Y.shape[1]))
    fb = {c: Y[fallback_train == c].mean(0) for c in np.unique(fallback_train)}
    pooled = Y.mean(0)
    stats = {c: (Y[cells_train == c].mean(0), (cells_train == c).sum()) for c in np.unique(cells_train)}
    for i, (c, f) in enumerate(zip(cells_eval, fallback_eval)):
        m, n = stats.get(c, (None, 0))
        out[i] = m if n >= min_n else fb.get(f, pooled)
    return out
