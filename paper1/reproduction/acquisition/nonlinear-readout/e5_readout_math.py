"""E5 readout-capacity math: random-feature ridge + small MLP on the SAME inputs.

Design: capacity is the ONLY change from the affine current-basis readout. We import
the frozen fitting/scoring/split/weight/bootstrap math (producer-reference legacy_math.py)
by explicit path and reuse it unchanged for standardization, dual ridge, splits, case
weights, component scoring and the whole-task-component bootstrap, so splits/targets/
weights/intervals are identical. We ADD only two readout families on top of each
standardized feature block, plus deterministic hyperparameter rules fixed on training
rows. numpy + stdlib only (the Baidu CPU capsule has numpy).

No cloud, model, credential or file access here besides importing legacy_math by path.
"""
from __future__ import annotations
import hashlib
import importlib.util
import sys
from pathlib import Path
import numpy as np


def load_legacy(path):
    """Import the frozen legacy_math.py by explicit path (no sys.path injection)."""
    path = Path(path).resolve()
    name = "e5_frozen_legacy_math"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod            # dataclass needs the module registered
    spec.loader.exec_module(mod)
    return mod


def _seed(text):
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "little")


# ----------------------------------------------------------------------------
# Random-feature ridge: RBF random Fourier features + the frozen weighted dual ridge.
# ----------------------------------------------------------------------------
def median_bandwidth(z, rng, cap=512):
    """Median pairwise Euclidean distance on standardized training rows (train only)."""
    n = len(z)
    if n > cap:
        idx = rng.choice(n, cap, replace=False)
        z = z[idx]
    d = np.sqrt(np.maximum(0.0, ((z[:, None, :] - z[None, :, :]) ** 2).sum(-1)))
    iu = np.triu_indices(len(z), k=1)
    med = float(np.median(d[iu])) if len(iu[0]) else 1.0
    return med if med > 0 else 1.0


def _rff(z, omega, bvec, D):
    return np.sqrt(2.0 / D) * np.cos(z @ omega + bvec)


def fit_rfridge(M, Xtr, Ytr, w, seed_text, cv_groups, *, D=2048, lambdas=(1e-3, 1e-2, 1e-1, 1.0, 10.0),
                n_cv=5, transform=None):
    """Return (predict_fn, proof). omega/b fixed by seed; sigma + lambda from TRAIN only."""
    xt = M.fit_transform(Xtr, w) if transform is None else transform
    ztr = xt.apply(Xtr)
    rng = np.random.default_rng(_seed("e5-rfridge|" + seed_text))
    sigma = median_bandwidth(ztr, rng)
    omega = rng.standard_normal((ztr.shape[1], D)) / sigma
    bvec = rng.uniform(0.0, 2.0 * np.pi, D)
    phi_tr = _rff(ztr, omega, bvec, D)
    lam = _select_lambda(M, phi_tr, Ytr, w, cv_groups, lambdas, "e5-rfridge-cv|" + seed_text, n_cv)
    model = M.fit_weighted_affine_ridge(phi_tr, Ytr, w, lam)

    def predict(Xte):
        return model.predict(_rff(xt.apply(Xte), omega, bvec, D))

    return predict, {"family": "rfridge", "D": int(D), "sigma": float(sigma), "lambda": float(lam),
                     "feature_dim_in": int(Xtr.shape[1])}


def _select_lambda(M, phi, Y, w, groups, lambdas, seed_text, n_cv):
    """Grouped K-fold CV over TRAIN task components (a component stays whole)."""
    uniq = sorted(set(groups))
    if len(uniq) < 2:
        return float(lambdas[len(lambdas) // 2])
    rng = np.random.default_rng(_seed(seed_text))
    order = rng.permutation(len(uniq))
    fold_of = {uniq[order[i]]: i % n_cv for i in range(len(uniq))}
    assign = np.array([fold_of[g] for g in groups])
    best, best_lam = np.inf, lambdas[0]
    for lam in lambdas:
        total, counted = 0.0, 0
        for k in range(min(n_cv, len(uniq))):
            tr = np.where(assign != k)[0]
            te = np.where(assign == k)[0]
            if len(tr) < 2 or len(te) < 1:
                continue
            wt = w[tr] / w[tr].sum()
            model = M.fit_weighted_affine_ridge(phi[tr], Y[tr], wt, lam)
            pred = model.predict(phi[te])
            we = w[te] / w[te].sum()
            total += float(we @ np.mean((pred - Y[te]) ** 2, axis=1))
            counted += 1
        if counted and total / counted < best:
            best, best_lam = total / counted, lam
    return float(best_lam)


# ----------------------------------------------------------------------------
# Small MLP: one hidden ReLU layer, early stopping on a grouped training val split.
# ----------------------------------------------------------------------------
def fit_mlp(M, Xtr, Ytr, w, seed_text, groups, *, hidden=256, max_epochs=500, patience=25,
            val_frac=0.2, lr=1e-3, transform=None):
    xt = M.fit_transform(Xtr, w) if transform is None else transform
    z = xt.apply(Xtr)
    uniq = sorted(set(groups))
    rng = np.random.default_rng(_seed("e5-mlp|" + seed_text))
    # Deterministic grouped validation split (by task component; never a test component).
    n_val = max(1, int(round(val_frac * len(uniq)))) if len(uniq) >= 2 else 0
    val_comps = set(np.array(uniq)[rng.permutation(len(uniq))[:n_val]].tolist()) if n_val else set()
    is_val = np.array([g in val_comps for g in groups])
    tr_i = np.where(~is_val)[0]
    va_i = np.where(is_val)[0] if is_val.any() else tr_i
    meanY = (w[tr_i] / w[tr_i].sum()) @ Ytr[tr_i]
    Yc = Ytr - meanY
    din, dout = z.shape[1], Ytr.shape[1]
    scale1 = np.sqrt(2.0 / din)
    W1 = rng.standard_normal((din, hidden)) * scale1
    b1 = np.zeros(hidden)
    W2 = rng.standard_normal((hidden, dout)) * np.sqrt(1.0 / hidden)
    b2 = np.zeros(dout)
    params = [W1, b1, W2, b2]
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    wtr = w[tr_i] / w[tr_i].sum()
    wva = w[va_i] / w[va_i].sum()
    ztr, ytr = z[tr_i], Yc[tr_i]
    zva, yva = z[va_i], Yc[va_i]
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    best_val, best, wait = np.inf, [p.copy() for p in params], 0

    def forward(zz, P):
        h = np.maximum(0.0, zz @ P[0] + P[1])
        return h, h @ P[2] + P[3]

    for t in range(1, max_epochs + 1):
        h, out = forward(ztr, params)
        resid = out - ytr                                   # [n,dout]
        gout = (2.0 * wtr[:, None] * resid)                 # weighted MSE grad
        gW2 = h.T @ gout
        gb2 = gout.sum(0)
        gh = gout @ params[2].T
        gh[h <= 0] = 0.0
        gW1 = ztr.T @ gh
        gb1 = gh.sum(0)
        grads = [gW1, gb1, gW2, gb2]
        for i, (p, g) in enumerate(zip(params, grads)):
            m[i] = beta1 * m[i] + (1 - beta1) * g
            v[i] = beta2 * v[i] + (1 - beta2) * (g * g)
            mhat = m[i] / (1 - beta1 ** t)
            vhat = v[i] / (1 - beta2 ** t)
            p -= lr * mhat / (np.sqrt(vhat) + eps)
        _, vout = forward(zva, params)
        vmse = float(wva @ np.mean((vout - yva) ** 2, axis=1))
        if vmse < best_val - 1e-12:
            best_val, best, wait = vmse, [p.copy() for p in params], 0
        else:
            wait += 1
            if wait >= patience:
                break
    stop_epoch = t

    def predict(Xte):
        _, out = forward(xt.apply(Xte), best)
        return out + meanY

    return predict, {"family": "mlp", "hidden": int(hidden), "stop_epoch": int(stop_epoch),
                     "best_val_mse": float(best_val), "val_components": int(len(val_comps)),
                     "train_components": int(len(uniq) - len(val_comps)),
                     "feature_dim_in": int(Xtr.shape[1])}


# ----------------------------------------------------------------------------
# Contrast + sign bookkeeping (mirrors the frozen affine contrast exactly).
# ----------------------------------------------------------------------------
def ci_sign(ci):
    if ci is None:
        return "undef"
    lo, hi = ci
    if lo > 0:
        return "pos"
    if hi < 0:
        return "neg"
    return "span0"


def point_sign(x):
    if x is None:
        return "undef"
    return "pos" if x > 0 else ("neg" if x < 0 else "zero")
