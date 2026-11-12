"""Horizon-parameterized port of the per-split position-centered dictionary fit.

Source methods: scientific-synthesis mean-controls mean_math.py (spectrum, fixed_random,
fit_centered position loop) + action-readout math_core.py hierarchical_weights. The ONLY
change is the action-position count POS (10 for pi0.5, 50 for pi0); the eigh/QR/sign-fix
and the (task_id,family,variant_id,noise_id,token_position) nested weighting are identical.
These build the `new` dict consumed by readout_hn.predict_split: position_delta__basis32,
position_clean__basis32, curve_random32, position_mean. Pure CPU numpy; no cloud/model.
"""
from collections import defaultdict
import hashlib
import math
import numpy as np

LEVELS = ('task_id', 'family', 'variant_id', 'noise_id', 'token_position')
CURVE_RANDOM_SEED = 'plus-mean-controls-random-rankcurve-v1'


def require(ok, code):
    if not ok:
        raise ValueError(code)


def hierarchical_weights(rows):
    """Equal task, then family, variant, noise and token weight; sums to 1."""
    require(bool(rows), 'nonempty_training_rows')
    keys = [tuple(r[k] for k in LEVELS) for r in rows]
    require(len(keys) == len(set(keys)), 'unique_task_family_variant_noise_token')
    counts = [defaultdict(set) for _ in LEVELS]
    for key in keys:
        for depth in range(len(LEVELS)):
            counts[depth][key[:depth]].add(key[depth])
    weights = []
    for key in keys:
        w = 1.0
        for depth in range(len(LEVELS)):
            w /= len(counts[depth][key[:depth]])
        weights.append(w)
    result = np.asarray(weights, np.float64)
    require(np.isfinite(result).all() and np.all(result > 0)
            and math.isclose(float(result.sum()), 1.0, rel_tol=0, abs_tol=1e-12), 'normalized_nested_weights')
    return result


def _sign_fixed(basis):
    b = np.array(basis, dtype=np.float64, copy=True)
    for col in range(b.shape[1]):
        if b[np.argmax(np.abs(b[:, col])), col] < 0:
            b[:, col] *= -1
    return b


def _spectrum(x, w, rank=32):
    second = x.T @ (w[:, None] * x)
    values, vectors = np.linalg.eigh(second)
    order = np.argsort(values)[::-1]
    return _sign_fixed(vectors[:, order[:rank]])


def fixed_random(channels, layer, rank, seed_text=CURVE_RANDOM_SEED):
    seed = int.from_bytes(hashlib.sha256((seed_text + '|' + str(layer)).encode()).digest()[:8], 'little')
    z = np.random.default_rng(seed).standard_normal((channels, rank))
    q, _ = np.linalg.qr(z, mode='reduced')
    return _sign_fixed(q)


def fit_new_bases(train_delta_layer, train_clean_layer, train_rows, layer, pos):
    """Fit the position-centered new dictionaries for one (family,layer,fold) training split.

    train_delta_layer: [N_tr, pos, 1024] = DeltaH (H(T)-H(R)) at j0 for this layer.
    train_clean_layer: [N_tr, pos, 1024] = clean H(R) at j0 for this layer.
    train_rows: list of row dicts (task_id, family, row_id, noise_label), one per case.
    Returns the `new` dict for readout_hn.predict_split.
    """
    x = np.asarray(train_delta_layer, np.float64)
    r = np.asarray(train_clean_layer, np.float64)
    require(x.ndim == 3 and x.shape[1] == pos and x.shape[2] == 1024 and x.shape == r.shape, 'delta_clean_ABI')
    n = x.shape[0]
    require(len(train_rows) == n, 'rows_match_cases')
    # Flatten to (N_tr*pos, 1024) with per-(case,position) records for nested weighting.
    recs = []
    for row in train_rows:
        for position in range(pos):
            recs.append({'task_id': row['task_id'], 'family': row['family'], 'variant_id': row['row_id'],
                         'noise_id': int(row['noise_label']), 'token_position': position})
    w = hierarchical_weights(recs)                       # [N_tr*pos], sums to 1
    xf = x.reshape(n * pos, 1024)
    rf = r.reshape(n * pos, 1024)
    p = np.asarray([position for _ in range(n) for position in range(pos)], np.int64)
    require(set(p.tolist()) == set(range(pos)), 'all_positions_present')
    # per-position weighted means
    pm = np.empty((pos, 1024)); pr = np.empty((pos, 1024)); pw = np.empty(pos)
    for position in range(pos):
        sel = p == position
        pw[position] = w[sel].sum()
        pm[position] = (w[sel] @ xf[sel]) / pw[position]
        pr[position] = (w[sel] @ rf[sel]) / pw[position]
    position_delta = _spectrum(xf - pm[p], w, 32)
    position_clean = _spectrum(rf - pr[p], w, 32)
    curve_random32 = fixed_random(1024, layer, 32)
    return {'position_delta__basis32': position_delta, 'position_clean__basis32': position_clean,
            'curve_random32': curve_random32, 'position_mean': pm,
            'clean_position_mean': pr, 'position_weight': pw}
