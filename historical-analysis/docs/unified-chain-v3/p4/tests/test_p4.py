"""Synthetic tests for the P4 numerical core (no real data). Run: python3 test_p4.py"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'cloud', 'worker'))
from p4lib import (PCA, bf16_bits_to_f32, decomposition_logs, eef, group_folds, head_geometry, hier_mean,  # noqa: E402
                   rsvd_basis, step_terms, task_bootstrap)
import p4models  # noqa: E402

rng = np.random.default_rng(0)


def test_bf16():
    x = np.array([1.0, -2.5, 3.140625, 0.0], np.float32)
    bits = (x.view(np.uint32) >> 16).astype(np.uint16)
    assert np.array_equal(bf16_bits_to_f32(bits), x)


def test_decomposition_identity_and_geometry():
    W7 = rng.standard_normal((1024, 7)) * 0.02
    Ur, s, r, er = head_geometry(W7)
    assert r == 7 and Ur.shape == (1024, 7)
    dr = rng.standard_normal((10, 5, 1024)).astype(np.float32)
    rc = rng.standard_normal((10, 5, 1024)).astype(np.float32)
    lin = dr.astype(np.float64) @ W7
    u = (lin * -0.1 * (1 + 0.01 * rng.standard_normal(lin.shape))).astype(np.float32)
    t = step_terms(dr, rc, u, W7, Ur)
    d, ok, _ = decomposition_logs(t['R'][None], t['A'][None], t['L'][None], t['V'][None])
    assert ok.all()
    assert abs(d['T'][0] - (d['rho'][0] + d['g'][0] + d['nu'][0])) < 1e-9
    # readable fraction of isotropic noise ~ 7/1024
    assert abs(t['A'].sum() / t['R'].sum() - 7 / 1024) < 0.003
    # mapping residual small when u = dt * dr W7 (1% multiplicative noise)
    assert (t['M'] / t['V']).max() < 0.01


def test_zero_denominator_flag():
    R = np.ones((3, 10)); A = np.ones((3, 10)); L = np.ones((3, 10)); V = np.ones((3, 10))
    A[1, 0] = 0
    V[2, 9] = np.nan
    _, ok, _ = decomposition_logs(R, A, L, V)
    assert ok.tolist() == [True, False, False]


def test_hier_mean_and_bootstrap():
    task = np.array(['a', 'a', 'b', 'b', 'b', 'c'])
    cond = np.array(['x', 'y', 'x', 'x', 'y', 'x'])
    v = np.array([1., 2., 3., 5., 6., 9.])
    # cond x: tasks a(1), b(4), c(9) -> 14/3 ; cond y: a(2), b(6) -> 4 ; overall (14/3 + 4)/2
    assert abs(hier_mean(v, task, cond) - (14 / 3 + 4) / 2) < 1e-12
    reps = task_bootstrap(lambda w: hier_mean(v, task, cond, w), task, B=50)
    assert reps.shape == (50,) and np.isfinite(reps).all()


def test_rsvd_matches_svd():
    X = rng.standard_normal((300, 50)) @ np.diag(np.linspace(5, 0.1, 50)) @ np.linalg.qr(rng.standard_normal((50, 50)))[0]
    Xc = X - X.mean(0)
    V = rsvd_basis(Xc.astype(np.float32), 5)
    Vt = np.linalg.svd(Xc, full_matrices=False)[2][:5]
    assert np.allclose(np.abs((V @ Vt.T).diagonal()), 1, atol=1e-3)


def make_data(n_tasks=20, per=40, null_signal=0.0, seed=1):
    r = np.random.default_rng(seed)
    n = n_tasks * per
    task = np.repeat([f't{i:02d}' for i in range(n_tasks)], per)
    conds = np.array(['c0', 'c1', 'c2'])
    cond = conds[r.integers(0, 3, n)]
    anchor = np.array(['q0', 'lm1'])[r.integers(0, 2, n)]
    W7 = r.standard_normal((1024, 7)) * 0.03
    Ur, _, _, _ = head_geometry(W7)
    p = 5 * 1024
    basis = np.linalg.qr(r.standard_normal((p, 12)))[0].T          # 12 strong structured directions
    Bn = basis.reshape(12, 5, 1024)
    basis_null = (Bn - (Bn @ Ur) @ Ur.T).reshape(12, p)               # their null-space parts (no readable content)
    basis_null /= np.linalg.norm(basis_null, axis=1, keepdims=True)
    dr0 = (r.standard_normal((n, 12)) * np.linspace(30, 8, 12)) @ basis_null + r.standard_normal((n, p))
    dr0 = dr0.reshape(n, 5, 1024).astype(np.float32)
    lin = dr0.astype(np.float64) @ W7                      # (n,5,7)
    u0 = (-0.1 * lin).reshape(n, 35)
    a0 = (dr0.astype(np.float64) @ Ur).reshape(n, 35)
    n0 = (dr0 - (dr0.astype(np.float64) @ Ur @ Ur.T)).reshape(n, p).astype(np.float32)
    dir_null = basis_null[2]                                           # a high-variance, purely null direction
    proj = n0 @ dir_null.astype(np.float32)
    ysig = np.outer(proj, r.standard_normal(35))
    ysig *= np.linalg.norm(7.2 * u0) / np.linalg.norm(ysig)                    # comparable energy to the u0 part
    Y = 7.2 * u0 + null_signal * ysig + 0.02 * np.linalg.norm(u0, axis=1).mean() * r.standard_normal((n, 35))
    R0 = (dr0.astype(np.float64) ** 2).sum((1, 2))
    data = dict(Y=Y, u0=u0, a0=a0, n0=n0, dr0full=dr0.reshape(n, p), R0=R0, task=task, cond=cond, anchor=anchor,
                cell=np.char.add(np.char.add(cond, '|'), anchor), cond_levels=sorted(set(cond)),
                anchor_levels=sorted(set(anchor)))
    data['cell_levels'] = sorted(set(data['cell']))
    return data


def run_cv(data):
    folds = group_folds(data['task'], 5, 20261008)
    P = {}
    for f in range(5):
        tr, ev = np.where(folds != f)[0], np.where(folds == f)[0]
        data['_outer_fold'] = f
        preds, _ = p4models.predict_remaining(data, tr, ev)
        for k, v in preds.items():
            P.setdefault(k, np.zeros_like(data['Y']))[ev] = v
    return {k: eef(data['Y'], v) for k, v in P.items()}


def test_models_detect_and_reject_null_signal():
    no = run_cv(make_data(null_signal=0.0))
    yes = run_cv(make_data(null_signal=1.0))
    assert no['M2L'] > 0.9 and abs(no['M3'] - no['M2L']) < 0.02, no
    assert yes['M3'] - yes['M2L'] > 0.05, yes
    assert no['M2'] < no['M2L']  # 9*u0 over-predicts the 7.2 gain


def test_no_test_leakage_in_blocks():
    d = make_data()
    tr, ev = np.arange(400), np.arange(400, 800)
    b1 = p4models.Blocks(d, tr, z_keys=('u0',))
    d2 = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in d.items()}
    d2['u0'][ev] += 100.0
    d2['n0'][ev] += 50.0
    b2 = p4models.Blocks(d2, tr, z_keys=('u0',))
    assert np.allclose(b1.z['u0'][0], b2.z['u0'][0]) and np.allclose(b1.pca.V, b2.pca.V)


if __name__ == '__main__':
    for name, fn in sorted(globals().items()):
        if name.startswith('test_'):
            fn()
            print('PASS', name)
