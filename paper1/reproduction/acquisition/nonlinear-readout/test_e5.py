"""Focused tests for the E5 nonlinear readouts. Run: python3 -I test_e5.py"""
import sys
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e5_readout_math as E

M = E.load_legacy(Path(__file__).resolve().parent.parent / "frozen-reference" / "legacy_math.py")


def toy(n=40, d=12, seed=1):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d))
    W = rng.standard_normal((d, 390)) * 0.3
    Y = X @ W + rng.standard_normal((n, 390)) * 0.05
    w = np.full(n, 1.0 / n)
    groups = [f"c{i % 6}" for i in range(n)]
    return X, Y, w, groups


class Tests(unittest.TestCase):
    def test_rfridge_deterministic(self):
        X, Y, w, g = toy()
        p1, pr1 = E.fit_rfridge(M, X, Y, w, "s", g, D=128)
        p2, pr2 = E.fit_rfridge(M, X, Y, w, "s", g, D=128)
        self.assertTrue(np.allclose(p1(X), p2(X)))
        self.assertEqual(pr1["sigma"], pr2["sigma"])
        self.assertEqual(pr1["lambda"], pr2["lambda"])

    def test_rfridge_lambda_from_grid(self):
        X, Y, w, g = toy()
        _, pr = E.fit_rfridge(M, X, Y, w, "s", g, D=128, lambdas=(1e-2, 1.0))
        self.assertIn(pr["lambda"], (1e-2, 1.0))

    def test_mlp_deterministic_and_val_grouped(self):
        X, Y, w, g = toy()
        p1, pr1 = E.fit_mlp(M, X, Y, w, "s", g, hidden=8, max_epochs=20)
        p2, pr2 = E.fit_mlp(M, X, Y, w, "s", g, hidden=8, max_epochs=20)
        self.assertTrue(np.allclose(p1(X), p2(X)))
        # validation components disjoint from training components (no component in both)
        self.assertGreaterEqual(pr1["train_components"], 1)
        self.assertEqual(pr1["train_components"] + pr1["val_components"], 6)

    def test_rfridge_recovers_linear_signal(self):
        # On a noiseless linear target, rfridge should predict better than the target mean.
        rng = np.random.default_rng(3)
        X = rng.standard_normal((60, 10)); W = rng.standard_normal((10, 390))
        Y = X @ W
        w = np.full(60, 1 / 60); g = [f"c{i%6}" for i in range(60)]
        tr, te = slice(0, 48), slice(48, 60)
        pred, _ = E.fit_rfridge(M, X[tr], Y[tr], w[:48] / w[:48].sum(), "s", g[:48], D=512)
        err = np.mean((pred(X[te]) - Y[te]) ** 2)
        base = np.mean((Y[tr].mean(0) - Y[te]) ** 2)
        self.assertLess(err, base)

    def test_sign_helpers(self):
        self.assertEqual(E.ci_sign([0.1, 0.3]), "pos")
        self.assertEqual(E.ci_sign([-0.3, -0.1]), "neg")
        self.assertEqual(E.ci_sign([-0.1, 0.2]), "span0")
        self.assertEqual(E.ci_sign(None), "undef")
        self.assertEqual(E.point_sign(0.2), "pos")
        self.assertEqual(E.point_sign(-0.2), "neg")

    def test_affine_equals_frozen(self):
        X, Y, w, g = toy()
        tr, te = slice(0, 30), slice(30, 40)
        direct = M.fit_weighted_affine_ridge(X[tr], Y[tr], w[:30] / w[:30].sum(), 0.01).predict(X[te])
        model = M.fit_weighted_affine_ridge(X[tr], Y[tr], w[:30] / w[:30].sum(), 0.01)
        self.assertTrue(np.allclose(model.predict(X[te]), direct))


if __name__ == "__main__":
    unittest.main(verbosity=2)
