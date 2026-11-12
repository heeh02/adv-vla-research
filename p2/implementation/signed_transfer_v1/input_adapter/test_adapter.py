"""Synthetic arithmetic and source-metadata tests, no real neural arrays."""
from pathlib import Path
from types import SimpleNamespace
import importlib.util
import json
import tempfile
import unittest

import numpy as np
import adapter


class Field:
    def __init__(self, a):
        self.a = np.asarray(a)
    def value(self):
        return self.a


class AdapterTests(unittest.TestCase):
    def test_legacy_capsule_and_exact_unique_manifest(self):
        with adapter.legacy_modules(adapter.HERE / "legacy") as modules:
            _, _, _, planned = modules["prepare_runtime"].verify(adapter.HERE / "legacy")
        doc = json.loads((adapter.HERE / "INPUTS.json").read_bytes())
        rows = {r["uri"]: r for r in doc["input_objects"]}
        self.assertEqual(set(rows), set(planned["objects"]))
        self.assertEqual(len(rows), 1500)
        self.assertEqual(sum(r["bytes"] for r in rows.values()), 769677897)
        for uri, row in planned["objects"].items():
            self.assertEqual({k: row[k] for k in ("bytes", "sha256")},
                             {k: rows[uri][k] for k in ("bytes", "sha256")})

    def fixture(self):
        e = np.zeros((2, 256)); e[0, 0] = 2.; e[1, 1] = -3.
        p = np.zeros((8, 10, 12)); p[:, :, 0] = .125; p[:, :, 1] = .25; p[:, :, 2:] = .0625
        p[1, 4, 0] = .0625; p[1, 4, 2] += .0625
        w = np.zeros((8, 256, 1024))
        for h in range(8):
            w[h, 0, h] = h + 1; w[h, 1, 9 + h] = h - 2
        g = np.linspace(.2, .8, 1024)[None, None, :]
        context = np.arange(13, dtype=float)
        features = {"current": context, "pooled": context + np.r_[np.zeros(9), np.ones(4)],
                    "norm": context + 2}
        def local_write(prob, edit, wo, gate):
            a = np.einsum("hpn,nd->phd", prob[:, :, :2], edit)
            return (np.einsum("phd,hdf->pf", a, wo.value()) * gate.value()[0, 0]).ravel()
        P = SimpleNamespace(masked_mean=lambda x, mask: x, local_write=local_write)
        model = SimpleNamespace(pooled=p, wo=Field(w), features=lambda q, arm: features)
        query = SimpleNamespace(p_native=Field(p[None, None]), gate=Field(g),
                                mask=Field(np.ones((1, 10, 12), bool)), edit=lambda arm: e)
        row = SimpleNamespace(key=("synthetic", 11, "shared", "KT"), split="DEV",
                              current=features["current"], pooled=features["pooled"],
                              effect=np.zeros((10, 7)), zero_actual_E=False, provenance={})
        return SimpleNamespace(P=P), model, query, row

    def test_signed_position_head_and_full_denominator_retained(self):
        d, model, query, row = self.fixture()
        out = adapter.expand_row(d, model, query, row)
        self.assertEqual(out["current_headpos_read"].shape, (8, 10, 256))
        self.assertEqual(out["current_gated_write"].shape, (10, 1024))
        self.assertEqual(out["current_headpos_norms"].shape, (80,))
        self.assertEqual(out["current_headpos_read"][0, 0, 0], .25)
        self.assertEqual(out["current_headpos_read"][0, 0, 1], -.75)
        self.assertEqual(out["current_headpos_read"][1, 4, 0], .125)
        self.assertFalse(np.array_equal(out["current_headpos_read"][1, 4], out["current_headpos_read"][1, 3]))
        # Prefix mass=.375 is not renormalized to one.
        self.assertNotEqual(out["current_headpos_read"][0, 0, 0], .25 / .375)

    def test_no_effect_labels_enter_expanded_features(self):
        d, model, query, row = self.fixture()
        first = adapter.expand_row(d, model, query, row)
        row.effect = np.full((10, 7), 987654321.)
        second = adapter.expand_row(d, model, query, row)
        for name in ("context9", "current4", "pooled4", "originalnorm4", "current_headpos_read",
                     "current_gated_write", "current_headpos_norms"):
            self.assertTrue(np.array_equal(first[name], second[name]), name)


if __name__ == "__main__":
    unittest.main()
