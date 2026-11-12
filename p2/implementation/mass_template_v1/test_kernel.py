"""Synthetic-only regressions; no stored research arrays are opened."""
import copy
import json
import unittest

import numpy as np

import kernel as K


def pack(corners):
    g = np.einsum("ahpd,bhpd->hpab", corners, corners, optimize=True)
    i, j = np.triu_indices(4)
    return g[..., i, j]


def fixture(heads=2):
    p = np.zeros((2, heads, 10, 16), np.float64)
    p[..., :6] = .05
    p[..., 6:] = .02  # Original full row mass .5; never normalized to one.
    z = np.array([[2, 0, 0, 1, 0, 0], [-1, 0, 1, 0, 0, 0],
                  [0, 3, 0, 1, 0, 0], [1, -2, 0, 0, 1, 0],
                  [1, 0, 0, 0, 0, 2], [0, 1, 0, 0, 1, -2]], np.float64)
    return p, z


def saved(p, z, defined=None):
    """Independent small P/Z reference, emulating the frozen four-corner ABI."""
    h, positions = p.shape[1:3]
    if defined is None:
        defined = np.ones((h, positions), bool)
    mass = np.stack([p[..., 2*g:2*g+2].sum(-1) for g in range(3)], axis=-1)
    contributions = np.full((4, h, positions, 3, z.shape[1]), np.nan)
    group_defined = np.zeros((4, h, positions, 3), bool)
    for ci, (a, b) in enumerate(((0, 0), (1, 0), (0, 1), (1, 1))):
        for hi in range(h):
            for pos in range(positions):
                for group in range(3):
                    if not defined[hi, pos]:
                        continue
                    source, wanted = mass[b, hi, pos, group], mass[a, hi, pos, group]
                    use = slice(2*group, 2*group+2)
                    if source > 0:
                        weights = wanted * p[b, hi, pos, use] / source
                    elif wanted == 0:
                        weights = np.zeros(2)
                    else:
                        continue
                    contributions[ci, hi, pos, group] = weights @ z[use]
                    group_defined[ci, hi, pos, group] = True
    corners = contributions.sum(-2)
    cd = group_defined.all(-1)
    bases = (np.eye(z.shape[1])[:, :3], np.eye(z.shape[1])[:, -3:])
    projections = []
    for b in bases:
        projected = (corners @ b) @ b.T
        projections.append(np.stack((pack(projected), pack(corners-projected))))
    args = dict(raw_gram_upper=pack(corners), masses=mass,
                corner_defined=cd, projection_gram_upper=np.stack(projections),
                group_corner_defined=group_defined, flow_pair=(0, 9))
    return args, corners


def row(result, view="full", window="full10"):
    return next(r for r in result["rows"] if (r["view"], r["window"]) == (view, window))


def direct_sse(corners, scale):
    return float(np.sum((corners[3] - scale[..., None] * corners[0])**2))


class Tests(unittest.TestCase):
    def test_uniform_prefix_mass_only_exact_in_every_view(self):
        p, z = fixture()
        p[1, ..., :6] *= .4
        args, corners = saved(p, z)
        out = K.analyze_saved_grams(**args)
        for r in out["rows"]:
            for name in (*K.SCALINGS, "modality_mass", "full_late"):
                self.assertLess(r["baselines"][name]["SSE"], 1e-13)
            self.assertGreater(r["baselines"]["zero_change"]["SSE"], 0)
        np.testing.assert_allclose(out["arrays"]["prefix_mass"], args["masses"].sum(-1))
        self.assertAlmostEqual(row(out)["target_delta_energy"], float(np.sum((corners[3]-corners[0])**2)))

    def test_head_only_scaling_exact_but_position_global_fails(self):
        p, z = fixture()
        p[1, 0, :, :6] *= .5
        p[1, 1, :, :6] *= 1.5
        args, corners = saved(p, z)
        out = K.analyze_saved_grams(**args)
        for r in out["rows"]:
            self.assertLess(r["baselines"]["per_head_position_prefix_mass"]["SSE"], 1e-13)
            self.assertLess(r["baselines"]["modality_mass"]["SSE"], 1e-13)
            self.assertGreater(r["baselines"]["per_position_prefix_mass"]["SSE"], 0)
        np.testing.assert_allclose(out["arrays"]["scales"][0, 1], 1.)
        expected = np.broadcast_to(np.array((.5, 1.5))[:, None], (2, 10))
        np.testing.assert_allclose(out["arrays"]["scales"][0, 2], expected)
        self.assertAlmostEqual(row(out)["baselines"]["per_head_position_prefix_mass"]["SSE"],
                               direct_sse(corners, expected))

    def test_modality_additional_information_is_necessary(self):
        p, z = fixture()
        p[1, ..., :2] *= 1.5
        p[1, ..., 2:4] *= .5
        args, _ = saved(p, z)
        out = K.analyze_saved_grams(**args)
        for r in out["rows"]:
            self.assertGreater(r["target_delta_energy"], 0.)
            self.assertLess(r["baselines"]["modality_mass"]["SSE"], 1e-13)
            self.assertAlmostEqual(r["baselines"]["per_head_position_prefix_mass"]["relative_SSE"], 1.)

    def test_within_only_counterexample_is_retained(self):
        p, z = fixture()
        p[1, ..., 0] = .09
        p[1, ..., 1] = .01
        args, _ = saved(p, z)
        out = K.analyze_saved_grams(**args)
        r = row(out)
        self.assertLess(r["baselines"]["within_modality"]["SSE"], 1e-13)
        for name in (*K.SCALINGS, "modality_mass"):
            self.assertAlmostEqual(r["baselines"][name]["relative_SSE"], 1.)

    def test_global_window_and_per_position_are_distinct(self):
        p, z = fixture()
        factors = np.linspace(.2, 1.8, 10)
        p[1, ..., :6] *= factors[None, :, None]
        args, corners = saved(p, z)
        out = K.analyze_saved_grams(**args)
        for wi, (_, start, end) in enumerate(K.WINDOWS):
            r = out["rows"][wi]
            self.assertLess(r["baselines"]["per_position_prefix_mass"]["SSE"], 1e-13)
            self.assertGreater(r["baselines"]["global_window_prefix_mass"]["SSE"], 0.)
            ratio = factors[start:end].mean()
            expected = np.sum((corners[3, :, start:end] - ratio * corners[0, :, start:end])**2)
            self.assertAlmostEqual(r["baselines"]["global_window_prefix_mass"]["SSE"], expected)
        self.assertFalse(np.isfinite(out["arrays"]["scales"][1, :, :, 5:]).any())

    def test_signed_cancellation_and_zero_target_keep_nonzero_errors(self):
        p, z = fixture()
        z[:] = 0
        z[0, 0] = 1
        z[2, 0] = -1
        p[1, ..., 0] = .1
        p[1, ..., 2] = .1
        args, _ = saved(p, z)
        out = K.analyze_saved_grams(**args)
        r = row(out)
        self.assertEqual(r["target_delta_energy"], 0.)
        self.assertIsNone(r["baselines"]["within_modality"]["relative_SSE"])
        self.assertAlmostEqual(r["decomposition"]["component_energy_sum"] +
                               r["decomposition"]["signed_cross_sum"], 0.)
        # A mixed corner can be nonzero even when endpoints coincide.
        p, z = fixture()
        z[:] = 0
        z[:2, 0] = (1, -1)
        p[0, ..., :2] = (.075, .025)  # mean .5, mass .1 -> .05
        p[1, ..., :2] = (.125, .075)  # mean .25, mass .2 -> .05
        args, _ = saved(p, z)
        out = K.analyze_saved_grams(**args)
        r = row(out)
        self.assertLess(r["target_delta_energy"], 1e-15)
        self.assertGreater(r["baselines"]["modality_mass"]["SSE"], 0.)
        self.assertLess(r["decomposition"]["signed_cross_sum"], 0.)
        self.assertAlmostEqual(r["target_delta_energy"], r["decomposition"]["component_energy_sum"] +
                               r["decomposition"]["signed_cross_sum"])

    def test_head_spaces_are_concatenated_never_summed(self):
        p, z = fixture()
        z[:] = 0
        z[0, 0] = 1
        z[2, 0] = -1
        p[1, 0, :, 0] += .1
        p[1, 1, :, 2] += .1
        args, corners = saved(p, z)
        out = K.analyze_saved_grams(**args)
        delta = corners[3] - corners[0]
        np.testing.assert_allclose(delta.sum(0), 0.)
        self.assertGreater(row(out)["target_delta_energy"], 0.)
        self.assertAlmostEqual(row(out)["target_delta_energy"], np.sum(delta**2))

    def test_zero_modality_does_not_censor_known_head_baseline(self):
        p, z = fixture()
        p[0, ..., :2] = 0.
        args, _ = saved(p, z)
        out = K.analyze_saved_grams(**args)
        r = row(out)
        self.assertEqual(r["baselines"]["modality_mass"]["status"], "UNAVAILABLE_ZERO_MASS_ALLOCATION")
        self.assertEqual(r["baselines"]["per_head_position_prefix_mass"]["status"], "DEFINED")
        self.assertEqual(r["baselines"]["within_modality"]["status"], "DEFINED")
        self.assertEqual(r["baselines"]["full_late"]["SSE"], 0.)
        self.assertEqual(r["decomposition"]["status"], "UNAVAILABLE_MIXED_CORNER")
        p[1, ..., :2] = 0.
        out = K.analyze_saved_grams(**saved(p, z)[0])
        self.assertIsNotNone(row(out)["baselines"]["modality_mass"]["SSE"])

    def test_zero_prefix_undefined_vs_uniquely_zero_contribution(self):
        p, z = fixture()
        p[0, 0, :, :6] = 0.
        out = K.analyze_saved_grams(**saved(p, z)[0])
        r = row(out)
        self.assertIsNone(r["baselines"]["per_head_position_prefix_mass"]["SSE"])
        self.assertIsNotNone(r["baselines"]["per_position_prefix_mass"]["SSE"])
        p[1, 0, :, :6] = 0.
        out = K.analyze_saved_grams(**saved(p, z)[0])
        self.assertIsNotNone(row(out)["baselines"]["per_head_position_prefix_mass"]["SSE"])
        self.assertTrue(out["arrays"]["scale_defined"][0, 2, 0].all())
        self.assertFalse(out["arrays"]["scale_identified"][0, 2, 0].any())
        self.assertTrue(np.isnan(out["arrays"]["scales"][0, 2, 0]).all())
        p[1, 1, :, :6] *= .5
        out = K.analyze_saved_grams(**saved(p, z)[0])
        self.assertEqual(row(out)["baselines"]["per_head_position_prefix_mass"]["status"],
                         "DEFINED_WITH_ZERO_MASS_NOOP")
        self.assertEqual(row(out)["baselines"]["per_head_position_prefix_mass"]["zero_mass_noop_head_positions"], 10)

    def test_query_missingness_never_survivor_reweights_windows(self):
        p, z = fixture()
        defined = np.ones((2, 10), bool)
        defined[:, 6] = False
        p[:, :, 6] = 0.
        out = K.analyze_saved_grams(**saved(p, z, defined)[0])
        self.assertIsNone(row(out)["target_delta_energy"])
        self.assertEqual(row(out)["status"], "UNAVAILABLE_QUERY")
        self.assertIsNotNone(row(out, window="first5")["target_delta_energy"])
        self.assertIsNone(row(out, window="tail5")["target_delta_energy"])
        for b in row(out)["baselines"].values():
            self.assertIsNone(b["SSE"])

    def test_full_softmax_mass_preserved_and_suffix_not_renormalized(self):
        p, z = fixture()
        p[1, ..., :6] *= .5
        p[1, ..., 6:] = .08
        self.assertFalse(np.allclose(p.sum(-1), 1.))
        args, _ = saved(p, z)
        out = K.analyze_saved_grams(**args)
        np.testing.assert_allclose(out["arrays"]["prefix_mass"][0], .3)
        np.testing.assert_allclose(out["arrays"]["prefix_mass"][1], .15)
        np.testing.assert_allclose(out["arrays"]["scales"][0], .5)
        self.assertGreater(row(out)["target_delta_energy"], 0.)
        # Prefix-normalizing the original rows would erase this real change.
        normalized = p.copy()
        normalized[..., :6] /= normalized[..., :6].sum(-1, keepdims=True)
        wrong = K.analyze_saved_grams(**saved(normalized, z)[0])
        self.assertEqual(row(wrong)["target_delta_energy"], 0.)

    def test_basis_split_every_baseline_and_signed_closure(self):
        p, z = fixture()
        p[1, ..., :6] *= np.array((.2, .7, 1.3, .4, 1.8, .8))
        out = K.analyze_saved_grams(**saved(p, z)[0])
        for window, _, _ in K.WINDOWS:
            full = row(out, window=window)
            for basis in ("response", "clean"):
                a, b = row(out, basis+"_B3", window), row(out, basis+"_residual", window)
                self.assertAlmostEqual(full["target_delta_energy"], a["target_delta_energy"]+b["target_delta_energy"])
                for name in K.BASELINES:
                    self.assertAlmostEqual(full["baselines"][name]["SSE"],
                                           a["baselines"][name]["SSE"] + b["baselines"][name]["SSE"])
                self.assertLess(abs(a["decomposition"]["energy_closure_error"]), 1e-12)
                for r in (a, b):
                    self.assertEqual(r["full_target_delta_energy"], full["target_delta_energy"])
                    for name in K.BASELINES:
                        self.assertAlmostEqual(r["baselines"][name]["relative_SSE_to_full_target"],
                                               r["baselines"][name]["SSE"] / full["target_delta_energy"])

    def test_oracle_uses_one_full_coefficient_in_every_part(self):
        p, z = fixture()
        p[1, ..., :2] *= .2
        p[1, ..., 2:4] *= 1.4
        p[1, ..., 4:6] *= .7
        args, corners = saved(p, z)
        out = K.analyze_saved_grams(**args)
        x, y = corners[0], corners[3]
        alpha = np.sum(x*y)/np.sum(x*x)
        full = row(out)
        self.assertAlmostEqual(full["oracle_full_window_alpha"], alpha)
        self.assertAlmostEqual(full["baselines"]["signed_scalar_oracle"]["SSE"], np.sum((y-alpha*x)**2))
        part_optima = []
        for name, dimensions in (("response_B3", slice(0, 3)), ("response_residual", slice(3, 6))):
            r = row(out, name)
            self.assertEqual(r["oracle_full_window_alpha"], full["oracle_full_window_alpha"])
            self.assertAlmostEqual(r["baselines"]["signed_scalar_oracle"]["SSE"],
                                   np.sum((y[..., dimensions]-alpha*x[..., dimensions])**2))
            xp, yp = x[..., dimensions], y[..., dimensions]
            part_optima.append(np.sum(xp*yp)/np.sum(xp*xp))
        self.assertNotAlmostEqual(part_optima[0], part_optima[1])
        self.assertEqual(full["baselines"]["full_late"]["comparison_role"], "diagnostic_only")

    def test_oracle_zero_source_keeps_known_zero_prediction(self):
        p, z = fixture()
        p[0, ..., :6] = 0.
        out = K.analyze_saved_grams(**saved(p, z)[0])
        r = row(out)
        self.assertIsNone(r["oracle_full_window_alpha"])
        self.assertEqual(r["baselines"]["signed_scalar_oracle"]["status"], "SOURCE_ZERO_NOOP")
        self.assertAlmostEqual(r["baselines"]["signed_scalar_oracle"]["relative_SSE"], 1.)

    def test_negative_roundoff_is_bounded_reported_not_science_threshold(self):
        g = np.ones((1, 1, 4, 4), np.float64)
        g[..., 0, 3] = g[..., 3, 0] = 1. + np.finfo(np.float64).eps
        r = K.quadratic_energy(g, K.DELTA_COEFFICIENTS)
        self.assertLess(r["raw"].item(), 0.)
        self.assertEqual(r["energy"].item(), 0.)
        self.assertTrue(r["corrected"].item())
        g[..., 0, 3] = g[..., 3, 0] = 1. + 1e-8
        with self.assertRaisesRegex(ValueError, "material_negative"):
            K.quadratic_energy(g, K.DELTA_COEFFICIENTS)
        g[..., 0, 3] = g[..., 3, 0] = 1. - np.finfo(np.float64).eps
        self.assertGreater(K.quadratic_energy(g, K.DELTA_COEFFICIENTS)["energy"].item(), 0.)

    def test_bad_shape_mask_and_projection_refused(self):
        p, z = fixture()
        args, _ = saved(p, z)
        broken = copy.deepcopy(args)
        broken["raw_gram_upper"] = broken["raw_gram_upper"].astype(np.float32)
        with self.assertRaisesRegex(ValueError, "FP64"):
            K.analyze_saved_grams(**broken)
        broken = copy.deepcopy(args)
        broken["group_corner_defined"][1, 0, 0, 0] = False
        with self.assertRaisesRegex(ValueError, "group_corner"):
            K.analyze_saved_grams(**broken)
        broken = copy.deepcopy(args)
        broken["projection_gram_upper"] *= 2
        with self.assertRaisesRegex(ValueError, "basis_split"):
            K.analyze_saved_grams(**broken)

    def test_input_immutable_full_only_and_small_json(self):
        p, z = fixture()
        p[1, ..., :6] *= .6
        args, _ = saved(p, z)
        original = {k: a.copy() for k, a in args.items() if isinstance(a, np.ndarray)}
        for a in original:
            args[a].setflags(write=False)
        out = K.analyze_saved_grams(**args)
        for key, a in original.items():
            np.testing.assert_array_equal(args[key], a)
        json.dumps({k: v for k, v in out.items() if k != "arrays"}, allow_nan=False)
        args["projection_gram_upper"] = None
        self.assertEqual(len(K.analyze_saved_grams(**args)["rows"]), 3)

    def test_aggregation_absolute_before_ratio_and_missing_is_preserved(self):
        p, z = fixture()
        p[1, ..., :6] *= .5
        base = row(K.analyze_saved_grams(**saved(p, z)[0]))
        a, b = copy.deepcopy(base), copy.deepcopy(base)
        a["target_delta_energy"], b["target_delta_energy"] = 1., 9.
        for name in K.BASELINES:
            a["baselines"][name]["SSE"], b["baselines"][name]["SSE"] = 1., 0.
        aggregate = K.aggregate_absolute([a, b])
        self.assertEqual(aggregate["baselines"]["modality_mass"]["relative_SSE"], .1)
        b["baselines"]["modality_mass"]["SSE"] = None
        aggregate = K.aggregate_absolute([a, b])
        self.assertIsNone(aggregate["baselines"]["modality_mass"]["relative_SSE"])
        self.assertEqual(aggregate["baselines"]["modality_mass"]["observed_rows"], 1)
        self.assertIsNotNone(aggregate["baselines"]["zero_change"]["relative_SSE"])


if __name__ == "__main__":
    unittest.main()
