"""Synthetic-only checks for fixed P08 geometry and support semantics."""
import importlib.util
from pathlib import Path
import unittest
import numpy as np

spec = importlib.util.spec_from_file_location("p08_statistics", Path(__file__).with_name("statistics.py"))
S = importlib.util.module_from_spec(spec); spec.loader.exec_module(S)


def row(task=0, slot=1, condition="camera_plus15", suite="s", noise=0, present=True):
    eid = f"{suite}:{task}:{condition}:n{noise}"
    return dict(anchor_id=eid + f":slot{slot}", episode_id=eid, suite=suite, task=str(task),
                task_fold=task % 5, root=f"{suite}:{task}:i0", noise_repeat=noise, slot=slot,
                history_condition=condition, declared_current_condition=condition,
                original_present=present)


def fixture():
    rows = [row(t, s) for t in range(5) for s in (1, 2)]
    split = S.freeze_metadata(rows)
    rng = np.random.default_rng(7)
    delta = rng.standard_normal((10, 3, 5, 8)).astype(np.float32)
    clean = rng.standard_normal(delta.shape).astype(np.float32) + 20
    return rows, delta, clean, split


class TestStatistics(unittest.TestCase):
    def test_bf16_exact_bit_expansion(self):
        np.testing.assert_array_equal(S.bf16_bits_to_float32(np.array([0x3f80, 0xbf80, 0], np.uint16)), [1, -1, 0])
        with self.assertRaises(ValueError): S.bf16_bits_to_float32([1, 2])

    def test_existing_fold_split_balanced_and_no_outcome_selection(self):
        rows = [row(t, s, suite=u) for u in ("a", "b") for t in range(10) for s in (1, 2)]
        split = S.freeze_metadata(rows)
        self.assertEqual(split["suite_balance"], {"a": {"train": 6, "heldout": 4}, "b": {"train": 6, "heldout": 4}})
        chosen = split["heatmap_examples"][0]["episode_id"]
        for r in rows:
            r["original_success"] = r["task_fold"] % 2
            if r["episode_id"] == chosen: r["original_present"] = False
        self.assertEqual(S.freeze_metadata(rows)["heatmap_examples"][0]["episode_id"], chosen)

    def test_task_condition_root_noise_weights(self):
        rows = [row(0, noise=0), row(0, noise=1), row(1)]
        np.testing.assert_allclose(S.owner_weights(rows), [.25, .25, .5])
        with self.assertRaises(ValueError): S.owner_weights(rows + [rows[0]])

    def test_heldout_and_slot2_do_not_change_fit(self):
        rows, d, c, split = fixture()
        fit = S.fit_bases(rows, d, c, split, rank=3)
        changed = d.copy(); clean = c.copy()
        for i, r in enumerate(rows):
            if r["task_fold"] >= 3 or r["slot"] == 2:
                changed[i] = 1e6; clean[i] = -1e6
        again = S.fit_bases(rows, changed, clean, split, rank=3)
        for layer in fit["layers"]:
            np.testing.assert_array_equal(fit["layers"][layer]["center"], again["layers"][layer]["center"])
            for key in ("response", "clean", "random"):
                np.testing.assert_array_equal(fit["layers"][layer]["bases"][key], again["layers"][layer]["bases"][key])

    def test_position_centers_and_constant_offsets(self):
        rows, d, c, split = fixture()
        shift = np.arange(5, dtype=np.float32)[None, None, :, None] * 10
        f1, f2 = S.fit_bases(rows, d, c, split, 3), S.fit_bases(rows, d + shift, c, split, 3)
        for layer in f1["layers"]:
            np.testing.assert_allclose(f2["layers"][layer]["center"] - f1["layers"][layer]["center"], np.broadcast_to(shift[0,0], (5,8)), atol=1e-5)
            b1, b2 = (f["layers"][layer]["bases"]["response"] for f in (f1, f2))
            np.testing.assert_allclose(b1 @ b1.T, b2 @ b2.T, atol=1e-5)

    def test_shared_control_center_and_reconstruction_budget(self):
        x = np.array([[3., 2., 1., 0.]], np.float32)
        mu = np.array([[1., 0., 0., 0.]], np.float32)
        b = np.eye(4, 3, dtype=np.float32)
        z = S._projection_metrics(x, mu, b)
        self.assertAlmostEqual(z["sse"], 0.)
        self.assertAlmostEqual(z["retention_centered"], 1.)
        self.assertEqual(S._projection_metrics(mu, mu, b)["retention_centered"], None)
        self.assertIsNone(S._projection_metrics(x, mu, None)["sse"])

    def test_rank_deficiency_not_filled_with_arbitrary_axes(self):
        rows, d, c, split = fixture()
        fit = S.fit_bases(rows, np.zeros_like(d), c, split, 3)
        self.assertIsNone(fit["layers"]["1"]["bases"]["response"])
        z = S.describe_owner(rows[-1], np.zeros_like(d[-1]), c[-1], fit)
        self.assertIsNone(z["layers"]["1"]["projection"]["points"][0]["xyz"])

    def test_pr_is_coordinate_dependent_not_dimension(self):
        self.assertEqual(S.coordinate_pr([1., 0]), 1.)
        self.assertAlmostEqual(S.coordinate_pr([1., 1.]), 2.)
        self.assertIsNone(S.coordinate_pr([0., 0]))
        self.assertEqual(S.signed_cosine([1., 0], [-1., 0])["value"], -1.)
        self.assertIsNone(S.signed_cosine([0., 0], [1., 0])["value"])

    def test_attention_unconditional_mass_not_conditioned_tv(self):
        c = np.zeros((3,8,2,16,16), np.float32); c[...,0,0] = .25
        a = c * 2
        mask = np.ones((3,8,4), bool)
        z = S.attention_summary(c, a, mask, mask, mask)
        self.assertEqual(z["fixed8"][0]["camera_mass_clean"], .25)
        self.assertEqual(z["fixed8"][0]["unconditional_camera_half_L1"], .125)
        # Conditioning each map would give TV=0; the new metric must not.
        bad = mask.copy(); bad[0,0,0] = False
        z = S.attention_summary(c, a, bad, mask, bad)
        self.assertIsNone(z["fixed8"][0]["unconditional_camera_half_L1"])
        self.assertEqual(z["fixed8"][0]["unconditional_camera_half_L1_valid_heads"], 7)

    def test_valid_zero_mass_and_invalid_masks(self):
        a = np.zeros((3,8,2,16,16), np.float32)
        mask = np.ones((3,8,4), bool)
        self.assertEqual(S.attention_summary(a,a,mask,mask,mask)["fixed8"][0]["unconditional_camera_half_L1"], 0.)
        with self.assertRaises(ValueError): S.attention_summary(a,a,mask,mask,~mask)

    def test_small_output_shapes_roles_and_fixed_axes(self):
        rows, d, c, split = fixture(); fit = S.fit_bases(rows,d,c,split,3)
        z = S.describe_owner(rows[-1],d[-1],c[-1],fit)
        self.assertEqual(z["evaluation_role"], "primary_heldout_slot2")
        points = z["layers"]["18"]["projection"]["points"]
        self.assertEqual(len(points), 5); self.assertEqual(len(points[0]["xyz"]), 3)
        self.assertNotIn("hidden_delta", z)
        lim = S.frozen_axis_limits([z]); self.assertEqual(len(lim["18"]),3)
        self.assertEqual(len(S.describe_pair(d[0],d[1])["1"]["position_cosines"]), 5)

    def test_bootstrap_missing_and_signed_control_difference(self):
        rows = [dict(row(t), gain=(-1.) ** t) for t in range(5)]
        rows[-1]["gain"] = None
        tasks = [S.task_key(r) for r in rows]
        z = S.summarize_fixed_metric(rows,"gain",tasks)
        self.assertEqual(z["defined_rows"],4); self.assertEqual(z["planned_rows"],5)
        self.assertEqual(z["mean"],0.)
        self.assertEqual(z,S.summarize_fixed_metric(rows,"gain",tasks))


if __name__ == "__main__": unittest.main()
