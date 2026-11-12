import copy
import hashlib
import json
from pathlib import Path
import unittest
import numpy as np
import core as C


def fixture():
    rng = np.random.default_rng(872)
    wo = rng.normal(size=(2, 3, 5))
    queries, groups = [], []
    output = rng.normal(size=(3, 70))
    for ri, root in enumerate(C.ROOTS):
        task, cluster = "T"+str(ri//2), "verified-init-"+str(ri)
        groups.append(dict(
            root_id=root, cluster_id=cluster, task_id=task,
            library_sha256=str(ri//2)*64, init_index=ri % 2,
            init_state_sha256=str(ri+2)*64, source_ref={"synthetic": True}))
        for qidx in (11, 12, 13):
            state, delta = rng.normal(size=8), rng.normal(size=6)
            p = rng.uniform(.01, 1, size=(2, 10, 13))
            p /= p.sum(-1, keepdims=True)
            edits, targets = {}, {}
            for ai, arm in enumerate(C.ARMS):
                e = rng.normal(size=(3, 3))*(ai+1)/4
                edits[arm] = e
                latent = np.array([state[0]+e.sum(), state[1]+e[0, 0], delta[0]+ai])
                targets[arm] = (latent @ output).reshape(10, 7)
            queries.append(dict(
                root_id=root, cluster_id=cluster, task_id=task, query_index=qidx,
                source_identity="a"*64, state_vector=state, first_flow_delta=delta,
                time_scalar=5*qidx/520, p=p, mask=np.ones((10, 13), bool),
                gate=rng.uniform(.1, 1, size=5), edits=edits, targets=targets,
                provenance={"synthetic": True}, split="DEV"))
    return dict(
        schedule="persistent_j5_to_j9", arms=list(C.ARMS), wo=wo, groups=groups,
        queries=queries,
        independence={"status": "SYNTHETIC_FOUR_DISTINCT_GROUPS", "iid_claim": False})


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = fixture()
        cls.result = C.run(cls.raw, synthetic=True)

    def test_pin_and_bounded_fit_accounting(self):
        raw = Path(C.__file__).with_name("legacy_core.py").read_bytes()
        self.assertEqual(hashlib.sha256(raw).hexdigest(), C.LEGACY_SHA256)
        summary = self.result["summary"]
        self.assertEqual(summary["new_fits"], 150)
        self.assertEqual(summary["incremental_comparator_fits"], 60)
        self.assertEqual(summary["old_family_replay_fits"], 90)
        self.assertEqual(summary["fit_counts_by_family"], {f: 30 for f in C.FAMILIES})
        self.assertEqual(summary["preprocessing_fit_sets"], 10)
        self.assertEqual(summary["PCA_fit_count"], 30)
        self.assertFalse(summary["all_DEV_refit_executed"])
        self.assertFalse(summary["bandwidth_grid_expanded"])
        self.assertFalse(summary["family_winner_selection"])

    def test_exact_legacy_replay_of_all_fifteen_tracks(self):
        legacy = C.L.run(self.raw, synthetic=True)
        for family in C.OLD_FAMILIES:
            keys = [f"fixed__{family}__{lam:g}" for lam in C.LAMBDAS]
            keys += ["nested__"+family, "contrast_nested__"+family]
            for key in keys:
                np.testing.assert_array_equal(
                    self.result["arrays"]["prediction__"+key],
                    legacy["arrays"]["prediction__"+key])
                self.assertEqual(
                    self.result["summary"]["OOF_root_arm_scores"][key],
                    legacy["summary"]["OOF_root_arm_scores"][key])
                self.assertEqual(
                    self.result["summary"]["paired_persistent_contrast"]["root_scores"][key],
                    legacy["summary"]["paired_persistent_contrast"]["root_scores"][key])

    def test_heldout_labels_never_change_geometry_tuning_or_predictions(self):
        data = C.Dataset(self.raw, lambda: None, True)
        held, training = data.clusters[0], list(data.clusters[1:])
        selected = C.nested_choices(data, training)
        fold = data.fold(training)
        predictions = {
            f: fold.predict(f, selected["selected_lambda"][f])[0] for f in C.FAMILIES}
        modified = copy.deepcopy(self.raw)
        for q in modified["queries"]:
            if q["cluster_id"] == held:
                for arm in C.ARMS:
                    q["targets"][arm] *= -1000
        other = C.Dataset(modified, lambda: None, True)
        selected2 = C.nested_choices(other, training)
        fold2 = other.fold(training)
        self.assertEqual(selected, selected2)
        self.assertEqual(fold.manifest(), fold2.manifest())
        for family in C.FAMILIES:
            pred, _ = fold2.predict(family, selected2["selected_lambda"][family])
            np.testing.assert_array_equal(predictions[family], pred)
        self.assertEqual(fold.geometry(), fold2.geometry())
        for row in selected["inner_folds"]:
            self.assertNotIn(held, row["training_clusters"])
            self.assertNotIn(row["validation_cluster"], row["training_clusters"])

    def test_heldout_features_never_change_shared_bandwidth_or_preprocessing(self):
        data = C.Dataset(self.raw, lambda: None, True)
        held, training = data.clusters[0], list(data.clusters[1:])
        a = data.fold(training)
        for family in C.FAMILIES:
            a.predict(family, 1.)
        modified = copy.deepcopy(self.raw)
        for q in modified["queries"]:
            if q["cluster_id"] == held:
                q["state_vector"] *= 100
                q["first_flow_delta"] *= 200
                q["p"][:] = 1/q["p"].shape[-1]
                q["gate"] *= 300
                for arm in C.ARMS:
                    q["edits"][arm] *= 400
        b = C.Dataset(modified, lambda: None, True).fold(training)
        for family in C.FAMILIES:
            b.predict(family, 1.)
        self.assertEqual(a.manifest(), b.manifest())
        self.assertEqual(a.geometry(), b.geometry())
        for family in C.FAMILIES:
            np.testing.assert_array_equal(a.kernels[family].cross[a.indices],
                                          b.kernels[family].cross[b.indices])

    def test_fixed_bandwidth_matches_legacy_when_bandwidth_matches(self):
        rng = np.random.default_rng(50)
        x, y = rng.normal(size=(30, 6)), rng.normal(size=(24, 3))
        indices = np.arange(24)
        w = np.arange(1, 25, dtype=float)
        w /= w.sum()
        z = np.eye(4)[np.arange(30) % 4]
        old = C.L.Kernel(x, indices, w, z, rbf=True)
        fixed = C.FixedBandwidthKernel(x, indices, w, z, old.bandwidth)
        np.testing.assert_array_equal(old.cross, fixed.cross)
        for lam in C.LAMBDAS:
            a, ai, _ = old.predict(y, lam)
            b, bi, _ = fixed.predict(y, lam)
            np.testing.assert_array_equal(a, b)
            self.assertEqual(ai, bi)
        with self.assertRaisesRegex(ValueError, "UNAVAILABLE"):
            C.FixedBandwidthKernel(x, indices, w, z, 0.)

    def test_edf_equals_direct_smoother_trace(self):
        rng = np.random.default_rng(71)
        x = rng.normal(size=(20, 6))
        z = np.eye(4)[np.arange(20) % 4]
        w = np.arange(1, 21, dtype=float)
        w /= w.sum()
        kernel = C.FixedBandwidthKernel(x, np.arange(20), w, z, 1.3)
        for lam in C.LAMBDAS:
            # Each target column is a weighted-coordinate basis vector.
            values, info, _ = kernel.predict(np.eye(20)/np.sqrt(w)[:, None], lam)
            smoother = np.sqrt(w)[:, None]*values
            self.assertAlmostEqual(np.trace(smoother), info["total_EDF"], places=10)

    def test_shared_bandwidth_is_exactly_train_full_and_compression_nested(self):
        summary = self.result["summary"]
        for fold in summary["folds"]:
            for lam in C.LAMBDAS:
                full = fold["model_diagnostics"][f"fixed__rbf__{lam:g}"]
                small = fold["model_diagnostics"][
                    f"fixed__rbf_current4_full_bandwidth__{lam:g}"]
                self.assertEqual(full["RBF_median_squared_distance"],
                                 small["RBF_median_squared_distance"])
        for geometry in summary["geometry_by_training_set"]:
            self.assertLessEqual(geometry["write_PCA_rank"], 4)
            self.assertGreaterEqual(geometry["compression_variance_retained"], 0)
            self.assertLessEqual(geometry["compression_variance_retained"], 1+1e-10)
            self.assertLessEqual(geometry["median_ratio_current4_over_full"], 1+1e-10)

    def test_comparison_roots_counterexamples_and_no_winner_selection(self):
        summary = self.result["summary"]
        records = summary["comparison_report"]["comparisons"]
        self.assertEqual(len(records), 30)
        primary = [r for r in records if r["role"] == "PRIMARY_SAME_ALGORITHM_RULE"]
        secondary = [r for r in records if r["role"] == "SECONDARY_COMMON_TRAIN_FULL_BANDWIDTH"]
        self.assertEqual(len(primary), 5)
        self.assertEqual(len(secondary), 5)
        for record in records:
            for kind in ("endpoint", "shared_minus_random_contrast"):
                comparison = record[kind]
                self.assertEqual(comparison["roots"], 4)
                self.assertEqual(comparison["favorable_roots"] +
                                 comparison["unfavorable_roots"] +
                                 comparison["tied_roots"], 4)
                expected = []
                for root in comparison["per_root"]:
                    differences = np.array(root["MSE_difference_baseline_minus_candidate"])
                    expected += [
                        (root["cluster"], f"first{end}", d)
                        for wi, end in enumerate((5, 10))
                        for d in range(7) if differences[wi, 1, d] < 0]
                actual = [
                    (r["cluster"], r["window"], r["action_dimension"])
                    for r in comparison["all_unfavorable_standardized_root_window_coordinates"]]
                self.assertEqual(expected, actual)
                self.assertFalse(comparison["hypothesis_test_or_power_claim"])
        json.dumps(summary, allow_nan=False)

    def test_zero_override_and_constant_targets_retained(self):
        raw = copy.deepcopy(self.raw)
        for q in raw["queries"]:
            for arm in C.ARMS:
                q["targets"][arm][:, 6] = 0
        raw["queries"][0]["edits"]["shared"][:] = 0
        raw["queries"][0]["targets"]["shared"][:] = 0
        result = C.run(raw, synthetic=True)
        self.assertEqual(result["summary"]["known_zero_edits"], 1)
        for name, value in result["arrays"].items():
            if name.startswith("prediction__fixed__"):
                np.testing.assert_array_equal(value[0], 0)
        self.assertTrue(result["summary"]["unavailable_predictions"])
        json.dumps(result["summary"], allow_nan=False)
        fixed_records = [
            record for record in result["summary"]["comparison_report"]["comparisons"]
            if record["track"].startswith("fixed")]
        self.assertTrue(all(
            record["endpoint"]["status"] == "UNDEFINED_STANDARDIZED_SCORE"
            for record in fixed_records))

    def test_zero_training_median_is_unavailable_without_epsilon_or_fallback(self):
        raw = copy.deepcopy(self.raw)
        for q in raw["queries"]:
            q["state_vector"][:] = 0
            q["first_flow_delta"][:] = 0
            q["time_scalar"] = 0
            q["gate"][:] = 0
            for arm in C.ARMS:
                q["edits"][arm][:] = 1
        data = C.Dataset(raw, lambda: None, True)
        fold = data.fold(data.clusters[:3])
        for family in ("rbf", *C.NEW_FAMILIES):
            for lam in C.LAMBDAS:
                prediction, info = fold.predict(family, lam)
                self.assertIsNone(prediction)
                self.assertEqual(info["status"], "UNAVAILABLE")
                self.assertIn("zero_training_median", info["reason"])
        self.assertEqual(data.fit_count, 0)
        self.assertEqual(data.fit_attempt_count, 9)


if __name__ == "__main__":
    unittest.main()
