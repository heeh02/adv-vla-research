"""Bounded, train-group-only nonlinear compression comparator; NumPy and no IO.

The byte-pinned legacy module is read solely to import the exact frozen math.
There is no data, network, model, environment, or provider IO in this module.
"""
import hashlib
import importlib.util
from pathlib import Path
import numpy as np

_legacy_path = Path(__file__).with_name("legacy_core.py")
LEGACY_SHA256 = "9b915a6e3a42b355868ac62f7dc89db27a5c2441c565432296423b76b0b412e1"
if hashlib.sha256(_legacy_path.read_bytes()).hexdigest() != LEGACY_SHA256:
    raise ValueError("exact_frozen_v2_math_source_required")
_spec = importlib.util.spec_from_file_location("rss_kernel_comparator_legacy", _legacy_path)
L = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(L)

ARMS, ROOTS, LAMBDAS = L.ARMS, L.ROOTS, L.LAMBDAS
OLD_FAMILIES = ("current4", "current_full", "rbf")
NEW_FAMILIES = ("rbf_current4", "rbf_current4_full_bandwidth")
FAMILIES = OLD_FAMILIES + NEW_FAMILIES
need, serial = L.need, L.serial


class FixedBandwidthKernel(L.Kernel):
    """The legacy weighted KRR algebra, with a prespecified train-only bandwidth."""
    def __init__(self, x, indices, w, z, bandwidth):
        self.x = L.arr(x)
        self.indices, self.w, self.z, self.rbf = indices, w, z, True
        xt = x[indices]
        dot = x @ xt.T
        need(np.isfinite(dot).all(), "finite_kernel_dot_product")
        distances = np.maximum(np.sum(x*x, axis=1)[:, None] +
                               np.sum(xt*xt, axis=1)[None] - 2*dot, 0.)
        self.bandwidth = float(bandwidth)
        need(np.isfinite(distances).all() and np.isfinite(self.bandwidth),
             "finite_kernel_distance")
        need(self.bandwidth > 0, "RBF_bandwidth_UNAVAILABLE_zero_training_median")
        self.cross = np.exp(-distances / (2*self.bandwidth))
        train = self.cross[indices]
        sw = np.sqrt(w)
        self.zw = sw[:, None]*z[indices]
        self.ztz = self.zw.T @ self.zw
        need(np.linalg.matrix_rank(self.ztz) == z.shape[1], "all_training_arms_present")
        self.projection = self.zw @ np.linalg.solve(self.ztz, self.zw.T)
        self.residual = np.eye(len(w)) - self.projection
        self.weighted = sw[:, None]*train*sw[None]
        centered = self.residual @ self.weighted @ self.residual
        centered = (centered + centered.T)/2
        eigen, self.u = np.linalg.eigh(centered)
        tol = 1e-10*max(1., float(np.max(np.abs(eigen))))
        need(np.isfinite(eigen).all() and np.isfinite(self.u).all(),
             "finite_kernel_eigensystem")
        need(eigen.min() >= -tol, "kernel_PSD_beyond_roundoff")
        self.eigen = np.maximum(eigen, 0)
        self.rank = int(np.count_nonzero(
            self.eigen > max(1., self.eigen.max())*len(w)*np.finfo(float).eps))


class Dataset(L.Dataset):
    def fold(self, clusters):
        key = tuple(sorted(clusters))
        if key not in self.folds:
            self.folds[key] = Fold(self, key)
        return self.folds[key]


class Fold(L.Fold):
    def __init__(self, data, clusters):
        super().__init__(data, clusters)
        self.x["rbf_current4"] = self.x["current4"]
        self.x["rbf_current4_full_bandwidth"] = self.x["current4"]

    def predict(self, family, lam=1.):
        if family in OLD_FAMILIES:
            return super().predict(family, lam)
        need(family in NEW_FAMILIES, "bounded_comparator_family_roster")
        key = family, lam
        if key in self.cache:
            return self.cache[key]
        self.data.check()
        self.data.fit_attempt_count += 1
        if family not in self.kernels:
            try:
                if family == "rbf_current4":
                    self.kernels[family] = L.Kernel(
                        self.x[family], self.indices, self.w, self.data.z, rbf=True)
                else:
                    # Constructing a target-free kernel is not an estimator fit.
                    # The original full RBF is reused, never retuned here.
                    if "rbf" not in self.kernels:
                        self.kernels["rbf"] = L.Kernel(
                            self.x["rbf"], self.indices, self.w, self.data.z, rbf=True)
                    self.kernels[family] = FixedBandwidthKernel(
                        self.x[family], self.indices, self.w, self.data.z,
                        self.kernels["rbf"].bandwidth)
            except ValueError as exc:
                if str(exc).startswith("RBF_bandwidth_UNAVAILABLE"):
                    self.unavailable[family] = str(exc)
                    value = None, {"family": family, "status": "UNAVAILABLE",
                                   "reason": str(exc)}
                    self.cache[key] = value
                    return value
                raise
        scaled, info, _ = self.kernels[family].predict(self.y, lam)
        prediction = (scaled*self.divisor + self.target_mean).reshape(-1, 10, 7)
        need(np.isfinite(prediction).all(), "finite_predictions_before_zero_override")
        prediction = np.array(prediction, copy=True)
        prediction[self.data.zero] = 0.
        need(np.isfinite(prediction).all(), "finite_predictions")
        info["family"] = family
        info["bandwidth_rule"] = (
            "own training design median squared distance" if family == "rbf_current4"
            else "same training Full design median as old rbf; no labels or held-out rows")
        self.data.fit_count += 1
        self.cache[key] = prediction, info
        return prediction, info

    def geometry(self):
        current = self.current_normalized[self.indices]
        projected = self.write_pca.apply(self.current_normalized)[self.indices]
        full_energy = float(self.w @ np.sum(current*current, axis=1))
        projected_energy = float(self.w @ np.sum(projected*projected, axis=1))
        if full_energy == 0:
            retained = None
        else:
            retained = projected_energy/full_energy
            need(-1e-10 <= retained <= 1+1e-10, "PCA_nested_energy_accounting")
        old = self.kernels.get("rbf")
        small = self.kernels.get("rbf_current4")
        full_median = None if old is None else old.bandwidth
        small_median = None if small is None else small.bandwidth
        result = {
            "training_clusters": list(self.clusters),
            "write_PCA_rank": len(self.write_pca.s),
            "complete_write_centered_weighted_energy": full_energy,
            "projected4_centered_weighted_energy": projected_energy,
            "compression_variance_retained": retained,
            "full_design_training_median_squared_distance": full_median,
            "current4_design_training_median_squared_distance": small_median,
            "median_ratio_current4_over_full": (
                None if full_median is None or small_median is None
                else small_median/full_median),
            "retained_energy_scope": "write block only; medians include the common context+dose",
            "capacity_caution": "Same lambda and bandwidth do not guarantee equal EDF.",
        }
        if old is not None and small is not None:
            # These are diagnostics on training geometry, not new kernel fits.
            def distances(x):
                x = x[self.indices]
                return np.maximum(np.sum(x*x, axis=1)[:, None] +
                                  np.sum(x*x, axis=1)[None] - 2*(x @ x.T), 0.)
            fd = distances(self.x["current_full"])
            cd = distances(self.x["current4"])
            tri = np.triu_indices(len(fd), 1)
            gap = fd[tri] - cd[tri]
            need(float(gap.min()) >= -1e-8*max(1., float(fd.max())),
                 "nested_PCA_pairwise_distance_nonincrease")
            result["training_pairwise_distance_residual"] = {
                "pairs": len(gap),
                "minimum_full_minus_current4": float(gap.min()),
                "median_full_minus_current4": float(np.median(gap)),
                "maximum_full_minus_current4": float(gap.max()),
                "ratio_of_mean_distances_current4_over_full": (
                    float(cd[tri].mean()/fd[tri].mean()) if fd[tri].mean() > 0 else None),
            }
        return result


def nested_choices(data, available):
    by_model = {family: [] for family in FAMILIES}
    details = []
    for held in available:
        training = [cluster for cluster in available if cluster != held]
        fold = data.fold(training)
        indices = data.row_indices([held])
        for family in FAMILIES:
            for lam in LAMBDAS:
                prediction, info = fold.predict(family, lam)
                row = {
                    "validation_cluster": held, "validation_task": data.tasks[held],
                    "training_clusters": training, "family": family, "lambda": lam,
                    "score": L.tune_score(fold, prediction, indices),
                    "contrast_score": L.tune_contrast_score(fold, prediction, indices),
                    "diagnostics": info,
                }
                details.append(row)
                by_model[family].append(row)
    candidates, contrast_candidates = {}, {}
    selected, contrast_selected = {}, {}
    for family, rows in by_model.items():
        grid, contrast_grid = [], []
        for lam in LAMBDAS:
            rr = [row for row in rows if row["lambda"] == lam]
            weights = L.group_weights([row["validation_task"] for row in rr],
                                      [row["validation_cluster"] for row in rr])
            for score_name, out in (("score", grid), ("contrast_score", contrast_grid)):
                score = (None if any(row[score_name] is None for row in rr)
                         else float(weights @ np.array([row[score_name] for row in rr])))
                out.append({"lambda": lam, "score": score})
        candidates[family], contrast_candidates[family] = grid, contrast_grid
        selected[family], contrast_selected[family] = L.choose(grid), L.choose(contrast_grid)
    return {
        "inner_folds": details,
        "candidate_scores": candidates,
        "selected_lambda": selected,
        "contrast_candidate_scores": contrast_candidates,
        "contrast_selected_lambda": contrast_selected,
        "simple_family_reselection": False,
        "endpoint_selection_criterion": "inner all-arm first5 standardized score only",
        "contrast_selection_criterion": (
            "inner same-query persistent shared-minus-random first5 standardized score"),
        "lambda_tie_rule": "largest lambda on exact score tie",
    }


def _root_comparison(data, root_scores, candidate, baseline):
    if candidate not in root_scores or baseline not in root_scores:
        return {"candidate": candidate, "baseline": baseline, "status": "UNAVAILABLE"}
    a, b = root_scores[candidate], root_scores[baseline]
    differences = b-a
    primary_a, primary_b = a[:, 0, 1].mean(1), b[:, 0, 1].mean(1)
    primary_d = primary_b-primary_a
    weights = L.group_weights([data.tasks[c] for c in data.clusters], data.clusters)
    defined = bool(np.isfinite(primary_a).all() and np.isfinite(primary_b).all())
    denom = float(weights @ primary_b) if defined else None
    improvement = float(weights @ primary_d) if defined else None
    rows = []
    counterexamples = []
    for i, cluster in enumerate(data.clusters):
        rows.append({
            "cluster": cluster, "task": data.tasks[cluster],
            "candidate_first5_standardized_MSE": float(primary_a[i]),
            "baseline_first5_standardized_MSE": float(primary_b[i]),
            "improvement_baseline_minus_candidate": float(primary_d[i]),
            "relative_improvement": (
                float(primary_d[i]/primary_b[i])
                if np.isfinite(primary_b[i]) and primary_b[i] > 0 else None),
            "MSE_difference_baseline_minus_candidate": differences[i],
        })
        for window, end in enumerate((5, 10)):
            for dimension in range(7):
                delta = differences[i, window, 1, dimension]
                if np.isfinite(delta) and delta < 0:
                    counterexamples.append({
                        "cluster": cluster, "task": data.tasks[cluster],
                        "window": f"first{end}", "action_dimension": dimension,
                        "candidate_standardized_MSE": float(a[i, window, 1, dimension]),
                        "baseline_standardized_MSE": float(b[i, window, 1, dimension]),
                        "improvement_baseline_minus_candidate": float(delta),
                        "candidate_over_baseline": (
                            float(a[i, window, 1, dimension]/b[i, window, 1, dimension])
                            if b[i, window, 1, dimension] > 0 else None),
                    })
    return {
        "candidate": candidate, "baseline": baseline,
        "status": "DESCRIPTIVE_FOUR_ROOT_DIAGNOSTIC" if defined else "UNDEFINED_STANDARDIZED_SCORE",
        "primary_metric": "equal-task/root first5 standardized seven-coordinate MSE",
        "candidate_mean_loss": float(weights @ primary_a) if defined else None,
        "baseline_mean_loss": denom,
        "mean_improvement_baseline_minus_candidate": improvement,
        "relative_mean_improvement": (
            improvement/denom if denom is not None and denom > 0 else None),
        "favorable_roots": int(np.count_nonzero(primary_d > 0)) if defined else None,
        "unfavorable_roots": int(np.count_nonzero(primary_d < 0)) if defined else None,
        "tied_roots": int(np.count_nonzero(primary_d == 0)) if defined else None,
        "roots": len(data.clusters), "per_root": rows,
        "all_unfavorable_standardized_root_window_coordinates": counterexamples,
        "hypothesis_test_or_power_claim": False,
        "raw_coordinate_errors": "all seven native coordinates retained separately; not summed across physical units",
    }


def comparisons(data, endpoint_roots, contrast_roots):
    pairs = (
        ("rbf", "rbf_current4", "PRIMARY_SAME_ALGORITHM_RULE"),
        ("rbf", "rbf_current4_full_bandwidth", "SECONDARY_COMMON_TRAIN_FULL_BANDWIDTH"),
        ("current_full", "current4", "LINEAR_REPLAY_CONTROL"),
        ("rbf_current4", "current4", "NONLINEARITY_WITH_COMPRESSED_REPRESENTATION"),
        ("rbf", "current_full", "NONLINEARITY_WITH_FULL_REPRESENTATION"),
        ("rbf_current4_full_bandwidth", "rbf_current4", "BANDWIDTH_CONVENTION_SENSITIVITY"),
    )
    records = []
    tracks = [(f"fixed_lambda_{lam:g}", "fixed__", f"__{lam:g}") for lam in LAMBDAS]
    tracks += [("endpoint_nested", "nested__", ""),
               ("contrast_nested", "contrast_nested__", "")]
    for candidate, baseline, role in pairs:
        for label, prefix, suffix in tracks:
            a, b = prefix+candidate+suffix, prefix+baseline+suffix
            records.append({
                "role": role, "track": label,
                "endpoint": _root_comparison(data, endpoint_roots, a, b),
                "shared_minus_random_contrast": _root_comparison(data, contrast_roots, a, b),
                "selection_scope": (
                    "fixed common lambda; no outer outcome selection" if prefix == "fixed__"
                    else "each family selected separately by the named inner-only criterion"),
            })
    zero_refs = []
    for family in ("rbf", "rbf_current4", "rbf_current4_full_bandwidth"):
        for label, prefix in (("endpoint_nested", "nested__"),
                              ("contrast_nested", "contrast_nested__")):
            zero_refs.append({
                "family": family, "track": label,
                "endpoint": _root_comparison(data, endpoint_roots, prefix+family, "zero"),
                "shared_minus_random_contrast": _root_comparison(
                    data, contrast_roots, prefix+family, "zero"),
            })
    return {"comparisons": records, "zero_references": zero_refs}


def run(dataset, check=lambda: None, synthetic=False):
    data = Dataset(dataset, check, synthetic)
    predictions, fold_receipts = {}, []
    scales = np.full_like(data.y, np.nan)
    for held in data.clusters:
        check()
        training = [cluster for cluster in data.clusters if cluster != held]
        fold, test = data.fold(training), data.row_indices([held])
        scales[test] = fold.target_std.reshape(10, 7)
        selection = nested_choices(data, training)
        receipt = {"held_out_cluster": held, "selection": selection, "model_diagnostics": {}}
        for family in FAMILIES:
            for lam in LAMBDAS:
                name = f"fixed__{family}__{lam:g}"
                prediction, info = fold.predict(family, lam)
                predictions.setdefault(name, np.full_like(data.y, np.nan))
                if prediction is not None:
                    predictions[name][test] = prediction[test]
                receipt["model_diagnostics"][name] = info
            for prefix, selector in (("nested__", "selected_lambda"),
                                     ("contrast_nested__", "contrast_selected_lambda")):
                name = prefix+family
                predictions.setdefault(name, np.full_like(data.y, np.nan))
                lam = selection[selector][family]
                if lam is not None:
                    prediction, info = fold.predict(family, lam)
                    if prediction is not None:
                        predictions[name][test] = prediction[test]
                    receipt["model_diagnostics"][name] = info
        receipt["kernel_geometry"] = fold.geometry()
        fold_receipts.append(receipt)
    predictions["zero"] = np.zeros_like(data.y)
    unavailable = {
        name: "one_or_more_fold_predictions_unavailable"
        for name, prediction in predictions.items() if not np.isfinite(prediction).all()
    }
    usable = {name: prediction for name, prediction in predictions.items() if name not in unavailable}
    grouped, loss_arrays = L.score_groups(data, usable, scales)
    contrast, contrast_arrays, contrast_roots = L.contrast(data, usable, scales)
    endpoint_roots = {
        name: np.stack([g["MSE"] for g in groups if g["arm"] == "equal_arm"])
        for name, groups in grouped.items()
    }
    fit_counts = {family: 0 for family in FAMILIES}
    attempts = {family: 0 for family in FAMILIES}
    for fold in data.folds.values():
        for (family, _), (prediction, _) in fold.cache.items():
            attempts[family] += 1
            fit_counts[family] += prediction is not None
    need(sum(fit_counts.values()) == data.fit_count, "fit_count_exact_cache_accounting")
    need(sum(attempts.values()) == data.fit_attempt_count, "fit_attempt_count_exact_accounting")
    need(len(data.folds) == 10, "exact_six_pair_and_four_triple_training_sets")
    need(data.fit_attempt_count == 150, "bounded_150_estimator_attempts")
    summary = {
        "schema": "rss-kernel-compression-comparator-v1",
        "analysis_complete": True, "synthetic": synthetic,
        "scope": "Historical persistent DEV compression versus nonlinear estimation; no pulse result",
        "schedule": dataset["schedule"], "arms": list(data.arms),
        "queries": len(data.q), "rows": len(data.y),
        "groups": dataset["groups"], "independence": dataset["independence"],
        "known_zero_edits": int(data.zero.sum()),
        "known_zero_override": "all models, all folds; no rows dropped",
        "families": list(FAMILIES), "old_replayed_families": list(OLD_FAMILIES),
        "new_comparator_families": list(NEW_FAMILIES), "fixed_lambdas": list(LAMBDAS),
        "folds": fold_receipts,
        "preprocessing": [fold.manifest() for fold in data.folds.values()],
        "geometry_by_training_set": [fold.geometry() for fold in data.folds.values()],
        "OOF_root_arm_scores": grouped, "unavailable_predictions": unavailable,
        "paired_persistent_contrast": contrast,
        "comparison_report": comparisons(data, endpoint_roots, contrast_roots),
        "new_fits": data.fit_count, "fit_attempts": data.fit_attempt_count,
        "fit_counts_by_family": fit_counts, "fit_attempts_by_family": attempts,
        "incremental_comparator_fits": sum(fit_counts[f] for f in NEW_FAMILIES),
        "old_family_replay_fits": sum(fit_counts[f] for f in OLD_FAMILIES),
        "expected_when_all_estimators_available": {"incremental": 60, "replay": 90, "total": 150},
        "preprocessing_fit_sets": len(data.folds), "PCA_fit_count": 3*len(data.folds),
        "new_fits_scope": "successful estimator fits, 60 incremental and 90 replay when available",
        "primary": "RBF Full versus RBF current4, each own train-only median bandwidth, equal three-lambda budget",
        "secondary": "RBF Full versus RBF current4 using the same train-Full bandwidth; no bandwidth search",
        "capacity": "EDF and kernel rank reported for every fit; equal lambda is not equal capacity",
        "causal_limit": "Estimator-conditional held-init prediction diagnostic, not proof that discarded dimensions carry or lack causal information",
        "outer_estimand": "new initialization within two represented tasks; NOT held-task validation",
        "inner_support_caution": "An inner fold can train only on the other task; tuning mixes those supports.",
        "primary_outcome_cut": "all-arm and shared-minus-random first5 standardized seven-coordinate diagnostic",
        "full_retention": "first5 and full10, seven native coordinates, every root and unfavorable cell retained",
        "bandwidth_grid_expanded": False, "family_winner_selection": False,
        "learning_curve_executed": False, "all_DEV_refit_executed": False,
        "CONF_used": False, "new_blind_test": False, "pulse_precision": None,
        "model_calls": 0, "environment_steps": 0, "policy_training_updates": 0,
        "old_frozen_results_replaced": False, "legacy_core_sha256": LEGACY_SHA256,
    }
    arrays = {
        "actual_targets": data.y, "OOF_target_scale": scales, "zero_actual_E": data.zero,
        "row_keys": np.array([[str(value) for value in row[:3]] for row in data.row_meta]),
        **{"prediction__"+name: value for name, value in predictions.items()},
        **{"query_loss__"+name: value for name, value in loss_arrays.items()},
        **contrast_arrays,
    }
    summary["fold_prediction_cache"] = []
    for i, fold in enumerate(data.folds.values()):
        arrays[f"fold_{i:02d}__target_scale"] = fold.target_std
        arrays[f"fold_{i:02d}__training_indices"] = fold.indices
        for (family, lam), (prediction, info) in fold.cache.items():
            key = f"fold_{i:02d}__{family}__lambda{lam:g}"
            if prediction is not None:
                arrays[key] = prediction
            summary["fold_prediction_cache"].append({
                "array": key if prediction is not None else None,
                "training_clusters": list(fold.clusters), "family": family,
                "lambda": lam, "diagnostics": info,
                "row_order": "same global row_keys, including train and validation rows",
            })
    return {"summary": serial(summary), "arrays": arrays}
