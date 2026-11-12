"""Fixed DEV16 native observability readout; no predictor or physical effect."""
import copy
import math
import statistics

ROOTS = tuple("SDRU-CDEV-" + t + "-" + r
              for t in ("T00", "T01") for r in ("R00", "R01"))
KEYS = tuple((root, q) for root in ROOTS for q in (11, 22))
ROLE_PHASES = (
    ("baseline_none_1", "baseline_none_2", "V_own_null_pulse", "V_own_null_persistent"),
    ("shared_pulse", "shared_persistent", "random_pulse"),
    ("shared_pulse_repeat", "shared_persistent_repeat"),
)
CALL_IDS = tuple(f"{root}-Q{q:02d}-{role}"
                 for roles in ROLE_PHASES for root, q in KEYS for role in roles)
DIMENSIONS = tuple("translation_command_" + d for d in "xyz") + tuple(
    "rotation_command_" + d for d in "xyz") + ("gripper_command",)
WINDOWS = (("first5", 5), ("full10", 10))
CONTRASTS = ("S_minus_N", "R_minus_N", "P_minus_N", "S_minus_R", "P_minus_S")
RUN = "jinwen-paper2-dev-pulse-native-20261020-v2"
PRODUCER_SOURCE_SHA = "a00c21f8299c033f431d7a9cc3b5593eb58602c99c7dd8f7fe22b964ccccebf3"
UNIT = "command_adapter_units"
CONSISTENCY_ATOL = 1e-12
CONSISTENCY_RTOL = 1e-10


def need(ok, code):
    if not ok:
        raise ValueError(code)


def finite_number(value, code):
    need(type(value) in (int, float) and math.isfinite(value), code)
    return float(value)


def nonnegative(value, code):
    value = finite_number(value, code)
    need(value >= 0, code)
    return value


def native_array(value):
    need(type(value) is list and len(value) == 1 and
         type(value[0]) is list and len(value[0]) == 10,
         "adapter7_shape_1_10_7")
    result = []
    for position in value[0]:
        need(type(position) is list and len(position) == 7, "adapter7_shape_1_10_7")
        result.append([finite_number(x, "adapter7_finite_numeric") for x in position])
    return result


def consistent(a, b):
    return abs(a - b) <= CONSISTENCY_ATOL + CONSISTENCY_RTOL * max(abs(a), abs(b))


def _call_list(value, code):
    need(type(value) is list and value == list(CALL_IDS), code)


def _keys(rows, code):
    need(type(rows) is list and len(rows) == 8 and
         all(type(r) is dict and type(r.get("query_index")) is int for r in rows),
         code)
    keys = [(r["root_id"], r["query_index"]) for r in rows]
    need(len(set(keys)) == 8 and set(keys) == set(KEYS), code)
    return {key: row for key, row in zip(keys, rows)}


def qualify(terminal, launcher, runtime, refs, *, synthetic=False):
    """Input byte SHA is checked by adapter; here verify producer semantics."""
    need(type(synthetic) is bool, "explicit_synthetic_scope")
    need(terminal["schema"] == "rss-pulse-compact-terminal-v1", "terminal_schema")
    w = terminal["result"]
    need(w["schema"] == "DEV-pulse-worker-result-v1", "worker_schema")
    need(w["synthetic"] is synthetic and launcher["synthetic"] is synthetic,
         "producer_synthetic_scope")
    need(w["status"] == "DEV_PULSE_NATIVE_COMPLETE" and
         launcher["status"] == "DEV_PULSE_NATIVE_COMPLETE", "native_status")
    need(all(w[k] is True for k in ("complete", "qualification_complete",
                                   "initial_qualification_complete",
                                   "source_roster_native_qualified")),
         "all_native_qualification_flags")
    need(w["planned_queries"] == [list(k) for k in KEYS], "fixed_eight_keys")
    need(w["failures"] == [], "no_native_failures")
    need(w["environment_steps"] == w["training_updates"] == w["new_statistical_fits"] == 0,
         "native_no_environment_training_or_fit")
    need(w["full_generations_charged"] == 72 and w["prefix_prefills_charged"] == 144 and
         w["suffix_evaluations_charged"] == 720, "native72_counters")
    for key in ("charged_calls", "returned_native_calls", "completed_calls", "passed_calls"):
        _call_list(w[key], "exact72_worker_" + key)
        _call_list(launcher[key], "exact72_launcher_" + key)
    _call_list(terminal["charged_calls"], "exact72_storage_charged")
    persisted = terminal["persisted_calls"]
    need(type(persisted) is list and len(persisted) == 72 and
         len(set(persisted)) == 72 and set(persisted) == set(CALL_IDS),
         "exact72_storage_persisted")
    need(terminal["unexecuted_call_ids"] == [] and
         terminal["stopped_on_storage_failure"] is False and
         terminal["native_graph_modified"] is False, "storage_complete_unmodified_graph")
    need(w["R_delayed_repeat"] is False and
         w["physical_meaningful_threshold"] is None and
         w["old_G_endpoint_bitwise_parity_claim"] is False and
         w["old_persistent_results_relabelled_as_pulse"] is False,
         "native_scope_not_expanded")
    qs = _keys(w["query_status"], "query_status_fixed8")
    for row in qs.values():
        need(all(type(row[k]) is int and row[k] == 9
                 for k in ("planned_calls", "charged_calls", "passed_calls")) and
             row["status"] == "COMPLETE" and row["unexecuted_call_ids"] == [],
             "all8queries9calls_pass")
    need(launcher["schema"] == "DEV-pulse-native-launcher-result-v1" and
         launcher["run_id"] == runtime["run_id"] == RUN, "exact_producer_run")
    need(launcher["source_manifest_sha256"] == PRODUCER_SOURCE_SHA,
         "pinned_executed_native_source")
    need(all(launcher[k] is True for k in ("worker_complete", "worker_returned",
                                         "worker_started", "qualification_complete")) and
         launcher["worker_status"] == w["status"], "launcher_worker_complete")
    need(launcher["model_binding"]["model_bound"] is True and
         launcher["model_binding"]["adapter_bound"] is True, "bound_model_adapter")
    need(launcher["model_loads_started"] == launcher["model_loads_completed"] == 1 and
         launcher["environment_steps"] == launcher["training_updates"] == 0 and
         launcher["automatic_retry"] is False and launcher["new_scientific_grid"] is False,
         "launcher_scope")
    expected_terminal = {"path": "ledger/TERMINAL.json",
                         **{k: refs["TERMINAL.json"][k] for k in ("bytes", "sha256")}}
    need(launcher["worker_terminal"] == expected_terminal, "launcher_terminal_binding")
    need(runtime["schema"] == "DEV-pulse-native-parent-result-v1" and
         runtime["failure"] is None and runtime["child"]["returncode"] == 0 and
         runtime["automatic_retry"] is False, "runtime_completed_child")
    archive = runtime["science_archive"]
    need(archive["complete_for_observed_files"] is True and archive["errors"] == [],
         "verified_runtime_science_archive")
    for name, path in (("TERMINAL.json", "ledger/TERMINAL.json"), ("RESULT.json", "RESULT.json")):
        matches = [r for r in archive["objects"] if r["path"] == path]
        need(len(matches) == 1, "one_archived_producer_record")
        row = matches[0]
        need(row["state"] == "VERIFIED" and
             {k: row[k] for k in ("bytes", "sha256")} ==
             {k: refs[name][k] for k in ("bytes", "sha256")} and
             row["receipt"]["relative"] == "science/" + path and
             all(row["receipt"][k] == refs[name][k] for k in ("bytes", "sha256", "uri")),
             "producer_record_archive_SHA_URI_binding")
    need(runtime["storage"]["JPFS_new_bytes"] == 0, "producer_JPFS_zero")
    observations = _keys(w["observability"], "observability_fixed8")
    for row in observations.values():
        need(row["qualification_status"] == "ALL72_CHECKS_PASSED" and
             row["eligible_for_scientific_summary"] is True,
             "terminal_all72_qualified_observability")
        need(row["physical_motion_and_task_success"] is None and
             row["P_minus_S_identifies_duration_or_feedback_alone"] is False and
             row["R_delayed_repeat_performed"] is False, "observation_scope")
    return w, observations


def metrics(values):
    n = len(values)
    return {
        "RMS": math.sqrt(math.fsum(x * x for x in values) / n),
        "mean_signed": math.fsum(values) / n,
        "meanabs": math.fsum(abs(x) for x in values) / n,
        "maxabs": max(abs(x) for x in values),
        "zero_positions": sum(x == 0 for x in values),
        "positions": n,
        "all_zero": all(x == 0 for x in values),
    }


def descriptive(values):
    need(bool(values) and all(math.isfinite(x) for x in values), "finite_summary_support")
    return {"minimum": min(values), "median": statistics.median(values), "maximum": max(values)}


def sr_classification(s, r):
    if s == 0 and r == 0:
        return "both_zero"
    if s == 0:
        return "S_zero_only"
    if r == 0:
        return "R_zero_only"
    if s == r:
        return "equal_nonzero_RMS"
    return "S_larger_RMS" if s > r else "R_larger_RMS"


def analyze(terminal, launcher, runtime, refs, *, synthetic=False, check=lambda: None):
    w, observations = qualify(terminal, launcher, runtime, refs, synthetic=synthetic)
    rows, snapshots, comparisons, arithmetic = [], [], [], []
    for root, q in KEYS:
        check()
        item = observations[root, q]
        need(set(item["arms"]) == {"S", "R", "P"}, "exact_S_R_P_arms")
        arrays = {}
        for arm in ("S", "R", "P"):
            a = item["arms"][arm]
            need(a["native_compute_precision_changed"] is False, "native_precision_unchanged")
            arrays[arm + "_minus_N"] = native_array(a["adapter7_effect"])
            for field, shape, coordinates in (
                ("adapter7_first5", [1, 5, 7], 35),
                ("adapter7_full10", [1, 10, 7], 70),
            ):
                old = a["stage_changes"][field]
                need(old["shape"] == shape and old["coordinates"] == coordinates and
                     old["logical_dtype"] == "float64", "old_adapter_counter_ABI")
        for contrast, a, b in (("S_minus_R", "S", "R"), ("P_minus_S", "P", "S")):
            explicit = native_array(item[contrast + "_adapter7"])
            old = item[contrast]
            need(old["shape"] == [1, 10, 7] and old["coordinates"] == 70 and
                 old["logical_dtype"] == "float64", "old_contrast_counter_ABI")
            residuals = []
            for p in range(10):
                for d in range(7):
                    implied = arrays[a + "_minus_N"][p][d] - arrays[b + "_minus_N"][p][d]
                    actual = explicit[p][d]
                    need(consistent(actual, implied), "explicit_contrast_numerical_consistency")
                    residuals.append(abs(actual - implied))
            arithmetic.append({"root_id": root, "query_index": q, "contrast": contrast,
                               "coordinates_checked": 70, "maximum_absolute_residual": max(residuals),
                               "numeric_consistent": True, "typed_byte_identity_claim": False})
            arrays[contrast] = explicit
        snapshot_rows = []
        for window, n in WINDOWS:
            for dimension, label in enumerate(DIMENSIONS):
                by_contrast = {}
                for contrast in CONTRASTS:
                    values = [v[dimension] for v in arrays[contrast][:n]]
                    row = {"root_id": root, "task_id": root.split("-")[2], "query_index": q,
                           "dimension_index": dimension, "command_dimension": label,
                           "window": window, "contrast": contrast, **metrics(values),
                           "unit": UNIT, "proposed_not_issued": True}
                    rows.append(row)
                    by_contrast[contrast] = row
                s, r = (by_contrast[k] for k in ("S_minus_N", "R_minus_N"))
                comparison = {"root_id": root, "query_index": q, "command_dimension": label,
                              "window": window, "S_RMS": s["RMS"], "R_RMS": r["RMS"],
                              "class": sr_classification(s["RMS"], r["RMS"]),
                              "S_zero_positions": s["zero_positions"],
                              "R_zero_positions": r["zero_positions"], "unit": UNIT,
                              "comparison_is_descriptive_not_significance": True}
                comparisons.append(comparison)
                snapshot_rows.append(comparison)
        dose = copy.deepcopy(item["actual_dose"])
        s = nonnegative(dose["shared_global_norm"], "finite_shared_dose")
        r = nonnegative(dose["random_global_norm"], "finite_random_dose")
        delta = finite_number(dose["global_random_minus_shared_norm"], "finite_dose_delta")
        need(consistent(delta, r - s), "dose_delta_numerical_consistency")
        need(type(dose["exact_actual_dose_match"]) is bool and
             dose["redraw_or_rescale"] is False, "unchanged_actual_dose")
        snapshots.append({
            "root_id": root, "query_index": q, "qualification_status": item["qualification_status"],
            "source_query_status": copy.deepcopy(next(x for x in w["query_status"]
                                                    if (x["root_id"], x["query_index"]) == (root, q))),
            "actual_dose_original": dose,
            "global_dose_relative_error": None if s == 0 else (r - s) / s,
            "global_dose_relative_error_status": "undefined_shared_zero" if s == 0 else "defined",
            "old_stage_counters": {
                arm: {k: copy.deepcopy(v) for k, v in item["arms"][arm].items()
                      if k != "adapter7_effect"} for arm in ("S", "R", "P")},
            "old_contrast_counters": {k: copy.deepcopy(item[k]) for k in ("S_minus_R", "P_minus_S")},
            "old_mixed_coordinate_squared_norm_is_not_new_physical_metric": True,
            "S_R_axis_window_comparisons": snapshot_rows,
        })
    need(len(rows) == 560, "fixed560_scalar_cells")
    summaries = []
    for contrast in CONTRASTS:
        for window, _ in WINDOWS:
            for dimension in DIMENSIONS:
                group = [r for r in rows if (r["contrast"], r["window"], r["command_dimension"])
                         == (contrast, window, dimension)]
                need(len(group) == 8, "eight_snapshots_per_scalar_summary")
                summaries.append({
                    "contrast": contrast, "window": window, "command_dimension": dimension,
                    "snapshots": 8, "unit": UNIT,
                    "metrics": {k: descriptive([r[k] for r in group])
                                for k in ("RMS", "mean_signed", "meanabs", "maxabs", "zero_positions")},
                    "all_zero_snapshots": sum(r["all_zero"] for r in group),
                    "independent_samples_claim": False,
                })
    classes = ("both_zero", "S_zero_only", "R_zero_only", "equal_nonzero_RMS",
               "S_larger_RMS", "R_larger_RMS")
    sr_summary = []
    for window, _ in WINDOWS:
        for dimension in DIMENSIONS:
            group = [r for r in comparisons if (r["window"], r["command_dimension"])
                     == (window, dimension)]
            counts = {k: sum(r["class"] == k for r in group) for k in classes}
            need(sum(counts.values()) == 8, "S_R_comparison_full_denominator")
            sr_summary.append({"window": window, "command_dimension": dimension,
                               "snapshots": 8, "counts": counts, "unit": UNIT})
    dose_errors = [r["global_dose_relative_error"] for r in snapshots
                   if r["global_dose_relative_error"] is not None]
    scope = {"synthetic": synthetic, "new_model_calls": 0, "new_env_steps": 0,
             "training_updates": 0, "new_fits": 0, "JPFS_new_bytes": 0,
             "new_blind_holdout": False, "physical_meaningful_threshold": None,
             "prediction_or_policy_efficacy_claim": False,
             "P_minus_S_identifies_duration_or_feedback_alone": False}
    summary = {"schema": "paper2-dev16-axis-window-summary-v1", **scope,
               "scalar_cells": 560, "summaries": summaries, "S_R_RMS_comparisons": sr_summary,
               "dose": {"snapshots": 8, "defined_relative_errors": len(dose_errors),
                        "undefined_shared_zero": 8 - len(dose_errors),
                        "relative_error_definition": "(random_global_norm - shared_global_norm) / shared_global_norm",
                        "signed_relative_error": descriptive(dose_errors) if dose_errors else None,
                        "absolute_relative_error": descriptive([abs(x) for x in dose_errors])
                        if dose_errors else None,
                        "original_exact_actual_dose_match_true_count": sum(
                            r["actual_dose_original"]["exact_actual_dose_match"] for r in snapshots),
                        "approximate_global_norm_is_not_exact_token_dose_match": True},
               "population": {"tasks": 2, "initialization_roots": 4, "snapshots": 8,
                              "queries_per_root": 2, "windows_overlap": True,
                              "all_observed_support_retained": True,
                              "independent_snapshots_or_axes_claim": False}}
    qa = {"schema": "paper2-dev16-readout-QA-v1", **scope, "pass": True,
          "fixed560_scalar_cells": True, "fixed70_summaries": len(summaries) == 70,
          "fixed14_S_R_summary_groups": len(sr_summary) == 14,
          "all72_producer_calls_passed": True, "original_native_qualification_reexecuted": False,
          "explicit_contrast_arithmetic_checks": arithmetic,
          "numerical_consistency_tolerance": {"atol": CONSISTENCY_ATOL, "rtol": CONSISTENCY_RTOL,
                                             "purpose": "JSON FP64 subtraction consistency only",
                                             "not_physical_or_scientific_effect_threshold": True},
          "producer_source_manifest_sha256": PRODUCER_SOURCE_SHA,
          "producer_model_binding": copy.deepcopy(launcher["model_binding"]),
          "producer_counters_preserved": {
              k: copy.deepcopy(w[k]) for k in ("charged_calls", "returned_native_calls", "completed_calls",
                                              "passed_calls", "full_generations_charged",
                                              "prefix_prefills_charged", "suffix_evaluations_charged")},
          "historical_flags_preserved_not_authority_or_release": {
              "terminal_GPU_authorization_granted": terminal["GPU_authorization_granted"],
              "terminal_release_proven_by_storage": terminal["release_proven_by_storage"],
              "launcher_GPU_release_confirmed": launcher["GPU_release_confirmed"],
              "runtime_provider_release_proven": runtime["provider_release_proven"]},
          "actual_GPU_release_not_reassessed_by_this_CPU_analysis": True}
    snapshot_support = {"schema": "paper2-dev16-snapshot-support-v1", **scope,
                        "snapshots": snapshots, "snapshot_count": 8,
                        "numerical_zero_not_physical_no_effect": True}
    return rows, summary, qa, snapshot_support
