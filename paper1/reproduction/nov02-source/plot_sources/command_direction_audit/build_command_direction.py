#!/usr/bin/env python3
"""Recompute recorded q0 command directions; no inference, fits or bootstrap.

All inputs are local, hash-bound copies. CONF vectors are the actual first-query
issued adapter7 prefixes. DEV fields describe unexecuted saved-query plans and
are kept separate. Uses only the Python standard library.
"""
from collections import Counter, defaultdict
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "inputs"
FAMILIES = ("Camera Viewpoints", "Sensor Noise")
ARMS = ("N_C", "N_T", "R_resp", "R_clean", "R_rand", "R_full")
GROUPS = {"translation": (0, 1, 2), "rotation": (3, 4, 5), "gripper": (6,)}
METRICS = ("signed_along_clean", "orthogonal_norm", "edit_displacement_norm",
           "native_error", "edited_error", "error_gain", "positive_overshoot")

def require(condition, message):
    if not condition:
        raise AssertionError(message)

def sha(data):
    return hashlib.sha256(data).hexdigest()

def load_json(name):
    return json.loads((INPUT / name).read_text())

def write_json(name, value):
    (ROOT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                      allow_nan=False) + "\n")

def write_csv(name, rows):
    with (ROOT / name).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

def norm(values):
    return math.sqrt(math.fsum(x * x for x in values))

def anchor_key(row):
    return (row["family"], row["task_id"], row["row_id"],
            row["noise_label"], row["init_index"])

def weighted_metric(rows, metric):
    selected = [(r[metric], r["weight"]) for r in rows if r[metric] is not None]
    mass = math.fsum(w for _, w in selected)
    complete = len(selected) == len(rows)
    return {
        "value": math.fsum(x * w for x, w in selected) / mass if complete and mass else None,
        "defined_records": len(selected), "total_records": len(rows),
        "defined_weight_mass": mass,
        "status": "complete" if complete else "undefined_on_fixed_support",
    }

def main():
    manifest = load_json("SOURCE-MANIFEST.json")
    for identity in manifest["files"]:
        data = (INPUT / identity["name"]).read_bytes()
        require(sha(data) == identity["stored_sha256"], "Stored input hash mismatch")
        if identity["storage_transform"].startswith("gzip with"):
            data = gzip.decompress(data)
        require(sha(data) == identity["source_sha256"], "Original input hash mismatch")

    with gzip.open(INPUT / "CASE-BEHAVIOR.jsonl.gz", "rt") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    require(len(rows) == 1296, "CONF row count")
    split = load_json("FROZEN-SPLIT.json")["families"]
    scales = load_json("FROZEN-SCALES.json")["scales"]
    scalar_rows = load_json("CONF-Q0-COMMAND-SCALARS-v1.json")["records"]
    scalars = {(anchor_key(r), r["arm"]): r for r in scalar_rows}
    require(len(scalars) == len(scalar_rows) == 1296, "q0 scalar keys")
    anchors = defaultdict(dict)
    variants = defaultdict(lambda: defaultdict(set))
    source_errors = defaultdict(float)
    source_NA = Counter()
    for row in rows:
        key = anchor_key(row)
        require(row["arm"] not in anchors[key], "Duplicate CONF arm")
        require(row["status"] == "COMPLETE", "Missing CONF record")
        issued = row["issued"]
        require(issued["first_query_actual_controls"] == 5, "Not five returned controls")
        prefix = issued["first_query_issued_prefix7"]
        require(len(prefix) == 5 and all(len(p) == 7 for p in prefix), "Prefix shape")
        require(all(type(x) in (float, int) and math.isfinite(x) for p in prefix for x in p),
                "Nonfinite prefix")
        require(issued["plan_prefix_L2"] == issued["plan_prefix_maxabs"] == 0,
                "Original issued/plan identity diagnostic")
        require(issued["no_unissued_commands_fabricated"] is True, "Issued provenance")
        require(row["noise_label"] in (0, 1) and row["init_index"] == 0, "CONF identity")
        family_split = split[row["family"]]["confirm"]
        require(row["row_id"] in family_split["row_ids"] and
                row["task_id"] in family_split["task_ids"], "Not frozen CONF member")
        scalar = scalars[(key, row["arm"])]
        require(scalar["status"] == "COMPLETE", "Missing original q0 scalar")
        require(row["source_q0_plan_identity"]["sha256"] ==
                scalar["source_plan_identity"]["sha256"], "Plan identity differs")
        anchors[key][row["arm"]] = row
        variants[row["family"]][row["task_id"]].add(row["row_id"])
    require(len(anchors) == 216, "CONF anchor support")
    require(len({key[1] for key in anchors}) == 31, "CONF task union")

    records = []
    support = {}
    for family in FAMILIES:
        ss = split[family]["confirm"]
        actual_variants = {r["row_id"] for r in rows if r["family"] == family}
        require(actual_variants == set(ss["row_ids"]), "Incomplete frozen CONF variants")
        require(set(variants[family]) == set(ss["task_ids"]), "Incomplete frozen CONF tasks")
        support[family] = {"tasks": len(variants[family]), "variants": len(actual_variants),
                           "anchors": 2 * len(actual_variants), "arms": len(ARMS)}

    for key, arm_rows in sorted(anchors.items()):
        require(set(arm_rows) == set(ARMS), "Incomplete six-arm anchor")
        family, task, variant, noise, init = key
        weight = 1 / (len(variants[family]) * len(variants[family][task]) * 2)
        for group, indices in GROUPS.items():
            vectors = {arm: [p[i] for p in row["issued"]["first_query_issued_prefix7"]
                             for i in indices] for arm, row in arm_rows.items()}
            target = [x - y for x, y in zip(vectors["N_C"], vectors["N_T"])]
            native_error = norm(target)
            direction = [x / native_error for x in target] if native_error else None
            for arm in ARMS:
                row = arm_rows[arm]
                delta = [x - y for x, y in zip(vectors[arm], vectors["N_T"])]
                along = math.fsum(x * y for x, y in zip(delta, direction)) if direction else None
                off = norm([x - along * u for x, u in zip(delta, direction)]) if direction else None
                displacement = norm(delta)
                edited_error = norm([x - y for x, y in zip(vectors[arm], vectors["N_C"])])
                overshoot = max(0.0, along - native_error) if along is not None else None
                record = {
                    "family": family, "task_id": task, "row_id": variant,
                    "noise_label": noise, "init_index": init, "arm": arm,
                    "command_group": group, "vector_dimension": len(delta),
                    "weight": weight, "signed_along_clean": along,
                    "orthogonal_norm": off, "edit_displacement_norm": displacement,
                    "native_error": native_error, "edited_error": edited_error,
                    "error_gain": native_error - edited_error,
                    "positive_overshoot": overshoot, "direction_defined": native_error > 0,
                    "units": "adapter_command_units_within_named_component",
                    "source_episode_sha256": row["source_episode_identity"]["sha256"],
                    "source_plan_sha256": row["source_q0_plan_identity"]["sha256"],
                }
                archived = next(q for q in row["q0_direction"] if q["command_group"] == group)
                require(archived["units"] == "adapter_command_units", "Direction units")
                require(archived["direction_NA"] is (native_error == 0), "Direction NA mismatch")
                source_NA[(group, native_error == 0)] += 1
                for metric, source_metric in [("signed_along_clean", "signed_along_clean"),
                       ("orthogonal_norm", "off_target"), ("native_error", "native_error"),
                       ("edited_error", "edited_error"), ("error_gain", "error_gain"),
                       ("positive_overshoot", "positive_overshoot")]:
                    new = record[metric]
                    old = archived[source_metric]
                    if new is not None:
                        require(old is not None and math.isfinite(old), "Source scalar missing")
                        source_errors[f"{group}:{source_metric}"] = max(
                            source_errors[f"{group}:{source_metric}"], abs(new - old))
                        require(math.isclose(new, old, abs_tol=2e-14, rel_tol=2e-12),
                                "Recomputed/source direction disagreement")
                if group == "translation":
                    require(edited_error == scalars[(key, arm)]["command_error"],
                            "Original confirm q0 error not exactly reproduced")
                records.append(record)

    family_rows = []
    task_rows = []
    for family in FAMILIES:
        for group in GROUPS:
            for arm in ARMS:
                selected = [r for r in records if r["family"] == family and
                            r["command_group"] == group and r["arm"] == arm]
                require(math.isclose(math.fsum(r["weight"] for r in selected), 1,
                                     abs_tol=1e-14), "Family fixed weight mass")
                record = {"family": family, "command_group": group, "arm": arm,
                          **support[family], "metrics": {m: weighted_metric(selected, m) for m in METRICS}}
                family_rows.append(record)
                for task in sorted(variants[family]):
                    task_selected = [r for r in selected if r["task_id"] == task]
                    task_rows.append({"family": family, "command_group": group, "arm": arm,
                        "task_id": task, "variants": len(variants[family][task]),
                        "noise_paired_executions": len(task_selected),
                        **{m: weighted_metric(task_selected, m)["value"] for m in METRICS}})

    q0_verification = []
    lookup = {(r["family"], r["command_group"], r["arm"]): r for r in family_rows}
    readout = load_json("CONF-Q0-COMMAND-READOUT-v1.json")
    for comparison in readout["comparisons"]:
        family, left, right = comparison["family"], comparison["lhs"], comparison["rhs"]
        calc = (lookup[family, "translation", right]["metrics"]["edited_error"]["value"] -
                lookup[family, "translation", left]["metrics"]["edited_error"]["value"]) / scales[family]
        require(math.isclose(calc, comparison["point"], abs_tol=1e-14, rel_tol=1e-12),
                "Original six q0 comparison means not reproduced")
        q0_verification.append({"family": family, "lhs": left, "rhs": right,
                               "recomputed_point": calc, "original_point": comparison["point"],
                               "absolute_error": abs(calc-comparison["point"]),
                               "original_interval_not_reused_for_new_direction_metrics": True})

    # DEV means use stored plan-component scalars on a separate frozen support.
    with gzip.open(INPUT / "DEV-TRANSLATION-ENDPOINTS.jsonl.gz", "rt") as stream:
        dev = [json.loads(line) for line in stream]
    require(len(dev) == 2000 and len({r["endpoint_id"] for r in dev}) == 2000, "DEV endpoints")
    dev_means = []
    dev_identity_errors = defaultdict(float)
    for row in dev:
        require(row["component"] == "translation" and row["executed_command"] is False,
                "DEV endpoint scope")
        c, a, b, disp, e = (row["clean_target_norm"], row["signed_along_clean"],
            row["off_target"], row["edit_displacement_norm"], row["remaining_clean_error"])
        dev_identity_errors["target_error_squared"] = max(dev_identity_errors["target_error_squared"],
                                                         abs(e*e-((c-a)**2+b*b)))
        dev_identity_errors["displacement_squared"] = max(dev_identity_errors["displacement_squared"],
                                                         abs(disp*disp-(a*a+b*b)))
        require(math.isclose(e*e, (c-a)**2+b*b, abs_tol=1e-12, rel_tol=1e-10), "DEV error identity")
        require(math.isclose(disp*disp, a*a+b*b, abs_tol=1e-12, rel_tol=1e-10), "DEV displacement identity")
    for family in FAMILIES:
        ss = split[family]["dev"]
        family_dev = [r for r in dev if r["family"] == family]
        require({r["row_id"] for r in family_dev} == set(ss["row_ids"]), "DEV variants")
        condition_keys = sorted({(r["arm"], r["schedule_role"]) for r in family_dev})
        require(len(condition_keys) == 10, "DEV ten conditions")
        var_by_task = defaultdict(set)
        for r in family_dev:
            var_by_task[r["task_id"]].add(r["row_id"])
        for arm, schedule in condition_keys:
            selected = [r for r in family_dev if (r["arm"], r["schedule_role"]) == (arm, schedule)]
            require(len(selected) == 8*ss["variants"], "DEV condition support")
            for r in selected:
                w = 1 / (ss["tasks"]*len(var_by_task[r["task_id"]])*2*4)
                require(math.isclose(w, r["endpoint_available_query_weight"], abs_tol=1e-14),
                        "DEV original hierarchy")
                require(r["query"] in (0, 2, 5, 10), "DEV query")
            metrics = ("signed_along_clean", "off_target", "edit_displacement_norm",
                       "clean_target_norm", "remaining_clean_error", "absolute_error_improvement",
                       "normalized_translation_error_improvement")
            dev_means.append({"family": family, "arm": arm, "schedule_role": schedule,
                "tasks": ss["tasks"], "variants": ss["variants"], "anchors": len(selected),
                "metrics": {m: math.fsum(r[m]*r["endpoint_available_query_weight"] for r in selected)
                            for m in metrics}, "units": "adapter_plan_component_units",
                "statistical_status": "descriptive_original_dev_support_no_new_interval"})

    translation = [r for r in family_rows if r["command_group"] == "translation"]
    summary = {
        "schema": "issued-command-direction-descriptive-audit-v1",
        "analysis_status": "new_descriptive_decomposition_on_original_confirmation_population",
        "inputs": manifest,
        "CONF": {"support": support, "distinct_task_union": 31, "anchors": 216,
            "original_records": 1296, "derived_component_records": len(records),
            "all_six_arms_retained": True, "outcomes_used_for_filtering": False,
            "actual_returned_controls_per_record": 5, "issued_prefix_coordinates": 7,
            "primary_display_component": "translation: first five positions times x/y/z = 15 coordinates",
            "weighting": "1 / (family task count * frozen variants in that task * 2 noise samples)",
            "definitions": {
                "c": "A_N_C - A_N_T on the same family/task/variant/noise/init anchor",
                "d": "A_arm - A_N_T on that anchor",
                "signed_along_clean": "dot(d,c)/norm(c)",
                "orthogonal_norm": "norm(d - signed_along_clean*c/norm(c))",
                "edited_error": "norm(A_arm-A_N_C)",
                "error_identity": "edited_error^2 = (norm(c)-signed_along_clean)^2 + orthogonal_norm^2 per execution",
                "averaging": "Average each scalar with original fixed task hierarchy; never apply the nonlinear identity to displayed means",
                "units": "Euclidean norm/projection in adapter command units within the named component; not RMS and not physical displacement",
                "zero_target_rule": "If norm(c)=0, along/orthogonal/overshoot are null; the error and displacement remain defined. No partial-support renormalization or zero filling.",
            },
            "zero_target_counts_by_component": {g: sum(r["native_error"] == 0 for r in records
                  if r["command_group"] == g and r["arm"] == "N_T") for g in GROUPS},
            "missing_prefix_or_outcome_filter_exclusions": 0,
            "source_q0_direction_max_abs_disagreement": dict(source_errors),
            "original_q0_error_matches_all_1296_exactly": True,
            "original_six_comparison_mean_reproduction": q0_verification,
            "translation_family_arm_means": translation,
            "all_component_family_arm_means": family_rows},
        "DEV": {"separate_population": True, "endpoint": "unexecuted first-five translation plan at saved q=0/2/5/10",
            "all_2000_translation_rows_retained": True, "scalar_geometry_identity_max_errors": dict(dev_identity_errors),
            "condition_means": dev_means,
            "norm_rule_scope": "Common requested target at each position, actual BF16 writes and future paths can differ; not exactly equal cumulative dose."},
        "inference": {"new_model_calls": 0, "new_episodes": 0, "new_fits": 0,
            "new_bootstrap_draws": 0, "new_confidence_intervals": 0,
            "original_error_CIs_reassigned_to_direction_metrics": False,
            "descriptive_means_do_not_establish_direction_optimality": True,
            "not_a_failure_predictor_or_mediation_analysis": True,
            "native_attention_P_W_validation_supplied_by_this_audit": False},
    }
    write_json("SUMMARY.json", summary)
    write_json("CONF-PER-EXECUTION.json", records)
    write_csv("CONF-PER-EXECUTION.csv", records)
    write_json("CONF-TASK-MEANS.json", task_rows)
    write_csv("CONF-TASK-MEANS.csv", task_rows)
    write_json("DEV-CONDITION-MEANS.json", dev_means)
    flat_means = [{"family": r["family"], "command_group": r["command_group"],
                   "arm": r["arm"], "tasks": r["tasks"], "variants": r["variants"],
                   "anchors": r["anchors"], **{m: r["metrics"][m]["value"] for m in METRICS}}
                  for r in family_rows]
    write_csv("CONF-FAMILY-ARM-MEANS.csv", flat_means)
    print(json.dumps({"status": "PASS", "confirm_records": len(rows),
        "confirm_anchors": len(anchors), "confirm_task_union": 31,
        "translation_means": [r for r in flat_means if r["command_group"] == "translation"],
        "new_bootstrap_draws": 0, "new_model_calls": 0}, indent=2))

if __name__ == "__main__":
    main()
