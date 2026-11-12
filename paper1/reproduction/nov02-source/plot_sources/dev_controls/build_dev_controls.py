#!/usr/bin/env python3
"""Rebuild DEV descriptive control summaries using only saved scalar inputs.

No fitting, inference, bootstrap, confidence intervals or outcome filtering.
Only writes dev_controls_summary.json and dev_controls_table.tex beside this file.
"""
from __future__ import annotations

import gzip
import hashlib
import itertools
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
SUPPORT = {"Camera Viewpoints": {"tasks": 5, "variants": 8, "anchors": 64},
           "Sensor Noise": {"tasks": 9, "variants": 17, "anchors": 136}}
CONDITIONS = [("N_C", "all_calls"), ("N_T", "all_calls"),
              ("R_resp", "first_call"), ("R_resp", "all_calls"),
              ("R_clean", "all_calls"), ("R_rand", "all_calls"),
              ("R_full", "all_calls"), ("eq_B", "all_calls"),
              ("eq_C", "all_calls"), ("eq_G", "all_calls")]
METRIC = "normalized_translation_error_improvement"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def finite(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def anchor(r: dict) -> tuple:
    return (r["task_id"], r["row_id"], r["noise_label"], r["init_index"], r["query"])


def condition_key(arm: str, schedule: str) -> str:
    return arm + ":" + schedule


def read_subset(provenance: dict, name: str) -> list[dict]:
    spec = provenance["subsets"][name]
    compressed = (HERE / spec["path"]).read_bytes()
    require(sha(compressed) == spec["gzip_sha256"], name + " compressed hash")
    raw = gzip.decompress(compressed)
    require(sha(raw) == spec["uncompressed_sha256"], name + " row-byte hash")
    rows = [json.loads(line) for line in raw.splitlines()]
    require(len(rows) == spec["rows"], name + " row count")
    return rows


def mean(rows: list[dict], metric: str) -> float:
    return math.fsum(r["endpoint_available_query_weight"] * r[metric] for r in rows) / math.fsum(
        r["endpoint_available_query_weight"] for r in rows)


def dose_mean(rows: list[dict], field: str) -> dict:
    present = [(r, r["dose"].get(field) if isinstance(r["dose"], dict) else None)
               for r in rows]
    defined = [(r, v) for r, v in present if finite(v)]
    native_no_edit = all(r["arm"] in {"N_C", "N_T"} and r["dose"] is None for r in rows)
    complete = len(defined) == len(rows)
    return {
        "weighted_mean": (math.fsum(r["endpoint_available_query_weight"] * v for r, v in defined)
                          / math.fsum(r["endpoint_available_query_weight"] for r in rows))
                         if complete else None,
        "status": "defined_on_all_anchors" if complete else
                  "not_applicable_native_no_edit" if native_no_edit else "incomplete_original_fields",
        "defined_anchors": len(defined), "planned_anchors": len(rows),
        "defined_weight_sum": math.fsum(r["endpoint_available_query_weight"] for r, _ in defined),
        "missing_values_zero_filled": False, "partial_support_renormalized": False,
        "scope": "Original endpoint-local dose field for this complete generated plan",
    }


def paired_difference(left: dict, right: dict) -> dict:
    require(set(left) == set(right), "paired anchors differ")
    require(all(left[k]["endpoint_available_query_weight"] ==
                right[k]["endpoint_available_query_weight"] for k in left), "paired weights differ")
    value = math.fsum(left[k]["endpoint_available_query_weight"] *
                     (left[k][METRIC] - right[k][METRIC]) for k in sorted(left))
    total = math.fsum(left[k]["endpoint_available_query_weight"] for k in left)
    return {"point": value / total, "paired_anchors": len(left), "weight_sum": total,
            "metric": METRIC, "confidence_interval_computed": False}


def main() -> None:
    provenance_bytes = (HERE / "provenance.json").read_bytes()
    provenance = json.loads(provenance_bytes)
    scale_spec = provenance["frozen_scales"]
    scale_bytes = (HERE / scale_spec["path"]).read_bytes()
    require(sha(scale_bytes) == scale_spec["sha256"], "frozen scale file hash")
    scale_data = json.loads(scale_bytes)
    scales = scale_data["scales"]
    endpoints = read_subset(provenance, "translation_endpoints")
    local = read_subset(provenance, "local_eqB_eqC_task_cells")
    require(len(endpoints) == 2000 and len(local) == 1120, "total support")
    require(len({r["endpoint_id"] for r in endpoints}) == 2000, "endpoint ID duplication")
    require({r["family"] for r in endpoints} == set(SUPPORT), "endpoint families")
    require({r["family"] for r in local} == set(SUPPORT), "local families")
    families = {}
    all_local_counts = Counter()

    for family, expected in SUPPORT.items():
        family_rows = [r for r in endpoints if r["family"] == family]
        require(all(r["component"] == "translation" for r in family_rows), "component filter")
        require(all(r["executed_command"] is False for r in family_rows), "plan/issued role")
        require(all(r["dose"] is None for r in family_rows if r["arm"] in {"N_C", "N_T"}),
                "native references retain their original null dose records")
        require(all(isinstance(r["dose"], dict) and
                    r["dose"]["strict_equal_cumulative_dose_claimed"] is False
                    for r in family_rows if r["arm"] not in {"N_C", "N_T"}),
                "edited-condition cumulative dose role")
        require(all(finite(r[METRIC]) and finite(r["absolute_error_improvement"])
                    for r in family_rows), "endpoint metric missing or nonfinite")
        require({(r["arm"], r["schedule_role"]) for r in family_rows} == set(CONDITIONS),
                "condition set")
        groups = {}
        summaries = []
        reference = None
        max_weight_error = 0.0
        max_scale_error = 0.0
        for arm, schedule in CONDITIONS:
            rows = [r for r in family_rows if (r["arm"], r["schedule_role"]) == (arm, schedule)]
            keyed = {anchor(r): r for r in rows}
            require(len(keyed) == len(rows) == expected["anchors"], "anchor duplicate/count")
            require({r["query"] for r in rows} == {0, 2, 5, 10}, "query support")
            require({r["noise_label"] for r in rows} == {0, 1}, "noise support")
            require({r["init_index"] for r in rows} == {0}, "initial-state support")
            tasks = {r["task_id"] for r in rows}
            variants_by_task = defaultdict(set)
            for r in rows:
                variants_by_task[r["task_id"]].add(r["row_id"])
            require(len(tasks) == expected["tasks"], "task support")
            require(len({r["row_id"] for r in rows}) == expected["variants"], "variant support")
            for r in rows:
                w = r["endpoint_available_query_weight"]
                require(finite(w) and w > 0, "invalid endpoint weight")
                target = 1 / (len(tasks) * len(variants_by_task[r["task_id"]]) * 2 * 4)
                max_weight_error = max(max_weight_error, abs(w - target))
                require(math.isclose(w, target, rel_tol=0, abs_tol=1e-14), "weight hierarchy")
                require(math.isclose(r["planned_fixed_query_weight"], 4 * w,
                                     rel_tol=0, abs_tol=1e-14), "fixed-query weight")
                max_scale_error = max(max_scale_error, abs(
                    r[METRIC] - r["absolute_error_improvement"] / scales[family]))
            require(math.isclose(math.fsum(r["endpoint_available_query_weight"] for r in rows),
                                 1, rel_tol=0, abs_tol=1e-12), "weight sum")
            if reference is None:
                reference = keyed
            else:
                require(set(keyed) == set(reference), "conditions have different anchors")
                require(all(keyed[k]["endpoint_available_query_weight"] ==
                            reference[k]["endpoint_available_query_weight"] for k in keyed),
                        "conditions have different weights")
            key = condition_key(arm, schedule)
            groups[key] = keyed
            summaries.append({"arm": arm, "schedule_role": schedule, "anchors": len(rows),
                              "weight_sum": math.fsum(r["endpoint_available_query_weight"] for r in rows),
                              "mean_normalized_translation_error_improvement": mean(rows, METRIC),
                              "mean_absolute_error_improvement": mean(rows, "absolute_error_improvement"),
                              "dose": {field: dose_mean(rows, field) for field in
                                       ("actual_write_path_length", "call0_actual_write_RMS_token")}})
        require(max_scale_error < 1e-12, "stored normalized endpoint versus original S_f")

        cells = [r for r in local if r["family"] == family]
        require(all(r["view"] == "local_pair_32" and r["axes"]["left"] == "eq_B"
                    and r["axes"]["right"] == "eq_C" for r in cells), "local filter")
        tasks = {r["task_id"] for r in family_rows}
        keys = [(r["task_id"], r["query"], r["axes"]["call"],
                 r["axes"]["position_group"]) for r in cells]
        grid = set(itertools.product(tasks, [0, 2, 5, 10], range(10),
                                     ["positions_0_4", "positions_5_9"]))
        require(len(keys) == len(set(keys)) and set(keys) == grid, "local grid support")
        ordering = Counter({"C_greater_than_B": 0, "C_equal_B": 0, "C_less_than_B": 0})
        write_signs = Counter({"B_minus_C_positive": 0, "B_minus_C_zero": 0,
                               "B_minus_C_negative": 0})
        metric_status = defaultdict(Counter)
        for r in cells:
            for metric, record in r["metrics"].items():
                metric_status[metric][record["status"]] += 1
            for metric in ("left_norm32", "right_norm32", "actual_write_norm_difference_left_minus_right"):
                record = r["metrics"][metric]
                require(record["status"] == "defined" and finite(record["mean"]),
                        "required local scalar undefined")
                require(record["defined_variant_noise_units"] == record["planned_variant_noise_units"],
                        "local norm underlying support incomplete")
                require(record["planned_variant_noise_units"] == 2 * len(variants_by_task[r["task_id"]]),
                        "local variant/noise support")
                require(math.isclose(record["sum_saved_fixed_query_weights"], 1 / expected["tasks"],
                                     rel_tol=0, abs_tol=1e-12), "local saved task weight")
            b = r["metrics"]["left_norm32"]["mean"]
            c = r["metrics"]["right_norm32"]["mean"]
            ordering["C_greater_than_B" if c > b else "C_less_than_B" if c < b else "C_equal_B"] += 1
            d = r["metrics"]["actual_write_norm_difference_left_minus_right"]["mean"]
            write_signs["B_minus_C_positive" if d > 0 else "B_minus_C_negative" if d < 0
                        else "B_minus_C_zero"] += 1
        all_local_counts.update(ordering)
        contrasts = {"response_all_minus_first": paired_difference(
            groups["R_resp:all_calls"], groups["R_resp:first_call"])}
        for left, right in [("eq_B", "eq_C"), ("eq_B", "eq_G"), ("eq_C", "eq_G")]:
            contrasts[left + "_minus_" + right] = paired_difference(
                groups[left + ":all_calls"], groups[right + ":all_calls"])
        families[family] = {
            "support": {**expected, "noise_labels": [0, 1], "init_indices": [0],
                        "queries": [0, 2, 5, 10], "conditions": 10,
                        "translation_endpoint_rows": len(family_rows)},
            "frozen_S_f": scales[family], "condition_means": summaries,
            "paired_contrasts": contrasts,
            "endpoint_checks": {"all_conditions_same_anchors_and_weights": True,
                "all_endpoints_are_unexecuted_plans": True,
                "strict_equal_cumulative_dose_claimed": False,
                "native_reference_null_dose_records_preserved": sum(
                    r["dose"] is None for r in family_rows),
                "max_weight_error_from_task_variant_noise_query_rule": max_weight_error,
                "max_stored_normalization_error": max_scale_error,
                "direction_status_counts": dict(Counter(r["direction_status"] for r in family_rows)),
                "exact_zero_clean_direction_count": sum(r["exact_zero_clean_direction"] for r in family_rows)},
            "local_common_target_norm": {"task_mean_cells": len(cells),
                "grid_complete": True, "flow_calls": list(range(10)),
                "position_groups": ["positions_0_4", "positions_5_9"],
                "norm_direction_counts": dict(ordering),
                "actual_BF16_write_norm_difference_signs": dict(write_signs),
                "metric_status_counts": {k: dict(v) for k, v in metric_status.items()},
                "cell_role": "Existing task/query/call/position-group means; repeated observations",
                "velocity_role": "Mean norm of local signed 32-D normalized flow-velocity change "
                    "relative to N_T at the same saved native x_j,t_j"},
        }

    output = {
        "schema": "cfr-dev-controls-descriptive-replay-v1", "status": "VERIFIED_REPLAY",
        "input_identity": {"provenance_sha256": sha(provenance_bytes),
            "subsets": {k: {q: v[q] for q in ("path", "rows", "gzip_sha256", "uncompressed_sha256")}
                        for k, v in provenance["subsets"].items()},
            "frozen_scales_sha256": sha(scale_bytes)},
        "families": families,
        "all_local_task_cell_direction_counts": dict(all_local_counts),
        "aggregation": "Use original endpoint_available_query_weight. It equals "
            "1/(family task count * variants in task * 2 noise labels * 4 saved queries). "
            "Paired contrasts join the same task/variant/noise/init/query anchors.",
        "scientific_scope": {
            "development_only": True, "endpoint_is_plan_not_issued_or_success": True,
            "saved_queries": [0, 2, 5, 10],
            "natural_and_common_target_are_different_edit_rules": True,
            "common_target_rule": "Per action position s=min(||p_B||,||p_C||,||p_G||); "
                "candidate edit is -s*p_k/||p_k|| with the original zero/no-op rules",
            "actual_BF16_writes_need_not_equal": True,
            "dose_fields_are_original_per_plan_scalars_not_episode_exposure": True,
            "full_generation_states_can_diverge_after_editing": True,
            "local_task_cells_are_not_independent_sample_size": True,
            "clean_instantaneous_velocity_not_supplied": True,
            "local_full32_vectors_not_included": True,
            "native_attention_PW_fidelity_not_measured": True,
            "new_CI_bootstrap_fits_or_inference": False,
            "original_interval_table_not_included_or_recomputed": True,
        },
    }
    (HERE / "dev_controls_summary.json").write_text(json.dumps(output, ensure_ascii=False,
        indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    table_lines = [r"% Author-side draft. Requires booktabs. Descriptive DEV means only.",
        r"\begin{table}[t]", r"\centering\small", r"\setlength{\tabcolsep}{3.2pt}",
        r"\begin{tabular}{lrrrrr}", r"\toprule",
        r"& \multicolumn{2}{c}{Response schedule} & \multicolumn{3}{c}{Common target norm, all calls}\\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-6}",
        r"Family & First & All & Response & Clean & Random\\", r"\midrule"]
    for family, short in [("Camera Viewpoints", "Camera"), ("Sensor Noise", "Sensor")]:
        lookup = {(r["arm"], r["schedule_role"]): r["mean_normalized_translation_error_improvement"]
                  for r in families[family]["condition_means"]}
        values = [lookup[k] for k in [("R_resp", "first_call"), ("R_resp", "all_calls"),
                                     ("eq_B", "all_calls"), ("eq_C", "all_calls"), ("eq_G", "all_calls")]]
        table_lines.append(short + " & " + " & ".join(f"{v:.5f}" for v in values) + r"\\")
    table_lines.extend([r"\bottomrule", r"\end{tabular}",
        r"\caption{\textbf{Development controls for generation schedule and write rule.} "
        r"Task-equal mean translation-error improvement over the first five plan positions, "
        r"normalized by the frozen family scale $S_f$. Camera and Sensor use 5/9 tasks "
        r"and 8/17 variants, each with two noise labels and saved queries $q=0,2,5,10$. "
        r"The common target norm is set at each local state; BF16 writes are measured "
        r"separately and each complete generation follows its resulting states.}",
        r"\label{tab:dev_schedule_target_norm}", r"\end{table}"])
    (HERE / "dev_controls_table.tex").write_text("\n".join(table_lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": "VERIFIED_REPLAY", "translation_endpoints": len(endpoints),
                      "local_task_cells": len(local), "counts": dict(all_local_counts)}, sort_keys=True))


if __name__ == "__main__":
    main()
