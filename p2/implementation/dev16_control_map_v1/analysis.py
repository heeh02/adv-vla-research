"""Source-defined command-interface algebra; no controller or simulator import."""
import math
import numpy as np
from codec import need
from binding import DIMS

CONTRASTS = (("S-N", "S", "N"), ("R-N", "R", "N"), ("S-R", "S", "R"))
SCALE = np.array([0.05, 0.05, 0.05, 0.5, 0.5, 0.5], dtype=np.float64)


def bits(x):
    return np.asarray(x, dtype="<f8").tobytes().hex()


def mapping(raw):
    raw = np.asarray(raw)
    need(raw.dtype == np.dtype("float64") and raw.shape == (10, 7) and
         np.isfinite(raw).all(), "finite_native_adapter7_ABI")
    clipped = np.clip(raw[:, :6], -1.0, 1.0)
    # Preserve source's scale_action subtract/multiply/add order, including ±0.
    mapped = (clipped - np.zeros(6)) * SCALE + np.zeros(6)
    signs = np.sign(raw[:, 6]).astype(np.int8)
    return clipped, mapped, signs


def rms(values):
    return math.hypot(*values) / math.sqrt(len(values))


def clip_state(x):
    return "below_-1" if x < -1.0 else "above_+1" if x > 1.0 else "inside_including_boundary"


def scalar_pair(reference, candidate, axis):
    r, c = np.float64(reference), np.float64(candidate)
    need(np.isfinite(r) and np.isfinite(c), "finite_command_scalar")
    with np.errstate(over="raise", invalid="raise"):
        raw_delta = float(c - r)
    raw_equal = bool(c == r)
    if axis < 6:
        rc, cc = np.clip(np.array([r, c]), -1.0, 1.0)
        rm, cm = (np.array([rc, cc]) - np.float64(0)) * SCALE[axis] + np.float64(0)
        out = {
            "mapping": "arm_clip_then_goal_delta_scale",
            "reference_clipped": float(rc), "candidate_clipped": float(cc),
            "clipped_delta": float(cc - rc), "clipped_numeric_equal": bool(cc == rc),
            "clipped_bitwise_equal": bits(cc) == bits(rc),
            "reference_clip_state": clip_state(r), "candidate_clip_state": clip_state(c),
            "reference_at_input_boundary": bool(abs(r) == 1),
            "candidate_at_input_boundary": bool(abs(c) == 1),
            "reference_clipped_signbit": bool(np.signbit(rc)),
            "candidate_clipped_signbit": bool(np.signbit(cc)),
            "erased_at_clip": not raw_equal and bool(cc == rc),
            "erased_only_at_scale_rounding": bool(cc != rc) and bool(cm == rm),
            "arm_scale": float(SCALE[axis]),
            "mapped_units": "controller_goal_translation_increment" if axis < 3
                            else "controller_goal_axis_angle_increment",
        }
    else:
        rm, cm = int(np.sign(r)), int(np.sign(c))
        out = {
            "mapping": "gripper_sign_symbol_not_stateful_target",
            "reference_gripper_sign": rm, "candidate_gripper_sign": cm,
            "sign_transition": f"{rm}_to_{cm}",
            "opposite_nonzero_signs": rm * cm == -1,
            "one_zero_other_nonzero": (rm == 0) != (cm == 0),
            "both_zero": rm == cm == 0,
            "same_nonzero_sign": rm == cm and rm != 0,
            "raw_magnitude_difference_erased_by_sign": not raw_equal and rm == cm,
            "mapped_units": "gripper_sign_symbol",
        }
    out.update({
        "reference_raw": float(r), "candidate_raw": float(c), "raw_delta": raw_delta,
        "raw_numeric_equal": raw_equal, "raw_bitwise_equal": bits(r) == bits(c),
        "reference_raw_float64_le_hex": bits(r), "candidate_raw_float64_le_hex": bits(c),
        "reference_raw_zero": bool(r == 0), "candidate_raw_zero": bool(c == 0),
        "reference_raw_signbit": bool(np.signbit(r)), "candidate_raw_signbit": bool(np.signbit(c)),
        "reference_mapped": float(rm), "candidate_mapped": float(cm),
        "mapped_delta": float(cm - rm), "mapped_numeric_equal": bool(cm == rm),
        "mapped_bitwise_equal": bits(rm) == bits(cm),
        "raw_difference_erased_by_mapping": not raw_equal and bool(cm == rm),
        "mapping_difference_survives": bool(cm != rm),
    })
    need(not raw_equal or out["mapped_numeric_equal"], "deterministic_mapping_consistency")
    return out


def snapshot(key, commands, *, semantic_claim_eligible):
    for value in commands.values():
        mapping(value)
    need(set(commands) == {"N", "S", "R"}, "exact_native3_arms")
    points, windows, window_cases = [], [], []
    for contrast, candidate, reference in CONTRASTS:
        rows = []
        for pos in range(10):
            for axis, dimension in enumerate(DIMS):
                row = {"root_id": key[0], "query_index": key[1], "contrast": contrast,
                       "reference_arm": reference, "candidate_arm": candidate,
                       "position": pos, "axis": axis, "dimension": dimension,
                       "static_controller_binding_eligible": semantic_claim_eligible,
                       **scalar_pair(commands[reference][pos, axis],
                                     commands[candidate][pos, axis], axis)}
                rows.append(row)
                points.append(row)
        for window, stop in (("first5", 5), ("full10", 10)):
            selected = [r for r in rows if r["position"] < stop]
            arm = [r for r in selected if r["axis"] < 6]
            grip = [r for r in selected if r["axis"] == 6]
            case = {"root_id": key[0], "query_index": key[1], "contrast": contrast,
                    "window": window, "positions": stop,
                    "static_controller_binding_eligible": semantic_claim_eligible,
                    "arm6_all_mapped_numeric_equal": all(r["mapped_numeric_equal"] for r in arm),
                    "arm6_all_mapped_bitwise_equal": all(r["mapped_bitwise_equal"] for r in arm),
                    "gripper_all_signs_equal": all(r["mapped_numeric_equal"] for r in grip),
                    "all7_raw_numeric_equal": all(r["raw_numeric_equal"] for r in selected),
                    "all7_raw_bitwise_equal": all(r["raw_bitwise_equal"] for r in selected)}
            case["all7_frontend_numeric_equivalent"] = (
                case["arm6_all_mapped_numeric_equal"] and case["gripper_all_signs_equal"])
            case["physical_equivalence_measured"] = False
            window_cases.append(case)
            for axis, dimension in enumerate(DIMS):
                part = [r for r in selected if r["axis"] == axis]
                raw = [r["raw_delta"] for r in part]
                mapped = [r["mapped_delta"] for r in part]
                wr = {"root_id": key[0], "query_index": key[1], "contrast": contrast,
                      "window": window, "positions": stop, "dimension": dimension, "axis": axis,
                      "static_controller_binding_eligible": semantic_claim_eligible,
                      "raw_numeric_differences": sum(not r["raw_numeric_equal"] for r in part),
                      "raw_payload_differences": sum(not r["raw_bitwise_equal"] for r in part),
                      "raw_negative_deltas": sum(x < 0 for x in raw),
                      "raw_positive_deltas": sum(x > 0 for x in raw),
                      "raw_zero_deltas": sum(x == 0 for x in raw),
                      "raw_RMS_difference": rms(raw), "raw_max_abs_difference": max(map(abs, raw)),
                      "mapped_numeric_differences": sum(not r["mapped_numeric_equal"] for r in part),
                      "mapped_payload_differences": sum(not r["mapped_bitwise_equal"] for r in part),
                      "mapped_RMS_difference": rms(mapped),
                      "mapped_max_abs_difference": max(map(abs, mapped)),
                      "raw_differences_erased": sum(r["raw_difference_erased_by_mapping"] for r in part),
                      "mapped_units": part[0]["mapped_units"]}
                if axis < 6:
                    clipped = [r["clipped_delta"] for r in part]
                    wr.update(clipped_numeric_differences=sum(not r["clipped_numeric_equal"] for r in part),
                              clipped_payload_differences=sum(not r["clipped_bitwise_equal"] for r in part),
                              clipped_RMS_difference=rms(clipped),
                              erased_at_clip=sum(r["erased_at_clip"] for r in part),
                              erased_only_at_scale_rounding=sum(r["erased_only_at_scale_rounding"] for r in part),
                              reference_clipped_positions=sum(r["reference_clip_state"] != "inside_including_boundary" for r in part),
                              candidate_clipped_positions=sum(r["candidate_clip_state"] != "inside_including_boundary" for r in part))
                else:
                    wr.update(opposite_nonzero_signs=sum(r["opposite_nonzero_signs"] for r in part),
                              one_zero_other_nonzero=sum(r["one_zero_other_nonzero"] for r in part),
                              both_zero=sum(r["both_zero"] for r in part),
                              same_nonzero_sign=sum(r["same_nonzero_sign"] for r in part),
                              sign_transition_counts={f"{i}_to_{j}": sum(r["sign_transition"] == f"{i}_to_{j}" for r in part)
                                                      for i in (-1, 0, 1) for j in (-1, 0, 1)})
                windows.append(wr)
    return points, windows, window_cases


def summarize(points, windows, cases):
    groups = []
    for contrast, _, _ in CONTRASTS:
        for window in ("first5", "full10"):
            for axis, dimension in enumerate(DIMS):
                rows = [r for r in windows if r["contrast"] == contrast and r["window"] == window
                        and r["axis"] == axis]
                need(len(rows) == 8, "all8_window_axis_support")
                fields = ["raw_numeric_differences", "mapped_numeric_differences",
                          "raw_differences_erased", "raw_negative_deltas", "raw_positive_deltas",
                          "raw_zero_deltas"]
                fields += (["erased_at_clip", "erased_only_at_scale_rounding"]
                           if axis < 6 else ["opposite_nonzero_signs", "one_zero_other_nonzero",
                                             "both_zero", "same_nonzero_sign"])
                groups.append({"contrast": contrast, "window": window, "dimension": dimension,
                               "snapshots": 8, "positions_per_snapshot": rows[0]["positions"],
                               "dependent_position_comparisons": sum(r["positions"] for r in rows),
                               "unit_of_independent_support": "4 deterministic initializations in2tasks,notpositioncells",
                               **{k: sum(r[k] for r in rows) for k in fields}})
    return {"schema": "paper2-dev16-control-map-summary-v1", "groups": groups,
            "all_case_window_equivalence": cases,
            "claim": "Source-formula command-interface diagnostic, not executed torque or physical motion.",
            "same_gripper_sign_scope": "Equal target sequence conditional on equalstartingcurrent_action andupdatecounts; finger/object motion may differ via arm/contact coupling.",
            "specific_gripper_target_sequence_computed": False,
            "zero_and_negative_cases_retained": True, "new_fit": False,
            "point_comparisons": len(points), "axis_window_comparisons": len(windows)}
