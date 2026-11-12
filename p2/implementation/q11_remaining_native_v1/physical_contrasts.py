"""Descriptive within-case contrasts only; no pooling, fitted thresholds or SR."""
import numpy as np
from campaign_core import need


def summarize(campaign, retained):
    if not campaign["completed_remaining_roles"]:
        return {"status": "NO_SCIENTIFIC_CONTRAST_GATE_FAILED", "comparisons": [],
                "no_unobserved_tail_filled": True}
    by = {r["role"]: r for r in campaign["roles"]}
    old_N=retained["roles"][1]
    need(old_N["role"]=="N" and old_N["status"]=="FAILED_NO_RETRY" and
         old_N["close_failure"]=={"code":"one_window_finish","type":"PairingError"},
         "retained_N_observation_and_lifecycle_kept")
    need(by["N_repeat"]["trajectory_identity"]==old_N["trajectory_identity"],"actual_N_repeat_gate")
    rows = []
    for name in ("S", "R"):
        baseline, changed = old_N["trajectory"], by[name]["trajectory"]
        for slot in range(min(len(baseline), len(changed))):
            left = baseline[slot]["physical_outputs"]
            right = changed[slot]["physical_outputs"]
            a, b = left["returned_nonimage_observations"], right["returned_nonimage_observations"]
            delta = np.asarray(b["robot0_eef_pos"])-np.asarray(a["robot0_eef_pos"])
            qa, qb = np.asarray(a["robot0_eef_quat"]), np.asarray(b["robot0_eef_quat"])
            na, nb = np.linalg.norm(qa), np.linalg.norm(qb)
            need(np.isfinite(delta).all() and abs(na-1) < 1e-5 and abs(nb-1) < 1e-5,
                 "finite_EEF_and_unit_quaternions")
            # Align antipodal quaternions, then use a stable chord formula.
            qa, qb = qa/na, qb/nb
            if np.dot(qa, qb) < 0:
                qb = -qb
            angle = 4*np.arctan2(np.linalg.norm(qb-qa), np.linalg.norm(qb+qa))
            rows.append({"contrast": name+"-N", "issued_slot": slot,
                         "eef_position_delta_m": delta,
                         "eef_position_distance_m": float(np.linalg.norm(delta)),
                         "eef_orientation_geodesic_rad": float(angle),
                         "gripper_qpos_delta": np.asarray(b["robot0_gripper_qpos"])-
                                              np.asarray(a["robot0_gripper_qpos"]),
                         "gripper_actuator_target_equal": bool(np.array_equal(
                             left["gripper_actuator_ctrl"], right["gripper_actuator_ctrl"])),
                         "N_contact_count": left["contact_count"],
                         "arm_contact_count": right["contact_count"],
                         "N_done": baseline[slot]["done"], "arm_done": changed[slot]["done"],
                         "N_success": baseline[slot]["official_success"],
                         "arm_success": changed[slot]["official_success"]})
    return {"status": "DESCRIPTIVE_ONE_CASE_OBSERVED_PREFIX_ONLY", "comparisons": rows,
            "source_root": "SDRU-CDEV-T00-R00", "source_query": 11,
            "matched_start_and_exact_N_repeat_required": True,
            "retained_N_original_status":"FAILED_NO_RETRY",
            "retained_N_original_close_failure":old_N["close_failure"],
            "old_run_status_modified":False,
            "N_baseline_source":"previous_run_observed_trajectory_exactly_repeated_in_this_run",
            "no_unobserved_tail_filled": True, "success_rate_or_task_transfer": False,
            "dose_mediation_isolated": False, "statistical_significance_claimed": False}
