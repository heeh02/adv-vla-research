"""All observed five-step pairs from archived physical trajectories; no native imports."""
import math
import numpy as np
import pinned_campaign as P
from physical_codec import need, identity

PAIRS = (("S-N", "S", "N"), ("R-N", "R", "N"), ("S-R", "S", "R"))


def validate(docs, inputs, source_folder):
    refs = {r["logical_path"]: r for r in inputs["input_objects"]}
    for prefix in ("old", "new"):
        archive, runtime = (docs[prefix+"/control/"+n] for n in ("ARCHIVE-RECEIPT.json", "RUNTIME-RESULT.json"))
        result = docs[prefix+"/science/PHYSICAL-RESULT.json"]
        need(archive["archive_complete"] is True and archive["science_archive"]["errors"] == [] and
             archive["run_id"] == runtime["run_id"], "actual_observed_archive_complete")
        control = refs[prefix+"/control/RUNTIME-RESULT.json"]
        need(all(archive["control_receipt"][k] == control[k] for k in ("uri", "bytes", "sha256")),
             "runtime_archive_SHA")
        files = {r["name"]: r for r in archive["science_archive"]["objects"]}
        need(len(files) == archive["science_archive"]["observed_files"], "unique_observed_archive_members")
        for logical, ref in refs.items():
            if logical.startswith(prefix+"/science/"):
                row = files[logical.split('/')[-1]]
                need(all(ref[k] == row[k] for k in ("uri", "bytes", "sha256")), "scientific_archive_SHA")
        need(result["source_manifest"] == runtime["source_abi"]["source_manifest"] and
             result["resources"] == {"CPU": 8, "GPU": 0, "memory_GiB": 32} and result["new_model_calls"] == 0,
             "source_bound_actual_CPU_physics")
    old, new, witness = docs["old/science/PHYSICAL-CAMPAIGN.json.gz"], docs["new/science/PHYSICAL-CAMPAIGN.json.gz"], docs["witness.json"]
    old_n, new_n = docs["old/science/NATIVE-SUMMARY.json"], docs["new/science/NATIVE-SUMMARY.json"]
    need(old["synthetic"] is new["synthetic"] is False and old["completed_all_roles"] is False and
         new["completed_remaining_roles"] is True and new["source_and_N_case_reexecuted"] is False,
         "original_incomplete_run_and_completed_remaining_roles_distinct")
    need(old["total_explicit_env_steps"] == new["previous_environment_steps"] == 140 and
         new["new_environment_steps"] == 210 and new["cumulative_environment_steps"] == 350 and
         old["model_calls"] == new["model_calls"] == old["GPU"] == new["GPU"] == 0,
         "original_and_new_denominators")
    need([r["role"] for r in new["roles"]] == ["N_repeat", "S", "R"] and
         all(r["status"] == "OBSERVED_FIVE_COMMANDS" and r["env_steps"] == 70 for r in new["roles"]),
         "complete_actual_missing_three_roles")
    for full, summary in ((old, old_n), (new, new_n)):
        for row, small in zip(full["roles"], summary["roles"]):
            need({k: v for k, v in row.items() if k not in ("trajectory", "queries", "start_critical_witness")} == small,
                 "full_record_matches_archived_summary")
    need(old["roles"][0]["status"] == "QUALIFIED_SOURCE_PATH" and
         old["roles"][1]["status"] == "FAILED_NO_RETRY" and
         old["roles"][1]["close_failure"] == {"code": "one_window_finish", "type": "PairingError"},
         "retained_source_qualification_and_N_close_failure")
    by = {r["role"]: r for r in [old["roles"][1], *new["roles"]]}
    for role, row in by.items():
        need(len(row["trajectory"]) == row["observed_continuation_steps"] == 5 and
             P.trajectory_identity(row["trajectory"]) == row["trajectory_identity"], "full_trajectory_hash_recomputed")
        need(row["start_identity"] == witness["recorded_complete_q11_start_identity"], "same_q11_recorded_start")
        for step in row["trajectory"]:
            need(step["done"] is False and step["official_success"] is False, "complete_observed_nonterminal_window")
    need(by["N_repeat"]["trajectory_identity"] == by["N"]["trajectory_identity"] == witness["N_trajectory_identity_recomputed"],
         "N_repeat_full_physical_trajectory_exact")
    need(P.digest(by["N"]["start_critical_witness"]) == witness["critical_state_identity_recomputed"] and
         all(P.digest(r["start_critical_witness"]) == witness["critical_state_identity_recomputed"] for r in new["roles"]),
         "all_critical_state_witnesses_recomputed")
    from pathlib import Path
    pinned = identity((Path(source_folder)/"pinned_campaign.py").read_bytes())
    row = next(r for r in docs["old/control/RUNTIME-RESULT.json"]["source_abi"]["files"] if r["path"] == "campaign_core.py")
    need(pinned == {k: row[k] for k in ("bytes", "sha256")}, "same_trajectory_hash_source_domain")
    return by


def orientation(a, b):
    a, b = np.asarray(a), np.asarray(b)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    need(abs(na-1) < 1e-5 and abs(nb-1) < 1e-5, "unit_native_quaternions")
    a, b = a/na, b/nb
    if np.dot(a, b) < 0:
        b = -b
    return float(4*np.arctan2(np.linalg.norm(b-a), np.linalg.norm(b+a)))


def compute(by):
    records = []
    for name, candidate, reference in PAIRS:
        for slot, (left, right) in enumerate(zip(by[reference]["trajectory"], by[candidate]["trajectory"])):
            a, b = left["physical_outputs"], right["physical_outputs"]
            oa, ob = a["returned_nonimage_observations"], b["returned_nonimage_observations"]
            delta = ob["robot0_eef_pos"]-oa["robot0_eef_pos"]
            fingers = ob["robot0_gripper_qpos"]-oa["robot0_gripper_qpos"]
            need(delta.shape == (3,) and fingers.shape == (2,) and np.isfinite(delta).all() and np.isfinite(fingers).all(),
                 "finite_pose_and_two_finger_dimensions")
            t0 = float(left["before"]["sim_time"])
            t1 = float(left["after"]["sim_time"])
            need(t0 == float(right["before"]["sim_time"]) and t1 == float(right["after"]["sim_time"]) and
                 math.isclose(t1-t0, .05, abs_tol=1e-10), "same_native_clock_20Hz")
            records.append({"contrast": name, "candidate": candidate, "reference": reference,
                "issued_slot": slot, "elapsed_seconds": (slot+1)*.05,
                "eef_delta_xyz_m": delta.tolist(), "eef_position_distance_m": float(np.linalg.norm(delta)),
                "eef_orientation_geodesic_rad": orientation(oa["robot0_eef_quat"], ob["robot0_eef_quat"]),
                "gripper_qpos_delta_original_units": fingers.tolist(),
                "gripper_actuator_target_equal": bool(np.array_equal(a["gripper_actuator_ctrl"], b["gripper_actuator_ctrl"])),
                "reference_contact_count": a["contact_count"], "candidate_contact_count": b["contact_count"],
                "reference_success": left["official_success"], "candidate_success": right["official_success"],
                "reference_done": left["done"], "candidate_done": right["done"]})
    need(len(records) == 15, "all3pairs_all5times")
    return records


def compare_original(records, original):
    existing = {(r["contrast"], r["issued_slot"]): r for r in original["comparisons"]}
    need(len(existing) == 10, "all10_original_comparisons")
    for row in records:
        if row["contrast"] == "S-R":
            continue
        saved = existing[(row["contrast"], row["issued_slot"])]
        for key in ("eef_position_distance_m", "eef_orientation_geodesic_rad", "gripper_actuator_target_equal"):
            need(row[key] == saved[key], "recomputed_original_scalar_exact")
        need(np.array_equal(row["eef_delta_xyz_m"], saved["eef_position_delta_m"]) and
             np.array_equal(row["gripper_qpos_delta_original_units"], saved["gripper_qpos_delta"]),
             "recomputed_original_signed_vectors_exact")
        need(row["reference_contact_count"] == saved["N_contact_count"] and
             row["candidate_contact_count"] == saved["arm_contact_count"], "recomputed_original_contacts_exact")
