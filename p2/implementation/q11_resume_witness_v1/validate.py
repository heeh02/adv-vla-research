"""Readout eligibility only; never transforms a failed attempt into a completed one."""
import hashlib
from pathlib import Path
import numpy as np
import pinned_campaign as P
from physical_codec import need, array_identity, identity

COUNTS = {"retained_roles": 2, "continuation_steps": 10, "query_witnesses": 25,
          "source_target_transitions": 5, "missing_roles": 3}
LIMITS = {"construct": 10, "reset": 10, "set_init_state": 10, "mj_step": 8750,
          "mj_step1": 20, "mj_step2": 0, "mj_forward": 30000}


def provenance(docs, inputs, source_folder):
    refs = {r["logical_path"]: r for r in inputs["input_objects"]}
    archive, runtime = docs["control/ARCHIVE-RECEIPT.json"], docs["control/RUNTIME-RESULT.json"]
    result = docs["science/PHYSICAL-RESULT.json"]
    need(archive["run_id"] == runtime["run_id"] == inputs["source_run_id"] and
         archive["archive_complete"] is True and archive["science_archive"]["errors"] == [] and
         archive["science_archive"]["complete_for_observed_files"] is True,
         "actual_complete_observed_archive")
    control = refs["control/RUNTIME-RESULT.json"]
    need(all(archive["control_receipt"][k] == control[k] for k in ("uri", "bytes", "sha256")),
         "source_runtime_independent_archive_SHA")
    archived = {r["name"]: r for r in archive["science_archive"]["objects"]}
    need(len(archived) == archive["science_archive"]["observed_files"] == 9,
         "exact_source_observed_archive_roster")
    for logical, ref in refs.items():
        if logical.startswith("science/"):
            actual = archived[logical.removeprefix("science/")]
            need(all(actual[k] == ref[k] for k in ("uri", "bytes", "sha256")),
                 "source_science_independent_archive_SHA")
    need(result["synthetic"] is False and result["completed_all_roles"] is False and
         result["new_env_steps"] == runtime["new_env_steps"] == 140 and
         result["new_model_calls"] == runtime["new_model_calls"] == 0 and
         result["source_manifest"] == runtime["source_abi"]["source_manifest"] == inputs["source_manifest"] and
         result["inputs_identity"] == inputs["source_input_manifest"] and
         runtime["source_abi"]["source_zip"] == inputs["source_zip"], "original_native_run_and_source")
    need(runtime["resources"] == {"CPU": 8, "GPU": 0, "memory_GiB": 32} and
         runtime["input_body_bytes"] == 796390625 and len(runtime["verified_inputs"]) == 243,
         "original_input_and_resource_scope")
    files = {r["path"]: r for r in runtime["source_abi"]["files"]}
    for local, name in (("pinned_campaign.py", "campaign_core.py"),
                        ("pinned_writer.py", "physical_writer.py")):
        need(identity((Path(source_folder)/local).read_bytes()) ==
             {k: files[name][k] for k in ("bytes", "sha256")}, "original_hash_domain_and_writer_pins")


def inspect_campaign(campaign, native, qualified):
    need(campaign["schema"] == "paper2-q11-cached-command-campaign-v1" and
         campaign["synthetic"] is False and campaign["completed_all_roles"] is False and
         campaign["total_explicit_env_steps"] == native["total_explicit_env_steps"] == 140,
         "precise_incomplete_native_attempt")
    need([r["role"] for r in campaign["roles"]] == list(P.ROLES), "original_five_role_roster")
    need(native["failure"] is None and not native["completed_all_roles"] and
         campaign["GPU"] == native["GPU"] == campaign["model_calls"] == campaign["model_attempts"] ==
         native["new_model_calls"] == native["model_attempts"] == native["model_import_attempts"] == 0,
         "no_model_GPU_or_outer_failure")
    for row, summary in zip(campaign["roles"], native["roles"]):
        need({k: v for k, v in row.items() if k not in
              ("trajectory", "queries", "start_critical_witness")} == summary,
             "full_campaign_matches_native_summary")
    source, none = campaign["roles"][:2]
    need(source["status"] == "QUALIFIED_SOURCE_PATH" and
         "failure" not in source and "close_failure" not in source,
         "retained_source_qualification_completed")
    need(none["status"] == "FAILED_NO_RETRY" and "failure" not in none and
         none["close_failure"] == {"code": "one_window_finish", "type": "PairingError"},
         "only_exact_post_observation_close_failure")
    for row in campaign["roles"][2:]:
        need(row["status"] == "NOT_ATTEMPTED_PREVIOUS_GATE_FAILED" and
             row["env_steps"] == row["model_calls"] == row["model_attempts"] == 0,
             "missing_roles_remain_unattempted")
    parent = qualified["parent"]
    need(parent["root_id"] == P.ROOT and parent["all_source_continuity_exact"] is True and
         parent["q12_commands_used"] == 0 and len(parent["steps"]) == 60,
         "old_parent_input_qualification")
    join = qualified["join"]
    need(join["native_final72_qualification_bound"] is True and
         join["actual_saved_input_body_decoded"] is True and
         join["cached_commands"]["N"]["entry"]["receiver"] == "T" and
         join["cached_commands"]["N"]["entry"]["role"] == "baseline_none_1",
         "cached_receiver_T_N_not_source_parent_or_N_repeat")
    expected_groups = {r["query_index"]: r["source_groups_before"]
                       for r in parent["protected_queries"]}
    anchors = {r["query_index"]: r for r in parent["anchors"]}
    mixed, query_count, unmixed_count = [], 0, 0
    for row, queries in ((source, list(range(13))), (none, list(range(12)))):
        need(row["env_steps"] == 70 and row["observed_continuation_steps"] == 5 and
             len(row["trajectory"]) == 5, "complete_retained_five_command_observation")
        need([q["query"] for q in row["queries"]] == queries, "exact_retained_query_support")
        for q in row["queries"]:
            statuses = q["source_group_status"]
            need(set(statuses) == set(expected_groups[q["query"]]) and
                 all(v in ("EXACT", "DIFFERENT") for v in statuses.values()) and
                 all(v == "EXACT" for k, v in statuses.items() if k not in P.OLD_MIXED),
                 "old_unmixed_groups_remain_exact")
            need(q["old_all_declared_groups_exact"] == all(v == "EXACT" for v in statuses.values()) and
                 q["old_full_replay_qualification_relabelled"] is False, "preserved_old_full_replay_boundary")
            query_count += 1
            unmixed_count += len(set(statuses)-P.OLD_MIXED)
            mixed.append({"role": row["role"], "query": q["query"],
                          "old_mixed_groups": {k: statuses[k] for k in sorted(P.OLD_MIXED)}})
            if q["query"] in anchors:
                observed = q["anchor"]
                need(observed["physics_exact"] is True and
                     set(observed["observation_typed_exact"]) == set(anchors[q["query"]]["observations"]) and
                     all(v is True for k, v in observed["observation_typed_exact"].items()
                         if k not in P.IMAGE_KEYS), "recorded_anchor_nonimage_qualification")
        need(P.trajectory_identity(row["trajectory"]) == row["trajectory_identity"],
             "independently_recomputed_complete_trajectory_identity")
        previous = None
        for step in row["trajectory"]:
            need(step["done"] is False and step["official_success"] is False,
                 "exact_retained_nonterminal_observations")
            for phase in ("before", "after"):
                need(step[phase]["flat_physics"].shape == (123,) and
                     step[phase]["flat_physics"].dtype == np.float64, "native_physics_ABI")
            if previous is not None:
                need(all(P.equal(previous[k], step["before"][k]) for k in P.STEP_FIELDS),
                     "retained_step_continuity")
            previous = step["after"]
            obs = step["physical_outputs"]["returned_nonimage_observations"]
            need(P.equal(obs["robot0_eef_pos"], step["after"]["returned_eef_pos"]) and
                 P.equal(obs["robot0_eef_quat"], step["after"]["returned_eef_quat"]),
                 "actual_returned_pose_not_new_refresh")
    old_steps = [r for r in parent["steps"] if r["query_index"] == 11]
    need([r["slot"] for r in old_steps] == list(range(5)), "five_original_q11_transitions")
    for step, old in zip(source["trajectory"], old_steps):
        for phase in ("before", "after"):
            need(all(array_identity(step[phase][k]) == old[phase][k] for k in P.STEP_FIELDS),
                 "independent_original_target_transition_match")
    need(all(P.equal(source["trajectory"][0]["before"][k], none["trajectory"][0]["before"][k])
             for k in P.STEP_FIELDS), "same_actual_first_before_physics_time_pose")
    need(source["start_identity"] == none["start_identity"] and
         P.digest(source["start_critical_witness"]) == P.digest(none["start_critical_witness"]),
         "same_recorded_complete_start_and_independent_critical_state")
    budget = native["native_budget"]
    expected = {"construct": 4, "reset": 4, "set_init_state": 4, "mj_step": 3500,
                "mj_step1": 8, "mj_step2": 0, "mj_forward": 7038}
    need(budget["limits"] == LIMITS and budget["aborted"] is None and
         budget["entered_calls"] == budget["attempted_calls"] == expected and
         budget["reset_attempts"] == budget["reset_successes"] == 4,
         "actual_native_call_budget_kept")
    need(query_count == 25, "all25_query_witnesses_checked")
    witness = {"schema": "paper2-q11-retained-observation-resume-witness-v1",
        "root_id": P.ROOT, "query_index": 11, "retained_observation_input_eligible": True,
        "original_N_attempt_status": none["status"], "original_N_close_failure": none["close_failure"],
        "original_attempt_relabelled_complete": False,
        "recorded_complete_q11_start_identity": source["start_identity"],
        "complete_start_hash_recomputed_from_full_raw_declared_state": False,
        "critical_state_identity_recomputed": P.digest(source["start_critical_witness"]),
        "first_before_physics_time_pose_identity": P.digest(none["trajectory"][0]["before"]),
        "source_trajectory_identity_recomputed": source["trajectory_identity"],
        "N_trajectory_identity_recomputed": none["trajectory_identity"],
        "observed_env_steps": 140, "observed_native_calls": expected,
        "remaining_native_call_ceilings": {k: LIMITS[k]-v for k, v in expected.items()},
        "remaining_roles": ["N_repeat", "S", "R"], "maximum_new_environment_steps": 210,
        "source_and_N_reexecution_allowed": False,
        "future_mandatory_checks": ["same frozen original inputs/runtime", "new job exact q11 start fingerprint",
                                    "N_repeat full trajectory exactly equals retained N before S/R"],
        "cross_job_parity_established": False, "N_repeat_passed": False,
        "pulse_motion_contrast_established": False, "new_environment_execution_authorized": False,
        "source_parent_is_not_cached_receiver_T_N": True}
    checks = {"counts": COUNTS, "old_unmixed_group_comparisons_exact": unmixed_count,
              "all_old_mixed_group_statuses": mixed, "recomputed_trajectory_identities": 2,
              "source_target_transitions_independently_rechecked": 5,
              "source_prefix55_checks": "attested by verified frozen native execution, not raw-step redecoding here",
              "original_source_full_replay_status_unchanged": True}
    return witness, checks
