"""Only N-repeat/S/R continuation. No native imports or executable launcher.

The original failed run remains unchanged. Its checked observation witnesses
are references for a new repeatability gate, never a declaration that it passed.
"""
import copy
import hashlib
import json
import math
import time
import campaign_core as P
from native_budget import Budget
from failure_guards import ResetBudget

ROLES = ("N_repeat", "S", "R")
WITNESS = {"bytes": 2439, "sha256": "02fc7768d3cc5b8601341bf5a632f48d796170c3fae5d6ec5e737dc13b08efb2"}
PREVIOUS_CALLS = {"construct": 4, "reset": 4, "set_init_state": 4, "mj_step": 3500,
                  "mj_step1": 8, "mj_step2": 0, "mj_forward": 7038}


def load_witness(raw):
    P.need(type(raw) is bytes and {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()} == WITNESS,
           "exact_actual_cloud_resume_witness")
    return json.loads(raw)


def witness_gate(value):
    P.need(value["schema"] == "paper2-q11-retained-observation-resume-witness-v1" and
           value["root_id"] == P.ROOT and value["query_index"] == 11 and
           value["retained_observation_input_eligible"] is True and
           value["original_N_attempt_status"] == "FAILED_NO_RETRY" and
           value["original_N_close_failure"] == {"code": "one_window_finish", "type": "PairingError"} and
           value["original_attempt_relabelled_complete"] is False,
           "exact_retained_failure_and_observation_scope")
    P.need(value["observed_env_steps"] == 140 and value["observed_native_calls"] == PREVIOUS_CALLS and
           value["remaining_roles"] == list(ROLES) and value["maximum_new_environment_steps"] == 210 and
           value["source_and_N_reexecution_allowed"] is False and
           value["source_parent_is_not_cached_receiver_T_N"] is True and
           value["N_repeat_passed"] is False and value["new_environment_execution_authorized"] is False,
           "only_three_missing_roles_and_no_witness_execution_grant")


def cumulative_budget(value, deadline_epoch):
    """Precharge all prior entries; this pure helper never creates an environment."""
    witness_gate(value)
    return Budget(deadline_epoch, counts=copy.deepcopy(PREVIOUS_CALLS),
                  attempted=copy.deepcopy(PREVIOUS_CALLS),
                  resets=ResetBudget(maximum=10, attempted=4, successful=4))


def run_remaining(source, cached, runtime_factory, *, witness, authority, emit,
                  deadline_epoch, synthetic=False):
    P.need(math.isfinite(deadline_epoch) and source.root_id == P.ROOT, "fixed_root_and_new_absolute_deadline")
    if not synthetic:
        P.need(isinstance(authority, dict) and authority.get("direct_human_approved") is True and
               authority.get("scope") == "q11_remaining_physical_continuation" and
               authority.get("root_id") == P.ROOT and authority.get("query_index") == 11 and
               authority.get("maximum_new_environment_steps") == 210 and
               authority.get("maximum_cumulative_environment_steps") == 350 and
               authority.get("previous_environment_steps") == 140 and
               authority.get("resume_witness_identity") == WITNESS and
               authority.get("new_model_calls") == authority.get("GPU") == 0,
               "new_exact_continuation_authority_required")
        witness = load_witness(witness)
    else:
        P.need(isinstance(witness, dict), "synthetic_witness_dictionary")
        witness = copy.deepcopy(witness)
    witness_gate(witness)
    P.need(set(cached) == {"N", "S", "R"}, "exact_cached_three_commands")
    for row in cached.values():
        P.need(row["actions"].shape == (1, 10, 7) and row["actions"].dtype == P.np.float64 and
               row["raw32"].shape == (1, 10, 32) and row["raw32"].dtype == P.np.float32 and
               P.np.isfinite(row["actions"]).all() and P.np.isfinite(row["raw32"]).all(),
               "cached_original_command_ABI")
    results, new_steps = [], 0
    def check():
        P.need(time.time() < deadline_epoch, "continuation_absolute_science_deadline")
    for role in ROLES:
        runtime, queries, trajectory = None, [], []
        result = {"role": role, "status": "NOT_STARTED", "env_steps": 0,
                  "model_calls": 0, "model_attempts": 0}
        try:
            check()
            runtime = runtime_factory(role)
            P.need(getattr(runtime, "role", None) == role and
                   bool(getattr(runtime, "synthetic", False)) == synthetic,
                   "only_requested_missing_role_and_runtime_kind")
            runtime.open(source, check=check)
            P.need(runtime.env_step_calls == 10 and runtime.model_calls == 0,
                   "ten_original_dummy_steps_no_model")
            for q in range(11):
                check()
                queries.append(P.query_gate(source, runtime, q))
                chunk = source.parent_actions(q)
                runtime.bind_recorded_actions(chunk, q)
                for slot, action in enumerate(chunk[0, :5]):
                    check()
                    P.need(runtime.env_step_calls < 70, "missing_branch70_step_cap")
                    actual = runtime.step(action, query=q, slot=slot, steps=5*q+slot)
                    P.need(runtime.model_calls == getattr(runtime, "model_attempts", 0) == 0,
                           "unexpected_model_activity")
                    P.source_step_gate(source, actual, q, slot)
                runtime.query_done(q)
            queries.append(P.query_gate(source, runtime, 11))
            P.need(runtime.env_step_calls == 65, "q11_after_original_prefix")
            critical = runtime.critical_state()
            start = P.digest({"declared": runtime.group_ids(), "critical": critical})
            P.need(start == witness["recorded_complete_q11_start_identity"] and
                   P.digest(critical) == witness["critical_state_identity_recomputed"] and
                   P.digest(runtime.step_snapshot()) == witness["first_before_physics_time_pose_identity"],
                   "cross_job_exact_start_critical_physics_time_pose")
            key = "N" if role == "N_repeat" else role
            row = cached[key]
            runtime.bind_cached_commands(row["actions"], row["raw32"], 11, key)
            for slot, action in enumerate(row["actions"][0, :5]):
                check()
                P.need(runtime.env_step_calls < 70, "missing_branch70_step_cap")
                actual = runtime.step(action, query=11, slot=slot, steps=55+slot)
                P.need(runtime.model_calls == getattr(runtime, "model_attempts", 0) == 0,
                       "unexpected_model_activity")
                P.need(isinstance(actual["done"], (bool, P.np.bool_)) and
                       isinstance(actual["official_success"], (bool, P.np.bool_)), "native_boolean_outcomes")
                P.canonical(actual["physical_outputs"])
                trajectory.append(actual)
                if actual["done"] or actual["official_success"]:
                    break
            runtime.query_done(11)
            signature = P.trajectory_identity(trajectory)
            if role == "N_repeat":
                P.need(len(trajectory) == 5 and signature == witness["N_trajectory_identity_recomputed"],
                       "N_repeat_exact_retained_observation_before_S_R")
            P.need(66 <= runtime.env_step_calls <= 70, "actual_bounded_missing_branch")
            result.update(status="OBSERVED_TERMINAL_PREFIX" if len(trajectory) < 5 else "OBSERVED_FIVE_COMMANDS",
                start_identity=start, trajectory_identity=signature, queries=queries, trajectory=trajectory,
                observed_continuation_steps=len(trajectory), start_critical_witness=critical)
        except BaseException as error:
            result.update(status="FAILED_NO_RETRY", failure={"type": type(error).__name__,
                "code": str(error)[:180]}, queries=queries, trajectory=trajectory,
                observed_continuation_steps=len(trajectory))
        finally:
            if runtime is not None:
                result.update(env_steps=runtime.env_step_calls, model_calls=runtime.model_calls,
                              model_attempts=getattr(runtime, "model_attempts", 0))
                try:
                    runtime.close()
                except BaseException as error:
                    result.update(status="FAILED_NO_RETRY", close_failure={"type": type(error).__name__,
                                  "code": str(error)[:180]})
                finally:
                    result.update(env_steps=runtime.env_step_calls, model_calls=runtime.model_calls,
                                  model_attempts=getattr(runtime, "model_attempts", 0))
                    if result["model_calls"] or result["model_attempts"]:
                        result["status"] = "FAILED_NO_RETRY"
                        result["model_activity_present_at_branch_close"] = True
            new_steps += result["env_steps"]
        P.need(new_steps <= 210 and 140+new_steps <= 350, "new_and_cumulative_step_caps")
        results.append(result)
        emit("REMAINING_BRANCH_RESULT", result)
        if result["status"] == "FAILED_NO_RETRY":
            results.extend({"role": remaining, "status": "NOT_ATTEMPTED_PREVIOUS_GATE_FAILED",
                            "env_steps": 0, "model_calls": 0, "model_attempts": 0}
                           for remaining in ROLES[len(results):])
            break
    complete = all(r["status"] in ("OBSERVED_TERMINAL_PREFIX", "OBSERVED_FIVE_COMMANDS") for r in results)
    return {"schema": "paper2-q11-remaining-physical-campaign-v1", "synthetic": synthetic,
        "roles": results, "previous_environment_steps": 140, "new_environment_steps": new_steps,
        "cumulative_environment_steps": 140+new_steps, "maximum_new_environment_steps": 210,
        "maximum_cumulative_environment_steps": 350, "completed_remaining_roles": complete,
        "combined_physical_readout_available": complete,
        "model_calls": sum(r["model_calls"] for r in results),
        "model_attempts": sum(r["model_attempts"] for r in results), "GPU": 0,
        "retained_N_original_status": witness["original_N_attempt_status"],
        "retained_N_original_close_failure": witness["original_N_close_failure"],
        "old_run_status_modified": False, "source_and_N_case_reexecuted": False,
        "task_transfer_or_success_rate_established": False}
