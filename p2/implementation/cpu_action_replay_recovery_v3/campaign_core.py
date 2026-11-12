"""Bounded campaign component; no simulator/model imports or CLI execution.

The production native adapter, resource transport and direct approval binding
remain separate required components. Synthetic tests use fake runtimes only.
"""
import hashlib
import json
import math
import time
import numpy as np

ROOT = "SDRU-CDEV-T00-R00"
ROLES = ("source_qualification", "N", "N_repeat", "S", "R")
OLD_MIXED = frozenset(("environment", "root.observation",
                      "observable.agentview_image", "observable.robot0_eye_in_hand_image"))
IMAGE_KEYS = frozenset(("agentview_image", "robot0_eye_in_hand_image"))
STEP_FIELDS = ("sim_time", "flat_physics", "returned_eef_pos", "returned_eef_quat")


def need(ok, code):
    if not ok:
        raise ValueError(code)


def canonical(value):
    if isinstance(value, np.ndarray):
        need(not value.dtype.hasobject, "no_object_snapshot")
        if value.dtype.kind in "biufc":
            need(np.isfinite(value).all(), "finite_snapshot")
        return {"array": {"shape": list(value.shape), "dtype": value.dtype.str,
                          "sha256": hashlib.sha256(value.tobytes(order="C")).hexdigest()}}
    if isinstance(value, np.generic):
        return canonical(np.asarray(value))
    if isinstance(value, dict):
        need(all(isinstance(k, str) for k in value), "named_snapshot_fields")
        return {k: canonical(v) for k, v in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return {"sequence_type": type(value).__name__, "items": [canonical(v) for v in value]}
    if type(value) is float:
        need(math.isfinite(value), "finite_snapshot_float")
        return {"float_hex": value.hex()}
    need(value is None or type(value) in (str, int, bool), "supported_snapshot_value")
    return value


def digest(value):
    raw = json.dumps(canonical(value), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def equal(a, b):
    return digest(np.asarray(a)) == digest(np.asarray(b))


def query_gate(source, runtime, q):
    live = runtime.query(q)
    need(runtime.model_calls == 0 and getattr(runtime, "model_attempts", 0) == 0,
         "unexpected_model_activity")
    actual, expected = runtime.group_ids(), source.diagnostic(q)["before"]
    need(set(expected) <= set(actual), "all_source_declared_groups_present")
    statuses = {k: "EXACT" if actual[k] == v else "DIFFERENT" for k, v in expected.items()}
    need(all(v == "EXACT" for k, v in statuses.items() if k not in OLD_MIXED),
         "source_unmixed_state_mismatch")
    anchor_proof = None
    if q in (0, 11, 12):
        anchor = source.anchor(q)
        need(equal(live["flat_physics"], anchor["physical_sim_state"]),
             "source_anchor_physics_mismatch")
        need(set(anchor["observation"]) <= set(live["observation"]), "source_observation_keys")
        matches = {k: equal(live["observation"][k], v) for k, v in anchor["observation"].items()}
        need(all(v for k, v in matches.items() if k not in IMAGE_KEYS),
             "source_nonimage_observation_mismatch")
        anchor_proof = {"physics_exact": True, "observation_typed_exact": matches}
    return {"query": q, "source_group_status": statuses, "anchor": anchor_proof,
            "old_all_declared_groups_exact": all(v == "EXACT" for v in statuses.values()),
            "old_full_replay_qualification_relabelled": False}


def source_step_gate(source, actual, q, slot):
    expected, outcome = source.issued(q, slot), source.outcome(q, slot)
    need(all(equal(actual[phase][field], expected["values"][phase][field])
             for phase in ("before", "after") for field in STEP_FIELDS),
         "source_step_transition_mismatch")
    need(actual["done"] == outcome["done"] and
         actual["official_success"] == outcome["official_success"], "source_outcome_mismatch")
    need(not actual["done"] and not actual["official_success"], "source_early_terminal")


def trajectory_identity(rows):
    return digest([{k: row[k] for k in ("before", "after", "done", "official_success",
                                      "physical_outputs")} for row in rows])


def run_campaign(source, cached, runtime_factory, *, authority, emit, deadline_epoch,
                 synthetic=False):
    """One source qualification then N, N-repeat, S, R; never retry a branch."""
    need(math.isfinite(deadline_epoch), "finite_absolute_deadline")
    need(source.root_id == ROOT, "single_frozen_root")
    if not synthetic:
        need(isinstance(authority, dict) and authority.get("direct_human_approved") is True and
             authority.get("scope") == "q11_cached_command_physical_pilot" and
             authority.get("root_id") == ROOT and authority.get("query_index") == 11 and
             authority.get("maximum_explicit_env_steps") == 350 and
             authority.get("new_model_calls") == 0 and authority.get("GPU") == 0,
             "new_direct_environment_authority_required")
    need(set(cached) == {"N", "S", "R"}, "exact_three_cached_commands")
    for row in cached.values():
        need(row["actions"].shape == (1, 10, 7) and row["actions"].dtype == np.float64 and
             row["raw32"].shape == (1, 10, 32) and row["raw32"].dtype == np.float32 and
             np.isfinite(row["actions"]).all() and np.isfinite(row["raw32"]).all(),
             "cached_command_ABI")
    results, total_steps, reference_start, n_identity = [], 0, None, None
    def check():
        need(time.time() < deadline_epoch, "absolute_science_deadline")
    for role in ROLES:
        runtime, queries, trajectory = None, [], []
        result = {"role": role, "status": "NOT_STARTED", "env_steps": 0,
                  "model_calls": 0, "model_attempts": 0}
        try:
            check()
            runtime = runtime_factory(role)
            need(bool(getattr(runtime, "synthetic", False)) == synthetic, "runtime_kind_matches_mode")
            runtime.open(source, check=check)
            need(runtime.env_step_calls == 10 and runtime.model_calls == 0,
                 "ten_original_dummies_no_model")
            for q in range(11):
                check()
                queries.append(query_gate(source, runtime, q))
                chunk = source.parent_actions(q)
                runtime.bind_recorded_actions(chunk, q)
                for slot, action in enumerate(chunk[0, :5]):
                    check()
                    need(runtime.env_step_calls < 70, "branch_explicit_step_cap")
                    actual = runtime.step(action, query=q, slot=slot, steps=5*q+slot)
                    need(runtime.model_calls == 0 and getattr(runtime, "model_attempts", 0) == 0,
                         "unexpected_model_activity")
                    source_step_gate(source, actual, q, slot)
                runtime.query_done(q)
            queries.append(query_gate(source, runtime, 11))
            need(runtime.env_step_calls == 65 and runtime.model_calls == 0, "q11_after_exact_prefix")
            critical = runtime.critical_state()
            need({"gripper_current_action", "environment", "controller", "compiled_model"} <= set(critical),
                 "explicit_gripper_and_control_state_required")
            start = digest({"declared": runtime.group_ids(), "critical": critical})
            if role == "source_qualification":
                reference_start = start
            else:
                need(start == reference_start, "independent_rebuild_start_mismatch")
            if role == "source_qualification":
                commands = source.parent_actions(11)
                runtime.bind_recorded_actions(commands, 11)
            else:
                key = "N" if role == "N_repeat" else role
                commands = cached[key]["actions"]
                runtime.bind_cached_commands(commands, cached[key]["raw32"], 11, key)
            for slot, action in enumerate(commands[0, :5]):
                check()
                need(runtime.env_step_calls < 70, "branch_explicit_step_cap")
                actual = runtime.step(action, query=11, slot=slot, steps=55+slot)
                need(runtime.model_calls == 0 and getattr(runtime, "model_attempts", 0) == 0,
                     "unexpected_model_activity")
                need(isinstance(actual["done"], (bool, np.bool_)) and
                     isinstance(actual["official_success"], (bool, np.bool_)),
                     "native_boolean_outcome")
                canonical(actual["physical_outputs"])
                trajectory.append(actual)
                if role == "source_qualification":
                    source_step_gate(source, actual, 11, slot)
                elif actual["done"] or actual["official_success"]:
                    break  # Observed terminal prefix retained, no invented remaining samples.
            runtime.query_done(11)
            if role == "source_qualification":
                queries.append(query_gate(source, runtime, 12))
                need(len(trajectory) == 5, "full_source_qualification_continuation")
            signature = trajectory_identity(trajectory)
            if role == "N":
                n_identity = signature
            elif role == "N_repeat":
                need(signature == n_identity, "N_repeat_trajectory_mismatch")
            need(runtime.model_calls == 0 and 65 <= runtime.env_step_calls <= 70,
                 "bounded_no_model_branch")
            result.update(status="QUALIFIED_SOURCE_PATH" if role == "source_qualification"
                          else "OBSERVED_TERMINAL_PREFIX" if len(trajectory) < 5 else "OBSERVED_FIVE_COMMANDS",
                          start_identity=start, trajectory_identity=signature,
                          start_critical_witness=critical,
                          observed_continuation_steps=len(trajectory), queries=queries,
                          trajectory=trajectory, old_full_replay_qualified=False)
        except BaseException as error:
            result.update(status="FAILED_NO_RETRY", failure={"type": type(error).__name__,
                          "code": str(error)[:180]}, queries=queries,
                          observed_continuation_steps=len(trajectory), trajectory=trajectory)
        finally:
            if runtime is not None:
                result["env_steps"] = runtime.env_step_calls
                result["model_calls"] = runtime.model_calls
                result["model_attempts"] = getattr(runtime, "model_attempts", 0)
                try:
                    runtime.close()
                except BaseException as error:
                    result.update(status="FAILED_NO_RETRY",
                                  close_failure={"type": type(error).__name__, "code": str(error)[:180]})
            total_steps += result["env_steps"]
        need(total_steps <= 350, "campaign_explicit_step_cap")
        results.append(result)
        emit("BRANCH_RESULT", result)
        if result["status"] == "FAILED_NO_RETRY":
            for remaining in ROLES[len(results):]:
                results.append({"role": remaining, "status": "NOT_ATTEMPTED_PREVIOUS_GATE_FAILED",
                                "env_steps": 0, "model_calls": 0, "model_attempts": 0})
            break
    return {"schema": "paper2-q11-cached-command-campaign-v1", "roles": results,
            "total_explicit_env_steps": total_steps, "maximum_explicit_env_steps": 350,
            "model_calls": sum(r["model_calls"] for r in results),
            "model_attempts": sum(r["model_attempts"] for r in results),
            "GPU": 0, "synthetic": synthetic,
            "completed_all_roles": all(r["status"] != "FAILED_NO_RETRY" and
                                      not r["status"].startswith("NOT_") for r in results),
            "old_full_replay_qualification_changed": False,
            "task_transfer_or_success_rate_established": False}
