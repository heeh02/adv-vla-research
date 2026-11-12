"""Arm queries, open-loop rollouts and closed-loop episodes for one unit.

Adapted from NSA/execution/closed_loop/runner.py:241-455 (SHA 58f27780...): per-query noise
hash schedule bound before any policy call, caps, five-step execution chunk, official
success/terminal booleans, technical-missing semantics (status ``technical_missing`` with
success None) and terminal-state recording. Changes: LIBERO-Plus backend, S0 restore
instead of reset, the counterfactual observation on every chunk-final step of repair
arms, and every-call repair proofs. The Plus loop semantics (10 dummy steps already in
S0, at most five actions per query, stop on done or check_success) follow A/adapter.py:110-170.
"""

from __future__ import annotations

import time

import numpy as np

from .call_log import call_records
from .common import TechnicalFailure, array_id, need, safe_error
from .policy import arm_config, require_route_bits

EXECUTE_CHUNK = 5
NATIVE_ARMS = ("N_C", "N_T")
REPAIR_ARMS = ("R_resp", "R_clean", "R_rand", "R_full")


def query_arm(ctx, arm, obs_t, obs_c, noise, *, clean_layers, out=None, log_base=None, teacher=None, cfg=None):
    rt, policy = ctx.rt, ctx.rt.policy
    timing = {}
    t0 = time.monotonic()
    if arm in NATIVE_ARMS:
        obs = obs_t if arm == "N_T" else obs_c
        enc, batch = policy.encode(obs, ctx.prompt)
        timing["encode_s"] = time.monotonic() - t0
        result = policy.canonical(enc, batch, noise, label="canonical_" + arm)
        timing["model_s"] = result["model_seconds"]
        return {**result, "timing": timing, "arm": arm}
    cfg = cfg or arm_config(arm, g_assigned=rt.g_assigned)
    enc_t, enc_c, batch, pair = policy.encode_pair(obs_t, obs_c, ctx.prompt)
    timing["encode_s"] = time.monotonic() - t0
    result = policy.repair(enc_t, enc_c, batch, noise, ctx.bases_stack, cfg, clean_layers=clean_layers,
                           teacher=teacher, label="repair_" + cfg["route"])
    timing["model_s"] = result["model_seconds"]
    require_route_bits(result["trace"], cfg["route"])
    t1 = time.monotonic()
    if out is not None and log_base is not None:
        for record in call_records(result["trace"], base=log_base, cfg=cfg, bases_by_name=ctx.bases_by_name,
                                   template=ctx.template, gram=ctx.gram, identity=ctx.identity_log):
            out.log_call(record)
    timing["log_s"] = time.monotonic() - t1
    return {**result, "timing": timing, "arm": arm, "cfg": cfg, "encode_pair": pair}


def rollout_native(ctx, n_queries, *, policy_route="canonical", capture_every_chunk=True):
    """N_T rollout from S0 (canonical or record route); returns per-query inputs incl. obs_C."""
    ctx.restore_T()
    env, h = ctx.E_T, ctx.h_T
    obs_t, obs_c, receipt = ctx.obs_T0, ctx.obs_C0, ctx.cf_receipt0
    rows, steps = [], 10
    for q in range(n_queries):
        noise = ctx.noise(q)
        rows.append({"query": q, "obs_t": obs_t, "obs_c": obs_c, "cf_receipt": receipt, "noise": noise,
                     "physics": np.asarray(env.env.get_sim_state()).copy()})
        enc, batch = ctx.rt.policy.encode(obs_t, ctx.prompt)
        result = (ctx.rt.policy.canonical(enc, batch, noise, label="rollout_canonical") if policy_route == "canonical"
                  else ctx.rt.policy.record(enc, batch, noise, label="rollout_record"))
        rows[-1]["native"] = result
        terminal = False
        for i, action in enumerate(result["adapter7"][0, :EXECUTE_CHUNK]):
            last = i == EXECUTE_CHUNK - 1
            if last and capture_every_chunk:
                obs_t, done = ctx.tapped_step(env, h, action, step_index=steps + 1)
                obs_c, receipt = ctx._last_obs_c, ctx._last_cf
            else:
                obs_t, done = env.step(action)
            steps += 1
            if done or env.check_success():
                terminal = True
                break
        if terminal:
            rows.append({"query": q + 1, "terminal_before_query": True})
            break
    else:
        rows.append({"query": n_queries, "obs_t": obs_t, "obs_c": obs_c, "cf_receipt": receipt,
                     "noise": ctx.noise(n_queries), "physics": np.asarray(env.env.get_sim_state()).copy()})
    return rows


def run_episode(ctx, arm, *, clean_layers, out, sample_id, test_id, deadline, max_queries=None, label_role="formal_1A_tech"):
    """One closed-loop episode for one arm (outcome and issued commands are sealed by the caller)."""
    rt = ctx.rt
    record = {"schema": "cf-repair-1a-episode-v1", "sample_id": sample_id, "test_id": test_id, "arm": arm,
              "unit": {"row_id": ctx.unit["row_id"], "family": ctx.family, "suite": ctx.case["suite"],
                       "classification_id": ctx.case["classification_id"], "variant": ctx.unit["variant_name"]},
              "noise_label": ctx.noise_label, "noise_schedule": ctx.noise_schedule, "clean_variant": "A" if clean_layers == 9 else "B",
              "queries": [], "execution_status": "running", "technical_missing_reason": None, "role": label_role}
    sealed = {"issued_adapter7": [], "success": None, "termination": None, "policy_steps": 0}
    is_base = arm == "N_C"
    try:
        restore_info, restore_check = ctx.restore_C() if is_base else ctx.restore_T()
        record["restore"] = {"check": restore_check, "into_base": is_base}
        env, h = (ctx.E_C, ctx.h_C) if is_base else (ctx.E_T, ctx.h_T)
        obs_t, obs_c = ctx.obs_T0, ctx.obs_C0
        steps, max_steps = 0, ctx.case["max_policy_steps"]
        horizon = ctx.horizon_queries if max_queries is None else min(max_queries, ctx.horizon_queries)
        done = success = False
        for q in range(horizon):
            need(time.time() < deadline, "absolute_deadline")
            noise = ctx.noise(q)
            log_base = {"job_test": test_id, "sample_id": sample_id, "query_index": q, "noise_label": ctx.noise_label,
                        "unit_row_id": ctx.unit["row_id"], "episode": True}
            result = query_arm(ctx, arm, obs_t, obs_c, noise, clean_layers=clean_layers,
                               out=out if arm in REPAIR_ARMS else None, log_base=log_base)
            entry = {"query": q, "noise_sha256": array_id(noise)["sha256"], "raw32": array_id(result["raw32"]),
                     "timing": dict(result["timing"]), "policy_step": steps}
            if arm in REPAIR_ARMS:
                entry["generation"] = result["generation"]
            env_t = 0.0
            cf_t = 0.0
            for i, action in enumerate(result["adapter7"][0, :EXECUTE_CHUNK]):
                if steps >= max_steps:
                    break
                last = i == EXECUTE_CHUNK - 1
                ts = time.monotonic()
                if arm in REPAIR_ARMS and last:
                    obs, done = ctx.tapped_step(env, h, action, step_index=10 + steps + 1)
                    cf_t += ctx._last_cf["seconds"]
                    obs_c = ctx._last_obs_c
                else:
                    obs, done = env.step(action)
                env_t += time.monotonic() - ts
                sealed["issued_adapter7"].append(np.asarray(action, np.float64).tolist())
                steps += 1
                success = bool(env.check_success())
                if is_base:
                    obs_c = obs
                obs_t = obs
                if done or success:
                    break
            entry["timing"]["env_steps_s"] = env_t - cf_t
            entry["timing"]["counterfactual_capture_s"] = cf_t
            record["queries"].append(entry)
            if done or success or steps >= max_steps:
                break
        sealed.update(success=bool(success), policy_steps=steps,
                      termination="success" if success else ("natural_terminal" if done else "policy_budget_or_query_cap"))
        record["execution_status"] = "complete"
    except TechnicalFailure as error:
        record.update(execution_status="technical_missing", technical_missing_reason=safe_error(error))
    except Exception as error:  # noqa: BLE001 - retained as data, never retried
        record.update(execution_status="technical_missing", technical_missing_reason=safe_error(error))
    record["query_count"] = len(record["queries"])
    record["episode_length_policy_steps"] = sealed["policy_steps"]
    record["sealed"] = sealed
    record["sealed_fields"] = ["sealed"]
    return record
