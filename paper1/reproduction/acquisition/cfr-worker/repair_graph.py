"""cf-repair-v1 query graphs for the loaded pi05 model (TECH-CONTRACT-v1 4.1).

Adapted from NSA/execution/native_query.py:51-125 (SHA 3664de7a...): same preprocessing,
one prefix prefill per observation inside the compiled query function, the native Euler
``while_loop`` with explicit FP32 noise, ``dt = -1/10`` and ``time >= -dt/2``.
Changes for this experiment:
  * two prefills per query (treated obs_T and counterfactual clean obs_C); never per call;
  * on EVERY generation call j: a clean action-expert pass over layers 0..8 (variant A) or
    0..17 (variant B) at the arm's current (x_j, t_j) with the clean KV gives H_C; the
    treated 18-layer pass is edited at L9 (index 8) by cfr_repair_rules.edit_rule and
    continues with the treated KV (mixed context);
  * per-call capture of H_T, H_C, H', L10, L18 and the rule's FP32 auxiliaries (j0-only
    capture and the j0-only dose of native_query.py:86,95-100 are removed);
  * teacher-forced mode (T4 only): x_j, t_j from a supplied native path, selected with
    ``jnp.where`` inside the same compiled online graph;
  * the edit schedule is a dynamic boolean mask [10] (recorded in the generation proof).
The record-all route uses the same per-layer scan without an edit.

Nothing here loads a checkpoint, builds an environment or touches the network.
"""
# ruff: noqa: PLC0415 - JAX/openpi imports stay lazy so source audits need no JAX.

from __future__ import annotations

NUM_STEPS = 10
ACTION_HORIZON = 10
ACTION_DIM = 32
WIDTH = 1024


def _deps():
    import einops
    import jax
    import jax.numpy as jnp

    from openpi.models import model as model_lib
    from openpi.models.pi0 import make_attn_mask
    return einops, jax, jnp, model_lib, make_attn_mask


def _check_model(model, noise, observation):
    import jax.numpy as jnp
    if not getattr(model, "pi05", False):
        raise ValueError("pi05_model_required")
    if model.action_horizon != ACTION_HORIZON or model.action_dim != ACTION_DIM:
        raise ValueError("pi05_action_ABI_mismatch")
    if noise.shape != (observation.state.shape[0], ACTION_HORIZON, ACTION_DIM) or noise.dtype != jnp.float32:
        raise ValueError("explicit_canonical_FP32_noise_required")


def _prefill(model, obs):
    _, _, jnp, _, make_attn_mask = _deps()
    tokens, prefix_mask, prefix_ar_mask = model.embed_prefix(obs)
    attn = make_attn_mask(prefix_mask, prefix_ar_mask)
    positions = jnp.cumsum(prefix_mask, axis=1) - 1
    _, kv_cache = model.PaliGemma.llm([tokens, None], mask=attn, positions=positions)
    return tokens, prefix_mask, kv_cache


def _suffix_inputs(model, obs, prefix_mask, x_t, time, batch):
    einops, _, jnp, _, make_attn_mask = _deps()
    suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = model.embed_suffix(
        obs, x_t, jnp.broadcast_to(time, batch))
    suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
    prefix_to_suffix = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
    full_attn_mask = jnp.concatenate([prefix_to_suffix, suffix_attn_mask], axis=-1)
    positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1
    return suffix_tokens, full_attn_mask, positions, adarms_cond


def record_query(model, observation, noise):
    """Record-all-calls route: native Euler loop, per-layer scan, L9/L10/L18 at every call."""
    _, jax, jnp, model_lib, _ = _deps()
    _check_model(model, noise, observation)
    obs = model_lib.preprocess_observation(None, observation, train=False)
    batch = obs.state.shape[0]
    dt = -1.0 / NUM_STEPS
    prefix_tokens, prefix_mask, kv_cache = _prefill(model, obs)
    hidden_dtype = prefix_tokens.dtype
    h_shape = (NUM_STEPS, batch, ACTION_HORIZON, WIDTH)
    zeros_h = jnp.zeros(h_shape, hidden_dtype)
    xs = jnp.zeros((NUM_STEPS + 1, *noise.shape), noise.dtype).at[0].set(noise)
    velocities = jnp.zeros((NUM_STEPS, *noise.shape), hidden_dtype)
    times = jnp.zeros((NUM_STEPS,), jnp.float32)

    def step(carry):
        x_t, time, index, xs, velocities, times, h9, h10, h18 = carry
        suffix, mask, positions, cond = _suffix_inputs(model, obs, prefix_mask, x_t, time, batch)
        readout, trace = model.PaliGemma.llm(
            [None, suffix], mask=mask, positions=positions, kv_cache=kv_cache,
            adarms_cond=[None, cond], layers=18, method="cfr_scan_suffix")
        velocity = model.action_out_proj(readout[:, -ACTION_HORIZON:])
        if velocity.shape != x_t.shape or velocity.dtype != hidden_dtype:
            raise ValueError("native_velocity_ABI_mismatch")
        next_x = x_t + dt * velocity
        return (next_x, time + dt, index + 1, xs.at[index + 1].set(next_x),
                velocities.at[index].set(velocity), times.at[index].set(time),
                h9.at[index].set(trace["L9_H_before"]), h10.at[index].set(trace["L10_H_after"]),
                h18.at[index].set(trace["L18_H_after"]))

    def cond(carry):
        return carry[1] >= -dt / 2

    raw, final_time, completed, xs, velocities, times, h9, h10, h18 = jax.lax.while_loop(
        cond, step, (noise, 1.0, jnp.asarray(0, jnp.int32), xs, velocities, times, zeros_h, zeros_h, zeros_h))
    return raw, {"flow_x": xs, "flow_velocity": velocities, "flow_time": times,
                 "L9_H": h9, "L10_H": h10, "L18_H": h18,
                 "completed_updates": completed, "final_time": final_time}


def repair_query(model, observation_t, observation_c, noise, bases, route, alpha, arm, eq_index,
                 c_noop, schedule_mask, teacher_forced, teacher_x, teacher_t, *, clean_layers):
    """One complete repair-arm query (all ten calls), TECH-CONTRACT-v1 4.1 graph.

    bases: FP32 [5,1024,16] (B, C, G1, G2, G3); route/arm int32; eq_index int32[3];
    alpha/c_noop float32; schedule_mask bool[10]; teacher_forced bool; teacher_x
    FP32 [10,b,10,32]; teacher_t FP32 [10]; clean_layers static (9 = variant A, 18 = B).
    """
    _, jax, jnp, model_lib, _ = _deps()
    _check_model(model, noise, observation_t)
    if clean_layers not in (9, 18):
        raise ValueError("clean_variant_A9_or_B18_required")
    if bases.shape != (5, WIDTH, 16) or bases.dtype != jnp.float32:
        raise ValueError("five_FP32_rank16_bases_required")
    obs_t = model_lib.preprocess_observation(None, observation_t, train=False)
    obs_c = model_lib.preprocess_observation(None, observation_c, train=False)
    batch = obs_t.state.shape[0]
    dt = -1.0 / NUM_STEPS
    prefix_t, mask_t, kv_t = _prefill(model, obs_t)
    _prefix_c, mask_c, kv_c = _prefill(model, obs_c)
    hidden_dtype = prefix_t.dtype
    h_shape = (NUM_STEPS, batch, ACTION_HORIZON, WIDTH)
    zeros_h = jnp.zeros(h_shape, hidden_dtype)
    tok = (NUM_STEPS, batch, ACTION_HORIZON)
    carry0 = {
        "xs": jnp.zeros((NUM_STEPS + 1, *noise.shape), noise.dtype).at[0].set(noise),
        "x_in": jnp.zeros((NUM_STEPS, *noise.shape), noise.dtype),
        "t_in": jnp.zeros((NUM_STEPS,), jnp.float32),
        "velocity": jnp.zeros((NUM_STEPS, *noise.shape), hidden_dtype),
        "H_T": zeros_h, "H_C": zeros_h, "H_P": zeros_h, "L10": zeros_h, "L18": zeros_h,
        "L18_clean": zeros_h,
        "w": jnp.zeros(h_shape, jnp.float32),
        "declared_noop": jnp.zeros(tok, jnp.bool_), "s": jnp.zeros(tok, jnp.float32),
        "argmin": jnp.zeros(tok, jnp.int32), "tau": jnp.zeros(tok, jnp.float32),
        "p_norm": jnp.zeros((NUM_STEPS, 5, batch, ACTION_HORIZON), jnp.float32),
        "applied": jnp.zeros((NUM_STEPS,), jnp.bool_),
    }

    def step(carry):
        x_t, time, index, rec = carry
        x_in = jnp.where(teacher_forced, teacher_x[index], x_t)
        t_in = jnp.where(teacher_forced, teacher_t[index], time)
        suffix_c, attn_c, pos_c, cond_c = _suffix_inputs(model, obs_c, mask_c, x_in, t_in, batch)
        _clean_out, clean = model.PaliGemma.llm(
            [None, suffix_c], mask=attn_c, positions=pos_c, kv_cache=kv_c,
            adarms_cond=[None, cond_c], layers=clean_layers, method="cfr_scan_suffix")
        h_c = clean["L9_H_before"]
        edit_inputs = {"h_c": h_c, "bases": bases, "route": route, "alpha": alpha, "arm": arm,
                       "eq_index": eq_index, "c_noop": c_noop, "active": schedule_mask[index]}
        suffix_t, attn_t, pos_t, cond_t = _suffix_inputs(model, obs_t, mask_t, x_in, t_in, batch)
        readout, treated = model.PaliGemma.llm(
            [None, suffix_t], mask=attn_t, positions=pos_t, kv_cache=kv_t,
            adarms_cond=[None, cond_t], layers=18, edit_at=8, edit_inputs=edit_inputs,
            method="cfr_scan_suffix")
        velocity = model.action_out_proj(readout[:, -ACTION_HORIZON:])
        if velocity.shape != x_in.shape or velocity.dtype != hidden_dtype:
            raise ValueError("native_velocity_ABI_mismatch")
        next_x = x_in + dt * velocity
        aux = treated["edit_aux"]
        rec = dict(rec)
        rec["xs"] = rec["xs"].at[index + 1].set(next_x)
        rec["x_in"] = rec["x_in"].at[index].set(x_in)
        rec["t_in"] = rec["t_in"].at[index].set(t_in)
        rec["velocity"] = rec["velocity"].at[index].set(velocity)
        rec["H_T"] = rec["H_T"].at[index].set(treated["L9_H_before"])
        rec["H_C"] = rec["H_C"].at[index].set(h_c)
        rec["H_P"] = rec["H_P"].at[index].set(treated["L9_H_after"])
        rec["L10"] = rec["L10"].at[index].set(treated["L10_H_after"])
        rec["L18"] = rec["L18"].at[index].set(treated["L18_H_after"])
        rec["L18_clean"] = rec["L18_clean"].at[index].set(clean["L18_H_after"])
        for key in ("w", "declared_noop", "s", "argmin", "tau", "p_norm", "applied"):
            rec[key] = rec[key].at[index].set(aux[key])
        return (next_x, time + dt, index + 1, rec)

    def cond(carry):
        return carry[1] >= -dt / 2

    raw, final_time, completed, rec = jax.lax.while_loop(
        cond, step, (noise, 1.0, jnp.asarray(0, jnp.int32), carry0))
    rec["completed_updates"] = completed
    rec["final_time"] = final_time
    return raw, rec
