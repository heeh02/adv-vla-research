"""Independent scan-native JAX capsule for an ALREADY loaded Pi0. No loading/launcher.

Import is stdlib-only. Root must stage the explicitly pinned scan-native-capture-v1/overlay at the original Gemma
module path in an isolated source package, and compile the WHOLE query boundary
with its actual NNX module_jit. Do not JIT/prefill independently and call that a
stock-equivalent graph. All changed/capture/remaining graphs need real BF16
qualification. This module intentionally contains no parameter restoration.
"""
from typing import Any, NamedTuple

from contract import Intervention, ResumeState, dispatch_query, require


class Prepared(NamedTuple):
    """Immutable tuple and JAX pytree; no donated/mutable cache state."""
    observation: Any
    cache: Any
    donor_cache: Any
    prefix_mask: Any
    donor_prefix_mask: Any
    prefix_positions: Any
    donor_prefix_positions: Any
    token_mask: Any


def _deps():
    import jax
    import jax.numpy as jnp
    from openpi.models import model as model_module
    from openpi.models.pi0 import make_attn_mask
    return jax, jnp, model_module.preprocess_observation, make_attn_mask


def _architecture(model):
    require(model.pi05 and model.action_horizon == 10 and model.action_dim == 32, "pi05_10x32_only")
    llm = model.PaliGemma.llm.module
    cfg = llm.configs[1]
    require((cfg.depth, cfg.width, cfg.num_heads, cfg.num_kv_heads, cfg.head_dim) == (18, 1024, 8, 1, 256), "pi05_architecture")
    require(llm.embed_dtype == "bfloat16" and not llm.dropout, "native_BF16_deterministic")


def prepare_candidate(model, observation, donor_observation, *, token_mask=None):
    """Both observations must come from the externally bound SAME state/query.

    This function cannot certify rendering/state/prompt identity. No no-op
    calls this function; runtime masks/positions are separately returned gates.
    """
    _architecture(model)
    _, jnp, preprocess, make_mask = _deps()

    def prefill(obs):
        obs = preprocess(None, obs, train=False)
        prefix, valid, ar = model.embed_prefix(obs)
        positions = jnp.cumsum(valid, axis=1) - 1
        _, cache = model.PaliGemma.llm([prefix, None], mask=make_mask(valid, ar), positions=positions)
        return obs, valid, positions, cache

    obs, valid, positions, cache = prefill(observation)
    if donor_observation is None:
        dv, dp, dc = valid, positions, cache
    else:
        _, dv, dp, dc = prefill(donor_observation)
    require(valid.shape == dv.shape and positions.shape == dp.shape, "prefix_ABI_mismatch")
    require(all(a.shape == b.shape and a.dtype == b.dtype for a, b in zip(cache, dc)), "donor_cache_ABI")
    if token_mask is None:
        token_mask = valid
    require(token_mask.shape == valid.shape and token_mask.dtype == jnp.bool_, "prefix_token_mask_BN")
    return Prepared(obs, cache, dc, valid, dv, positions, dp, token_mask)


def _changed_cache(prepared, plan):
    """Functional replacements only; unchanged leaves reuse original values."""
    _, jnp, _, _ = _deps()
    result = []
    for old, donor, edit in zip(prepared.cache, prepared.donor_cache, (plan.edits_k, plan.edits_v)):
        if edit:
            replacement = jnp.where(prepared.token_mask[..., None, None], donor[plan.layer], old[plan.layer])
            result.append(old.at[plan.layer].set(replacement))
        else:
            result.append(old)
    return tuple(result)


def _prepared_abi(prepared):
    _, jnp, _, _ = _deps()
    require(isinstance(prepared, Prepared), "prepared_capsule_required")
    require(prepared.prefix_mask.ndim == 2 and prepared.prefix_mask.dtype == jnp.bool_, "prefix_mask_BN")
    b, n = prepared.prefix_mask.shape
    require(n > 0 and len(prepared.cache) == len(prepared.donor_cache) == 2, "nonempty_prefix_KV_pair")
    for array in (*prepared.cache, *prepared.donor_cache):
        require(array.shape == (18, b, n, 1, 256) and array.dtype == jnp.bfloat16, "actual_BF16_prefix_cache_LBNKH")
    for mask in (prepared.donor_prefix_mask, prepared.token_mask):
        require(mask.shape == (b, n) and mask.dtype == jnp.bool_, "prefix_selection_ABI")
    require(prepared.prefix_positions.shape == prepared.donor_prefix_positions.shape == (b, n), "prefix_position_ABI")


def _suffix(model, prepared, x, time):
    _, jnp, _, make_mask = _deps()
    batch = x.shape[0]
    suffix, valid, ar, condition = model.embed_suffix(prepared.observation, x, jnp.broadcast_to(time, batch))
    mask = jnp.concatenate([jnp.broadcast_to(prepared.prefix_mask[:, None, :],
                           (batch, suffix.shape[1], prepared.prefix_mask.shape[1])), make_mask(valid, ar)], axis=-1)
    positions = jnp.sum(prepared.prefix_mask, axis=-1)[:, None] + jnp.cumsum(valid, axis=-1) - 1
    return suffix, mask, positions, condition


def _ordinary_velocity(model, prepared, x, time, cache):
    suffix, mask, positions, condition = _suffix(model, prepared, x, time)
    (unused, readout), _discard_new_cache = model.PaliGemma.llm(
        [None, suffix], mask=mask, positions=positions, kv_cache=cache, adarms_cond=[None, condition])
    assert unused is None
    return model.action_out_proj(readout[:, -10:])


def _empty_capture(model, prepared, x, time):
    _, jnp, _, _ = _deps()
    suffix, mask, positions, condition = _suffix(model, prepared, x, time)
    cfg = model.PaliGemma.llm.module.configs[1]
    b, t, d = suffix.shape
    n = prepared.prefix_mask.shape[1]
    k, g, h = cfg.num_kv_heads, cfg.num_heads // cfg.num_kv_heads, cfg.head_dim
    z = lambda shape: jnp.zeros(shape, jnp.bfloat16)
    operands = {
        "query_post_rope_scaled": z((b, t, k, g, h)),
        "prefix_key_used": z((b, n, k, h)), "prefix_value_used": z((b, n, k, h)),
        "suffix_key_post_rope": z((b, t, k, h)), "suffix_value": z((b, t, k, h)),
        "probabilities_native": z((b, k, g, t, n+t)),
        "attention_output_post_WO_pre_gate": z((b, t, d)),
        "receiver_residual_input": z((b, t, d)),
        "attention_gate": z((b, 1, d)),
        "hidden_post_block": z((b, t, d)),
    }
    operands = {name: jnp.zeros((3, *a.shape), a.dtype) for name, a in operands.items()}
    operands.update({
        "mask": jnp.zeros_like(mask), "suffix_positions": jnp.zeros_like(positions),
        "adarms_condition": jnp.zeros_like(condition), "query_defined": jnp.zeros_like(mask.any(-1)),
        "final_readout": z((b, t, d)), "velocity": z(x.shape),
        "actual_x": jnp.zeros_like(x), "actual_time": jnp.zeros_like(time),
        "time_weak_type": jnp.asarray(False),
    })
    return operands


def continue_candidate(model, prepared, state, plan, *, capture=False, num_steps=10):
    """Complete the remaining ORIGINAL Euler schedule on fresh branch states.

    A resumed no-op uses an ordinary suffix/while continuation (stock has no
    arbitrary-x resume API). It is explicitly a new graph, requiring diagonal
    continuation qualification. Whole-query noop dispatches stock directly.
    The prefix modification is current-call-only; returned all-key caches are
    NEVER fed into the next layer/call. Full18 base prefix remains immutable.
    """
    _architecture(model)
    _prepared_abi(prepared)
    require(isinstance(state, ResumeState) and isinstance(plan, Intervention), "explicit_resume_and_plan")
    require(plan.mode == "noop" or state.start_call <= plan.call_index, "target_call_already_passed")
    require(not capture or plan.mode != "noop", "noop_cannot_claim_operand_capture")
    jax, jnp, _, _ = _deps()
    x, time = state.x_t, state.time
    require(x.shape == (prepared.prefix_mask.shape[0], 10, 32) and x.dtype == jnp.float32, "native_x32_shape_dtype")
    require(jnp.shape(time) == (), "scalar_actual_time_required")
    changed = _changed_cache(prepared, plan)
    require(jnp.shape(num_steps) == () and jnp.issubdtype(jnp.asarray(num_steps).dtype, jnp.integer),
            "scalar_integer_num_steps_ABI")
    # The actual old C producer passes num_steps=10 through module_jit as a
    # dynamic argument. A Python constant -0.1 is a different compiled graph.
    dt = -1.0 / num_steps
    trace0 = _empty_capture(model, prepared, x, time) if capture else {}
    if capture:
        trace0 = {name: jnp.zeros((3, *a.shape), a.dtype) for name, a in trace0.items()}
    calls = jnp.asarray((0, 5, 9), jnp.int32)

    def body(carry):
        x, time, index, trace, reached, filled = carry
        active = index == plan.call_index
        cache = jax.lax.cond(active, lambda: changed, lambda: prepared.cache)
        if capture:
            # This graph actually consumes the returned P; no CPU/side replay.
            suffix, mask, positions, condition = _suffix(model, prepared, x, time)
            readout, operands = model.PaliGemma.llm(
                [None, suffix], mask=mask, positions=positions, adarms_cond=[None, condition],
                kv_cache=cache, method="scan_native_suffix")
            velocity = model.action_out_proj(readout[:, -10:])
            operands.update(final_readout=readout, velocity=velocity, actual_x=x, actual_time=time,
                            time_weak_type=jnp.asarray(bool(getattr(time, "weak_type", False))))
            slot = jnp.argmax(calls == index)
            selected = jnp.any(calls == index)
            trace = jax.lax.cond(selected,
                lambda: jax.tree.map(lambda old, new: old.at[slot].set(new), trace, operands), lambda: trace)
            filled = jax.lax.cond(selected, lambda: filled.at[slot].set(True), lambda: filled)
        else:
            velocity = _ordinary_velocity(model, prepared, x, time, cache)
        # Not Δx + dt*Δv: update each branch with the original expression.
        return x + dt * velocity, time + dt, index + 1, trace, reached | active, filled

    def cond(carry):
        _, time, index, _, _, _ = carry
        return (time >= -dt / 2) & (index < 10)

    final, final_time, final_index, trace, reached, filled = jax.lax.while_loop(
        cond, body, (x, time, jnp.asarray(state.start_call, jnp.int32), trace0, jnp.asarray(False), jnp.zeros(3, bool)))
    gates = {
        "prefix_mask_identical": jnp.array_equal(prepared.prefix_mask, prepared.donor_prefix_mask),
        "prefix_positions_identical": jnp.array_equal(prepared.prefix_positions, prepared.donor_prefix_positions),
        "selected_tokens_valid": jnp.all(~prepared.token_mask | prepared.prefix_mask),
        "schedule_finished_at_ten": final_index == 10,
        "num_steps_is_ten": num_steps == 10,
        "schedule_terminal_time": final_time < -dt / 2,
        "target_call_reached": reached if plan.mode != "noop" else jnp.asarray(True),
        "inputs_finite": jnp.all(jnp.isfinite(x)) & jnp.isfinite(time),
        "cache_finite": jnp.all(jnp.stack([jnp.all(jnp.isfinite(a)) for a in (*prepared.cache, *prepared.donor_cache)])),
        "final_finite": jnp.all(jnp.isfinite(final)),
    }
    return final, {"operands": trace, "gates": gates, "final_time": final_time,
                   "sampling_abi": {"num_steps": jnp.asarray(num_steps),
                       "num_steps_weak_type": jnp.asarray(bool(getattr(jnp.asarray(num_steps), "weak_type", False))),
                       "num_steps_is_trace_input": jnp.asarray(isinstance(num_steps, jax.core.Tracer)),
                       "actual_dt": jnp.asarray(dt),
                       "dt_weak_type": jnp.asarray(bool(getattr(jnp.asarray(dt), "weak_type", False)))},
                   "completed_updates": final_index - state.start_call, "capture_filled": filled,
                   "capture_calls": calls, "capture_layers": jnp.asarray((0, 8, 17), jnp.int32),
                   "prefix_cache": prepared.cache if capture else (),
                   "prefix_mask": prepared.prefix_mask, "prefix_positions": prepared.prefix_positions}


def query_candidate(model, observation, donor_observation, noise, plan, *, capture=False, token_mask=None, num_steps=10):
    """Pure module_jit boundary. No-op invokes unchanged stock directly.

    The caller binds same physical state/prompt/noise, charges the call, and
    must check every result gate before using any returned action. Until the
    independent real qualification succeeds, outputs are diagnostic only.
    """
    require(isinstance(plan, Intervention), "explicit_intervention_required")
    jax, _, _, _ = _deps()

    def stock():
        return model.sample_actions(jax.random.key(0), observation, noise=noise, num_steps=num_steps), {}

    def candidate():
        require(donor_observation is not None or plan.mode == "observe", "donor_observation_required")
        prepared = prepare_candidate(model, observation, donor_observation, token_mask=token_mask)
        return continue_candidate(model, prepared, ResumeState(0, noise, 1.0), plan, capture=capture, num_steps=num_steps)

    return dispatch_query(plan, stock_sample=stock, candidate_sample=candidate, capture=capture)


def resume_candidate(model, prepared, x_t, time, *, start_call, plan, capture=False, num_steps=10):
    """JIT-friendly remaining-sampling entry; start_call/plan/capture are static.

    The Prepared pytree, opaque typed time and num_steps remain dynamic. Independently
    prepared/resumed graphs are NOT given whole-query native parity by alias.
    """
    return continue_candidate(model, prepared, ResumeState(start_call, x_t, time), plan, capture=capture, num_steps=num_steps)


def check_result_gates(metadata):
    """Host-side scalar gate; must run AFTER block_until_ready and BEFORE issue.

    Passing only validates these input/schedule checks, not numerical parity.
    """
    gates = metadata.get("gates", {})
    required = {"prefix_mask_identical", "prefix_positions_identical", "selected_tokens_valid",
                "schedule_finished_at_ten", "num_steps_is_ten", "schedule_terminal_time", "target_call_reached",
                "inputs_finite", "cache_finite", "final_finite"}
    require(set(gates) == required, "candidate_result_gates_missing")
    for name in sorted(required):
        require(bool(gates[name]), "candidate_gate_" + name)
    return True
