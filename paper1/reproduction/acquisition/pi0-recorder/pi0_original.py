import logging

import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.eval.paired_trace import hidden_group_rms
from openpi.eval.paired_trace import shadow_recompute_error
from openpi.eval.paired_trace import summarize_pair
from openpi.models import model as _model
from openpi.models import pi0_config
import openpi.models.gemma as _gemma
import openpi.models.siglip as _siglip
from openpi.shared import array_typing as at

logger = logging.getLogger("openpi")


@at.typecheck
def sample_flow_targets(
    noise_rng: at.KeyArrayLike, time_rng: at.KeyArrayLike, actions: _model.Actions
) -> tuple[
    at.Float[at.Array, "*b ah ad"],
    at.Float[at.Array, "*b"],
    at.Float[at.Array, "*b ah ad"],
    at.Float[at.Array, "*b ah ad"],
]:
    """Sample flow-matching noise, time, noisy action input, and velocity target."""
    batch_shape = actions.shape[:-2]
    noise = jax.random.normal(noise_rng, actions.shape)
    time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
    time_expanded = time[..., None, None]
    x_t = time_expanded * noise + (1 - time_expanded) * actions
    u_t = noise - actions
    return noise, time, x_t, u_t


def make_attn_mask(input_mask, mask_ar):
    """Adapted from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` bool[?B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: bool[?B, N] mask that's true where previous tokens cannot depend on
        it and false where it shares the same attention mask as the previous token.
    """
    mask_ar = jnp.broadcast_to(mask_ar, input_mask.shape)
    cumsum = jnp.cumsum(mask_ar, axis=1)
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
    return jnp.logical_and(attn_mask, valid_mask)


@at.typecheck
def posemb_sincos(
    pos: at.Real[at.Array, " b"], embedding_dim: int, min_period: float, max_period: float
) -> at.Float[at.Array, "b {embedding_dim}"]:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if embedding_dim % 2 != 0:
        raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by 2")

    fraction = jnp.linspace(0.0, 1.0, embedding_dim // 2)
    period = min_period * (max_period / min_period) ** fraction
    sinusoid_input = jnp.einsum(
        "i,j->ij",
        pos,
        1.0 / period * 2 * jnp.pi,
        precision=jax.lax.Precision.HIGHEST,
    )
    return jnp.concatenate([jnp.sin(sinusoid_input), jnp.cos(sinusoid_input)], axis=-1)


class Pi0(_model.BaseModel):
    def __init__(self, config: pi0_config.Pi0Config, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        self.pi05 = config.pi05
        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)
        # TODO: rewrite gemma in NNX. For now, use bridge.
        llm = nnx_bridge.ToNNX(
            _gemma.Module(
                configs=[paligemma_config, action_expert_config],
                embed_dtype=config.dtype,
                adarms=config.pi05,
            )
        )
        llm.lazy_init(rngs=rngs, method="init", use_adarms=[False, True] if config.pi05 else [False, False])
        img = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=paligemma_config.width,
                variant="So400m/14",
                pool_type="none",
                scan=True,
                dtype_mm=config.dtype,
            )
        )
        img.lazy_init(next(iter(config.fake_obs().images.values())), train=False, rngs=rngs)
        self.PaliGemma = nnx.Dict(llm=llm, img=img)
        self.action_in_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
        if config.pi05:
            self.time_mlp_in = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        else:
            self.state_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_in = nnx.Linear(2 * action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        self.action_out_proj = nnx.Linear(action_expert_config.width, config.action_dim, rngs=rngs)

        # This attribute gets automatically set by model.train() and model.eval().
        self.deterministic = True

    @at.typecheck
    def embed_prefix(
        self, obs: _model.Observation
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:
        tokens, input_mask, ar_mask, _ = self.embed_prefix_with_spans(obs)
        return tokens, input_mask, ar_mask

    def embed_prefix_with_spans(
        self, obs: _model.Observation
    ) -> tuple[
        at.Float[at.Array, "b s emb"],
        at.Bool[at.Array, "b s"],
        at.Bool[at.Array, " s"],
        dict[str, tuple[int, int]],
    ]:
        input_mask = []
        ar_mask = []
        tokens = []
        spans: dict[str, tuple[int, int]] = {}
        cursor = 0
        # embed images
        for name in obs.images:
            image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False)

            tokens.append(image_tokens)
            input_mask.append(
                einops.repeat(
                    obs.image_masks[name],
                    "b -> b s",
                    s=image_tokens.shape[1],
                )
            )
            # image tokens attend to each other
            ar_mask += [False] * image_tokens.shape[1]
            spans[f"image:{name}"] = (cursor, cursor + image_tokens.shape[1])
            cursor += image_tokens.shape[1]
        spans["visual_tokens"] = (0, cursor)

        # add language (aka tokenized inputs)
        if obs.tokenized_prompt is not None:
            tokenized_inputs = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
            tokens.append(tokenized_inputs)
            input_mask.append(obs.tokenized_prompt_mask)
            # full attention between image and language inputs
            ar_mask += [False] * tokenized_inputs.shape[1]
            spans["language_tokens"] = (cursor, cursor + tokenized_inputs.shape[1])
            cursor += tokenized_inputs.shape[1]
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        spans["all_prefix_tokens"] = (0, cursor)
        return tokens, input_mask, ar_mask, spans

    @at.typecheck
    def embed_suffix(
        self, obs: _model.Observation, noisy_actions: _model.Actions, timestep: at.Float[at.Array, " b"]
    ) -> tuple[
        at.Float[at.Array, "b s emb"],
        at.Bool[at.Array, "b s"],
        at.Bool[at.Array, " s"],
        at.Float[at.Array, "b emb"] | None,
    ]:
        input_mask = []
        ar_mask = []
        tokens = []
        if not self.pi05:
            # add a single state token
            state_token = self.state_proj(obs.state)[:, None, :]
            tokens.append(state_token)
            input_mask.append(jnp.ones((obs.state.shape[0], 1), dtype=jnp.bool_))
            # image/language inputs do not attend to state or actions
            ar_mask += [True]

        action_tokens = self.action_in_proj(noisy_actions)
        # embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = posemb_sincos(timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0)
        if self.pi05:
            # time MLP (for adaRMS)
            time_emb = self.time_mlp_in(time_emb)
            time_emb = nnx.swish(time_emb)
            time_emb = self.time_mlp_out(time_emb)
            time_emb = nnx.swish(time_emb)
            action_expert_tokens = action_tokens
            adarms_cond = time_emb
        else:
            # mix timestep + action information using an MLP (no adaRMS)
            time_tokens = einops.repeat(time_emb, "b emb -> b s emb", s=self.action_horizon)
            action_time_tokens = jnp.concatenate([action_tokens, time_tokens], axis=-1)
            action_time_tokens = self.action_time_mlp_in(action_time_tokens)
            action_time_tokens = nnx.swish(action_time_tokens)
            action_time_tokens = self.action_time_mlp_out(action_time_tokens)
            action_expert_tokens = action_time_tokens
            adarms_cond = None
        tokens.append(action_expert_tokens)
        input_mask.append(jnp.ones(action_expert_tokens.shape[:2], dtype=jnp.bool_))
        # image/language/state inputs do not attend to action tokens
        ar_mask += [True] + ([False] * (self.action_horizon - 1))
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask, adarms_cond

    def consumes_state_continuously(self) -> bool:
        return not self.pi05

    def has_memory_tokens(self) -> bool:
        return False

    def compute_flow_prediction_from_tokens(
        self,
        prefix_tokens: at.Float[at.Array, "b p emb"],
        prefix_mask: at.Bool[at.Array, "b p"],
        prefix_ar_mask: at.Bool[at.Array, " p"],
        suffix_tokens: at.Float[at.Array, "b s emb"],
        suffix_mask: at.Bool[at.Array, "b s"],
        suffix_ar_mask: at.Bool[at.Array, " s"],
        adarms_cond: at.Float[at.Array, "b emb"] | None,
    ) -> at.Float[at.Array, "b ah ad"]:
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        attn_mask = make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1
        (prefix_out, suffix_out), _ = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens], mask=attn_mask, positions=positions, adarms_cond=[None, adarms_cond]
        )
        del prefix_out
        return self.action_out_proj(suffix_out[:, -self.action_horizon :])

    def compute_flow_prediction_from_xt(
        self,
        observation: _model.Observation,
        x_t: _model.Actions,
        time: at.Float[at.Array, " b"],
        *,
        prefix_tokens: at.Float[at.Array, "b p emb"] | None = None,
        prefix_mask: at.Bool[at.Array, "b p"] | None = None,
        prefix_ar_mask: at.Bool[at.Array, " p"] | None = None,
    ) -> at.Float[at.Array, "b ah ad"]:
        if prefix_tokens is None or prefix_mask is None or prefix_ar_mask is None:
            prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, time)
        return self.compute_flow_prediction_from_tokens(
            prefix_tokens,
            prefix_mask,
            prefix_ar_mask,
            suffix_tokens,
            suffix_mask,
            suffix_ar_mask,
            adarms_cond,
        )

    def compute_flow_loss_from_xt(
        self,
        observation: _model.Observation,
        x_t: _model.Actions,
        time: at.Float[at.Array, " b"],
        u_t: _model.Actions,
        *,
        prefix_tokens: at.Float[at.Array, "b p emb"] | None = None,
        prefix_mask: at.Bool[at.Array, "b p"] | None = None,
        prefix_ar_mask: at.Bool[at.Array, " p"] | None = None,
        return_pred: bool = False,
    ) -> at.Float[at.Array, "*b ah"] | tuple[at.Float[at.Array, "*b ah"], at.Float[at.Array, "b ah ad"]]:
        v_t = self.compute_flow_prediction_from_xt(
            observation,
            x_t,
            time,
            prefix_tokens=prefix_tokens,
            prefix_mask=prefix_mask,
            prefix_ar_mask=prefix_ar_mask,
        )
        loss = jnp.mean(jnp.square(v_t - u_t), axis=-1)
        if return_pred:
            return loss, v_t
        return loss

    @override
    def compute_loss(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ) -> at.Float[at.Array, "*b ah"]:
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)

        _, time, x_t, u_t = sample_flow_targets(noise_rng, time_rng, actions)
        return self.compute_flow_loss_from_xt(observation, x_t, time, u_t)

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
    ) -> _model.Actions:
        observation = _model.preprocess_observation(None, observation, train=False)
        # note that we use the convention more common in diffusion literature, where t=1 is noise and t=0 is the target
        # distribution. yes, this is the opposite of the pi0 paper, and I'm sorry.
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        # first fill KV cache with a forward pass of the prefix
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

        def step(carry):
            x_t, time = carry
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                observation, x_t, jnp.broadcast_to(time, batch_size)
            )
            # `suffix_attn_mask` is shape (b, suffix_len, suffix_len) indicating how the suffix tokens can attend to each
            # other
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            # `prefix_attn_mask` is shape (b, suffix_len, prefix_len) indicating how the suffix tokens can attend to the
            # prefix tokens
            prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            # `combined_mask` is shape (b, suffix_len, prefix_len + suffix_len) indicating how the suffix tokens (which
            # generate the queries) can attend to the full prefix + suffix sequence (which generates the keys and values)
            full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
            assert full_attn_mask.shape == (
                batch_size,
                suffix_tokens.shape[1],
                prefix_tokens.shape[1] + suffix_tokens.shape[1],
            )
            # `positions` is shape (b, suffix_len) indicating the positions of the suffix tokens
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=positions,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
            )
            assert prefix_out is None
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

            return x_t + dt * v_t, time + dt

        def cond(carry):
            _x_t, time = carry
            # robust to floating-point error
            return time >= -dt / 2

        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        return x_0

    def trace_actions(
        self,
        observation,
        noise,
        *,
        num_steps=10,
        selected=(0, 8, 17),
        reference_states=None,
        reference_times=None,
        scalar_only=False,
    ):
        """Return a free sampling path and three-layer diagnostics explicitly.

        Compile with static_argnames=("num_steps", "selected", "scalar_only").
        This research entry point must pass native-sampler parity on the actual
        checkpoint before use. It does not replace ordinary policy inference.
        """
        observation = _model.preprocess_observation(None, observation, train=False)
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        prefix, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        _, cache = self.PaliGemma.llm(
            [prefix, None],
            mask=make_attn_mask(prefix_mask, prefix_ar_mask),
            positions=jnp.cumsum(prefix_mask, axis=1) - 1,
        )

        if (reference_states is None) != (reference_times is None):
            raise ValueError("Shared-x diagnostics require both reference states and times")

        def step(carry, reference):
            x_t, time = carry
            if reference_states is not None:
                x_t, time = reference
            suffix, suffix_mask, suffix_ar_mask, cond = self.embed_suffix(
                observation, x_t, jnp.broadcast_to(time, batch_size)
            )
            mask = jnp.concatenate(
                [
                    einops.repeat(prefix_mask, "b p -> b s p", s=suffix.shape[1]),
                    make_attn_mask(suffix_mask, suffix_ar_mask),
                ],
                axis=-1,
            )
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1
            readout, trace = self.PaliGemma.llm(
                [None, suffix],
                positions=positions,
                mask=mask,
                adarms_cond=[None, cond],
                kv_cache=cache,
                selected=selected,
                method="trace_suffix",
            )
            velocity = self.action_out_proj(readout[:, -self.action_horizon :])
            if scalar_only:
                trace = {"layer_rms": hidden_group_rms(trace["hidden"], xp=jnp), "time": time}
            else:
                trace = {**trace, "readout": readout, "velocity": velocity, "x_t": x_t, "time": time}
            return (x_t + dt * velocity, time + dt), trace

        references = None if reference_states is None else (reference_states, reference_times)
        length = num_steps if references is None else reference_states.shape[0]
        (actions, _), trace = jax.lax.scan(step, (noise, 1.0), references, length=length)
        if scalar_only:
            return actions, trace
        return actions, {**trace, "prefix": prefix, "prefix_mask": prefix_mask}

    def sample_actions_with_scalars(self, observation, noise):
        """Campaign sampling with only 10x3x2 scalars retained per observation.

        The canonical sample_actions remains the independent parity reference.
        Unused attention/activation trace outputs are not returned from scan.
        """
        return self.trace_actions(observation, noise, scalar_only=True)

    def sample_actions_with_trajectory(self, observation, noise):
        """Hidden-only sampling candidate, independently exact-qualified on GPU.

        No probabilities, prefix embeddings or attention branch outputs leave
        the denoising scan. The old sample_actions and scalar sampler remain
        unchanged references. A caller must not use this recording candidate
        as a control action unless the native reference comparison is exact.
        """
        observation = _model.preprocess_observation(None, observation, train=False)
        if self.action_horizon != 10 or noise.shape != (observation.state.shape[0], 10, self.action_dim):
            raise ValueError("Native campaign trajectory requires per-request ten-token noise")
        dt = -1.0 / 10
        batch_size = observation.state.shape[0]
        prefix, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        _, cache = self.PaliGemma.llm(
            [prefix, None],
            mask=make_attn_mask(prefix_mask, prefix_ar_mask),
            positions=jnp.cumsum(prefix_mask, axis=1) - 1,
        )

        def step(carry, _unused):
            x_t, time = carry
            suffix, suffix_mask, suffix_ar_mask, cond = self.embed_suffix(
                observation, x_t, jnp.broadcast_to(time, batch_size)
            )
            mask = jnp.concatenate(
                [
                    einops.repeat(prefix_mask, "b p -> b s p", s=suffix.shape[1]),
                    make_attn_mask(suffix_mask, suffix_ar_mask),
                ],
                axis=-1,
            )
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1
            readout, trace = self.PaliGemma.llm(
                [None, suffix],
                positions=positions,
                mask=mask,
                adarms_cond=[None, cond],
                kv_cache=cache,
                selected=(0, 8, 17),
                capture_attention=False,
                method="trace_suffix",
            )
            velocity = self.action_out_proj(readout[:, -self.action_horizon :])
            recorded = {
                "x_t": x_t,
                "time": time,
                "velocity": velocity,
                "readout": readout,
                "hidden": trace["hidden"],
                "layer_rms": hidden_group_rms(trace["hidden"], xp=jnp),
            }
            return (x_t + dt * velocity, time + dt), recorded

        (actions, _), path = jax.lax.scan(step, (noise, 1.0), None, length=10)
        return actions, path

    def velocity_at(self, observation, x_t, time):
        """Unrecorded cached-prefix velocity reference for qualification only.

        Reuses the native prefix/suffix computation, with no H/probability
        capture and no full-sequence training path. A caller evaluates the ten
        recorder states separately, rather than tracing this into recording.
        """
        observation = _model.preprocess_observation(None, observation, train=False)
        batch_size = observation.state.shape[0]
        if x_t.shape != (batch_size, self.action_horizon, self.action_dim) or time.shape != ():
            raise ValueError("Unrecorded velocity requires one native BxQxA state and scalar loop time")
        prefix, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        _, cache = self.PaliGemma.llm(
            [prefix, None],
            mask=make_attn_mask(prefix_mask, prefix_ar_mask),
            positions=jnp.cumsum(prefix_mask, axis=1) - 1,
        )
        suffix, suffix_mask, suffix_ar_mask, cond = self.embed_suffix(
            observation, x_t, jnp.broadcast_to(time, batch_size)
        )
        mask = jnp.concatenate(
            [
                einops.repeat(prefix_mask, "b p -> b s p", s=suffix.shape[1]),
                make_attn_mask(suffix_mask, suffix_ar_mask),
            ],
            axis=-1,
        )
        positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1
        (prefix_out, suffix_out), _ = self.PaliGemma.llm(
            [None, suffix], mask=mask, positions=positions, kv_cache=cache, adarms_cond=[None, cond]
        )
        assert prefix_out is None
        return self.action_out_proj(suffix_out[:, -self.action_horizon :])

    def trace_at_native_states(self, observation, states, times):
        """Instrumented shadow at explicit native inputs; never integrates v_t.

        This is a numerical re-evaluation, not the original inference graph.
        Its hidden/readout/velocity may differ and are never control actions.
        All times are supplied loop values, rather than a reconstructed grid.
        """
        observation = _model.preprocess_observation(None, observation, train=False)
        batch_size = observation.state.shape[0]
        if states.ndim != 4 or states.shape[1:] != (batch_size, self.action_horizon, self.action_dim):
            raise ValueError("Shadow states must match the native time/batch/action layout")
        if times.shape != (states.shape[0],):
            raise ValueError("Shadow times must match every supplied native state")
        prefix, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        _, cache = self.PaliGemma.llm(
            [prefix, None],
            mask=make_attn_mask(prefix_mask, prefix_ar_mask),
            positions=jnp.cumsum(prefix_mask, axis=1) - 1,
        )

        def step(_unused, reference):
            x_t, time = reference
            suffix, suffix_mask, suffix_ar_mask, cond = self.embed_suffix(
                observation, x_t, jnp.broadcast_to(time, batch_size)
            )
            mask = jnp.concatenate(
                [
                    einops.repeat(prefix_mask, "b p -> b s p", s=suffix.shape[1]),
                    make_attn_mask(suffix_mask, suffix_ar_mask),
                ],
                axis=-1,
            )
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1
            readout, trace = self.PaliGemma.llm(
                [None, suffix],
                positions=positions,
                mask=mask,
                adarms_cond=[None, cond],
                kv_cache=cache,
                selected=(0, 8, 17),
                method="trace_suffix",
            )
            velocity = self.action_out_proj(readout[:, -self.action_horizon :])
            # No dt/update or action endpoint is computed in this sidepass.
            return None, {**trace, "readout": readout, "velocity": velocity, "x_t": x_t, "time": time}

        _, trace = jax.lax.scan(step, None, (states, times))
        return {**trace, "prefix": prefix, "prefix_mask": prefix_mask}

    def trace_pair_on_native_states(
        self, clean_observation, actual_observation, clean_path, actual_path, *, save_visual=False, save_attention=False
    ):
        """Mixed-provenance summary after separate native recording calls.

        Own-path H/readout/velocity are native-recorded values. Probabilities,
        visual references and attention branch outputs are shadow values.
        Shared-x/t H is a shadow counterfactual, with both conditions evaluated
        using the same three-tau ABI. Never JIT native recording into this call.
        """
        if clean_path["x_t"].shape[0] != 10 or actual_path["x_t"].shape[0] != 10:
            raise ValueError("Paired campaign diagnostics require ten native denoising states")
        clean = self.trace_at_native_states(clean_observation, clean_path["x_t"], clean_path["time"])
        actual = self.trace_at_native_states(actual_observation, actual_path["x_t"], actual_path["time"])
        # Selected native features replace the shadow feature families. The
        # visual/attention families retain explicitly instrumented provenance.
        native_clean = {**clean, **{key: clean_path[key] for key in ("hidden", "readout", "velocity")}}
        native_actual = {**actual, **{key: actual_path[key] for key in ("hidden", "readout", "velocity")}}
        summary = summarize_pair(
            native_clean, native_actual, save_visual=save_visual, save_attention=save_attention, xp=jnp
        )
        summary["hidden_actual_rms"] = actual_path["layer_rms"]
        indices = jnp.array([0, 5, 9])
        shared_states, shared_times = clean_path["x_t"][indices], clean_path["time"][indices]
        shared_clean = self.trace_at_native_states(clean_observation, shared_states, shared_times)
        shared_actual = self.trace_at_native_states(actual_observation, shared_states, shared_times)
        shared_summary = summarize_pair(shared_clean, shared_actual, xp=jnp)
        # The existing schema uses the ten-tau clean baseline in the shared
        # branch. Do not claim it equals the separately reevaluated 3-tau clean
        # baseline: preserve that shadow baseline and its error explicitly.
        clean_subset = {key: value[indices] for key, value in clean_path.items()}
        return {
            "free": summary,
            "shared_xt": shared_summary,
            "times": clean_path["time"],
            "shared_indices": indices,
            "shadow_error": {
                "clean": shadow_recompute_error(clean_path, clean, xp=jnp),
                "actual": shadow_recompute_error(actual_path, actual, xp=jnp),
                "shared_clean": shadow_recompute_error(clean_subset, shared_clean, xp=jnp),
            },
        }

    def trace_pair(self, clean_observation, actual_observation, noise, *, save_visual=False, save_attention=False):
        """Paired diagnostics reduced on device; static flags select archive size.

        This entry point is fixed to the campaign's 10 steps and blocks 1/9/18.
        Compile with static_argnames=("save_visual", "save_attention").
        """
        clean_actions, clean = self.trace_actions(clean_observation, noise)
        actual_actions, actual = self.trace_actions(actual_observation, noise)
        summary = summarize_pair(clean, actual, save_visual=save_visual, save_attention=save_attention, xp=jnp)
        indices = jnp.array([0, 5, 9])
        _, shared = self.trace_actions(
            actual_observation,
            noise,
            reference_states=clean["x_t"][indices],
            reference_times=clean["time"][indices],
        )
        clean_subset = {
            key: value if key in ("prefix", "prefix_mask") else value[indices] for key, value in clean.items()
        }
        shared_summary = summarize_pair(clean_subset, shared, xp=jnp)
        # Full clean activations are already retained for all ten free steps.
        for key in ("hidden_clean", "readout_clean", "velocity_clean"):
            shared_summary.pop(key)
        return {
            "clean_actions": clean_actions,
            "actual_actions": actual_actions,
            "free": summary,
            "shared_xt": shared_summary,
            "times": clean["time"],
            "shared_indices": indices,
        }
