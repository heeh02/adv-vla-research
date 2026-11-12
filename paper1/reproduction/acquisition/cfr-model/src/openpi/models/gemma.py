# Copyright 2024 Big Vision Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Gemma adaptation for Pi, taken from big_vision.

We follow this einsum axis naming convention:
  B: batch
  T: query length
  S: k/v length
  N: num query heads
  K: num k/v heads
  G: num query heads per k/v head
  H: head dim
  D: d_model ("features")
"""

from collections.abc import Sequence
import dataclasses
from typing import Literal, TypeAlias

import einops
import flax.linen as nn
import jax
import jax.numpy as jnp

import openpi.models.lora as lora
import openpi.shared.array_typing as at
import openpi.training.sharding as sharding

PALIGEMMA_VOCAB_SIZE = 257_152


@dataclasses.dataclass
class Config:
    width: int
    depth: int
    mlp_dim: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    lora_configs: dict[str, lora.LoRAConfig] = dataclasses.field(default_factory=dict)


Variant = Literal["dummy", "gemma_300m", "gemma_300m_lora", "gemma_2b", "gemma_2b_lora"]


def get_config(variant: Variant) -> Config:
    """Returns config for specified gemma variant."""
    if variant == "dummy":
        return Config(
            width=64,
            depth=4,
            mlp_dim=128,
            num_heads=8,
            num_kv_heads=1,
            head_dim=16,
        )
    if variant == "gemma_300m":
        # 311M params
        return Config(
            width=1024,
            depth=18,
            mlp_dim=4096,
            num_heads=8,
            num_kv_heads=1,
            head_dim=256,
        )
    if variant == "gemma_2b":
        return Config(
            width=2048,
            depth=18,
            mlp_dim=16_384,
            num_heads=8,
            num_kv_heads=1,
            head_dim=256,
        )
    if variant == "gemma_2b_lora":
        return Config(
            width=2048,
            depth=18,
            mlp_dim=16_384,
            num_heads=8,
            num_kv_heads=1,
            head_dim=256,
            lora_configs={"attn": lora.LoRAConfig(rank=16, alpha=16.0), "ffn": lora.LoRAConfig(rank=16, alpha=16.0)},
        )
    if variant == "gemma_300m_lora":
        # 311M params
        return Config(
            width=1024,
            depth=18,
            mlp_dim=4096,
            num_heads=8,
            num_kv_heads=1,
            head_dim=256,
            lora_configs={"attn": lora.LoRAConfig(rank=32, alpha=32.0), "ffn": lora.LoRAConfig(rank=32, alpha=32.0)},
        )
    raise ValueError(f"Unknown variant: {variant}")


@at.typecheck
class RMSNorm(nn.Module):
    @nn.compact
    def __call__(self, x, cond):
        dtype = x.dtype  # original dtype, could be half-precision
        var = jnp.mean(jnp.square(x.astype(jnp.float32)), axis=-1, keepdims=True)  # compute variance in float32
        normed_inputs = jnp.asarray(x * jnp.reciprocal(jnp.sqrt(var + 1e-06)))  # compute normalization in float32
        if cond is None:
            # regular RMSNorm
            scale = self.param("scale", nn.initializers.zeros_init(), (x.shape[-1]))
            normed_inputs = normed_inputs * (
                1 + scale
            )  # scale by learned parameter in float32 (matches Flax implementation)
            return normed_inputs.astype(dtype), None  # return in original dtype

        # adaptive RMSNorm
        modulation = nn.Dense(x.shape[-1] * 3, kernel_init=nn.initializers.zeros, dtype=dtype)(cond)
        scale, shift, gate = jnp.split(modulation[:, None, :], 3, axis=-1)
        normed_inputs = normed_inputs * (1 + scale) + shift  # scale and shift in float32
        return normed_inputs.astype(dtype), gate


@at.typecheck
class Embedder(nn.Module):
    """Embedder module."""

    vocab_size: int
    embed_dim: int

    def setup(self):
        self.input_embedding_table = self.param(
            "input_embedding",
            nn.initializers.normal(),
            (self.vocab_size, self.embed_dim),
        )

    def encode(self, x):
        x = self.input_embedding_table[(x,)]
        x *= jnp.sqrt(self.embed_dim).astype(x.dtype)
        return x

    def decode(self, x):
        return jnp.dot(x, self.input_embedding_table.T)


@at.typecheck
class Attention(nn.Module):
    """Attention module."""

    configs: Sequence[Config]

    @nn.compact
    def __call__(self, xs, positions, attn_mask, kv_cache, *, capture=False, capture_operands=False):
        # all experts must share the same head dim, num heads, and num kv heads for self-attention to work
        assert all(config.head_dim == self.configs[0].head_dim for config in self.configs)
        assert all(config.num_heads == self.configs[0].num_heads for config in self.configs)
        assert all(config.num_kv_heads == self.configs[0].num_kv_heads for config in self.configs)

        dtype = next(x.dtype for x in xs if x is not None)  # original dtype, could be half-precision

        qkvs = []
        for i, (x, config) in enumerate(zip(xs, self.configs, strict=True)):
            if x is None:
                continue
            if config.num_kv_heads == config.num_heads:
                qkv_einsum = lora.Einsum(
                    shape=(3, config.num_heads, config.width, config.head_dim),
                    name=_name("qkv_einsum", i),
                    init_fn=nn.initializers.lecun_normal(in_axis=-2, out_axis=-1, batch_axis=(0, 1)),
                    lora_config=config.lora_configs.get("attn"),
                )
                qkvs.append(qkv_einsum("BSD,3KDH->3BSKH", x))
            else:
                q_einsum = lora.Einsum(
                    shape=(config.num_heads, config.width, config.head_dim),
                    name=_name("q_einsum", i),
                    init_fn=nn.initializers.lecun_normal(in_axis=-2, out_axis=-1, batch_axis=(0,)),
                    lora_config=config.lora_configs.get("attn"),
                )
                q = q_einsum("BTD,NDH->BTNH", x)
                kv_einsum = lora.Einsum(
                    shape=(2, config.num_kv_heads, config.width, config.head_dim),
                    name=_name("kv_einsum", i),
                    init_fn=nn.initializers.lecun_normal(in_axis=-2, out_axis=-1, batch_axis=(0, 1)),
                    lora_config=config.lora_configs.get("attn"),
                )
                k, v = kv_einsum("BSD,2KDH->2BSKH", x)
                qkvs.append((q, k, v))

        q, k, v = (jnp.concatenate(y, axis=1) for y in zip(*qkvs, strict=True))

        q = _apply_rope(q, positions=positions)
        q *= self.configs[0].head_dim ** -0.5

        k = _apply_rope(k, positions=positions)

        # should still be half-precision here (if input was half-precision)
        assert q.dtype == k.dtype == v.dtype == dtype

        if capture_operands:
            if kv_cache is None or xs[0] is not None or xs[1] is None:
                raise ValueError("SRU operand capture requires cached action-only input")
            suffix_key, suffix_value = k, v
        if kv_cache is not None:
            cache_k, cache_v = kv_cache
            k = jnp.concatenate([cache_k, k], axis=1)
            v = jnp.concatenate([cache_v, v], axis=1)

        q = einops.rearrange(q, "B T (K G) H -> B T K G H", K=self.configs[0].num_kv_heads)
        logits = jnp.einsum("BTKGH,BSKH->BKGTS", q, k, preferred_element_type=jnp.float32)

        if attn_mask.shape != (q.shape[0], 1, q.shape[1], k.shape[1]):
            raise ValueError(
                f"Attention mask with shape {attn_mask.shape} but shapes for q and k are: {q.shape} and {k.shape}"
            )

        # big_neg = jnp.finfo(logits.dtype).min
        big_neg = -2.3819763e38  # See gemma/modules.py
        masked_logits = jnp.where(attn_mask[:, :, None, :, :], logits, big_neg)

        probs = jax.nn.softmax(masked_logits, axis=-1).astype(dtype)

        encoded = jnp.einsum("BKGTS,BSKH->BTKGH", probs, v)
        encoded = einops.rearrange(encoded, "B T K G H -> B T (K G) H")

        out = []
        start = 0
        for i, (x, config) in enumerate(zip(xs, self.configs, strict=True)):
            if x is not None:
                end = start + x.shape[1]
                out_einsum = lora.Einsum(
                    shape=(config.num_heads, config.head_dim, config.width),
                    name=_name("attn_vec_einsum", i),
                    init_fn=nn.initializers.lecun_normal(in_axis=(-3, -2), out_axis=-1),
                    lora_config=config.lora_configs.get("attn"),
                )
                out.append(out_einsum("BTNH,NHD->BTD", encoded[:, start:end]))
                start = end
            else:
                out.append(None)

        if capture_operands:
            return out, (k, v), {
                "query_post_rope_scaled": q, "prefix_key_used": cache_k, "prefix_value_used": cache_v,
                "suffix_key_post_rope": suffix_key, "suffix_value": suffix_value,
                "probabilities_native": probs,
                "attention_encoded_pre_WO": encoded,
                "attention_output_post_WO_pre_gate": out[1],
            }
        if capture:
            return out, (k, v), probs
        return out, (k, v)


@at.typecheck
class FeedForward(nn.Module):
    """Feed forward module."""

    features: int
    hidden_dim: int

    @nn.compact
    def __call__(self, x):
        dtype = x.dtype  # original dtype, could be half-precision
        w_gating = self.param(
            "gating_einsum",
            nn.initializers.lecun_normal(in_axis=-2, out_axis=-1, batch_axis=(0,)),
            (2, self.features, self.hidden_dim),
        ).astype(dtype)
        ff_gate = jnp.dot(x, w_gating[0])
        gate_value = nn.gelu(ff_gate)

        ff1 = jnp.dot(x, w_gating[1])
        activations = gate_value * ff1

        w_linear = self.param(
            "linear",
            nn.initializers.lecun_normal(in_axis=-2, out_axis=-1),
            (self.hidden_dim, self.features),
        ).astype(dtype)
        outputs = jnp.dot(activations, w_linear)
        assert outputs.dtype == dtype
        return outputs


@at.typecheck
class Block(nn.Module):
    """Transformer block."""

    configs: tuple[Config, ...]

    dropout: float = 0.0
    dropout_bdims: tuple[int, ...] = ()

    @nn.compact
    def __call__(
        self, xs, kv_cache, positions, attn_mask, adarms_cond, deterministic=True, *,  # noqa: FBT002
        capture=False, capture_prefix_interface=False, sru_capture=False,
    ):
        xs = sharding.activation_sharding_constraint(xs)
        if sru_capture:
            if capture_prefix_interface or capture:
                raise ValueError("SRU cannot combine with another capture")
            sru_residual_input = xs[1]
        if capture_prefix_interface:
            if not xs or xs[0] is None:
                raise ValueError("Prefix interface capture requires expert 0 input")
            prefix_hidden_input = xs[0]
        drop = nn.Dropout(self.dropout, self.dropout_bdims) if self.dropout else lambda x, _: x

        attn = Attention(configs=self.configs, name="attn")

        pre_attn = []
        gates = []
        for i, x in enumerate(xs):
            if x is not None:
                x, gate = RMSNorm(name=_name("pre_attention_norm", i))(x, adarms_cond[i])  # noqa: PLW2901
            pre_attn.append(x)
            gates.append(gate if x is not None else None)

        pre_attn = sharding.activation_sharding_constraint(pre_attn)
        if sru_capture:
            sru_gate = gates[1]
            attention_result = attn(pre_attn, positions, attn_mask, kv_cache, capture_operands=True)
        else:
            attention_result = attn(pre_attn, positions, attn_mask, kv_cache, capture=capture)
        post_attn, kv_cache = attention_result[:2]
        post_attn = jax.tree.map(lambda x: drop(x, deterministic), post_attn)
        post_attn = sharding.activation_sharding_constraint(post_attn)
        xs = [_gated_residual(x, y, gate) for x, y, gate in zip(xs, post_attn, gates, strict=True)]
        if sru_capture:
            sru_after_attention = xs[1]
        xs = sharding.activation_sharding_constraint(xs)

        out = []
        gates = []
        for i, (x, config) in enumerate(zip(xs, self.configs, strict=True)):
            if x is not None:
                x, gate = RMSNorm(name=_name("pre_ffw_norm", i))(x, adarms_cond[i])  # noqa: PLW2901
                x = lora.FeedForward(  # noqa: PLW2901
                    features=config.width,
                    hidden_dim=config.mlp_dim,
                    name=_name("mlp", i),
                    lora_config=config.lora_configs.get("ffn"),
                )(x)
            out.append(x)
            gates.append(gate if x is not None else None)

        out = sharding.activation_sharding_constraint(out)
        out = jax.tree.map(lambda x: drop(x, deterministic), out)
        xs = [_gated_residual(x, y, gate) for x, y, gate in zip(xs, out, gates, strict=True)]
        xs = sharding.activation_sharding_constraint(xs)

        if sru_capture:
            return xs, kv_cache, {**attention_result[2],
                "receiver_residual_input": sru_residual_input,
                "receiver_pre_attention_norm": pre_attn[1], "attention_gate": sru_gate,
                "hidden_after_attention_residual": sru_after_attention, "hidden_post_block": xs[1]}
        if capture:
            trace = {"probabilities": attention_result[2], "attention_output": post_attn}
            if capture_prefix_interface:
                trace.update(hidden_input=prefix_hidden_input, hidden_pre_attention_norm=pre_attn[0])
            return xs, kv_cache, trace
        if capture_prefix_interface:
            # Return the very tensor fed to Attention, without another norm.
            # Default return structures and parameter collections stay intact.
            return xs, kv_cache, {
                "hidden_input": prefix_hidden_input,
                "hidden_pre_attention_norm": pre_attn[0],
            }
        return xs, kv_cache


SCAN_NATIVE_FIELDS = ('receiver_residual_input', 'hidden_after_attention_residual', 'hidden_post_block', 'query_post_rope_scaled', 'probabilities_native', 'attention_output_post_WO_pre_gate', 'attention_gate', 'prefix_key_used', 'prefix_value_used', 'suffix_key_post_rope', 'suffix_value')


class ScanNativeCaptureBlock(Block):
    """Same parameter scope/math; xs is carry, (cache, trace) is scan output."""

    @nn.compact
    def __call__(self, xs, kv_cache, positions, attn_mask, adarms_cond, deterministic=True):
        xs, next_cache, trace = super().__call__(
            xs, kv_cache, positions, attn_mask, adarms_cond, deterministic,
            sru_capture=True,
        )
        return xs, (next_cache, {name: trace[name] for name in SCAN_NATIVE_FIELDS})


class PrefixCaptureBlock(Block):
    """Original PG math/scopes; only xs is carry, cache and H/Z are outputs."""

    @nn.compact
    def __call__(self, xs, kv_cache, positions, attn_mask, adarms_cond, deterministic=True):
        xs, next_cache, trace = super().__call__(
            xs, kv_cache, positions, attn_mask, adarms_cond, deterministic,
            capture_prefix_interface=True,
        )
        return xs, (next_cache, trace)


KVCache: TypeAlias = tuple[at.Float[at.Array, "l b _t _k _h"], at.Float[at.Array, "l b _t _v _h"]]


@at.typecheck
class Module(nn.Module):
    """Transformer model, supporting a mixture of different weights for different tokens."""

    configs: Sequence[Config]  # list of configs, one for each expert
    embed_dtype: str

    dropout: float = 0.0
    dropout_bdims: tuple[int, ...] = ()  # Every float is dropped independently.
    adarms: bool = False

    def setup(self):
        # all experts must have the same depth
        assert all(config.depth == self.configs[0].depth for config in self.configs)

        self.embedder = Embedder(
            vocab_size=PALIGEMMA_VOCAB_SIZE,
            embed_dim=self.configs[0].width,  # embedder for first expert only
            name="embedder",
        )
        block_cls = nn.remat(
            Block,
            prevent_cse=False,
            static_argnums=(5,),  # 0=self, 6=deterministic
            policy=jax.checkpoint_policies.nothing_saveable,
        )
        self.layers = nn.scan(
            block_cls,
            variable_axes={"params": 0},
            split_rngs={"params": True, "dropout": True},
            in_axes=(
                0,
                nn.broadcast,
                nn.broadcast,
                nn.broadcast,
                nn.broadcast,
            ),  # 0=kv_cache, 1=positions, 2=mask, 3=adarms_cond, 4=deterministic
            length=self.configs[0].depth,
        )(
            configs=self.configs,
            dropout=self.dropout,
            dropout_bdims=self.dropout_bdims,
        )
        self.final_norms = [RMSNorm(name=_name("final_norm", i)) for i in range(len(self.configs))]

    @at.typecheck
    def embed(self, tokens: at.Int[at.Array, "b t"]) -> at.Float[at.Array, "b t d"]:
        return self.embedder.encode(tokens).astype(self.embed_dtype)

    @at.typecheck
    def __call__(
        self,
        # list of token arrays, one for each expert, or None if that expert should not be run
        embedded: Sequence[at.Float[at.Array, "b _t _d"] | None],
        positions: at.Int[at.Array, "b t"],
        mask: at.Bool[at.Array, "b t s"],
        adarms_cond: Sequence[at.Float[at.Array, "b _d"] | None] | None = None,
        *,
        kv_cache: KVCache | None = None,
        deterministic: bool = True,
    ) -> tuple[Sequence[at.Float[at.Array, "b _t _d"] | None], KVCache]:
        embedded = jax.tree.map(lambda e: e.astype(self.embed_dtype), embedded)
        mask = jnp.asarray(mask)[:, None, :, :]
        if adarms_cond is None:
            adarms_cond = [None] * len(self.configs)

        embedded, kv_cache = self.layers(embedded, kv_cache, positions, mask, adarms_cond, deterministic)

        assert all(e.dtype == jnp.dtype(self.embed_dtype) for e in embedded if e is not None)

        return [
            f(e, a)[0] if e is not None else e for f, e, a in zip(self.final_norms, embedded, adarms_cond, strict=True)
        ], kv_cache

    def trace_suffix(
        self, embedded, positions, mask, adarms_cond, *, kv_cache, selected=(0, 8, 17), capture_attention=True
    ):
        """Inference-only explicit outputs at selected action-expert blocks.

        Reuses the original Block and its exact scanned parameters. The scan
        carries only selected tensors, never a stack of all-layer attention.
        Normal inference and the checkpoint parameter tree remain unchanged.
        The hidden-only candidate uses the original capture=False Block call
        and never returns attention probabilities or branch outputs. Actual
        checkpoint/action/scalar parity is required before calling it native.
        """
        if len(embedded) != 2 or embedded[0] is not None or embedded[1] is None:
            raise ValueError("trace_suffix requires cached prefix and action expert only")
        if self.dropout:
            raise ValueError("Trace parity requires deterministic inference without dropout")
        depth = self.configs[0].depth
        if not selected or tuple(sorted(set(selected))) != tuple(selected) or selected[-1] >= depth or selected[0] < 0:
            raise ValueError("Selected layers must be sorted unique valid zero-based indices")
        embedded = jax.tree.map(lambda x: x.astype(self.embed_dtype), embedded)
        x = embedded[1]
        config = self.configs[1]
        slots = len(selected)
        hidden = jnp.zeros((slots, *x.shape), x.dtype)
        probabilities = (
            jnp.zeros(
                (
                    slots,
                    x.shape[0],
                    config.num_kv_heads,
                    config.num_heads // config.num_kv_heads,
                    x.shape[1],
                    mask.shape[-1],
                ),
                x.dtype,
            )
            if capture_attention
            else None
        )
        branch = jnp.zeros_like(hidden) if capture_attention else None
        block = Block(configs=self.configs, parent=None)
        params = self.variables["params"]["layers"]
        selected_array = jnp.asarray(selected)

        def step(carry, inputs):
            xs, h, p, a = carry
            index, layer_params, cache = inputs
            output = block.apply(
                {"params": layer_params},
                xs,
                cache,
                positions,
                mask[:, None],
                adarms_cond,
                deterministic=True,
                capture=capture_attention,
            )
            xs = output[0]
            slot = jnp.argmax(selected_array == index)

            def save(buffers):
                h, p, a = buffers
                if not capture_attention:
                    return h.at[slot].set(xs[1]), None, None
                trace = output[2]
                return (
                    h.at[slot].set(xs[1]),
                    p.at[slot].set(trace["probabilities"]),
                    a.at[slot].set(trace["attention_output"][1]),
                )

            h, p, a = jax.lax.cond(jnp.any(selected_array == index), save, lambda z: z, (h, p, a))
            return (xs, h, p, a), None

        (xs, hidden, probabilities, branch), _ = jax.lax.scan(
            step, (embedded, hidden, probabilities, branch), (jnp.arange(depth), params, kv_cache)
        )
        readout = self.final_norms[1](xs[1], adarms_cond[1])[0]
        trace = {"hidden": hidden}
        if capture_attention:
            trace.update(probabilities=probabilities, attention_output=branch)
        return readout, trace


    def ablate_suffix(self, embedded, positions, mask, adarms_cond, *, kv_cache, basis, mean, dose):
        """L9 post-block action H edit; complete L10..L18 native continuation.

        New graph: canonical parity must be measured with the actual checkpoint.
        ``dose==0`` returns the original BF16 H without an FP32 round trip.
        """
        if (len(embedded) != 2 or embedded[0] is not None or embedded[1] is None
                or self.dropout or self.configs[0].depth != 18):
            raise ValueError("H ablation requires deterministic action-only depth18")
        embedded = jax.tree.map(lambda x: x.astype(self.embed_dtype), embedded)
        h = embedded[1]
        if (h.ndim != 3 or basis.shape != (h.shape[-1], 16)
                or mean.shape != h.shape[1:] or h.shape[1] != 10):
            raise ValueError("H ablation requires B[d,16], mean[10,d], all ten action tokens")
        if jnp.shape(dose) != ():
            raise ValueError("H ablation dose must be scalar")
        block = Block(configs=self.configs, parent=None)
        params = self.variables["params"]["layers"]
        empty = jnp.zeros_like(h)

        def step(carry, inputs):
            xs, h_before, h_after, l10_out, l18_out = carry
            index, layer_params, cache = inputs
            xs, _ = block.apply(
                {"params": layer_params}, xs, cache, positions,
                jnp.asarray(mask)[:, None, :, :], adarms_cond, True,
            )
            # Capture L10 post-block output as proof that the edited L9 H
            # continues through the following action-expert computation.
            l10_out = jax.lax.cond(index == 9, lambda: xs[1], lambda: l10_out)
            l18_out = jax.lax.cond(index == 17, lambda: xs[1], lambda: l18_out)

            def edit(x):
                before = x[1]
                def project():
                    hf = before.astype(jnp.float32)
                    bf = basis.astype(jnp.float32)
                    mf = mean.astype(jnp.float32)
                    coeff = jnp.matmul(hf - mf[None, ...], bf)
                    delta = jnp.matmul(coeff, jnp.swapaxes(bf, 0, 1))
                    return (hf - jnp.asarray(dose, jnp.float32) * delta).astype(before.dtype)
                after = jax.lax.cond(jnp.asarray(dose) == 0, lambda: before, project)
                return [None, after], before, after

            xs, before, after = jax.lax.cond(
                index == 8, edit,
                lambda x: (x, h_before, h_after), xs,
            )
            h_before = jax.lax.cond(index == 8, lambda: before, lambda: h_before)
            h_after = jax.lax.cond(index == 8, lambda: after, lambda: h_after)
            return (xs, h_before, h_after, l10_out, l18_out), None

        (xs, before, after, l10_out, l18_out), _ = jax.lax.scan(
            step, (embedded, empty, empty, empty, empty),
            (jnp.arange(18), params, kv_cache),
        )
        readout = self.final_norms[1](xs[1], adarms_cond[1])[0]
        return readout, {"L9_H_before": before, "L9_H_after": after,
                         "L10_H_after": l10_out, "L18_H_after": l18_out}


    def scan_native_suffix(self, embedded, positions, mask, adarms_cond, *, kv_cache):
        """Unqualified measurement graph. No trace carry or per-layer scatter.

        Same nn.scan(nn.remat(Block)) settings and actual layers parameters as
        stock; selected traces are scan outputs. BF16 native P only. Added
        outputs can still change XLA scheduling: real native parity is pending.
        """
        if (len(embedded) != 2 or embedded[0] is not None or embedded[1] is None
                or self.dropout or self.configs[0].depth != 18):
            raise ValueError("scan-native requires deterministic action-only depth18")
        embedded = jax.tree.map(lambda x: x.astype(self.embed_dtype), embedded)
        block_cls = nn.remat(
            ScanNativeCaptureBlock,
            prevent_cse=False,
            static_argnums=(5,),
            policy=jax.checkpoint_policies.nothing_saveable,
        )
        scanned = nn.scan(
            block_cls,
            variable_axes={"params": 0},
            split_rngs={"params": True, "dropout": True},
            in_axes=(0, nn.broadcast, nn.broadcast, nn.broadcast, nn.broadcast),
            length=self.configs[0].depth,
        )(configs=self.configs, dropout=self.dropout, dropout_bdims=self.dropout_bdims, parent=None)
        # Bind existing stacked weights directly. No new scope, copy, setter,
        # initialization, renamed parameter, or all-layer trace carry.
        xs, (_unused_next_cache, all_trace) = scanned.apply(
            {"params": self.variables["params"]["layers"]},
            embedded, kv_cache, positions, jnp.asarray(mask)[:, None, :, :], adarms_cond, True,
        )
        selected = jnp.asarray((0, 8, 17), jnp.int32)
        trace = jax.tree.map(lambda value: jnp.take(value, selected, axis=0), all_trace)
        trace.update(mask=mask, suffix_positions=positions,
                     adarms_condition=adarms_cond[1], query_defined=mask.any(-1))
        return self.final_norms[1](xs[1], adarms_cond[1])[0], trace


    def scan_native_prefix(self, embedded, positions, mask):
        """PG H/Z/K/V outputs using the stock scan/remat and existing weights.

        New measurement graph: no native parity is granted by source reuse.
        No prefix-prefix attention probabilities or recomputed normalization.
        """
        if (len(embedded) != 2 or embedded[0] is None or embedded[1] is not None
                or self.dropout or self.configs[0].depth != 18):
            raise ValueError("scan-native PG requires deterministic prefix-only depth18")
        embedded = jax.tree.map(lambda x: x.astype(self.embed_dtype), embedded)
        block_cls = nn.remat(
            PrefixCaptureBlock,
            prevent_cse=False,
            static_argnums=(5,),
            policy=jax.checkpoint_policies.nothing_saveable,
        )
        scanned = nn.scan(
            block_cls,
            variable_axes={"params": 0},
            split_rngs={"params": True, "dropout": True},
            in_axes=(0, nn.broadcast, nn.broadcast, nn.broadcast, nn.broadcast),
            length=self.configs[0].depth,
        )(configs=self.configs, dropout=self.dropout, dropout_bdims=self.dropout_bdims, parent=None)
        xs, (cache, all_trace) = scanned.apply(
            {"params": self.variables["params"]["layers"]},
            embedded, None, positions, jnp.asarray(mask)[:, None, :, :], [None, None], True,
        )
        selected = jnp.asarray((0, 8, 17), jnp.int32)
        trace = jax.tree.map(lambda value: jnp.take(value, selected, axis=0), all_trace)
        trace.update(cache_k=jnp.take(cache[0], selected, axis=0),
                     cache_v=jnp.take(cache[1], selected, axis=0),
                     selected_layers=selected, positions=positions, mask=mask,
                     query_defined=mask.any(-1), key_visible=mask.any(-2))
        return self.final_norms[0](xs[0], None)[0], cache, trace


    def cfr_scan_suffix(self, embedded, positions, mask, adarms_cond, *, kv_cache, layers,
                        edit_at=None, edit_inputs=None):
        """cf-repair-v1: action-expert per-layer Block.apply scan over the first ``layers`` layers.

        Adapted from ``ablate_suffix`` (same Block.apply binding of the stacked parameters,
        same lax.scan over (index, params, cache)). ``layers`` is 18 (treated pass, record
        route, clean variant B) or 9 (clean variant A, layers 0..8 with kv_cache[0:9]).
        With ``edit_at == 8`` the L9 post-block action H is replaced by
        ``cfr_repair_rules.edit_rule``; layers 9..17 then continue from H' with the same
        kv_cache (mixed context). Returns (readout or None, trace).
        """
        if (len(embedded) != 2 or embedded[0] is not None or embedded[1] is None
                or self.dropout or self.configs[0].depth != 18):
            raise ValueError("cfr scan requires deterministic action-only depth18")
        if layers not in (9, 18):
            raise ValueError("cfr scan supports 9 (clean partial) or 18 layers")
        if (edit_at is None) != (edit_inputs is None):
            raise ValueError("cfr edit needs both edit_at and edit_inputs")
        if edit_at is not None and (edit_at != 8 or layers != 18):
            raise ValueError("cfr edit site is the L9 post-block output of the 18-layer pass")
        embedded = jax.tree.map(lambda x: x.astype(self.embed_dtype), embedded)
        h = embedded[1]
        if h.ndim != 3 or h.shape[1] != 10:
            raise ValueError("cfr scan requires all ten action tokens")
        block = Block(configs=self.configs, parent=None)
        params = self.variables["params"]["layers"]
        if layers != 18:
            params = jax.tree.map(lambda value: value[:layers], params)
            kv_cache = jax.tree.map(lambda value: value[:layers], kv_cache)
        empty = jnp.zeros_like(h)
        mask4 = jnp.asarray(mask)[:, None, :, :]
        if edit_inputs is not None:
            from openpi.models import cfr_repair_rules as rules  # noqa: PLC0415
            if set(edit_inputs) != set(rules.EDIT_INPUT_KEYS):
                raise ValueError("cfr edit inputs differ from the frozen rule signature")
            aux0 = rules.empty_aux(h, edit_inputs["bases"])
        else:
            rules = None
            aux0 = {}

        def step(carry, inputs):
            xs, h8_before, h8_after, l10_out, l18_out, aux = carry
            index, layer_params, cache = inputs
            xs, _ = block.apply(
                {"params": layer_params}, xs, cache, positions, mask4, adarms_cond, True,
            )
            l10_out = jax.lax.cond(index == 9, lambda: xs[1], lambda: l10_out)
            l18_out = jax.lax.cond(index == 17, lambda: xs[1], lambda: l18_out)
            h8_before = jax.lax.cond(index == 8, lambda: xs[1], lambda: h8_before)
            if rules is not None:
                def edit(state):
                    x, _old = state
                    after, new_aux = rules.edit_rule(x[1], **edit_inputs)
                    return [None, after], new_aux

                xs, aux = jax.lax.cond(index == 8, edit, lambda state: state, (xs, aux))
            h8_after = jax.lax.cond(index == 8, lambda: xs[1], lambda: h8_after)
            return (xs, h8_before, h8_after, l10_out, l18_out, aux), None

        (xs, before, after, l10_out, l18_out, aux), _ = jax.lax.scan(
            step, (embedded, empty, empty, empty, empty, aux0),
            (jnp.arange(layers), params, kv_cache),
        )
        readout = self.final_norms[1](xs[1], adarms_cond[1])[0] if layers == 18 else None
        return readout, {"L9_H_before": before, "L9_H_after": after,
                         "L10_H_after": l10_out, "L18_H_after": l18_out, "edit_aux": aux}


    def init(self, use_adarms: Sequence[bool]):
        """Convenience method for initializing all parameters, necessary due to the quirks of linen."""
        self.embed(jnp.zeros((1, 1), dtype=jnp.int32))
        self(
            [jnp.zeros((1, 1, c.width)) for c in self.configs],
            jnp.zeros((1, len(self.configs)), dtype=jnp.int32),
            jnp.zeros((1, len(self.configs), len(self.configs)), dtype=bool),
            adarms_cond=[jnp.zeros((1, c.width)) if u else None for u, c in zip(use_adarms, self.configs, strict=True)],
        )


def _apply_rope(x, *, positions, max_wavelength=10_000):
    """Applies RoPE positions [B, L] to x [B, L, H, D]."""
    freq_exponents = (2.0 / x.shape[-1]) * jnp.arange(x.shape[-1] // 2, dtype=jnp.float32)
    timescale = max_wavelength**freq_exponents
    radians = positions[..., None] / timescale[None, None, :]
    radians = radians[..., None, :]
    assert radians.dtype == jnp.float32
    # radians.shape = [...,L,1,d=D/2]
    sin, cos = jnp.sin(radians), jnp.cos(radians)
    x1, x2 = jnp.split(x, 2, axis=-1)
    res = jnp.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)
    assert res.dtype == jnp.float32
    # The original bigvision impl allows RoPE to upcast to float32. It is then immediately downcast again to the cache
    # dtype when in inference mode (but not in training mode). I don't think any of this was intentional. Based on the
    # original DeepMind impl, as well as the widely-used transformers impl, it is ok to always downcast back to bfloat16
    # here.
    return res.astype(x.dtype)


def _name(name, i):
    # we name layers like this because we want the first expert's weights to have no suffix (e.g., "attn"), so that they
    # can be loaded seamlessly from the existing PaliGemma checkpoint. subsequent experts will have a suffix (e.g.,
    # "attn_1") and their weights will be initialized from scratch. in practice, we only use two experts -- PaliGemma,
    # and the action expert.
    if i == 0:
        return name
    return f"{name}_{i}"


def _gated_residual(x, y, gate):
    assert (x is None) == (y is None)
    if x is None:
        return None
    if gate is None:
        return x + y
    return x + y * gate
