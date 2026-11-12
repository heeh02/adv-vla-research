"""cf-repair-v1 L9 write rules for the pi05 action expert (TECH-CONTRACT-v1 section 4.2).

New code for counterfactual-repair-v1 Stage 1A (no reuse source). Imported lazily by
the overlay ``gemma.Module.cfr_scan_suffix``; also unit-tested directly on CPU.

All projection arithmetic is FP32 with ``precision=HIGHEST``; the bases are consumed
as FP32 [1024,16] and never cast to the activation dtype. The written value is the
cast of the FP32 result to the native activation dtype (BF16); a declared no-op or a
non-writing route selects the original bits without an FP32 round trip.

Routes (dynamic int32 so that every arm shares one compiled executable):
  0 HOOK_NO_WRITE        H' = H_T bits; all logging computed
  1 NATURAL              per token: declared no-op -> H_T bits, else bf16(f32(H_T) - alpha*p_arm)
  2 ALPHA_FP32_NO_RULE   bf16(f32(H_T) - alpha*p_arm) for every token (alpha=0 identity route)
  3 FULL                 H' = H_C bits (R_full)
  4 EQUAL_NORM           s_t = min_k in eq ||p_k,t||; declared no-op if s_t <= tau_t,
                         else bf16(f32(H_T) - s_t * p_arm,t / ||p_arm,t||)
Declared no-op (rule frozen in TECH-CONTRACT-v1 4.2): ||w_t|| <= tau_t = c_noop * 2^-9 * ||H_T,t||,
or D_t == 0 exactly. ``w`` is the vector subtracted from H_T (H' = H_T - w). The template
split (M) is logging only and never enters the write.
"""

import jax
import jax.numpy as jnp

HOOK_NO_WRITE = 0
NATURAL = 1
ALPHA_FP32_NO_RULE = 2
FULL = 3
EQUAL_NORM = 4
ROUTE_NAMES = ("hook_no_write", "natural", "alpha_fp32_no_rule", "full", "equal_norm")
BF16_NOOP_FACTOR = 2.0 ** -9  # Frozen no-op threshold, not BF16's unit roundoff.
EDIT_INPUT_KEYS = ("h_c", "bases", "route", "alpha", "arm", "eq_index", "c_noop", "active")


def empty_aux(h, bases):
    """Zero-initialised carry with the exact structure returned by ``edit_rule``."""
    batch, tokens, width = h.shape
    k = bases.shape[0]
    return {
        "w": jnp.zeros((batch, tokens, width), jnp.float32),
        "declared_noop": jnp.zeros((batch, tokens), jnp.bool_),
        "s": jnp.zeros((batch, tokens), jnp.float32),
        "argmin": jnp.zeros((batch, tokens), jnp.int32),
        "p_norm": jnp.zeros((k, batch, tokens), jnp.float32),
        "tau": jnp.zeros((batch, tokens), jnp.float32),
        "applied": jnp.zeros((), jnp.bool_),
    }


def edit_rule(h_t, *, h_c, bases, route, alpha, arm, eq_index, c_noop, active):
    """Return (H', aux) for one generation call at the L9 post-block site.

    h_t, h_c: [b,10,d] native dtype (bf16). bases: [K,d,16] float32 (order B, C, G1, G2, G3).
    route, arm: int32 scalars; eq_index: int32[3]; alpha, c_noop: float32 scalars;
    active: bool scalar (this call is in the edit schedule).
    """
    if h_t.dtype != h_c.dtype or h_t.shape != h_c.shape or h_t.ndim != 3:
        raise ValueError("cfr_edit_requires_same_native_dtype_and_shape_HT_HC")
    if bases.dtype != jnp.float32 or bases.ndim != 3 or bases.shape[1] != h_t.shape[-1]:
        raise ValueError("cfr_edit_requires_FP32_bases_K_d_r")
    hp = jax.lax.Precision.HIGHEST
    hf = h_t.astype(jnp.float32)
    cf = h_c.astype(jnp.float32)
    d = hf - cf
    coeff = jnp.einsum("btd,kdr->kbtr", d, bases, precision=hp)
    p = jnp.einsum("kbtr,kdr->kbtd", coeff, bases, precision=hp)
    p_norm = jnp.sqrt(jnp.sum(p * p, axis=-1))
    h_norm = jnp.sqrt(jnp.sum(hf * hf, axis=-1))
    tau = jnp.asarray(c_noop, jnp.float32) * jnp.float32(BF16_NOOP_FACTOR) * h_norm
    d_zero = jnp.all(d == 0, axis=-1)
    p_arm = jnp.take(p, arm, axis=0)
    n_arm = jnp.take(p_norm, arm, axis=0)

    w_alpha = jnp.asarray(alpha, jnp.float32) * p_arm
    w_alpha_norm = jnp.sqrt(jnp.sum(w_alpha * w_alpha, axis=-1))
    noop_natural = (w_alpha_norm <= tau) | d_zero

    eq_norms = jnp.take(p_norm, eq_index, axis=0)
    s = jnp.min(eq_norms, axis=0)
    argmin = jnp.argmin(eq_norms, axis=0).astype(jnp.int32)
    safe = jnp.where(n_arm > 0, n_arm, jnp.float32(1.0))
    w_equal = (s / safe)[..., None] * p_arm
    noop_equal = (s <= tau) | d_zero

    def cast(w):
        return (hf - w).astype(h_t.dtype)

    h_natural = jnp.where(noop_natural[..., None], h_t, cast(w_alpha))
    h_alpha_rule_free = cast(w_alpha)
    h_equal = jnp.where(noop_equal[..., None], h_t, cast(w_equal))
    h_full = h_c

    # Nested selects: the chosen element is copied bit-exactly.
    h_new = jnp.where(route == FULL, h_full, h_t)
    h_new = jnp.where(route == NATURAL, h_natural, h_new)
    h_new = jnp.where(route == ALPHA_FP32_NO_RULE, h_alpha_rule_free, h_new)
    h_new = jnp.where(route == EQUAL_NORM, h_equal, h_new)
    h_new = jnp.where(active, h_new, h_t)

    zero_w = jnp.zeros_like(hf)
    w = jnp.where(route == FULL, d, zero_w)
    w = jnp.where((route == NATURAL) | (route == ALPHA_FP32_NO_RULE), w_alpha, w)
    w = jnp.where(route == EQUAL_NORM, w_equal, w)
    w = jnp.where(active, w, zero_w)
    noop = jnp.zeros_like(noop_natural)
    noop = jnp.where(route == NATURAL, noop_natural, noop)
    noop = jnp.where(route == EQUAL_NORM, noop_equal, noop)
    noop = jnp.where(active, noop, jnp.zeros_like(noop))
    aux = {"w": w, "declared_noop": noop, "s": s, "argmin": argmin, "p_norm": p_norm,
           "tau": tau, "applied": jnp.asarray(active, jnp.bool_)}
    return h_new, aux
