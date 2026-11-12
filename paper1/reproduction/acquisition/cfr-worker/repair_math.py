"""Host-side (NumPy, float64) measurement of the L9 repair writes (TECH-CONTRACT-v1 4.2-4.3, 7.0).

New code. Every quantity is computed from the exact BF16 bits returned by the graph
(H_T, H_C, H'), the FP32 bases actually consumed by the graph, and the graph's FP32
intended write ``w``. float64 measurement is strictly more precise than the FP32
write, so the deterministic bounds of test T6/T7 can be checked without calibration.

Also provides ``reference_rule``: a NumPy emulation of cfr_repair_rules.edit_rule used
only by local tests (it is never used to produce scientific values).
"""

from __future__ import annotations

import numpy as np

U_BF16 = 2.0 ** -8          # BF16 unit roundoff; PRECAL-FREEZE-v1 section 2
U_BF16_OUTPUT = U_BF16 / (1.0 - U_BF16)
BF16_NOOP_FACTOR = 2.0 ** -9  # Frozen no-op rule; this is NOT the unit roundoff.
F32_EPS_HALF = 2.0 ** -24
WIDTH = 1024
ROUTES = ("hook_no_write", "natural", "alpha_fp32_no_rule", "full", "equal_norm")
ROUTE_CODE = {name: index for index, name in enumerate(ROUTES)}
BASIS_ORDER = ("B", "C", "G1", "G2", "G3")


def gram_error(basis):
    """g_k = ||I - B^T B||_2 (spectral norm) in float64 of the basis as consumed."""
    b = np.asarray(basis, np.float64)
    return float(np.linalg.norm(np.eye(b.shape[1]) - b.T @ b, ord=2))


def gram_max_abs(basis):
    b = np.asarray(basis, np.float64)
    return float(np.max(np.abs(b.T @ b - np.eye(b.shape[1]))))


def _norm(x):
    return np.sqrt(np.sum(np.square(x), axis=-1))


def project(x, basis):
    b = np.asarray(basis, np.float64)
    return (x @ b) @ b.T


def eps32(h_t_norm, h_c_norm, width=WIDTH):
    return width * F32_EPS_HALF * (h_t_norm + h_c_norm)


def call_measurements(*, h_t, h_c, h_p, w, declared_noop, route, arm_name, bases, template,
                      gram, s=None, argmin=None, eq_names=None, tau=None, active=True):
    """Per-token measurements for one generation call (all arrays [10, d] for batch 1).

    bases: dict name -> FP32 [d,16] as consumed; template: FP32 [10,d] (M, logging only);
    gram: dict name -> g_k. Returns a dict of per-token lists (length 10).
    """
    ht, hc, hp = (np.asarray(v).astype(np.float64).reshape(-1, WIDTH) for v in (h_t, h_c, h_p))
    wf = np.asarray(w, np.float64).reshape(-1, WIDTH)
    m = np.asarray(template, np.float64).reshape(-1, WIDTH)
    d = ht - hc
    n_ht, n_hc, n_d = _norm(ht), _norm(hc), _norm(d)
    out = {"H_T_norm": n_ht, "H_C_norm": n_hc, "D_norm": n_d, "bases": {}}
    p_norms = {}
    for name, basis in bases.items():
        p = project(d, basis)
        pm = project(m, basis)
        pr = project(d - m, basis)
        pn = _norm(p)
        p_norms[name] = pn
        share = np.where(n_d > 0, pn ** 2 / np.where(n_d > 0, n_d, 1.0) ** 2, np.nan)
        out["bases"][name] = {"p_norm": pn, "pM_norm": _norm(pm), "pR_norm": _norm(pr),
                              "energy_share": share}
    big_w = hp - ht
    cast_target = big_w + wf
    noop = np.asarray(declared_noop, bool).reshape(-1)
    applies = active and route in ("natural", "alpha_fp32_no_rule", "full", "equal_norm")
    out.update({
        "w_intended_norm": _norm(wf),
        "W_actual_norm": _norm(big_w),
        "W_actual_maxabs": np.max(np.abs(big_w), axis=-1),
        "cast_error_norm": np.where(noop | (not applies), np.nan, _norm(cast_target)),
        "declared_noop": noop,
        "actual_noop": (~noop) & np.all(big_w == 0, axis=-1),
        "H_prime_equals_H_T_bits": np.all(
            np.ascontiguousarray(h_p).reshape(-1, WIDTH).view(np.uint8).reshape(-1, WIDTH * np.asarray(h_p).dtype.itemsize)
            == np.ascontiguousarray(h_t).reshape(-1, WIDTH).view(np.uint8).reshape(-1, WIDTH * np.asarray(h_t).dtype.itemsize), axis=-1),
        "cast_bound": U_BF16_OUTPUT * _norm(hp) + eps32(n_ht, n_hc),
        "hidden_distance_before": n_d,
        "hidden_distance_after": _norm(hp - hc),
    })
    if tau is not None:
        out["tau"] = np.asarray(tau, np.float64).reshape(-1)
    if arm_name is not None:
        b = bases[arm_name]
        delta = hp - ht
        r_in = _norm((hp - hc) @ np.asarray(b, np.float64))
        r_out = _norm(delta - project(delta, b))
        e32 = eps32(n_ht, n_hc)
        g = gram[arm_name]
        out.update({
            "identity_basis": arm_name,
            "r_in": r_in, "r_out": r_out,
            "bound_in": U_BF16_OUTPUT * _norm(hp) + 2 * g * n_d + e32,
            "bound_out": U_BF16_OUTPUT * _norm(hp) + 2 * g * p_norms[arm_name] + e32,
        })
    if route == "equal_norm" and s is not None:
        s64 = np.asarray(s, np.float64).reshape(-1)
        pb = p_norms["B"]
        out["equal_norm"] = {
            "s": s64, "f": np.where(pb > 0, s64 / np.where(pb > 0, pb, 1.0), np.nan),
            "argmin": [eq_names[int(i)] for i in np.asarray(argmin).reshape(-1)] if eq_names else None,
        }
    return out


def identity_verdict(measure):
    """T6 per-token check: r_in <= bound_in and r_out <= bound_out (deterministic bounds)."""
    r_in, r_out = np.asarray(measure["r_in"]), np.asarray(measure["r_out"])
    b_in, b_out = np.asarray(measure["bound_in"]), np.asarray(measure["bound_out"])
    noop = np.asarray(measure["declared_noop"], bool)
    bit_equal = np.asarray(measure["H_prime_equals_H_T_bits"], bool)
    cast_error, cast_bound = np.asarray(measure["cast_error_norm"]), np.asarray(measure["cast_bound"])
    ok = np.where(noop, bit_equal, (r_in <= b_in) & (r_out <= b_out) & (cast_error <= cast_bound))
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio_in = np.where(noop, np.nan, np.where(b_in > 0, r_in / b_in, np.where(r_in == 0, 0.0, np.inf)))
        ratio_out = np.where(noop, np.nan, np.where(b_out > 0, r_out / b_out, np.where(r_out == 0, 0.0, np.inf)))
    return {"pass": bool(ok.all()), "tokens": int(ok.size), "failing_tokens": int((~ok).sum()),
            "max_ratio_in": float(np.max(ratio_in[~noop])) if (~noop).any() else None,
            "max_ratio_out": float(np.max(ratio_out[~noop])) if (~noop).any() else None,
            "declared_noop_tokens": int(noop.sum()), "declared_noop_bit_failures": int((noop & ~bit_equal).sum()),
            "cast_failures": int(((~noop) & ~(cast_error <= cast_bound)).sum())}


def equal_norm_verdict(*, w, s, tau, h_t, h_c, h_p):
    """T7 j0 checks: |‖s p̂‖ - s| <= 1e-6 s + 1e-12; ‖W + s p̂‖ <= u‖H'‖ + eps32; s <= tau excluded."""
    wf = np.asarray(w, np.float64).reshape(-1, WIDTH)
    s64 = np.asarray(s, np.float64).reshape(-1)
    t64 = np.asarray(tau, np.float64).reshape(-1)
    ht, hc, hp = (np.asarray(v).astype(np.float64).reshape(-1, WIDTH) for v in (h_t, h_c, h_p))
    active = s64 > t64
    cand = np.abs(_norm(wf) - s64) <= 1e-6 * s64 + 1e-12
    actual = _norm((hp - ht) + wf) <= U_BF16_OUTPUT * _norm(hp) + eps32(_norm(ht), _norm(hc))
    bit_equal = np.all(np.ascontiguousarray(h_p).reshape(-1, WIDTH).view(np.uint8).reshape(len(s64), -1)
                       == np.ascontiguousarray(h_t).reshape(-1, WIDTH).view(np.uint8).reshape(len(s64), -1), axis=-1)
    ok = np.where(active, cand & actual, bit_equal)
    return {"pass": bool(ok.all()), "checked_tokens": int(active.sum()), "excluded_s_le_tau": int((~active).sum()),
            "candidate_fail": int((active & ~cand).sum()), "actual_fail": int((active & ~actual).sum()),
            "declared_noop_bit_failures": int((~active & ~bit_equal).sum()),
            "max_candidate_rel_error": float(np.max(np.where(active, np.abs(_norm(wf) - s64) / np.where(s64 > 0, s64, 1), 0.0)))}


def reference_rule(h_t, h_c, bases_f32, *, route, alpha, arm, eq_index, c_noop=1.0, active=True):
    """NumPy emulation of cfr_repair_rules.edit_rule (tests only). bases_f32: [K,d,16]."""
    import ml_dtypes  # noqa: PLC0415
    bf16 = ml_dtypes.bfloat16
    ht = np.asarray(h_t)
    hf = ht.astype(np.float32)
    cf = np.asarray(h_c).astype(np.float32)
    d = hf - cf
    bases_f32 = np.asarray(bases_f32, np.float32)
    p = np.stack([(d @ b) @ b.T for b in bases_f32]).astype(np.float32)
    pn = np.sqrt(np.sum(p * p, axis=-1)).astype(np.float32)
    tau = np.float32(c_noop) * np.float32(BF16_NOOP_FACTOR) * np.sqrt(np.sum(hf * hf, axis=-1)).astype(np.float32)
    d_zero = np.all(d == 0, axis=-1)
    p_arm, n_arm = p[arm], pn[arm]
    w_alpha = (np.float32(alpha) * p_arm).astype(np.float32)
    noop_nat = (np.sqrt(np.sum(w_alpha * w_alpha, axis=-1)) <= tau) | d_zero
    eq = pn[list(eq_index)]
    s = eq.min(axis=0)
    w_eq = ((s / np.where(n_arm > 0, n_arm, 1.0)).astype(np.float32)[..., None] * p_arm).astype(np.float32)
    noop_eq = (s <= tau) | d_zero

    def cast(w):
        return (hf - w).astype(np.float32).astype(bf16)

    name = ROUTES[route]
    if not active or name == "hook_no_write":
        return ht.copy(), np.zeros_like(hf), np.zeros(ht.shape[:-1], bool)
    if name == "full":
        return np.asarray(h_c).copy(), d, np.zeros(ht.shape[:-1], bool)
    if name == "alpha_fp32_no_rule":
        return cast(w_alpha), w_alpha, np.zeros(ht.shape[:-1], bool)
    if name == "natural":
        out = np.where(noop_nat[..., None], ht, cast(w_alpha))
        return out.astype(bf16), w_alpha, noop_nat
    out = np.where(noop_eq[..., None], ht, cast(w_eq))
    return out.astype(bf16), w_eq, noop_eq
