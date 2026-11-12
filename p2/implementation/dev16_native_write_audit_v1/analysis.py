"""Signed native-output/proxy audit; no fitting or native-kernel replay."""
import hashlib
import math
import numpy as np
from codec import need, from_array
from pinned import quantization_frozen as Q

KEYS = tuple((f"SDRU-CDEV-T0{t}-R0{r}", q) for t in range(2)
             for r in range(2) for q in (11, 22))
WINDOWS = tuple((f"position_{p}", p, p + 1) for p in range(10)) + (
    ("first5", 0, 5), ("tail5", 5, 10), ("full10", 0, 10))
DIMS = tuple("translation_command_" + c for c in "xyz") + tuple(
    "rotation_command_" + c for c in "xyz") + ("gripper_command",)


def vector_metrics(reference, candidate):
    a, b = (np.asarray(x, np.float64).ravel(order="C") for x in (reference, candidate))
    need(a.shape == b.shape and a.size > 0 and np.isfinite(a).all()
         and np.isfinite(b).all(), "paired_finite_vectors")
    error = b - a
    aa, bb, ab, ee = (float(np.dot(x, y)) for x, y in
                      ((a, a), (b, b), (a, b), (error, error)))
    na, nb = math.sqrt(aa), math.sqrt(bb)
    az, bz = a == 0, b == 0
    both_nonzero = ~(az | bz)
    sign_same = int(np.count_nonzero((np.signbit(a) == np.signbit(b)) & both_nonzero))
    gain = ab / aa if aa else None
    residual = b - gain * a if gain is not None else None
    gain_error = float(np.dot(residual, residual)) if residual is not None else None
    return {
        "coordinates": int(a.size), "reference_norm": na, "candidate_norm": nb,
        "reference_squared_norm": aa, "candidate_squared_norm": bb, "inner_product": ab,
        "difference_norm": math.sqrt(ee), "difference_RMS": math.sqrt(ee / a.size),
        "difference_maxabs": float(np.max(np.abs(error))),
        "relative_vector_error_to_reference": math.sqrt(ee / aa) if aa else None,
        "relative_vector_error_to_candidate": math.sqrt(ee / bb) if bb else None,
        "candidate_over_reference_norm": nb / na if na else None,
        "cosine": ab / (na * nb) if na and nb else None,
        "reference_zero": not bool(aa), "candidate_zero": not bool(bb),
        "both_zero_coordinates": int(np.count_nonzero(az & bz)),
        "reference_zero_candidate_nonzero": int(np.count_nonzero(az & ~bz)),
        "reference_nonzero_candidate_zero": int(np.count_nonzero(~az & bz)),
        "both_nonzero_coordinates": int(np.count_nonzero(both_nonzero)),
        "same_sign_both_nonzero": sign_same,
        "opposite_sign_both_nonzero": int(np.count_nonzero(both_nonzero)) - sign_same,
        "same_sign_fraction_of_both_nonzero": sign_same / int(np.count_nonzero(both_nonzero))
        if both_nonzero.any() else None,
        "descriptive_best_scalar_candidate_from_reference": gain,
        "descriptive_best_scalar_residual_norm": math.sqrt(gain_error)
        if gain_error is not None else None,
        "descriptive_best_scalar_residual_over_candidate_norm": math.sqrt(gain_error / bb)
        if gain_error is not None and bb else None,
        "projection_is_target_visible_not_prediction": True,
    }


def compare_typed(a, b):
    return {"reference": a.identity(), "candidate": b.identity(),
            "typed_exact": a.identity() == b.identity(),
            "numeric_equal": bool(np.array_equal(a.numeric(), b.numeric())),
            **vector_metrics(a.numeric(), b.numeric())}


def proxy(p, e, wo, gate):
    p, e, wo, gate = (np.asarray(x, np.float64) for x in (p, e, wo, gate))
    need(p.shape == (8, 10, e.shape[0] + 10) and e.shape[1] == 256 and
         wo.shape == (8, 256, 1024) and gate.shape == (1024,), "proxy_ABI")
    # Preserve the legacy signed-transfer contraction order. P includes the
    # complete denominator; its prefix is selected without renormalization.
    read = np.einsum("hpn,nd->phd", p[:, :, :len(e)], e)
    pre_gate = np.einsum("phd,hdf->pf", read, wo)
    return pre_gate, pre_gate * gate[None, :]


def scalar_check(actual, saved):
    actual, saved = float(actual), float(saved)
    need(math.isfinite(actual) and math.isfinite(saved), "finite_scalar_QA")
    return {"recomputed": actual, "saved": saved, "difference": actual - saved,
            "exact_scalar_equal": actual == saved,
            "scalar_close": math.isclose(actual, saved, rel_tol=5e-13, abs_tol=1e-24),
            "relative_tolerance": 5e-13, "absolute_tolerance": 1e-24,
            "not_a_native_equality_tolerance": True}


def dyadic_float_units(value):
    """Exact integer representation of a recorded finite FP64 value at 2^-1074."""
    value = float(value)
    need(math.isfinite(value), "finite_proxy_value")
    n, d = value.as_integer_ratio()
    need(d & (d - 1) == 0 and d.bit_length() <= 1075, "binary_float_ratio")
    return n << (1074 - (d.bit_length() - 1))


def verify_energy_closure(total, first, second, cross):
    need(total == first + second + cross, "exact_proxy_native_signed_energy_closure")


def exact_decomposition(a_before, a_after, h_before, h_after, gate, qhat,
                        *, check=lambda: None):
    """Exact dyadic identity for the saved native values and FP64 proxy.

    This does not identify the native multiply/add implementation. A mismatched
    pre-state remains recorded and the caller disables that interpretation.
    """
    arrays = [np.asarray(x) for x in (a_before, a_after, h_before, h_after)]
    need(all(x.dtype == np.uint16 and x.shape == (10, 1024) for x in arrays)
         and np.asarray(gate).shape == (1024,), "dyadic_BF16_ABI")
    need(np.asarray(gate).dtype == np.uint16 and np.asarray(qhat).shape == (10, 1024),
         "dyadic_gate_proxy_ABI")
    by_position = []
    for p in range(10):
        check()
        values = {k: [] for k in ("total", "projection", "gate_residual", "native_ideal", "native_actual")}
        for c in range(1024):
            a0, a1, h0, h1 = (Q.units(x[p, c]) for x in arrays)
            g = Q.units(gate[c])
            # BF16 units = 2^-133; proxy units = 2^-1074; products below use 2^-1207.
            da, dh, q = a1 - a0, h1 - h0, dyadic_float_units(qhat[p, c])
            ideal = (g * da) << 941
            actual = dh << 1074
            projection = g * ((da << 941) - q)
            residual = actual - ideal
            total = actual - g * q
            need(total == projection + residual, "exact_proxy_native_vector_closure")
            for name, value in (("total", total), ("projection", projection),
                                ("gate_residual", residual), ("native_ideal", ideal),
                                ("native_actual", actual)):
                values[name].append(value)
        by_position.append(values)
    rows = []
    for window, start, end in WINDOWS:
        values = {k: [x for pos in by_position[start:end] for x in pos[k]]
                  for k in by_position[0]}
        energies = {k: sum(x * x for x in v) for k, v in values.items()}
        cross = 2 * sum(x * y for x, y in zip(values["projection"], values["gate_residual"]))
        native_cross = 2 * sum(x * y for x, y in zip(values["native_ideal"], values["gate_residual"]))
        verify_energy_closure(energies["total"], energies["projection"],
                              energies["gate_residual"], cross)
        need(energies["native_actual"] == energies["native_ideal"] +
             energies["gate_residual"] + native_cross, "exact_native_signed_energy_closure")
        digest = hashlib.sha256()
        for name, vector in values.items():
            digest.update(name.encode())
            for value in vector:
                digest.update((str(value) + "\n").encode())
        reports = {}
        for name, vector in values.items():
            energy = Q.scaled(energies[name], -2414)
            reports[name] = {"squared_norm": energy, "norm": math.sqrt(energy),
                             "nonzero_coordinates": sum(x != 0 for x in vector)}
        rows.append({"window": window, "coordinates": (end - start) * 1024,
                     "terms": reports, "signed_cross_term": Q.scaled(cross, -2414),
                     "native_ideal_residual_signed_cross_term": Q.scaled(native_cross, -2414),
                     "exact_vector_closure": True, "exact_energy_closure": True,
                     "exact_native_energy_closure": True, "exact_units_exponent": -1207,
                     "exact_vector_digest": digest.hexdigest(),
                     "FP64_proxy_is_reference_not_native_kernel": True,
                     "exact_products_use_saved_FP64_Qhat_and_BF16_gate_as_dyadic_values": True,
                     "fidelity_gated_proxy_uses_rounded_FP64_multiplication": True,
                     "exact_vs_FP64_product_last_bit_equality_claimed": False,
                     "native_gated_product_captured": False,
                     "multiplication_addition_attribution": "NOT_IDENTIFIED"})
    return rows


def site(leaves, name):
    x = leaves[("reads__" + name,)]
    need(x.logical == "bfloat16", "site_native_BF16")
    expected = (5, 3, 1, 1, 1024) if name == "attention_gate" else (5, 3, 1, 10, 1024)
    need(x.storage.shape == expected, "site_shape")
    return x.select((0, 1, 0, 0) if name == "attention_gate" else (0, 1, 0))


def state_fields(leaves, n):
    fields = {
        "query": leaves[("j5_L9_query_post_rope_scaled",)],
        "P": leaves[("j5_L9_probabilities_native",)],
        "gate": site(leaves, "attention_gate"),
        "receiver_input": site(leaves, "receiver_residual_input"),
        "mask": leaves[("j5_mask",)], "positions": leaves[("j5_suffix_positions",)],
        "query_defined": leaves[("j5_query_defined",)],
        "x0_through_x5": leaves[("flow__x",)].select(slice(0, 6)),
        "v0_through_v4": leaves[("flow__velocity_native",)].select(slice(0, 5)),
        "time": leaves[("flow__time",)], "dt": leaves[("flow__dt",)],
        "time_weak_type": leaves[("flow__time_weak_type",)],
    }
    abi = {"query": ((1, 10, 1, 8, 256), "bfloat16"),
           "P": ((1, 1, 8, 10, n + 10), "bfloat16"),
           "mask": ((1, 10, n + 10), "bool"), "positions": ((1, 10), "int32"),
           "query_defined": ((1, 10), "bool"), "x0_through_x5": ((6, 1, 10, 32), "float32"),
           "v0_through_v4": ((5, 1, 10, 32), "bfloat16"), "time": ((11,), "float32"),
           "dt": ((10,), "float32"), "time_weak_type": ((11,), "bool")}
    for name, (shape, dtype) in abi.items():
        need(fields[name].storage.shape == shape and fields[name].logical == dtype, "state_field_ABI")
    return fields


def snapshot(key, leaves, edits, wo, obs, *, check=lambda: None,
             prefix_binding_exact=True, valid_prefix_tokens=None):
    root, query = key
    n = edits["S"].shape[0]
    states = {arm: state_fields(leaves[arm], n) for arm in ("N", "S", "R")}
    rows, fidelity, decomp, dose, endpoint, qa = [], [], [], [], [], []
    base = {"root_id": root, "query_index": query}
    for arm in ("S", "R"):
        for name in states["N"]:
            rows.append({**base, "arm": arm, "field": name,
                         **compare_typed(states["N"][name], states[arm][name])})
    same = {arm: prefix_binding_exact and all(x["typed_exact"] for x in rows if x["arm"] == arm)
            for arm in ("S", "R")}
    same["S-R"] = same["S"] and same["R"]
    gates = states["N"]["gate"]
    g = gates.numeric().astype(np.float64)
    p = states["N"]["P"].numeric()[0, 0]
    proxies = {arm: proxy(p, edits[arm], wo, g) for arm in ("S", "R")}
    native_a = {arm: site(leaves[arm], "attention_output_post_WO_pre_gate") for arm in leaves}
    native_h = {arm: site(leaves[arm], "hidden_after_attention_residual") for arm in leaves}
    a = {arm: native_a[arm].numeric().astype(np.float64) for arm in leaves}
    h = {arm: native_h[arm].numeric().astype(np.float64) for arm in leaves}
    end = {arm: leaves[arm][("adapter7",)].numeric().astype(np.float64)[0] for arm in leaves}
    for arm in ("S", "R"):
        need(end[arm].shape == (10, 7), "endpoint_7axis_shape")
        need(np.array_equal(end[arm] - end["N"],
                            np.asarray(obs["arms"][arm]["adapter7_effect"])[0]),
             "terminal_endpoint_effect_exact")
        q = scalar_check(float(np.sum(edits[arm] ** 2)),
                         obs["arms"][arm]["actual_E_squared_norm"])
        need(q["scalar_close"], "actual_E_existing_scalar_close")
        qa.append({**base, "arm": arm, "stage": "actual_E", **q})
        for stage, delta, l0, l1 in (
            ("L9_j5_write_pre_gate", a[arm] - a["N"], native_a["N"], native_a[arm]),
            ("L9_j5_post_attention_residual", h[arm] - h["N"], native_h["N"], native_h[arm])):
            saved = obs["arms"][arm]["stage_changes"][stage]
            c = scalar_check(float(np.sum(delta ** 2)), saved["squared_norm"])
            need(c["scalar_close"] and int(np.count_nonzero(delta)) == saved["numeric_nonzero_coordinates"]
                 and int(np.count_nonzero(l0.storage != l1.storage)) == saved["bitwise_changed_coordinates"]
                 and float(np.max(np.abs(delta))) == saved["maximum_absolute_difference"],
                 "existing_native_stage_summary_reproduced")
            qa.append({**base, "arm": arm, "stage": stage, **c})
    need(np.array_equal(end["S"] - end["R"], np.asarray(obs["S_minus_R_adapter7"])[0]),
         "terminal_SR_endpoint_exact")
    for contrast, before, after, qhat in (
        ("S-N", "N", "S", proxies["S"][0]), ("R-N", "N", "R", proxies["R"][0]),
        ("S-R", "R", "S", proxies["S"][0] - proxies["R"][0])):
        da, dh = a[after] - a[before], h[after] - h[before]
        valid = same[after] if before == "N" else same["S-R"]
        comparisons = (
            ("FP64_proxy_preGate_vs_native_deltaA", da, qhat),
            ("FP64_proxy_gated_vs_native_deltaH", dh, qhat * g),
            ("FP64_gate_times_native_deltaA_vs_native_deltaH", dh, da * g))
        for window, start, stop in WINDOWS:
            for name, actual, proposed in comparisons:
                fidelity.append({**base, "contrast": contrast, "comparison": name, "window": window,
                                 "reference_role": "native_deltaA" if name ==
                                 "FP64_proxy_preGate_vs_native_deltaA" else "native_deltaH",
                                 "candidate_role": name.split("_vs_native_")[0],
                                 "ratio_orientation": "FP64_proposal_over_native_output_difference",
                                 "same_state_interpretation_eligible": valid,
                                 **vector_metrics(actual[start:stop], proposed[start:stop])})
        for row in exact_decomposition(native_a[before].storage, native_a[after].storage,
                                       native_h[before].storage, native_h[after].storage,
                                       gates.storage, qhat, check=check):
            decomp.append({**base, "contrast": contrast,
                           "same_state_interpretation_eligible": valid,
                           "gate_residual_arithmetic_interpretation_eligible": valid,
                           "gate_residual_term_interpretation":
                           "combined_difference_with_same_recorded_pre_state_product_uncaptured"
                           if valid else "may_include_context_or_gate_change_not_pure_arithmetic",
                           **row})
        for window, stop in (("first5", 5), ("full10", 10)):
            effect = end[after][:stop] - end[before][:stop]
            for dim, name in enumerate(DIMS):
                x = effect[:, dim]
                endpoint.append({**base, "contrast": contrast, "window": window, "dimension": name,
                                 "RMS": float(np.sqrt(np.mean(x ** 2))),
                                 "maximum_absolute": float(np.max(np.abs(x))),
                                 "numeric_nonzero_positions": int(np.count_nonzero(x)),
                                 "positions": stop, "physical_threshold": None})
    dose.append({**base, "stage": "actual_E", "window": "all_prefix_tokens",
                 "same_state_interpretation_eligible": same["S-R"],
                 "ratio_orientation": "S_over_R",
                 "total_prefix_tokens": n, "valid_prefix_tokens": valid_prefix_tokens,
                 "coordinate_denominator_includes_zero_padding": True,
                 **vector_metrics(edits["R"], edits["S"])})
    for stage, sr, rr in (
        ("native_deltaA", a["S"] - a["N"], a["R"] - a["N"]),
        ("native_deltaH", h["S"] - h["N"], h["R"] - h["N"]),
        ("FP64_proxy_preGate", proxies["S"][0], proxies["R"][0]),
        ("FP64_proxy_gated", proxies["S"][1], proxies["R"][1])):
        for window, start, stop in WINDOWS:
            dose.append({**base, "stage": stage, "window": window,
                         "same_state_interpretation_eligible": same["S-R"],
                         "ratio_orientation": "S_over_R",
                         **vector_metrics(rr[start:stop], sr[start:stop])})
    return rows, fidelity, decomp, dose, endpoint, qa
