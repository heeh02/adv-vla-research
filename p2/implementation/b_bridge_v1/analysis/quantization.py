"""Pure CPU readout of actual native W/gate/residual tensors.

Two declared rounding models are evaluated as CPU candidates, never asserted
to be the compiled native path. Exact dyadic identities compare a real
recorded post-attention residual increment with its ideal W*gate increment.
Without an explicitly captured consumed multiplication result, multiplication
and addition rounding are not separately attributed.
"""
import math
import numpy as np

ROLES = ("none", "natural_d", "common_c")
FIELDS = ("W_before", "W_used", "gate", "residual_input", "after_attention_residual")
PAIRS = (("d_minus_none", "none", "natural_d"), ("c_minus_none", "none", "common_c"),
         ("c_minus_d", "natural_d", "common_c"))
WINDOWS = (("first5", 0, 5), ("tail5", 5, 10), ("full10", 0, 10))


class QuantizationError(ValueError):
    pass


def need(ok, code):
    if not ok:
        raise QuantizationError(code)


def units(bits):
    b = int(bits)
    exponent, fraction = (b >> 7) & 255, b & 127
    need(0 <= b <= 65535 and exponent != 255, "finite_original_BF16_bits")
    v = fraction if exponent == 0 else (128 + fraction) << (exponent - 1)
    return -v if b >> 15 else v


def scaled(value, exponent):
    """Report-only FP64 conversion after exact integer sums (no admission use)."""
    if value == 0:
        return 0.
    sign, magnitude = (-1 if value < 0 else 1), abs(value)
    shift = max(0, magnitude.bit_length() - 53)
    result = math.ldexp(float(sign * (magnitude >> shift)), exponent + shift)
    need(math.isfinite(result), "finite_report_range")
    return result


def round_product_units_to_bf16(value, *, zero_sign=0):
    """Exact RN-even of integer *2^-266 to BF16 bits, including signed zero.

    Returns signed infinity on finite-input rounding overflow, rather than
    saturating or silently accepting a finite candidate. No float is used.
    """
    need(type(value) is int and zero_sign in (0, 1), "exact_rounding_input")
    if value == 0:
        return zero_sign << 15
    sign, magnitude = int(value < 0), abs(value)
    shift = max(133, magnitude.bit_length()-8)
    q, remainder = divmod(magnitude, 1 << shift)
    half = 1 << (shift-1)
    if remainder > half or (remainder == half and q & 1):
        q += 1
    target = (q << shift) >> 133
    if target == 0:
        return sign << 15
    if target > (255 << 253):
        return (sign << 15) | 0x7f80
    if target < 128:
        return (sign << 15) | target
    exponent = target.bit_length()-7
    spacing = exponent-1
    need(target % (1 << spacing) == 0, "RN_exact_BF16_lattice")
    fraction = (target >> spacing)-128
    need(1 <= exponent <= 254 and 0 <= fraction <= 127, "RN_finite_internal_range")
    return (sign << 15) | (exponent << 7) | fraction


def _sum_zero_sign(a_units, a_sign, b_units, b_sign):
    # Under RN-even, exact cancellation is +0 except -0 + -0.
    return int(a_units == b_units == 0 and a_sign == b_sign == 1)


def rounding_candidates(w_bits, gate_bits, residual_bits):
    """Predict fused and split candidates for ONE coordinate from native bits."""
    w, g, r = units(w_bits), units(gate_bits), units(residual_bits)
    product = w*g
    product_sign = ((int(w_bits) >> 15) ^ (int(gate_bits) >> 15)) & 1
    residual_sign = int(residual_bits) >> 15
    fused_n = (r << 133)+product
    fused = round_product_units_to_bf16(fused_n,
        zero_sign=_sum_zero_sign(r, residual_sign, product, product_sign))
    rounded_product = round_product_units_to_bf16(product, zero_sign=product_sign)
    product_overflow = ((rounded_product >> 7) & 255) == 255
    if product_overflow:
        split = rounded_product  # finite r cannot cancel an infinite product
    else:
        p = units(rounded_product)
        split = round_product_units_to_bf16((r+p) << 133,
            zero_sign=_sum_zero_sign(r, residual_sign, p, rounded_product >> 15))
    return {"fused": fused, "split": split, "split_product": rounded_product,
            "split_product_overflow": product_overflow}


def analyze_rounding_candidates(records, *, check=lambda: None):
    """Accept old pulse/persistent and/or new none/d/c leaves, one record each.

    Each record supplies W_used, gate, residual_input, after_attention_residual.
    All arrays must be actual native BF16 bits; no residual or gate is inferred.
    """
    need(bool(records) and len(records) <= 5, "bounded_original_and_new_native_records")
    output = []
    for role, record in records.items():
        arrays = {k: _array(record[k], gate=k == "gate") for k in ("W_used", "gate", "residual_input", "after_attention_residual")}
        per_position = []
        for p in range(10):
            check()
            counts = {k: 0 for k in ("coordinates", "fused_bitwise_match", "split_bitwise_match", "both_bitwise_match",
                "neither_bitwise_match", "models_differ", "fused_zero_sign_only_mismatch", "split_zero_sign_only_mismatch",
                "fused_overflow", "split_overflow", "split_product_overflow")}
            mismatch = {"fused_channels": [], "split_channels": []}
            for c in range(1024):
                predictions = rounding_candidates(arrays["W_used"][p,c], arrays["gate"][p,c], arrays["residual_input"][p,c])
                actual = int(arrays["after_attention_residual"][p,c])
                f, s = predictions["fused"], predictions["split"]
                fm, sm = f == actual, s == actual
                values = {"coordinates": 1, "fused_bitwise_match": int(fm), "split_bitwise_match": int(sm),
                    "both_bitwise_match": int(fm and sm), "neither_bitwise_match": int(not fm and not sm),
                    "models_differ": int(f != s), "fused_zero_sign_only_mismatch": int(not fm and (f & 0x7fff) == (actual & 0x7fff) == 0),
                    "split_zero_sign_only_mismatch": int(not sm and (s & 0x7fff) == (actual & 0x7fff) == 0),
                    "fused_overflow": int(((f >> 7) & 255) == 255), "split_overflow": int(((s >> 7) & 255) == 255),
                    "split_product_overflow": int(predictions["split_product_overflow"])}
                for key, value in values.items():
                    counts[key] += value
                if not fm:
                    mismatch["fused_channels"].append(c)
                if not sm:
                    mismatch["split_channels"].append(c)
            per_position.append({"action_position": p, "counts": counts, "mismatches": mismatch})
        total = {key: sum(row["counts"][key] for row in per_position) for key in per_position[0]["counts"]}
        f_all, s_all = total["fused_bitwise_match"] == 10240, total["split_bitwise_match"] == 10240
        status = "BOTH_MATCH_PATH_NOT_IDENTIFIED" if f_all and s_all else "ONLY_FUSED_CANDIDATE_MATCHES" if f_all else \
                 "ONLY_SPLIT_CANDIDATE_MATCHES" if s_all else "NEITHER_CANDIDATE_MATCHES_ALL_COORDINATES"
        output.append({"role": role, "status": status, "counts": total, "per_position": per_position})
    return {"schema": "B-BRIDGE-two-RN-even-candidates-v1", "analysis_complete": True, "records": output,
        "predictions": {"fused": "RN_BF16(r + W*g)", "split": "RN_BF16(r + RN_BF16(W*g))"},
        "arithmetic": "exact_dyadic_integer_RN_even_no_float_or_epsilon", "compiled_path_uniquely_identified": False,
        "fused_is_ideal_single_rounding_not_FP32_FMA_claim": True,
        "FP32_FMA_then_BF16_double_rounding_not_covered": True,
        "only_two_candidate_models_tested": True, "native_product_capture_required": False,
        "native_nonfinite_inputs_rejected": True, "rounding_overflow_is_not_saturated": True,
        "new_model_calls": 0}


def _array(value, *, gate=False):
    a = np.asarray(value)
    need(a.dtype == np.uint16 and a.shape in (((1, 1024), (10, 1024)) if gate else ((10, 1024),)),
         "native_uint16_ten_positions_1024channels")
    need(not np.any(((a.astype(np.uint32) >> 7) & 255) == 255), "no_native_NaN_or_Inf")
    return np.broadcast_to(a, (10, 1024)) if gate else a


def _metrics(values, exponent):
    n = len(values)
    total, energy = sum(values), sum(v * v for v in values)
    nonzero = sum(v != 0 for v in values)
    return {"coordinates": n, "nonzero_coordinates": nonzero, "zero_coordinates": n-nonzero,
        "signed_sum_exact_integer": str(total), "square_sum_exact_integer": str(energy),
        "unit_exponent": exponent, "signed_mean": scaled(total, exponent) / n,
        "energy": scaled(energy, 2 * exponent), "RMS": math.sqrt(scaled(energy, 2 * exponent) / n),
        "maxabs": scaled(max(map(abs, values), default=0), exponent)}


def analyze_native_triplet(records, *, check=lambda: None):
    """records[role][FIELDS] are the actual, consumed BF16 uint16 leaves.

    Optional `consumed_product_native` must be supplied for all three roles
    and each role must explicitly assert `product_is_consumed is True`.
    This readout never creates such a product by replaying multiplication.
    """
    need(set(records) == set(ROLES), "exact_none_d_c_triplet")
    arrays = {r: {f: _array(records[r][f], gate=f == "gate") for f in FIELDS} for r in ROLES}
    for role in ROLES[1:]:
        for field in ("W_before", "gate", "residual_input"):
            need(np.array_equal(arrays[role][field], arrays["none"][field]), "same_B_pre_site_" + field)
    need(np.array_equal(arrays["none"]["W_before"], arrays["none"]["W_used"]), "none_actual_W_identity")
    products = ["consumed_product_native" in records[r] for r in ROLES]
    need(all(products) or not any(products), "native_product_all_roles_or_absent")
    has_product = all(products)
    if has_product:
        for role in ROLES:
            need(records[role].get("product_is_consumed") is True, "no_reconstructed_product_attribution")
            arrays[role]["consumed_product_native"] = _array(records[role]["consumed_product_native"])
    rows = []
    for name, before, after in PAIRS:
        by_position = []
        for p in range(10):
            check()
            measures = {k: [] for k in ("native_W_increment", "ideal_gated_increment", "native_after_attention_increment",
                                         "combined_native_rounding_delta", "baseline_combined_residual", "treated_combined_residual")}
            if has_product:
                measures.update(native_product_increment=[], product_rounding_delta=[], residual_addition_rounding_delta=[])
            for c in range(1024):
                wb, wt = units(arrays[before]["W_used"][p, c]), units(arrays[after]["W_used"][p, c])
                gate = units(arrays[before]["gate"][p, c])
                residual = units(arrays[before]["residual_input"][p, c])
                yb, yt = units(arrays[before]["after_attention_residual"][p, c]), units(arrays[after]["after_attention_residual"][p, c])
                dw = wt-wb
                ideal = dw*gate  # unit 2^-266
                actual = (yt-yb) << 133
                eb = (yb << 133) - ((residual << 133)+wb*gate)
                et = (yt << 133) - ((residual << 133)+wt*gate)
                difference = actual-ideal
                need(difference == et-eb, "exact_combined_rounding_difference_identity")
                for key, value in (("native_W_increment", dw), ("ideal_gated_increment", ideal),
                    ("native_after_attention_increment", actual), ("combined_native_rounding_delta", difference),
                    ("baseline_combined_residual", eb), ("treated_combined_residual", et)):
                    measures[key].append(value)
                if has_product:
                    pb, pt = units(arrays[before]["consumed_product_native"][p,c]), units(arrays[after]["consumed_product_native"][p,c])
                    product_increment = (pt-pb) << 133
                    product_error = product_increment-ideal
                    addition_error = actual-product_increment
                    need(product_error+addition_error == difference, "consumed_product_addition_exact_split")
                    for key, value in (("native_product_increment", product_increment), ("product_rounding_delta", product_error),
                                       ("residual_addition_rounding_delta", addition_error)):
                        measures[key].append(value)
            by_position.append(measures)
        panels = tuple(("position_"+str(p),p,p+1) for p in range(10)) + WINDOWS
        for window, start, end in panels:
            all_values = {k: [x for pos in by_position[start:end] for x in pos[k]] for k in by_position[0]}
            ideal, error, actual = (all_values[k] for k in ("ideal_gated_increment", "combined_native_rounding_delta", "native_after_attention_increment"))
            cross = 2 * sum(a*b for a,b in zip(ideal,error))
            ie, ee, ae = (sum(v*v for v in values) for values in (ideal,error,actual))
            need(ae == ie+ee+cross, "native_after_attention_signed_energy_closure")
            rows.append({"contrast": name, "window": window, "positions": end-start,
                "metrics": {k: _metrics(v, -133 if k == "native_W_increment" else -266) for k,v in all_values.items()},
                "ideal_error_signed_cross_exact_units_squared": str(cross),
                "ideal_error_signed_cross": scaled(cross,-532),
                "combined_error_over_ideal_energy": None if ie == 0 else ee/ie,
                "combined_error_over_actual_energy": None if ae == 0 else ee/ae,
                "ideal_nonzero_but_native_zero_coordinates": sum(i != 0 and a == 0 for i,a in zip(ideal,actual)),
                "ideal_zero_but_native_nonzero_coordinates": sum(i == 0 and a != 0 for i,a in zip(ideal,actual)),
                "exact_closure": True})
    return {"schema": "B-BRIDGE-native-quantization-readout-v1", "analysis_complete": True,
        "rows": rows, "native_product_captured": has_product,
        "attribution": "native_product_and_residual_split" if has_product else "combined_only_product_not_captured",
        "ideal_arithmetic_is_not_GPU_emulation": True, "rounding_fraction_is_not_causal_share": True,
        "new_model_calls": 0, "new_env_steps": 0, "scientific_equivalence_or_GO": False}
