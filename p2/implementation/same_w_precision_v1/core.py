"""Exact BF16 lattice arithmetic. No NumPy, model, filesystem, or network IO.

Every finite BF16 value is an integer multiple of 2**-133. Python integers
preserve the exact difference and target, including far-separated exponents.
Signed zero is retained as a bit-pattern fact, never counted as a nonzero delta.
"""
from operator import index

UNIT_EXPONENT = -133
MAX_FINITE_UNITS = 255 << 253


class PrecisionError(ValueError):
    pass


def need(ok, code):
    if not ok:
        raise PrecisionError(code)


def bits_to_units(value):
    """Return (exact signed integer, sign bit); reject NaN/Inf patterns."""
    need(not isinstance(value, bool), "boolean_is_not_bf16_bits")
    try:
        bits = index(value)
    except TypeError:
        raise PrecisionError("integer_bf16_storage_required") from None
    need(0 <= bits <= 65535, "uint16_bf16_range")
    sign, exponent, fraction = bits >> 15, (bits >> 7) & 255, bits & 127
    need(exponent != 255, "nonfinite_bf16_pattern")
    magnitude = fraction if exponent == 0 else (128 + fraction) << (exponent - 1)
    return (-magnitude if sign else magnitude), sign


def units_to_bits(value, *, zero_sign=0):
    """Exact inverse, or (None, reason); never round, clip, or fit epsilon.

    Zero's sign is not determined by exact real addition. The caller may retain
    an existing zero sign for a no-op; otherwise zero_sign=0 is only a canonical
    diagnostic encoding, not a proposed native setter target.
    """
    need(type(value) is int and type(zero_sign) is int and zero_sign in (0, 1),
         "exact_integer_units_and_zero_sign")
    if value == 0:
        return zero_sign << 15, "exact_zero"
    sign, magnitude = int(value < 0), abs(value)
    if magnitude > MAX_FINITE_UNITS:
        return None, "outside_finite_bf16_range"
    if magnitude < 128:
        return (sign << 15) | magnitude, "exact_subnormal"
    exponent = magnitude.bit_length() - 7
    shift = exponent - 1
    if magnitude & ((1 << shift) - 1):
        return None, "not_on_bf16_lattice"
    significand = magnitude >> shift
    need(1 <= exponent <= 254 and 128 <= significand <= 255, "inverse_internal_range")
    return (sign << 15) | (exponent << 7) | (significand - 128), "exact_normal"


def analyze_triplet(a, b_before, b_after, *, expected_shape=(10, 1024), check=lambda: None):
    """A=none; B_before=pulse; B_after=persistent, all at the fixed site.

    Inputs are rectangular sequences of integer BF16 bits. No target tensor or
    alternative common probe is returned. Only support/count/location metadata.
    """
    positions, channels = expected_shape
    need(type(positions) is type(channels) is int and positions > 0 and channels > 0,
         "positive_expected_shape")
    values = (a, b_before, b_after)
    need(all(len(v) == positions and all(len(row) == channels for row in v) for v in values),
         "exact_position_channel_shape")
    totals = {k: 0 for k in ("coordinates", "reference_delta_nonzero", "reference_delta_zero",
        "A_target_representable", "A_target_nonrepresentable", "A_target_representable_nonzero_delta",
        "A_target_zero", "A_target_zero_from_nonzero_delta", "A_target_overflow",
        "B_zero_sign_transitions", "A_negative_zero", "B_before_negative_zero", "B_after_negative_zero")}
    rows = []
    for p in range(positions):
        row = {k: 0 for k in totals}
        row.update(action_position=p, nonrepresentable_channels=[], overflow_channels=[])
        for c in range(channels):
            if c % 128 == 0:
                check()
            av, az = bits_to_units(a[p][c])
            bv, bz = bits_to_units(b_before[p][c])
            ev, ez = bits_to_units(b_after[p][c])
            delta = ev - bv
            target = av + delta
            # No-op retains A's signed-zero bit; other exact-zero targets have
            # two valid BF16 encodings. Neither is a new requested probe.
            encoded, reason = units_to_bits(target, zero_sign=az if delta == 0 and av == 0 else 0)
            nonzero = delta != 0
            representable = encoded is not None
            counts = {"coordinates": 1, "reference_delta_nonzero": int(nonzero),
                "reference_delta_zero": int(not nonzero), "A_target_representable": int(representable),
                "A_target_nonrepresentable": int(not representable),
                "A_target_representable_nonzero_delta": int(representable and nonzero),
                "A_target_zero": int(target == 0),
                "A_target_zero_from_nonzero_delta": int(target == 0 and nonzero),
                "A_target_overflow": int(reason == "outside_finite_bf16_range"),
                "B_zero_sign_transitions": int(bv == ev == 0 and bz != ez),
                "A_negative_zero": int(av == 0 and az == 1),
                "B_before_negative_zero": int(bv == 0 and bz == 1),
                "B_after_negative_zero": int(ev == 0 and ez == 1)}
            for key, count in counts.items():
                row[key] += count
                totals[key] += count
            if not representable:
                row["nonrepresentable_channels"].append(c)
                if reason == "outside_finite_bf16_range":
                    row["overflow_channels"].append(c)
            else:
                need(bits_to_units(encoded)[0] == target, "exact_target_inverse_closure")
            # The reference B target must remain its original finite value.
            need(bv + delta == ev and units_to_bits(ev, zero_sign=ez)[0] == index(b_after[p][c]),
                 "original_B_target_roundtrip")
        need(row["A_target_representable"] + row["A_target_nonrepresentable"] == channels,
             "complete_position_denominator")
        rows.append(row)
    need(totals["coordinates"] == positions * channels, "complete_fixed_denominator")
    if totals["reference_delta_nonzero"] == 0:
        status = "NO_GO_ZERO_REFERENCE_DELTA"
    elif totals["A_target_nonrepresentable"]:
        status = "NO_GO_NATURAL_SAME_DELTA_UNREPRESENTABLE"
    else:
        status = "PASS_EXACT_REPRESENTABILITY_ONLY"
    return {"schema": "BF16-exact-dyadic-preflight-core-v1", "analysis_complete": True,
        "status": status, "unit": "2**-133", "integer_arithmetic": "Python_unbounded_int_only",
        "shape": [positions, channels], "counts": totals, "per_position": rows,
        "signed_zero_policy": "preserve input signs and transitions; numeric zero delta is a no-op; no new zero-sign target selected",
        "same_W_setter_qualified": False, "scientific_receiver_hypothesis_rejected": False,
        "new_probe_generated": False, "epsilon_or_float_equality_used": False}
