"""Exact common finite-BF16 increment projection; pure Python and no IO.

Selection uses A, B, and the fixed B-after minus B-before increment only.
No model response, fit, numeric tolerance, renormalization, or pass threshold.
Large coordinate arrays are not returned; integer geometry sums are preserved.
"""
from __future__ import annotations

import math

from exact_bf16 import bits_to_units, units_to_bits, MAX_FINITE_UNITS


UNIT_EXPONENT = -133
ENERGY_EXPONENT = 2 * UNIT_EXPONENT
COUNT_NAMES = (
    "coordinates", "reference_nonzero", "reference_zero", "natural_common_representable",
    "natural_not_common_representable", "unchanged", "nonzero_preserved", "strictly_shrunk_nonzero",
    "zeroed_reference_nonzero", "zero_reference_kept", "candidate_nonzero", "sign_reversals",
    "magnitude_increases", "natural_failed_rescued_nonzero", "A_target_zero", "B_target_zero",
    "A_negative_zero_input", "B_negative_zero_input", "B_after_negative_zero_input",
)
ENERGY_NAMES = ("reference_energy", "candidate_energy", "error_energy", "candidate_dot_reference",
                "initial_failed_reference_energy", "initial_failed_candidate_energy", "initial_failed_error_energy",
                "zero_drop_error_energy", "zero_drop_candidate_energy", "zero_drop_dot_reference")


def need(ok, message):
    if not ok:
        raise ValueError(message)


def _finite_bins():
    # [-255,255] joins subnormals and the first normal exponent, all stride1.
    negative = [(-(255 << k), -(128 << k), 1 << k) for k in range(253, 0, -1)]
    positive = [(128 << k, 255 << k, 1 << k) for k in range(1, 254)]
    return tuple(negative + [(-255, 255, 1)] + positive)


FINITE_BINS = _finite_bins()


def _increment_bins(baseline, sign, limit):
    """Sorted u intervals and residues such that baseline + sign*u is BF16."""
    ordered = FINITE_BINS if sign > 0 else reversed(FINITE_BINS)
    out = []
    for low, high, stride in ordered:
        lo, hi = (low-baseline, high-baseline) if sign > 0 else (baseline-high, baseline-low)
        lo, hi = max(0, lo), min(limit, hi)
        if lo <= hi:
            out.append((lo, hi, stride, (-sign*baseline) % stride))
    return out


def project_common_units(a, b, d, *, check=lambda: None):
    """Unique closest same-sign nonamplified increment with TWO native targets.

    All inputs/outputs are exact Python integers in units2^-133. d itself need
    not be a BF16 value. a and b must be finite BF16; b+d must be finite BF16
    because the reference is an actual B-after minus B-before difference.
    """
    need(all(type(v) is int for v in (a, b, d)), "Python_integer_dyadic_units")
    need(units_to_bits(a)[0] is not None and units_to_bits(b)[0] is not None,
         "finite_BF16_baselines")
    need(units_to_bits(b+d)[0] is not None, "native_reference_B_after")
    if d == 0 or units_to_bits(a+d)[0] is not None:
        return d
    sign, limit = (1 if d > 0 else -1), abs(d)
    aa, bb = _increment_bins(a, sign, limit), _increment_bins(b, sign, limit)
    ia = ib = 0
    best = 0  # zero always feasible because both original baselines are finite.
    while ia < len(aa) and ib < len(bb):
        check()
        la, ha, qa, ra = aa[ia]
        lb, hb, qb, rb = bb[ib]
        low, high = max(la, lb), min(ha, hb)
        # Power-of-two strides are nested: compatible iff residues agree mod min.
        if low <= high and (ra-rb) % min(qa, qb) == 0:
            stride, residue = (qa, ra) if qa >= qb else (qb, rb)
            candidate = high - ((high-residue) % stride)
            if candidate >= low:
                best = max(best, candidate)
        if ha <= hb:
            ia += 1
        if hb <= ha:
            ib += 1
    result = sign*best
    need(abs(result) <= abs(d) and result*d >= 0 and
         units_to_bits(a+result)[0] is not None and units_to_bits(b+result)[0] is not None,
         "exact_selected_common_increment")
    return result


def diagnostic_target_bits(baseline_bits, increment_units):
    """Deterministic diagnostic encoding, not a qualified native setter.

    A numeric no-op preserves original zero's sign. A nonzero increment landing
    at exact zero uses +0; that zero-sign convention is explicit and fixed.
    """
    value, sign = bits_to_units(baseline_bits)
    target = value + increment_units
    bits, reason = units_to_bits(target, zero_sign=sign if increment_units == 0 and value == 0 else 0)
    need(bits is not None, "diagnostic_target_must_be_exactly_representable")
    return bits


def _scaled(integer, exponent):
    # BF16² sums for the prescribed10240 coordinates fit FP64 exponent range.
    # Selection and all sums were completed as Python integers before this step.
    value = math.ldexp(float(integer), exponent)
    need(math.isfinite(value), "finite_small_float_reporting_range")
    return value


def _ratio(numerator, denominator):
    return None if denominator == 0 else numerator/denominator


def _report_counts_energy(counts, energy):
    d2, c2, e2, dot = (energy[k] for k in ENERGY_NAMES[:4])
    need(e2 == d2+c2-2*dot, "exact_error_energy_closure")
    need(0 <= c2 <= dot <= d2 and 0 <= e2 <= d2 and e2 <= energy["zero_drop_error_energy"],
         "same_sign_nonamplified_geometry_bounds")
    need(energy["zero_drop_error_energy"] == energy["initial_failed_reference_energy"],
         "zero_drop_error_equals_failed_reference_energy")
    det = c2*d2-dot*dot
    need(det >= 0, "exact_Cauchy_Schwarz")
    # Algebraic denominator tests use exact integers, never epsilon.
    if c2 == 0 or d2 == 0:
        cosine = angle = None
        direction_status = "UNDEFINED_ZERO_VECTOR"
    else:
        cosine = math.sqrt((dot*dot)/(c2*d2))
        angle = math.degrees(math.atan2(math.sqrt(det/(c2*d2)), cosine))
        direction_status = "DEFINED"
    return {"counts": counts,
            "energy_exact_integer_units_squared": {k: str(v) for k, v in energy.items()},
            "energy": {k: _scaled(v, ENERGY_EXPONENT) for k, v in energy.items()},
            "relative_error_energy": _ratio(e2, d2), "retained_reference_energy": _ratio(c2, d2),
            "candidate_to_reference_norm_ratio": None if d2 == 0 else math.sqrt(c2/d2),
            "relative_error_norm": None if d2 == 0 else math.sqrt(e2/d2),
            "initial_failed_reference_energy_fraction": _ratio(energy["initial_failed_reference_energy"], d2),
            "zero_drop_relative_error_energy": _ratio(energy["zero_drop_error_energy"], d2),
            "candidate_minus_zero_drop_error_energy": _scaled(e2-energy["zero_drop_error_energy"], ENERGY_EXPONENT),
            "direction_status": direction_status, "cosine_to_reference": cosine,
            "angle_to_reference_degrees": angle,
            "parallel_gain": _ratio(dot, d2),
            "orthogonal_error_energy": None if d2 == 0 else math.ldexp(det/d2, ENERGY_EXPONENT),
            "orthogonal_error_exact_rational_units_squared": None if d2 == 0 else {"numerator": str(det), "denominator": str(d2)},
            "error_dot_reference": _scaled(dot-d2, ENERGY_EXPONENT),
            "error_dot_candidate": _scaled(c2-dot, ENERGY_EXPONENT),
            "exact_energy_closure": True}


def _signed_sum_report(dsum, csum, d_abs, c_abs, *, coordinate_space):
    err = [c-d for c, d in zip(csum, dsum)]
    d2, c2, e2, dot = (sum(v*v for v in dsum), sum(v*v for v in csum),
                       sum(v*v for v in err), sum(c*d for c, d in zip(csum, dsum)))
    need(e2 == d2+c2-2*dot, "exact_signed_sum_error_closure")
    return {"coordinate_space": coordinate_space,
            "reference_sum_exact_units": [str(v) for v in dsum],
            "candidate_sum_exact_units": [str(v) for v in csum],
            "change_sum_exact_units": [str(v) for v in err],
            "reference_sum": [_scaled(v, UNIT_EXPONENT) for v in dsum],
            "candidate_sum": [_scaled(v, UNIT_EXPONENT) for v in csum],
            "change_sum": [_scaled(v, UNIT_EXPONENT) for v in err],
            "reference_sum_energy": _scaled(d2, ENERGY_EXPONENT),
            "candidate_sum_energy": _scaled(c2, ENERGY_EXPONENT),
            "sum_error_energy": _scaled(e2, ENERGY_EXPONENT),
            "candidate_dot_reference_sum": _scaled(dot, ENERGY_EXPONENT),
            "exact_sum_energy_units_squared": {"reference": str(d2), "candidate": str(c2), "error": str(e2), "dot": str(dot)},
            "reference_sum_absolute_coordinate_magnitudes": _scaled(d_abs, UNIT_EXPONENT),
            "candidate_sum_absolute_coordinate_magnitudes": _scaled(c_abs, UNIT_EXPONENT),
            "interpretation": "coordinate_basis_dependent_signed_cancellation_diagnostic_not_action_or_motion"}


def analyze_triplet(a_bits, b_before_bits, b_after_bits, *, expected_shape=(10, 1024), check=lambda: None):
    """In-memory common candidate geometry; return small JSON, never target arrays.

    Pure exact projection at every coordinate. Preserves full reference support,
    real zeros, initial failures and position labels; does not select a subset.
    """
    positions, channels = expected_shape
    need(type(positions) is type(channels) is int and positions > 0 and channels > 0,
         "positive_position_channel_shape")
    values = (a_bits, b_before_bits, b_after_bits)
    need(all(len(v) == positions and all(len(row) == channels for row in v) for v in values),
         "exact_position_channel_shape")
    totals = dict.fromkeys(COUNT_NAMES, 0)
    energies = dict.fromkeys(ENERGY_NAMES, 0)
    per_position, row_d, row_c, row_error = [], [], [], []
    for p in range(positions):
        counts, energy = dict.fromkeys(COUNT_NAMES, 0), dict.fromkeys(ENERGY_NAMES, 0)
        ds, cs = [], []
        affected = {"initial_not_common_channels": [], "shrunk_nonzero_channels": [], "zeroed_nonzero_channels": []}
        for ch in range(channels):
            if ch % 128 == 0:
                check()
            a, az = bits_to_units(a_bits[p][ch])
            b, bz = bits_to_units(b_before_bits[p][ch])
            after, ez = bits_to_units(b_after_bits[p][ch])
            d = after-b
            natural = units_to_bits(a+d)[0] is not None
            c = project_common_units(a, b, d, check=check)
            error = c-d
            drop = d if natural else 0
            if not natural:
                affected["initial_not_common_channels"].append(ch)
            if 0 < abs(c) < abs(d):
                affected["shrunk_nonzero_channels"].append(ch)
            if d != 0 and c == 0:
                affected["zeroed_nonzero_channels"].append(ch)
            need((not natural or c == d) and c*d >= 0 and abs(c) <= abs(d), "preserve_natural_and_sign_nonamplification")
            for baseline, target in ((a_bits[p][ch], a+c), (b_before_bits[p][ch], b+c)):
                need(bits_to_units(diagnostic_target_bits(baseline, c))[0] == target, "common_target_exact_roundtrip")
            observed = {
                "coordinates": 1, "reference_nonzero": int(d != 0), "reference_zero": int(d == 0),
                "natural_common_representable": int(natural), "natural_not_common_representable": int(not natural),
                "unchanged": int(c == d), "nonzero_preserved": int(d != 0 and c == d),
                "strictly_shrunk_nonzero": int(0 < abs(c) < abs(d)), "zeroed_reference_nonzero": int(d != 0 and c == 0),
                "zero_reference_kept": int(d == c == 0), "candidate_nonzero": int(c != 0),
                "sign_reversals": int(c*d < 0), "magnitude_increases": int(abs(c) > abs(d)),
                "natural_failed_rescued_nonzero": int(not natural and c != 0),
                "A_target_zero": int(a+c == 0), "B_target_zero": int(b+c == 0),
                "A_negative_zero_input": int(a == 0 and az), "B_negative_zero_input": int(b == 0 and bz),
                "B_after_negative_zero_input": int(after == 0 and ez),
            }
            ev = {"reference_energy": d*d, "candidate_energy": c*c, "error_energy": error*error,
                  "candidate_dot_reference": c*d, "initial_failed_reference_energy": 0 if natural else d*d,
                  "initial_failed_candidate_energy": 0 if natural else c*c,
                  "initial_failed_error_energy": 0 if natural else error*error,
                  "zero_drop_error_energy": (drop-d)**2, "zero_drop_candidate_energy": drop*drop,
                  "zero_drop_dot_reference": drop*d}
            for key in COUNT_NAMES:
                counts[key] += observed[key]; totals[key] += observed[key]
            for key in ENERGY_NAMES:
                energy[key] += ev[key]; energies[key] += ev[key]
            ds.append(d); cs.append(c)
        need(counts["coordinates"] == channels and
             counts["nonzero_preserved"]+counts["strictly_shrunk_nonzero"]+counts["zeroed_reference_nonzero"] == counts["reference_nonzero"],
             "full_position_support_partition")
        row = {"action_position": p, **_report_counts_energy(counts, energy), "affected_support": affected}
        row["signed_channel_sum"] = _signed_sum_report([sum(ds)], [sum(cs)], sum(map(abs, ds)), sum(map(abs, cs)),
                                                       coordinate_space="scalar_sum_across_this_position_channels")
        per_position.append(row); row_d.append(ds); row_c.append(cs)
        row_error.append([c-d for c, d in zip(cs, ds)])
    # Position vectors share W channel coordinates. These signed Gram tables
    # expose cancellations across positions without pretending positions add as actions.
    def gram(rows):
        return [[sum(x*y for x, y in zip(a, b)) for b in rows] for a in rows]
    dgram, cgram, egram = gram(row_d), gram(row_c), gram(row_error)
    dsum = [sum(row[ch] for row in row_d) for ch in range(channels)]
    csum = [sum(row[ch] for row in row_c) for ch in range(channels)]
    cross_d = sum(map(sum, dgram))-energies["reference_energy"]
    cross_c = sum(map(sum, cgram))-energies["candidate_energy"]
    need(sum(x*x for x in dsum) == energies["reference_energy"]+cross_d and
         sum(x*x for x in csum) == energies["candidate_energy"]+cross_c, "exact_cross_position_energy_closure")
    full = _report_counts_energy(totals, energies)
    full["signed_all_coordinate_sum"] = _signed_sum_report([sum(dsum)], [sum(csum)],
                                                           sum(abs(x) for row in row_d for x in row),
                                                           sum(abs(x) for row in row_c for x in row),
                                                           coordinate_space="scalar_sum_over_position_and_channel")
    # Do not return1024-channel signed sums: aggregated energies/cross and per-position
    # scalar sums suffice for small reporting, with exact integers preserving auditability.
    summed_report = _signed_sum_report(dsum, csum, sum(map(abs, dsum)), sum(map(abs, csum)),
                                      coordinate_space="sum_across_positions_preserving_W_channels")
    for key in ("reference_sum_exact_units", "candidate_sum_exact_units", "change_sum_exact_units",
                "reference_sum", "candidate_sum", "change_sum"):
        summed_report.pop(key)
    return {"schema": "BF16-common-probe-exact-geometry-v1", "analysis_complete": True,
            "shape": [positions, channels], "unit": "2**-133", "energy_unit": "2**-266",
            "selection_rule": "coordinatewise_closest_same_sign_nonamplified_common_finite_BF16_increment",
            "optimality": "unique_numeric_minimizer_of_sum_coordinate_squared_distortion_under_fixed_constraints",
            "full": full, "per_position": per_position,
            "cross_position": {"reference_Gram_exact_units_squared": [[str(v) for v in r] for r in dgram],
                               "candidate_Gram_exact_units_squared": [[str(v) for v in r] for r in cgram],
                               "error_Gram_exact_units_squared": [[str(v) for v in r] for r in egram],
                               "reference_Gram": [[_scaled(v, ENERGY_EXPONENT) for v in r] for r in dgram],
                               "candidate_Gram": [[_scaled(v, ENERGY_EXPONENT) for v in r] for r in cgram],
                               "error_Gram": [[_scaled(v, ENERGY_EXPONENT) for v in r] for r in egram],
                               "reference_signed_cross_energy": _scaled(cross_d, ENERGY_EXPONENT),
                               "candidate_signed_cross_energy": _scaled(cross_c, ENERGY_EXPONENT),
                               "signed_cross_energy_change": _scaled(cross_c-cross_d, ENERGY_EXPONENT),
                               "summed_position_vector": summed_report, "exact_energy_closure": True},
            "zero_encoding_tie_rule": "numeric_noop_preserve_each_baseline_zero_sign_other_exact_zero_canonical_plus",
            "candidate_geometry_computed": True, "native_target_arrays_returned": False,
            "same_W_setter_qualified": False, "model_or_behavior_effect_evaluated": False,
            "scientific_acceptance_threshold": None, "reference_energy_fraction_not_equal_functional_importance": True,
            "new_model_calls": 0, "new_env_steps": 0}
