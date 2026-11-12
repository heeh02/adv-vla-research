"""New mass-scaling baselines from frozen natural-reading-use corner Grams.

Pure NumPy, no IO, no original tensor reads, no fit, no model or cloud calls.
This module does not reconstruct signed vectors, spatial alpha, or native W.
"""
from __future__ import annotations

import numpy as np


CORNERS = ("F00", "F10", "F01", "F11")
TERMS = ("mass", "within", "interaction")
WINDOWS = (("full10", 0, 10), ("first5", 0, 5), ("tail5", 5, 10))
BASELINES = (
    "zero_change", "global_window_prefix_mass", "per_position_prefix_mass",
    "per_head_position_prefix_mass", "modality_mass", "within_modality", "signed_scalar_oracle", "full_late",
)
SCALINGS = BASELINES[1:4]
VIEWS = ("full", "response_B3", "response_residual", "clean_B3", "clean_residual")
MAI_COEFFICIENTS = np.array(((-1., 1., 0., 0.), (-1., 0., 1., 0.),
                             (1., -1., -1., 1.)))
DELTA_COEFFICIENTS = np.array((-1., 0., 0., 1.))
# A quadratic form uses at most sixteen products. This bound concerns FP64
# accumulation cancellation only, never BF16 capture or a scientific threshold.
ROUNDOFF_FACTOR = 64.0
PROJECTION_RTOL = 1e-10


def _need(condition, code):
    if not condition:
        raise ValueError(code)


def unpack_corner_gram(packed):
    """Mirror the recorded upper triangle, without averaging or re-fitting."""
    p = np.asarray(packed)
    _need(p.dtype == np.float64 and p.ndim == 3 and p.shape[1:] == (10, 10),
          "packed_Gram_FP64_head_10positions_upper4x4")
    _need(not np.isinf(p).any(), "Gram_no_infinity")
    g = np.empty(p.shape[:2] + (4, 4), np.float64)
    i, j = np.triu_indices(4)
    g[..., i, j] = p
    g[..., j, i] = p
    return g


def quadratic_energy(gram, coefficients):
    """Return energy, availability, raw value, bound, and roundoff correction.

    Only nonzero coefficient pairs require defined Gram entries. An undefined
    unused corner cannot contaminate a known endpoint or another baseline.
    Tiny *negative* cancellation is corrected to zero and explicitly retained
    in the report; any more negative form raises rather than hiding bad input.
    Small positive energies remain positive, with no epsilon zero-target rule.
    """
    g = np.asarray(gram, np.float64)
    q = np.asarray(coefficients, np.float64)
    _need(g.shape[-2:] == (4, 4) and q.shape[-1:] == (4,), "quadratic_axes")
    _need(np.isfinite(q).all() and not np.isinf(g).any(), "finite_coefficients_no_infinite_Gram")
    weights = q[..., :, None] * q[..., None, :]
    active = weights != 0
    defined = np.all(~active | np.isfinite(g), axis=(-2, -1))
    products = np.where(active, g, 0.) * weights
    raw = products.sum(axis=(-2, -1), dtype=np.float64)
    scale = np.abs(products).sum(axis=(-2, -1), dtype=np.float64)
    bound = ROUNDOFF_FACTOR * np.finfo(np.float64).eps * scale
    _need(np.all(~defined | (raw >= -bound)), "material_negative_quadratic_energy")
    corrected = defined & (raw < 0)
    value = np.where(defined, np.maximum(raw, 0.), np.nan)
    return {"energy": value, "defined": defined, "raw": raw,
            "roundoff_bound": bound, "corrected": corrected}


def _scale(source, target):
    """A zero mass has no allocation; both-zero contribution is still zero."""
    ratio = np.full(np.broadcast_shapes(source.shape, target.shape), np.nan)
    np.divide(target, source, out=ratio, where=source > 0)
    identifiable = source > 0
    defined = identifiable | (target == 0)
    # Internal zero is used only for a uniquely zero contribution. Public
    # coefficients remain NaN when their ratio is not identifiable.
    computational = np.where(identifiable, ratio, 0.)
    return ratio, defined, identifiable, computational


def analyze_saved_grams(raw_gram_upper, masses, corner_defined, *,
                        projection_gram_upper=None, group_corner_defined=None,
                        flow_pair=None):
    """Analyze one frozen condition/layer/flow pair; identities stay external.

    raw_gram_upper [H,10,10]; masses [early/late,H,10,3]; corner_defined
    [F00/F10/F01/F11,H,10]. Optional projection_gram_upper is
    [response/clean,B3/residual,H,10,10], computed before WO in the old run.
    Optional group_corner_defined [4,H,10,3] checks zero-mass semantics.
    Arrays must be the original FP64/bool artifact fields, not inferred from
    norms. No original P row is normalized here; masses are used as recorded.
    """
    g = unpack_corner_gram(raw_gram_upper)
    heads, positions = g.shape[:2]
    m = np.asarray(masses)
    cd = np.asarray(corner_defined)
    _need(m.dtype == np.float64 and m.shape == (2, heads, positions, 3),
          "original_FP64_mass_endpoint_head_position_modality")
    _need(np.isfinite(m).all() and (m >= 0).all(), "finite_nonnegative_recorded_mass")
    _need(cd.dtype == bool and cd.shape == (4, heads, positions), "boolean_corner_defined_axes")
    _need(np.array_equal(cd[0], cd[3]), "same_early_late_query_support")
    qd = cd[0]
    _need(np.array_equal(qd, np.broadcast_to(qd[:1], qd.shape)), "query_support_shared_across_heads")
    pairs = ((0, 0), (1, 0), (0, 1), (1, 1))
    expected_group = np.stack([((m[a] == 0) | (m[b] > 0)) & qd[..., None]
                               for a, b in pairs])
    _need(np.array_equal(cd, expected_group.all(-1)), "corner_zero_mass_identifiability")
    if group_corner_defined is not None:
        gc = np.asarray(group_corner_defined)
        _need(gc.dtype == bool and gc.shape == expected_group.shape and
              np.array_equal(gc, expected_group), "original_group_corner_defined_matches_mass")
    expected_entries = np.moveaxis(cd, 0, -1)
    expected_entries = expected_entries[..., :, None] & expected_entries[..., None, :]
    _need(np.array_equal(np.isfinite(g), expected_entries), "Gram_defined_mask_matches_corners")
    diagonal = np.diagonal(g, axis1=-2, axis2=-1)
    _need(np.all(diagonal[np.isfinite(diagonal)] >= 0), "saved_corner_norm_nonnegative")
    if flow_pair is not None:
        _need(len(flow_pair) == 2 and tuple(flow_pair) in ((0, 5), (5, 9), (0, 9)),
              "predeclared_flow_pair")

    grams = [g]
    checks = {"input_reconstruction": "saved_Grams_only_no_P_V_or_alpha_read",
              "negative_roundoff_factor": ROUNDOFF_FACTOR,
              "negative_roundoff_formula": "64 * eps64 * sum(abs(q_i G_ij q_j))",
              "projection_sum_maxabs": {}}
    if projection_gram_upper is not None:
        pg = np.asarray(projection_gram_upper)
        _need(pg.dtype == np.float64 and pg.shape == (2, 2, heads, positions, 10),
              "projection_Gram_response_clean_B3_residual_axes")
        for bi, basis in enumerate(("response", "clean")):
            parts = [unpack_corner_gram(pg[bi, part]) for part in range(2)]
            for part in parts:
                _need(np.array_equal(np.isfinite(part), expected_entries),
                      "projection_Gram_defined_mask_matches_corners")
            error = (g - parts[0] - parts[1])[expected_entries]
            scale = (np.abs(g) + np.abs(parts[0]) + np.abs(parts[1]))[expected_entries]
            _need(np.all(np.abs(error) <= PROJECTION_RTOL * np.maximum(1., scale)),
                  "orthogonal_pre_WO_basis_split_Gram_closure")
            checks["projection_sum_maxabs"][basis] = float(np.max(np.abs(error), initial=0.))
            grams.extend(parts)
    views = VIEWS if len(grams) == 5 else VIEWS[:1]
    prefix_mass = m.sum(-1)
    mass0, mass1 = prefix_mass
    per_p = _scale(mass0.sum(0), mass1.sum(0))
    per_hp = _scale(mass0, mass1)
    scale_values = np.full((len(WINDOWS), 3, heads, positions), np.nan)
    scale_defined = np.zeros_like(scale_values, dtype=bool)
    scale_identified = np.zeros_like(scale_values, dtype=bool)
    baseline_defined = np.zeros((len(WINDOWS), len(BASELINES), heads, positions), bool)
    sse_shape = (len(grams), len(WINDOWS), len(BASELINES), heads, positions)
    sse = np.full(sse_shape, np.nan)
    sse_raw = np.full(sse_shape, np.nan)
    sse_bound = np.full(sse_shape, np.nan)
    corrected = np.zeros(sse_shape, bool)
    target = np.full((len(grams), heads, positions), np.nan)
    target_raw = target.copy()
    target_bound = target.copy()
    target_corrected = np.zeros_like(target, bool)
    oracle_alpha = np.full(len(WINDOWS), np.nan)
    oracle_source_energy = np.full(len(WINDOWS), np.nan)
    oracle_source_zero = np.zeros(len(WINDOWS), bool)
    full_target_values = quadratic_energy(g, DELTA_COEFFICIENTS)["energy"]
    for wi, (_, start, end) in enumerate(WINDOWS):
        if qd[:, start:end].all():
            source = float(g[:, start:end, 0, 0].sum())
            dot = float(g[:, start:end, 0, 3].sum())
            oracle_source_energy[wi] = source
            oracle_source_zero[wi] = source == 0
            if source > 0:
                oracle_alpha[wi] = dot / source
            else:
                _need(dot == 0, "zero_source_energy_requires_zero_endpoint_dot")
    # Signed inner products identify cancellation without reconstructing
    # coordinates. Undefined corner rows are never silently treated as zero.
    term_grams = np.stack([np.einsum("ai,hpij,bj->hpab", MAI_COEFFICIENTS, gg,
                                   MAI_COEFFICIENTS, optimize=True) for gg in grams])
    rows = []
    for vi, gg in enumerate(grams):
        t = quadratic_energy(gg, DELTA_COEFFICIENTS)
        target[vi] = np.where(qd, t["energy"], np.nan)
        target_raw[vi] = np.where(qd, t["raw"], np.nan)
        target_bound[vi] = np.where(qd, t["roundoff_bound"], np.nan)
        target_corrected[vi] = qd & t["corrected"]
        for wi, (window, start, end) in enumerate(WINDOWS):
            sl = slice(start, end)
            scope = np.zeros_like(qd)
            scope[:, sl] = True
            query_complete = bool(qd[:, sl].all())
            global_s = _scale(np.asarray(mass0[:, sl].sum()), np.asarray(mass1[:, sl].sum()))
            scales = (global_s, per_p, per_hp)
            coeffs = []
            validity = []
            noops = []
            # Error is late - estimated late; target is late - early.
            qzero = np.broadcast_to(DELTA_COEFFICIENTS, qd.shape + (4,)).copy()
            coeffs.append(qzero)
            validity.append(qd)
            noops.append(np.zeros_like(qd))
            for si, (rat, known, identified, internal) in enumerate(scales):
                q = np.zeros(qd.shape + (4,))
                q[..., 0] = -internal
                q[..., 3] = 1.
                coeffs.append(q)
                ok = np.broadcast_to(known, qd.shape) & qd
                if si == 0 and not query_complete:
                    ok = np.zeros_like(ok)
                validity.append(ok)
                noops.append(np.broadcast_to(known & ~identified, qd.shape) & ok)
                scale_values[wi, si, :, sl] = np.broadcast_to(rat, qd.shape)[:, sl]
                scale_defined[wi, si, :, sl] = ok[:, sl]
                scale_identified[wi, si, :, sl] = (np.broadcast_to(identified, qd.shape) & ok)[:, sl]
            for ci in (1, 2):
                q = np.zeros(qd.shape + (4,))
                q[..., 3] = 1.
                q[..., ci] -= 1.
                coeffs.append(q)
                validity.append(qd & cd[ci])
                noops.append(np.zeros_like(qd))
            # The historical oracle is computed once in full space for this
            # window, then the SAME scalar is applied in every projected view.
            qo = np.zeros(qd.shape + (4,))
            qo[..., 0] = 0. if oracle_source_zero[wi] or not query_complete else -oracle_alpha[wi]
            qo[..., 3] = 1.
            coeffs.append(qo)
            validity.append(qd & query_complete)
            noops.append(np.zeros_like(qd))
            coeffs.append(np.zeros(qd.shape + (4,)))
            validity.append(qd)
            noops.append(np.zeros_like(qd))
            base_rows = {}
            total = float(target[vi, :, sl].sum()) if query_complete else None
            full_total = float(full_target_values[:, sl].sum()) if query_complete else None
            for mi, (name, q, ok, noop) in enumerate(zip(BASELINES, coeffs, validity, noops)):
                result = quadratic_energy(gg, q)
                use = scope & ok & result["defined"]
                baseline_defined[wi, mi] = use
                sse[vi, wi, mi] = np.where(use, result["energy"], np.nan)
                sse_raw[vi, wi, mi] = np.where(use, result["raw"], np.nan)
                sse_bound[vi, wi, mi] = np.where(use, result["roundoff_bound"], np.nan)
                corrected[vi, wi, mi] = use & result["corrected"]
                complete = bool(use[:, sl].all())
                value = float(sse[vi, wi, mi, :, sl].sum()) if complete else None
                noop_count = int(noop[:, sl].sum())
                base_rows[name] = {
                    "status": "UNAVAILABLE_QUERY" if not query_complete else
                              "UNAVAILABLE_ZERO_MASS_ALLOCATION" if not complete else
                              "ZERO_TARGET" if total == 0 else
                              "SOURCE_ZERO_NOOP" if name == "signed_scalar_oracle" and oracle_source_zero[wi] else
                              "DEFINED_WITH_ZERO_MASS_NOOP" if noop_count else "DEFINED",
                    "SSE": value,
                    "relative_SSE": None if value is None or total is None or total <= 0 else value / total,
                    "relative_SSE_to_full_target": None if value is None or full_total is None or full_total <= 0 else value / full_total,
                    "comparison_role": "diagnostic_only" if name == "full_late" else "baseline",
                    "zero_mass_noop_head_positions": noop_count,
                    "defined_head_positions": int(use[:, sl].sum()),
                    "planned_head_positions": heads * (end - start),
                    "negative_roundoff_corrections": int(corrected[vi, wi, mi, :, sl].sum()),
                }
            tg = term_grams[vi, :, sl]
            decomp_complete = query_complete and bool(cd[:, :, sl].all())
            decomp = {"status": "DEFINED" if decomp_complete else "UNAVAILABLE_MIXED_CORNER",
                      "term_gram": None, "component_energy_sum": None,
                      "signed_cross_sum": None, "energy_closure_error": None}
            if decomp_complete:
                tgs = tg.sum(axis=(0, 1))
                diag = float(np.trace(tgs))
                cross = float(2 * (tgs[0, 1] + tgs[0, 2] + tgs[1, 2]))
                closure = total - (diag + cross)
                # Bound cancellation by the underlying corner Gram products,
                # not by the possibly near-zero target energy.
                allowance = PROJECTION_RTOL * max(1., float(np.abs(gg[:, sl]).sum()))
                _need(abs(closure) <= allowance, "signed_MAI_cross_energy_closure")
                decomp.update(term_gram=tgs.tolist(), component_energy_sum=diag,
                              signed_cross_sum=cross, energy_closure_error=closure)
            head_sse = base_rows["per_head_position_prefix_mass"]["SSE"]
            modal_sse = base_rows["modality_mass"]["SSE"]
            gain = None if head_sse is None or modal_sse is None else head_sse - modal_sse
            rows.append({"view": views[vi], "window": window,
                         "flow_pair": None if flow_pair is None else list(flow_pair),
                         "target_delta_energy": total,
                         "full_target_delta_energy": full_total,
                         "head_minus_modality_SSE": gain,
                         "head_minus_modality_SSE_over_full_target": None if gain is None or full_total is None or full_total <= 0 else gain / full_total,
                         "oracle_full_window_alpha": None if not np.isfinite(oracle_alpha[wi]) else float(oracle_alpha[wi]),
                         "oracle_full_window_source_energy": None if not query_complete else float(oracle_source_energy[wi]),
                         "status": "UNAVAILABLE_QUERY" if not query_complete else
                                   "ZERO_TARGET" if total == 0 else "DEFINED",
                         "baselines": base_rows, "decomposition": decomp})
    checks["negative_SSE_corrections"] = int(corrected.sum())
    checks["negative_target_corrections"] = int(target_corrected.sum())
    checks["max_corrected_SSE_magnitude"] = float(np.max(-sse_raw[corrected], initial=0.))
    checks["max_SSE_roundoff_bound"] = float(np.max(np.where(np.isfinite(sse_bound), sse_bound, 0.)))
    checks["part_energy_closure"] = []
    if len(grams) == 5:
        for basis, bi in (("response", 1), ("clean", 3)):
            for wi, (window, _, _) in enumerate(WINDOWS):
                fullrow, brow, rrow = rows[wi], rows[bi*3+wi], rows[(bi+1)*3+wi]
                for name in BASELINES:
                    values = [r["baselines"][name]["SSE"] for r in (fullrow, brow, rrow)]
                    remainder = None if any(v is None for v in values) else values[0]-values[1]-values[2]
                    if remainder is not None:
                        _need(abs(remainder) <= PROJECTION_RTOL * max(1., sum(abs(v) for v in values)),
                              "same_baseline_B3_residual_SSE_closure")
                    checks["part_energy_closure"].append({"basis": basis, "window": window,
                                                         "baseline": name, "full_minus_B3_minus_residual": remainder})
    arrays = {
        "prefix_mass": prefix_mass.copy(), "scales": scale_values,
        "scale_defined": scale_defined, "scale_identified": scale_identified,
        "target_energy": target, "target_energy_raw": target_raw,
        "target_roundoff_bound": target_bound, "target_roundoff_corrected": target_corrected,
        "baseline_SSE": sse, "baseline_SSE_raw": sse_raw,
        "baseline_SSE_roundoff_bound": sse_bound, "baseline_SSE_roundoff_corrected": corrected,
        "baseline_defined": baseline_defined, "term_gram": term_grams,
        "oracle_full_window_alpha": oracle_alpha,
        "oracle_full_window_source_energy": oracle_source_energy,
        "oracle_source_zero": oracle_source_zero,
    }
    return {"rows": rows, "arrays": arrays, "checks": checks,
            "schema": "SAVED-GRAM-PREFIX-MASS-BASELINES-v1",
            "views": list(views), "baselines": list(BASELINES), "windows": [w[0] for w in WINDOWS],
            "precision": "original_saved_D32_FP64_corner_Grams; not native_D64 or W",
            "scope": "conditional_reconstruction_using_observed_late_mass_no_new_fit",
            "unavailable": ["signed_vector_coordinates", "spatial_alpha", "native_W", "causal_or_behavior_effect"]}


def aggregate_absolute(rows):
    """Equal-weight complete declared rows; no survivor reweighting.

    Caller supplies the entire planned membership for exactly one view/window
    and cohort. Hierarchical root/task aggregation repeats this operation on
    absolute mean quantities; it never averages relative SSEs.
    """
    _need(len(rows) > 0, "nonempty_planned_rows")
    _need(len({(r["view"], r["window"]) for r in rows}) == 1, "one_view_window_per_aggregate")
    view, window = rows[0]["view"], rows[0]["window"]
    targets = [r["target_delta_energy"] for r in rows]
    target = None if any(x is None for x in targets) else float(np.mean(targets))
    full_targets = [r["full_target_delta_energy"] for r in rows]
    full_target = None if any(x is None for x in full_targets) else float(np.mean(full_targets))
    output = {"view": view, "window": window, "planned_rows": len(rows),
              "target_delta_energy": target, "full_target_delta_energy": full_target, "baselines": {}}
    for name in BASELINES:
        values = [r["baselines"][name]["SSE"] for r in rows]
        value = None if any(x is None for x in values) else float(np.mean(values))
        output["baselines"][name] = {
            "SSE": value, "relative_SSE": None if value is None or target is None or target <= 0 else value / target,
            "relative_SSE_to_full_target": None if value is None or full_target is None or full_target <= 0 else value / full_target,
            "comparison_role": "diagnostic_only" if name == "full_late" else "baseline",
            "observed_rows": sum(x is not None for x in values), "planned_rows": len(rows),
            "status": "UNAVAILABLE" if value is None or target is None else "ZERO_TARGET" if target == 0 else "DEFINED",
        }
    head = output["baselines"]["per_head_position_prefix_mass"]["SSE"]
    modal = output["baselines"]["modality_mass"]["SSE"]
    gain = None if head is None or modal is None else head-modal
    output["head_minus_modality_SSE"] = gain
    output["head_minus_modality_SSE_over_full_target"] = None if gain is None or full_target is None or full_target <= 0 else gain/full_target
    return output
