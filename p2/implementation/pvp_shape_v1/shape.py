"""Descriptive response projections from three same-reference RMS summaries."""
import math
from itertools import product
import audit as A

MOMENT_TOL = 1e-10


def project(rs, rp, rd, positions):
    """Per-coordinate vectors; a gain is estimated from these same responses."""
    A.need(positions in (5, 10), 'fixed_window')
    A.need(all(type(v) in (int, float) and math.isfinite(v) and v >= 0
               for v in (rs, rp, rd)), 'finite_nonnegative_RMS')
    scale = max(rs, rp, rd)
    out = dict(pulse_RMS=rs, persistent_RMS=rp, difference_RMS=rd,
               positions=positions, normalization_RMS=scale,
               moment_tolerance=MOMENT_TOL,
               signed_gain=None, nonnegative_gain=None, cosine=None,
               signed_residual_fraction=None, nonnegative_residual_fraction=None,
               signed_fraction_lower=None, signed_fraction_upper=None,
               positive_fraction_lower=None, positive_fraction_upper=None,
               signed_gain_lower=None, signed_gain_upper=None,
               gain_sign_uncertain=None,
               raw_signed_residual_fraction=None,
               raw_nonnegative_residual_fraction=None,
               raw_normalized_signed_residual=None,
               moment_clamped=False, rounding_sensitive=False)
    if scale == 0:
        return dict(out, status='both_zero', signed_residual_RMS=0.,
                    nonnegative_residual_RMS=0., gain_identifiable=False)
    # Normalized mean energies suffice: the common window length cancels.
    aa, bb, dd = ((r / scale)**2 for r in (rs, rp, rd))
    cc = math.fsum((aa, bb, -dd)) / 2
    bound = math.sqrt(aa) * math.sqrt(bb)
    cond_a = MOMENT_TOL/aa if aa > 0 else None
    cond_b = MOMENT_TOL/bb if bb > 0 else None
    overflow = any(v is not None and not math.isfinite(v) for v in (cond_a, cond_b))
    out.update(normalized_A=aa, normalized_B=bb, normalized_D=dd,
               normalized_dot_raw=cc, gain_identifiable=rs > 0,
               cauchy_bound=bound, raw_cauchy_slack=bound-abs(cc),
               dot_adjustment=None, condition_overflow=overflow,
               normalized_nonzero_underflow=(aa == 0 and rs > 0) or (bb == 0 and rp > 0),
               tolerance_over_A=cond_a if cond_a is not None and math.isfinite(cond_a) else None,
               tolerance_over_B=cond_b if cond_b is not None and math.isfinite(cond_b) else None)
    if abs(cc) > bound + 1.5*MOMENT_TOL:
        return dict(out, status='invalid_moment_geometry', gain_identifiable=False)
    used = min(bound, max(-bound, cc))
    out.update(normalized_dot_used=used, moment_clamped=used != cc,
               dot_adjustment=used-cc)
    if rs == 0:
        if abs(bb-dd) > 3*MOMENT_TOL:
            return dict(out, status='invalid_zero_pulse_geometry',
                        gain_identifiable=False)
        return dict(out, status='pulse_zero', signed_residual_RMS=rp,
                    nonnegative_residual_RMS=rp,
                    signed_residual_fraction=1. if bb else None,
                    nonnegative_residual_fraction=1. if bb else None)
    if rp == 0:
        return dict(out, status='persistent_zero', signed_gain=0.,
                    nonnegative_gain=0., signed_residual_RMS=0.,
                    nonnegative_residual_RMS=0.)
    if min(aa, bb) <= MOMENT_TOL:
        return dict(out, status='nonzero_below_guard_scale', rounding_sensitive=True,
                    signed_fraction_lower=0., signed_fraction_upper=1.,
                    positive_fraction_lower=0., positive_fraction_upper=1.)
    cosine = used / bound
    out.update(raw_normalized_signed_residual=bb-cc*cc/aa,
               raw_signed_residual_fraction=1.-cc*cc/(aa*bb),
               raw_nonnegative_residual_fraction=1.-max(cc, 0.)**2/(aa*bb))
    signed_fraction = max(0., 1.-cosine*cosine)
    positive_cosine = max(0., cosine)
    positive_fraction = max(0., 1.-positive_cosine*positive_cosine)
    gain = used / aa
    out.update(status='defined', signed_gain=gain, nonnegative_gain=max(0., gain),
               cosine=cosine, signed_residual_fraction=signed_fraction,
               nonnegative_residual_fraction=positive_fraction,
               signed_residual_RMS=rp*math.sqrt(signed_fraction),
               nonnegative_residual_RMS=rp*math.sqrt(positive_fraction))
    # Conditional sensitivity envelope, NOT a sampling CI or native noise bound.
    denom_lo = (aa-MOMENT_TOL)*(bb-MOMENT_TOL)
    denom_hi = (aa+MOMENT_TOL)*(bb+MOMENT_TOL)
    gain_extrema = [(cc+dc)/(aa+da) for dc in (-1.5*MOMENT_TOL, 1.5*MOMENT_TOL)
                    for da in (-MOMENT_TOL, MOMENT_TOL)]
    out.update(signed_gain_lower=min(gain_extrema), signed_gain_upper=max(gain_extrema),
               gain_sign_uncertain=cc-1.5*MOMENT_TOL <= 0 <= cc+1.5*MOMENT_TOL)
    for prefix, c_lo, c_hi in (
        ('signed', max(0., abs(cc)-1.5*MOMENT_TOL), abs(cc)+1.5*MOMENT_TOL),
        ('positive', max(0., cc-1.5*MOMENT_TOL), max(0., cc+1.5*MOMENT_TOL)),
    ):
        out[prefix+'_fraction_lower'] = max(0., 1.-min(1., c_hi*c_hi/denom_lo))
        out[prefix+'_fraction_upper'] = min(1., 1.-max(0., c_lo*c_lo/denom_hi))
    return out


def analyze(tables):
    rows = tables['WINDOWS']
    A.need(len(rows) == 1008, 'all_three_contrasts_1008_rows')
    idx = A.index(rows, ('window', 'command_dimension', 'contrast'))
    expected = set(product(A.SNAPSHOTS, A.WINDOWS, A.DIMENSIONS, A.CONTRASTS))
    A.need(set((key[:4], *key[4:]) for key in idx) == expected, 'exact_source_roster')
    for row in rows:
        A.validate_common(row)
        A.need(row['positions'] == (5 if row['window'] == 'first5' else 10)
               and row['unit'] == 'command_adapter_units'
               and row['proposed_not_issued'] is True, 'same_command_scope_and_window')
        A.need(all(math.isfinite(row[k]) for k in ('RMS', 'mean_signed')),
               'finite_RMS_and_mean')
        A.need((row['RMS'] == 0) == (row['exactzero_slots'] == row['positions']),
               'RMS_zero_matches_actual_zero_slots')
    result = []
    for snap, window, dim in product(A.SNAPSHOTS, A.WINDOWS, A.DIMENSIONS):
        r = [idx[(*snap, window, dim, c)] for c in A.CONTRASTS]
        # Necessary independent consistency check from the retained signed means.
        A.need(A.close(r[2]['mean_signed'], r[1]['mean_signed']-r[0]['mean_signed']),
               'shared_none_mean_identity')
        value = project(*(row['RMS'] for row in r), r[0]['positions'])
        value.update(dict(zip(A.IDENTITY, snap)), window=window,
                     command_dimension=dim, unit='command_adapter_units',
                     pulse_mean=r[0]['mean_signed'], persistent_mean=r[1]['mean_signed'],
                     difference_mean=r[2]['mean_signed'])
        result.append(value)
    A.need(len(result) == 336, 'all336_pairs')
    return result


def distribution(values):
    values = sorted(v for v in values if v is not None and math.isfinite(v))
    if not values:
        return {'n': 0, 'min': None, 'median': None, 'max': None}
    n = len(values)
    mid = values[n//2] if n % 2 else (values[n//2-1]+values[n//2])/2
    return {'n': n, 'min': values[0], 'median': mid, 'max': values[-1]}


def summarize(rows):
    out = []
    for window, dim in product(A.WINDOWS, A.DIMENSIONS):
        cells = [r for r in rows if r['window'] == window and r['command_dimension'] == dim]
        A.need(len(cells) == 24, 'all24_in_each_axis_window')
        defined = [r for r in cells if r['status'] == 'defined']
        out.append({
            'window': window, 'command_dimension': dim, 'snapshots': 24,
            'status_counts': {s: sum(r['status'] == s for r in cells)
                              for s in sorted({r['status'] for r in cells})},
            'cosine': distribution(r['cosine'] for r in defined),
            'signed_gain': distribution(r['signed_gain'] for r in defined),
            'signed_residual_fraction': distribution(r['signed_residual_fraction'] for r in defined),
            'nonnegative_residual_fraction': distribution(r['nonnegative_residual_fraction'] for r in defined),
            'signed_numeric_lower_positive': sum(r['signed_fraction_lower'] > 0 for r in defined),
            'positive_numeric_lower_positive': sum(r['positive_fraction_lower'] > 0 for r in defined),
            'rounding_sensitive': sum(r['rounding_sensitive'] for r in cells),
            'not_independent_samples': True,
        })
    return out
