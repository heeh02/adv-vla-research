"""Fixed descriptive reductions of archived scalar comparisons, without fits."""
from itertools import product
import math
import statistics

KEYS = tuple((f'SDRU-CDEV-T0{t}-R0{r}', q) for t in range(2) for r in range(2) for q in (11, 22))
DIMS = tuple('translation_command_'+s for s in 'xyz') + tuple(
    'rotation_command_'+s for s in 'xyz') + ('gripper_command',)
FIELDS = ('T_x0', 'T_v0', 'T_x5', 'T_v5', 'T_P_j5_L9', 'T_gate_j5_L9',
          'T_mask_j5', 'T_suffix_positions_j5', 'time_j0_j5', 'time_weak_type_j0_j5',
          'prefix_mask', 'prefix_positions')
FEATURES = ('cache_headpos', 'cache_full_write', 'cache_c9', 'cache_current4',
            'current_headpos', 'current_full_write', 'cache_old_P_effect')
WINDOWS = ('first5', 'full10')
COMPARISONS = ('none_endpoint', 'persistent_endpoint', 'persistent_effect_own_baselines')
COUNTS = {'snapshots': 8, 'native_field_comparisons': 96, 'feature_comparisons': 56,
          'endpoint_comparisons': 848, 'persistent_own_baseline_comparisons': 112}
FIELDS_A = ('T_v0', 'T_v5', 'T_P_j5_L9', 'current_headpos', 'current_full_write')
COMMON_METRICS = ('relative_RMS_to_reference', 'relative_RMS_to_candidate')
SIGNAL_METRICS = ('residual_RMS_over_new_S_RMS', 'residual_RMS_over_new_S_minus_R_RMS')


def need(ok, code):
    if not ok:
        raise ValueError(code)


def key(row):
    return row['root_id'], row['query_index']


def close(a, b):
    return abs(a-b) <= 5e-12*max(1., abs(a), abs(b))


def ratio_check(row, num, den, ratio, zero):
    n, d, r = row[num], row[den], row[ratio]
    need(type(row[zero]) is bool and row[zero] == (d == 0), 'explicit_zero_denominator')
    need((r is None) == (d == 0), 'undefined_iff_zero_denominator')
    if r is not None:
        need(isinstance(r, (float, int)) and math.isfinite(r) and r >= 0 and close(r, n/d),
             'cached_ratio_arithmetic')


def validate_row(row):
    need(key(row) in KEYS, 'fixed8_snapshot_identity')
    need(type(row['query_index']) is int, 'integer_query')
    for k in ('RMS_difference', 'RMS_reference', 'RMS_candidate', 'max_abs'):
        need(isinstance(row[k], (int, float)) and math.isfinite(row[k]) and row[k] >= 0,
             'finite_nonnegative_scalar')
    n = row['numeric_nonzero_differences']
    need(type(n) is int and n >= 0, 'integer_difference_count')
    for k in ('typed_exact', 'raw_payload_exact', 'numeric_equal'):
        need(type(row[k]) is bool, 'explicit_exactness_boolean')
    need(row['numeric_equal'] == (n == 0) == (row['RMS_difference'] == 0) ==
         (row['max_abs'] == 0), 'numeric_zero_consistency')
    need(row['RMS_difference'] <= row['max_abs']+1e-12, 'RMS_max_consistency')
    rr, cr, dr = row['RMS_reference'], row['RMS_candidate'], row['RMS_difference']
    tol = 5e-12*max(1., rr, cr, dr)
    need(abs(rr-cr) <= dr+tol and dr <= rr+cr+tol, 'RMS_triangle_consistency')
    need(not row['numeric_equal'] or rr == cr, 'equal_values_equal_norms')
    left, right = row['reference'], row['candidate']
    for desc in (left, right):
        need(set(desc) == {'shape', 'logical_dtype', 'storage_dtype', 'bytes', 'payload_sha256'},
             'typed_descriptor_schema')
        need(all(type(v) is int and v >= 0 for v in desc['shape']) and
             type(desc['bytes']) is int and desc['bytes'] > 0 and
             len(desc['payload_sha256']) == 64, 'typed_descriptor_shape_hash')
    need(left['shape'] == right['shape'], 'same_paired_shape')
    need(n <= math.prod(left['shape']), 'difference_count_bounded')
    need(row['typed_exact'] == (left == right) and
         row['raw_payload_exact'] == (left['payload_sha256'] == right['payload_sha256']),
         'typed_and_raw_equality_preserved')
    need(not row['typed_exact'] or row['numeric_equal'], 'typed_equality_implies_numeric')
    ratio_check(row, 'RMS_difference', 'RMS_reference', COMMON_METRICS[0], 'reference_zero')
    ratio_check(row, 'RMS_difference', 'RMS_candidate', COMMON_METRICS[1], 'candidate_zero')
    if row.get('comparison') == 'persistent_effect_own_baselines':
        for den, metric, zero in zip(('new_S_RMS', 'new_S_minus_R_RMS'), SIGNAL_METRICS,
                                     ('new_S_zero', 'new_S_minus_R_zero')):
            need(isinstance(row[den], (float, int)) and math.isfinite(row[den]) and row[den] >= 0,
                 'finite_pulse_denominator')
            ratio_check(row, 'RMS_difference', den, metric, zero)


def index(rows, extra):
    result = {}
    for row in rows:
        validate_row(row)
        k = key(row)+tuple(row[e] for e in extra)
        need(k not in result, 'no_duplicate_support')
        result[k] = row
    return result


def summary(values):
    defined = [v for v in values if v is not None]
    need(len(values) == 8, 'all8_in_each_summary')
    return {'samples': 8, 'defined': len(defined), 'undefined': 8-len(defined),
            'minimum': min(defined) if defined else None,
            'median': statistics.median(defined) if defined else None,
            'maximum': max(defined) if defined else None,
            'zero': sum(v == 0 for v in defined),
            'less_than_one': sum(v < 1 for v in defined),
            'equal_to_one': sum(v == 1 for v in defined),
            'greater_than_one': sum(v > 1 for v in defined),
            'near_one_display_only': [
                {'snapshot_index': i, 'snapshot_key': list(KEYS[i]), 'unrounded_value': v}
                for i, v in enumerate(values) if v is not None and abs(v-1.) <= 1e-12],
            'near_one_is_not_equivalence_or_reclassification': True}


def reduce_group(rows, metrics):
    return {'snapshot_order': [list(key(r)) for r in rows],
            'typed_exact': sum(r['typed_exact'] for r in rows),
            'numeric_equal': sum(r['numeric_equal'] for r in rows),
            'reference_zero': sum(r['reference_zero'] for r in rows),
            'candidate_zero': sum(r['candidate_zero'] for r in rows),
            'metrics': {m: {**summary([r[m] for r in rows]), 'all8_values': [r[m] for r in rows]}
                        for m in metrics}}


def aggregate(fields, features, endpoints, producer, qa, protocol, *, synthetic=False):
    need(producer['schema'] == 'paper2-dev16-feature-bridge-result-v1' and
         producer['status'] == 'DEV16_FEATURE_BRIDGE_COMPLETE' and
         producer['synthetic'] is synthetic and producer['input_qualified'] is True and
         producer['analysis_complete'] is True and producer['snapshots'] == 8,
         'complete_source_bridge')
    need(qa['schema'] == 'paper2-dev16-feature-bridge-QA-v1' and qa['synthetic'] is synthetic and
         qa['pass'] is True and qa['all8_retained'] is True and
         qa['native72_final_lists_checked'] is True and qa['new_N_P_checks_events_bound'] == 16,
         'inherited_source_qualification')
    for doc in (producer, qa):
        need(all(type(doc[k]) is int and doc[k] == 0 for k in
                 ('new_model_calls', 'new_env_steps', 'new_fits')), 'source_no_new_calls_or_fit')
        need(doc['whole_graph_equivalence'] == 'UNMEASURED', 'source_graph_boundary')
        for k in ('snapshots', 'native_field_comparisons', 'feature_comparisons', 'endpoint_comparisons'):
            need((producer['comparison_counts'] if doc is producer else qa['counts'])[k] == COUNTS[k],
                 'source_fixed_comparison_counts')
    need(producer['prediction_validated'] is False and qa['fresh_C_context'] == 'UNMEASURED' and
         protocol['statistical_or_physical_acceptance_threshold'] is None, 'no_unmeasured_acceptance')
    f = index(fields, ('field',)); p = index(features, ('feature',))
    e = index(endpoints, ('comparison', 'window', 'dimension'))
    need(set(f) == {s+(name,) for s, name in product(KEYS, FIELDS)}, 'all96_fields')
    need(set(p) == {s+(name,) for s, name in product(KEYS, FEATURES)}, 'all56_features')
    expected = {s+(c, w, d) for s, c, w, d in product(KEYS, COMPARISONS, WINDOWS, DIMS)}
    expected |= {s+(c, 'full10', f'normalized_raw_coordinate_{d}')
                 for s, c, d in product(KEYS, ('none_raw32', 'persistent_raw32'), range(32))}
    need(set(e) == expected, 'all848_endpoints')
    out = {'schema': 'paper2-dev16-bridge-readout-summary-v1', 'synthetic': synthetic,
           'counts': COUNTS, 'field': {}, 'feature': {}, 'endpoint': [],
           'no_threshold_or_causal_amplification': True,
           'full_support_table': 'FULL-SUPPORT.csv.gz', 'source_qualification_inherited': True,
           'fresh_C_context': 'UNMEASURED', 'whole_graph_equivalence': 'UNMEASURED',
           'historical_source_domain_preserved': True, 'new_Gstar_internal_contrasts_preserved': True}
    for group, idx, names in (('field', f, FIELDS), ('feature', p, FEATURES)):
        for name in names:
            rows = [idx[s+(name,)] for s in KEYS]
            out[group][name] = reduce_group(rows, COMMON_METRICS)
    for c, w, d in sorted({k[2:] for k in e}):
        rows = [e[s+(c, w, d)] for s in KEYS]
        metrics = COMMON_METRICS+(SIGNAL_METRICS if c == 'persistent_effect_own_baselines' else ())
        out['endpoint'].append({'comparison': c, 'window': w, 'dimension': d,
                                **reduce_group(rows, metrics)})
    support = [{'source_kind': kind, **row} for kind, rows in
               (('field', fields), ('feature', features), ('endpoint', endpoints)) for row in rows]
    return {'summary': out, 'support': support, 'fields': f, 'features': p, 'endpoints': e,
            'synthetic': synthetic}
