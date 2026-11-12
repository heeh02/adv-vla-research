"""Strict scalar-cache parser and descriptive CONF24 checks; no fitting."""
from pathlib import Path
from itertools import product
import csv
import math

TASKS = ('CCONF-T06', 'CCONF-T07', 'CCONF-T08', 'CCONF-T09')
VARIANTS = ('V00700', 'V00774', 'V01029', 'V00827')
ROOTS = tuple((task, f'SDRU-{task}-{variant}-I00-P{parent}', parent)
              for task, variant in zip(TASKS, VARIANTS) for parent in ('C', 'T'))
SNAPSHOTS = tuple((*root, q) for root in ROOTS for q in (11, 16, 22))
LAYERS = (1, 9, 18)
FLOWS = (5, 6, 7, 8, 9)
WINDOWS = ('first5', 'full10')
CONTRASTS = ('pulse_minus_none', 'persistent_minus_none', 'persistent_minus_pulse')
FIELDS = ('receiver_residual_input', 'attention_output_post_WO_pre_gate',
          'hidden_after_attention_residual', 'hidden_post_block')
EXACT = ('query_post_rope_scaled_typed_exact', 'probabilities_native_typed_exact',
         'attention_gate_typed_exact', 'suffix_key_post_rope_typed_exact',
         'suffix_value_typed_exact')
DIMENSIONS = ('translation_command_x', 'translation_command_y',
              'translation_command_z', 'rotation_command_x', 'rotation_command_y',
              'rotation_command_z', 'gripper_command')
IDENTITY = ('task_id', 'root_id', 'parent_condition', 'query_index')
BASE = IDENTITY + ('policy_env_step', 'absolute_env_step', 'graph', 'contrast')
FLOW_HEAD = BASE + ('field', 'flow_index', 'reference_flow_time',
                   'candidate_flow_time', 'coordinates')
STATS = ('action_window', 'action_positions', 'coordinates_in_window', 'signed_mean',
         'RMS_difference', 'maxabs_difference', 'difference_energy', 'reference_energy',
         'candidate_energy', 'all_exactzero_difference', 'different_coordinates',
         'summary_not_independent_samples')
HEADERS = {
    'FLOW-SUMMARY': FLOW_HEAD + STATS,
    'MECHANISM-SUMMARY': FLOW_HEAD + ('layer_index', 'layer_label') + STATS,
    'READING_EXACT': BASE + ('flow_index', 'reference_flow_time',
                            'candidate_flow_time', 'layer_index', 'layer_label') + EXACT,
    'WINDOWS': BASE + ('window', 'command_dimension', 'mean_signed', 'meanabs', 'RMS',
                       'maxabs', 'exactzero_slots', 'positions', 'unit', 'proposed_not_issued'),
    'COVERAGE': IDENTITY + ('status', 'same_j5_and_x6', 'same_j6_L1_L9_QP',
                            'actual_E_exactzero'),
}
INTS = set(('query_index', 'policy_env_step', 'absolute_env_step', 'flow_index',
            'layer_index', 'layer_label', 'action_positions', 'coordinates_in_window',
            'different_coordinates', 'exactzero_slots', 'positions'))
BOOLS = set(EXACT) | {'all_exactzero_difference', 'summary_not_independent_samples',
                     'proposed_not_issued', 'same_j5_and_x6', 'same_j6_L1_L9_QP',
                     'actual_E_exactzero'}
FLOATS = {'reference_flow_time', 'candidate_flow_time', 'signed_mean', 'RMS_difference',
          'maxabs_difference', 'difference_energy', 'reference_energy', 'candidate_energy',
          'mean_signed', 'meanabs', 'RMS', 'maxabs'}
INPUT_NAMES = tuple(f'{t}-PVP-{k}-v1.csv' for t in TASKS
                    for k in ('FLOW-SUMMARY', 'MECHANISM-SUMMARY', 'READING_EXACT', 'WINDOWS')
                    ) + ('PVP-COVERAGE-v1.csv', 'RESULT-v1.json')


def need(condition, code):
    if not condition:
        raise ValueError(code)


def snapshot(row):
    return tuple(row[k] for k in IDENTITY)


def parse_csv(path, kind):
    """CSV bodies may only be supplied by the Linux adapter or synthetic fixture."""
    rows = []
    with Path(path).open(newline='', encoding='utf-8') as source:
        reader = csv.DictReader(source)
        need(tuple(reader.fieldnames or ()) == HEADERS[kind], 'exact_CSV_header_' + kind)
        for row in reader:
            need(None not in row and all(v is not None and v != '' for v in row.values()),
                 'no_missing_source_cells')
            parsed = {}
            for key, value in row.items():
                if key in BOOLS:
                    need(value in ('True', 'False'), 'strict_boolean')
                    parsed[key] = value == 'True'
                elif key in INTS:
                    parsed[key] = int(value)
                    need(str(parsed[key]) == value, 'strict_integer')
                elif key in FLOATS:
                    parsed[key] = float(value)
                    need(math.isfinite(parsed[key]), 'finite_source_scalar')
                else:
                    parsed[key] = value
            rows.append(parsed)
            need(len(rows) <= 6000, 'bounded_CSV_rows')
    return rows


def index(rows, extra):
    keys = IDENTITY + extra
    result = {}
    for row in rows:
        key = tuple(row[k] for k in keys)
        need(key not in result, 'no_duplicate_rows')
        result[key] = row
    return result


def close(a, b):
    return abs(a-b) <= 5e-12 * max(1., abs(a), abs(b))


def validate_common(row):
    need(snapshot(row) in SNAPSHOTS, 'fixed_24_snapshot_identity')
    if 'graph' in row:
        need(row['graph'] == 'Gstar' and row['contrast'] in CONTRASTS, 'same_Gstar_contrast')
        need(row['policy_env_step'] == row['query_index']*5 and
             row['absolute_env_step'] == row['policy_env_step']+10, 'separate_two_clocks')
    if 'layer_label' in row:
        need(row['layer_label'] in LAYERS and
             row['layer_index'] == row['layer_label']-1, 'exact_layer_axes')
    if 'reference_flow_time' in row:
        need(row['reference_flow_time'] == row['candidate_flow_time'], 'same_native_flow_time')


def validate_stats(row, kind):
    window = row['action_window']
    positions = 5 if window == 'first5' else 10
    need(window in WINDOWS and row['action_positions'] == positions, 'complete_action_window')
    coordinates = positions * (1024 if kind == 'MECHANISM-SUMMARY' else 32)
    need(row['coordinates_in_window'] == coordinates, 'exact_coordinate_denominator')
    need(row['coordinates'] == ('AE_1024_native' if kind == 'MECHANISM-SUMMARY'
                               else 'native_32_not_physical_units'), 'native_coordinate_units')
    for key in ('RMS_difference', 'maxabs_difference', 'difference_energy',
                'reference_energy', 'candidate_energy'):
        need(row[key] >= 0, 'nonnegative_energy_magnitude')
    energy = row['difference_energy']
    need(close(row['RMS_difference']**2, energy/coordinates), 'RMS_energy_identity')
    need(abs(row['signed_mean']) <= row['RMS_difference']+1e-12 and
         row['RMS_difference'] <= row['maxabs_difference']+1e-12, 'mean_RMS_max_order')
    zero = row['all_exactzero_difference']
    need(zero == (energy == 0) == (row['RMS_difference'] == 0) ==
         (row['maxabs_difference'] == 0) == (row['different_coordinates'] == 0),
         'exact_zero_preserved')
    need(0 <= row['different_coordinates'] <= coordinates and
         row['summary_not_independent_samples'] is True, 'descriptive_summary_support')


def verify(tables, producer, *, synthetic=False):
    need(set(tables) == set(HEADERS), 'exact_five_table_kinds')
    need(producer['schema'] == 'PVP-existing-Gstar-CPU-readout-v1' and
         producer['complete'] is True and producer['synthetic'] is synthetic and
         producer['planned_queries'] == producer['completed_queries'] == 24 and
         producer['errors'] == [], 'completed_fixed_producer')
    need(producer['new_model_calls'] == producer['new_environment_steps'] == 0 and
         producer['new_fit'] is False and producer['old_R_arrays_used'] is False and
         producer['behavior_or_equal_dose_timing_claim'] is False, 'producer_boundaries')
    for kind, rows in tables.items():
        for row in rows:
            validate_common(row)
            if kind in ('MECHANISM-SUMMARY', 'FLOW-SUMMARY'):
                validate_stats(row, kind)
    coverage = index(tables['COVERAGE'], ())
    need(set(coverage) == set(SNAPSHOTS), 'coverage_all24_no_selection')
    need(coverage == index(producer['coverage'], ()), 'coverage_matches_producer')
    for row in coverage.values():
        need(row['status'] == 'COMPLETE' and row['same_j5_and_x6'] and
             row['same_j6_L1_L9_QP'] and not row['actual_E_exactzero'],
             'inherited_native_qualification')

    reading = index(tables['READING_EXACT'], ('contrast', 'flow_index', 'layer_label'))
    need(set(reading) == {s+(c, j, l) for s, c, j, l in
                         product(SNAPSHOTS, CONTRASTS, FLOWS, LAYERS)},
         'all1080_reading_rows')
    mechanism = index(tables['MECHANISM-SUMMARY'],
                      ('contrast', 'field', 'flow_index', 'layer_label', 'action_window'))
    need(set(mechanism) == {s+('persistent_minus_pulse', f, j, l, w)
                           for s, f, j, l, w in product(SNAPSHOTS, FIELDS, FLOWS, LAYERS, WINDOWS)},
         'all2880_mechanism_rows')
    flow = index(tables['FLOW-SUMMARY'], ('contrast', 'field', 'flow_index', 'action_window'))
    need(set(flow) == {s+(c, f, j, 'full10') for s, c in product(SNAPSHOTS, CONTRASTS)
                      for f, end in (('x', 11), ('velocity_native', 10)) for j in range(end)},
         'all1512_flow_rows')
    windows = index(tables['WINDOWS'], ('contrast', 'window', 'command_dimension'))
    need(set(windows) == {s+(c, w, d) for s, c, w, d in
                         product(SNAPSHOTS, CONTRASTS, WINDOWS, DIMENSIONS)},
         'all1008_window_rows')
    for row in windows.values():
        n = 5 if row['window'] == 'first5' else 10
        need(row['positions'] == n and row['unit'] == 'command_adapter_units' and
             row['proposed_not_issued'] is True, 'candidate_command_units')
        need(0 <= row['exactzero_slots'] <= n and 0 <= row['meanabs'] <= row['RMS']+1e-12 and
             row['RMS'] <= row['maxabs']+1e-12 and
             abs(row['mean_signed']) <= row['meanabs']+1e-12, 'window_magnitude_order')
        need((row['exactzero_slots'] == n) == (row['RMS'] == 0) ==
             (row['maxabs'] == 0) == (row['meanabs'] == 0), 'window_zero_preserved')
    # The common prefix is checked using cached typed flags and exact-zero
    # summaries; the original native tensors are intentionally not reopened.
    for s in SNAPSHOTS:
        for j, l in product(FLOWS, LAYERS):
            r = reading[s+('persistent_minus_pulse', j, l)]
            if j == 5 or (j == 6 and l in (1, 9)):
                need(all(r[k] for k in EXACT), 'common_early_receiver_cache')
            for f, w in product(FIELDS, WINDOWS):
                m = mechanism[s+('persistent_minus_pulse', f, j, l, w)]
                if j == 5 or (j == 6 and l == 1) or (
                        j == 6 and l == 9 and f == 'receiver_residual_input'):
                    need(m['all_exactzero_difference'], 'common_early_neural_cache')
        for f, stop in (('x', 7), ('velocity_native', 6)):
            for j in range(stop):
                need(flow[s+('persistent_minus_pulse', f, j, 'full10')]
                     ['all_exactzero_difference'], 'common_j5_x6_flow_cache')

    cells = []
    for s, w, d in product(SNAPSHOTS, WINDOWS, DIMENSIONS):
        p = windows[s+('persistent_minus_none', w, d)]
        a = windows[s+('pulse_minus_none', w, d)]
        difference = windows[s+('persistent_minus_pulse', w, d)]
        need(close(p['mean_signed']-a['mean_signed'], difference['mean_signed']),
             'same_N_signed_window_mean_identity')
        need(abs(p['RMS']-a['RMS']) <= difference['RMS']+1e-12 and
             difference['RMS'] <= p['RMS']+a['RMS']+1e-12, 'window_RMS_triangle')
        den, num = a['RMS'], p['RMS']
        ratio = num/den if den > 0 else None
        cells.append({**dict(zip(IDENTITY, s)), 'window': w, 'command_dimension': d,
                      'pulse_RMS': den, 'persistent_RMS': num,
                      'RMS_ratio': ratio, 'ratio_defined': den > 0,
                      'pulse_mean_signed': a['mean_signed'],
                      'persistent_mean_signed': p['mean_signed'],
                      'signed_window_means_opposite': (
                          (a['mean_signed'] < 0 < p['mean_signed']) or
                          (p['mean_signed'] < 0 < a['mean_signed'])),
                      'pulse_mean_exactzero': a['mean_signed'] == 0,
                      'persistent_mean_exactzero': p['mean_signed'] == 0,
                      'pulse_window_exactzero': den == 0,
                      'persistent_window_exactzero': num == 0,
                      'RMS_order': 'smaller' if num < den else 'larger' if num > den else 'equal'})
    counts = []
    for j, l in product(FLOWS, LAYERS):
        rows = [reading[s+('persistent_minus_pulse', j, l)] for s in SNAPSHOTS]
        counts.append({'flow_index': j, 'layer_label': l, 'snapshots': 24,
                       **{k: sum(r[k] for r in rows) for k in EXACT},
                       **{f+'_zero': sum(mechanism[s+('persistent_minus_pulse', f, j, l, 'full10')]
                                         ['all_exactzero_difference'] for s in SNAPSHOTS) for f in FIELDS}})
    command_counts = []
    for w in WINDOWS:
        group = [c for c in cells if c['window'] == w]
        command_counts.append({'window': w, 'descriptive_cells': 168,
            **{k: sum(c['RMS_order'] == k for c in group) for k in ('smaller', 'equal', 'larger')},
            **{k: sum(c[k] for c in group) for k in (
                'signed_window_means_opposite', 'pulse_window_exactzero',
                'persistent_window_exactzero', 'pulse_mean_exactzero', 'persistent_mean_exactzero')},
            'undefined_ratios': sum(not c['ratio_defined'] for c in group)})
    return {'synthetic': synthetic, 'tables': tables, 'reading': reading,
            'mechanism': mechanism, 'flow': flow, 'windows': windows,
            'command_cells': cells, 'separation_counts': counts, 'command_counts': command_counts,
            'qa': {'schema': 'paper2-conf24-cache-QA-v1', 'status': 'PASS',
                   'synthetic': synthetic, 'rows': {k: len(v) for k, v in tables.items()},
                   'snapshot_count': 24, 'task_count': 4, 'parent_roots': 8,
                   'all_snapshot_dimension_window_cells': 336, 'new_fits': 0,
                   'original100objects_reopened': False, 'native_qualification_reexecuted': False,
                   'typed_exact_source': 'inherited cached native storage flags',
                   'zero_source': 'cached decoded-coordinate exact-zero summaries',
                   'arithmetic_relative_tolerance': 5e-12,
                   'zero_denominators': 'undefined null; never pseudocount',
                   'independence_claim': False,
                   'separation_counts': counts, 'command_counts': command_counts}}
