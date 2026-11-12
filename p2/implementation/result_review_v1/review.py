"""Cloud-only acceptance and bounded scalar export of a frozen CPU result.

No model, NumPy, network, NPZ decode, job control, or input deletion. Hash all
objects in the result manifest on cloud; only small JSON values leave cloud.
"""
from pathlib import Path
import argparse
import gzip
import hashlib
import io
import itertools
import json
import math
import os
import stat
import sys
import time

HERE = Path(__file__).resolve().parent
SCHEMA = 'PAPER2-result-review-v1'
MAX_EXPORT = 512 * 1024
PRODUCER_MANIFEST = {'bytes': 1467, 'sha256': 'd27ebe3828ded5f42dfb9e2aa670f637e9972770c9742b4e96ab886e3ce741d3'}
INVENTORY = {'bytes': 21030, 'sha256': '217c4c2be45393977663501f377625f6403d5e0bbd248fefb79994e2a3eeb117'}
PARENT_SUMMARY = {'bytes': 88120, 'sha256': 'b4c3a2d6af8ff3a1e55126400e68c3c32b9d17ebea82252afa3d6c87d7673dcc'}
PARENT_RUN = 'jinwen-sru-natural-reading-20261016-v1'
BASELINES = ('zero_change', 'global_window_prefix_mass', 'per_position_prefix_mass',
             'per_head_position_prefix_mass', 'modality_mass', 'within_modality', 'signed_scalar_oracle', 'full_late')
FIGURE_BASELINES = BASELINES[1:5]
VIEWS = ('full', 'response_B3', 'response_residual', 'clean_B3', 'clean_residual')
WINDOWS = ('full10', 'first5', 'tail5')
PAIRS = ((0, 5), (5, 9), (0, 9))
TASKS = {f'CCONF-T{t:02d}': tuple(f'SDRU-CCONF-T{t:02d}-V{v:05d}-I00-P{p}' for p in ('C', 'T'))
         for t, v in ((6, 700), (7, 774), (8, 1029), (9, 827))}
ROOTS = tuple(r for roots in TASKS.values() for r in roots)
EXPECTED_QUERIES = {(r, q) for r in ROOTS for q in (11, 16, 22)}
TAGS = ('condition', 'layer_index', 'flow_from', 'flow_to', 'view', 'window')
EXPECTED_TAGS = {(c, l, a, b, v, w) for c, l, (a, b), v, w in
                 itertools.product(('C', 'T'), (0, 8, 17), PAIRS, VIEWS, WINDOWS)}
FAILURE_MARKERS = ('FINALIZATION-FAILURE.json', 'FAILURE.json')
RESULT_NAMES = {f'{prefix}-{i:02d}{suffix}' for i in range(24)
                for prefix, suffix in (('DERIVED', '.npz'), ('QUERY', '.json.gz'))} | {'SUMMARY.json', 'AGGREGATES.json.gz'}
DERIVED_ABI = {'prefix_mass': ((2, 8, 10), 'float64'), 'scales': ((3, 3, 8, 10), 'float64'),
               'scale_defined': ((3, 3, 8, 10), 'bool'), 'scale_identified': ((3, 3, 8, 10), 'bool'),
               'target_energy': ((5, 8, 10), 'float64'), 'baseline_SSE': ((5, 3, 8, 8, 10), 'float64'),
               'baseline_defined': ((3, 8, 8, 10), 'bool'), 'oracle_full_window_alpha': ((3,), 'float64'),
               'oracle_full_window_source_energy': ((3,), 'float64'), 'oracle_source_zero': ((3,), 'bool')}


class Rejected(ValueError):
    pass


def need(ok, code):
    if not ok:
        raise Rejected(code)


def ident(raw):
    return {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


def same(ref, other):
    return all(ref[k] == other[k] for k in ('bytes', 'sha256'))


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()


def unique_json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            need(key not in result, 'duplicate_JSON_key_' + key)
            result[key] = value
        return result
    def nonfinite(value):
        raise Rejected('nonfinite_JSON_literal_' + value)
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=nonfinite)


class Deadline:
    def __init__(self, epoch):
        need(isinstance(epoch, (int, float)) and math.isfinite(epoch) and epoch > time.time(), 'future_absolute_review_deadline')
        self.epoch = epoch
        self.monotonic_end = time.monotonic() + 300

    def check(self):
        need(time.time() < self.epoch and time.monotonic() < self.monotonic_end, 'review_deadline')


def read_or_hash(path, *, ref=None, cap=4 * 1024**2, keep=True, deadline=None):
    path = Path(path)
    need(path.is_file() and not path.is_symlink(), 'regular_input_file_' + path.name)
    fd = os.open(path, os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0))
    blocks, digest, total = [], hashlib.sha256(), 0
    with os.fdopen(fd, 'rb') as stream:
        before = os.fstat(stream.fileno())
        need(stat.S_ISREG(before.st_mode) and 0 < before.st_size <= cap, 'bounded_input_file_' + path.name)
        if ref is not None:
            need(before.st_size == ref['bytes'], 'input_file_size_' + path.name)
        while True:
            if deadline is not None:
                deadline.check()
            block = stream.read(65536)
            if not block:
                break
            total += len(block)
            need(total <= cap, 'input_expanded_while_reading')
            digest.update(block)
            if keep:
                blocks.append(block)
        after = os.fstat(stream.fileno())
    need((before.st_ino, before.st_size, before.st_mtime_ns) ==
         (after.st_ino, after.st_size, after.st_mtime_ns), 'input_unchanged_while_reading')
    actual = {'bytes': total, 'sha256': digest.hexdigest()}
    if ref is not None:
        need(same(actual, ref), 'input_file_SHA_' + path.name)
    return (b''.join(blocks) if keep else None), actual


def uncompress_json(raw, cap):
    with gzip.GzipFile(fileobj=io.BytesIO(raw)) as stream:
        value = stream.read(cap + 1)
    need(len(value) <= cap, 'bounded_decompressed_JSON')
    return unique_json(value)


class Checks:
    def __init__(self):
        self.count = 0
        self.max_scaled_error = 0.0
        self.by_family = {}

    def equal(self, actual, expected, family, *, tolerance=1e-9):
        if actual is None or expected is None:
            need(actual is expected, 'null_support_mismatch_' + family)
            error = 0.
        else:
            need(type(actual) in (int, float) and type(expected) in (int, float) and
                 math.isfinite(actual) and math.isfinite(expected), 'finite_scalar_' + family)
            error = abs(actual - expected) / max(1., abs(actual), abs(expected))
            need(error <= tolerance, f'scalar_mismatch_{family}: actual={actual!r}, expected={expected!r}')
        self.count += 1
        self.max_scaled_error = max(self.max_scaled_error, error)
        self.by_family[family] = self.by_family.get(family, 0) + 1


def ratio(numerator, denominator):
    return None if numerator is None or denominator is None or denominator <= 0 else numerator / denominator


def mean(values):
    return None if any(v is None for v in values) else sum(values) / len(values)


def row_key(row):
    return tuple(row[k] for k in TAGS)


def role(row):
    main = (row['condition'], row['layer_index'], row['flow_from'], row['flow_to']) == ('C', 17, 0, 9)
    if main and row['view'] == 'full':
        return 'primary' if row['window'] == 'full10' else 'support_window'
    return 'descriptive_control'


def validate_scalar_row(row, checks, *, aggregate):
    need(row_key(row) in EXPECTED_TAGS and row['comparison_role'] == role(row), 'fixed_row_tags_and_role')
    target, full = row['target_delta_energy'], row['full_target_delta_energy']
    for value in (target, full):
        need(value is None or (type(value) in (int, float) and math.isfinite(value) and value >= 0), 'nonnegative_target_or_null')
    need(set(row['baselines']) == set(BASELINES), 'all_eight_baselines')
    for name in BASELINES:
        baseline = row['baselines'][name]
        sse = baseline['SSE']
        need(sse is None or (type(sse) in (int, float) and math.isfinite(sse) and sse >= 0), 'nonnegative_SSE_or_null')
        need(baseline['comparison_role'] == ('diagnostic_only' if name == 'full_late' else 'baseline'), 'full_late_only_diagnostic')
        checks.equal(baseline['relative_SSE'], ratio(sse, target), 'own_target_ratio')
        checks.equal(baseline['relative_SSE_to_full_target'], ratio(sse, full), 'full_target_ratio')
        need(baseline['status'] in ('DEFINED', 'ZERO_TARGET', 'UNAVAILABLE', 'UNAVAILABLE_QUERY',
                                  'UNAVAILABLE_ZERO_MASS_ALLOCATION', 'DEFINED_WITH_ZERO_MASS_NOOP',
                                  'SOURCE_ZERO_NOOP'), 'known_baseline_status')
        if baseline['status'] in ('DEFINED', 'DEFINED_WITH_ZERO_MASS_NOOP', 'SOURCE_ZERO_NOOP'):
            need(sse is not None and target is not None and target > 0, 'defined_status_numerics')
        elif baseline['status'] == 'ZERO_TARGET':
            need(target == 0 and sse is not None, 'zero_target_status_numerics')
        else:
            need(sse is None or target is None, 'unavailable_status_numerics')
        if not aggregate:
            planned = 80 if row['window'] == 'full10' else 40
            need(baseline['planned_head_positions'] == planned and
                 0 <= baseline['defined_head_positions'] <= planned, 'head_position_denominators')
            need(type(baseline['negative_roundoff_corrections']) is int and baseline['negative_roundoff_corrections'] >= 0,
                 'recorded_roundoff_count')
            noops = baseline['zero_mass_noop_head_positions']
            need(type(noops) is int and 0 <= noops <= baseline['defined_head_positions'], 'zero_mass_noop_count')
            if baseline['status'] == 'DEFINED_WITH_ZERO_MASS_NOOP':
                need(name in BASELINES[1:4] and noops > 0, 'identified_noop_status')
            if baseline['status'] == 'SOURCE_ZERO_NOOP':
                need(name == 'signed_scalar_oracle' and row['oracle_full_window_alpha'] is None and
                     row['oracle_full_window_source_energy'] == 0, 'oracle_zero_source_unidentified_scalar')
    checks.equal(row['baselines']['zero_change']['SSE'], target, 'zero_change_equals_target')
    if target is not None:
        checks.equal(row['baselines']['full_late']['SSE'], 0., 'full_late_identity')
    if not aggregate:
        decomposition = row['decomposition']
        if decomposition['status'] == 'DEFINED':
            gram = decomposition['term_gram']
            need(len(gram) == 3 and all(len(x) == 3 for x in gram), 'signed_MAI_Gram_shape')
            diagonal = sum(gram[i][i] for i in range(3))
            cross = 2 * (gram[0][1] + gram[0][2] + gram[1][2])
            checks.equal(decomposition['component_energy_sum'], diagonal, 'signed_MAI_diagonal')
            checks.equal(decomposition['signed_cross_sum'], cross, 'signed_MAI_cross')
            checks.equal(target, diagonal + cross, 'signed_MAI_target_closure')
        else:
            need(decomposition['status'] == 'UNAVAILABLE_MIXED_CORNER' and decomposition['term_gram'] is None,
                 'undefined_MAI_preserved')
    h = row['baselines']['per_head_position_prefix_mass']['SSE']
    m = row['baselines']['modality_mass']['SSE']
    gain = None if h is None or m is None else h - m
    checks.equal(row['head_minus_modality_SSE'], gain, 'head_minus_modality_difference')
    checks.equal(row['head_minus_modality_SSE_over_full_target'], ratio(gain, full), 'head_minus_modality_full_ratio')


def validate_parts(indexed, checks):
    for condition, layer, pair, window in itertools.product(('C', 'T'), (0, 8, 17), PAIRS, WINDOWS):
        a, b = pair
        full = indexed[(condition, layer, a, b, 'full', window)]
        for basis in ('response', 'clean'):
            inner = indexed[(condition, layer, a, b, basis + '_B3', window)]
            outer = indexed[(condition, layer, a, b, basis + '_residual', window)]
            for part in (inner, outer):
                checks.equal(part['full_target_delta_energy'], full['target_delta_energy'], 'shared_full_denominator')
            checks.equal(full['full_target_delta_energy'], full['target_delta_energy'], 'full_view_denominator')
            vals = (inner['target_delta_energy'], outer['target_delta_energy'])
            checks.equal(full['target_delta_energy'], None if None in vals else sum(vals), 'part_target_closure')
            for name in BASELINES:
                values = (inner['baselines'][name]['SSE'], outer['baselines'][name]['SSE'])
                checks.equal(full['baselines'][name]['SSE'], None if None in values else sum(values), 'part_SSE_closure')


def fixed_index(rows, *, identity, aggregate, checks):
    indexed = {}
    for row in rows:
        key = row_key(row)
        need(key not in indexed, 'duplicate_view_window_row')
        for name, expected in identity.items():
            need(row[name] == expected, 'row_identity_' + name)
        validate_scalar_row(row, checks, aggregate=aggregate)
        indexed[key] = row
    need(set(indexed) == EXPECTED_TAGS, 'complete_270_view_window_rows')
    validate_parts(indexed, checks)
    return indexed


def derived_descriptors(descriptors):
    expected = {}
    for condition, layer, (a, b) in itertools.product(('C', 'T'), (0, 8, 17), PAIRS):
        prefix = f'{condition}_L{layer}_j{a}to{b}__'
        for name, (shape, dtype) in DERIVED_ABI.items():
            expected[prefix + name] = (shape, dtype)
    need(set(descriptors) == set(expected), 'exact180_derived_field_descriptors')
    total = 0
    for key, (shape, dtype) in expected.items():
        desc = descriptors[key]
        size = math.prod(shape) * (8 if dtype == 'float64' else 1)
        need(desc['shape'] == list(shape) and desc['dtype'] == dtype and desc['bytes'] == size,
             'derived_descriptor_ABI_' + key)
        need(len(desc['payload_sha256']) == 64 and all(c in '0123456789abcdef' for c in desc['payload_sha256']),
             'derived_descriptor_payload_SHA')
        total += size
    need(total == 1628118, 'derived_query_raw_byte_total')


def check_mean(row, members, checks):
    for field in ('target_delta_energy', 'full_target_delta_energy'):
        checks.equal(row[field], mean([r[field] for r in members]), 'fixed_membership_' + field)
    for name in BASELINES:
        checks.equal(row['baselines'][name]['SSE'], mean([r['baselines'][name]['SSE'] for r in members]),
                     'fixed_membership_SSE')


def old_consistency(aggregates, parent, checks):
    need(parent['analysis_complete'] is True and parent['synthetic'] is False and
         parent['qualified_queries'] == 24 and parent['qualified_roots'] == 8 and
         parent['qualified_comparisons'] == 432 and parent['missing_queries'] == parent['numerical_failures'] == 0,
         'complete_pinned_parent_summary')
    mapping = {'modality_mass': 'mass', 'within_modality': 'within', 'signed_scalar_oracle': 'oracle', 'zero_change': 'nochange'}
    count = 0
    for level, groups in aggregates.items():
        old = {(r['aggregation_id'], r['window']): r for r in parent['main_aggregates'][level]}
        need(len(old) == len(groups) * 3, 'complete_parent_main_aggregates')
        for label, indexed in groups.items():
            for window in WINDOWS:
                row = indexed[('C', 17, 0, 9, 'full', window)]
                previous = old[label, window]
                need(previous['planned_queries'] == row['planned_queries'] and
                     previous['observed_queries'] == row['observed_queries'], 'old_baseline_same_population')
                checks.equal(row['target_delta_energy'], previous['target_delta_energy_mean'], 'old_target')
                for name, old_name in mapping.items():
                    checks.equal(row['baselines'][name]['SSE'], previous[old_name + '_error_energy_mean'], 'old_' + old_name + '_SSE')
                    checks.equal(row['baselines'][name]['relative_SSE'], previous[old_name + '_error_ratio_of_mean_energy'],
                                 'old_' + old_name + '_ratio')
                count += 1
    return {'matched_aggregate_panels': count, 'baselines': mapping, 'scope': 'C/L18/0-to-9; all three windows;8roots4tasks1population; old summary is not a perquery oracle reference'}


def compact(row, *, identity_fields=(), baselines=FIGURE_BASELINES, decomposition=False):
    names = tuple(identity_fields) + ('view', 'window', 'comparison_role', 'target_delta_energy',
                                   'full_target_delta_energy', 'head_minus_modality_SSE')
    result = {k: row[k] for k in names}
    result['baselines'] = {b: {k: row['baselines'][b][k] for k in ('SSE', 'status')} for b in baselines}
    for name in baselines:
        for field in ('zero_mass_noop_head_positions', 'defined_head_positions', 'planned_head_positions',
                      'observed_rows', 'planned_rows'):
            if field in row['baselines'][name]:
                result['baselines'][name][field] = row['baselines'][name][field]
    if decomposition:
        result['decomposition'] = row['decomposition']
        result['oracle_full_window_alpha'] = row['oracle_full_window_alpha']
        result['oracle_full_window_source_energy'] = row['oracle_full_window_source_energy']
    return result


def validate_and_extract(summary, details, aggregate_doc, inventory, parent, refs):
    """Pure small-JSON checking. No arrays or new scientific estimands."""
    checks = Checks()
    need(summary['schema'] == 'PAPER2-saved-Gram-run-v1' and summary['analysis_complete'] is True and
         summary['synthetic'] is False and not summary['failures'] and not summary['missing_queries'], 'successful_summary_no_failures')
    need(summary['planned_queries'] == summary['completed_queries'] == 24 and
         summary['planned_comparisons'] == summary['completed_comparisons'] == 432, 'fixed24_and432_summary')
    need(summary['parent_run_id'] == PARENT_RUN and summary['posthoc_CONF_reuse'] is True and
         summary['new_model_calls'] == summary['new_env_steps'] == summary['cloud_calls'] == summary['input_deletions'] == 0 and
         summary['new_fit'] is False, 'unchanged_analysis_scope')
    need(same(summary['source_manifest'], refs['producer_source_manifest']) and
         same(summary['input_inventory'], refs['input_inventory']), 'summary_source_and_inventory_SHA')
    selected = {(q['root_id'], q['query_index']): q for q in inventory['queries']}
    need(len(selected) == len(inventory['queries']) == 24 and set(selected) == EXPECTED_QUERIES, 'fixed_inventory_population')
    completed = {(q['root_id'], q['query_index']): q for q in summary['completed']}
    need(len(completed) == len(summary['completed']) == 24 and set(completed) == EXPECTED_QUERIES and
         all(q['comparisons'] == 18 for q in completed.values()), 'complete_query_receipts')
    need(summary['verified_query_reads'] == inventory['queries'] and summary['verified_local_input_body_bytes'] == 26236769 and
         summary['planned_input_body_bytes'] == 26236769 and summary['failed_unverified_read_bytes'] == 0,
         'full_verified_input_read_accounting')
    need(summary['derived_raw_array_bytes'] == 39074832, 'exact_derived_raw_bytes')
    queries = {}
    for name, detail in details.items():
        source = detail['input']
        key = (source['root_id'], source['query_index'])
        need(key not in queries and key in selected and source == selected[key], 'exact_query_input_provenance')
        need(name == source['path'].replace('.npz', '.json.gz') and
             same(detail['source_manifest'], refs['producer_source_manifest']), 'query_file_and_source')
        need(detail['posthoc_reuse'] is True and detail['signed_vector_reconstructed'] is False and
             detail['native_W_inference'] is False, 'query_scope')
        derived_descriptors(detail['array_descriptors'])
        need(detail['derived'] == completed[key]['derived'] and same(refs['outputs'][name], completed[key]['detail']) and
             same(refs['outputs'][detail['derived']['path']], detail['derived']), 'detail_and_derived_in_summary_manifest')
        need(len(detail['checks']) == 18 and len({(x['tags']['condition'], x['tags']['layer_index'],
             x['tags']['flow_from'], x['tags']['flow_to']) for x in detail['checks']}) == 18, 'eighteen_query_check_records')
        identity = {k: source[k] for k in ('root_id', 'query_index', 'task_id', 'source_identity')}
        need(all(all(check['tags'][k] == v for k, v in identity.items()) for check in detail['checks']),
             'query_check_record_identity')
        indexed = fixed_index(detail['rows'], identity=identity, aggregate=False, checks=checks)
        need(sum(r['comparison_role'] == 'primary' for r in indexed.values()) == 1 and
             sum(r['comparison_role'] == 'support_window' for r in indexed.values()) == 2, 'one_primary_panel_perquery')
        queries[key] = indexed
    need(set(queries) == EXPECTED_QUERIES, 'all24_query_details')
    need(aggregate_doc['schema'] == 'PAPER2-fixed-membership-aggregate-v1', 'aggregate_schema')
    groups = {'root': {r: [] for r in ROOTS}, 'task': {t: [] for t in TASKS}, 'population': {'all': []}}
    for level in groups:
        for row in aggregate_doc[level]:
            need(row['aggregation_id'] in groups[level] and row['aggregation_level'] == level, 'known_aggregate_identity')
            groups[level][row['aggregation_id']].append(row)
    aggregate_indices = {level: {label: fixed_index(rows, identity={'aggregation_id': label, 'aggregation_level': level},
                        aggregate=True, checks=checks) for label, rows in bins.items()} for level, bins in groups.items()}
    for root, indexed in aggregate_indices['root'].items():
        for tag, row in indexed.items():
            need(row['planned_queries'] == row['observed_queries'] == 3 and
                 row['planned_roots'] == row['observed_roots'] == 1 and row['planned_rows'] == 3, 'root_fixed3_support')
            check_mean(row, [queries[root, q][tag] for q in (11, 16, 22)], checks)
    for level, memberships in (('task', TASKS), ('population', {'all': ROOTS})):
        for label, roots in memberships.items():
            for tag, row in aggregate_indices[level][label].items():
                need(row['planned_queries'] == row['observed_queries'] == 3 * len(roots) and
                     row['planned_roots'] == row['observed_roots'] == len(roots) and row['planned_rows'] == len(roots),
                     'aggregate_fixed_root_support')
                check_mean(row, [aggregate_indices['root'][r][tag] for r in roots], checks)
    old = old_consistency(aggregate_indices, parent, checks)
    figure = {'schema': 'PAPER2-mass-baseline-figure-data-v1', 'synthetic': False,
              'planned_roots': [{'root_id': r, 'task_id': t, 'parent_condition': r[-1], 'query_indices': [11, 16, 22]}
                                for t, rr in TASKS.items() for r in rr],
              'root_rows': [], 'task_rows': [], 'population_rows': [], 'query_rows': []}
    for level in aggregate_indices:
        for label, indexed in aggregate_indices[level].items():
            for view, window in itertools.product(VIEWS, WINDOWS):
                row = indexed[('C', 17, 0, 9, view, window)]
                value = compact(row, identity_fields=('aggregation_id',))
                if level == 'root':
                    value['task_id'] = next(t for t, rr in TASKS.items() if label in rr)
                figure[level + '_rows'].append(value)
    decision = {'primary_query_rows': [], 'primary_root_rows': [], 'primary_task_rows': [], 'primary_population_rows': []}
    for key in sorted(queries):
        row = queries[key][('C', 17, 0, 9, 'full', 'full10')]
        figure['query_rows'].append(compact(row, identity_fields=('root_id', 'task_id', 'query_index')))
        decision['primary_query_rows'].append(compact(row, identity_fields=('root_id', 'task_id', 'query_index', 'source_identity'),
                                                    baselines=BASELINES, decomposition=True))
    for level in aggregate_indices:
        decision['primary_' + level + '_rows'] = [compact(indexed[('C', 17, 0, 9, 'full', 'full10')],
                  identity_fields=('aggregation_id',), baselines=BASELINES) for indexed in aggregate_indices[level].values()]
    return {'support': {'queries': 24, 'comparisons': 432, 'query_view_window_rows': 6480,
             'root_view_window_rows': 2160, 'task_view_window_rows': 1080, 'population_view_window_rows': 270,
             'roots': 8, 'tasks': 4, 'primary_query_panels': 24, 'posthoc_CONF_reuse': True},
            'checks': {'scalar_checks': checks.count, 'max_scaled_error': checks.max_scaled_error,
                       'by_family': checks.by_family, 'old_baseline_consistency': old},
            'figure_data': figure, 'decision_data': decision}


def validate_sources(inventory_path, producer_path, parent_path, review_manifest, review_sha, deadline):
    sources, docs = {}, {}
    for name, path, pin in (('producer_source_manifest', producer_path, PRODUCER_MANIFEST),
                            ('input_inventory', inventory_path, INVENTORY), ('parent_summary', parent_path, PARENT_SUMMARY)):
        raw, ref = read_or_hash(path, ref=pin, cap=2 * 1024**2, deadline=deadline)
        sources[name] = ref
        docs[name] = unique_json(raw)
    raw, ref = read_or_hash(review_manifest, cap=1024**2, deadline=deadline)
    need(ref['sha256'] == review_sha, 'review_source_manifest_SHA')
    manifest = unique_json(raw)
    need(manifest['schema'] == 'PAPER2-result-review-source-manifest-v1', 'review_source_manifest_schema')
    by = {r['path']: r for r in manifest['files']}
    need(len(by) == len(manifest['files']) and 'review.py' in by, 'review_source_pinned')
    need(Path(review_manifest).parent.resolve() == HERE, 'review_manifest_with_executed_source')
    read_or_hash(HERE / 'review.py', ref=by['review.py'], cap=2 * 1024**2, keep=False, deadline=deadline)
    sources['review_source_manifest'] = ref
    producer_files = {r['path']: r for r in docs['producer_source_manifest']['files']}
    need(same(producer_files['SOURCE-INPUT-INVENTORY.json'], sources['input_inventory']), 'producer_inventory_pin')
    need(same(docs['input_inventory']['metadata']['PARENT-SUMMARY.json'], sources['parent_summary']), 'old_summary_inventory_pin')
    return sources, docs


def review_result(result_dir, inventory_path, producer_path, parent_path, review_manifest, review_sha,
                  output_manifest_sha, *, deadline_epoch):
    need(sys.platform.startswith('linux'), 'actual_result_review_cloud_Linux_only')
    result = {'schema': SCHEMA, 'status': 'REJECTED', 'accepted': False, 'synthetic': False,
              'source_files': [], 'source_bindings': {}, 'support': {}, 'checks': {},
              'failures': [], 'figure_data': None, 'decision_data': None,
              'new_model_calls': 0, 'new_env_steps': 0, 'NPZ_decodes': 0,
              'scope': 'Acceptance and extraction of existing scalar results only. Provider release and BOS persistence remain separate root checks.'}
    result['requested_source_bindings'] = {'review_source_manifest_sha256': review_sha,
                                           'output_manifest_sha256': output_manifest_sha}
    try:
        deadline = Deadline(deadline_epoch)
        base = Path(result_dir)
        need(base.is_dir() and not base.is_symlink(), 'regular_result_directory')
        # These markers take precedence even if their bytes are malformed,
        # unlisted in MANIFEST, or precede a stale successful SUMMARY.
        for name in FAILURE_MARKERS:
            if (base / name).exists() or (base / name).is_symlink():
                failure = {'stage': 'producer_failure_precedence', 'path': name}
                try:
                    raw, ref = read_or_hash(base / name, cap=65536, deadline=deadline)
                    failure.update(identity=ref, content=unique_json(raw))
                except Exception as error:
                    failure['unreadable_marker'] = {'type': type(error).__name__, 'error': str(error)[:300]}
                result['failures'].append(failure)
        if result['failures']:
            return result
        refs, docs = validate_sources(inventory_path, producer_path, parent_path, review_manifest, review_sha, deadline)
        result['source_bindings'] = refs.copy()
        raw, mref = read_or_hash(base / 'MANIFEST.json', cap=65536, deadline=deadline)
        need(mref['sha256'] == output_manifest_sha, 'independently_bound_result_manifest_SHA')
        manifest = unique_json(raw)
        need(manifest['schema'] == 'PAPER2-saved-Gram-output-manifest-v1' and manifest['analysis_complete'] is True and
             same(manifest['source_manifest'], refs['producer_source_manifest']), 'complete_source_bound_output_manifest')
        files = {r['path']: r for r in manifest['files']}
        need(len(files) == len(manifest['files']) and set(files) == RESULT_NAMES, 'exact_fifty_manifest_objects')
        need({p.name for p in base.iterdir()} == RESULT_NAMES | {'MANIFEST.json'}, 'no_unknown_result_files')
        need(sum(r['bytes'] for r in files.values()) + mref['bytes'] <= 64 * 1024**2, 'physical_output_budget')
        refs['output_manifest'], refs['outputs'] = mref, files
        result['source_bindings']['output_manifest'] = mref
        result['source_files'].append({'role': 'output_manifest', 'path': 'MANIFEST.json', **mref})
        loaded = {}
        for name, ref in sorted(files.items()):
            # DERIVED NPZ bytes are hashed in cloud and immediately discarded.
            raw, identity = read_or_hash(base / name, ref=ref, cap=65536 if name == 'SUMMARY.json' else 4 * 1024**2,
                                        keep=not name.endswith('.npz'), deadline=deadline)
            result['source_files'].append({'role': 'producer_output', 'path': name, **identity})
            if raw is not None:
                loaded[name] = uncompress_json(raw, 16 * 1024**2 if name == 'AGGREGATES.json.gz' else 4 * 1024**2) if name.endswith('.gz') else unique_json(raw)
        summary = loaded['SUMMARY.json']
        if summary.get('failures') or summary.get('missing_queries'):
            result['failures'].append({'stage': 'producer_summary', 'failures': summary.get('failures'),
                                      'missing_queries': summary.get('missing_queries'),
                                      'summary_identity': files['SUMMARY.json']})
        need(same(summary['aggregates'], files['AGGREGATES.json.gz']), 'summary_aggregate_identity')
        details = {k: v for k, v in loaded.items() if k.startswith('QUERY-')}
        extracted = validate_and_extract(summary, details, loaded['AGGREGATES.json.gz'], docs['input_inventory'], docs['parent_summary'], refs)
        deadline.check()
        result.update(extracted, status='ACCEPTED_COMPLETE', accepted=True)
        result['checks'].update(output_files_hashed=51, NPZ_objects_hashed=24, NPZ_decodes=0,
                                derived_descriptor_shapes_checked=24 * 180, NPZ_payloads_decoded_for_validation=False,
                                physical_output_bytes=sum(r['bytes'] for r in files.values()) + mref['bytes'])
    except Exception as error:
        result['failures'].append({'stage': 'review', 'type': type(error).__name__, 'error': str(error)[:1000]})
        result.update(status='REJECTED', accepted=False, figure_data=None, decision_data=None)
    return result


def write_export(directory, result):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    raw = encoded(result)
    if len(raw) > MAX_EXPORT:
        result = {'schema': SCHEMA, 'status': 'REJECTED', 'accepted': False, 'synthetic': False,
                  'source_bindings': result.get('source_bindings', {}), 'figure_data': None, 'decision_data': None,
                  'failures': result.get('failures', []) + [{'stage': 'export', 'error': 'result_review_exceeds_512KiB', 'bytes': len(raw)}]}
        raw = encoded(result)
    need(len(raw) <= MAX_EXPORT, 'failure_export_within_512KiB')
    path = directory / 'result-review.json'
    with path.open('xb') as stream:
        stream.write(raw)
    receipt = {'schema': 'PAPER2-result-review-export-v1', 'status': result['status'], 'accepted': result['accepted'],
               'file': {'path': path.name, **ident(raw)}, 'source_bindings': result.get('source_bindings', {})}
    with (directory / 'REVIEW-RECEIPT.json').open('xb') as stream:
        stream.write(encoded(receipt))
    return receipt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('result-dir', 'inventory', 'producer-source-manifest', 'parent-summary', 'source-manifest', 'out'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--source-manifest-sha256', required=True)
    p.add_argument('--output-manifest-sha256', required=True)
    p.add_argument('--deadline-epoch', type=float, required=True)
    a = p.parse_args()
    need(not a.out.resolve().is_relative_to(a.result_dir.resolve()), 'review_output_outside_producer_directory')
    result = review_result(a.result_dir, a.inventory, a.producer_source_manifest, a.parent_summary,
                           a.source_manifest, a.source_manifest_sha256, a.output_manifest_sha256,
                           deadline_epoch=a.deadline_epoch)
    receipt = write_export(a.out, result)
    print(json.dumps(receipt))
    return 0 if receipt['accepted'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
