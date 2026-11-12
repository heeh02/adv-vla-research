#!/usr/bin/env python3
"""Describe existing pi0 Sensor Noise cases, without inference or resampling.

Uses only the Python standard library.  Run from any directory:
    python3 build_case_audit.py
Inputs are byte-identical copies of the original local derived-record package.
The selected record objects are copied as original JSON byte fragments.
"""
from collections import Counter, defaultdict
import csv
import gzip
import hashlib
import io
import json
import math
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent
INPUT = ROOT / 'inputs'
MODEL = 'pi0_libero'
FAMILY = 'Sensor Noise'
ARMS = ('R0', 'T0', 'R1', 'T1', 'R0prime')
PAIRS = {'TR0': ('R0', 'T0'), 'TR1': ('R1', 'T1'),
         'R0_R1': ('R0', 'R1'), 'R0_R0prime': ('R0', 'R0prime')}
GROUPS = ('SF', 'FS', 'SS', 'FF')
RELATIONS = ('below', 'equal', 'above', 'missing')
ORIGINAL_RATIO = 0.5803425149195939
ORIGINAL_RD = -0.14285714285714285


def sha(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False,
                               allow_nan=False) + '\n')


def fragments(raw):
    """Read the top-level rows array, retaining each exact object byte sequence."""
    text = raw.decode('utf-8')
    match = re.search(r'"rows"\s*:\s*\[', text)
    assert match is not None
    pos = match.end()
    decoder = json.JSONDecoder()
    result = []
    while True:
        while text[pos].isspace() or text[pos] == ',':
            pos += 1
        if text[pos] == ']':
            break
        start = pos
        value, pos = decoder.raw_decode(text, pos)
        fragment = text[start:pos].encode('utf-8')
        result.append((value, fragment, start, pos))
    return result


def selected(row):
    return (row['model'] == MODEL and row['family'] == FAMILY
            and row['common28'] is True)


def outcome_cell(a, b):
    if a is None or b is None:
        return None
    assert type(a) is bool and type(b) is bool
    return ('S' if a else 'F') + ('S' if b else 'F')


def motion(row, pair):
    reasons = []
    if not row['identity_qualified'] or row['validation_errors']:
        reasons.append('identity_unqualified')
    if row['motion_status'] != 'processed':
        reasons.append('motion_not_processed')
    for arm in PAIRS[pair]:
        q = row['arm_qualification'][arm]
        if 5 not in (q['available_pose_clocks'] or []):
            reasons.append('control_5_pose_not_verified:' + arm)
        if q['issues']:
            reasons.append('arm_issues:' + arm)
    value = row['motion'][pair]['5']['position_distance']
    assert value['unit'] == 'metre'
    if value['status'] != 'measured' or value['value'] is None:
        reasons.append(value.get('reason') or 'motion_unavailable')
    if pair == 'R0_R0prime' and row['outcomes']['R0prime'] is None:
        reasons.append('repeat_outcome_unvalidated')
    if reasons:
        return None, reasons
    x = value['value']
    assert type(x) in (int, float) and math.isfinite(x) and x >= 0
    return x, []


def compare(value, reference):
    if value is None or reference is None:
        return 'missing'
    return 'below' if value < reference else 'above' if value > reference else 'equal'


def summarize_cells(pairs, relation_field):
    out = {}
    for group in GROUPS:
        group_pairs = [p for p in pairs if p['outcome_cell'] == group]
        group_mass = math.fsum(p['task_variant_noise_weight'] for p in group_pairs)
        out[group] = {
            'pair_count': len(group_pairs),
            'unique_variants': len({p['variant_id'] for p in group_pairs}),
            'unique_tasks': len({p['task_id'] for p in group_pairs}),
            'whole_panel_task_weight_mass': group_mass,
            'relations': {},
        }
        for relation in RELATIONS:
            rp = [p for p in group_pairs if p[relation_field] == relation]
            mass = math.fsum(p['task_variant_noise_weight'] for p in rp)
            out[group]['relations'][relation] = {
                'pair_count': len(rp),
                'unique_variants': len({p['variant_id'] for p in rp}),
                'unique_tasks': len({p['task_id'] for p in rp}),
                'whole_panel_task_weight_mass': mass,
                'conditional_weight_share_given_outcome': mass / group_mass if group_mass else None,
            }
        assert sum(r['pair_count'] for r in out[group]['relations'].values()) == len(group_pairs)
    assert sum(v['pair_count'] for v in out.values()) == len(pairs)
    return out


def main():
    provenance = json.loads((INPUT / 'SOURCE-MANIFEST.json').read_text())
    for filename, metadata in provenance['files'].items():
        raw = (INPUT / filename).read_bytes()
        assert sha(raw) == metadata['sha256'] and len(raw) == metadata['bytes'], filename

    original_gz = (INPUT / 'MOTION-C-PLOT-v1.json.gz').read_bytes()
    original_bytes = gzip.decompress(original_gz)
    payload = json.loads(original_bytes)
    plan_bytes = (INPUT / 'PANEL-PLAN.json').read_bytes()
    plan = json.loads(plan_bytes)
    config = json.loads((INPUT / 'CONFIG.json').read_text())
    assert sha(original_gz) == config['input_sha256']
    assert len(original_gz) == config['input_bytes']
    assert sha(plan_bytes) == config['plan_sha256']
    assert payload['schema'] == 'joint-motion-c-plot-v1'
    assert payload['data_class'] == 'observed_existing_scalars'
    assert len(payload['rows']) == len(plan['rows']) == 536

    pindex = {(p['model'], p['variant_id']): p for p in plan['rows']}
    rindex = {(r['model'], r['variant_id']): r for r in payload['rows']}
    assert len(pindex) == len(rindex) == 536 and set(pindex) == set(rindex)
    for key, row in rindex.items():
        p = pindex[key]
        assert all(row[k] == p[k] for k in ('suite', 'family', 'common28', 'condition', 'official_level'))
        assert row['task_id'] == p['suite'] + ':' + p['task']
    rows = [r for r in payload['rows'] if selected(r)]
    planned_keys = {(p['model'], p['variant_id']) for p in plan['rows'] if selected(p)}
    assert {(r['model'], r['variant_id']) for r in rows} == planned_keys
    tasks = sorted({r['task_id'] for r in rows})
    assert len(rows) == 55 and len(tasks) == 28
    variants_per_task = Counter(r['task_id'] for r in rows)

    selected_fragments = [(v, f, start, end) for v, f, start, end in fragments(original_bytes) if selected(v)]
    assert [v for v, f, start, end in selected_fragments] == rows
    header = json.dumps({'schema': 'byte-preserved-source-row-subset-v1',
                         'original_compressed_source_sha256': sha(original_gz),
                         'original_decompressed_source_sha256': sha(original_bytes),
                         'selection': {'model': MODEL, 'family': FAMILY, 'common28': True}},
                        separators=(',', ':')).encode()[:-1]
    selected_bytes = header + b',"rows":[\n' + b',\n'.join(f for v, f, start, end in selected_fragments) + b'\n]}\n'
    selected_gz = gzip.compress(selected_bytes, mtime=0)
    (ROOT / 'SELECTED-55-ORIGINAL-ROWS.json.gz').write_bytes(selected_gz)
    reread_fragments = fragments(gzip.decompress(selected_gz))
    assert [f for v, f, start, end in selected_fragments] == [f for v, f, start, end in reread_fragments]
    fragment_hashes = [{'model': v['model'], 'variant_id': v['variant_id'],
                       'original_object_sha256': sha(f), 'object_bytes': len(f)}
                      for v, f, start, end in selected_fragments]

    with gzip.open(INPUT / 'C-COMMAND-MOTION-POINTS-v1.csv.gz', 'rt') as stream:
        points = list(csv.DictReader(stream))
    point_index = {}
    for p in points:
        key = (p['model'], p['variant_id'], int(p['noise']), int(p['step']))
        assert key not in point_index
        point_index[key] = p

    pairs = []
    variants = []
    checks = Counter()
    point_deltas = []
    for row in rows:
        assert row['identity_qualified'] is True and not row['validation_errors']
        assert row['motion_status'] == 'processed'
        assert set(row['outcomes']) == set(row['arm_qualification']) == set(ARMS)
        for arm in ARMS:
            o = row['outcomes'][arm]
            q = row['arm_qualification'][arm]
            assert type(o) is bool
            assert q['status'] == ('accepted_success' if o else 'accepted_failure')
            assert not q['issues'] and 5 in q['available_pose_clocks']
            checks['qualified_arm_outcome_and_control5_pose'] += 1
        rr, rr_reasons = motion(row, 'R0_R1')
        repeat, repeat_reasons = motion(row, 'R0_R0prime')
        assert not rr_reasons and not repeat_reasons
        visual = []
        for noise in (0, 1):
            n = str(noise)
            cell = outcome_cell(row['outcomes']['R'+n], row['outcomes']['T'+n])
            assert cell == row['noise_pair_cells'][n]
            value, reasons = motion(row, 'TR'+n)
            assert not reasons
            visual.append(value)
            point = point_index[(MODEL, row['variant_id'], noise, 5)]
            assert point['task_id'] == row['task_id'] and point['family'] == row['family']
            assert point['Y'] == cell and point['source_sha256'] == row['source_sha256']
            assert point['included'] == 'True' and point['motion_status'] == 'measured'
            delta = abs(float(point['eef_mm']) / 1000 - value)
            assert delta == 0
            point_deltas.append(delta)
            checks['exact_same_source_figure_point_matches'] += 1
            pairs.append({
                'model': MODEL, 'family': FAMILY, 'variant_id': row['variant_id'],
                'task_id': row['task_id'], 'suite': row['suite'], 'condition': row['condition'],
                'noise_label': noise, 'returned_control_step': 5,
                'outcome_cell': cell, 'clean_success': row['outcomes']['R'+n],
                'perturbed_success': row['outcomes']['T'+n],
                'visual_distance_metres': value, 'clean_resampling_distance_metres': rr,
                'clean_R0_repeat_distance_metres': repeat,
                'relative_to_clean_resampling': compare(value, rr),
                'relative_to_clean_R0_repeat': compare(value, repeat),
                'visual_over_clean_resampling': value / rr if rr and rr > 0 else None,
                'visual_over_clean_R0_repeat': value / repeat if repeat and repeat > 0 else None,
                'resampling_ratio_status': 'defined' if rr and rr > 0 else 'undefined_zero_reference',
                'repeat_ratio_status': 'defined' if repeat and repeat > 0 else 'undefined_zero_reference',
                'task_variant_noise_weight': 1 / len(tasks) / variants_per_task[row['task_id']] / 2,
                'source_sha256': row['source_sha256'],
                'run_id': row['run_id'], 'job_id': row['job_id'], 'sequence': row['sequence'],
                'source_identity_qualified': row['identity_qualified'],
            })
        variants.append({'variant_id': row['variant_id'], 'task_id': row['task_id'],
                         'mean_visual_distance_metres': math.fsum(visual) / 2,
                         'clean_resampling_distance_metres': rr,
                         'clean_R0_repeat_distance_metres': repeat,
                         'resampling_outcome_cell': outcome_cell(row['outcomes']['R0'], row['outcomes']['R1']),
                         'repeat_outcome_cell': outcome_cell(row['outcomes']['R0'], row['outcomes']['R0prime']),
                         'weight': 1 / len(tasks) / variants_per_task[row['task_id']]})

    def task_mean(field):
        return math.fsum(math.fsum(v[field] for v in variants if v['task_id'] == t) / variants_per_task[t]
                         for t in tasks) / len(tasks)

    visual_mean = task_mean('mean_visual_distance_metres')
    reference_mean = task_mean('clean_resampling_distance_metres')
    ratio = visual_mean / reference_mean
    clean_success = math.fsum(p['task_variant_noise_weight'] * p['clean_success'] for p in pairs)
    treated_success = math.fsum(p['task_variant_noise_weight'] * p['perturbed_success'] for p in pairs)
    rd = math.fsum(p['task_variant_noise_weight'] * (int(p['perturbed_success']) - int(p['clean_success'])) for p in pairs)
    assert abs(ratio - ORIGINAL_RATIO) < 1e-14 and abs(rd - ORIGINAL_RD) < 1e-14
    assert len(pairs) == 110 and abs(math.fsum(p['task_variant_noise_weight'] for p in pairs) - 1) < 1e-14

    resampling = summarize_cells(pairs, 'relative_to_clean_resampling')
    repeat = summarize_cells(pairs, 'relative_to_clean_R0_repeat')
    reference_outcomes = {}
    for ref, field in [('R0_R1', 'resampling_outcome_cell'), ('R0_R0prime', 'repeat_outcome_cell')]:
        reference_outcomes[ref] = {
            g: {'variant_count': sum(v[field] == g for v in variants),
                'whole_panel_task_weight_mass': math.fsum(v['weight'] for v in variants if v[field] == g)}
            for g in GROUPS}

    summary = {
        'schema': 'pi0-sensor-casewise-descriptive-audit-v1',
        'purpose': 'descriptive within-record comparison of existing motion and outcomes; no new confidence interval or predictor',
        'inputs': provenance,
        'selection': {'model': MODEL, 'family': FAMILY, 'common28': True,
                      'variant_count': len(rows), 'task_count': len(tasks),
                      'visual_pair_count': len(pairs), 'arm_outcome_count': len(rows)*5,
                      'variant_count_distribution_across_tasks': dict(Counter(variants_per_task.values()))},
        'matching': {
            'pair_identity': 'same source model-variant object; noise0=R0/T0, noise1=R1/T1',
            'clock': 'returned physical control5, position_distance in metres; no interpolation or terminal substitution',
            'reference': 'each variant uses its own R0/R1 clean-resampling separation for both symmetric noise-pair comparisons',
            'repeat': 'same-case R0/R0prime; no R1prime arm exists, so this is not a noise1-specific clean repeat',
            'initial_state_scope': 'original derived row identity_qualified and validation_errors are checked; the selected bundle has no explicit initial-state index/hash or raw poses to revalidate physical acquisition',
            'no_cross_record_initial_state_join': True,
            'independent_plan_membership_verified': True,
            'figure_points_key': ['model', 'variant_id', 'noise', 'step'],
            'figure_points_additional_fields_checked': ['task_id', 'family', 'Y', 'source_sha256', 'included', 'motion_status'],
            'figure_points_max_absolute_metres_difference': max(point_deltas),
            'checks': dict(checks),
        },
        'missing_and_zero_rules': {
            'classification': 'compare original stored distances directly with <, ==, >; no epsilon, clipping, or rounding',
            'missing': 'a missing/unqualified outcome or either distance remains missing, with no replacement denominator',
            'zero_reference': 'distance comparison stays defined; ratio is null when reference is zero, including 0/0',
            'observed_missing_arms': sum(r['outcomes'][a] is None for r in rows for a in ARMS),
            'observed_missing_visual_or_reference_pairs': sum(p['relative_to_clean_resampling']=='missing' for p in pairs),
            'zero_resampling_reference_variants': sum(v['clean_resampling_distance_metres']==0 for v in variants),
            'zero_R0_repeat_reference_variants': sum(v['clean_R0_repeat_distance_metres']==0 for v in variants),
            'zero_visual_distance_pairs': sum(p['visual_distance_metres']==0 for p in pairs),
            'min_absolute_visual_minus_resampling_gap_metres': min(abs(p['visual_distance_metres']-p['clean_resampling_distance_metres']) for p in pairs),
        },
        'weighting': {
            'pair_weight': '1 / (28 * planned Sensor variants in the task * 2)',
            'whole_panel_task_weight_mass': 'sum fixed pair weights for a cell; all four outcome groups and all relations sum to1',
            'conditional_share': 'cell mass / total fixed-weight mass of its outcome group; this is not an equal average over only outcome-eligible tasks',
            'unique_counts': 'variant/task counts can overlap across noise outcomes or below/above categories and are not additive',
            'dependence': 'two noise pairs share each variant and its reference; variants share tasks; raw pair counts are descriptive, not independent Bernoulli observations',
        },
        'same_input_reproduction': {
            'visual_task_equal_mean_metres': visual_mean,
            'resampling_task_equal_mean_metres': reference_mean,
            'ratio_of_task_equal_means': ratio,
            'original_ratio': ORIGINAL_RATIO,
            'ratio_absolute_error': abs(ratio-ORIGINAL_RATIO),
            'clean_success': clean_success, 'perturbed_success': treated_success,
            'risk_difference': rd, 'risk_difference_percentage_points': rd*100,
            'original_risk_difference': ORIGINAL_RD,
            'risk_difference_absolute_error': abs(rd-ORIGINAL_RD),
            'new_bootstrap_draws': 0, 'new_confidence_intervals': 0, 'new_model_calls': 0,
        },
        'all_110_pairs_relative_to_clean_resampling': resampling,
        'all_110_pairs_relative_to_R0_clean_repeat': repeat,
        'by_noise_relative_to_clean_resampling': {
            str(n): summarize_cells([p for p in pairs if p['noise_label']==n], 'relative_to_clean_resampling')
            for n in (0,1)},
        'reference_outcomes_on_55_variants': reference_outcomes,
        'original_object_preservation': {
            'selected_gzip_file': 'SELECTED-55-ORIGINAL-ROWS.json.gz',
            'selected_gzip_sha256': sha(selected_gz),
            'selected_decompressed_sha256': sha(selected_bytes),
            'all55_object_byte_sequences_match_source': True,
            'record_objects': fragment_hashes,
        },
        'candidate_sentence': 'Twelve of the 18 recorded harm pairs also show smaller five-control separation than their own clean-resampling reference.',
        'candidate_sentence_interpretation': '12 noise-paired executions in 10 variants and 8 tasks; all 18 harm pairs occupy 15 variants and 10 tasks. Descriptive and conditional on these recorded cases, with no independent-pair or prediction claim.',
    }
    write_json(ROOT / 'SUMMARY.json', summary)
    write_json(ROOT / 'PAIR-RECORDS-110.json', pairs)
    with (ROOT / 'PAIR-RECORDS-110.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(pairs[0]))
        writer.writeheader(); writer.writerows(pairs)
    write_json(ROOT / 'VARIANT-SUPPORT-55.json', variants)
    print(json.dumps({'support':[len(rows),len(tasks),len(pairs)],
                      'ratio':ratio, 'risk_difference_pp':rd*100,
                      'harm':resampling['SF'], 'all55_original_objects_byte_preserved':True}, indent=2))


if __name__ == '__main__':
    main()
