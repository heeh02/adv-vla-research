"""Reconcile exact small metadata identities. No array reads, fitting or cloud calls."""
from pathlib import Path
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import re
from eligibility import check, check_split

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[1]
OUT = BASE / 'analysis/prospective-prediction-reuse-20261021-v1'

def read(p):
    return json.loads(p.read_text())

def identity(p):
    raw = p.read_bytes()
    return {'path': str(p), 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}

def save(name, value):
    with (OUT / name).open('x') as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write('\n')

field_path = BASE / 'analysis/same-gstar-next-20261021-v1/FIELD-AND-COHORT-AUDIT.json'
field = read(field_path)
review = read(OUT / 'INDEPENDENT-REUSE-REVIEW.json')
for ref in review['sources'].values():
    if isinstance(ref, dict) and {'path', 'bytes', 'sha256'} <= set(ref):
        assert all(identity(Path(ref['path']))[k] == ref[k] for k in ('bytes', 'sha256'))
receipt = read(OUT / 'source-readback-v1/RECEIPT.json')
assert receipt['status'] == 'ALL_FOUR_SMALL_JSON_GET_SHA_VERIFIED'
assert receipt['compute_calls'] == receipt['array_GETs'] == 0

features = [{'field': x, 'origin': 'ownN'} for x in ['x0', 'v0', 'x5', 'v5', 'P_j5', 'gate_j5', 'mask', 'positions']]
features += [{'field': 'state', 'origin': 'original_input'}, {'field': 'query_clock', 'origin': 'original_input'}]
features += [{'field': x, 'origin': 'known_predeclared_payload'} for x in ['actual_E', 'WO']]
cases = []
for row in field['DEV8']['snapshots']:
    root = row['root_id']; task = re.fullmatch(r'SDRU-CDEV-(T\d+)-R\d+', root).group(1)
    cases.append({'cohort': 'DEV8', 'task': task, 'root_id': root, 'query_index': row['query_index'],
        'initialization_cluster': root, 'parent_branch': 'T', 'receiver': 'T',
        'roles': sorted(row['calls']), 'ownN_pairing_verified': True,
        'fresh_C_AE_observed': False, 'label_exposure': 'already_analyzed',
        'preregistration_before_label_exposure_verified': False, 'independent_new_unit_verified': False,
        'features': features, 'input_ref': row['new_input'],
        'known_E_source': row['old_actual_E_source'], 'state_source': row['old_state_noise_audit'],
        'call_refs': {k: v['compact'] for k, v in row['calls'].items()},
        'source_scope': 'Prior native72 acceptance and source metadata; no new NPZ GET in this audit.'})

matched_calls = 0
matched_inputs = 0
for ref in receipt['objects']:
    p = Path(ref['local_path'])
    assert all(identity(p)[k] == ref[k] for k in ('bytes', 'sha256'))
    source = read(p)
    assert source['qualification_from_complete_original_worker'] is True and len(source['queries']) == 6
    task = re.search(r'CCONF-(T\d+)', p.name).group(1)
    old = next(t for t in field['CONF24']['tasks'] if t['shard_id'] == 'CCONF-' + task)
    old_refs = {r['uri']: r for r in old['input_refs']}
    old_calls = {(r['root_id'], r['query_index'], r['role']): r for r in old['calls']}
    for row in source['queries']:
        root = row['root_id']; q = row['query_index']
        match = re.fullmatch(r'(SDRU-CCONF-T\d+-V\d+-I\d+)-P([CT])', root)
        assert match and row['all_readout_calls_source_qualified'] is True
        inp = row['input_ref']; old_inp = old_refs[inp['uri']]
        assert all(inp[k] == old_inp[k] for k in ('bytes', 'sha256'))
        matched_inputs += 1
        calls = {}
        for c in row['selected_calls']:
            prev = old_calls[(root, q, c['role'])]
            assert prev['call_id'] == c['call_id']
            assert all(prev['ref'][k] == c['ref'][k] for k in ('uri', 'bytes', 'sha256'))
            assert c['role'] not in calls
            calls[c['role']] = c['ref']; matched_calls += 1
        assert set(calls) == {'baseline_none_1', 'shared_pulse', 'shared_persistent'}
        cases.append({'cohort': 'CONF24', 'task': task, 'root_id': root, 'query_index': q,
            'initialization_cluster': match.group(1), 'parent_branch': match.group(2), 'receiver': 'T',
            'roles': sorted(calls), 'ownN_pairing_verified': True,
            'fresh_C_AE_observed': False, 'label_exposure': 'already_analyzed',
            'preregistration_before_label_exposure_verified': False, 'independent_new_unit_verified': False,
            'features': features, 'input_ref': inp, 'call_refs': calls,
            'source_metadata_ref': ref,
            'source_scope': 'Fresh small SOURCE JSON GET/SHA plus prior original-array readout qualification; no new array revalidation.'})
assert len(cases) == 32 and matched_inputs == 24 and matched_calls == 72
assert len({(r['root_id'], r['query_index']) for r in cases}) == 32
checks = []
for c in cases:
    checks.append({'cohort': c['cohort'], 'root_id': c['root_id'], 'query_index': c['query_index'],
        'T_only_shared_retrospective': check(c, target='shared_pulse', information_set='T_only_shadow', cutoff='after_unedited_j5_before_edited_output'),
        'T_only_SR_retrospective': check(c, target='shared_minus_random_pulse', information_set='T_only_shadow', cutoff='after_unedited_j5_before_edited_output'),
        'fresh_CT_shared_retrospective': check(c, target='shared_pulse', information_set='fresh_CT_shadow', cutoff='after_unedited_j5_before_edited_output'),
        'T_only_shared_prospective': check(c, target='shared_pulse', information_set='T_only_shadow', cutoff='after_unedited_j5_before_edited_output', prospective=True)})
counts = {k: sum(r[k]['eligible_for_requested_scientific_use'] for r in checks) for k in checks[0] if k.startswith(('T_only', 'fresh_CT'))}
assert counts == {'T_only_shared_retrospective': 32, 'T_only_SR_retrospective': 8, 'fresh_CT_shared_retrospective': 0, 'T_only_shared_prospective': 0}
folds = []
for cohort in ('DEV8', 'CONF24'):
    subset = [c for c in cases if c['cohort'] == cohort]
    units = sorted({c['initialization_cluster'] for c in subset})
    assert len(units) == 4
    for unit in units:
        train = [c for c in subset if c['initialization_cluster'] != unit]
        test = [c for c in subset if c['initialization_cluster'] == unit]
        assert check_split(train, test)['group_disjoint']
        folds.append({'cohort': cohort, 'held_cluster': unit, 'training_snapshots': len(train),
                      'test_snapshots': len(test), 'training_clusters': 3, 'test_clusters': 1,
                      'prospective': False, 'fits_executed': 0})
native = read(BASE / 'analysis/rss-dev16-native-v2/ACTUAL-ACCEPTANCE.json')
save('CASE-ROSTER.json', {'schema': 'paper2-existing-prediction-case-roster-v1', 'cases': cases,
    'fresh_conf_metadata_inputs_joined': matched_inputs, 'fresh_conf_metadata_calls_joined': matched_calls,
    'source_audit': identity(field_path), 'source_readback': identity(OUT / 'source-readback-v1/RECEIPT.json'),
    'independent_review': identity(OUT / 'INDEPENDENT-REUSE-REVIEW.json'), 'model_or_array_reads': 0})
save('ELIGIBILITY-RESULT.json', {'schema': 'paper2-prediction-metadata-eligibility-v1', 'status': 'COMPLETE_WITH_SCIENTIFIC_GAPS',
    'counts': counts, 'cases': checks, 'grouped_fold_metadata_only': folds,
    'body_level_dataset_ready': False, 'execution_authorized': False, 'new_fits': 0,
    'interpretation': '32 shared-only retrospective metadata candidates;8 S/R candidates. None is fresh CT or new prospective evidence. Passing metadata eligibility does not establish fit-worthiness or body-integrity qualification.'})
save('MISSING-EVIDENCE-AND-COST.json', {
    'schema': 'paper2-prospective-prediction-missing-evidence-cost-v1', 'UTC': datetime.now(timezone.utc).isoformat(),
    'existing_reuse': {'DEV_pulse_labels': 16, 'DEV_snapshots': 8, 'DEV_initializations': 4,
                       'CONF_shared_pulse_labels': 24, 'CONF_random_pulse_labels': 0,
                       'CONF_parent_trajectories': 8, 'CONF_initialization_pairs': 4,
                       'new_independent_unrevealed_units_in_this_roster': 0},
    'gaps': [
        {'name': 'fresh_C_AE_for_historical_CT_context', 'DEV_missing_snapshots': 8, 'CONF_missing_snapshots': 24,
         'minimum_new_full_C_baseline_calls_if_retaining_CT': 32, 'complete_budget': False,
         'excludes': ['repeat/zero controls', 'restore/model load', 'new independent populations'],
         'alternative': 'T-only is a new information-set definition; it does not reproduce historical context9.'},
        {'name': 'CONF_random_pulse_for_SR_contrast', 'missing_labels': 24,
         'minimum_new_edited_full_calls': 24, 'complete_budget': False,
         'requires_before_execution': ['outcome-independent random-edit definition', 'fresh-run none/parity and null/repeat gates', 'exact frozen payload roster'],
         'not_recommended_as_prospective_replacement': 'These24 cases are already analyzed; addingR doesnotcreate new initialization/task support.'},
        {'name': 'full_DEV48_pulse_fill', 'already_qualified_labels': 16, 'missing_labels': 80,
         'missing_snapshots': 40, 'edited_call_lower_bound': 80,
         'reference_cost_if_original9call_protocol_repeated_on_missing40only': 360,
         'reference_protocol_also_collects_P_and_qualification': True,
         'not_automatically_recommended': 'No new independent roots; this fill is not required merely to reach96 labels.'},
        {'name': 'new_independent_prospective_support', 'verified_eligible_units': 0,
         'new_unit_count': None, 'full_call_count': None, 'GPU_hours': None,
         'reason_unpriced': 'No source-bound new initialization/task roster, parent acquisition contract or valid population precision basis is fixed. Unknown isnotzero.'}
    ],
    'measured_reference_only': {'native_DEV16_acceptance': identity(BASE / 'analysis/rss-dev16-native-v2/ACTUAL-ACCEPTANCE.json'),
        'full_calls': native['full_calls_charged_returned_completed_passed'], 'prefix_prefills': native['prefix_prefills'],
        'suffix_evaluations': native['suffix_evaluations'], 'science_bytes': native['science_archive_bytes'],
        'create_to_release_observation_seconds': native['Create_to_release_seconds'],
        'not_a_per_call_throughput_estimate': True},
    'current_compute_calls': 0, 'new_model_calls': 0, 'new_environment_steps': 0,
    'GPU_authorized': False, 'execution_authorized': False, 'whole_goal_complete': False})
print(json.dumps({'snapshots': len(cases), 'CONF_inputs_joined': matched_inputs, 'CONF_calls_joined': matched_calls,
                  'counts': counts, 'fits': 0, 'cloud_compute': 0}))
