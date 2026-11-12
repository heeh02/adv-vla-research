"""Pure metadata identity, true policy-clock and payload-cost inventory logic."""
from collections import Counter, defaultdict
import csv
import hashlib
import io
import re

SUBSET_SHA = 'b58b6382617996b89b8e874ee1c9643ffcc1b73035a59089816dac976646d64e'
CASE_SET_SHA = '7fa65c3429d5ab414cf8f1b59085511756989d5255e4a26a52654bde7e235811'
INPUT_FILES = ('FINAL-READSET-v1.csv', 'CASE-ARTIFACT-INDEX-v1.csv', 'CASE-SUBSETS-v1.csv', 'case-overview-v1.csv')
FAMILY_COUNTS = {'Background Textures': 96, 'Camera Viewpoints': 102, 'Light Conditions': 96, 'Sensor Noise': 110}
OBS_FIELDS = ('robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos', 'physical_sim_state')
INPUT_FIELDS = (*OBS_FIELDS, 'noise')
ACTION_FIELDS = ('raw_actions', 'actions')
RGB_FIELDS = ('agentview_image', 'robot0_eye_in_hand_image')


class InventoryError(ValueError):
    pass


def need(ok, code):
    if not ok: raise InventoryError(code)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def indexed_csv(raw):
    rows = list(csv.DictReader(io.StringIO(raw.decode('utf-8'))))
    need(len(rows) == 536 and len({r['case_id'] for r in rows}) == 536, 'full536_unique_source_rows')
    return {r['case_id']: r for r in rows}


def build_plan(inputs):
    need(set(inputs) == set(INPUT_FILES), 'exact_four_input_tables')
    need(sha(inputs['CASE-SUBSETS-v1.csv']) == SUBSET_SHA, 'frozen_common28_subset_file')
    tables = {name: indexed_csv(raw) for name, raw in inputs.items()}
    need(all(set(t) == set(tables[INPUT_FILES[0]]) for t in tables.values()), 'same536_source_case_set')
    selected = sorted(cid for cid, r in tables['CASE-SUBSETS-v1.csv'].items() if r['common_28_labelled'] == 'True')
    need(len(selected) == 404 and sha(('\n'.join(selected) + '\n').encode()) == CASE_SET_SHA, 'exact_outcome_blind404_case_set')
    plan = []
    for cid in selected:
        r, a, s, o = [tables[name][cid] for name in INPUT_FILES]
        for key in ('case_id', 'row_id', 'noise_label', 'family', 'suite', 'task', 'Y_cell'):
            need(len({row[key] for row in (r, a, s, o)}) == 1, 'four_table_identity_' + key)
        for key in ('clean_id', 'cohort', 'run_id'):
            need(r[key] == a[key] == o[key], 'source_identity_' + key)
        need(r['source_package_sha256'] == a['source_package_sha256']
             and r['source_index_sha256'] == a['source_index_sha256'], 'source_package_index_identity')
        need(s['light_null_separate'] == 'False' and r['M_status'] == 'complete', 'labelled_population_only')
        scope = a['source_case_scope']
        need(scope.startswith('redacted-90e00ae5ddb1')
             and scope.endswith('/') and r['run_id'] + '/' in scope
             and r['source_index_uri'] == scope + 'RUNTIME_RESTORE_INDEX-v1.json'
             and r['source_manifest_uri'].startswith(scope), 'same_frozen_owned_case_scope')
        need(a['case_uri'] == o['source_case_uri'] == s['source_case_uri']
             and a['case_sha256'] == o['source_case_sha256'] == s['source_case_sha256'], 'existing_case_summary_identity')
        expected = {}
        for arm, cell in zip(('reference', 'treatment'), r['Y_cell']):
            need(cell in 'SF' and a[arm + '_success'] in ('0', '1'), 'frozen_Y_cell_boolean')
            success = a[arm + '_success'] == '1'
            need(success == (cell == 'S'), 'existing_Y_label_agrees')
            expected[arm] = {'status': o[arm + '_status'], 'success': success,
                             'policy_steps': int(o[arm + '_policy_steps']), 'queries': int(o[arm + '_queries'])}
        plan.append({**{k: r[k] for k in ('case_id', 'row_id', 'noise_label', 'family', 'suite', 'task', 'clean_id', 'cohort', 'run_id', 'source_package_sha256', 'Y_cell')},
                     'case_scope': scope, 'index_uri': r['source_index_uri'], 'index_sha256': r['source_index_sha256'],
                     'manifest_uri': r['source_manifest_uri'], 'manifest_sha256': r['source_manifest_sha256'],
                     'native_bytes': int(r['native_bytes']) if r['native_bytes'] else None,
                     'native_sha256': r['native_sha256'] or None,
                     'expected_Y': expected, 'source_case_summary': {'uri': a['case_uri'], 'bytes': int(a['case_bytes']), 'sha256': a['case_sha256']}})
    need(dict(Counter(r['family'] for r in plan)) == FAMILY_COUNTS and len({r['task'] for r in plan}) == 28,
         'fixed404_family_and_parent_task_support')
    counts = {arm: sum(r['expected_Y'][arm]['queries'] for r in plan) for arm in ('reference', 'treatment')}
    need(counts == {'reference': 14753, 'treatment': 16308}, 'known_query_count_metadata')
    return plan


def validate_trace(y, expected):
    """No success-rate computation: check saved trace against original per-case Y."""
    n, q = y.get('policy_steps'), y.get('queries')
    need(type(n) is int and n > 0 and type(q) is int and q > 0, 'positive_saved_step_query_count')
    for key in ('status', 'success', 'policy_steps', 'queries'):
        need(y.get(key) == expected[key], 'Y_overview_' + key + '_mismatch')
    need(type(y['success']) is bool and y['status'] in ('accepted_success', 'accepted_failure'), 'accepted_saved_Y_boolean')
    need((y['status'] == 'accepted_success') == y['success'], 'Y_status_success_consistency')
    trace = y.get('policy_trace')
    need(isinstance(trace, list) and len(trace) == n, 'trace_length_equals_actual_policy_steps')
    groups, controls = defaultdict(list), []
    previous_query = 0
    for step, event in enumerate(trace, 1):
        need(all(type(event.get(k)) is int for k in ('query', 'slot', 'policy_step')), 'integer_trace_coordinates')
        need(event['policy_step'] == step and 0 <= event['query'] < q and 0 <= event['slot'] < 5, 'continuous_real_policy_step')
        need(type(event.get('done')) is bool and type(event.get('official_success')) is bool, 'strict_trace_flags')
        need(event['query'] in (previous_query, previous_query + 1), 'ordered_query_trace')
        need(event['slot'] == len(groups[event['query']]), 'ordered_executed_slot')
        if step < n:
            need(not event['done'] and not event['official_success'], 'trace_continues_after_terminal')
        groups[event['query']].append(event)
        previous_query = event['query']
        controls.append({'query': event['query'], 'slot': event['slot'], 's_before': step - 1, 's_after': step,
                         'done': event['done'], 'official_success': event['official_success']})
    need(set(groups) == set(range(q)), 'all_saved_queries_have_executed_controls')
    queries = []
    for query in range(q):
        events = groups[query]
        need(1 <= len(events) <= 5 and (query == q - 1 or len(events) == 5), 'only_last_query_can_be_partial')
        before = events[0]['policy_step'] - 1
        queries.append({'query': query, 's_before': before, 'last_s_after': events[-1]['policy_step'],
                        'executed_slots': [e['slot'] for e in events], 'executed_count': len(events),
                        'agrees_with_5q': before == 5 * query})
    need(queries[0]['s_before'] == 0, 'first_query_before_step_zero')
    last = trace[-1]
    need(last['official_success'] == y['success'], 'last_success_equals_saved_Y')
    reason = 'official_success' if last['official_success'] else ('environment_done' if last['done'] else 'policy_budget_exhausted')
    need(y.get('terminal_reason') == reason, 'terminal_reason_matches_actual_trace')
    return {'queries': queries, 'controls': controls, 'terminal_policy_step': n,
            'query_policy_steps': [r['s_before'] for r in queries],
            'terminal_observation_physics_alignment': y.get('terminal_observation_physics_alignment'),
            'status': y['status'], 'success': y['success'], 'terminal_reason': y['terminal_reason']}


def expected_members(arm, query_ids):
    prefix = arm + '/Y/'
    return {prefix + 'Y-v1.json': 'Y_json', prefix + 'terminal.npz': 'terminal',
            **{prefix + f'query-{q:04d}-input.npz': 'query_input' for q in query_ids},
            **{prefix + f'query-{q:04d}-actions.npz': 'query_actions' for q in query_ids}}


def classify_member(name, arm):
    prefix = arm + '/Y/'
    if name == prefix + 'Y-v1.json': return 'Y_json'
    if name == prefix + 'terminal.npz': return 'terminal'
    match = re.fullmatch(re.escape(prefix) + r'query-(\d{4})-(input|actions)\.npz', name)
    return ('query_input' if match[2] == 'input' else 'query_actions') if match else None


def member_record(name, kind, mapped, native, expected=True):
    """Manifest field declarations only; NPZ payloads are deliberately NOT read."""
    record = {'member': name, 'kind': kind, 'expected_by_Y_trace': expected,
              'payload_read_and_verified': False, 'field_typed_SHA_verified_against_array': False}
    issues = []
    if name not in mapped:
        return {**record, 'availability': 'source_member_absent', 'bytes': None, 'sha256': None}, ['source_member_absent']
    item = mapped[name]; arc = item['row']
    record.update(bytes=arc['bytes'], sha256=arc['sha256'], header_offset=item['header_offset'],
                  payload_offset=item['payload_offset'], header_bytes=len(item['header']), availability='manifest_indexed')
    if kind == 'Y_json':
        record['field_identity_scope'] = 'JSON_no_native_field_schema_by_producer_design'
        return record, issues
    if name not in native:
        return {**record, 'availability': 'native_schema_absent'}, ['native_schema_absent']
    proof = native[name]
    if proof['bytes'] != arc['bytes'] or proof['sha256'] != arc['sha256']:
        return {**record, 'availability': 'archive_native_identity_mismatch'}, ['archive_native_identity_mismatch']
    selected = INPUT_FIELDS if kind == 'query_input' else (ACTION_FIELDS if kind == 'query_actions' else OBS_FIELDS)
    expected_role = {'query_input': 'actual_Y_wrapper_input', 'query_actions': 'canonical_native_sampler',
                     'terminal': 'actual_last_returned_wrapper_terminal'}[kind]
    if proof.get('role') != expected_role: issues.append('producer_role_mismatch')
    fields = proof.get('fields', {})
    missing = [k for k in selected if k not in fields]
    if missing: issues.append('field_absent')
    record.update(native_role=proof.get('role'), selected_fields={k: fields[k] for k in selected if k in fields},
                  missing_fields=missing, complete_field_names=sorted(fields),
                  selected_declared_stored_bytes=sum(fields[k]['bytes'] for k in selected if k in fields),
                  RGB_declared_stored_bytes=sum(fields[k]['bytes'] for k in RGB_FIELDS if k in fields),
                  total_declared_field_stored_bytes=sum(v['bytes'] for v in fields.values()),
                  field_identity_scope='native_manifest_declaration_only_pending_payload_read')
    shapes = {'robot0_eef_pos': [3], 'robot0_eef_quat': [4], 'robot0_gripper_qpos': [2],
              'noise': [1, 10, 32], 'raw_actions': [1, 10, 32], 'actions': [1, 10, 7]}
    for key, meta in record['selected_fields'].items():
        if key in shapes and meta.get('shape') != shapes[key]: issues.append('declared_field_shape_mismatch:' + key)
        if not isinstance(meta.get('sha256'), str) or not re.fullmatch('[0-9a-f]{64}', meta['sha256']): issues.append('invalid_field_SHA:' + key)
        if meta.get('finite') is not True: issues.append('declared_field_nonfinite:' + key)
    record['availability'] = 'manifest_indexed_with_field_issues' if issues else 'manifest_indexed_schema_declared'
    return record, issues


def costs(records):
    groups = {}
    for kind in ('Y_json', 'query_input', 'query_actions', 'terminal'):
        selected = [r for r in records if r['kind'] == kind]
        known = [r for r in selected if r.get('bytes') is not None]
        groups[kind] = {'member_count': len(selected), 'members_with_known_bytes': len(known),
                        'whole_member_payload_bytes': sum(r['bytes'] for r in known),
                        'max_member_bytes': max((r['bytes'] for r in known), default=0),
                        'selected_field_stored_bytes': sum(r.get('selected_declared_stored_bytes', 0) for r in known),
                        'RGB_stored_bytes': sum(r.get('RGB_declared_stored_bytes', 0) for r in known),
                        'missing_member_bytes_unknown': len(selected) - len(known)}
    return groups
