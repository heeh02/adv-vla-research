"""Stage B1 inventory only: metadata/Y GET + Range, zero NPZ/model/GPU reads."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import time
import traceback
import zipfile
from bos_stdlib import BOSWriter
from metadata_reader import MetadataReader
import inventory_math as M
import readset_contract as Q1

RUN = 'jinwen-adv-vla-plus-query-motion-inventory-20261010-v1'
OUTPUT = 'redacted-7450c3eb7535'


def safe_error(error):
    # Our custom exception carries only fixed codes. Foreign transport/parser
    # exception messages may contain URLs; never serialize their text.
    known = {'metadata_whole_SHA', 'actual_member_bytes_SHA_differ', 'actual_tar_header_bytes_differ',
             'bounded_metadata_response', 'selected_member_only', 'real_tar_header_and_member_SHA'}
    code = str(error) if isinstance(error, M.InventoryError) or str(error) in known else None
    category = ('SHA_mismatch' if code and ('SHA' in code or 'identity' in code or 'header_bytes_differ' in code)
                else 'reader_limit' if code in ('bounded_metadata_response', 'selected_member_only')
                else 'source_member_absent' if code == 'source_member_absent'
                else 'temporary_transport' if type(error).__name__ in ('TimeoutError', 'URLError', 'HTTPError', 'ConnectionError')
                else 'schema_or_trace_inconsistency')
    return {'error_type': type(error).__name__, 'code': code, 'category': category,
            'frames': [{'file': Path(f.filename).name, 'function': f.name, 'line': f.lineno}
                       for f in traceback.extract_tb(error.__traceback__)], 'raw_exception_saved': False}


def inventory_case(row, signer, deadline, cache_root, reader_class=MetadataReader):
    M.need(time.time() < deadline - 120, 'inventory_science_deadline')
    r = reader_class(signer, row, deadline - 120, cache_root)
    report = {'schema': 'plus-query-motion-case-inventory-v1',
              'case': row, 'members': [], 'arms': {}, 'issues': [],
              'NPZ_payload_reads': 0, 'NPZ_field_payload_verifications': 0,
              'source_index': r.index, 'metadata_cache_key': row['index_sha256']}
    for arm in ('reference', 'treatment'):
        y_name = arm + '/Y/Y-v1.json'
        try:
            M.need(y_name in r.mapped, 'source_member_absent')
            y_raw = r.member(y_name, limit=256 * 1024)
            y = json.loads(y_raw)
            trace = M.validate_trace(y, row['expected_Y'][arm])
            report['arms'][arm] = {'trace_status': 'verified_against_original_case_overview', **trace,
                                  'Y_json': {'member': y_name, 'bytes': len(y_raw), 'sha256': M.sha(y_raw)}}
            query_ids = [v['query'] for v in trace['queries']]
        except BaseException as error:
            report['arms'][arm] = {'trace_status': 'unavailable_or_inconsistent', 'failure': safe_error(error)}
            report['issues'].append({'arm': arm, 'stage': 'Y_trace', **safe_error(error)})
            # Still enumerate the true archive inventory; a bad Y never hides
            # available members. Expected names are clearly labelled fallback.
            query_ids = list(range(row['expected_Y'][arm]['queries']))
        expected = M.expected_members(arm, query_ids)
        actual = {name: M.classify_member(name, arm) for name in r.mapped if M.classify_member(name, arm) is not None}
        for name in sorted(set(expected) | set(actual)):
            kind = actual.get(name, expected.get(name))
            rec, issues = M.member_record(name, kind, r.mapped, r.native, name in expected)
            rec.update(arm=arm, expected_basis='actual_Y_trace' if report['arms'][arm]['trace_status'].startswith('verified') else 'frozen_overview_pending_trace')
            if name == y_name and report['arms'][arm].get('Y_json'):
                rec['payload_read_and_verified'] = True
            if name not in expected: issues.append('query_member_not_declared_by_Y')
            for issue in issues: report['issues'].append({'arm': arm, 'member': name, 'code': issue})
            report['members'].append(rec)
        report['arms'][arm]['manifest_query_input_count'] = sum(k == 'query_input' for k in actual.values())
        report['arms'][arm]['manifest_query_action_count'] = sum(k == 'query_actions' for k in actual.values())
    report['costs'] = M.costs(report['members'])
    report['metadata_members_actually_read'] = r.read_members
    report['range_requests'] = len(r.range.range_requests)
    report['metadata_transfer_payload_bytes'] = sum(len(raw) for raw in r.metadata_raw.values())
    report['status'] = 'complete_manifest_and_trace_inventory' if not report['issues'] else 'inventory_complete_with_explicit_issues'
    return report, r.metadata_raw


def save_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False); stream.write('\n')


def publish_case(writer, output, report, metadata):
    key = hashlib.sha256(report['case']['case_id'].encode()).hexdigest()[:24]
    folder = output / key; folder.mkdir(exist_ok=False)
    cache = folder / 'METADATA-CACHE-v1.zip'
    with zipfile.ZipFile(cache, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        for name, raw in sorted(metadata.items()): archive.writestr(name, raw)
    cache_ref = writer.put(cache, f'cases/{key}/METADATA-CACHE-v1.zip')
    report['cache'] = {'object': cache_ref, 'members': {name: {'bytes': len(raw), 'sha256': M.sha(raw)} for name, raw in metadata.items()},
                       'contains_arrays': False, 'original_index_manifest_native_Y_bytes_preserved': True}
    path = folder / 'B_MEMBER_READSET-v1.json.gz'
    with gzip.open(path, 'xt') as stream: json.dump(report, stream, sort_keys=True, allow_nan=False)
    ref = writer.put(path, f'cases/{key}/B_MEMBER_READSET-v1.json.gz')
    # Small host receipt; full member/field lists and true query/control clocks
    # stay cloud-side and are referenced by exact bytes/SHA.
    return {'case_id': report['case']['case_id'], 'task': report['case']['task'],
            'family': report['case']['family'], 'suite': report['case']['suite'],
            'status': report['status'], 'issue_count': len(report['issues']), 'costs': report['costs'],
            'readset': ref, 'metadata_cache': cache_ref,
            'source_index_sha256': report['case']['index_sha256'],
            'actual_metadata_payload_bytes': report['metadata_transfer_payload_bytes'],
            'query_counts': {arm: len(value.get('queries', [])) for arm, value in report['arms'].items()},
            'trace_verified': {arm: value['trace_status'].startswith('verified') for arm, value in report['arms'].items()},
            'terminal_policy_steps': {arm: value.get('terminal_policy_step') for arm, value in report['arms'].items()},
            'NPZ_payload_reads': 0}


def summarize(receipts, planned):
    totals = {}
    for kind in ('Y_json', 'query_input', 'query_actions', 'terminal'):
        fields = ('member_count', 'members_with_known_bytes', 'whole_member_payload_bytes', 'selected_field_stored_bytes',
                  'RGB_stored_bytes', 'missing_member_bytes_unknown')
        totals[kind] = {key: sum(r['costs'][kind][key] for r in receipts) for key in fields}
        totals[kind]['max_member_bytes'] = max((r['costs'][kind]['max_member_bytes'] for r in receipts), default=0)
    read = {r['case_id'] for r in receipts}
    return {'planned_case_count': len(planned), 'inventoried_case_count': len(receipts),
            'complete_trace_and_field_inventory_cases': sum(r['status'] == 'complete_manifest_and_trace_inventory' for r in receipts),
            'uninventoried_case_ids': [r['case_id'] for r in planned if r['case_id'] not in read],
            'whole_member_payload_costs': totals, 'costs_complete_for_all_planned_cases': len(receipts) == len(planned)
                and all(v['missing_member_bytes_unknown'] == 0 for v in totals.values()),
            'actual_metadata_payload_bytes': sum(r['actual_metadata_payload_bytes'] for r in receipts),
            'planned_input_query_count_from_previous_overview': 31061,
            'actual_verified_trace_queries': {arm: sum(r['query_counts'][arm] for r in receipts) for arm in ('reference', 'treatment')},
            'actual_NPZ_payload_reads': 0, 'success_rate_recomputed': False,
            'cost_interpretation': 'whole NPZ payload bytes are future transfer cost; selected field bytes are not network savings'}


def run_cloud(root, deadline):
    root = Path(root).resolve(strict=True)
    M.need(root == (Path('/tmp/Jinwen/adv-vla') / RUN).resolve(), 'exact_owned_CPU_root')
    meta = root / 'package/metadata'
    req = json.loads((meta / 'REQUEST-v1.json').read_bytes())
    M.need(req['run_id'] == RUN and req['output_prefix'] == OUTPUT and req['planned_case_count'] == 404
           and req['resources'] == {'CPU': 16, 'GPU': 0, 'memory_GiB': 64}
           and req['parallel_readers'] == 8 and req['max_runtime_seconds'] == 1800, 'frozen_inventory_request')
    inputs = {name: (meta / name).read_bytes() for name in M.INPUT_FILES}
    for name, raw in inputs.items():
        M.need({'bytes': len(raw), 'sha256': M.sha(raw)} == req['inputs'][name], 'frozen_source_table_identity')
    plan = M.build_plan(inputs)
    for row in plan:
        M.need(row['source_package_sha256'] == Q1.SOURCE_ROLES[row['cohort']][1], 'original_runtime_source_package')
    output = root / 'inventory'; output.mkdir(exist_ok=False)
    cache = root / 'metadata-cache'; cache.mkdir(exist_ok=False)
    writer = BOSWriter(OUTPUT, deadline - 30)
    receipts, failures = [], []
    def one(row):
        report, payload = inventory_case(row, writer, deadline, cache)
        return publish_case(writer, output, report, payload)
    try:
        # One actual source enters the original denominator; no separate probe.
        receipts.append(one(plan[0]))
    except BaseException as error:
        failures.append({'case_id': plan[0]['case_id'], 'stage': 'first_source_metadata_gate', **safe_error(error)})
    if receipts:
        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = {pool.submit(one, row): row for row in plan[1:]}
            for future in as_completed(futures):
                try: receipts.append(future.result())
                except BaseException as error:
                    failures.append({'case_id': futures[future]['case_id'], 'stage': 'case_inventory', **safe_error(error)})
    receipts.sort(key=lambda r: r['case_id'])
    result = {'schema': 'plus-query-motion-inventory-result-v1', 'run_id': RUN, **summarize(receipts, plan),
              'case_set_sha256': M.CASE_SET_SHA, 'inputs': req['inputs'], 'case_receipts': receipts, 'failures': failures,
              'GPU': 0, 'model_calls': 0, 'phase': 'B1_metadata_and_Y_trace_inventory_only',
              'position_quaternion_arrays_measured': False, 'motion_analysis_complete': False}
    result_path = output / 'RESULT-v1.json'; save_json(result_path, result)
    M.need(result_path.stat().st_size < 8 * 1024 * 1024, 'small_inventory_result')
    result_ref = writer.put(result_path, 'summary/RESULT-v1.json')
    delivery = {'schema': 'plus-query-motion-inventory-delivery-v1', 'run_id': RUN, 'planned_case_count': 404,
                'inventoried_case_count': len(receipts), 'result': result_ref, 'GPU': 0, 'model_calls': 0,
                'raw_array_to_Mac': False}
    path = output / 'DELIVERY-v1.json'; save_json(path, delivery)
    writer.put(path, 'summary/DELIVERY-v1.json')
    clean = len(receipts) == 404 and not failures and all(r['issue_count'] == 0 for r in receipts)
    return {'status': 'inventory_published' if clean else 'partial_or_issues_inventory_published',
            'result': result_ref, 'cases': len(receipts), 'failure_count': len(failures), 'GPU': 0}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--execute-cloud', action='store_true'); parser.add_argument('--root'); parser.add_argument('--deadline-epoch', type=float)
    args = parser.parse_args()
    if not args.execute_cloud:
        print(json.dumps({'status': 'dry_query_motion_inventory', 'cloud_calls': 0, 'credential_reads': 0, 'NPZ_payload_reads': 0})); return 0
    try:
        M.need(args.root and args.deadline_epoch, 'explicit_cloud_arguments')
        result = run_cloud(args.root, args.deadline_epoch); print(json.dumps(result), flush=True)
        return 0 if result['status'] == 'inventory_published' else 2
    except BaseException as error:
        print(json.dumps({'status': 'inventory_source_or_publication_failure', **safe_error(error)}), flush=True); return 2


if __name__ == '__main__': raise SystemExit(main())
