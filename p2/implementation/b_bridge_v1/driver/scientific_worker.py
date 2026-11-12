"""R-only native B bridge driver; cloud child has no transport or credentials.

This frozen stage runs exactly two original full continuations. Candidate stage
uses a separately frozen successor after these reference objects are archived.
No retry, warmup, synthetic production switch, or extra model call exists here.
"""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import math
import pickle
import random
import sys
import time

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent
ROOT = 'SDRU-CCONF-T06-V00700-I00-PC'
CALLS = (('REF-01', 'pulse', 'shared_pulse'), ('REF-02', 'persistent', 'shared_persistent'))
RAW_ROLES = ('ROOT_RUNTIME_BINDING', 'ROOT_INPUT_BINDING', 'ACTUAL_PROMPT_BINDING',
             'PAIRED_INPUT', 'QUERY_REACHED', 'CURRENT_GPU_INPUT_BINDING',
             'FROZEN_QUERY_EDITS', 'natural_T_common')
RAW_ARGS = ('root_runtime', 'root_input', 'prompt_binding', 'pair', 'clock', 'binding', 'edits', 'old_natural_T')
OLD_MANIFEST = {'bytes': 104372, 'sha256': '76ae8f0820370ebf0ae738177b1a1154d7a051d0e93896c0bda7b64a180c5cbb'}
PRECISION_MANIFEST_SHA = '192b4fa8f1cb8987eadab3a74e24ca5c3e1588546d5df93b10f2e23b996f112e'
PRIOR_REF = {'bytes': 16547, 'sha256': '25b1c8b47745460277c329aa909e0f3fe3c177cf9fa5e751dd84b8e1313d9b30'}
AUDIT_REF = {'bytes': 94694, 'sha256': 'fcf6c7835d01a4e4eaeb12b4b96938e49d70182dc457c348f6bdc5084ee469b1'}
MAX_FILE = 64 * 1024**2
MAX_TOTAL = 512 * 1024**2
MAX_RESULT = 2 * 1024**2
OUTPUT_NAMES = frozenset(['RESULT.json', 'INPUT-QUALIFICATION.json', 'MODEL-BINDING.json', 'PG-C.npz', 'PG-T.npz'] +
    [f'{kind}-{call}.{ext}' for call, _, _ in CALLS for kind, ext in [('CALL', 'npz'), ('CHARGE', 'json'), ('CHECK', 'json')]] +
    [f'UNEXPECTED-PG-{side}-{call}.npz' for call, _, _ in CALLS for side in ('C', 'T')])


class WorkerFault(ValueError):
    pass


def need(ok, code):
    if not ok:
        raise WorkerFault(code)


def digest(raw):
    return {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


def json_bytes(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()


def bounded_read(path, ref, cap):
    p = Path(path)
    need(p.is_file() and not p.is_symlink() and p.stat().st_size <= cap, 'bounded_regular_input')
    raw = p.read_bytes()
    need(digest(raw) == {k: ref[k] for k in ('bytes', 'sha256')}, 'input_wholefile_SHA')
    return raw


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, Path(path))
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


def verify_source(expected_sha):
    need(isinstance(expected_sha, str) and len(expected_sha) == 64, 'source_SHA_argument')
    path = HERE / 'REFERENCE-SOURCE-MANIFEST.json'
    raw = path.read_bytes()
    need(digest(raw)['sha256'] == expected_sha, 'externally_bound_source_manifest')
    doc = json.loads(raw)
    need(doc['schema'] == 'B-BRIDGE-reference-science-source-v1', 'reference_source_schema')
    names = [r['path'] for r in doc['files']]
    need(len(names) == len(set(names)) and {'driver/scientific_worker.py', 'driver/native_codec.py',
         'native/frozen/candidate_query.py', 'native/frozen/input_contract.py', 'native/targets.py',
         'driver/frozen_precision/adapter.py', 'driver/ORIGINAL-PRECISION-RESULT.json',
         'driver/COMMON-PROBE-AUDIT-RESULT.json'} <= set(names), 'unique_required_science_closure')
    for ref in doc['files']:
        p = Path(ref['path'])
        need(not p.is_absolute() and '..' not in p.parts and (SOURCE / p).resolve().is_relative_to(SOURCE), 'source_relative_path')
        bounded_read(SOURCE / p, ref, 2 * 1024**2)
    return digest(raw)


def host(x):
    import numpy as np
    if isinstance(x, dict):
        return {k: host(v) for k, v in x.items()}
    if isinstance(x, (tuple, list)):
        return type(x)(host(v) for v in x)
    return np.asarray(x) if hasattr(x, 'dtype') else x


def exact(a, b):
    import numpy as np
    if isinstance(a, dict):
        return isinstance(b, dict) and set(a) == set(b) and all(exact(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(exact(x, y) for x, y in zip(a, b))
    a, b = np.asarray(a), np.asarray(b)
    return a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()


def array_identity(x):
    import numpy as np
    a = np.asarray(x)
    return {'shape': list(a.shape), 'dtype': a.dtype.str, 'logical_dtype': a.dtype.name,
            'bytes': a.nbytes, 'sha256': hashlib.sha256(a.tobytes()).hexdigest()}


def rng_identity():
    import numpy as np
    return {'numpy': hashlib.sha256(pickle.dumps(np.random.get_state(), protocol=4)).hexdigest(),
            'python_diagnostic_only': hashlib.sha256(pickle.dumps(random.getstate(), protocol=4)).hexdigest()}


def pulse_pair(pulse, persistent, fields):
    """Original PVP pair gate, without equating intentionally edited j6 V/W."""
    a, b = pulse['metadata'], persistent['metadata']
    for name, count in (('x', 7), ('velocity_native', 6), ('time', 7), ('time_weak_type', 7), ('dt', 10)):
        need(exact(a['flow'][name][:count], b['flow'][name][:count]), 'same_j5_x6_' + name)
    for name in fields:
        need(exact(a['reads'][name][0], b['reads'][name][0]), 'same_j5_three_layers_' + name)
    for name in ('query_post_rope_scaled', 'probabilities_native', 'receiver_residual_input',
                 'suffix_key_post_rope', 'suffix_value', 'attention_gate'):
        need(exact(a['reads'][name][1, :2], b['reads'][name][1, :2]), 'same_j6_L1_L9_receiver_' + name)
    for name in ('mask', 'suffix_positions', 'query_defined', 'adarms_condition'):
        need(exact(a['reads'][name][:2], b['reads'][name][:2]), 'same_j5_j6_' + name)
    need(exact(a['Vmod_argument'], b['Vmod_argument']) and exact(a['pg'], b['pg']), 'same_actual_E_and_complete_fresh_PG')
    return True


def compare_old_call(record, actual, I, Q, source_identity):
    """All saved old payload leaves plus 30 used-K/V typed identities.

    Historical PG payload and used-prefix arrays were NOT stored in old CALL.
    They are not fabricated here; fresh PG has a separate new complete archive.
    """
    old = I.value(record.arrays)
    m = actual['metadata']
    new = {k: v for k, v in m.items() if k not in ('pg', 'reads', 'Vmod_argument')}
    new['reads'] = {k: v for k, v in m['reads'].items() if k not in ('prefix_key_used', 'prefix_value_used')}
    new.update(raw32=actual['raw32'], adapter7=actual['adapter7'])
    need(exact(old, new), 'all_saved_old_CALL_payload_leaves_exact')
    need(record.metadata['input_identity'] == source_identity, 'old_original_encoded_input_identity')
    need(record.metadata['Vmod_actual_identity'] == array_identity(m['Vmod_argument']), 'old_actual_Vmod_native_identity')
    used = record.metadata['actual_prefix_used']
    expected = {(j, li, name) for j in Q.CALLS for li in Q.LAYERS for name in ('prefix_key_used', 'prefix_value_used')}
    keys = [(row['flow_call'], row['layer_index'], row['field']) for row in used]
    need(len(keys) == len(set(keys)) == 30 and set(keys) == expected, 'old_actual_prefix_identity_complete_roster')
    for row in used:
        ji, si = Q.CALLS.index(row['flow_call']), Q.LAYERS.index(row['layer_index'])
        need(row['actual_identity'] == array_identity(m['reads'][row['field']][ji, si]), 'old_actual_used_prefix_typed_identity')
    return {'all_saved_old_CALL_payload_leaves_exact': True, 'old_used_prefix_native_identities_checked': 30,
            'historical_full_PG_payload_reloaded': False, 'full_historical_PG_value_parity_claim': False}


class OutputStore:
    def __init__(self, output, codec):
        self.path, self.codec = Path(output), codec
        need(not self.path.exists() or (self.path.is_dir() and not any(self.path.iterdir())), 'exclusive_empty_output_directory')
        self.path.mkdir(parents=True, exist_ok=True)
        need(not self.path.is_symlink(), 'owned_output_directory')
        self.refs, self.pg = {}, None

    def _record(self, name, ref):
        self.refs[name] = {'path': name, **ref}
        need(ref['bytes'] <= MAX_FILE, 'per_file_64MiB_cap')
        need(sum(p.stat().st_size for p in self.path.iterdir() if p.is_file()) <= MAX_TOTAL, 'reference_output_512MiB_cap')
        return self.refs[name]

    def json(self, name, value):
        need(name in OUTPUT_NAMES, 'declared_output_name')
        raw = json_bytes(value)
        need(len(raw) <= MAX_RESULT, 'small_json_2MiB_cap')
        with (self.path / name).open('xb') as stream:
            stream.write(raw)
        return self._record(name, digest(raw))

    def tree(self, name, value):
        need(name in OUTPUT_NAMES, 'declared_output_name')
        return self._record(name, self.codec.save_tree(self.path / name, value, allow_nonfinite=True))

    def actual(self, call_id, actual):
        """Persist actual main output and all changed PG BEFORE qualification."""
        m = actual['metadata']
        main = {**actual, 'metadata': {k: v for k, v in m.items() if k != 'pg'}}
        self.tree('CALL-' + call_id + '.npz', main)
        need('pg' in m and set(m['pg']) == {'C', 'T'}, 'actual_PG_roster')
        equal = self.pg is None or exact(self.pg, m['pg'])
        if self.pg is None:
            for side in ('C', 'T'):
                self.tree('PG-' + side + '.npz', m['pg'][side])
            self.pg = m['pg']
        elif not equal:
            for side in ('C', 'T'):
                self.tree(f'UNEXPECTED-PG-{side}-{call_id}.npz', m['pg'][side])
        return equal


def prepare_inputs(context, check):
    """Read the exact eight raw records and six frozen precision objects once."""
    A = module('_b_bridge_precision_adapter', HERE / 'frozen_precision/adapter.py')
    A.verify_sources(PRECISION_MANIFEST_SHA)
    N, M = A.load_helpers()
    plan = A.load_inputs()
    refs = {r['role']: r for r in plan['input_objects']}
    staged = context['precision_inputs']
    need(set(staged) == set(refs), 'exact_six_precision_roles')
    for role, ref in refs.items():
        need(staged[role]['ref'] == ref, 'original_precision_full_reference_identity')
    source = json.loads(A.read_small(staged['source_evidence']['path'], refs['source_evidence'], 65536))
    query = A.select_source_query(source, plan)
    planned = N.load(staged['planned']['path'], refs['planned'])
    entries = A.planned_entries(planned, query)
    check()
    inp = N.load(staged['input']['path'], refs['input'])
    calls = [N.load(staged[role]['path'], refs[role]) for role in A.ROLES]
    prior = json.loads(bounded_read(HERE / 'ORIGINAL-PRECISION-RESULT.json', PRIOR_REF, MAX_RESULT))
    computed = A.analyze_records(inp, calls, entries, query, N, M, check=check)
    need(all(prior.get(k) == v for k, v in computed.items()), 'original_precision_result_exact_reproduced')
    need(computed['counts']['A_target_nonrepresentable'] == 623 and
         computed['counts']['reference_delta_nonzero'] == 10091, 'fixed_623_10091_reproduced')
    I = module('_b_bridge_original_input_contract', SOURCE / 'native/frozen/input_contract.py')
    need(set(context['raw_inputs']) == set(RAW_ROLES), 'exact_eight_raw_roles')
    records = {}
    for role in RAW_ROLES:
        check()
        entry = context['raw_inputs'][role]
        records[role] = N.load(entry['path'], entry['ref'])
    saved = I.from_records(**{arg: records[role] for arg, role in zip(RAW_ARGS, RAW_ROLES)}, synthetic=False)
    need(saved.key == (ROOT, 11) and saved.sources == inp.metadata['sources'], 'same_fixed_raw_and_precision_input_sources')
    # This validates the frozen c geometry using old W only; it is not a setter
    # qualification, candidate model execution, or automatic scientific GO.
    T = module('_b_bridge_targets_qualification', SOURCE / 'native/targets.py')
    old_w = [r.field('reads', 'attention_output_post_WO_pre_gate').storage[1, 1, 0] for r in calls]
    audit = bounded_read(HERE / 'COMMON-PROBE-AUDIT-RESULT.json', AUDIT_REF, MAX_RESULT)
    _, common_receipt = T.qualify_and_build_targets(*old_w, old_w[1], old_w[2], old_w[1], audit, check=check)
    proof = {'schema': 'B-BRIDGE-fixed-input-qualification-v1', 'synthetic': False, 'input_qualified': True,
             'root_id': ROOT, 'query_index': 11, 'raw_sources': saved.sources, 'precision_inputs': plan['input_objects'],
             'prior_precision_ref': PRIOR_REF, 'common_audit_ref': AUDIT_REF,
             'original_precision_result_exact_reproduced': True, 'precision_counts': computed['counts'],
             'fixed_common_geometry_reproduced': True, 'common_geometry_receipt': common_receipt,
             'geometry_receipt_scope': 'old_archived_W_only_not_fresh_model_or_native_setter',
             'historical_full_PG_payload_reloaded': False, 'new_model_forward_calls': 0}
    return I, saved, dict(zip(A.ROLES, calls)), proof


def run_reference(*, campaign, saved, old_calls, I, Q, runtime, guard, store, result, check):
    """Dependency-injected orchestration for synthetic tests; no extra forwards."""
    result['step'] = 'encode_original_raw_inputs'
    branches = I.encode_loaded(campaign, saved)
    original_identity = I.identity((branches, saved.noise, saved.modified_v))
    snapshots = {}
    for call_id, schedule, role in CALLS:
        result['active_call'] = call_id
        result['step'] = 'pre_call_model_and_source_guard'
        check()
        guard(campaign)
        before = rng_identity()
        need(I.identity((branches, saved.noise, saved.modified_v)) == original_identity, 'input_unchanged_before_call')
        charge = {'call_id': call_id, 'schedule': schedule, 'role': role, 'root_id': ROOT, 'query_index': 11,
                  'kind': 'full_continuation', 'prefix_prefills_reserved': 2, 'suffix_evaluations_reserved': 10,
                  'charge_before_invoke': True, 'charged_epoch': time.time()}
        store.json('CHARGE-' + call_id + '.json', charge)
        result['charged_entries'].append(call_id)
        result['prefix_prefills_reserved'] += 2
        result['suffix_evaluations_reserved'] += 10
        result['step'] = 'model_call'
        raw, metadata = runtime['common'](branches['T'], branches['C'], saved.noise, saved.modified_v, schedule)
        runtime['wait']((raw, metadata))
        result['returned_entries'].append(call_id)
        # Decoding is the original non-neural normalization adapter. It is only
        # historical full-continuation qualification, never a single-v action.
        result['step'] = 'host_and_original_adapter'
        actual = {'raw32': host(raw), 'metadata': host(metadata)}
        try:
            adapter = runtime['decode'](raw, branches['T'])
            runtime['wait'](adapter)
            actual['adapter7'] = host(adapter)
        except BaseException:
            store.actual(call_id, actual)
            raise
        result['step'] = 'persist_actual_before_gates'
        pg_equal = store.actual(call_id, actual)
        result['completed_entries'].append(call_id)
        result['step'] = 'qualify_saved_actual'
        check()
        Q.check_result(actual['metadata'])
        I.current_prefix(saved, actual['metadata'])
        need(pg_equal, 'complete_fresh_PG_same_across_reference_calls')
        need(exact(actual['metadata']['Vmod_argument'], saved.modified_v), 'actual_original_Vmod_argument')
        need(exact(actual['metadata']['flow']['x'][0], saved.noise), 'actual_original_noise')
        own = Q.selectors(schedule)
        need(all(exact(actual['metadata']['selectors'][name], v) for name, v in
                 zip(('kv_mask', 'layer_mask', 'flow_mask', 'use_vmod'), own)), 'actual_original_schedule')
        comparison = compare_old_call(old_calls[role], actual, I, Q, original_identity)
        need(I.identity((branches, saved.noise, saved.modified_v)) == original_identity, 'input_unchanged_after_call')
        after = rng_identity()
        need(before['numpy'] == after['numpy'], 'numpy_RNG_unchanged')
        snapshots[role] = actual
        pair_ok = None
        if role == 'shared_persistent':
            pair_ok = pulse_pair(snapshots['shared_pulse'], actual, Q.FIELDS)
        guard(campaign)
        check()
        store.json('CHECK-' + call_id + '.json', {'call_id': call_id, 'passed': True, **comparison,
                   'current_prefix_L9_KT_VT_actual_E_exact': True, 'encoded_input_identity': original_identity,
                   'complete_fresh_PG_equal': True, 'pulse_pair_checked': pair_ok,
                   'rng_before': before, 'rng_after': after})
        result['passed_entries'].append(call_id)
    result.update(analysis_complete=True, scientific_complete=True, qualification_complete=True,
                  status='REFERENCE_QUALIFIED', step='reference_complete', active_call=None,
                  historical_full_PG_payload_reloaded=False, full_historical_PG_value_parity_claim=False,
                  fresh_reference_PG_complete_saved=True, same_W_setter_qualified=False,
                  B_bridge_scientific_effect_evaluated=False, A_model_calls=0)
    return snapshots


def validate_context(context):
    need(sys.platform.startswith('linux'), 'actual_model_and_arrays_cloud_Linux_only')
    need(context['schema'] == 'B-bridge-exact-cloud-stage-context-v1' and context['phase'] == 'reference', 'R_only_exact_restore_context')
    need(type(context['run_id']) is str and 0 < len(context['run_id']) < 256, 'bounded_run_identity')
    need(context.get('reference_dependencies', []) == [], 'R_has_no_reference_dependencies')
    need(Path(context['source_root']).resolve() == SOURCE and Path(context['new_package_root']).resolve() == SOURCE,
         'executed_source_root_is_context_source')
    need(Path(context['package_root']).resolve() == Path(context['old_package_root']).resolve(), 'original_reference_package')
    need(Path(context['repo_root']).resolve() == (Path(context['old_package_root']) / 'repo').resolve(), 'original_reference_repo')
    need(math.isfinite(context['science_deadline_epoch']) and time.time() < context['science_deadline_epoch'], 'future_absolute_science_deadline')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--context', type=Path, required=True)
    parser.add_argument('--source-manifest-sha256', required=True)
    args = parser.parse_args()
    need(args.context.is_file() and not args.context.is_symlink() and args.context.stat().st_size <= MAX_RESULT, 'small_immutable_context')
    context = json.loads(args.context.read_bytes())
    validate_context(context)
    source_ref = verify_source(args.source_manifest_sha256)
    codec = module('_b_bridge_native_codec', HERE / 'native_codec.py')
    store = OutputStore(context['output_dir'], codec)
    result = {'schema': 'B-BRIDGE-reference-result-v1', 'run_id': context['run_id'], 'phase': 'reference',
              'synthetic': False, 'root_id': ROOT, 'query_index': 11, 'flow_index': 6, 'layer_index': 8,
              'source_manifest': source_ref, 'context_identity': digest(args.context.read_bytes()),
              'status': 'FAILED_PRESERVED', 'analysis_complete': False, 'scientific_complete': False,
              'qualification_complete': False, 'charged_entries': [], 'returned_entries': [],
              'completed_entries': [], 'passed_entries': [], 'prefix_prefills_reserved': 0,
              'suffix_evaluations_reserved': 0, 'model_call_caps': {'full_continuations': 2, 'single_forwards': 0,
              'prefix_prefills': 4, 'suffix_evaluations': 20}, 'new_environment_steps': 0,
              'training_calls': 0, 'automatic_retry': False, 'same_W_setter_qualified': False,
              'A_model_calls': 0, 'scientific_output_budget_bytes': MAX_TOTAL,
              'historical_full_PG_payload_reloaded': False, 'full_historical_PG_value_parity_claim': False}
    def check():
        need(time.time() < context['science_deadline_epoch'], 'absolute_science_deadline')
    exit_code = 2
    try:
        result['step'] = 'verify_old_source_and_inputs'
        check()
        old_package = Path(context['old_package_root'])
        bounded_read(old_package / 'metadata/SOURCE_MANIFEST-v1.json', OLD_MANIFEST, MAX_RESULT)
        I, saved, old_calls, proof = prepare_inputs(context, check)
        store.json('INPUT-QUALIFICATION.json', proof)
        result['step'] = 'load_original_campaign'
        sys.path.insert(0, str(Path(context['repo_root']) / 'src'))
        from openpi.eval.campaign_policy import CampaignPolicy
        campaign = CampaignPolicy(Path(context['checkpoint_root']))
        result['step'] = 'bind_original_model_source_and_adapter'
        guard_module = module('_b_bridge_actual_original_model_guard', old_package / 'study/continuous/cloud-v1/model_guard.py')
        guard = guard_module.bind(campaign, package_root=old_package, checkpoint_root=context['checkpoint_root'],
            restore_receipt_path=context['model_restore_receipt_path'], request={'model_plan': {'path': context['model_plan_relative']}})
        model_summary = guard(campaign)
        store.json('MODEL-BINDING.json', model_summary)
        result['model_binding'] = {k: model_summary[k] for k in ('model_bound', 'adapter_bound',
            'loaded_parameter_tree_sha256', 'normalization_sha256')}
        Q = module('_b_bridge_original_query', SOURCE / 'native/frozen/candidate_query.py')
        runtime = Q.production_runtime(campaign)
        run_reference(campaign=campaign, saved=saved, old_calls=old_calls, I=I, Q=Q,
                      runtime=runtime, guard=guard, store=store, result=result, check=check)
        exit_code = 0
    except BaseException as error:
        result.update(status='FAILED_PRESERVED', analysis_complete=False, scientific_complete=False,
                      qualification_complete=False, error_type=type(error).__name__,
                      error_code=str(error) if isinstance(error, WorkerFault) else 'external_exception_redacted')
    result['unexecuted_entries'] = [call for call, _, _ in CALLS if call not in result['charged_entries']]
    result['charged_but_completion_unproven'] = [call for call in result['charged_entries'] if call not in result['completed_entries']]
    result['outputs'] = list(store.refs.values())
    result['observed_files'] = [p.name for p in sorted(store.path.iterdir()) if p.is_file()]
    result['finished_epoch'] = time.time()
    store.json('RESULT.json', result)
    print(json.dumps({'status': result['status'], 'result': store.refs['RESULT.json'], 'exit_code': exit_code}, sort_keys=True))
    return exit_code


if __name__ == '__main__':
    raise SystemExit(main())
