"""C-only B bridge. Requires root-accepted exact R artifacts; no transport IO."""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import math
import shutil
import sys
import time

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent
spec = importlib.util.spec_from_file_location('_b_bridge_frozen_reference_driver', HERE / 'scientific_worker.py')
R = importlib.util.module_from_spec(spec); sys.modules[spec.name] = R; spec.loader.exec_module(R)
need = R.need
REFERENCE_SOURCE_SHA = '186c664f0b82e980446485c6800b8f06af6d9afc21e16f3dc95d6bfea55a494d'
NEW_GEMMA_SHA = '85ed2c5e0c07f3ccedbc5c5639c106c4fb31b2ce49df129bb5dd8e6bf3e67f74'
OLD_GEMMA_SHA = 'dfec7510ee7341dc90ef0c58111642bf102fdd24f70129890ecc98b210b61efe'
SINGLES = (
    ('S01', 'single', 'none', False, 'resume_diagonal'),
    ('S02', 'single', 'none', True, 'own_target'),
    ('S03', 'independent_single', 'none', False, 'independent_none'),
    ('S04', 'independent_single', 'natural_d', True, 'independent_d'),
    ('S05', 'independent_single', 'common_c', True, 'independent_c'),
    ('S06', 'single', 'natural_d', True, 'main_d'),
    ('S07', 'single', 'common_c', True, 'main_c'),
    ('S08', 'single', 'none', False, 'delayed_none'),
    ('S09', 'single', 'natural_d', True, 'delayed_d'),
    ('S10', 'single', 'common_c', True, 'delayed_c'),
)
CALL_IDS = ('CAN-01',) + tuple(x[0] for x in SINGLES)
REFERENCE_NAMES = {'reference_result': 'RESULT.json', 'reference_pulse': 'CALL-REF-01.npz',
    'reference_persistent': 'CALL-REF-02.npz', 'reference_PG_C': 'PG-C.npz', 'reference_PG_T': 'PG-T.npz'}
MAX_TOTAL = 1536 * 1024**2
OUTPUT_NAMES = frozenset(['RESULT.json', 'INPUT-QUALIFICATION.json', 'MODEL-BINDING.json',
    'SOURCE-BINDING.json', 'REFERENCE-ADMISSION.json', 'TARGET-QUALIFICATION.json',
    'VELOCITY-READOUT.json', 'QUANTIZATION-READOUT.json', 'ROUNDING-DIAGNOSTIC.json', 'PG-C.npz', 'PG-T.npz'] +
    [f'{kind}-{call}.{ext}' for call in CALL_IDS for kind, ext in [('CALL', 'npz'), ('CHARGE', 'json'), ('CHECK', 'json')]] +
    [f'UNEXPECTED-PG-{side}-{call}.npz' for call in CALL_IDS for side in ('C', 'T')])


def verify_source(expected_sha):
    R.verify_source(REFERENCE_SOURCE_SHA)
    raw = (HERE / 'CANDIDATE-SOURCE-MANIFEST.json').read_bytes()
    need(R.digest(raw)['sha256'] == expected_sha, 'externally_bound_candidate_source_manifest')
    doc = json.loads(raw)
    need(doc['schema'] == 'B-BRIDGE-candidate-science-source-v1' and doc['reference_source_sha256'] == REFERENCE_SOURCE_SHA,
         'candidate_source_schema_and_frozen_R')
    paths = [r['path'] for r in doc['files']]
    need(len(paths) == len(set(paths)) and {'driver/candidate_worker.py', 'native/query.py', 'native/overlay/gemma.py',
        'analysis/acceptance.py', 'analysis/quantization.py'} <= set(paths), 'candidate_unique_runtime_roster')
    for ref in doc['files']:
        path = Path(ref['path'])
        need(not path.is_absolute() and '..' not in path.parts and (SOURCE / path).resolve().is_relative_to(SOURCE),
             'candidate_source_relative_path')
        R.bounded_read(SOURCE / path, ref, 2 * 1024**2)
    return R.digest(raw)


class OutputStore(R.OutputStore):
    def _record(self, name, ref):
        self.refs[name] = {'path': name, **ref}
        need(ref['bytes'] <= R.MAX_FILE, 'candidate_per_file_64MiB_cap')
        need(sum(p.stat().st_size for p in self.path.iterdir() if p.is_file()) <= MAX_TOTAL, 'candidate_1536MiB_cap')
        return self.refs[name]

    def json(self, name, value):
        need(name in OUTPUT_NAMES, 'candidate_declared_output_name')
        raw = R.json_bytes(value)
        need(len(raw) <= R.MAX_RESULT, 'candidate_JSON_2MiB_cap')
        with (self.path / name).open('xb') as f:
            f.write(raw)
        return self._record(name, R.digest(raw))

    def tree(self, name, value):
        need(name in OUTPUT_NAMES, 'candidate_declared_output_name')
        return self._record(name, self.codec.save_tree(self.path / name, value, allow_nonfinite=True))


def load_reference(context, codec):
    """Full local native arrays come only from root-frozen complete R archives."""
    rows = context['reference_dependencies']
    need(len(rows) == 5 and {x['role'] for x in rows} == set(REFERENCE_NAMES), 'exact_five_reference_dependencies')
    by = {r['role']: r for r in rows}
    for role, name in REFERENCE_NAMES.items():
        ref = by[role]['ref']
        need(ref['path'] == name and type(ref['uri']) is str and ref['uri'].startswith('bos://'), 'root_frozen_reference_object_role')
    rr = by['reference_result']
    result = json.loads(R.bounded_read(rr['path'], rr['ref'], R.MAX_RESULT))
    adm = context['reference_admission']
    need(adm['archive_complete'] is True and adm['readback_complete'] is True and
         adm['reference_run_id'] == result['run_id'] and
         adm['reference_result_sha256'] == rr['ref']['sha256'] and
         adm['source_manifest_sha256'] == REFERENCE_SOURCE_SHA, 'root_R_archive_readback_admission_bound')
    need(result['schema'] == 'B-BRIDGE-reference-result-v1' and result['status'] == 'REFERENCE_QUALIFIED' and
         result['synthetic'] is False and result['phase'] == 'reference' and
         result['root_id'] == R.ROOT and result['query_index'] == 11 and
         result['source_manifest']['sha256'] == REFERENCE_SOURCE_SHA and
         all(result[k] is True for k in ('analysis_complete', 'scientific_complete', 'qualification_complete')),
         'actual_reference_completed_and_source_bound')
    need(all(result[k] == ['REF-01', 'REF-02'] for k in ('charged_entries', 'returned_entries', 'completed_entries', 'passed_entries'))
         and result['prefix_prefills_reserved'] == 4 and result['suffix_evaluations_reserved'] == 20 and
         result['unexecuted_entries'] == [] and result['charged_but_completion_unproven'] == [], 'actual_reference_exact_call_budget')
    published = {row['path']: row for row in result['outputs']}
    need(len(published) == len(result['outputs']) == 10, 'successful_reference_ten_payload_roster')
    for role, name in REFERENCE_NAMES.items():
        if role == 'reference_result':
            continue
        need(name in published and all(published[name][k] == by[role]['ref'][k] for k in ('bytes', 'sha256')),
             'reference_dependency_in_actual_complete_result')
    pg = {side: codec.load_tree(by['reference_PG_' + side]['path'], by['reference_PG_' + side]['ref']) for side in ('C', 'T')}
    calls = {}
    for role, key in (('reference_pulse', 'pulse'), ('reference_persistent', 'persistent')):
        value = codec.load_tree(by[role]['path'], by[role]['ref'])
        need(set(value) == {'raw32', 'adapter7', 'metadata'} and 'pg' not in value['metadata'], 'full_reference_native_call_container')
        value['metadata']['pg'] = pg
        calls[key] = value
    return calls, result, {'schema': 'B-BRIDGE-reference-dependency-proof-v1', 'admission': adm,
                           'references': [{**row['ref'], 'role': row['role']} for row in rows],
                           'native_reloaded': True, 'historical_full_PG_payload_reloaded': False}


def isolate_candidate_package(context, source_ref):
    """Copy only the bounded original source package and change one GM module."""
    old = Path(context['old_package_root'])
    original = R.bounded_read(old / 'metadata/SOURCE_MANIFEST-v1.json', R.OLD_MANIFEST, R.MAX_RESULT)
    gm_path = 'repo/src/openpi/models/gemma.py'
    need(R.digest((old / gm_path).read_bytes())['sha256'] == OLD_GEMMA_SHA, 'original_Gstar_Gemma')
    files = [p for p in old.rglob('*') if p.is_file()]
    need(not any(p.is_symlink() for p in old.rglob('*')) and len(files) <= 500 and
         sum(p.stat().st_size for p in files) <= 7 * 1024**2, 'bounded_original_source_only_copy')
    dest = Path(context['output_dir']).parent / 'candidate-source-package'
    need(not dest.exists() and not dest.is_symlink(), 'new_exclusive_candidate_source_package')
    shutil.copytree(old, dest, copy_function=shutil.copyfile)
    changed = dest / gm_path
    raw = (SOURCE / 'native/overlay/gemma.py').read_bytes()
    need(R.digest(raw)['sha256'] == NEW_GEMMA_SHA, 'frozen_candidate_Gemma')
    changed.chmod(0o600); changed.write_bytes(raw); changed.chmod(0o440)
    manifest = json.loads(original)
    rows = [row for row in manifest['files'] if row['path'] == gm_path]
    need(len(rows) == 1 and rows[0]['sha256'] == OLD_GEMMA_SHA, 'single_original_Gemma_manifest_row')
    rows[0].update(R.digest(raw))
    rows[0]['origin'] = 'B-bridge-v1/native/overlay/gemma.py'
    manifest['B_bridge_candidate_provenance'] = {'phase': 'candidate', 'run_id': context['run_id'],
        'science_source_manifest': source_ref, 'original_manifest': R.OLD_MANIFEST,
        'only_repo_source_changed': gm_path, 'original_Gemma_sha256': OLD_GEMMA_SHA,
        'candidate_Gemma_sha256': NEW_GEMMA_SHA, 'old_request_is_historical_not_execution_authority': True}
    mpath = dest / 'metadata/SOURCE_MANIFEST-v1.json'
    mpath.chmod(0o600); mpath.write_bytes(R.json_bytes(manifest)); mpath.chmod(0o440)
    need((old / 'metadata/SOURCE_MANIFEST-v1.json').read_bytes() == original and
         R.digest((old / gm_path).read_bytes())['sha256'] == OLD_GEMMA_SHA, 'original_package_unchanged')
    # Check every copied original source byte except the two declared edits.
    for old_file in files:
        rel = str(old_file.relative_to(old))
        if rel not in (gm_path, 'metadata/SOURCE_MANIFEST-v1.json'):
            need((dest / rel).read_bytes() == old_file.read_bytes(), 'candidate_copy_original_other_bytes')
    return dest, {'schema': 'B-BRIDGE-isolated-candidate-source-v1',
        **manifest['B_bridge_candidate_provenance'], 'actual_candidate_manifest': R.digest(mpath.read_bytes()),
        'original_package_unchanged': True, 'model_files_copied': False}


def full_parity(reference, candidate, Q):
    m = candidate['metadata']
    Q.check_result(m)
    need(set(m['bridge_gates']) == {'full_W_selector_inactive', 'full_schedule_is_B_pulse', 'all_before_equals_used'} and
         all(bool(v) for v in m['bridge_gates'].values()), 'full_inactive_bridge_gates')
    need(R.exact(m['reads']['bridge_W_before'], m['reads']['bridge_W_used']) and
         R.exact(m['reads']['bridge_W_before'], m['reads']['attention_output_post_WO_pre_gate']), 'full_before_used_original_W_equal')
    projected = {k: v for k, v in m.items() if k != 'bridge_gates'}
    projected['reads'] = {k: v for k, v in m['reads'].items() if not k.startswith('bridge_')}
    need(R.exact(reference, {**candidate, 'metadata': projected}), 'candidate_full_inactive_complete_R_native_parity')


def single_diagonal(full, single):
    m, s = full['metadata'], single['metadata']
    need(R.exact(single['velocity_native'], m['flow']['velocity_native'][6]), 'single_resume_native_velocity')
    need(R.exact({k: v[1] for k, v in m['reads'].items()}, s['reads']), 'single_resume_all_three_layer_native_reads')
    for name in ('pg', 'prefix_mask', 'prefix_positions', 'sampling_abi'):
        need(R.exact(m[name], s[name]), 'single_resume_' + name)
    for name, old in (('actual_x', m['flow']['x'][6]), ('actual_time', m['flow']['time'][6]),
                      ('time_weak_type', m['flow']['time_weak_type'][6])):
        need(R.exact(s[name], old), 'single_resume_' + name)


def own_target_identity(none, own):
    def canonical(actual):
        meta = {k: v for k, v in actual['metadata'].items() if k != 'W_selector'}
        meta['reads'] = {k: v for k, v in meta['reads'].items() if k != 'bridge_active'}
        return {**actual, 'metadata': meta}
    need(R.exact(canonical(none), canonical(own)), 'active_own_target_full_native_identity')


def run_candidate(*, campaign, saved, old_calls, I, Q, A, T, Z, runtime, guard, store, result,
                  references, reference_result, source_ref, audit, check):
    import numpy as np
    branches = I.encode_loaded(campaign, saved)
    input_identity = I.identity((branches, saved.noise, saved.modified_v))
    snapshots = {}; sites = {}
    def invoke(call_id, method, args, *, full=False):
        result['active_call'] = call_id; result['step'] = 'pre_call_guard'
        check(); guard(campaign)
        old = I.identity((branches, saved.noise, saved.modified_v, args))
        need(I.identity((branches, saved.noise, saved.modified_v)) == input_identity, 'candidate_input_unchanged_before_call')
        before = R.rng_identity()
        suffix = 10 if full else 1
        store.json('CHARGE-' + call_id + '.json', {'call_id': call_id, 'kind': 'full_continuation' if full else 'single_forward',
                   'method': method, 'prefix_prefills_reserved': 2, 'suffix_evaluations_reserved': suffix,
                   'charge_before_invoke': True, 'charged_epoch': time.time()})
        result['charged_entries'].append(call_id); result['prefix_prefills_reserved'] += 2; result['suffix_evaluations_reserved'] += suffix
        result['step'] = 'model_call'
        value, metadata = runtime[method](*args)
        runtime['wait']((value, metadata)); result['returned_entries'].append(call_id)
        actual = {'raw32' if full else 'velocity_native': R.host(value), 'metadata': R.host(metadata)}
        if full:
            try:
                decoded = campaign.decode(value, branches['T'].decode_states); runtime['wait'](decoded)
                actual['adapter7'] = R.host(decoded)
            except BaseException:
                store.actual(call_id, actual); raise
        result['step'] = 'persist_actual_before_gates'
        pg_equal = store.actual(call_id, actual)
        result['completed_entries'].append(call_id)
        result['step'] = 'qualify_actual'
        check()
        need(pg_equal and R.exact(actual['metadata']['pg'], references['pulse']['metadata']['pg']), 'candidate_all_prefills_equal_complete_R_PG')
        I.current_prefix(saved, actual['metadata'])
        need(I.identity((branches, saved.noise, saved.modified_v, args)) == old, 'candidate_input_target_and_tau_unchanged')
        need(R.rng_identity()['numpy'] == before['numpy'], 'candidate_numpy_RNG_unchanged')
        return actual, before
    def passed(call_id, evidence, before):
        guard(campaign); check()
        after = R.rng_identity()
        need(before['numpy'] == after['numpy'], 'candidate_numpy_RNG_unchanged_after_gates')
        store.json('CHECK-' + call_id + '.json', {'call_id': call_id, 'passed': True,
                   'evidence': evidence, 'rng_before': before, 'rng_after': after})
        result['passed_entries'].append(call_id)
    full, rng = invoke('CAN-01', 'full_inactive', (branches['T'], branches['C'], saved.noise, saved.modified_v), full=True)
    full_parity(references['pulse'], full, Q.Q)
    fullsite = A.extract_site(None, full['metadata'], full=True)
    pulse = A.extract_site(None, references['pulse']['metadata'], full=True, legacy_reference=True)
    persistent = A.extract_site(None, references['persistent']['metadata'], full=True, legacy_reference=True)
    old_w = [old_calls[r].field('reads', 'attention_output_post_WO_pre_gate').storage[1, 1, 0]
             for r in ('baseline_none_1', 'shared_pulse', 'shared_persistent')]
    targets, target_receipt = T.qualify_and_build_targets(*old_w, pulse['W_used'], persistent['W_used'], fullsite['W_before'], audit, check=check)
    store.json('TARGET-QUALIFICATION.json', target_receipt)
    passed('CAN-01', {'complete_inactive_R_parity': True, 'fixed_old_fresh_native_targets_qualified': True}, rng)
    identity = {**A.IDENTITY, 'snapshot_identity': I.identity(saved.raw_observations),
                'noise_identity': I.identity(saved.noise), 'prefix_identity': I.identity(full['metadata']['pg']),
                'model_identity': result['model_binding']['loaded_parameter_tree_sha256'],
                'source_manifest_sha256': source_ref['sha256']}
    x6 = np.array(full['metadata']['flow']['x'][6], copy=True)
    tau = runtime['restore_time'](full['metadata']['flow']['time'][6], full['metadata']['flow']['time_weak_type'][6])
    for call_id, method, role, active, purpose in SINGLES:
        target = T.decode_native_target(targets[role])
        actual, rng = invoke(call_id, method, (branches['T'], branches['C'], x6, tau, target, active))
        m = actual['metadata']
        need(R.exact(m['W_target_argument'], target) and R.exact(m['W_selector'], np.asarray(active)), 'actual_target_argument_and_selector')
        need(int(m['prefix_prefills']) == 2 and int(m['suffix_evaluations']) == 1 and
             not bool(m['product_is_consumed']) and bool(m['sampling_abi']['num_steps_is_trace_input']) and
             int(m['sampling_abi']['num_steps']) == 10, 'single_call_native_ABI_no_product_capture')
        need(R.exact(m['reads']['bridge_W_before'][[0, 2]], m['reads']['bridge_W_used'][[0, 2]]),
             'non_L9_actual_W_remains_own')
        need(R.exact(m['reads']['bridge_W_before'], m['reads']['attention_output_post_WO_pre_gate']),
             'before_trace_matches_actual_attention_output')
        site = A.extract_site(actual['velocity_native'], m)
        evidence = A.validate_site(fullsite, site, targets[role][0], actual_identity=identity,
                                   reference_identity=identity, active=active)
        snapshots[call_id], sites[call_id] = actual, site
        if call_id == 'S01':
            single_diagonal(full, actual)
        elif call_id == 'S02':
            own_target_identity(snapshots['S01'], actual)
        elif call_id == 'S03':
            need(R.exact(snapshots['S01'], actual), 'independent_none_complete_native_parity')
        elif call_id == 'S04':
            A.require_exact_fields(persistent, site, A.OUT_FIELDS)
            evidence['matches_fresh_original_persistent_j6_OUT_FIELDS'] = True
        elif call_id == 'S06':
            need(R.exact(snapshots['S04'], actual), 'independent_and_main_natural_d_complete_native_parity')
            A.require_exact_fields(persistent, site, A.OUT_FIELDS)
            evidence['matches_fresh_original_persistent_j6_OUT_FIELDS'] = True
        elif call_id == 'S07':
            need(R.exact(snapshots['S05'], actual), 'independent_and_main_common_c_complete_native_parity')
        elif call_id in ('S08', 'S09', 'S10'):
            first = {'S08': 'S01', 'S09': 'S06', 'S10': 'S07'}[call_id]
            need(R.exact(snapshots[first], actual), 'delayed_repeat_complete_native_parity')
        evidence['purpose'] = purpose
        passed(call_id, evidence, rng)
    check(); result['step'] = 'fixed_CPU_readout_after_all_native_gates'
    triplet = {role: sites[key] for role, key in (('none', 'S01'), ('natural_d', 'S06'), ('common_c', 'S07'))}
    velocity = A.velocity_readout({role: site['velocity'] for role, site in triplet.items()})
    for row in velocity['rows']:
        for label in ('d_effect', 'c_effect', 'c_minus_d'):
            row[label + '_RMS_model_coordinate_units'] = math.sqrt(row[label + '_energy'] / row['coordinates'])
    quantization = Z.analyze_native_triplet(triplet, check=check)
    rounding = Z.analyze_rounding_candidates({'fresh_Gstar_pulse': pulse, 'fresh_Gstar_persistent': persistent, **triplet}, check=check)
    store.json('VELOCITY-READOUT.json', velocity)
    store.json('QUANTIZATION-READOUT.json', quantization)
    store.json('ROUNDING-DIAGNOSTIC.json', rounding)
    check(); guard(campaign)
    result.update(status='B_BRIDGE_NATIVE_COMPLETE', analysis_complete=True, scientific_complete=True,
                  qualification_complete=True, active_call=None, step='candidate_complete',
                  same_W_setter_qualified=True, B_bridge_scientific_effect_evaluated=True,
                  A_model_calls=0, cross_state_same_delta_scientific_GO=False,
                  adapter7_single_action_claim=False, all_native_gates_passed=True,
                  native_product_captured=False, quantization_attribution='combined_only',
                  downstream_or_reading_sufficiency_claim=False)


def validate_context(context):
    need(sys.platform.startswith('linux'), 'candidate_model_and_arrays_cloud_Linux_only')
    need(context['schema'] == 'B-bridge-exact-cloud-stage-context-v1' and context['phase'] == 'candidate', 'C_only_restore_context')
    need(Path(context['source_root']).resolve() == SOURCE and Path(context['new_package_root']).resolve() == SOURCE, 'candidate_source_context_identity')
    need(Path(context['package_root']).resolve() == Path(context['old_package_root']).resolve() and
         Path(context['repo_root']).resolve() == (Path(context['old_package_root']) / 'repo').resolve(), 'candidate_original_source_restore')
    need(type(context['run_id']) is str and 0 < len(context['run_id']) < 256 and
         math.isfinite(context['science_deadline_epoch']) and time.time() < context['science_deadline_epoch'], 'candidate_run_deadline')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--context', type=Path, required=True)
    parser.add_argument('--source-manifest-sha256', required=True)
    args = parser.parse_args()
    need(args.context.is_file() and not args.context.is_symlink() and args.context.stat().st_size <= R.MAX_RESULT, 'small_candidate_context')
    context = json.loads(args.context.read_bytes()); validate_context(context)
    source_ref = verify_source(args.source_manifest_sha256)
    codec = R.module('_b_bridge_candidate_codec', HERE / 'native_codec.py')
    store = OutputStore(context['output_dir'], codec)
    result = {'schema': 'B-BRIDGE-candidate-result-v1', 'run_id': context['run_id'], 'phase': 'candidate',
        'synthetic': False, 'root_id': R.ROOT, 'query_index': 11, 'flow_index': 6, 'layer_index': 8,
        'source_manifest': source_ref, 'context_identity': R.digest(args.context.read_bytes()),
        'status': 'FAILED_PRESERVED', 'analysis_complete': False, 'scientific_complete': False,
        'qualification_complete': False, 'charged_entries': [], 'returned_entries': [], 'completed_entries': [], 'passed_entries': [],
        'prefix_prefills_reserved': 0, 'suffix_evaluations_reserved': 0,
        'model_call_caps': {'full_continuations': 1, 'single_forwards': 10, 'prefix_prefills': 22, 'suffix_evaluations': 20},
        'new_environment_steps': 0, 'training_calls': 0, 'automatic_retry': False, 'A_model_calls': 0,
        'same_W_setter_qualified': False, 'B_bridge_scientific_effect_evaluated': False,
        'scientific_output_budget_bytes': MAX_TOTAL, 'cross_state_same_delta_scientific_GO': False}
    def check(): need(time.time() < context['science_deadline_epoch'], 'candidate_absolute_science_deadline')
    exit_code = 2
    try:
        result['step'] = 'accepted_R_and_original_input_qualification'
        check()
        references, reference_result, admission = load_reference(context, codec)
        store.json('REFERENCE-ADMISSION.json', admission)
        I, saved, old_calls, proof = R.prepare_inputs(context, check)
        store.json('INPUT-QUALIFICATION.json', proof)
        result['step'] = 'isolate_native_candidate_source'
        package, binding = isolate_candidate_package(context, source_ref)
        store.json('SOURCE-BINDING.json', binding)
        result['step'] = 'load_actual_candidate_model'
        sys.path.insert(0, str(package / 'repo/src'))
        from openpi.eval.campaign_policy import CampaignPolicy
        campaign = CampaignPolicy(Path(context['checkpoint_root']))
        result['step'] = 'bind_actual_candidate_model_and_adapter'
        G = R.module('_b_bridge_candidate_model_guard', package / 'study/continuous/cloud-v1/model_guard.py')
        guard = G.bind(campaign, package_root=package, checkpoint_root=context['checkpoint_root'],
            restore_receipt_path=context['model_restore_receipt_path'], request={'model_plan': {'path': context['model_plan_relative']}})
        model = guard(campaign); store.json('MODEL-BINDING.json', model)
        result['model_binding'] = {k: model[k] for k in ('model_bound', 'adapter_bound', 'loaded_parameter_tree_sha256', 'normalization_sha256')}
        need(result['model_binding'] == reference_result['model_binding'], 'candidate_same_R_loaded_model_and_normalization')
        Q = R.module('_b_bridge_candidate_query', SOURCE / 'native/query.py')
        A = R.module('_b_bridge_site_acceptance', SOURCE / 'analysis/acceptance.py')
        T = R.module('_b_bridge_candidate_targets', SOURCE / 'native/targets.py')
        Z = R.module('_b_bridge_quantization', SOURCE / 'analysis/quantization.py')
        runtime = Q.production_runtime(campaign)
        audit = R.bounded_read(HERE / 'COMMON-PROBE-AUDIT-RESULT.json', R.AUDIT_REF, R.MAX_RESULT)
        run_candidate(campaign=campaign, saved=saved, old_calls=old_calls, I=I, Q=Q, A=A, T=T, Z=Z,
                      runtime=runtime, guard=guard, store=store, result=result, references=references,
                      reference_result=reference_result, source_ref=source_ref, audit=audit, check=check)
        exit_code = 0
    except BaseException as error:
        result.update(status='FAILED_PRESERVED', analysis_complete=False, scientific_complete=False,
                      qualification_complete=False, same_W_setter_qualified=False,
                      error_type=type(error).__name__, error_code=str(error) if isinstance(error, R.WorkerFault) else 'external_exception_redacted')
    result['unexecuted_entries'] = [call for call in CALL_IDS if call not in result['charged_entries']]
    result['charged_but_completion_unproven'] = [call for call in result['charged_entries'] if call not in result['completed_entries']]
    result['outputs'] = list(store.refs.values())
    result['observed_files'] = [p.name for p in sorted(store.path.iterdir()) if p.is_file()]
    result['finished_epoch'] = time.time()
    store.json('RESULT.json', result)
    print(json.dumps({'status': result['status'], 'result': store.refs['RESULT.json'], 'exit_code': exit_code}, sort_keys=True))
    return exit_code


if __name__ == '__main__': raise SystemExit(main())
