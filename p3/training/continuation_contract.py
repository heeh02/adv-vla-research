"""T20 full-state parent and two independently identified T30 continuations."""
import copy
import hashlib
import json
from pathlib import Path, PurePosixPath

HERE = Path(__file__).resolve().parent
LR = dict(warmup_steps=1000, peak_lr=2.5e-5, decay_steps=10000, decay_lr=2.5e-6)

def need(ok, why):
    if not ok:
        raise ValueError(why)

def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()

def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()

def identity(path):
    path = Path(path)
    need(path.is_file() and not path.is_symlink(), 'regular_small_file')
    need(path.stat().st_size <= 2*1024**2, 'small_metadata_only')
    raw = path.read_bytes()
    return dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())

def read(path):
    identity(path)
    return json.loads(Path(path).read_bytes())

def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        stream.write(canonical(value) + b'\n')

def ref(value):
    need(type(value) is dict and type(value.get('bytes')) is int and value['bytes'] > 0, 'positive_ref_bytes')
    sha = value.get('sha256', '')
    need(type(sha) is str and len(sha) == 64 and set(sha) <= set('0123456789abcdef'), 'ref_SHA256')
    need(value.get('uri', '').startswith(('redacted-011187c171d5',
        'redacted-7c45a5749fb6')), 'exact_Jinwen_BOS_ref')
    return {key:value[key] for key in ('uri', 'bytes', 'sha256')}

def exact_bytes(path, reference):
    need(identity(path) == {key:reference[key] for key in ('bytes', 'sha256')}, 'metadata_whole_SHA')
    return read(path)

def validate_parent(parent, bundle):
    need(parent['schema'] == 'ct-full-state-continuation-20k-contract-v1', 'actual_T20_contract_schema')
    need(parent['arm'] == 'T' and type(parent['seed']) is int and parent['seed'] in (0, 1), 'T_two_original_seeds_only')
    need((parent['total_updates'], parent['full_epochs'], parent['batch_size'], parent['batches_per_epoch']) == (20000,100,32,200), 'T20_full_endpoint')
    need(parent['selector'] == 'clean' and parent['initialization_seed'] == 0 and parent['LR'] == LR and parent['EMA'] == .999, 'parent_objective_and_numerics')
    commit, manifest, local, cursor = (bundle[key] for key in ('commit','manifest','local','cursor'))
    from source_loader import load_legacy
    legacy = load_legacy()
    legacy.checkpoint.validate_envelope(commit, parent)
    native = commit['native_commit']
    need(commit['effective_updates'] == native['actual_stock_step'] == 20000, 'parent_is_complete_20k')
    need(native['checkpoint_role'] == 'new_endpoint20k' and native['eval_eligible'] is True, 'parent_prespecified_endpoint')
    for obj in (commit, native, manifest):
        need(obj['scientific_contract_sha256'] == digest(parent) and obj['logical_run_id'] == parent['logical_run_id'], 'exact_parent_scientific_identity')
        need(obj['effective_updates'] == 20000, 'same_parent_step')
    need(manifest['params_online_opt_EMA_cursor_complete'] is True and manifest['whole_member_independent_GET_SHA_verified'] is True, 'full_restore_manifest')
    rows = manifest['files']
    need(4 < len(rows) <= 4096 and len({row['uri'] for row in rows}) == len(rows), 'bounded_unique_objects')
    paths = set()
    for row in rows:
        ref(row)
        name = row['cache_relative']; path = PurePosixPath(name)
        need(not path.is_absolute() and '..' not in path.parts and str(path) == name and name not in paths, 'canonical_unique_member_path')
        paths.add(name)
    stock = [row for row in rows if row['role'] == 'stock_native_checkpoint']
    side = [row for row in rows if row['role'] == 'matching_state_cursor_sidecar']
    need(len(stock)+len(side) == len(rows) and {row['cache_relative'] for row in side} ==
         {f'sidecar/20000/{name}' for name in ('CURSOR-v1.json','STATE_IDENTITY-v1.json','PREFIX_IDENTITY-v1.json','LOCAL_COMMIT-v1.json')}, 'full_sidecar_roster')
    need(all(row['cache_relative'] == 'checkpoint/20000/'+row['stock_relative'] for row in stock), 'native_restore_paths')
    for top in ('params', 'train_state'):
        need(any(row['stock_relative'].startswith(top+'/') for row in stock), 'EMA_only_not_full_resume')
    need([{'relative':r['stock_relative'],'bytes':r['bytes'],'sha256':r['sha256']} for r in stock] == local['stock_files'], 'native_inventory_matches_local')
    need(local['schema'] == 'cul-closed-stock-state-v1' and local['contract_sha256'] == digest(parent), 'parent_local_contract')
    need(local['step'] == local['effective_updates'] == 20000 and local['bad_or_skipped_update_included'] is False, 'valid_parent_updates')
    need(local['params_optimizer_EMA_format_unchanged'] is True and local['native_dtype_unchanged'] is True, 'unchanged_native_format')
    need(local['train_rng_identity']['training_seed'] == parent['seed'] and local['train_rng_identity']['stock_fold_in_global_step'] is True, 'parent_training_RNG')
    need(cursor == dict(schema='native-multistep-cursor-v1', contract_sha256=digest(parent),
        completed_updates=20000, next_epoch=100, next_batch_in_epoch=0,
        active_epoch_plan=None, active_epoch_plan_sha256=None), 'exact_done_parent_cursor')
    need(bundle['state']['tree_sha256'] == local['full_state_tree_sha256'] == manifest['native_full_state_tree_sha256'], 'whole_state_tree_metadata')
    b = bundle['temporal_binding']
    need(b['schema'] == 'ct-tail-temporal-next-cursor-binding-v1' and b['tail_contract_sha256'] == digest(parent), 'T20_temporal_contract')
    need(b['native_cursor_sha256'] == digest(cursor) and b['effective_updates'] == 20000 and
        (b['next_global_epoch'],b['next_batch_in_epoch'],b['next_physical_epoch'],b['next_epoch_map_sha256']) == (100,0,None,None), 'T20_done_physical_binding')
    need(b['original_schedule_sha256'] == parent['temporal_schedule_canonical_sha256'], 'same_original_50_epoch_schedule')
    need(bundle['next_input'] is None, 'done_parent_has_no_next_input_proof')
    proof = bundle['migration']
    need(proof['status'] == 'PASS' and proof['tail_contract_sha256'] == digest(parent) and proof['parent_commit'] == parent['continuation']['parent_commit'], 'parent_migration_ancestry')
    return ref(bundle['commit_ref'])

def schedule_config(tail):
    from lr_schedule import ContinuationSchedule
    import dataclasses
    return dataclasses.asdict(ContinuationSchedule(**tail['optimization_continuation']['future_LR']))

def build(parent, bundle, logical_run_id, branch, *, scientific_attempt_id, decision_identity, source_identity):
    parent_ref = validate_parent(parent, bundle)
    need(branch in ('floor','reheat'), 'two_frozen_branches')
    for label in (logical_run_id, scientific_attempt_id):
        need(type(label) is str and label and all(c.isalnum() or c == '-' for c in label), 'independent_identity_string')
    need(logical_run_id != parent['logical_run_id'], 'new_logical_run_required')
    from lr_schedule import ContinuationSchedule
    import dataclasses
    tail = copy.deepcopy(parent)
    tail.update(schema='T20-optimization-continuation-30k-contract-v1', logical_run_id=logical_run_id,
        total_updates=30000, full_epochs=150, total_row_exposures=960000,
        mandatory_checkpoint_steps=[25000,30000],
        scientific_change='T20_full_state_to_T30;320000_additional_rows;repeat_original_50_physical_epochs;future_LR_only',
        main_evaluation='prespecified30000_EMA;25000_recovery_and_process_only;no_best_checkpoint_selection')
    tail['validation']['effective_steps'] = list(range(20000,30001,1000))
    tail['parent20k_continuation'] = copy.deepcopy(parent['continuation'])
    tail['continuation'] = dict(parent_contract_sha256=digest(parent), parent_commit=parent_ref,
        parent_local_commit=ref(bundle['commit']['native_commit']['local_commit']),
        parent_cursor_sha256=digest(bundle['cursor']), parent_full_state_tree_sha256=bundle['local']['full_state_tree_sha256'],
        start_global_step=20000, first_new_effective_step=20001, start_global_epoch=100,
        target_global_step=30000, additional_row_exposures=320000, parent_row_exposures=640000,
        optimizer_EMA_RNG_reset=False, optimizer_schedule_count_reset=False,
        T_physical_epoch_rule='global_epoch%50', T_original_schedule_sha256=parent['temporal_schedule_canonical_sha256'],
        user_decision_identity=copy.deepcopy(decision_identity), continuation_source_identity=copy.deepcopy(source_identity))
    tail['optimization_continuation'] = dict(scientific_attempt_id=scientific_attempt_id, branch=branch,
        future_LR=dataclasses.asdict(ContinuationSchedule(branch)),
        parent_LR_field_is_historical=True, static_tx_rebuilt_with_future_callable=True,
        dynamic_opt_state_fully_restored=True, last_update_schedule_index=29999,
        independent_allocation_id_required=True, no_automatic_retry=True)
    return tail

def verify(tail, parent, bundle):
    c, o = tail['continuation'], tail['optimization_continuation']
    need(tail == build(parent,bundle,tail['logical_run_id'],o['branch'],
        scientific_attempt_id=o['scientific_attempt_id'],decision_identity=c['user_decision_identity'],
        source_identity=c['continuation_source_identity']), 'continuation_contract_drift')
    return tail

def load_bundle(folder, commit_reference, parent):
    folder = Path(folder)
    def get(name, reference):
        return exact_bytes(folder/name, reference)
    commit = get('OUTER_COMMIT-v1.json', ref(commit_reference)); native = commit['native_commit']
    manifest = get('CHECKPOINT_MANIFEST-v1.json', ref(native['checkpoint_manifest']))
    local = get('LOCAL_COMMIT-v1.json', ref(native['local_commit']))
    bundle = dict(commit=commit,commit_ref=ref(commit_reference),manifest=manifest,local=local)
    for name,key in [('CURSOR-v1.json','cursor'),('STATE_IDENTITY-v1.json','state'),('PREFIX_IDENTITY-v1.json','prefix')]:
        bundle[key] = get(name,local['sidecars'][name])
    for name,key in [('TEMPORAL_BINDING-v1.json','temporal_binding'),('NEXT_INPUT-v1.json','next_input'),('MIGRATION_VERIFIED-v1.json','migration')]:
        bundle[key] = get(name,ref(commit['continuation_objects'][name]))
    validate_parent(parent,bundle)
    return bundle
