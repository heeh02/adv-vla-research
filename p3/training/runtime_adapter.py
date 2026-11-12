"""Reuse the original run code object; change only explicit restore migration hooks."""
import ast
import copy
import dataclasses
import inspect
import types
from pathlib import Path
import continuation_contract as C


def migrate_cursor(old, parent, tail, bundle, cursor, saved):
    C.verify(tail,parent,bundle)
    C.need(cursor.snapshot() == bundle['cursor'] and saved == bundle['local'], 'actual_parent_restore_differs_from_pinned_metadata')
    snapshot = copy.deepcopy(cursor.snapshot())
    snapshot['contract_sha256'] = C.digest(tail)
    migrated = old.cursor.Cursor(tail,snapshot)
    C.need((migrated.completed,migrated.epoch,migrated.offset) == (20000,100,0), 'no_cursor_reset_or_skip')
    return migrated


def prepare_parts(old, parts, parent, tail, bundle):
    """Reuse qualified matched-C preparation; only future LR and target change."""
    from lr_schedule import ContinuationSchedule
    C.verify(tail,parent,bundle)
    C.need(C.digest(parts['contract']) == parent['matched_C_contract_sha256'], 'prepared_T_matched_C_contract')
    cfg = parts['config']
    C.need(cfg.num_train_steps == 10000 and cfg.seed == parent['seed'], 'qualified_original_prepare_config')
    C.need(dataclasses.asdict(cfg.lr_schedule) == parent['LR'] and dataclasses.asdict(cfg.optimizer) == parent['AdamW'] and cfg.ema_decay == parent['EMA'], 'same_parent_numerical_configuration')
    p = dict(parts)
    p['config'] = dataclasses.replace(cfg,num_train_steps=30000,lr_schedule=ContinuationSchedule(**C.schedule_config(tail)))
    p['contract'] = tail
    C.need(p['config'].optimizer is cfg.optimizer, 'unchanged_optimizer_configuration')
    # This only creates a template later. Orbax restores every dynamic leaf;
    # the callable tx is static and newly captures the future schedule.
    return p


def validate_output_roots(parts,resume,out,sidecar_root):
    """New output roots must be disjoint from the immutable restored parent."""
    protected=[Path(resume[k]).resolve(strict=True) for k in ('step_directory','sidecar_directory')]
    targets=[Path(parts['config'].checkpoint_dir).resolve(),Path(out).resolve(),Path(sidecar_root).resolve()]
    for target in targets:
        C.need(all(not(target==p or target.is_relative_to(p) or p.is_relative_to(target)) for p in protected),'resume_input_output_overlap')


def adapted_code(original, namespace):
    """Two observer calls; removing them must reproduce the exact original AST."""
    tree = ast.parse(inspect.getsource(original))
    baseline = ast.dump(tree, include_attributes=False)
    function = tree.body[0]
    mesh = next(node for node in function.body if isinstance(node, ast.With))
    positions = [i for i,node in enumerate(mesh.body) if isinstance(node,ast.Expr)
        and isinstance(node.value,ast.Call) and node.value.args
        and ast.unparse(node.value.args[0]) == "out / 'INITIAL_STATE-v1.json'"]
    C.need(len(positions) == 1, 'one_native_identity_verified_boundary')
    restore_call = ast.parse('_T30_audit_tx(state, template, cursor, cfg)').body[0]
    mesh.body.insert(positions[0],restore_call)
    monitor = next(node for node in mesh.body if isinstance(node,ast.FunctionDef) and node.name == 'monitor_current')
    probe_call = ast.parse('_T30_state_monitor(parts, state, count, science_check)').body[0]
    monitor.body.append(probe_call)
    # Verify the entire function except the two observers is unchanged.
    mesh.body.remove(restore_call);monitor.body.remove(probe_call)
    C.need(ast.dump(tree,include_attributes=False) == baseline, 'original_runtime_AST_preserved')
    mesh.body.insert(positions[0],restore_call);monitor.body.append(probe_call)
    ast.fix_missing_locations(tree)
    exec(compile(tree,inspect.getsourcefile(original),'exec'),namespace)
    return namespace[original.__name__]


def audit_tx(state,template,cursor,cfg):
    import jax
    import numpy as np
    C.need(state.tx is template.tx, 'restored_state_uses_new_template_tx')
    C.need(int(jax.device_get(state.step)) == cursor.completed, 'native_step_matches_cursor')
    counts=[]
    for path,value in jax.tree_util.tree_flatten_with_path(state.opt_state)[0]:
        if getattr(path[-1], 'name', None) == 'count':
            raw=np.asarray(jax.device_get(value))
            C.need(raw.shape == () and raw.dtype == np.int32 and int(raw) == cursor.completed, 'restored_Adam_and_schedule_global_counts')
            counts.append(dict(path=jax.tree_util.keystr(path),count=int(raw),dtype=str(raw.dtype)))
    C.need(len(counts) == 2, 'exact_Adam_and_callable_schedule_counts')
    return dict(schema='T30-native-static-tx-and-dynamic-count-proof-v1',
        effective_updates=cursor.completed,template_tx_retained=True,optimizer_counts=counts,
        future_LR=dataclasses.asdict(cfg.lr_schedule),
        next_update_LR=float(jax.device_get(cfg.lr_schedule.create()(state.step))),
        full_momentum_and_EMA_bytes_checked_by_original_runtime=True)


def run(old, aliases, parts, *, parent, bundle, resume_kind, resume, out, original_T_checkpoint=None, process_nonce=None, on_state_monitor=None, **kwargs):
    """No path initializes from weights. Native restore remains the original run.

    `old` and `aliases` must come from source_loader.load_original(). Initial
    parent restore verifies old contract/files, then actual full state before
    emitting MIGRATION_VERIFIED; tail resume uses its own contract directly.
    """
    from source_loader import verify_sources
    verify_sources()
    tail = parts['contract']; C.verify(tail,parent,bundle)
    C.need(tail['continuation']['continuation_source_identity']==C.identity(C.HERE/'SOURCE_MANIFEST-v1.json'),'exact_running_tail_source')
    C.need(tail['continuation']['user_decision_identity']==C.identity(C.HERE.parent/'SCIENTIFIC_DECISION-v1.json'),'exact_root_scientific_decision')
    C.need(resume is not None and resume_kind in ('parent20k','tail'), 'explicit_full_resume_required')
    C.need(parts['config'].num_train_steps == 30000, 'tail_config_steps')
    C.need(dataclasses.asdict(parts['config'].lr_schedule) == C.schedule_config(tail), 'exact_frozen_future_LR_callable')
    C.need(dataclasses.asdict(parts['config'].optimizer) == parent['AdamW'] and parts['config'].ema_decay == parent['EMA'], 'unchanged_optimizer_EMA')
    if parent['arm']=='T':
        C.need(parts['temporal_schedule'].tail_contract_sha256 == C.digest(tail), 'T_tail_schedule_required')
        C.need(parts['pipeline'].schedule is parts['temporal_schedule'], 'actual_T_pipeline_schedule')
    validate_output_roots(parts,resume,out,kwargs['sidecar_root'])
    out = Path(out); restored = {}
    if resume_kind == 'tail':
        from tail_checkpoint import validate_resume
        C.need(process_nonce and original_T_checkpoint is not None, 'tail_resume_validation_inputs')
        validate_resume(old,original_T_checkpoint,parts,resume,process_nonce,kwargs['archive_check'])
    def restore(step_directory, sidecar, contract, local_sha, check):
        C.need(C.digest(contract) == C.digest(tail), 'requested_tail_identity')
        if resume_kind == 'parent20k':
            C.need(local_sha == tail['continuation']['parent_local_commit']['sha256'], 'exact_parent_local_commit')
            cursor,saved = old.checkpoint_state.verify_restore(step_directory,sidecar,parent,local_sha,check)
            if parent['arm']=='T':
                # Verify original done binding using original 50-epoch schedule.
                C.validate_parent(parent,bundle)  # Full T20 endpoint binding was SHA-verified.
            cursor = migrate_cursor(old,parent,tail,bundle,cursor,saved)
        else:
            cursor,saved = old.checkpoint_state.verify_restore(step_directory,sidecar,tail,local_sha,check)
            C.need(20000 < cursor.completed <= 30000, 'tail_resume_cannot_remigrate_parent_or_reset')
            if parent['arm']=='T':
                parts['temporal_schedule'].verify_restore(C.read(resume['temporal_binding_path']),cursor.snapshot(),cursor.completed)
        restored.update(cursor=cursor,saved=saved)
        return cursor,saved
    def write(path, value):
        path = Path(path)
        if path.name == 'INTERNAL_EMA-step20000.json':
            C.need(resume_kind == 'parent20k', '20000_monitor_only_at_migration')
            value = {**value,'measurement_role':'restoration_check_not_new_sample','duplicate_parent_validation_state':True,
                     'included_as_new_learning_curve_observation':False}
            path = path.with_name('RESTORE_CHECK_INTERNAL_EMA-step20000.json')
        old.protocol.write_new(path,value)
        if path.name == 'INITIAL_STATE-v1.json':
            # Original runtime reaches this only after actual stock restoration,
            # native tree/prefix equality, RNG identity and parameter census.
            C.need(restored and value['from_original_initialization'] is False, 'native_restore_before_migration_PASS')
            C.need(value['state_identity']['tree_sha256'] == restored['saved']['full_state_tree_sha256'], 'full_native_identity')
            proof = dict(schema='T30-native-migration-verified-v1',status='PASS',resume_kind=resume_kind,
                parent_contract_sha256=C.digest(parent),tail_contract_sha256=C.digest(tail),
                parent_commit=tail['continuation']['parent_commit'],actual_restored_step=value['actual_stock_step'],
                full_state_tree_sha256=value['state_identity']['tree_sha256'],train_rng_identity=value['rng'],
                cursor=restored['cursor'].snapshot(),parent_document_rewritten=False,
                optimizer_EMA_RNG_reinitialized=False,next_effective_step=value['actual_stock_step']+1,
                native_initial_state_identity=C.identity(path),GPU_qualification_claimed=False)
            old.protocol.write_new(out/'MIGRATION_VERIFIED-v1.json',proof)
    callback=kwargs.get('on_monitor')
    if callback is not None:
        def monitor_callback(path,value):
            path=Path(path)
            if path.name=='INTERNAL_EMA-step20000.json':
                path=path.with_name('RESTORE_CHECK_INTERNAL_EMA-step20000.json');value=C.read(path)
            return callback(path,value)
        kwargs['on_monitor']=monitor_callback
    p = types.SimpleNamespace(**vars(old.protocol)); p.write_new = write
    def audit(state,template,cursor,cfg):
        old.protocol.write_new(out/'SCHEDULE_RESTORE-v1.json',audit_tx(state,template,cursor,cfg))
    def state_monitor(parts,state,count,check):
        if on_state_monitor is not None and count in (20000,25000,30000):
            return on_state_monitor(parts,state,count,check)
    ns = dict(old.runtime.run.__globals__)
    ns.update(P=p,verify_restore=restore,_T30_audit_tx=audit,_T30_state_monitor=state_monitor)
    adapted = adapted_code(old.runtime.run,ns)
    with aliases({name:getattr(old,name) for name in ('protocol','cursor','pipeline','checkpoint_state','monitor','train_loop')}):
        return adapted(parts,out=out,resume=resume,**kwargs)
