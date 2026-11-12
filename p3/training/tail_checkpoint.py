"""Independent tail envelope; preserves native format and original parent commit."""
from pathlib import Path
import continuation_contract as C


def validate_envelope(envelope,tail):
    C.need(envelope['schema']=='T30-full-checkpoint-commit-v1' and envelope['scientific_contract_sha256']==C.digest(tail),'tail_outer_contract')
    C.need(envelope['arm']==tail['arm'] and type(envelope['training_seed']) is int and envelope['training_seed']==tail['seed'] and envelope['logical_run_id']==tail['logical_run_id'],'tail_outer_arm_seed_run')
    count=envelope['effective_updates']; native=envelope['native_commit']
    C.need(type(count) is int and 20000<count<=30000 and count==envelope['actual_stock_step']==native['actual_stock_step']==native['effective_updates'],'tail_effective_step')
    C.need(native['schema']=='T30-native-checkpoint-commit-v1' and native['logical_target_updates']==30000,'new_tail_native_schema')
    C.need(native['scientific_contract_sha256']==C.digest(tail) and native['logical_run_id']==tail['logical_run_id'] and native['arm']==tail['arm'] and native['training_seed']==tail['seed'],'tail_native_identity')
    role='new_endpoint30k' if count==30000 else 'recovery_process25k' if count==25000 else 'resume_partial'
    C.need(native['checkpoint_role']==role,'exact_tail_checkpoint_role')
    C.need(native['eval_eligible'] is (count==30000) and native['main_evaluation_parameters']=='EMA','20k_only_new_endpoint')
    C.need(envelope['parent_commit']==tail['continuation']['parent_commit'] and envelope['parent_contract_sha256']==tail['continuation']['parent_contract_sha256'],'exact_parent_ancestry')
    C.need(envelope['full_checkpoint_restore_permitted'] is True and native['full_checkpoint_restore_permitted'] is True,'full_resume_only')
    for key in ('stock_params_item_contains_EMA','whole_member_independent_GET_SHA_verified'):C.need(native[key] is True,'native_complete')
    names={'MIGRATION_VERIFIED-v1.json'} | ({'TEMPORAL_BINDING-v1.json','NEXT_INPUT-v1.json'} if tail['arm']=='T' else set())
    C.need(set(envelope['continuation_objects'])==names,'exact_continuation_sidecars')
    for value in envelope['continuation_objects'].values():C.ref(value)
    return envelope


def archive(old,native_archive,original_T_checkpoint,parts,process_nonce,step,sidecar,record,tail,*,checkpoint_writer,data_writer,out,check):
    out=Path(out);count=record['step'];cursor=old.cursor.Cursor(tail,C.read(Path(sidecar)/'CURSOR-v1.json'))
    proof_path=out/'MIGRATION_VERIFIED-v1.json';proof=C.read(proof_path)
    C.need(proof['status']=='PASS' and proof['tail_contract_sha256']==C.digest(tail),'actual_native_migration_required')
    C.need(cursor.completed==count and 20000<count<=30000,'new_effective_tail_checkpoint')
    payloads={'MIGRATION_VERIFIED-v1.json':proof}
    if tail['arm']=='T':
        payloads['TEMPORAL_BINDING-v1.json']=parts['temporal_schedule'].checkpoint_binding(cursor.snapshot(),count,active_reader_epoch=parts['pipeline'].active_epoch)
        payloads['NEXT_INPUT-v1.json']=original_T_checkpoint.next_witness(parts,cursor)
    native=native_archive.archive(step,sidecar,record,tail,checkpoint_writer=checkpoint_writer,data_writer=data_writer,out=out,check_deadline=check)
    objects={}
    for name,value in payloads.items():
        path=out/f'tail-step-{count:06d}'/name;C.write(path,value)
        obj=checkpoint_writer.put(path,f'step-{count:06d}/continuation/{name}')
        C.need({k:obj[k] for k in ('bytes','sha256')}==C.identity(path),'tail_sidecar_independent_readback')
        objects[name]=obj
    envelope=dict(schema='T30-full-checkpoint-commit-v1',native_commit=native,
        logical_run_id=tail['logical_run_id'],arm=tail['arm'],training_seed=tail['seed'],
        scientific_contract_sha256=C.digest(tail),effective_updates=count,actual_stock_step=count,
        parent_contract_sha256=tail['continuation']['parent_contract_sha256'],parent_commit=tail['continuation']['parent_commit'],
        continuation_objects=objects,saving_process_nonce=process_nonce,full_checkpoint_restore_permitted=True,
        success_or_learning_claimed=False,parent20k_commit_rewritten=False)
    validate_envelope(envelope,tail)
    path=out/f'TAIL_CHECKPOINT_COMMIT-step{count}.json';C.write(path,envelope)
    obj=data_writer.put(path,f'tail-checkpoints/step-{count:06d}.json')
    C.need({k:obj[k] for k in ('bytes','sha256')}==C.identity(path),'tail_outer_independent_readback')
    return {**envelope,'trigger_object':obj}


def validate_resume(old,original_T_checkpoint,parts,resume,process_nonce,check):
    tail=parts['contract']
    envelope=C.exact_bytes(resume['tail_commit_path'],C.ref(resume['tail_commit_ref']))
    validate_envelope(envelope,tail)
    C.need(envelope['saving_process_nonce']!=process_nonce,'independent_restore_process')
    C.need(resume['local_commit_sha256']==envelope['native_commit']['local_commit']['sha256'],'same_local_commit')
    cursor,saved=old.checkpoint_state.verify_restore(resume['step_directory'],resume['sidecar_directory'],tail,resume['local_commit_sha256'],check)
    C.need(cursor.completed==envelope['effective_updates'],'same_native_tail_step')
    proof=C.exact_bytes(resume['migration_path'],envelope['continuation_objects']['MIGRATION_VERIFIED-v1.json'])
    C.need(proof['status']=='PASS' and proof['tail_contract_sha256']==C.digest(tail) and proof['parent_commit']==tail['continuation']['parent_commit'],'original_migration_provenance')
    if tail['arm']=='T':
        binding=C.exact_bytes(resume['temporal_binding_path'],envelope['continuation_objects']['TEMPORAL_BINDING-v1.json'])
        parts['temporal_schedule'].verify_restore(binding,cursor.snapshot(),cursor.completed)
        expected=C.exact_bytes(resume['next_input_path'],envelope['continuation_objects']['NEXT_INPUT-v1.json'])
        C.need(original_T_checkpoint.next_witness(parts,cursor)==expected,'next_global_bank_native_B32_LR_RNG')
    return dict(actual_step=cursor.completed,tail_commit_sha256=resume['tail_commit_ref']['sha256'],full_native_restore_still_required=True)
