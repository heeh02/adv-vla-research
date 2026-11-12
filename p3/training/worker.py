"""One root-admitted GPU child; bootstrap owns restore and host owns Create."""
import argparse
import contextlib
import os
from pathlib import Path
import signal
import sys
import threading
import time
import types
import uuid

import continuation_contract as C
import source_loader as S


def validate(request, admission, *, now=None):
    now=time.time() if now is None else now
    C.need(request['schema']=='T30-native-request-v1', 'native_request_schema')
    C.need(admission['schema']=='T30-native-child-admission-v1' and admission['execute'] is True, 'root_admitted_child_only')
    for key in ('run_id','allocation_id','science_deadline_epoch','allocation_deadline_epoch'):
        C.need(request[key]==admission[key], 'admission_binding:'+key)
    C.need(C.identity(request['request_path'])==admission['request_identity'], 'exact_native_request_bytes')
    C.need(request['branch'] in ('floor','reheat') and type(request['seed']) is int and request['seed'] in (0,1), 'two_branches_two_seeds')
    C.need(now < request['science_deadline_epoch'] <= request['allocation_deadline_epoch']-600, 'absolute_science_archive_clock')
    C.need(admission['account_active_and_reserved_GPU'] <= 8 and admission['resources']==dict(GPU=1,CPU=16,memory_GiB=128), 'root_one_GPU_admission_and_account_limit')
    output=Path(request['output_root'])
    C.need(output.is_absolute() and output.is_relative_to('/tmp/Jinwen/adv-vla') and output.name=='after-cold', 'isolated_scratch_segment')
    C.need(request['allocation_id'] in output.parts, 'allocation_owns_output')
    for field,prefix in [('data_output_prefix','redacted-24fc043a6cae'),
                         ('checkpoint_output_prefix','redacted-53ad6e2d54fe')]:
        value=request[field]
        C.need(value.startswith(prefix) and value.endswith('/'+request['allocation_id']+'/') and '/ct20k-continuation/' not in value, 'independent_new_output_scope')


def make_pipeline(old, common, epoch_module, prod, adapter, disk, parts, tail,
                  base_schedule, document, catalog, members, data, out, science_deadline):
    import temporal_tail as T
    parts['native_BatchPipeline']=old.pipeline.BatchPipeline
    schedule=T.TailSchedule(base_schedule,tail);parts['temporal_schedule']=schedule
    transport=adapter.cached_transport(prod.restore.ExactTransport(data,science_deadline),out,common)
    raw_restore=prod.restore.restore_epoch
    def guarded_restore(*args,**kwargs):
        disk.epoch(transport,out)
        return raw_restore(*args,**kwargs)
    restore=types.SimpleNamespace(**vars(prod.restore));restore.restore_epoch=guarded_restore
    guarded_prod=types.SimpleNamespace(**vars(prod));guarded_prod.restore=restore
    seq=[0]
    def cache_event(kind,epoch,value):
        seq[0]+=1
        C.write(out/'cache-events'/f'{seq[0]:04d}-{kind}-global{epoch}.json',value)
    parts['pipeline']=T.production(epoch_module,parts,guarded_prod,catalog,document,schedule,
        out,transport,members,cache_event)


def load_probe(request):
    cfg=request['fixed_train_probe']
    sources=[cfg[key] for key in ('runtime_source','candidate_source','dependencies_source')]
    expected=('probe_runtime.py','probe_candidate.py','PROBE-DEPENDENCIES-v1.json')
    parent=Path(sources[0]['path']).parent.resolve()
    for row,name in zip(sources,expected,strict=True):
        path=Path(row['path'])
        C.need(path.name==name and path.parent.resolve()==parent,'same_frozen_probe_source_directory')
        C.need(C.identity(path)=={k:row[k] for k in ('bytes','sha256')},'probe_source_whole_SHA')
    return S.load('_T30_fixed_probe_runtime',sources[0]['path']).factory(request)


def execute(request, admission, *, clock=time.time, on_state_monitor=None):
    C.need(sys.platform.startswith('linux') and os.environ.get('JAX_PLATFORMS')=='cuda', 'Baidu_CUDA_child_only')
    validate(request,admission,now=clock())
    C.need(float(os.environ['ONE_VLA_BOS_STS_EXPIRATION_EPOCH'])>request['allocation_deadline_epoch']+600, 'whole_allocation_BOS_expiry')
    legacy=S.configure(request['frozen_repo_root'])
    old,common,epoch_module,original_T_checkpoint=S.load_original()
    import runtime_adapter as A
    import tail_checkpoint as K
    native=S.load_native_archive(old,common)
    prod=common.production()
    with common.aliases({'common':common}):
        TC=common.load('_T30_original_T_contract',common.HERE/'contract.py')
    adapter=S.load('_T30_data_composite_adapter',legacy.root/S.FORMAL_REL/'composite_adapter.py')
    disk=S.load('_T30_disk_guard',legacy.root/S.FORMAL_REL/'disk_guard.py')
    parent=C.read(request['parent_contract_path']);tail=C.read(request['tail_contract_path'])
    original_T=C.read(request['original_T_contract_path'])
    C.need(C.digest(original_T)==parent['continuation']['parent_contract_sha256'], 'original_T10_input_contract_ancestry')
    bundle=C.load_bundle(request['parent_metadata_directory'],request['parent_commit'],parent)
    C.verify(tail,parent,bundle)
    C.need(tail['logical_run_id']==request['run_id'] and tail['seed']==request['seed'] and
        tail['optimization_continuation']['branch']==request['branch'], 'run_branch_seed_contract')
    C.need(request['data_output_prefix'] not in request['parent_commit']['uri'] and
        request['checkpoint_output_prefix'] not in request['parent_commit']['uri'], 'parent_scope_read_only')
    document,catalog,base_schedule,members=adapter.production_inputs(
        {**request,'contract':original_T},old,prod,common,TC)
    for field in ('source_support_manifest_sha256','physical_payload_manifest_sha256'):
        C.need(base_schedule._document[field]==bundle['temporal_binding'][field], 'same_T20_physical_support:'+field)
    out=Path(request['output_root']);out.mkdir(parents=True,exist_ok=False)
    nonce=uuid.uuid4().hex;stop=threading.Event()
    signal.signal(signal.SIGTERM,lambda *_:stop.set());signal.signal(signal.SIGINT,lambda *_:stop.set())
    def science():
        if stop.is_set() or clock()>=request['science_deadline_epoch']:
            raise old.cursor.ScienceStop()
    def archive_check():
        C.need(clock()<request['allocation_deadline_epoch']-120,'absolute_archive_deadline')
    def hard_exit():
        os._exit(124)
    timer=threading.Timer(max(0,request['allocation_deadline_epoch']-clock()),hard_exit)
    timer.daemon=True;timer.start()
    @contextlib.contextmanager
    def guard():
        watchdog=threading.Timer(max(0,min(480,request['allocation_deadline_epoch']-120-clock())),hard_exit)
        watchdog.daemon=True;watchdog.start()
        try:
            with disk.save_guard(out,hard_exit):
                yield
        finally:
            watchdog.cancel()
    Writer=old.checkpoint_archive.writer_class(request['data_output_prefix'],request['checkpoint_output_prefix'])
    data=Writer(request['data_output_prefix'],request['allocation_deadline_epoch']-120);parts=None
    try:
        if on_state_monitor is None:
            on_state_monitor=load_probe(request)
        def probe_and_publish(parts,state,count,check):
            paths=on_state_monitor(parts,state,count,check)
            for path in paths or []:
                path=Path(path)
                C.need(path.resolve().is_relative_to(out.resolve()) and path.is_file() and not path.is_symlink(),'owned_probe_output')
                archive_check();data.put(path,'training/'+path.name)
        with common.aliases(vars(old)):
            parts=old.runtime.prepare(TC.baseline(old,tail['seed']),index_path=request['index_path'],
                qualification_path=request['qualification_path'],qualification_sha256=request['qualification_sha256'],
                restore_root=request['native_restore_root'],original_checkpoint=request['original_checkpoint_root'],
                tokenizer=request['tokenizer_path'],checkpoint_output=out/'stock',exp_name='continuation',check_science=science)
            C.need(parts['config'].model.pi05 is True and parts['config'].model.discrete_state_input is False and
                parts['config'].data.extra_delta_transform is False,'unchanged_state_and_label_flags')
            parts=A.prepare_parts(old,parts,parent,tail,bundle)
            make_pipeline(old,common,epoch_module,prod,adapter,disk,parts,tail,
                base_schedule,document,catalog,members,data,out,request['science_deadline_epoch'])
            C.write(out/'PREPARE-v1.json',dict(parent_commit=request['parent_commit'],tail_contract_sha256=C.digest(tail),
                original_stock_B32_parity=parts['persistent_input_vs_stock_loader'],extra_training_update_calls=0,
                frozen_repo_root=str(legacy.root),source_identity=C.identity(C.HERE/'SOURCE_MANIFEST-v1.json')))
            def publish(step,sidecar,record,contract):
                disk.checkpoint_record(record,out,sidecar)
                writer=Writer(request['checkpoint_output_prefix'],min(clock()+480,request['allocation_deadline_epoch']-120))
                return K.archive(old,native,original_T_checkpoint,parts,nonce,step,sidecar,record,contract,
                    checkpoint_writer=writer,data_writer=data,out=out,check=archive_check)
            def log(name,value):
                with (out/name).open('ab') as stream:
                    stream.write(C.canonical(value)+b'\n')
            def publish_monitor(path,value):
                archive_check();data.put(path,'training/'+Path(path).name)
            result=A.run(old,common.aliases,parts,parent=parent,bundle=bundle,resume_kind=request.get('resume_kind','parent20k'),
                resume=request['resume'],out=out,original_T_checkpoint=original_T_checkpoint,process_nonce=nonce,
                sidecar_root=out/'sidecars',science_check=science,archive_check=archive_check,archive_checkpoint=publish,
                on_step=lambda value:log('UPDATES-v1.jsonl',value),on_epoch=lambda value:log('EPOCHS-v1.jsonl',value),
                on_monitor=publish_monitor,on_state_monitor=probe_and_publish,checkpoint_guard=guard)
            result.update(run_id=request['run_id'],allocation_id=request['allocation_id'],branch=request['branch'],
                arm='T',seed=tail['seed'],parent_commit=request['parent_commit'],process_nonce=nonce,
                new_updates_this_lineage=result['effective_updates']-20000,formal_learning_claimed=False)
            C.write(out/'ALLOCATION_RESULT-v1.json',result)
            for path in sorted(out.iterdir()):
                if path.is_file() and path.suffix in ('.json','.jsonl'):
                    archive_check();data.put(path,'training/'+path.name)
            return result
    except BaseException as error:
        C.write(out/'FAILURE-v1.json',dict(run_id=request['run_id'],exception_type=type(error).__name__,
            raw_error_or_credentials_logged=False,automatic_retry=False,completed_training_claimed=False))
        try:
            for path in sorted(out.iterdir()):
                if path.is_file() and path.suffix in ('.json','.jsonl') and path.stat().st_size<=16*1024**2:
                    archive_check();data.put(path,'training/failure-metadata/'+path.name)
        except BaseException:
            pass
        raise
    finally:
        timer.cancel()
        if on_state_monitor is not None and hasattr(on_state_monitor,'close'):
            on_state_monitor.close()
        if parts and hasattr(parts['store'],'close'):
            parts['store'].close()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--request',required=True);parser.add_argument('--admission',required=True)
    parser.add_argument('--execute',action='store_true')
    args=parser.parse_args()
    if not args.execute:
        print('{"execute":false,"credential_calls":0,"cloud_calls":0}')
        return
    request,admission=C.read(args.request),C.read(args.admission)
    C.need(Path(args.request).resolve()==Path(request['request_path']).resolve(),'exact_request_path')
    execute(request,admission)

if __name__=='__main__':
    main()
