"""One retained CPU parent: exact restore, credential-free replay, bounded archive."""
import argparse,json,os,shutil,subprocess,sys,time
from pathlib import Path
import contract as C
from prepare_runtime import verify

def private_bytes(root):
    total=0
    for p in Path(root).rglob('*'):
        C.need(not p.is_symlink(),'no_private_cache_symlink')
        if p.is_file():total+=p.stat().st_size
        C.need(total<=C.LIMITS['private_tmp_limit_bytes'],'private4GiB_cap')
    return total
def run(root,deadline,science_deadline,archive_target):
    root=Path(root);package=root/'package';request,manifest=verify(package)
    C.need(sys.platform.startswith('linux')and time.time()<science_deadline<archive_target<deadline,'cloud_CPU_absolute_phases')
    admission=json.loads((root/'RUNTIME-ADMISSION-v1.json').read_bytes())
    C.need((science_deadline,archive_target,deadline)==tuple(admission[k]for k in('science_deadline_epoch','archive_target_epoch','deadline_epoch')),'same_admitted_clock')
    from byte_budget import bounded_http
    from bos_stdlib import BOSWriter
    from transport import ExactRestore,safe_error
    import backend_probe
    import runtime_adapter
    writer=BOSWriter(request['output_prefix'],deadline-30);science=root/'science';science.mkdir()
    cache=root/'exact-inputs';cache.mkdir();input_paths={};error=None;result=None;probe=None;child=None;peak=private_bytes(root)
    source_bytes=admission['source_package_bytes'];reserved=3*source_bytes+C.LIMITS['mac_readback_limit_bytes']
    with bounded_http(total_limit_bytes=C.LIMITS['transfer_limit_bytes']-reserved,
        input_limit_bytes=C.LIMITS['input_limit_bytes']-2*source_bytes,output_limit_bytes=C.LIMITS['output_limit_bytes'])as budget:
        def check():C.need(time.time()<science_deadline-30,'absolute_CPU_science_cutoff')
        try:
            check();probe=backend_probe.probe_isolated(deadline_epoch=min(science_deadline-30,time.time()+65))
            (science/'EGL-PROBE-v1.json').write_bytes(C.encoded(probe))
            C.need(probe.get('status')=='SOFTWARE_EGL_AVAILABLE','same_job_software_EGL_before_any_runtime_GET')
            C.need(shutil.disk_usage(root).free>=C.LIMITS['private_tmp_limit_bytes'],'private4GiB_free_before_restore')
            objects={r['uri']:r for r in request['read_objects']};restore=ExactRestore(writer,objects)
            for i,ref in enumerate(request['read_objects']):
                check();target=cache/f'{i:04d}-{ref["sha256"]}.data'
                with budget.download_bound(ref['bytes']):restore(ref,destination=target,deadline_epoch=science_deadline-30)
                input_paths[ref['uri']]=str(target)
            C.need(sum(p.stat().st_size for p in cache.iterdir())==request['planned_input_body_bytes'],'all_exact_original_input_bytes')
            peak=max(peak,private_bytes(root))
            restored,env,proof=runtime_adapter.restore(package,root,input_paths,check=check)
            peak=max(peak,private_bytes(root));proof.update(actual_private_bytes_after_restore=peak,
                conservative_peak_bytes=request['conservative_private_peak_bytes'],input_objects=len(input_paths),input_bytes=request['planned_input_body_bytes'])
            (science/'RUNTIME-RESTORE-v1.json').write_bytes(C.encoded(proof))
            ctx={'package_root':str(package),'input_paths':input_paths,'restored':restored,'EGL_probe':probe,
                'science_dir':str(science),'deadline_epoch':science_deadline-30}
            context=root/'REPLAY-CONTEXT-v1.json';context.write_bytes(C.encoded(ctx));context.chmod(0o440)
            log=science/'NATIVE-CHILD-LOG-v1.txt'
            with log.open('xb')as stream:
                child=subprocess.Popen([sys.executable,'-B',str(package/'worker/science_child.py'),str(context)],
                    env=env,cwd=root/'science-tmp',stdout=stream,stderr=subprocess.STDOUT)
                while child.poll()is None:
                    check();C.need(log.stat().st_size<=2*1024**2,'native_child_log2MiB');peak=max(peak,private_bytes(root))
                    try:child.wait(timeout=1)
                    except subprocess.TimeoutExpired:pass
            C.need(child.returncode==0,'native_child_failed_no_retry')
            result=json.loads((science/'REPLAY-SUMMARY-v1.json').read_bytes())
        except BaseException as failure:error=safe_error(failure)
        finally:
            if child is not None and child.poll()is None:
                child.terminate()
                try:child.wait(timeout=5)
                except subprocess.TimeoutExpired:child.kill();child.wait(timeout=5)
        if result is None:
            result={'schema':'recorded-path-replay-campaign-v1','roots':[],'root_denominator':4,'targets_per_root':3,
                'development_qualification_passed':False,'total_env_step_calls':None if child is not None else 0,
                'model_calls':0,'training_steps':0,'scientific_complete':False,'full_state_restore_qualified':False,'new_reaction_policy_evaluated':False,'failure':error}
            p=science/'REPLAY-SUMMARY-v1.json'
            if p.exists():result=json.loads(p.read_bytes())
            else:p.write_bytes(C.encoded(result))
        files=sorted(p for p in science.iterdir()if p.is_file());C.need(sum(p.stat().st_size for p in files)<=C.LIMITS['output_limit_bytes']-128*1024,'bounded_all_replay_outputs')
        rows=result.get('roots',[])
        complete=error is None and len(rows)==4 and all(r.get('physical_only_qualification')is True for r in rows)
        usage=budget.snapshot();checks={'no_model_or_training':result.get('model_calls')==result.get('training_steps')==0,
            'declared_root_denominator':result.get('root_denominator')==4,'all_four_strict_declared_state_pass':complete,
            'transport_clean':not any(usage[k]for k in('rejected_count','requests_failed','inflight_requests'))}
        complete=complete and all(checks.values());outputs=[]
        with budget.archive_phase():
            for p in files:
                C.need(time.time()<archive_target,'archive_absolute_target');outputs.append(writer.put(p,'science/'+p.name))
            summary={'schema':'recorded-replay-CPU-delivery-result-v1','run_id':request['run_id'],'source_run_id':request['run_id'],'root_id':'RECORDED-FOUR-ROOTS',
                'qualification_complete':complete,'scientific_complete':False,'renderer_available':probe is not None and probe.get('status')=='SOFTWARE_EGL_AVAILABLE',
                'status':'complete'if complete else'partial_or_qualification_failed','analysis_status':'COMPLETE'if complete else'PARTIAL',
                'failure':error,'checks':checks,'root_denominator':4,'cloud_objects':outputs,'GPU':0,'model_calls':0,'new_env_steps':result.get('total_env_step_calls'),
                'full_state_restore_qualified':False,'new_reaction_policy_evaluated':False,'pixel_replay_qualified':complete and all(r.get('image_bytes_exact')is True for r in rows),
                'private_logical_bytes_sampled_peak':peak,'conservative_private_peak_bytes':request['conservative_private_peak_bytes'],'budget':budget.snapshot()}
            p=root/'RESULT-v1.json';p.write_bytes(C.encoded(summary));C.need(time.time()<archive_target,'result_archive_deadline');ref=writer.put(p,'summary/RESULT-v1.json')
            delivery={'schema':'recorded-replay-CPU-delivery-v1','run_id':request['run_id'],'qualification_complete':complete,
                'renderer_available':summary['renderer_available'],'scientific_complete':False,'result':ref,'GPU':0,'model_calls':0,'new_env_steps':summary['new_env_steps'],'budget':budget.snapshot()}
            d=root/'DELIVERY-v1.json';d.write_bytes(C.encoded(delivery));C.need(time.time()<archive_target,'delivery_archive_deadline');writer.put(d,'summary/DELIVERY-v1.json')
    return summary
def main():
    p=argparse.ArgumentParser();p.add_argument('--execute-cloud',action='store_true');p.add_argument('--root',type=Path)
    for name in('deadline-epoch','science-deadline-epoch','archive-target-epoch'):p.add_argument('--'+name,type=float)
    a=p.parse_args()
    if not a.execute_cloud:print(json.dumps({'status':'dry_recorded_replay_no_native_or_cloud','cloud_calls':0,'credential_reads':0}));return 0
    result=run(a.root,a.deadline_epoch,a.science_deadline_epoch,a.archive_target_epoch);print(json.dumps({'status':result['status']}));return 0 if result['qualification_complete']else 2
if __name__=='__main__':raise SystemExit(main())
