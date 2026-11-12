"""Credential-free original-collector replay, 0 model, same private CPU job."""
import gzip,importlib.abc,importlib.util,json,os,signal,sys,time
from pathlib import Path
import contract as C
from prepare_runtime import verify

class NoModelImports(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.')[0]in('jax','jaxlib','flax','tensorflow')or fullname.startswith(('openpi.models','openpi.policies','openpi.training'))or fullname=='openpi.eval.campaign_policy':
            raise RuntimeError('model_import_forbidden:'+fullname)
        return None
class EventLog:
    def __init__(self,folder):
        self.folder=Path(folder);self.path=self.folder/'REPLAY-EVENTS-v1.jsonl.gz';self.stream=gzip.GzipFile(filename=str(self.path),mode='wb',mtime=0);self.count=0
    def emit(self,event,metadata):
        self.count+=1;raw=C.encoded({'sequence':self.count,'event':event,'metadata':metadata})
        C.need(len(raw)<=2*1024**2,'bounded_single_small_event_no_arrays');self.stream.write(raw)
        if self.count%128==0:self.stream.flush();C.need(self.path.stat().st_size<=24*1024**2,'compressed_event_output24MiB')
        return 'events:'+str(self.count)
    def close(self):self.stream.close()
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
def run(context_path):
    ctx=json.loads(Path(context_path).read_bytes());package=Path(ctx['package_root']);request,manifest=verify(package)
    C.need(sys.platform.startswith('linux')and not any(k.startswith(('ONE_VLA_','AWS_','BCE_','BAIDU_','ADV_VLA_SOURCE'))for k in os.environ),'credential_free_Linux_child')
    sys.meta_path.insert(0,NoModelImports());sys.path.insert(0,str(package/'repo/src'))
    import replay_core as R
    import source_reader as Reader
    from native_replay import NativeReplay
    readout=module('_replay_original_reader',package/'worker/retained_readout.py')
    from runtime_adapter import ASSEMBLY
    canonical=module('canonical_step_capture',package/'repo/docs/experiments/first-paper-evidence-upgrade-20261008-v1/plus-primary-confirmation-20261010-v1/render-candidate-v1/canonical_step_capture.py')
    members=json.loads((package/'metadata/MEMBERS-v1.json').read_bytes());by={r['root_id']:r for r in members['roots']}
    output=Path(ctx['science_dir']);log=EventLog(output);result=None
    def stop(*_):raise TimeoutError('absolute_replay_science_deadline')
    signal.signal(signal.SIGALRM,stop);signal.setitimer(signal.ITIMER_REAL,max(.001,ctx['deadline_epoch']-time.time()))
    try:
        def source(root_id):return Reader.RootRecords(by[root_id],local_paths=ctx['input_paths'],readout=readout,rng_module=canonical)
        def runtime(root_id):
            original=json.loads((package/'metadata/roots'/f'{root_id}.json').read_bytes())
            return NativeReplay({'package_root':str(package),'source_manifest':manifest,'original_runtime_config':original,
                'restored':ctx['restored'],'resources':C.RESOURCES,'EGL_probe':ctx['EGL_probe']},emit=log.emit)
        result=R.run_campaign(request['replay_plan'],source_for_root=source,runtime_for_root=runtime,emit=log.emit,deadline_epoch=ctx['deadline_epoch'])
    except BaseException as error:
        result={'schema':'recorded-path-replay-campaign-v1','roots':[],'root_denominator':4,'targets_per_root':3,
            'development_qualification_passed':False,'total_env_step_calls':None,'model_calls':0,'training_steps':0,
            'full_state_restore_qualified':False,'new_reaction_policy_evaluated':False,
            'failure':{'type':type(error).__name__,'code':str(error)[:200]}}
    finally:signal.setitimer(signal.ITIMER_REAL,0);log.close()
    result['scientific_complete']=False;result['event_count']=log.count
    (output/'REPLAY-CAMPAIGN-v1.json').write_bytes(C.encoded(result))
    summary={k:v for k,v in result.items()if k!='roots'};summary['roots']=[]
    for row in result['roots']:
        brief={k:v for k,v in row.items()if k!='query_rows'}
        brief['target_witnesses']=[r for r in row.get('query_rows',[])if r['anchor']]
        brief['query_witness_count']=len(row.get('query_rows',[]));summary['roots'].append(brief)
    (output/'REPLAY-SUMMARY-v1.json').write_bytes(C.encoded(summary))
    C.need(sum(p.stat().st_size for p in output.iterdir()if p.is_file())<=C.LIMITS['output_limit_bytes']-512*1024,'all_replay_output_cap')
    return result
if __name__=='__main__':
    if len(sys.argv)!=2:print(json.dumps({'status':'dry_replay_child_no_native_import'}))
    else:run(sys.argv[1])
