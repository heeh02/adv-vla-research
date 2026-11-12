"""Retained CPU transport/archive with exact CONF source joins; no new fit."""
import argparse,json,signal,sys,time
from pathlib import Path
import contract as C
from prepare_runtime import verify

class Output:
 def __init__(self,path,limit):
  self.path=Path(path);self.path.mkdir(parents=True,exist_ok=True);C.need(not any(self.path.iterdir()),'empty_private_science_output');self.limit=limit;self.used=0;self.files=[];self.pools={'a':0,'b':0,'control':0}
 def put(self,pool,name,raw):
  C.relative(name);C.need('/'not in name and type(raw)is bytes and len(raw)>0,'flat_output_bytes');p=self.path/name
  caps={'a':4*1024**2,'b':2*1024**2,'control':2*1024**2-128*1024}
  C.need(not p.exists()and self.used+len(raw)<=self.limit and self.pools[pool]+len(raw)<=caps[pool],'output_prewrite_byte_and_pool_cap')
  self.used+=len(raw);self.pools[pool]+=len(raw)
  with p.open('xb')as f:f.write(raw)
  row={'path':name,'bytes':len(raw),'sha256':C.sha(raw)};self.files.append(row);return row
 def put_a(self,name,raw):return self.put('a',name,raw)
 def put_b(self,name,raw):return self.put('b',name,raw)
 def put_control(self,name,raw):return self.put('control',name,raw)

from resolver import Reads

class ReaderGroup:
 def __init__(self,package,root,planned,restore,budget,deadline):
  self.package,self.root,self.planned,self.restore,self.budget,self.deadline=package,root,planned,restore,budget,deadline;self.readers={}
 def get(self,name):
  if name not in self.readers:self.readers[name]=Reads(self.package,self.root/'work'/('cache-'+name),self.planned['phases'][name],self.restore,self.budget,self.deadline)
  return self.readers[name]
 def cleanup(self):
  for reader in self.readers.values():reader.cleanup()
 def receipt(self):
  return {'phases':{n:{'objects':r.receipts,'failures':r.failures,'max_live_cache_bytes':r.peak}for n,r in self.readers.items()},'read_count':sum(len(r.receipts)for r in self.readers.values()),'expected_count':self.planned['physical_read_count'],'input_body_bytes':sum(x['bytes']for r in self.readers.values()for x in r.receipts),'original_arrays_to_Mac':False}

def analyze(planned,*,package,root,restore,budget,deadline_epoch,synthetic=False):
 sys.path.insert(0,str(package/'worker/core'));import science
 output=Output(root/'work/science',C.LIMITS['output_limit_bytes']-128*1024);reads=ReaderGroup(package,root,planned,restore,budget,deadline_epoch)
 result=None;failure=None
 try:
  result=science.run(planned,reads,output,package=package,deadline_epoch=deadline_epoch,synthetic=synthetic)
  C.need(time.time()<deadline_epoch,'readout_before_science_deadline')
  C.need(not result['complete']or reads.receipt()['read_count']==planned['physical_read_count'],'all_planned_physical_reads_consumed')
 except BaseException as error:
  failure=error;frames=[];tb=error.__traceback__
  while tb is not None:
   frames.append({'file':Path(tb.tb_frame.f_code.co_filename).name,'function':tb.tb_frame.f_code.co_name,'line':tb.tb_lineno});tb=tb.tb_next
  result={'schema':'context-ablation-result-v1','complete':False,'synthetic':synthetic,'exception_type':type(error).__name__,'code':str(error)if len(str(error))<=160 and str(error).replace('_','').isalnum()else None,'frames':frames[-8:],'model_calls':0,'env_steps':0,'new_fit':True,'old_fit_identity':C.FIT_ID,'old_scores_replaced':False}
 finally:
  reads.cleanup();output.put_control('EXTRACTION-RECEIPT-v1.json',C.encoded(reads.receipt()));output.put_control('RESULT-v1.json',C.encoded(result))
 if failure is not None:raise failure
 return result

def classification(result,error,io):
 core=result is not None and result.get('complete')is True and result.get('synthetic')is False and error is None
 checks={'core_readout_complete':core,'full_planned_population':result is not None and result.get('DEV_shared_targets')==48 and result.get('CONF_KT_targets')==96 and result.get('CONF_KC_support_targets')==24,
 'original_preprocessing_fixed':result is not None and result.get('old_fit_identity')==C.FIT_ID and result.get('new_fit')is True and result.get('old_scores_replaced')is False,
 'transport_clean':not any(io[k]for k in('rejected_count','requests_failed','inflight_requests')),'model_free_CPU':True}
 return core,False,all(checks.values()),checks

def inventory(science):
 files=sorted(Path(science).iterdir());C.need(all(p.is_file()and not p.is_symlink()for p in files),'flat_immutable_readout_files')
 C.need(sum(p.stat().st_size for p in files)<=C.LIMITS['output_limit_bytes']-128*1024,'bounded_science_files');return files

def run(root,deadline,science_deadline,archive_target):
 root=Path(root);package=root/'package';C.need(sys.platform.startswith('linux')and time.time()<science_deadline<archive_target<deadline,'cloud_CPU_absolute_phases')
 request,config,manifest,planned=verify(package);admission=json.loads((root/'RUNTIME-ADMISSION-v1.json').read_bytes())
 C.need((science_deadline,archive_target,deadline)==tuple(admission[k]for k in('science_deadline_epoch','archive_target_epoch','deadline_epoch')),'exact_CPU_admitted_clock')
 from transport import ExactRestore,safe_error
 from byte_budget import bounded_http
 from bos_stdlib import BOSWriter
 writer=BOSWriter(request['output_prefix'],deadline-30);restore=ExactRestore(writer,planned['objects'])
 result=None;error=None;outputs=[];source_bytes=admission['source_package_bytes'];reserved=3*source_bytes+C.LIMITS['mac_readback_limit_bytes']
 with bounded_http(total_limit_bytes=C.LIMITS['transfer_limit_bytes']-reserved,input_limit_bytes=C.LIMITS['input_limit_bytes']-2*source_bytes,output_limit_bytes=C.LIMITS['output_limit_bytes'])as budget:
  previous=signal.getsignal(signal.SIGALRM)
  def scientific_cutoff(*_):raise TimeoutError('absolute_CPU_science_deadline')
  signal.signal(signal.SIGALRM,scientific_cutoff);signal.setitimer(signal.ITIMER_REAL,max(.001,science_deadline-time.time()))
  try:result=analyze(planned,package=package,root=root,restore=restore,budget=budget,deadline_epoch=science_deadline-10)
  except BaseException as failure:error=safe_error(failure)
  finally:signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,previous)
  science=root/'work/science';science.mkdir(parents=True,exist_ok=True);files=inventory(science);core,figures,complete,checks=classification(result,error,budget.snapshot())
  with budget.archive_phase():
   for p in files:
    C.need(time.time()<archive_target,'absolute_archive_target');outputs.append(writer.put(p,'science/'+p.name))
   summary={'schema':'context-ablation-CPU-delivery-result-v1','run_id':request['run_id'],'source_run_ids':request['source_run_ids'],'root_id':request['root_id'],
    'scientific_complete':complete,'core_readout_complete':core,'figure_complete':figures,'status':'complete'if complete else'partial_preserved','analysis_status':'COMPLETE'if core else'PARTIAL',
    'source_input_query_count':144,'task_count':4,'checks':checks,'failure':error,'worker_status':None if result is None else ('COMPLETE'if result['complete']else'PARTIAL'),'worker_result':result,
    'cloud_objects':outputs,'GPU':0,'model_calls':0,'new_env_steps':0,'visual_QA':'NOT_APPLICABLE_NO_PLOTS','budget':budget.snapshot()}
   p=root/'RESULT-v1.json';p.write_bytes(C.encoded(summary));C.need(time.time()<archive_target,'summary_archive_deadline');ref=writer.put(p,'summary/RESULT-v1.json')
   delivery={'schema':'context-ablation-CPU-delivery-v1','run_id':request['run_id'],'scientific_complete':complete,'result':ref,'GPU':0,'model_calls':0,'new_env_steps':0,'budget':budget.snapshot()}
   d=root/'DELIVERY-v1.json';d.write_bytes(C.encoded(delivery));C.need(sum(f.stat().st_size for f in files)+p.stat().st_size+d.stat().st_size<=C.LIMITS['output_limit_bytes'],'final8MiB')
   C.need(time.time()<archive_target,'delivery_archive_deadline');writer.put(d,'summary/DELIVERY-v1.json')
 return summary

def main():
 p=argparse.ArgumentParser();p.add_argument('--execute-cloud',action='store_true');p.add_argument('--root',type=Path)
 for name in('deadline-epoch','science-deadline-epoch','archive-target-epoch'):p.add_argument('--'+name,type=float)
 a=p.parse_args()
 if not a.execute_cloud:print(json.dumps({'status':'candidate_no_cloud','GPU':0,'cloud_calls':0,'credential_reads':0}));return 0
 r=run(a.root,a.deadline_epoch,a.science_deadline_epoch,a.archive_target_epoch);print(json.dumps({'status':r['status']}));return 0 if r['scientific_complete']else 2
if __name__=='__main__':raise SystemExit(main())
