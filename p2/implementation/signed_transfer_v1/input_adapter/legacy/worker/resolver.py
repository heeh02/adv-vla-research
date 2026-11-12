"""One exact physical object cache, small selected members, no network API."""
import gzip,hashlib,json,tarfile,time
from pathlib import Path
import contract as C
from prepare_runtime import document
import archive_pack as A
class Reads:
 def __init__(self,package,cache,planned,restore,budget,deadline):
  self.package=Path(package);self.cache=Path(cache);self.cache.mkdir(parents=True,exist_ok=False)
  self.plan=planned;self.restore=restore;self.budget=budget;self.deadline=deadline;self.current=None;self.seen=set();self.receipts=[];self.peak=0;self.noise_windows={};self.failures=[];self.failure_count=0
 def object_path(self,uri):
  suffix=Path(uri).suffix;C.need(suffix in('.json','.npz','.tar'),'declared_physical_object_suffix')
  return self.cache/('object'+suffix)
 def validate_windows(self):
  for ref in self.plan['allowed'].values():
   if ref.get('event')=='PARENT_STOCK':
    C.need(ref['key']in self.noise_windows and ref['window_input_key']in self.noise_windows and self.noise_windows[ref['key']]==self.noise_windows[ref['window_input_key']],'actual_parent_and_diagnostic_noise_same_root_query')
 def cleanup(self):
  if self.current is not None:
   p=self.object_path(self.current);C.need(p.is_file()and not p.is_symlink(),'owned_exact_cache');p.unlink();self.current=None
 def physical(self,uri):
  if self.current==uri:return self.object_path(uri)
  self.cleanup();C.need(uri not in self.seen,'member_order_one_GET_per_physical_object');row=self.plan['objects'][uri]
  C.need(row['bytes']<=32*1024**2 and time.time()<self.deadline,'cache_and_absolute_GET_bound');dest=self.object_path(uri)
  with self.budget.download_bound(row['bytes']):self.restore(row,destination=dest,deadline_epoch=self.deadline)
  C.need(C.ident(dest)=={k:row[k]for k in('bytes','sha256')},'whole_exact_object_SHA');self.seen.add(uri);self.current=uri
  if uri in self.plan['packs']:
   sink=A.Verifier(self.plan['packs'][uri])
   with dest.open('rb')as f:
    for chunk in iter(lambda:f.read(1024**2),b''):
     C.need(time.time()<self.deadline,'tar_verify_deadline');sink.consume(chunk)
   proof=sink.finish()
  else:proof=None
  self.receipts.append({'uri':uri,**{k:row[k]for k in('bytes','sha256')},'every_tar_member_proof':proof});self.peak=max(self.peak,row['bytes']);return dest
 def __call__(self,ref,consume):
  C.need(self.plan['allowed'].get(ref['key'])==ref and time.time()<self.deadline,'exact_callback_ref_and_deadline');selected=None
  try:
   if 'package_path'in ref:
    _,raw=document(self.package,ref);selected=self.cache/'selected.json';selected.write_bytes(raw);path=selected
   else:
    obj=self.physical(ref['transport_uri'])
    if ref.get('storage')=='tar':
     C.need(ref['bytes']<=31*1024**2,'bounded_selected_original');suffix=Path(ref['path']).suffix
     C.need(suffix in('.json','.npz'),'declared_native_member_suffix');selected=self.cache/('selected'+suffix)
     with tarfile.open(obj,'r:')as archive:
      member=archive.getmember(ref['path']);C.need(member.isfile()and not member.linkname and member.size==ref['bytes'],'exact_selected_tar_member')
      with archive.extractfile(member)as src,selected.open('xb')as dst:
       for b in iter(lambda:src.read(1024**2),b''):dst.write(b)
     path=selected
    else:path=obj
   live=sum(p.stat().st_size for p in self.cache.iterdir());C.need(live<=C.LIMITS['cache_limit_bytes'],'total_live_cache64MiB');self.peak=max(self.peak,live)
   C.need(C.ident(path)=={k:ref[k]for k in('bytes','sha256')},'exact_selected_original_SHA');before=path.stat();result=consume(path);after=path.stat()
   C.need((before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns)==(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns),'unchanged_consumed_original')
   return result
  except Exception as error:
   self.failure_count+=1
   if len(self.failures)<16:
    frames=[];tb=error.__traceback__
    while tb is not None:
     frames.append({'file':Path(tb.tb_frame.f_code.co_filename).name,'function':tb.tb_frame.f_code.co_name,'line':tb.tb_lineno});tb=tb.tb_next
    self.failures.append({'key':ref['key'],'logical_path':ref.get('path'),'storage':ref.get('storage','inline'),
     'cache_suffix':path.suffix if 'path'in locals()else None,'type':type(error).__name__,'frames':frames[-8:]})
   raise
  finally:
   if selected is not None and selected.exists():selected.unlink()
