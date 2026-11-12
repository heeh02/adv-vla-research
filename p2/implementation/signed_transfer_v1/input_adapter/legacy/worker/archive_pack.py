"""Bounded deterministic USTAR; fresh GET validates every original member.

No auth, remote listing, compression, deletion, or science call. Parent-only.
"""
import hashlib,io,json,os,re,tarfile,time
from pathlib import Path
import contract as C
PACK_LIMIT=32*1024**2
MEMBER_LIMIT=2048
MAX_FILES=131072
MAX_PACKS=256
PER_FILE_RESERVE=2048
FIXED_RESERVE=16*1024**2
INDEX='ARCHIVE-INDEX-v1.json'
PACKS={}

def round512(n):return (n+511)//512*512
def packed_size(rows,index_bytes):
 n=512+round512(index_bytes)+sum(512+round512(r['bytes'])for r in rows)+1024
 return (n+tarfile.RECORDSIZE-1)//tarfile.RECORDSIZE*tarfile.RECORDSIZE

def file_identity(path):
 p=Path(path);C.need(p.is_file()and not p.is_symlink(),'pack_regular_original');before=p.stat();h=hashlib.sha256()
 with p.open('rb')as f:
  for b in iter(lambda:f.read(1024**2),b''):h.update(b)
 after=p.stat();C.need((before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns)==(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns),'pack_original_unchanged')
 return {'bytes':after.st_size,'sha256':h.hexdigest()}

def pack(root,members,*,binding,deadline_epoch,output_remaining_bytes,inflight_limit_bytes):
 root=Path(root);C.need(len(members)<=MAX_FILES and not PACKS,'one_bounded_parent_pack_attempt');dest=root/'archive-packs';dest.mkdir(exist_ok=False)
 C.need(inflight_limit_bytes==64*1024**2,'actual_request_inflight64MiB')
 rows=sorted([{k:r[k]for k in('path','bytes','sha256')}for r in members],key=lambda r:r['path'])
 C.need(len({r['path']for r in rows})==len(rows),'unique_original_members')
 for r in rows:
  C.relative(r['path']);C.need(re.fullmatch('[A-Za-z0-9_./-]{1,100}',r['path'])is not None and len(C.encoded(r))<=256 and not r['path'].startswith('archive-packs/')and type(r['bytes'])is int and r['bytes']>=0,'bounded_USTAR_original_path_size')
  C.need(packed_size([r],1024+len(C.encoded(r)))<=PACK_LIMIT,'one_original_fits32MiB_USTAR')
 groups=[];current=[];body=0;entry_bytes=0
 def index(group,ordinal):return {'schema':'behavior-lossless-USTAR-index-v2','binding':binding,'pack_ordinal':ordinal,'compression':'none','original_members':group,'original_bytes':sum(r['bytes']for r in group),'original_files_deleted':False}
 C.need(len(C.encoded(index([],MAX_PACKS)))<=2048,'fixed_index_binding_header_bound')
 for r in rows:
  C.need(time.time()<deadline_epoch,'pack_absolute_archive_deadline')
  # Conservative exact-entry index bound avoids quadratic index serialization.
  extra=len(C.encoded(r));n=len(current)+1;index_bound=2048+entry_bytes+extra+n
  candidate_body=body+512+round512(r['bytes'])
  tar_bound=(512+round512(index_bound)+candidate_body+1024+tarfile.RECORDSIZE-1)//tarfile.RECORDSIZE*tarfile.RECORDSIZE
  if current and(n>MEMBER_LIMIT or tar_bound>PACK_LIMIT):groups.append(current);current=[];body=0;entry_bytes=0
  current.append(r);body+=512+round512(r['bytes']);entry_bytes+=extra
 if current:groups.append(current)
 C.need(len(groups)<=MAX_PACKS,'bounded_pack_count')
 planned=[];wire_total=0
 for i,group in enumerate(groups):
  raw=C.encoded(index(group,i));size=packed_size(group,len(raw));C.need(len(group)<=MEMBER_LIMIT and size<=PACK_LIMIT and len(raw)<=512*1024,'exact_pack_and_index_caps')
  C.need(size+len(raw)<=inflight_limit_bytes,'one_tar_index_READY_under_request64MiB');wire_total+=size+len(raw)
  planned.append((group,raw,size))
 # Broker READY/ACK overhead remains inside the per-file/fixed reservation.
 C.need(wire_total+len(groups)*8192<=output_remaining_bytes,'pack_and_broker_before_any_write_or_PUT')
 result=[]
 for i,(group,raw,size)in enumerate(planned):
  C.need(time.time()<deadline_epoch,'pack_before_file_write_deadline')
  prefix=f'archive-packs/pack-{i:04d}';path=root/(prefix+'.tar');idx=root/(prefix+'.index.json')
  with idx.open('xb')as f:f.write(raw)
  with tarfile.open(path,mode='w',format=tarfile.USTAR_FORMAT)as archive:
   def info(name,n):
    v=tarfile.TarInfo(name);v.size=n;v.mode=0o440;v.uid=v.gid=v.mtime=0;v.uname=v.gname='';return v
   archive.addfile(info(INDEX,len(raw)),io.BytesIO(raw))
   for row in group:
    C.need(time.time()<deadline_epoch,'pack_member_deadline');source=root/row['path'];C.need(file_identity(source)=={k:row[k]for k in('bytes','sha256')},'pack_original_source_SHA')
    with source.open('rb')as stream:archive.addfile(info(row['path'],row['bytes']),stream)
  p_ref={'path':prefix+'.tar',**file_identity(path)};i_ref={'path':prefix+'.index.json',**file_identity(idx)}
  C.need(p_ref['bytes']==size,'deterministic_USTAR_size')
  entry={'pack':p_ref,'index':i_ref,'original_members':group,'index_payload':raw,'fresh_GET':None}
  # Local streaming parse catches mutation during TAR write before publication.
  sink=Verifier(entry)
  with path.open('rb')as f:
   for b in iter(lambda:f.read(1024**2),b''):sink.consume(b)
  sink.finish();PACKS[p_ref['path']]=entry;result.append([p_ref,i_ref])
 return result

class Verifier:
 """Incremental regular-USTAR validator; never retains member payloads."""
 def __init__(self,entry):
  self.entry=entry;self.rows=[{'path':INDEX,**C.ident(entry['index_payload'])}]+entry['original_members'];self.index=0
  self.state='header';self.buf=bytearray();self.left=0;self.pad=0;self.h=None;self.bytes=0;self.footer=0;self.full=hashlib.sha256()
 def consume(self,chunk):
  self.bytes+=len(chunk);self.full.update(chunk);C.need(self.bytes<=self.entry['pack']['bytes'],'freshGET_pack_size_before_consuming');view=memoryview(chunk);pos=0
  while pos<len(view):
   if self.state=='footer':
    piece=view[pos:];C.need(not any(piece),'USTAR_no_extra_or_nonzero_trailing_members');self.footer+=len(piece);break
   if self.state=='header':
    count=min(512-len(self.buf),len(view)-pos);self.buf.extend(view[pos:pos+count]);pos+=count
    if len(self.buf)<512:continue
    C.need(self.index<len(self.rows),'USTAR_no_extra_member');header=tarfile.TarInfo.frombuf(bytes(self.buf),'ascii','strict');self.buf.clear();row=self.rows[self.index]
    C.need(header.name==row['path']and header.size==row['bytes']and header.type==tarfile.REGTYPE and not header.linkname,'USTAR_exact_member_name_size_regular')
    C.need((header.uid,header.gid,header.mtime,header.mode)==(0,0,0,0o440),'USTAR_fixed_metadata')
    self.left=row['bytes'];self.pad=(-self.left)%512;self.h=hashlib.sha256();self.state='data'
   if self.state=='data':
    count=min(self.left,len(view)-pos);self.h.update(view[pos:pos+count]);pos+=count;self.left-=count
    if self.left:continue
    C.need(self.h.hexdigest()==self.rows[self.index]['sha256'],'freshGET_original_member_payload_SHA');self.index+=1;self.state='padding'
   if self.state=='padding':
    count=min(self.pad,len(view)-pos);C.need(not any(view[pos:pos+count]),'USTAR_zero_padding');pos+=count;self.pad-=count
    if not self.pad:self.state='footer'if self.index==len(self.rows)else'header'
 def finish(self):
  C.need(self.state=='footer'and self.footer>=1024 and self.bytes==self.entry['pack']['bytes']and self.full.hexdigest()==self.entry['pack']['sha256'],'complete_freshGET_pack_and_all_members')
  return {'method':'fresh_GET_streaming_member_SHA','bytes':self.bytes,'sha256':self.full.hexdigest(),'verified_original_members':len(self.rows)-1,'original_bytes':sum(r['bytes']for r in self.rows[1:]),'index_sha256':self.rows[0]['sha256']}

def readback(store,key,size):
 matches=[entry for name,entry in PACKS.items()if key.endswith('/'+name)]
 if not matches:return store.stream(key,size,lambda _:None)
 C.need(len(matches)==1,'unique_current_pack_key');entry=matches[0];C.need(size==entry['pack']['bytes'],'freshGET_expected_pack_bytes');sink=Verifier(entry)
 result=store.stream(key,size,sink.consume);proof=sink.finish();C.need(result=={k:entry['pack'][k]for k in('bytes','sha256')},'store_and_member_verifier_agree');entry['fresh_GET']={**proof,'object_key':key};return result

def report():
 return {'schema':'behavior-packed-archive-v2','compression':'none','pack_limit_bytes':PACK_LIMIT,'max_original_members_per_pack':MEMBER_LIMIT,
  'original_files_deleted':False,'temporary_tar_and_index_copy_bytes':sum(e['pack']['bytes']+e['index']['bytes']for e in PACKS.values()),'original_member_bytes':sum(r['bytes']for e in PACKS.values()for r in e['original_members']),'packs':[{'pack':e['pack'],'index':e['index'],'original_member_count':len(e['original_members']),
  'original_bytes':sum(r['bytes']for r in e['original_members']),'fresh_GET':e['fresh_GET']}for e in PACKS.values()]}
