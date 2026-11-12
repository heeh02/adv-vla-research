"""Small immutable CPU archive writer and charge ledger; no submission/network."""
import hashlib,json,os,time
from pathlib import Path
import numpy as np

class LedgerError(RuntimeError):pass
def need(ok,code):
    if not ok:raise LedgerError(code)
def flatten(value,path='root',arrays=None):
    if isinstance(value,dict):
        need(all(isinstance(k,str)and'/'not in k for k in value),'safe_array_keys')
        return {k:flatten(v,path+'/'+k,arrays)for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [flatten(v,path+'/'+str(i),arrays)for i,v in enumerate(value)]
    if value is None or type(value)in (str,int,bool,float):
        if type(value)is float:need(np.isfinite(value),'finite_scalar')
        return value
    a=np.asarray(value);need(not a.dtype.hasobject,'no_object_arrays')
    need(a.dtype.kind in 'biufcUS'or a.dtype.name=='bfloat16','native_numeric_arrays_only')
    arrays[path]=a.view(np.uint16)if a.dtype.name=='bfloat16'else a
    return {'array':path,'shape':list(a.shape),'logical_dtype':a.dtype.name,'sha256':hashlib.sha256(a.tobytes()).hexdigest()}
class Ledger:
    def __init__(self,root,*,limits,deadline_epoch,plan,temporary_limit_bytes,inflight_limit_bytes,member_limit_bytes=64*1024**2,archive=None):
        self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True);self.events_path=self.root/'EVENTS.jsonl'
        need(not(self.root/'EVENTS.jsonl').exists(),'fresh_owned_directory')
        need(set(limits)=={'full','suffix','prefill','bytes'}and all(type(v)is int and v>0 for v in limits.values()),'explicit_frozen_limits')
        need(type(temporary_limit_bytes)is int and type(inflight_limit_bytes)is int and 0<inflight_limit_bytes<=temporary_limit_bytes,"explicit_temp_and_inflight_caps")
        need(type(member_limit_bytes)is int and 0<member_limit_bytes<=64*1024**2,"existing_BOS_member_cap")
        self.member_limit=member_limit_bytes
        self.temporary_limit=temporary_limit_bytes;self.inflight_limit=inflight_limit_bytes;self.hot_bytes=0
        self.limits=dict(limits);self.deadline=deadline_epoch;self.used={k:0 for k in limits};self.seen=set();self.seq=0;self.members=[];self.archive=archive;self.active_query=None;self.query_start=None;self.small_stream=None
        self.emit('PLAN',{'plan':plan,'limits':limits,'deadline_epoch':deadline_epoch},{})
    def check(self):need(time.time()<self.deadline,'absolute_science_deadline')
    def begin_query(self,query):
        self.seal_small()
        need(self.active_query is None and type(query)is int,"one_actual_query_archive_scope")
        self.active_query=query;self.query_start=len(self.members)
    def end_query(self,query):
        need(query==self.active_query and self.archive is not None,"fresh_GET_verified_archive_required")
        self.seal_small()
        rows=[dict(row)for row in self.members[self.query_start:]]
        self.archive.commit(query,rows)
        self.hot_bytes-=sum(row["bytes"]for row in rows)
        need(self.hot_bytes>=0,"archive_hot_byte_accounting")
        self.active_query=None;self.query_start=None
    def emit_small(self,event,metadata):
        """Bounded sensor update JSONL; append one record, no per-substep file."""
        raw=(json.dumps({'event':event,'metadata':metadata,'epoch':time.time()},sort_keys=True,allow_nan=False)+'\n').encode()
        need(len(raw)<=128*1024,'small_sensor_event_limit')
        if self.small_stream is not None and self.small_stream['bytes']+len(raw)>min(8*1024**2,self.inflight_limit):self.seal_small()
        if self.small_stream is None:
            path=self.root/f'{self.seq:06d}-SENSOR-EVENTS.jsonl';self.seq+=1
            with path.open('xb'):pass
            self.small_stream={'path':path,'bytes':0}
        need(self.used['bytes']+len(raw)+1024<=self.limits['bytes'],'small_stream_campaign_cap')
        need(self.hot_bytes+len(raw)+1024<=self.temporary_limit,'small_stream_temporary_cap')
        need(self.small_stream['bytes']+len(raw)<=self.inflight_limit,'small_stream_inflight_cap')
        with self.small_stream['path'].open('ab')as f:f.write(raw);f.flush();os.fsync(f.fileno())
        self.small_stream['bytes']+=len(raw);self.used['bytes']+=len(raw);self.hot_bytes+=len(raw)
        return {'stream':self.small_stream['path'].name,'end_offset':self.small_stream['bytes'],'locally_fsynced':True}
    def seal_small(self):
        if self.small_stream is None:return
        stream=self.small_stream;path=stream['path'];h=hashlib.sha256()
        with path.open('rb')as f:
            for data in iter(lambda:f.read(1024**2),b''):h.update(data)
        need(path.stat().st_size==stream['bytes'],'actual_small_stream_size')
        row={'path':path.name,'bytes':stream['bytes'],'sha256':h.hexdigest()}
        self.members.append(row);self.small_stream=None
        index=(json.dumps({'event':'SENSOR_STREAM_SEALED',**row},sort_keys=True)+'\n').encode()
        need(self.used['bytes']+len(index)<=self.limits['bytes']and self.hot_bytes+len(index)<=self.temporary_limit,'small_stream_index_cap')
        with self.events_path.open('ab')as f:f.write(index);f.flush();os.fsync(f.fileno())
        self.used['bytes']+=len(index);self.hot_bytes+=len(index)
    def charge(self,entry):
        self.check();key=(entry['root_trajectory_id'],entry['query_index'],entry['condition'],entry['noise_label'],entry.get('noise_role','parent_actual'if entry['kind']=='parent_stock'else'diagnostic_fixed'),entry['kind'])
        need(key not in self.seen,'no_model_retry_or_duplicate_call')
        need((entry['full'],entry['suffix'],entry['prefill'])==(1,10,1),'natural_full10_oneprefill_only')
        need(all(self.used[k]+entry[k]<=self.limits[k]for k in ('full','suffix','prefill')),'frozen_neural_budget_exceeded')
        self.emit('CHARGED',entry,{})
        for k in ('full','suffix','prefill'):self.used[k]+=entry[k]
        self.seen.add(key)
    def emit(self,event,metadata,values):
        arrays={};tree=flatten({'metadata':metadata,'values':values},arrays=arrays)
        raw=json.dumps(tree,sort_keys=True,allow_nan=False).encode()
        bound=len(raw)+sum(a.nbytes+2*len(k)+512 for k,a in arrays.items())+4096
        need(self.used['bytes']+bound+1024<=self.limits['bytes'],'reserve_before_archive_write')
        need(bound<=self.member_limit,'existing_BOS_64MiB_member_cap_before_write')
        need(bound<=self.inflight_limit,'one_file_inflight_cap_before_write')
        need(self.hot_bytes+bound+1024<=self.temporary_limit,'temporary_cap_before_write')
        name=f'{self.seq:06d}-{event}';path=self.root/(name+('.npz'if arrays else'.json'))
        if arrays:
            arrays['__schema__']=np.frombuffer(raw,dtype=np.uint8)
            with path.open('xb')as f:np.savez_compressed(f,**arrays);f.flush();os.fsync(f.fileno())
        else:
            with path.open('xb')as f:f.write(raw);f.flush();os.fsync(f.fileno())
        size=path.stat().st_size;need(size<=bound,'writer_bound_underestimated')
        digest=hashlib.sha256(path.read_bytes()).hexdigest()
        row={'seq':self.seq,'event':event,'path':path.name,'bytes':size,'sha256':digest,'epoch':time.time()}
        line=(json.dumps(row,sort_keys=True)+'\n').encode()
        with(self.root/'EVENTS.jsonl').open('ab')as f:f.write(line);f.flush();os.fsync(f.fileno())
        self.members.append({'path':path.name,'bytes':size,'sha256':digest})
        self.used['bytes']+=size+len(line);self.hot_bytes+=size+len(line);self.seq+=1
        return 'sha256:'+digest


class ProducerStore:
    """Route original producer writes into the same query archive/cap."""
    def __init__(self,ledger):self.ledger=ledger
    def arrays(self,name,values,role):
        receipt=self.ledger.emit('PRODUCER_ARRAYS',{'original_name':name,'role':role},values)
        return {'receipt':receipt,'original_name':name}
    def json(self,name,value):
        return self.ledger.emit('PRODUCER_JSON',{'original_name':name,'payload':value},{})
