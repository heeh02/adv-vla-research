"""Bounded call ledger and native-array writer for the qualification worker.

No cloud API/model loading. Records failures and charges before invoking a
model; cannot resume a half-consumed plan by resetting counters.
"""
from pathlib import Path
import hashlib,json,os,time
import numpy as np

class LedgerError(RuntimeError):pass
def need(ok,code):
    if not ok:raise LedgerError(code)
def sha(b):return hashlib.sha256(b).hexdigest()

def flatten(value, path="root", arrays=None):
    arrays={}if arrays is None else arrays
    if isinstance(value,dict):
        need(all(isinstance(k,str)and'/'not in k for k in value),'safe_dictionary_keys')
        return {k:flatten(v,path+'/'+k,arrays)for k,v in value.items()}
    if isinstance(value,(tuple,list)):
        return {'__sequence__':type(value).__name__,'values':[flatten(v,path+'/'+str(i),arrays)for i,v in enumerate(value)]}
    if value is None or type(value)in(str,int,float,bool):
        if type(value)is float:need(np.isfinite(value),'nonfinite_scalar')
        return value
    a=np.asarray(value);need(not a.dtype.hasobject,'object_arrays_forbidden')
    need(path not in arrays,'unique_array_path')
    logical=a.dtype.name
    if logical=='bfloat16':stored=a.view(np.uint16)
    else:
        need(a.dtype.kind in 'biufc','supported_native_dtype')
        stored=a
    arrays[path]=stored
    return {'__array__':path,'shape':list(a.shape),'logical_dtype':logical,'storage_dtype':str(stored.dtype),
            'bytes':a.nbytes,'payload_sha256':sha(a.tobytes(order='C'))}

class Ledger:
    def __init__(self,root,budget,deadline_epoch,limit_bytes=2*1024**3):
        self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True)
        need(not self.root.is_symlink()and not(self.root/'EVENTS.jsonl').exists(),'fresh_owned_ledger_required')
        self.entries=budget['entries'];need(len(self.entries)==54,'exact54_entries')
        self.deadline=deadline_epoch;self.limit=limit_bytes;self.written=0;self.reserved=0;self.next=0;self.seq=0
        self.suffix=self.prefill=0;self.events=self.root/'EVENTS.jsonl'
        self.record('START',{'planned_entries':54,'suffix_cap':316,'prefill_cap':16,'deadline_epoch':deadline_epoch,'limit_bytes':limit_bytes})

    def record(self,event,payload):
        raw=(json.dumps({'seq':self.seq,'epoch':time.time(),'event':event,'payload':payload},sort_keys=True,allow_nan=False)+'\n').encode()
        need(self.written+len(raw)<=self.limit,'output_metadata_limit')
        with self.events.open('ab')as f:f.write(raw);f.flush();os.fsync(f.fileno())
        self.written+=len(raw);self.seq+=1

    def charge(self,entry):
        need(time.time()<self.deadline,'science_deadline_before_charge')
        need(self.next<len(self.entries)and entry==self.entries[self.next],'next_exact_entry_required')
        suffix=self.suffix+entry['suffix_evaluations_max'];prefill=self.prefill+entry['prefix_prefills_max']
        need(suffix<=316 and prefill<=16,'call_budget_exceeded')
        self.record('CHARGED',{'call_id':entry['call_id'],'suffix_reserved_total':suffix,'prefill_reserved_total':prefill})
        self.suffix,self.prefill=suffix,prefill;self.next+=1

    def persist(self,event,payload):
        arrays={};tree=flatten(payload,arrays=arrays)
        metadata=json.dumps(tree,sort_keys=True,allow_nan=False).encode()
        # Uncompressed bound is conservative; no dependence on a hoped-for ratio.
        bound=sum(a.nbytes+512+2*len(k.encode())for k,a in arrays.items())+len(metadata)+4096
        need(self.written+bound<=self.limit-65536,'reserve_output_before_file_write')
        base=f'{self.seq:04d}-{event}';target=self.root/(base+'.npz'if arrays else base+'.json')
        need(not target.exists(),'immutable_output')
        if arrays:
            arrays['__schema__']=np.frombuffer(metadata,dtype=np.uint8)
            with target.open('xb')as f:
                np.savez_compressed(f,**arrays);f.flush();os.fsync(f.fileno())
        else:
            with target.open('xb')as f:f.write(metadata);f.flush();os.fsync(f.fileno())
        size=target.stat().st_size;need(size<=bound,'writer_bound_underestimated')
        h=hashlib.sha256()
        with target.open('rb')as f:
            for b in iter(lambda:f.read(1024**2),b''):h.update(b)
        self.written+=size
        self.record(event,{'path':target.name,'bytes':size,'sha256':h.hexdigest(),'arrays':len(arrays)-(1 if arrays else 0)})
        return target
