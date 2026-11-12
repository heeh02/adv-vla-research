"""Trusted official init loading through a scoped benchmark-only torch facade.

Import does not import torch/numpy or deserialize anything. Every selected file
and native init0 is compared to the independently archived real v6 reference.
"""
import contextlib
import hashlib
import json
from pathlib import Path

CONTRACT_ID='official-plus-trusted-init-cpu-v1'
OPTIONS={'map_location':'cpu','weights_only':False}
PROTOCOL='released_task_language'

class InitContractError(ValueError):pass

def need(ok,message):
    if not ok:raise InitContractError(message)

def identity(path):
    h=hashlib.sha256();size=0
    with Path(path).open('rb') as stream:
        while part:=stream.read(1024**2):h.update(part);size+=len(part)
    return {'bytes':size,'sha256':h.hexdigest()}

def signature(path):
    row=Path(path).stat();return tuple(getattr(row,k) for k in ('st_dev','st_ino','st_size','st_mtime_ns','st_ctime_ns','st_nlink'))

def read_reference(path,expected,*,full=True):
    need(identity(path)=={k:expected[k] for k in ('bytes','sha256')},'actual v6 variant-reference byte identity differs')
    rows={}
    with Path(path).open() as stream:
        for line in stream:
            need(len(line)<=65536,'variant metadata line bound');row=json.loads(line);key=(row['suite'],row['task_id'])
            need(key not in rows and row['status']=='source_resolved_requires_init_bridge','duplicate or nonresolved v6 reference')
            init=row['init'];need(init.get('official_default_load_compatible') is False and init.get('CPU_trust_bridge_used') is True and init.get('default_error_type')=='UnpicklingError','original default incompatibility evidence differs')
            need(type(row['released_task_language']) is str and row['released_task_language'],'released prompt missing')
            rows[key]=row
    if full:need(len(rows)==10030,'full official reference must retain10030 cases')
    return rows

class TrustedInitReader:
    def __init__(self,root,reference,torch_module,numpy_module):
        self.root=Path(root).resolve();self.reference=reference;self.torch=torch_module;self.np=numpy_module;self.real_load=torch_module.load
        self.files={};self.cache={};self.signatures={};self.cache_content={};self.last=None;self.active=None;self.loads=0
        for row in reference.values():
            init=row['init'];path=init['path'];need(path.endswith('.pruned_init'),'non-init reference rejected')
            ref={k:init[k] for k in ('bytes','sha256')}
            need(path not in self.files or self.files[path]==ref,'same official init path has conflicting source identity')
            self.files[path]=ref
    def __getattr__(self,name):return getattr(self.torch,name)
    def cache_identity(self,states):
        digest=hashlib.sha256();digest.update(str(len(states)).encode()+b'\0')
        for item in states:
            if isinstance(item,self.torch.Tensor):
                need(item.device.type=='cpu','cached init tensor left CPU');item=item.detach().numpy()
            array=self.np.asarray(item);need(not array.dtype.hasobject,'non-numeric cached init representation unsupported')
            digest.update(array.dtype.str.encode()+b'\0'+repr(tuple(array.shape)).encode()+b'\0'+array.tobytes(order='C'))
        return digest.hexdigest()
    def load(self,path,*args,**kwargs):
        need(self.active is not None,'init load outside scoped official selector')
        need(not args and not kwargs,'official init loader call signature changed')
        original=Path(path);p=original.resolve()
        need(not original.is_symlink() and p.is_relative_to(self.root) and p.is_file(),'official init source outside restored tree')
        relative=str(p.relative_to(self.root));expected=self.active['init']
        need(relative==expected['path'] and relative in self.files,'actual official init selector differs from v6')
        need(self.torch.load is self.real_load,'global torch.load was modified')
        if relative not in self.cache:
            need(identity(p)==self.files[relative],'official init source SHA differs')
            before=signature(p)
            value=self.real_load(p,map_location='cpu',weights_only=False)
            need(signature(p)==before,'official init source changed during read')
            self.cache[relative]=value;self.signatures[relative]=before;self.cache_content[relative]=self.cache_identity(value);self.loads+=1
        else:need(signature(p)==self.signatures[relative],'cached init source identity changed')
        need(self.cache_identity(self.cache[relative])==self.cache_content[relative],'cached init arrays changed after prior consumer use')
        need(self.torch.load is self.real_load,'global torch.load changed during trusted read')
        self.last={'path':relative,**self.files[relative],'contract_id':CONTRACT_ID,'loader_options':dict(OPTIONS),
            'source_SHA_verified_before_deserialization':True,'global_torch_load_unchanged':True,'cached_states_native_typed_sha256':self.cache_content[relative],
            'original_default_load_compatible':False,'original_default_error_type':expected['default_error_type'],
            'default_failure_evidence':'actual immutable v6 reference; not retried or rewritten here'}
        return self.cache[relative]
    @contextlib.contextmanager
    def scoped(self,benchmark_module,expected):
        need(self.active is None and benchmark_module.torch is self.torch,'nested or wrong benchmark torch binding')
        old=benchmark_module.torch;self.active=expected;self.last=None;benchmark_module.torch=self
        try:yield
        finally:
            benchmark_module.torch=old;self.active=None
            need(self.torch.load is self.real_load,'global torch.load changed across local facade')
    def select(self,benchmark_module,benchmark,task_id,suite):
        expected=self.reference[(suite,task_id)]
        task=benchmark.get_task(task_id)
        need(task.name==expected['task_name'] and task.language==expected['released_task_language'],'official task or released prompt changed')
        with self.scoped(benchmark_module,expected):states=benchmark.get_task_init_states(task_id)
        need(self.last is not None and len(states)>0,'official selector did not load init states')
        first=states[0]
        if isinstance(first,self.torch.Tensor):
            need(first.device.type=='cpu','trusted loader returned a non-CPU tensor');first=first.detach().numpy()
        initial=self.np.asarray(first)
        need(initial.ndim==1 and initial.dtype.kind=='f' and self.np.isfinite(initial).all(),'native init0 shape/dtype/finite mismatch')
        proof={'init0_shape':list(initial.shape),'init0_dtype':initial.dtype.str,'init0_raw_sha256':hashlib.sha256(initial.tobytes(order='C')).hexdigest(),
            'states_shape':list(getattr(states,'shape',(len(states),)))}
        for key,value in proof.items():need(value==expected['init'][key],'actual trusted native init identity differs from real v6: '+key)
        return states,{**self.last,**proof,'matched_real_v6_init0':True,'released_task_language_unchanged':True}
