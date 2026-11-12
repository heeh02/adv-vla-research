"""CONF native metadata/arrays codec. Real tensor reads are cloud-Linux only.

No JAX/model/environment/network imports. The caller owns exact transport and
releases each restored member after this function has copied its typed values.
"""
from dataclasses import dataclass
from pathlib import Path,PurePosixPath
import hashlib,json,re,sys,zipfile
import numpy as np

LAYERS=(0,8,17);CALLS=(0,5,9);ANCHORS=(11,16,22)
MEMBER_CAP=32*1024**2;SCHEMA_CAP=2*1024**2;QUERY_NATIVE_CAP=128*1024**2
PG_FLOAT=('hidden_input','hidden_pre_attention_norm','cache_k','cache_v')
AE_BASE=('receiver_residual_input','hidden_post_block','attention_output_post_WO_pre_gate')
class FieldError(ValueError):pass
def need(ok,code):
 if not ok:raise FieldError(code)
def sha(raw):return hashlib.sha256(raw).hexdigest()
def token(path):
 s=path.stat();return s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns
def same_ref(a,b):return all(a.get(k)==b.get(k)for k in ('path','bytes','sha256'))

@dataclass(frozen=True)
class Native:
 storage:np.ndarray
 descriptor:dict
 @property
 def shape(self):return self.storage.shape
 @property
 def dtype(self):return self.descriptor['logical_dtype']
 def numeric(self):
  if self.dtype=='bfloat16':return (self.storage.astype(np.uint32)<<16).view(np.float32)
  return self.storage
 def identity(self):return {k:self.descriptor[k]for k in ('shape','logical_dtype','storage_dtype','bytes','payload_sha256')}

@dataclass
class Record:
 event:str
 metadata:dict
 arrays:dict
 ref:dict
 native_bytes:int
 synthetic:bool=False
 def field(self,*keys):
  x=self.arrays
  for k in keys:x=x[k]
  need(isinstance(x,Native),'actual_typed_leaf_required');return x

def load(path,ref,*,synthetic=False):
 need(synthetic or sys.platform.startswith('linux'),'real_neural_arrays_cloud_Linux_only')
 p=Path(path);need(p.is_file()and not p.is_symlink(),'regular_owned_original')
 before=token(p);need(type(ref['bytes'])is int and 0<ref['bytes']<=MEMBER_CAP and before[2]==ref['bytes'],'exact_member_size_bound')
 h=hashlib.sha256()
 with p.open('rb')as f:
  for block in iter(lambda:f.read(1024**2),b''):h.update(block)
 need(h.hexdigest()==ref['sha256']and token(p)==before,'whole_member_SHA_and_stability')
 match=re.fullmatch(r'\d+-([A-Z0-9_]+)\.(npz|json)',PurePosixPath(ref['path']).name)
 need(match is not None,'original_event_logical_filename')
 values={};seen=set();total=0
 if match[2]=='npz':
  with zipfile.ZipFile(p)as z:
   entries=z.infolist();names=[x.filename for x in entries]
   need(len(names)==len(set(names))and len(names)<=4096 and all(x.endswith('.npy')for x in names),'bounded_unique_NPZ_arrays')
   need(sum(x.file_size for x in entries)<=MEMBER_CAP and all(not(x.flag_bits&1)for x in entries),'uncompressed_NPZ_bound')
  with np.load(p,allow_pickle=False)as z:
   schema=z['__schema__'];need(schema.dtype==np.uint8 and schema.ndim==1 and schema.size<=SCHEMA_CAP,'bounded_native_schema')
   tree=json.loads(schema.tobytes());values={k:np.array(z[k],copy=True)for k in z.files if k!='__schema__'}
 else:
  need(ref['bytes']<=SCHEMA_CAP,'small_native_JSON');tree=json.loads(p.read_bytes())
 def decode(x):
  nonlocal total
  if isinstance(x,dict):
   if '__array__'in x:
    need(set(x)=={'__array__','shape','logical_dtype','storage_dtype','bytes','payload_sha256'},'exact_native_leaf_descriptor')
    k=x['__array__'];need(k in values and k not in seen,'unique_declared_native_leaf');seen.add(k);a=values[k]
    need(not a.dtype.hasobject and list(a.shape)==x['shape']and a.nbytes==x['bytes']and str(a.dtype)==x['storage_dtype'],'typed_shape_dtype_bytes')
    need((x['logical_dtype']=='bfloat16'and a.dtype==np.uint16)or(x['logical_dtype']==a.dtype.name and a.dtype.kind in 'biufc')or(x['logical_dtype']==a.dtype.str and a.dtype.kind in 'SU'),'native_logical_storage_dtype')
    need(sha(a.tobytes(order='C'))==x['payload_sha256'],'native_payload_SHA')
    numeric=(a.astype(np.uint32)<<16).view(np.float32)if x['logical_dtype']=='bfloat16'else a
    need(a.dtype.kind in 'SU' or np.isfinite(numeric).all(),'finite_native_payload');total+=a.nbytes;a.setflags(write=False);return Native(a,dict(x))
   if '__sequence__'in x:
    need(set(x)=={'__sequence__','values'}and x['__sequence__']in ('tuple','list'),'native_sequence_schema')
    return [decode(v)for v in x['values']]
   return {k:decode(v)for k,v in x.items()}
  need(x is None or type(x)in (str,int,float,bool),'native_scalar_schema')
  if type(x)is float:need(np.isfinite(x),'finite_native_metadata_scalar')
  return x
 need(set(tree)=={'metadata','arrays'},'CONF_codec_metadata_arrays_not_old_values')
 tree=decode(tree);need(seen==set(values)and token(p)==before,'no_orphan_or_mutated_payload')
 return Record(match[1],tree['metadata'],tree['arrays'],dict(ref),total,synthetic)

def field(record,path,shape,dtype):
 a=record.field(*path);need(a.shape==shape and a.dtype==dtype,'native_ABI_'+'.'.join(path));return a
def equal(a,b):return a.identity()==b.identity()and a.storage.tobytes()==b.storage.tobytes()
def val(record,path,shape,dtype):return field(record,path,shape,dtype).numeric()

@dataclass
class Pair:
 root_id:str
 query_index:int
 parent_condition:str
 pg:dict
 ae:dict
 binding:Record
 n:int
 anchor:bool
 common_valid:np.ndarray
 source_identity:str
 native_bytes:int
 clock:Record
 synthetic:bool=False

def bind_pair(records,*,root_id,query_index,parent_condition):
 """Receipt/role/input joins first. Never adopt neighboring files as a pair."""
 need(query_index in range(11,23)and parent_condition in ('C','T'),'predeclared_root_query_population')
 records=list(records);need(sum(r.native_bytes for r in records)<=QUERY_NATIVE_CAP,'one_query_native_storage_cap')
 need(len({r.synthetic for r in records})==1,'no_mixed_synthetic_real_records')
 pg={};ae={};bindings=[];clocks=[]
 for r in records:
  if r.event not in ('QUERY_PG','CALL_OUTPUT','CURRENT_GPU_INPUT_BINDING','QUERY_REACHED'):continue
  m=r.metadata;need(m['root_id']==root_id and m['query_index']==query_index,'each_member_actual_root_query')
  if r.event=='CURRENT_GPU_INPUT_BINDING':bindings.append(r);continue
  if r.event=='QUERY_REACHED':clocks.append(r);continue
  role=m.get('role')
  if role not in ('record_C','record_T'):continue
  side=role[-1];target=pg if r.event=='QUERY_PG'else ae
  need(side not in target and m['kind']=='record'and m['receiver']==side,'unique_actual_recorder_condition');target[side]=r
 need(set(pg)==set(ae)=={'C','T'}and len(bindings)==len(clocks)==1,'complete_same_query_C_T_PG_AE_binding_and_clock')
 clock=clocks[0];need(clock.metadata['env_step_index']==5*query_index and clock.metadata['absolute_env_step_index']==10+5*query_index,'actual_query_physical_clock_indices')
 qt=clock.field('query_time_actual');need(qt.shape==()and qt.dtype in ('float32','float64'),'actual_query_clock_scalar')
 binding=bindings[0];source=binding.metadata['source'];need(source['same_state_pair']is True and source['root_id']==root_id and source['query_index']==query_index and source['env_step_index']==5*query_index and source['parent_condition']==source['restored_parent_condition']==parent_condition,'actual_same_state_parent_role')
 need(all(isinstance(binding.metadata[s],str)and re.fullmatch('[0-9a-f]{64}',binding.metadata[s])for s in ('C','T')),'actual_encoded_branch_hashes')
 noise=field(binding,('noise',),(1,10,32),'float32');field(binding,('consumed_decode_state',),(1,32),'float32')
 n=pg['C'].field('pg','hidden_input').shape[2];need(768<n<=968,'bounded_actual_prefix_capacity')
 anchor=query_index in ANCHORS;valid=[]
 for side in ('C','T'):
  p,a=pg[side],ae[side];m=a.metadata
  need(same_ref(m['PG_ref'],p.ref)and m['same_cache_verified']is True and m['input_source']==source['source_identity'],'exact_PG_reference_and_actual_input_source')
  need(set(p.arrays['pg'])==set(PG_FLOAT)|{'selected_layers','positions','mask','query_defined','key_visible'},'exact_PG_recorded_fields')
  for k,shape in [('hidden_input',(3,1,n,2048)),('hidden_pre_attention_norm',(3,1,n,2048)),('cache_k',(3,1,n,1,256)),('cache_v',(3,1,n,1,256))]:field(p,('pg',k),shape,'bfloat16')
  need(np.array_equal(val(p,('pg','selected_layers'),(3,),'int32'),LAYERS),'actual_PG_layer_axis')
  mask=val(a,('context','prefix_mask'),(1,n),'bool');pos=field(a,('context','prefix_positions'),(1,n),'int32');ar=val(a,('context','prefix_ar_mask'),(n,),'bool')
  need(equal(pos,field(p,('pg','positions'),(1,n),'int32'))and np.array_equal(pos.numeric(),np.cumsum(mask,axis=-1,dtype=np.int32)-1),'actual_cumulative_R_positions')
  block=np.cumsum(ar);expected=(block[None,:,None]>=block[None,None,:])&mask[:,:,None]&mask[:,None,:]
  pm=val(p,('pg','mask'),(1,n,n),'bool');qd=val(p,('pg','query_defined'),(1,n),'bool');kv=val(p,('pg','key_visible'),(1,n),'bool')
  need(np.array_equal(pm,expected)and np.array_equal(qd,pm.any(-1))and np.array_equal(kv,pm.any(-2)),'actual_PG_mask_support')
  need(np.array_equal(val(a,('context','capture_calls'),(3,),'int32'),CALLS)and np.array_equal(val(a,('context','capture_layers'),(3,),'int32'),LAYERS),'actual_R_capture_axes')
  need(val(a,('context','capture_filled'),(3,),'bool').all()and int(val(a,('context','completed_updates'),(),'int32'))==10,'actual_R_capture_complete')
  gates=a.arrays['context']['gates'];required={'prefix_mask_identical','prefix_positions_identical','selected_tokens_valid','schedule_finished_at_ten','num_steps_is_ten','schedule_terminal_time','target_call_reached','inputs_finite','cache_finite','final_finite'}
  need(set(gates)==required and all(isinstance(v,Native)and v.shape==()and v.dtype=='bool'and bool(v.numeric())for v in gates.values()),'actual_saved_R_structural_gates')
  for name in AE_BASE:field(a,('operands',name),(3,3,1,10,1024),'bfloat16')
  field(a,('raw32',),(1,10,32),'float32');field(a,('adapter7',),(1,10,7),'float64')
  if anchor:
   shapes={'query_post_rope_scaled':((3,3,1,10,1,8,256),'bfloat16'),'probabilities_native':((3,3,1,1,8,10,n+10),'bfloat16'),
    'attention_gate':((3,3,1,1,1024),'bfloat16'),'suffix_key_post_rope':((3,3,1,10,1,256),'bfloat16'),'suffix_value':((3,3,1,10,1,256),'bfloat16'),
    'mask':((3,1,10,n+10),'bool'),'suffix_positions':((3,1,10),'int32'),'query_defined':((3,1,10),'bool'),
    'adarms_condition':((3,1,1024),'float32'),'final_readout':((3,1,10,1024),'bfloat16'),'velocity':((3,1,10,32),'bfloat16'),
    'actual_x':((3,1,10,32),'float32'),'actual_time':((3,),'float32'),'time_weak_type':((3,),'bool')}
   need(set(a.arrays['operands'])==set(AE_BASE)|set(shapes),'complete_anchor_R_fields')
   for name,(shape,dtype)in shapes.items():field(a,('operands',name),shape,dtype)
   need(np.array_equal(a.field('operands','actual_x').numeric()[0],noise.numeric()),'recorder_initial_x_is_actual_diagnostic_noise')
   am=a.field('operands','mask').numeric();need(np.array_equal(am[...,:n],np.broadcast_to(mask,(3,1,10,n)))and am[...,n:].all(),'actual_R_full_key_mask')
  else:need(set(a.arrays['operands'])==set(AE_BASE),'nonanchor_cropped_fields_not_invented')
  valid.append(qd[0]&kv[0])
 # Equality is a pairing premise, not an intersection-based exclusion rule.
 for name in ('prefix_mask','prefix_positions','prefix_ar_mask'):
  need(equal(ae['C'].field('context',name),ae['T'].field('context',name)),'paired_actual_'+name)
 for name in ('mask','positions','query_defined','key_visible','selected_layers'):
  need(equal(pg['C'].field('pg',name),pg['T'].field('pg',name)),'paired_actual_PG_'+name)
 need(np.array_equal(valid[0],valid[1]),'paired_actual_token_support')
 return Pair(root_id,query_index,parent_condition,pg,ae,binding,n,anchor,valid[0],source['source_identity'],sum(r.native_bytes for r in records),clock,records[0].synthetic)
