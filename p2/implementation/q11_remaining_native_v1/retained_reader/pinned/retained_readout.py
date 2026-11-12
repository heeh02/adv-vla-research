"""Bounded, credential-free continuous archive readout core.

No model, environment, network, fitting, plotting or command-line execution.
An authorized cloud caller supplies exact restored members and root receipts.
Synthetic tests must opt in and can never produce a real-data result label.
"""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import sys
import zipfile
import numpy as np

MEMBER_CAP=64*1024**2
QUERY_CAP=512*1024**2
JSON_CAP=4*1024**2
LAYERS=(0,8,17)
CALLS=(0,5,9)
ROLES=(("diagnostic_fixed",0),("diagnostic_fixed",1),("parent_actual",0))
CONDITIONS=("clean","nativeperturb")
PG_FIELDS=("hidden_input","hidden_pre_attention_norm","cache_k","cache_v")
AE_FIELDS=("receiver_residual_input","hidden_post_block","query_post_rope_scaled","attention_output_post_WO_pre_gate","suffix_value")


class ReadoutError(ValueError):pass
def need(ok,code):
    if not ok:raise ReadoutError(code)
def sha(raw):return hashlib.sha256(raw).hexdigest()
def encoded(value):return (json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False)+"\n").encode()
def relative(value):
    need(isinstance(value,str)and value and "\\"not in value,"relative_member_path")
    p=PurePosixPath(value)
    need(not p.is_absolute()and str(p)==value and all(x not in("",".","..")for x in value.split("/")),"canonical_member_path")
    return value
def token(path):
    s=path.stat();return (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
def verified_local(path,ref,*,synthetic=False):
    need(synthetic or sys.platform.startswith("linux"),"real_neural_readout_requires_cloud_Linux")
    path=Path(path);need(path.is_file()and not path.is_symlink(),"regular_restored_member")
    need(type(ref["bytes"])is int and 0<ref["bytes"]<=MEMBER_CAP,"bounded_member_bytes")
    before=token(path);need(before[2]==ref["bytes"],"restored_member_size")
    h=hashlib.sha256()
    with path.open("rb")as stream:
        for block in iter(lambda:stream.read(1024**2),b""):h.update(block)
    need(token(path)==before and h.hexdigest()==ref["sha256"],"restored_member_SHA_or_mutation")
    return path,before


@dataclass(frozen=True)
class NativeArray:
    """Lazy typed payload. BF16 storage is uint16; no rounding is introduced."""
    path:Path
    file_token:tuple
    descriptor:dict

    @property
    def shape(self):return tuple(self.descriptor["shape"])
    @property
    def dtype(self):return self.descriptor["logical_dtype"]
    @property
    def identity(self):
        return {"shape":list(self.shape),"dtype":self.dtype,"sha256":self.descriptor["sha256"]}
    def storage(self):
        need(token(self.path)==self.file_token,"restored_file_changed_before_array")
        with np.load(self.path,allow_pickle=False)as z:a=np.array(z[self.descriptor["array"]],copy=True)
        need(token(self.path)==self.file_token,"restored_file_changed_during_array")
        need(not a.dtype.hasobject and a.shape==self.shape,"native_array_shape_or_object")
        need(a.dtype==np.dtype("uint16")if self.dtype=="bfloat16"else a.dtype.name==self.dtype,"native_storage_dtype")
        need(sha(a.tobytes(order="C"))==self.descriptor["sha256"],"native_array_payload_SHA")
        return a
    def numeric(self):
        a=self.storage()
        if self.dtype=="bfloat16":return (a.astype(np.uint32)<<np.uint32(16)).view(np.float32)
        need(a.dtype.kind in "biufc","numeric_field_required")
        return a


def _decode_tree(tree,path,file_token,keys,seen):
    if isinstance(tree,dict):
        if set(tree)=={"array","shape","logical_dtype","sha256"}:
            key=tree["array"];need(key in keys and key not in seen,"unique_array_descriptor")
            need(isinstance(tree["shape"],list)and all(type(n)is int and n>=0 for n in tree["shape"]),"array_shape_descriptor")
            need(isinstance(tree["logical_dtype"],str)and re.fullmatch(r"[0-9a-f]{64}",tree["sha256"]),"typed_array_descriptor")
            seen.add(key);return NativeArray(path,file_token,tree)
        return {k:_decode_tree(v,path,file_token,keys,seen)for k,v in tree.items()}
    if isinstance(tree,list):return [_decode_tree(v,path,file_token,keys,seen)for v in tree]
    need(tree is None or type(tree)in(str,int,float,bool),"ledger_JSON_value")
    return tree


@dataclass(frozen=True)
class Record:
    event:str
    metadata:dict
    values:dict
    member:dict
    @property
    def receipt(self):return "sha256:"+self.member["sha256"]


def read_member(path,ref,*,synthetic=False):
    """Whole-file SHA first; lazy arrays individually verified when consumed.

    SENSOR JSONL has a different schema and is intentionally not a ledger event.
    """
    path,before=verified_local(path,ref,synthetic=synthetic)
    name=PurePosixPath(ref["path"]).name
    match=re.fullmatch(r"[0-9]+-([A-Z0-9_]+)\.(npz|json)",name)
    need(match is not None,"ledger_event_filename_required")
    if match[2]=="npz":
        with zipfile.ZipFile(path)as z:
            infos=z.infolist();names=[i.filename for i in infos]
            need(len(names)==len(set(names))and len(names)<=4096,"unique_bounded_NPZ_members")
            need(sum(i.file_size for i in infos)<=MEMBER_CAP,"uncompressed_NPZ_cap")
            need(all(i.filename.endswith(".npy")and not(i.flag_bits&1)for i in infos),"NPZ_arrays_only")
        with np.load(path,allow_pickle=False)as z:
            need("__schema__"in z.files,"ledger_schema_array")
            a=z["__schema__"];need(a.dtype==np.uint8 and a.ndim==1 and a.size<=JSON_CAP,"bounded_schema_uint8")
            tree=json.loads(a.tobytes());keys=set(z.files)-{"__schema__"}
        seen=set();tree=_decode_tree(tree,path,before,keys,seen)
        need(seen==keys,"no_orphan_NPZ_arrays")
    else:
        need(ref["bytes"]<=JSON_CAP,"small_ledger_JSON")
        tree=json.loads(path.read_bytes())
        tree=_decode_tree(tree,path,before,set(),set())
    need(token(path)==before and set(tree)=={"metadata","values"},"immutable_ledger_tree_schema")
    need(isinstance(tree["metadata"],dict)and isinstance(tree["values"],dict),"metadata_and_values_objects")
    return Record(match[1],tree["metadata"],tree["values"],dict(ref))


class ArchiveQuery:
    """One exact remote READY/ACK pair. Restorer supplies paths, not credentials."""
    def __init__(self,ready_raw,ack_raw,*,ready_ref,ack_ref,expected_binding,restore_exact,synthetic=False):
        need(callable(restore_exact),"exact_restorer_callback")
        for raw,ref in ((ready_raw,ready_ref),(ack_raw,ack_ref)):
            need(0<len(raw)<=JSON_CAP and len(raw)==ref["bytes"]and sha(raw)==ref["sha256"],"control_object_identity")
        ready,ack=json.loads(ready_raw),json.loads(ack_raw)
        need(ready_raw==encoded(ready)and ack_raw==encoded(ack),"canonical_broker_JSON")
        need(ready.get("schema")=="continuous-query-ready-v1"and ack.get("schema")=="continuous-query-ack-v1"
             and ack.get("status")=="VERIFIED","actual_READY_ACK_schema")
        need(ready["binding"]==ack["binding"]==expected_binding,"run_root_source_request_plan_binding")
        for k in("sequence","nonce","query_index","role"):need(ready[k]==ack[k],"ACK_manifest_identity")
        need(ack["ready_sha256"]==sha(ready_raw),"ACK_READY_SHA")
        need(ready["role"]in("complete","partial")and(type(ready["query_index"])is int and ready["query_index"]>=0
             or ready["role"]=="partial"and ready["query_index"]is None),"actual_query_or_partial_role")
        qi=ready["query_index"];root="queries/"+expected_binding["root_id"]+"/"+("initialization"if qi is None else f"query-{qi:04d}")+"/"+ready["nonce"]
        need(ready_ref["object_key"]==root+"/READY-v1.json"and ack_ref["object_key"]==root+"/ACK-v1.json","exact_remote_control_keys")
        need(ack["manifest_key"]==ready_ref["object_key"]and ack["manifest_verification"]=={"method":"fresh_GET","bytes":len(ready_raw),"sha256":sha(ready_raw)},"manifest_fresh_GET_proof")
        rows=ready["members"];need(isinstance(rows,list)and 0<len(rows)<=4096 and len(rows)==len(ack["members"]),"complete_archive_member_count")
        need(len({r["path"]for r in rows})==len(rows)and sum(r["bytes"]for r in rows)==ready["total_bytes"]<=QUERY_CAP,"unique_bounded_query")
        self.members=[]
        for a,b in zip(rows,ack["members"]):
            relative(a["path"]);need(set(a)=={"path","bytes","sha256"},"exact_member_descriptor")
            need(type(a["bytes"])is int and 0<a["bytes"]<=MEMBER_CAP,"member_bound")
            need(all(b[k]==a[k]for k in a)and b["object_key"]==root+"/"+a["path"],"ACK_member_binding")
            need(b["verification"]=={"method":"fresh_GET","bytes":a["bytes"],"sha256":a["sha256"]},"member_fresh_GET_proof")
            self.members.append({**a,"object_key":b["object_key"]})
        self.ready,self.binding,self.restore,self.synthetic=ready,dict(expected_binding),restore_exact,synthetic
        self._records={}
    def records(self,events=None):
        for ref in self.members:
            m=re.fullmatch(r"[0-9]+-([A-Z0-9_]+)\.(npz|json)",PurePosixPath(ref["path"]).name)
            if m is None or events is not None and m[1]not in events:continue
            if ref["path"]not in self._records:
                self._records[ref["path"]]=read_member(self.restore(dict(ref)),ref,synthetic=self.synthetic)
            yield self._records[ref["path"]]


def exact_array(a,b):
    need(isinstance(a,NativeArray)and isinstance(b,NativeArray),"typed_array_operands")
    return a.identity==b.identity and a.storage().tobytes()==b.storage().tobytes()
def scalar(a):
    need(isinstance(a,NativeArray)and a.shape==(),"actual_scalar")
    return a.numeric().item()


@dataclass(frozen=True)
class NaturalRecord:
    record:Record
    pg_record:Record
    @property
    def meta(self):return self.record.values["metadata"]
    @property
    def pg(self):return self.pg_record.values["pg"]
    def ae(self,field,j):
        need(j in CALLS,"captured_generation_call")
        aliases=self.meta["shared_AE_prefix_fields"]
        if field in("prefix_key_used","prefix_value_used"):
            # Broadcast by reference only; do not materialize three repeated KV arrays.
            return self.pg[aliases[field]].numeric()
        return self.meta["ae"][field].numeric()[CALLS.index(j)]


def resolve_natural(records,*,root_id,query_index):
    """Exact same-query/condition receipt resolution; never a nearest match."""
    records=list(records);pg={};naturals={}
    for r in records:
        if r.event=="QUERY_PG":
            need(r.receipt not in pg,"unique_PG_receipt")
            need(r.metadata["root_trajectory_id"]==root_id and r.metadata["query_index"]==query_index,"PG_query_root_identity")
            pg[r.receipt]=r
    for r in records:
        if r.event!="NATURAL_RECORD":continue
        m=r.metadata;need(m["root_trajectory_id"]==root_id and m["query_index"]==query_index,"natural_query_root_identity")
        key=(m["noise_role"],m["noise_label"],m["condition"])
        need(key[:2]in ROLES and key[2]in CONDITIONS and key not in naturals,"distinct_three_noise_roles_and_conditions")
        meta=r.values["metadata"];ref=meta["shared_prefix_ref"]
        need(ref["receipt"]in pg,"missing_exact_PG_reference")
        p=pg[ref["receipt"]]
        need(p.metadata["condition"]==m["condition"],"cross_condition_PG_reference_forbidden")
        need(ref["identities"]==p.metadata["identities"],"PG_reference_identity_table")
        need(meta["pg"]=={}and meta["retention"]["PG_cross_noise_deduplicated"]is True,"actual_physical_PG_dedup_schema")
        need(meta["shared_AE_prefix_fields"]=={"prefix_key_used":"cache_k","prefix_value_used":"cache_v","generation_slots":[0,5,9],"same_typed_values_verified":True},"exact_shared_AE_prefix_map")
        need("prefix_key_used"not in meta["ae"]and"prefix_value_used"not in meta["ae"],"no_duplicate_AE_prefix_arrays")
        for name,a in p.values["pg"].items():
            need(isinstance(a,NativeArray)and a.identity==ref["identities"][name],"PG_payload_typed_descriptor")
        context=meta["context"]
        need(np.array_equal(context["capture_calls"].numeric(),CALLS)and np.array_equal(context["capture_layers"].numeric(),LAYERS),"actual_capture_axes")
        need(int(scalar(context["completed_updates"]))==10,"actual_ten_flow_updates")
        dense=meta["dense_P"]
        need(len(dense)==9 and {(d["layer_index"],d["generation_call"])for d in dense}=={(l,j)for l in LAYERS for j in CALLS},"complete_A_dense_P_axes")
        naturals[key]=NaturalRecord(r,p)
    return naturals


def validate_native_layout(value):
    """Read ABI descriptors, then validate requested payloads lazily on access."""
    p=value.pg;need(set(PG_FIELDS)<=set(p),"all_PG_fields_present")
    n=p["cache_k"].shape[2]
    for field,shape in {"hidden_input":(3,1,n,2048),"hidden_pre_attention_norm":(3,1,n,2048),
                        "cache_k":(3,1,n,1,256),"cache_v":(3,1,n,1,256)}.items():
        need(p[field].shape==shape and p[field].dtype=="bfloat16","PG_native_shape_dtype:"+field)
    for field in("query_defined","key_visible"):
        need(p[field].shape==(1,n)and p[field].dtype=="bool","PG_native_token_mask:"+field)
    shapes={"receiver_residual_input":(3,3,1,10,1024),"hidden_post_block":(3,3,1,10,1024),
        "query_post_rope_scaled":(3,3,1,10,1,8,256),"attention_output_post_WO_pre_gate":(3,3,1,10,1024),
        "attention_gate":(3,3,1,1,1024),"suffix_key_post_rope":(3,3,1,10,1,256),"suffix_value":(3,3,1,10,1,256)}
    for field,shape in shapes.items():
        a=value.meta["ae"][field];need(a.shape==shape and a.dtype=="bfloat16","AE_native_shape_dtype:"+field)
    for field,shape,dtype in (("actual_x",(3,1,10,32),"float32"),("actual_time",(3,),"float32"),
        ("mask",(3,1,10,n+10),"bool"),("query_defined",(3,1,10),"bool"),
        ("suffix_positions",(3,1,10),"int32"),("adarms_condition",(3,1,1024),"float32"),
        ("velocity",(3,1,10,32),"bfloat16"),("final_readout",(3,1,10,1024),"bfloat16")):
        a=value.meta["ae"][field];need(a.shape==shape and a.dtype==dtype,"context_native_shape_dtype:"+field)
    for row in value.meta["dense_P"]:
        a=row["probabilities_native"];need(a.shape==(1,1,8,10,n+10)and a.dtype=="bfloat16","native_all_heads_dense_P")
    for field,shape,dtype in (("raw32",(1,10,32),"float32"),("adapter7",(1,10,7),"float64")):
        a=value.record.values[field];need(a.shape==shape and a.dtype==dtype,"native_endpoint:"+field)
    return {"native_descriptors_valid":True,"payloads_verified_only_when_accessed":True,"prefix_tokens":n}


def pair_coverage(records,*,root_id,query_index):
    records=list(records);natural=resolve_natural(records,root_id=root_id,query_index=query_index)
    audits={};parent=next((r for r in records if r.event=="PARENT_ACTIONS"and r.metadata["query_index"]==query_index),None)
    for r in records:
        if r.event=="NEURAL_RNG_AUDIT"and r.metadata.get("kind")=="pg_ae":
            m=r.metadata;key=(m["noise_role"],m["noise_label"],m["condition"])
            audits.setdefault(key,[]).append(m["audit"]["explicit_model_noise_identity"])
    rows=[]
    for role,label in ROLES:
        a,b=(natural.get((role,label,c))for c in CONDITIONS)
        row={"root_id":root_id,"query_index":query_index,"noise_role":role,"noise_label":label,
             "clean_recorded":a is not None,"treated_recorded":b is not None,"paired":a is not None and b is not None,
             "initial_x_typed_equal":None,"explicit_noise_audit_equal":None,"parent_noise_typed_equal":None,
             "later_x_fixed":False,"pair_role":"same_physical_query_two_freely_generated_paths"}
        if a and b:
            aa=a.meta["ae"]["actual_x"];bb=b.meta["ae"]["actual_x"]
            av,bv=aa.numeric()[0],bb.numeric()[0]
            row["initial_x_typed_equal"]=aa.dtype==bb.dtype and av.shape==bv.shape and av.tobytes()==bv.tobytes()
            ia,ib=audits.get((role,label,"clean"),[]),audits.get((role,label,"nativeperturb"),[])
            row["explicit_noise_audit_equal"]=bool(ia and ib and all(x==ia[0]for x in ia+ib))if ia or ib else None
            if role=="parent_actual"and parent:
                pn=parent.values["noise"].numeric();row["parent_noise_typed_equal"]=aa.dtype==parent.values["noise"].dtype and av.shape==pn.shape and av.tobytes()==pn.tobytes()
        rows.append(row)
    return rows,natural


def planned_coverage(worker_result,*,root_id,planned_query_indices):
    """Root denominator comes from a separately exact-read WORKER-CLOSED object.

    Do not infer completion from a READY role='complete' (archive completeness).
    """
    need(worker_result.get("schema")=="continuous-native-root-result-v1"and worker_result.get("root_id")==root_id,"actual_root_result_identity")
    table=worker_result.get("query_table",{});rows=table.get("rows",[])if isinstance(table,dict)else table
    by={r["planned_query_index"]:r for r in rows}
    need(len(by)==len(rows)and set(by)<=set(planned_query_indices),"unique_frozen_query_denominator")
    output=[]
    for q in planned_query_indices:
        r=by.get(q);output.append({"root_id":root_id,"query_index":q,"planned_env_step_index":5*q,
            "pair_status":"unknown_missing_root_row"if r is None else r.get("status","unknown"),
            "neural_status":None if r is None else r.get("neural_status"),
            "reason":None if r is None else r.get("reason"),"observed_success":worker_result.get("observed_success"),
            "root_complete":worker_result.get("root_complete")is True,"root_terminal_reason":worker_result.get("root_terminal_reason"),
            "statistical_unit":"one_parent_root_correlated_query"})
    return output


def query_clocks(records,*,root_id,query_index):
    rows=[]
    for r in records:
        if r.event=="PAIRED_INPUT"and r.metadata.get("query_index")==query_index:
            m=r.metadata
            sensors={}
            for condition,view in r.values["views"].items():
                sensors[condition]={"sensor_env_step_index":view["sensor_env_step_index"],
                    "camera_times":{camera:scalar(value)for camera,value in view["sensor_time_actual"].items()}}
            rows.append({"root_id":root_id,"query_index":query_index,"env_step_index":m.get("env_step_index"),
                "absolute_env_step_index":m.get("absolute_env_step_index"),"query_time_actual":scalar(r.values["query_time_actual"]),
                "query_physical_snapshot_id":m.get("query_physical_snapshot_id"),"sensor_source_physical_snapshot_id":m.get("sensor_source_physical_snapshot_id"),
                "declared_runtime_snapshot_id":m.get("declared_runtime_snapshot_id"),"full_runtime_snapshot_id":m.get("full_runtime_snapshot_id"),
                "full_replay_qualified":m.get("full_replay_qualified",False),"pair_receipt":r.receipt,"sensor_clocks":sensors,
                "pair_snapshot_proof_is_neural_qualification":False})
    need(len(rows)<=1,"unique_actual_pair_query_clock")
    return rows


def parent_issued_rows(records,*,root_id,query_index,natural):
    records=list(records);parents=[r for r in records if r.event=="PARENT_ACTIONS"and r.metadata["query_index"]==query_index]
    need(len(parents)<=1,"one_parent_action_source")
    parent=parents[0]if parents else None;trace=natural.get(("parent_actual",0,"clean"));endpoint_checks=None
    if parent and trace:endpoint_checks={k:exact_array(parent.values[k],trace.record.values[k])for k in("raw32","adapter7")}
    issued={};outcomes={}
    for r in records:
        if r.metadata.get("query_index")!=query_index:continue
        if r.event in("ACTUAL_ISSUED_STEP","ACTUAL_STEP_OUTCOME"):
            p=r.metadata["action_position"];need(type(p)is int and 0<=p<5,"actual_first_five_issued_slot")
            target=issued if r.event=="ACTUAL_ISSUED_STEP"else outcomes
            need(p not in target,"unique_issued_or_outcome_slot");target[p]=r
    rows=[]
    for p in range(10):
        event=issued.get(p);outcome=outcomes.get(p);exact=None
        if parent and event:
            a=parent.values["adapter7"].numeric()[0,p];b=event.values["action"].numeric()
            exact=parent.values["adapter7"].dtype==event.values["action"].dtype and a.shape==b.shape and a.tobytes()==b.tobytes()
        def small_value(value):
            return None if value is None else value.numeric().tolist()if isinstance(value,NativeArray)else value
        rows.append({"root_id":root_id,"query_index":query_index,"action_position":p,"planned_for_issuance":p<5,
            "issued":event is not None,"issued_action_matches_parent":exact,"parent_trace_raw32_exact":None if endpoint_checks is None else endpoint_checks["raw32"],
            "parent_trace_adapter7_exact":None if endpoint_checks is None else endpoint_checks["adapter7"],
            "policy_step":None if event is None else event.metadata["policy_step"],"official_success":None if outcome is None else outcome.metadata["official_success"],
            "done":None if outcome is None else outcome.metadata["done"],"outcome_recorded":outcome is not None,
            "parent_command":None if parent is None else parent.values["adapter7"].numeric()[0,p].tolist(),
            "issued_command":None if event is None else small_value(event.values["action"]),
            "returned_eef_before":None if event is None else small_value(event.values["before"].get("returned_eef_pos")),
            "returned_eef_after":None if event is None else small_value(event.values["after"].get("returned_eef_pos")),
            "diagnostic_action_is_issued":False})
    return rows


def signed_summary(clean,treated,reference_delta=None):
    """Pure descriptive full-dimensional readout; no fit or inferred causality.

    Native arrays convert to FP32 for subtraction. Float64 only accumulates host
    descriptive scalars; model forward and archived values are untouched.
    """
    a,b=np.asarray(clean,dtype=np.float32),np.asarray(treated,dtype=np.float32)
    need(a.shape==b.shape and a.size>0 and np.isfinite(a).all()and np.isfinite(b).all(),"finite_same_shape_signed_operands")
    d=(b-a).reshape(-1);av,bv=a.reshape(-1),b.reshape(-1)
    def dot(x,y):return float(np.sum(x.astype(np.float64)*y.astype(np.float64)))
    cn,tn,dn=(np.sqrt(dot(x,x))for x in(av,bv,d))
    refcos=None
    if reference_delta is not None:
        ref=np.asarray(reference_delta,dtype=np.float32).reshape(-1);need(ref.shape==d.shape,"same_coordinates_direction_reference")
        rn=np.sqrt(dot(ref,ref));refcos=dot(d,ref)/(dn*rn)if dn and rn else None
    return {"coordinate_count":int(d.size),"clean_l2":float(cn),"treated_l2":float(tn),"delta_l2":float(dn),
        "delta_rms":float(dn/np.sqrt(d.size)),"delta_mean_signed":float(np.mean(d,dtype=np.float64)),
        "positive_coordinates":int(np.count_nonzero(d>0)),"negative_coordinates":int(np.count_nonzero(d<0)),"zero_coordinates":int(np.count_nonzero(d==0)),
        "delta_over_clean_l2":float(dn/cn)if cn else None,"clean_treated_cosine":dot(av,bv)/(cn*tn)if cn and tn else None,
        "delta_cosine_same_query_fixed0":refcos,"zero_direction_is_undefined":bool(not dn),
        "difference_definition":"native T minus C converted to FP32 before subtraction","causal_claim":False}


def signed_rows(natural,*,root_id,query_index):
    """Optional compact descriptors. No across-query aggregation or prediction."""
    rows=[];reference={}
    for role,label in ROLES:
        pair=[natural.get((role,label,c))for c in CONDITIONS]
        if not all(pair):continue
        c,t=pair
        for j in CALLS:
            for field in AE_FIELDS:
                a,b=c.ae(field,j),t.ae(field,j)
                for li,layer in enumerate(LAYERS):
                    key=(field,layer,j);delta=b[li].astype(np.float32)-a[li].astype(np.float32)
                    if(role,label)==ROLES[0]:reference[key]=delta
                    rows.append({"root_id":root_id,"query_index":query_index,"domain":"AE","field":field,"layer_index":layer,
                        "generation_call":j,"noise_role":role,"noise_label":label,"selection":"all_action_positions_native_coordinates",
                        "actual_time_clean":float(c.meta["ae"]["actual_time"].numeric()[CALLS.index(j)]),
                        "actual_time_treated":float(t.meta["ae"]["actual_time"].numeric()[CALLS.index(j)]),
                        **signed_summary(a[li],b[li],reference.get(key))})
    # PG is stored once per condition/query and has no flow/noise population.
    any_pair=next(([natural.get((r,n,c))for c in CONDITIONS]for r,n in ROLES if all((r,n,c)in natural for c in CONDITIONS)),None)
    if any_pair:
        c,t=any_pair
        for field in PG_FIELDS:
            mask_field="key_visible"if field in("cache_k","cache_v")else"query_defined"
            valid_c=c.pg[mask_field].numeric();valid_t=t.pg[mask_field].numeric()
            need(valid_c.shape==valid_t.shape,"paired_PG_mask_shape")
            common=valid_c.astype(bool)&valid_t.astype(bool)
            if np.any(common):
                a,b=c.pg[field].numeric(),t.pg[field].numeric()
                for li,layer in enumerate(LAYERS):
                    rows.append({"root_id":root_id,"query_index":query_index,"domain":"PG","field":field,"layer_index":layer,
                        "generation_call":None,"noise_role":None,"noise_label":None,"selection":"intersection_of_actual_"+mask_field+"_tokens",
                        "clean_defined_tokens":int(valid_c.sum()),"treated_defined_tokens":int(valid_t.sum()),"selected_tokens":int(common.sum()),
                        **signed_summary(a[li][common],b[li][common])})
            else:
                for layer in LAYERS:
                    rows.append({"root_id":root_id,"query_index":query_index,"domain":"PG","field":field,"layer_index":layer,
                        "generation_call":None,"noise_role":None,"noise_label":None,"selection":"intersection_of_actual_"+mask_field+"_tokens",
                        "clean_defined_tokens":int(valid_c.sum()),"treated_defined_tokens":int(valid_t.sum()),"selected_tokens":0,
                        "coordinate_count":0,"undefined_reason":"no_common_defined_tokens","delta_l2":None,"delta_mean_signed":None,
                        "delta_cosine_same_query_fixed0":None,"causal_claim":False})
    return rows


def read_query_core(archive,*,include_signed_descriptors=False):
    """One-query bounded result for a later authorized cloud analysis runner.

    This does not mark a whole root complete. Use planned_coverage separately on
    the exact WORKER-CLOSED receipt, including early-unreached/failed queries.
    No plot, fitted coordinates, treatment outcome or formal inference is made.
    """
    q=archive.ready["query_index"];need(type(q)is int,"query_core_requires_real_query_index")
    root=archive.binding["root_id"]
    events={"QUERY_PG","NATURAL_RECORD","NEURAL_RNG_AUDIT","PARENT_ACTIONS","PAIRED_INPUT",
        "ACTUAL_ISSUED_STEP","ACTUAL_STEP_OUTCOME","NATURAL_INPUT_COMPARISON","ACTUAL_PARENT_TRACE_ENDPOINT",
        "DIAGNOSTIC_FINISHED","DIAGNOSTIC_FAILED","RECORDING_FAILED"}
    records=list(archive.records(events))
    # Partial archives can also contain receipts from earlier queries; no join
    # by file order or slot name is allowed.
    records=[r for r in records if r.metadata.get("query_index")==q]
    coverage,natural=pair_coverage(records,root_id=root,query_index=q)
    if not archive.synthetic:
        for value in natural.values():validate_native_layout(value)
    finished=[r for r in records if r.event=="DIAGNOSTIC_FINISHED"]
    comparisons={}
    for r in records:
        if r.event=="NATURAL_INPUT_COMPARISON":
            m=r.metadata;key=(m["noise_role"],m["noise_label"],m["condition"])
            need(key not in comparisons,"unique_actual_input_comparison")
            comparisons[key]=m.get("all_comparisons_exact")
    inputs=[]
    for role,label in ROLES:
        for condition in CONDITIONS:
            key=(role,label,condition);value=natural.get(key)
            inputs.append({"root_id":root,"query_index":q,"noise_role":role,"noise_label":label,"condition":condition,
                "record_present":value is not None,"record_receipt":None if value is None else value.record.receipt,
                "PG_receipt":None if value is None else value.pg_record.receipt,
                "first_input_comparison_exact":comparisons.get(key),"query_diagnostic_finished":bool(finished),
                "qualification_pending_at_record_write":None if value is None else value.record.metadata.get("qualification_pending"),
                "receipt_presence_is_qualification":False})
    result={"schema":"continuous-query-readout-core-v1","synthetic":archive.synthetic,
        "status":"SYNTHETIC_SCHEMA_TEST_ONLY"if archive.synthetic else"REAL_ARCHIVE_DESCRIPTIVE_READOUT",
        "binding":archive.binding,"query_index":q,"archive_role":archive.ready["role"],
        "archive_complete_is_neural_complete":False,"input_coverage":inputs,"pair_coverage":coverage,
        "query_clocks":query_clocks(records,root_id=root,query_index=q),
        "parent_issued":parent_issued_rows(records,root_id=root,query_index=q,natural=natural),
        "signed_descriptors":signed_rows(natural,root_id=root,query_index=q)if include_signed_descriptors else[],
        "signed_descriptors_requested":include_signed_descriptors,"basis_or_3D":None,
        "treated_behavior_outcome":None,"independent_query_samples_claimed":False,"mechanism_claim":False,
        "source_members_used":[dict(r.member)for r in records]}
    need(len(encoded(result))<=JSON_CAP,"single_query_small_readout_cap")
    return result
