"""Fixed archived T06-PC/q11/j6/L9 natural-d CPU readout; no model or cloud IO."""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import math
import sys
import time

HERE=Path(__file__).resolve().parent
ROOT='SDRU-CCONF-T06-V00700-I00-PC'
INPUTS_SHA='186da1670aafa12ec370c29678a5a38886258bbea552be3665c333dace7c3d7a'
OLD_MANIFEST_SHA='192b4fa8f1cb8987eadab3a74e24ca5c3e1588546d5df93b10f2e23b996f112e'
QUANT_SHA='17234273673323ede28caf8f5d3d0cb9d964aabaa9f36d424ba2f0239f77c45d'
ROLES=('source_evidence','planned','input','shared_pulse','shared_persistent')
LIMIT=512*1024


def need(ok,code):
    if not ok:raise ValueError(code)


def identity(raw):return {'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
def encode(value):return (json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec)
    sys.modules[name]=m;spec.loader.exec_module(m);return m


def verify_sources(expected):
    raw=(HERE/'SOURCE-MANIFEST.json').read_bytes()
    need(identity(raw)['sha256']==expected,'exact_archived_quant_source_manifest_SHA')
    manifest=json.loads(raw)
    need(manifest['schema']=='B-BRIDGE-archived-natural-d-source-v1','archived_pair_source_schema')
    refs=manifest['files'];need(len(refs)==len({r['path']for r in refs}),'unique_source_roster')
    required={'adapter.py','pair.py','quantization_frozen.py','INPUTS.json',
              *('frozen_precision/'+n for n in ('adapter.py','core.py','native_reader.py','pvp_numeric.py','INPUTS.json','SOURCE-MANIFEST.json'))}
    need(required<={r['path']for r in refs},'complete_frozen_source_closure')
    for ref in refs:
        p=HERE/ref['path']
        need(not p.is_symlink()and p.is_file()and p.resolve().is_relative_to(HERE)and p.stat().st_size<=1024**2,'bounded_source_file')
        need(identity(p.read_bytes())=={k:ref[k]for k in('bytes','sha256')},'source_file_bytes_SHA')
    need(identity((HERE/'quantization_frozen.py').read_bytes())['sha256']==QUANT_SHA,'unchanged_quantization_source')
    return identity(raw)


def load_inputs():
    raw=(HERE/'INPUTS.json').read_bytes();need(identity(raw)['sha256']==INPUTS_SHA,'fixed_five_INPUTS_SHA')
    plan=json.loads(raw);refs=plan['input_objects']
    need(plan['schema']=='B-BRIDGE-archived-natural-d-inputs-v1'and plan['root_id']==ROOT and
         (plan['query_index'],plan['flow_index'],plan['layer_index'])==(11,6,8),'fixed_archived_pair_scope')
    need(tuple(r['role']for r in refs)==ROLES and len({r['path']for r in refs})==5 and
         sum(r['bytes']for r in refs)==6314847,'exact_five_archive_objects')
    need(all('/'not in r['path']and r['path']not in('','.','..')for r in refs),'flat_original_archive_paths')
    return plan


def validate_pair(inp,calls,entries,query,reader,numeric,*,synthetic=False):
    """Frozen PVP pair gates, retaining only the two actual roles under study."""
    import numpy as np
    N,M=reader,numeric
    need(len(calls)==len(entries)==2 and inp.synthetic is synthetic and all(r.synthetic is synthetic for r in calls),'two_same_data_scope_calls')
    need(inp.event=='PVP_INPUT_VERIFIED'and inp.metadata['root_id']==ROOT and inp.metadata['query_index']==11,'original_input_event_identity')
    n=calls[0].field('prefix_mask').shape[-1];need(n==query['prefix_capacity']and 0<n<=968,'actual_prefix_capacity')
    noise=N.field(inp,('noise',),(1,10,32),'float32')
    N.field(inp,('consumed_decode_state',),(1,32),'float32')
    vm=N.field(inp,('Vmod',),(1,n,1,256),'bfloat16');e=N.field(inp,('actual_E',),(1,n,1,256),'float32')
    need(inp.metadata['sources']==query['old_saved_sources'],'fixed_old_raw_source_identity')
    need(calls[0].metadata['input_identity']==calls[1].metadata['input_identity'],'same_encoded_inputs_noise')
    for call,entry,role,cid in zip(calls,entries,('shared_pulse','shared_persistent'),('PVP-097','PVP-098')):
        m=call.metadata
        need(call.event=='PVP_CALL_OUTPUT'and m['graph']=='Gstar'and all(m.get(k)==v for k,v in entry.items()),'actual_fixed_entry_graph')
        need(m['role']==role and m['call_id']==cid and m['root_id']==ROOT and m['query_index']==11 and m['receiver']=='T'and m['donor']=='C','actual_pair_roles')
        need(N.same_ref(m['input_ref'],inp.ref)and m['PG_refs']==calls[0].metadata['PG_refs'],'same_actual_input_PG_refs')
        for key,shape,dtype in (('prefix_mask',(1,n),'bool'),('prefix_positions',(1,n),'int32'),('capture_layers',(3,),'int32'),('capture_calls',(5,),'int32')):
            N.field(call,(key,),shape,dtype)
        need(np.array_equal(call.field('capture_layers').numeric(),M.LAYERS)and np.array_equal(call.field('capture_calls').numeric(),M.FLOWS),'fixed_native_capture_axes')
        for name,shape,dtype in [('x',(11,1,10,32),'float32'),('velocity_native',(10,1,10,32),'bfloat16'),('time',(11,),'float32'),('time_weak_type',(11,),'bool'),('dt',(10,),'float32')]:
            N.field(call,('flow',name),shape,dtype)
        need(M.exact(call.field('flow','x').storage[0],noise.storage),'same_original_noise')
        for name in M.FIELDS:N.field(call,('reads',name),(5,3,1,10,1024),'bfloat16')
        for name,shape in [('query_post_rope_scaled',(5,3,1,10,1,8,256)),('probabilities_native',(5,3,1,1,8,10,n+10)),
                          ('attention_gate',(5,3,1,1,1024)),('suffix_key_post_rope',(5,3,1,10,1,256)),('suffix_value',(5,3,1,10,1,256))]:
            N.field(call,('reads',name),shape,'bfloat16')
        for name,shape,dtype in [('mask',(5,1,10,n+10),'bool'),('suffix_positions',(5,1,10),'int32'),
                               ('query_defined',(5,1,10),'bool'),('adarms_condition',(5,1,1024),'float32')]:
            N.field(call,('reads',name),shape,dtype)
        for name in ('kv_mask','layer_mask','flow_mask','use_vmod'):
            need(np.array_equal(call.field('selectors',name).numeric(),np.asarray(entry[name],bool)),'actual_original_selector')
        need(set(call.arrays['gates'])==M.GATES and all(isinstance(v,N.Native)and v.dtype=='bool'and v.shape==()and bool(v.numeric())for v in call.arrays['gates'].values()),'actual_original_native_gates')
        vref=m['Vmod_actual_identity'];need(vref['shape']==list(vm.shape)and vref['logical_dtype']=='bfloat16'and
             vref['bytes']==vm.storage.nbytes and vref['sha256']==N.sha(vm.storage.tobytes()),'actual_same_Vmod_bytes')
        need(not np.any(e.numeric()[:,~call.field('prefix_mask').numeric()[0]]),'no_E_on_padding')
    p,s=calls
    for name,count in (('x',7),('velocity_native',6),('time',7),('time_weak_type',7),('dt',10)):
        need(M.exact(p.field('flow',name).storage[:count],s.field('flow',name).storage[:count]),'same_pulse_history_x6_'+name)
    need(set(p.arrays['reads'])==set(s.arrays['reads']),'same_original_read_field_roster')
    for name in p.arrays['reads']:
        need(M.exact(p.field('reads',name).storage[0],s.field('reads',name).storage[0]),'same_all_j5_native_reads')
    for name in ('query_post_rope_scaled','probabilities_native','receiver_residual_input','suffix_key_post_rope','suffix_value','attention_gate'):
        need(M.exact(p.field('reads',name).storage[1,:2],s.field('reads',name).storage[1,:2]),'same_j6_L1_L9_receiver_'+name)
    for name in ('mask','suffix_positions','query_defined','adarms_condition'):
        need(M.exact(p.field('reads',name).storage[:2],s.field('reads',name).storage[:2]),'same_j5_j6_context_'+name)
    for name in ('prefix_mask','prefix_positions'):
        need(N.equal(p.field(name),s.field(name)),'same_prefix_layout')
    need(int(p.field('prefix_mask').numeric().sum())==query['valid_prefix_tokens']and
         bool(not np.any(e.numeric()))==query['exactzero_actual_E'],'source_support_and_E_zero_status')
    return {'source_and_model_guard_from_original_PLANNED':True,'two_CALLs_only':True,
            'same_pulse_history_x6_tau_weak_dt':True,'same_j6_L1_L9_Q_P_r_gate_suffixKV':True,
            'new_model_calls':0,'native_product_captured':False}


def run_stage(stage,out,*,source_manifest_sha256,deadline_epoch):
    need(sys.platform.startswith('linux'),'real_native_arrays_cloud_Linux_only')
    need(math.isfinite(deadline_epoch)and deadline_epoch>time.time(),'future_absolute_deadline')
    stage,out=Path(stage),Path(out)
    need(stage.is_dir()and not stage.is_symlink()and not out.exists()and not out.resolve().is_relative_to(stage.resolve()),'new_output_outside_readonly_stage')
    source_ref=verify_sources(source_manifest_sha256);plan=load_inputs()
    result={'schema':'B-BRIDGE-archived-natural-d-quant-v1','analysis_complete':False,'synthetic':False,
            'status':'INPUT_OR_NUMERIC_FAILURE_NOT_SCIENTIFIC_NEGATIVE','source_manifest':source_ref,
            'scope':{k:plan[k]for k in('root_id','query_index','flow_index','layer_index','baseline','treated')},
            'inputs':plan['input_objects'],'input_body_bytes':6314847,'historical_archive_only':True,
            'input_objects':[{k:r[k]for k in('path','uri','bytes','sha256')}for r in plan['input_objects']],
            'new_model_calls':0,'new_env_steps':0,'common_c_evaluated':False,'same_W_setter_qualified':False,
            'new_common_c_generated':False,
            'new_graph_qualification':False,'native_product_captured':False,'scientific_GO':False}
    def check():need(time.time()<deadline_epoch,'archived_pair_absolute_deadline')
    try:
        A=module('_archived_pair_frozen_precision',HERE/'frozen_precision/adapter.py')
        A.verify_sources(OLD_MANIFEST_SHA);original=A.load_inputs();N,M=A.load_helpers()
        original_refs={r['role']:r for r in original['input_objects']}
        need(plan['input_objects']==[original_refs[r]for r in ROLES],'exact_five_original_reference_subset')
        refs={r['role']:r for r in plan['input_objects']}
        source=json.loads(A.read_small(stage/refs['source_evidence']['path'],refs['source_evidence'],65536))
        query=A.select_source_query(source,original)
        planned=N.load(stage/refs['planned']['path'],refs['planned']);entries=A.planned_entries(planned,query)[1:]
        check();inp=N.load(stage/refs['input']['path'],refs['input'])
        calls=[]
        for role in ('shared_pulse','shared_persistent'):
            check();calls.append(N.load(stage/refs[role]['path'],refs[role]))
        proof=validate_pair(inp,calls,entries,query,N,M)
        records=[];fields=[]
        mapping={'W_used':'attention_output_post_WO_pre_gate','gate':'attention_gate',
                 'residual_input':'receiver_residual_input','after_attention_residual':'hidden_after_attention_residual'}
        for role,call in zip(('archived_pulse','archived_persistent'),calls):
            record={};identities={}
            for key,name in mapping.items():
                leaf=call.field('reads',name);record[key]=leaf.storage[1,1,0]
                identities[key]={'full_leaf':leaf.identity(),'slice_axes':[1,1,0,'all_native_positions_channels'],
                                 'slice_shape':list(record[key].shape),'slice_dtype':str(record[key].dtype),
                                 **identity(record[key].tobytes())}
            records.append(record);fields.append({'role':role,'wholefile':call.ref,'fields':identities})
        Q=module('quantization_frozen',HERE/'quantization_frozen.py')
        P=module('_archived_pair_math',HERE/'pair.py')
        result.update(status='ARCHIVED_NATURAL_D_QUANT_COMPLETE',analysis_complete=True,input_qualified=True,
                      qualification=proof,actual_time_FP32=float(calls[0].field('flow','time').numeric()[6]),
                      native_source_leaves=fields,readout=P.analyze_pair(*records,check=check),
                      decoded_native_records=4,baseline_none_CALL_loaded=False,
                      interpretation='Archived Gstar local gated-residual natural-d diagnostic; not common-c action or new setter qualification.')
    except Exception as exc:
        result.update(input_qualified=False,error_type=type(exc).__name__)
    raw=encode(result);need(len(raw)<=LIMIT,'single_archived_pair_JSON_512KiB')
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('xb')as stream:stream.write(raw)
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--stage',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--source-manifest-sha256',required=True);p.add_argument('--deadline-epoch',type=float,required=True)
    a=p.parse_args();r=run_stage(a.stage,a.out,source_manifest_sha256=a.source_manifest_sha256,deadline_epoch=a.deadline_epoch)
    print(json.dumps({k:r[k]for k in('status','analysis_complete','new_model_calls','common_c_evaluated')}))
    return 0 if r['analysis_complete']else 2


if __name__=='__main__':raise SystemExit(main())
