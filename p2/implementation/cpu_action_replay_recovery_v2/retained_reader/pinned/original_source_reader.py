"""Consume exact restored original members, no auth/network/model operations."""
import copy,hashlib,importlib.util,json,sys
from pathlib import Path
import numpy as np

def need(ok,code):
    if not ok:raise ValueError(code)
def file_id(p):
    p=Path(p);need(p.is_file()and not p.is_symlink(),'regular_restored_source');h=hashlib.sha256()
    with p.open('rb')as f:
        for b in iter(lambda:f.read(1024**2),b''):h.update(b)
    return {'bytes':p.stat().st_size,'sha256':h.hexdigest()}
def load(name,path,ref):
    need(file_id(path)=={k:ref[k]for k in('bytes','sha256')},'original_reader_source_SHA')
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
def materialize(value):
    if hasattr(value,'storage')and hasattr(value,'descriptor'):return value.storage()
    if isinstance(value,dict):return {k:materialize(v)for k,v in value.items()}
    if isinstance(value,list):return [materialize(v)for v in value]
    return value

class RootRecords:
    def __init__(self,row,*,local_paths,readout,rng_module,synthetic=False):
        self.row=row;self.root_id=row['root_id'];self.records={};self.initial=[];self.rng_module=rng_module
        prefix=row['binding']['run_id'];base='redacted-2ac5a54c7015'+prefix+'/'
        def local(ref):
            p=Path(local_paths[ref['uri']]);need(file_id(p)=={k:ref[k]for k in('bytes','sha256')},'restored_original_file_SHA');return p
        for group in row['archives']:
            refs={r['path']:{**r,'uri':group['prefix_uri']+r['path'],'object_key':(group['prefix_uri']+r['path'])[len(base):]}for r in group['members']}
            controls={k:{**group[k],'object_key':group[k]['uri'][len(base):]}for k in('ready','ack')}
            def restore(ref):
                selected=refs.get(ref['path']);need(selected is not None and all(selected[k]==ref[k]for k in('bytes','sha256','object_key')),'only_explicit_original_member');return local(selected)
            archive=readout.ArchiveQuery(local(controls['ready']).read_bytes(),local(controls['ack']).read_bytes(),ready_ref=controls['ready'],ack_ref=controls['ack'],
                expected_binding=row['binding'],restore_exact=restore,synthetic=synthetic)
            actual={r['path']:r for r in archive.members};decoded=[]
            for name,ref in refs.items():
                need(name in actual and all(actual[name][k]==ref[k]for k in('bytes','sha256','object_key')),'actual_READY_ACK_selected_member')
                record=readout.read_member(local(ref),actual[name],synthetic=synthetic)
                need(record.event==ref['event'],'decoded_selected_event')
                if 'required_original_name'in ref:need(record.metadata.get('original_name')==ref['required_original_name'],'decoded_original_parent_input_role')
                if 'required_role'in ref:need(record.metadata.get('role')==ref['required_role'],'decoded_original_parent_input_role')
                decoded.append(record)
            if group['query_index']is None:self.initial.extend(decoded)
            else:self.records[group['query_index']]=decoded
        need(set(self.records)==set(range(23)),'all23_original_query_groups')
    def one(self,q,event):
        rows=[r for r in self.records[q]if r.event==event];need(len(rows)==1,'one_original_'+event);return rows[0]
    def parent_actions(self,q):
        r=self.one(q,'PARENT_ACTIONS');need(r.metadata['query_index']==q and r.metadata.get('proposed_first5_only')is True,'original_parent_action_role')
        return materialize(r.values['adapter7'])
    def parent_raw32(self,q):
        r=self.one(q,'PARENT_ACTIONS');need(r.metadata['query_index']==q,'original_parent_raw_query')
        return materialize(r.values['raw32'])
    def issued(self,q,slot):
        rows=[r for r in self.records[q]if r.event=='ACTUAL_ISSUED_STEP'and r.metadata['action_position']==slot];need(len(rows)==1,'one_issued_original_slot')
        r=rows[0];return {'metadata':r.metadata,'values':materialize(r.values),'source_receipt':r.receipt}
    def outcome(self,q,slot):
        rows=[r for r in self.records[q]if r.event=='ACTUAL_STEP_OUTCOME'and r.metadata['action_position']==slot];need(len(rows)==1,'one_original_step_outcome');return rows[0].metadata
    def diagnostic(self,q):
        r=self.one(q,'DIAGNOSTIC_FINISHED');need(r.metadata['query_index']==q,'original_diagnostic_query');return r.metadata['protected_state']
    def parent_audit(self,q):
        r=self.one(q,'PRODUCER_JSON');need(r.metadata['original_name'].startswith('parent/neural-calls/rng-call-')
             and r.metadata['original_name'].endswith('.json'),'original_parent_rng_audit_role')
        return r.metadata['payload']
    def anchor(self,q):
        r=self.one(q,'PRODUCER_ARRAYS');need(r.metadata=={'original_name':f'parent/query-{q:04d}-input.npz','role':'actual_parent_wrapper_input'},'exact_original_parent_input_payload')
        values=materialize(r.values);pair=self.one(q,'PAIRED_INPUT');clock=self.one(q,'QUERY_REACHED');sensor=self.one(q,'SENSOR_PROVENANCE')
        need(pair.metadata['query_index']==clock.metadata['query_index']==sensor.metadata['query_index']==q
             and clock.metadata['env_step_index']==5*q and clock.metadata['absolute_env_step_index']==10+5*q and sensor.metadata['accepted']is True,'original_anchor_clock_provenance')
        obs={k:v for k,v in values.items()if k not in('physical_sim_state','noise')and not k.startswith('rng_')}
        return {'physical_sim_state':values['physical_sim_state'],'observation':obs,'rng_payload':values,
                'source_receipts':{'input':r.receipt,'pair':pair.receipt,'clock':clock.receipt,'sensor':sensor.receipt}}
    def initial_rng(self):
        v=self.anchor(0)['rng_payload']
        names=('rng_numpy_algorithm','rng_numpy_keys','rng_numpy_position','rng_numpy_has_gauss','rng_numpy_cached_gaussian',
               'rng_python_version','rng_python_internal','rng_python_gaussian_present','rng_python_gaussian')
        need(all(k in v for k in names),'full_q0_global_RNG_payload_required')
        n=(str(np.asarray(v['rng_numpy_algorithm']).item()),np.asarray(v['rng_numpy_keys']).copy(),int(v['rng_numpy_position']),int(v['rng_numpy_has_gauss']),float(v['rng_numpy_cached_gaussian']))
        py=(int(v['rng_python_version']),tuple(int(x)for x in np.asarray(v['rng_python_internal'])),float(v['rng_python_gaussian'])if bool(v['rng_python_gaussian_present'])else None)
        return {'numpy':n,'python':py}
    def rng_id(self,state):return self.rng_module.rng_id(state)
