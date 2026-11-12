"""Credential-free continuous science factory; lazy native creation only.

Root alone stages/freeze/submits. No model loader, cloud SDK, credential or
recording fallback exists here. Imports use the parent's exact source manifest.
"""
from contextlib import contextmanager
import copy
import dataclasses
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import sys
import time
import numpy as np

ADAPTER='docs/experiments/analysis-system-20260929/native-write-flow-20261006-v1/libero-plus-policy-adapter-20261009-v2'
LEGACY='docs/experiments/first-paper-evidence-upgrade-20261008-v1/plus-primary-confirmation-20261010-v1'
STUDY='docs/experiments/state-dependent-response-use-20261013-v1'
BASE_FACTORY='docs/experiments/selective-correction-20261012-v1/execution/B/runtime/environment_factory.py'
class NativeError(RuntimeError):pass
def need(ok,code):
    if not ok:raise NativeError(code)
def sha(raw):return hashlib.sha256(raw).hexdigest()
def typed_id(value):
    a=np.asarray(value);need(not a.dtype.hasobject,'native_numeric_identity')
    header={'shape':list(a.shape),'dtype':a.dtype.str,'order':'C'}
    return {**header,'bytes':a.nbytes,'sha256':sha(json.dumps(header,sort_keys=True,separators=(',',':')).encode()+b'\0'+a.tobytes()),
            'hash_domain':'plus-policy-native-array-v1','raw_sha256':sha(a.tobytes())}
def noise(root,label,query=None):
    need(type(label)is int and label in(0,1),'declared_noise_label')
    common=f"{root['component_id']}|{root['root_initial_state_sha256']}|{label}"
    if query is None:preimage='state-response-use-diagnostic-noise-v1|'+common
    else:
        need(type(query)is int and query>=0,'actual_parent_query')
        preimage='state-response-use-parent-noise-v1|'+common+'|'+str(query)
    seed=int.from_bytes(hashlib.sha256(preimage.encode()).digest()[:8],'little')
    return np.random.Generator(np.random.PCG64(seed)).standard_normal((1,10,32)).astype(np.float32)

class DistinctInitialReader:
    """Reuse exact official trusted loader; select original-order typed vectors.

    The bridge returns a single selected vector to the original reset method,
    which indexes [0]. The caller MUST relabel the old init0 source metadata.
    """
    def __init__(self,reader,root,emit):self.reader,self.root,self.emit=reader,root,emit;self.selection=None
    def select(self,*args):
        states,proof=self.reader.select(*args)
        expected=self.root['root_selection']['library']
        need(all(proof[k]==expected[k]for k in ('bytes','sha256')),'actual_frozen_init_library')
        need(proof['path']==expected['path'],'actual_official_init_library_path')
        selected=[];seen=set()
        for index,value in enumerate(states):
            if hasattr(value,'detach'):value=value.detach().cpu().numpy()
            value=np.asarray(value);need(value.ndim==1 and value.dtype.kind=='f'and np.isfinite(value).all(),'native_initial_vector')
            identity=typed_id(value);key=(tuple(identity['shape']),identity['dtype'],identity['raw_sha256'])
            if key not in seen:seen.add(key);selected.append((index,np.array(value,copy=True),identity))
            if len(selected)==2:break
        rank=self.root['root_selection']['unique_vector_rank']
        need(type(rank)is int and rank in(0,1)and len(selected)>rank,'second_distinct_init_missing_no_replacement')
        index,value,identity=selected[rank]
        declared=self.root['root_selection'].get('root_initial_state_sha256')
        need(declared is None or declared==identity['sha256'],'frozen_root_initial_vector_identity')
        declared_index=self.root['root_selection'].get('original_library_index')
        need(declared_index is None or declared_index==index,'frozen_original_library_index')
        self.selection={'root_trajectory_id':self.root['root_trajectory_slot_id'],'unique_vector_rank':rank,'original_library_index':index,
            'native_initial_vector':identity,'source_library':copy.deepcopy(expected),'original_init0_proof':copy.deepcopy(proof),
            'rule':'first_two_typed_byte_distinct_vectors_in_original_library_order','old_optics_case_is_not_current_state':True}
        self.emit('ACTUAL_ROOT_INITIAL_VECTOR',self.selection,{'initial_vector':value})
        selected_proof={**proof,'init0_shape':list(value.shape),'init0_dtype':value.dtype.str,'init0_raw_sha256':identity['raw_sha256'],
            'matched_real_v6_init0':rank==0,'selected_original_library_index':index,'selected_unique_vector_rank':rank,
            'original_init0_binding_preserved_in_selection_receipt':True}
        return [value],selected_proof

class SourceModules:
    def __init__(self,repo_root,manifest):
        self.root=Path(repo_root).resolve();self.rows={r['path'].removeprefix('repo/'):r for r in manifest['files']if r['path'].startswith('repo/')}
    def load(self,name,relative):
        path=(self.root/relative).resolve();need(path.is_relative_to(self.root)and path.is_file()and not path.is_symlink(),'exact_module_path')
        row=self.rows.get(relative);need(row is not None,'required_module_in_source_manifest:'+name)
        raw=path.read_bytes();need((len(raw),sha(raw))==(row['bytes'],row['sha256']),'module_source_identity:'+name)
        if name in sys.modules:
            need(Path(sys.modules[name].__file__).resolve()==path,'foreign_module_alias:'+name);return sys.modules[name]
        sys.path.insert(0,str(path.parent))
        spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);sys.modules[name]=module
        try:spec.loader.exec_module(module)
        except BaseException:sys.modules.pop(name,None);raise
        return module
    def legacy(self):
        rows=[('plus_contract',ADAPTER+'/plus_contract.py'),('adapter',ADAPTER+'/adapter.py'),
            ('trusted_init',ADAPTER+'/trusted_init.py'),('q0_consumer',ADAPTER+'/q0_consumer.py'),
            ('official_namespace',ADAPTER+'/official_namespace.py'),('official_environment',ADAPTER+'/official_environment.py'),
            ('plus_runtime',ADAPTER+'/runtime.py'),('canonical_step_capture',LEGACY+'/render-candidate-v1/canonical_step_capture.py'),
            ('runtime_snapshot',LEGACY+'/render-runtime-v1/runtime_snapshot.py'),('orchestrator',LEGACY+'/recorder-candidate-v1/orchestrator.py'),
            ('production',LEGACY+'/parallel-collection-runtime-v3/production.py'),('environments',LEGACY+'/parallel-collection-runtime-v3/environments.py'),
            ('_continuous_state_helpers',BASE_FACTORY)]
        return {name:self.load(name,path)for name,path in rows}

@contextmanager
def patches(rows):
    saved=[]
    try:
        for owner,name,replacement in rows:
            need(callable(getattr(owner,name,None)),'required_actual_method:'+name)
            saved.append((owner,name,name in vars(owner),vars(owner).get(name)))
            setattr(owner,name,replacement(getattr(owner,name)))
        yield
    finally:
        for owner,name,own,old in reversed(saved):
            if own:setattr(owner,name,old)
            else:delattr(owner,name)

class Activity:
    """Monotonic call attempts; counts live outside every restorable binding."""
    def __init__(self,emit):self.counts={};self.phase='setup';self.physical=0;self.emit=emit
    def wrapper(self,label,physical=False):
        def wrap(original):
            def call(*args,**kwargs):
                key=self.phase+':'+label;self.counts[key]=self.counts.get(key,0)+1
                if physical:self.physical+=1
                return original(*args,**kwargs)
            return call
        return wrap
    @contextmanager
    def setup(self,official,native):
        # These are the same native wrappers used by the original factory.
        with patches([(official,'env_factory',self.wrapper('environment_construction')),
                      (native['wrapper_class'],'reset',self.wrapper('wrapper_reset',True)),
                      (native['wrapper_class'],'set_init_state',self.wrapper('initial_state_assignment',True)),
                      (native['sim_class'],'render',self.wrapper('initialization_render'))]):yield
    @contextmanager
    def rollout(self,official):
        sim=official.env.env.sim;wrapper=official.env
        rows=[(official,'step',self.wrapper('ordinary_control',True)),(sim,'step',self.wrapper('sim_step',True)),
            (sim,'forward',self.wrapper('sim_forward',True)),(wrapper,'set_state',self.wrapper('set_state',True)),
            (sim,'set_state_from_flattened',self.wrapper('set_state_from_flattened',True)),(sim,'render',self.wrapper('sensor_render'))]
        with patches(rows):yield


def snapshot(official,D,emit,role):
    def arrays(name,values):
        ref=emit('SETUP_COMPILED_ARRAYS',{'role':role,'name':name},values)
        return {'receipt':ref,'fields':{k:D['orchestrator'].array_id(v)for k,v in values.items()}}
    def text(name,value):
        raw=value.encode();ref=emit('SETUP_SOURCE_TEXT',{'role':role,'name':name},{'utf8':np.frombuffer(raw,np.uint8)})
        return {'receipt':ref,'bytes':len(raw),'sha256':sha(raw)}
    result=D['runtime_snapshot'].snapshot(official,arrays,text,np=np)
    result['physics_XML_projection']=D['environments'].physics_xml(official.env.env.model.get_xml())
    emit('SETUP_COMPILED_SNAPSHOT',{'role':role,'snapshot':result},{})
    return result


def camera_fields(model,A):
    index=model.camera_name2id('agentview')
    fields=set(A.EDIT)|set(A.STATIC)|{k for k in A.EXTENDED_OPTICS if hasattr(model,k)}
    return {name:np.array(getattr(model,name)[index],copy=True)for name in sorted(fields)}


def create_parent(root,frame,*,D,native,restored,init_reader,classification,emit,activity,check):
    """Sequential exact variant setup -> close -> clean parent; no dummy here."""
    E=D['environments'];case=frame['official_case']
    need(frame['identity']['case_id']==root['source_case_for_optics_and_task_identity_only']and frame['identity']['task']==root['parent_task_id'],'frozen_optics_case_and_task_identity')
    need(case['category']=='Camera Viewpoints','only_two_declared_camera_tasks')
    prompt,prompt_proof=E.matched_base_prompt(frame,restored['plus_root'],restored['runtime_manifest'])
    need(prompt==root['actual_prompt'],'actual_root_prompt_from_official_BDDL')
    emit('ACTUAL_PROMPT_BINDING',prompt_proof,{})
    import camera_sensor_pair as A
    variant=E.OfficialPlusEnvironment(classification,init_reader=init_reader)
    try:
        check();activity.phase='variant_optics_setup'
        with activity.setup(variant,native):variant.reset(case)
        vs=snapshot(variant,D,emit,'variant_optics_setup_init0_not_parent_state')
        vf=camera_fields(variant.env.env.sim.model,A)
        variant_source=copy.deepcopy(variant.source)
        emit('ACTUAL_VARIANT_CAMERA_FIELDS',{'source_case_for_optics_only':root['source_case_for_optics_and_task_identity_only']},vf)
    finally:variant.close()
    need(variant.env is None,'variant_closed_before_clean_parent')
    reader=DistinctInitialReader(init_reader,root,emit)
    clean=E.BaseReferenceEnvironment(classification,plus_root=restored['plus_root'],reference=frame['reference_contract'],
        init_reader=reader,runtime_manifest=restored['runtime_manifest'],variant_preinit_snapshot=vs)
    try:
        check();activity.phase='clean_parent_setup'
        with activity.setup(clean,native):clean.reset(case)
        need(clean.prompt==prompt and reader.selection is not None,'actual_parent_prompt_and_initial_selection')
        selection=reader.selection
        # Do not publish the original reset helper's historical [0] label as R01.
        clean.source['original_reset_helper_init0_label']=clean.source.pop('native_init0',None)
        clean.source.update(official_init_selector='explicit_first_two_byte_distinct_native_vector_selector',
            native_initial_state=selection['native_initial_vector'],current_root_initial_selection=selection,
            root_trajectory_id=root['root_trajectory_slot_id'],source_case_role='optics_and_task_only')
        cs=snapshot(clean,D,emit,'current_clean_parent_after_exact_selected_init')
        E.verify_preinit_compiled(clean.env.env.sim.model,clean.env.env.model.get_xml(),vs)
        cf=camera_fields(clean.env.env.sim.model,A)
        need(set(cf)==set(vf),'same_exposed_complete_camera_fields')
        for key in cf:
            if key not in A.EDIT:need(typed_id(cf[key])==typed_id(vf[key]),'nonedit_camera_field_changed:'+key)
        goal_keys=(set(cs['parsed_problem'])|set(vs['parsed_problem']))-{'problem_name','language','scene_properties'}
        need(goal_keys and all(cs['parsed_problem'].get(k)==vs['parsed_problem'].get(k)for k in goal_keys),
             'current_compiled_goal_and_entities_differ')
        receipt={'schema':'current-native-compiled-camera-setup-v1','root_trajectory_id':root['root_trajectory_slot_id'],
            'case_id':'SDRU-Q01'if 'v00705' in root['source_case_for_optics_and_task_identity_only']else'SDRU-Q02',
            'current_complete_profiles':True,'noncamera_compiled_physics_equal':True,'clean_fields_id':A.R._identity(cf),
            'variant_fields_id':A.R._identity(vf),'clean_source':copy.deepcopy(clean.source),'variant_source':variant_source,
            'old_q0_complete_fields_claimed':False,'current_q0_pair_qualified':False,'variant_init0_is_parent_R01':False}
        emit('ACTUAL_CAMERA_SETUP_BINDING',receipt,{'clean':cf,'variant':vf})
        return clean,selection,cf,vf,receipt
    except BaseException:clean.close();raise


def passive_groups(official,helpers,S,*,known_extra=()):
    """Reuse existing explicit inventory readers, wrap every restore as verify-only."""
    E=helpers._adapter_module();physical=official.env.env;sim=physical.sim
    known={id(v):k for k,v in [('sim',sim),('model',sim.model),('data',sim.data),
        ('raw_model',getattr(sim.model,'_model',sim.model)),('raw_data',getattr(sim.data,'_data',sim.data)),*known_extra]}
    raw=[E.physics_binding(sim)]
    need(all(hasattr(physical,k)for k in helpers._ENV_FIELDS),'actual_environment_field_inventory')
    raw.append(S.R.StateBinding('environment',lambda:{k:copy.deepcopy(getattr(physical,k))for k in helpers._ENV_FIELDS},lambda _:None))
    inventory={}
    for i,robot in enumerate(physical.robots):
        binding,proof=helpers._mutable_tree(f'controller.{i}',robot.controller,known_refs=known)
        raw.append(binding);inventory[binding.name]=proof
        for field in ('recent_qpos','recent_actions','recent_torques','recent_ee_forcetorques','recent_ee_pose','recent_ee_vel','recent_ee_acc','ee_vel_buffer'):
            value=getattr(robot,field,None)
            if value is not None:
                binding,proof=helpers._mutable_tree(f'robot.{i}.{field}',value,known_refs=known)
                raw.append(binding);inventory[binding.name]=proof
    for name,observable in physical._observables.items():
        binding,proof=helpers._mutable_tree('observable.'+name,observable,known_refs=known)
        raw.append(binding);inventory[binding.name]=proof
    need(physical.robots and physical._observables,'nonempty_real_state_inventory')
    return [S.passive_witness_binding(binding.name,binding.snapshot)for binding in raw],inventory


def make_camera_parameters(sim,A,*,case_id,clean,variant,receipt,camlight,activity,passive):
    """Current setup provenance, explicitly not the old q0-field receipt schema."""
    class CompiledCameraParameters(A.CameraParameters):
        def __init__(self):
            A.need(receipt['schema']=='current-native-compiled-camera-setup-v1'and receipt['case_id']==case_id
                and receipt['current_complete_profiles']is True and receipt['noncamera_compiled_physics_equal']is True,'current_compiled_camera_setup_required')
            A.need(receipt['old_q0_complete_fields_claimed']is False and receipt['current_q0_pair_qualified']is False,'no_old_q0_or_new_state_claim')
            A.need(receipt['clean_fields_id']==A.R._identity(clean)and receipt['variant_fields_id']==A.R._identity(variant),'current_setup_actual_array_identity')
            self.sim,self.case_id=sim,case_id;self.clean,self.variant=copy.deepcopy(clean),copy.deepcopy(variant)
            self.camlight,self.activity=camlight,activity;self.passive=tuple(passive);self.last_reports=[]
            self.camera=sim.model.camera_name2id('agentview')
            A.need(case_id in A.CASES and self.camera!=sim.model.camera_name2id('robot0_eye_in_hand'),'declared_task_and_distinct_cameras')
            A.need(any(b.name=='physics.current'for b in self.passive)and all(not b.allow_temporary_change for b in self.passive),'passive_physics_bindings')
            self.check_clean()
    return CompiledCameraParameters()


def _local(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module);return module

@contextmanager
def aliases(values):
    absent=object();old={k:sys.modules.get(k,absent)for k in values};sys.modules.update(values)
    try:yield
    finally:
        for key,value in old.items():
            if value is absent:sys.modules.pop(key,None)
            else:sys.modules[key]=value


def run_root(config,*,campaign,restored,archive_transport,model_guard=None):
    """One loaded model, one clean parent rollout; no authenticated child IO.

    restored is the existing source-bound Plus capsule context, never old q0
    arrays. Parent broker owns network and fresh-GET proof via commit_query.
    """
    need(sys.platform.startswith('linux'),'actual_runtime_Linux_only')
    need(callable(getattr(archive_transport,'commit_query',None)),'credential_free_parent_IPC_required')
    deadline=config['deadline']
    need(callable(model_guard),'live_original_model_guard_required')
    def check():
        need(time.time()<deadline,'absolute_science_deadline')
        need(not Path(config['stop_path']).exists(),'parent_requested_science_stop')
        model_guard(campaign)
    check()
    root=copy.deepcopy(config['root']);frame=copy.deepcopy(config['frame_record'])
    need(root['source_case_for_optics_and_task_identity_only']in('plus-visual-libero_10-v00705:n0','plus-visual-libero_10-v00979:n0'),'exact_two_optics_tasks')
    need(root['root_selection']['unique_vector_rank']in(0,1),'exact_first_two_distinct_roots')
    manifest=config['source_manifest'];imports=SourceModules(restored['repo_root'],manifest);D=imports.legacy()
    bindings=D['plus_contract'].read(Path(restored['repo_root'])/ADAPTER/'SOURCE_BINDINGS-v1.json')
    official=D['plus_runtime'].prepare_official(restored['plus_root'],restored['config_dir'],bindings)
    q0=D['plus_runtime'].validate_q0_gate(restored['q0_gate_path'],restored['q0_gate_identity'])
    need(q0==restored['q0_gate'],'actual_gate_object_matches_validated_file')
    reader=D['plus_runtime'].prepare_init_reader(restored['plus_root'],restored['init_reference'],q0)
    classification=D['plus_contract'].catalog(Path(restored['plus_root'])/'libero/libero/benchmark/task_classification.json')
    helper=D['_continuous_state_helpers']
    # Reuse the same proven native entry points, without invoking the D35B
    # loader (its module aliases are pinned to a different v6 collector).
    mujoco=importlib.import_module('mujoco')
    native={'wrapper_class':importlib.import_module('libero.libero.envs.env_wrapper').ControlEnv,
        'sim_class':importlib.import_module('robosuite.utils.binding_utils').MjSim,
        'camlight':lambda sim:mujoco.mj_camlight(sim.model._model,sim.data._data)}
    here=Path(__file__).resolve().parent.parent
    sys.path[:0]=[str(here/'collector-v2'),str(here/'native-pairing'),str(here)]
    modules={name:_local('_continuous_'+name,here/'collector-v2'/f'{name}.py')for name in ('collector','producer','recording','ledger','query_archive','camera_tap')}
    C=modules['collector'];S=C.S
    contract=_local('_continuous_neural_contract',here.parent/'execution/contract.py')
    with aliases({'contract':contract}):
        R=_local('_continuous_natural_recording',here/'native-recording-v2/native_recording.py')
        with aliases({'jax_candidate':R.J}):
            validator=_local('_continuous_scan_validator',here/'native-recording-v2/scan_native_worker.py')
    from openpi.eval.campaign_policy import CampaignPolicy
    need(isinstance(campaign,CampaignPolicy)and getattr(campaign.encode,'__func__',None)is CampaignPolicy.encode
        and getattr(campaign.decode,'__func__',None)is CampaignPolicy.decode,'original_loaded_campaign_encode_decode')
    model_receipt=model_guard(campaign)
    need(model_receipt.get('model_bound')is True and model_receipt.get('adapter_bound')is True
         and model_receipt.get('loaded_parameter_tree_sha256')=='d077865d5ea92ebbfafbd3b7084634013092c14f54b127fdb65561817a9fb6fc','parent_checked_original_parameter_and_adapter_binding')
    need(root['diagnostic_noise_labels']==[0,1]and root['parent_noise_stream_label']==0,'A_two_diagnostic_noises_and_parent_zero')
    need(root['recorded_query_indices']==list(range((root['official_max_policy_steps']+4)//5)),'A_all_actual_query_opportunities')
    all_ae=set(validator.LAYER_FIELDS+validator.CONTEXT_FIELDS)-{'probabilities_native'}
    for tier in root['tiers'].values():
        need(set(tier['ae_fields'])==all_ae and set(R.PG_FIELDS)<=set(tier['pg_fields']),'A_full_native_trace_fields')
        need({(x['layer_index'],x['generation_call'])for x in tier['dense_P']}=={(l,j)for l in(0,8,17)for j in(0,5,9)},'A_all_nine_native_P')
    rt=R.production_runtime(campaign)
    plan={'root_trajectory_id':root['root_trajectory_slot_id'],'max_policy_steps':root['official_max_policy_steps'],
          'recorded_query_indices':root['recorded_query_indices'],'tiers':root['tiers']}
    limits=config['limits'];need(limits['bytes']==int(8.5*1024**3),'root_8point5GiB_cap')
    ledger=modules['ledger'].Ledger(config['science_output'],limits=limits,deadline_epoch=deadline,plan=plan,
        temporary_limit_bytes=24*1024**3,inflight_limit_bytes=512*1024**2,member_limit_bytes=64*1024**2)
    ledger.archive=modules['query_archive'].QueryArchive(ledger.root,remote_prefix=config['archive_prefix'],transport=archive_transport,
        emit=ledger.emit,temporary_limit_bytes=24*1024**3,inflight_limit_bytes=512*1024**2)
    emit=ledger.emit;activity=Activity(emit);parent=None;state={};result=None;selection=None;collector=None;session=None
    emit('RUNTIME_SOURCE_BOUND',{'official':official,'root':root,'model_binding':model_receipt},{})
    try:
        parent,selection,clean,variant,profile=create_parent(root,frame,D=D,native=native,restored=restored,
            init_reader=reader,classification=classification,emit=emit,activity=activity,check=check)
        root['root_initial_state_sha256']=selection['native_initial_vector']['sha256']
        state={'env':parent,'observation':{}};clock={'query_index':0,'env_step_index':0,'absolute_env_step_index':0,'query_time_actual':np.asarray(parent.env.env.sim.data.time)}
        cursor=C.B.IssuedSliceCursor(lambda:{k:clock[k]for k in ('env_step_index','absolute_env_step_index')})
        import camera_sensor_pair as A
        sim=parent.env.env.sim;physics=helper._adapter_module().physics_binding(sim)
        passive=[S.passive_witness_binding('physics.current',physics.snapshot)]
        optics=make_camera_parameters(sim,A,case_id=profile['case_id'],clean=clean,variant=variant,receipt=profile,
            camlight=native['camlight'],activity=lambda:activity.physical,passive=passive)
        tap=modules['camera_tap'].CameraPairTap(sim,{camera+'_image':parent.env.env._observables[camera+'_image']for camera in A.CAMERAS},
            optics,clock_state=physics.snapshot)
        binding_module=_local('_continuous_observable_binding',here/'native-pairing/observable-binding-v2/observable_binding.py')
        observable_module=importlib.import_module('robosuite.utils.observables')
        binder=binding_module.ObservableEventBinding(tap,observable_module=observable_module,emit=ledger.emit_small)
        with activity.rollout(parent),tap.installed(),binder.installed():
            groups,inventory=passive_groups(parent,helper,S,known_extra=[('paired_sensor_owner',tap),('observable_binding_owner',binder)])
            optics.passive=tuple(groups)
            render= A._array_binding('render.parameters',sim.model,tuple(clean),selection=optics.camera,temporary=True)
            declared=S.DeclaredState([*groups,render,S.mapping_binding('root.clock',clock),cursor.binding,
                S.observation_binding(lambda:state['observation'])],source_id=sha(json.dumps(config['binding'],sort_keys=True).encode()),
                step_attempts=lambda:activity.physical)
            emit('ACTUAL_STATE_INVENTORY',{'inventory':inventory,'all_bindings_passive_except_camera':True,'full_replay_qualified':False},{})
            @dataclasses.dataclass(frozen=True)
            class Branch:
                observation:object
                decode_states:object
            def encode_pair(pair):
                output={}
                for condition in ('clean','nativeperturb'):
                    encoded,batch=campaign.encode([copy.deepcopy(pair[condition]['observation'])],[parent.prompt])
                    output[condition]=Branch(encoded,batch['state'])
                need(R.typed_exact(output['clean'].decode_states,output['nativeperturb'].decode_states),'paired_current_robot_decode_state')
                return output
            last_parent={}
            def input_id(encoded,states):
                return D['orchestrator'].tree_id(modules['recording'].explicit_tree({'observation':encoded,'decode_states':states}))
            def parent_reference(pending,branch,condition,label):
                if condition!='clean'or last_parent.get('query_index')!=pending['query_index']:return {}
                token=config['binding']['run_id']+':single_loaded_policy'
                return {'current_parent_input_sha256':last_parent['input_sha256'],
                    'diagnostic_input_sha256':input_id(branch.observation,branch.decode_states),
                    'current_runtime_identity':token,'diagnostic_runtime_identity':token}
            session=modules['recording'].NaturalRecording(runtime=rt,recording_api=R,scan_validator=validator,production=D['production'],record=frame,
                noise_labels=root['diagnostic_noise_labels'],encode_pair=encode_pair,charge=ledger.charge,emit=emit,check=check,
                diagnostic_noise=lambda label,q:noise(root,label),root_id=root['root_trajectory_slot_id'],parent_reference=parent_reference)
            def step_snapshot():
                obs=state['observation']
                return {'sim_time':np.asarray(sim.data.time),'flat_physics':np.asarray(parent.env.get_sim_state()),
                    'returned_eef_pos':obs.get('robot0_eef_pos'),'returned_eef_quat':obs.get('robot0_eef_quat'),
                    'pose_source':'actual_returned_observation_not_resampled'}
            collector=C.Collector(plan,declared,tap,cursor,clock,emit,sensor_gate=binder.query_gate,sensor_return=binder.record_step_return,
                diagnose=session,step_snapshot=step_snapshot,charge=ledger.charge,storage=ledger)
            activity.phase='clean_parent_dummy'
            modules['producer'].initialize_after_exact_reset(state,collector=collector,check=check,rng_copy=D['canonical_step_capture'].rng_copy)
            activity.phase='clean_parent_policy_and_diagnostics'
            class Backend:
                def baseline(self,branch,draw):
                    # Same encode/sample/decode and actual campaign.sample object
                    # as infer_batch(capture_scalars=False), exposing only the
                    # actual encoded-input identity for per-query parent linkage.
                    import jax
                    import jax.numpy as jnp
                    encoded,batch=campaign.encode([copy.deepcopy(branch['observation'])],[branch['prompt']])
                    last_parent.update(query_index=clock['query_index'],input_sha256=input_id(encoded,batch['state']))
                    raw=campaign.sample(jax.random.key(0),encoded,noise=jnp.asarray(draw),num_steps=10)
                    raw.block_until_ready()
                    return {'raw_actions':np.asarray(raw),'actions':campaign.decode(raw,batch['state'])}
            actual_record=copy.deepcopy(frame);actual_record['noise']={'protocol':'state-response-use-parent-noise-v1','noise_label':root['parent_noise_stream_label']}
            Aadapter=D['adapter'];rng=D['canonical_step_capture']
            result=modules['producer'].run_episode(state,actual_record,Backend(),modules['ledger'].ProducerStore(ledger),'parent',check,collector,
                production=D['production'],environments=D['environments'],orchestrator=D['orchestrator'],PolicyAdapter=Aadapter.PolicyAdapter,
                strict_bool=Aadapter.strict_bool,NonfiniteActionError=Aadapter.NonfiniteActionError,rng_copy=rng.rng_copy,rng_id=rng.rng_id,
                rng_arrays=rng.rng_arrays,observation_identity=Aadapter.observation_identity,
                parent_noise_for=lambda record,q:noise(root,root['parent_noise_stream_label'],q))
        complete=(result['status'].startswith('accepted_')and session.qualified
            and all(row.get('neural_status')=='recorded'for row in collector.rec.rows if row['status']=='recorded'))
        return {'schema':'continuous-native-root-result-v1','root_id':root['root_trajectory_slot_id'],
            'status':'ROOT_COMPLETE'if complete else'FAILED_PRESERVED','root_complete':complete,
            'observed_success':result.get('observed_success'),'root_terminal_reason':result.get('terminal_reason'),
            'parent':result,'actual_initial_selection':selection,'used':dict(ledger.used),'activity_attempts':activity.counts,
            'recording_gate_passed':session.qualified,'provider_release_proven':False,'query_table':collector.rec.summary(),
            'qualification_scope':'current_runtime_continuous_natural_pg_ae','behavior_claim':False}
    except BaseException as error:
        return {'schema':'continuous-native-root-result-v1','root_id':root['root_trajectory_slot_id'],
            'status':'FAILED_PRESERVED','root_complete':False,'observed_success':None,'root_terminal_reason':'technical_failure',
            'parent':result,'actual_initial_selection':selection,'used':dict(ledger.used),'activity_attempts':activity.counts,
            'recording_gate_passed':False if session is None else session.qualified,'provider_release_proven':False,
            'query_table':collector.rec.summary()if collector is not None else{'rows':[{'planned_query_index':q,'status':'unreached','reason':'native_setup_failed'}for q in plan['recorded_query_indices']]},
            'failure':{'type':type(error).__name__,'code':str(error)if isinstance(error,NativeError)else'native_factory_failed_no_retry'},
            'qualification_scope':'current_runtime_continuous_natural_pg_ae','behavior_claim':False}
    finally:
        ledger.seal_small()
        if parent is not None:parent.close()
        emit('ROOT_FACTORY_CLOSED',{'root_id':root['root_trajectory_slot_id'],'activity_attempts':activity.counts,
            'used':dict(ledger.used),'terminal_result_available':result is not None,'auto_retry':False},{})
