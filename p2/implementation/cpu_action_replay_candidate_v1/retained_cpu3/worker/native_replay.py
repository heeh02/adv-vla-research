"""Thin native facade over the exact successful continuous setup/collector.

Requires an already restored, hash-bound original runtime; never restores a
checkpoint, loads policy, changes backend, or installs a target snapshot/RNG.
"""
from contextlib import ExitStack
import copy,hashlib,importlib,importlib.util,json,sys
from pathlib import Path
import numpy as np
from replay_core import need

def module(name,path,ref):
    raw=Path(path).read_bytes();need({'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}=={k:ref[k]for k in('bytes','sha256')},'exact_original_runtime_source')
    if name in sys.modules:
        need(Path(sys.modules[name].__file__).resolve()==Path(path).resolve(),'foreign_original_module_alias:'+name)
        return sys.modules[name]
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m

class ReplayStorage:
    """Only the existing collector's per-query sink lifecycle, no new broker."""
    def __init__(self):self.active_query=None
    def begin_query(self,q):need(self.active_query is None,'one_query_open');self.active_query=q
    def end_query(self,q):need(self.active_query==q,'same_query_close');self.active_query=None

class NativeReplay:
    def __init__(self,context,*,emit):
        self.context=context;self.emit=emit;self.env_step_calls=0;self.model_calls=0;self.model_attempts=0
        self.stack=ExitStack();self.parent=None;self.latest={};self.closed=False
    def forbid_neural(self,*_a,**_k):self.model_attempts+=1;raise RuntimeError('no_model_or_diagnostic_callback_in_recorded_replay')
    def _small(self,event,metadata):return self.emit('REPLAY_SENSOR_EVENT',{'event':event,'metadata':metadata})
    def _record(self,event,metadata,values):
        # Current-query arrays are bounded RAM witnesses, not new stored traces.
        self.latest[event]={'metadata':copy.deepcopy(metadata),'values':copy.deepcopy(values)}
        witness=values;domain='selective_reference_explicit_tree_v1'
        if event=='PAIRED_INPUT':
            # Only this original event contains raw native OrderedDict views.
            # Reuse its existing lossless observation witness for AUXILIARY
            # record identity. Neither original PAIR nor latest arrays change.
            from observation_witness import observation_snapshot
            witness=copy.deepcopy(values)
            for role in('clean','nativeperturb'):
                witness['views'][role]['observation']=observation_snapshot(values['views'][role]['observation'])
            domain='paired_input_with_native_observation_witness_v1'
        value_hash=self.S.R._identity(witness)if hasattr(self,'S')else None
        return self.emit('REPLAY_NATIVE_EVENT',{'event':event,'metadata':metadata,'value_identity':value_hash,
            'value_identity_domain':domain,'original_event_is_reused_source':True})
    def open(self,source,*,check):
        self.source=source;c=self.context;need(sys.platform.startswith('linux'),'native_replay_cloud_Linux_only')
        need(c['resources']['GPU']==0 and c['EGL_probe']['status']=='SOFTWARE_EGL_AVAILABLE','same_job_CPU_software_EGL_before_runtime')
        package=Path(c['package_root']);manifest=c['source_manifest'];rows={r['path']:r for r in manifest['files']}
        def load(name,path):return module(name,package/path,rows[path])
        self.N=load('_recorded_original_native','study/continuous/cloud-v1/native_runtime.py');N=self.N
        config=copy.deepcopy(c['original_runtime_config']);root=config['root'];frame=config['frame_record'];restored=c['restored']
        need(root['root_trajectory_slot_id']==source.root_id and frame['official_case']['env_seed']==source.row['env_seed'],'same_original_root_seed_case')
        source_binding=source.row['binding']
        need(source_binding['root_id']==source.root_id,'original_verified_source_root_binding')
        sys.path[:0]=[str(package/'repo/src'),str(package/'study/continuous'),str(package/'study/continuous/native-pairing')]
        imports=N.SourceModules(restored['repo_root'],manifest);self.D=imports.legacy();D=self.D
        bindings=D['plus_contract'].read(Path(restored['repo_root'])/N.ADAPTER/'SOURCE_BINDINGS-v1.json')
        D['plus_runtime'].prepare_official(restored['plus_root'],restored['config_dir'],bindings)
        q0=D['plus_runtime'].validate_q0_gate(restored['q0_gate_path'],restored['q0_gate_identity'])
        need(q0==restored['q0_gate'],'exact_original_environment_capsule_gate')
        reader=D['plus_runtime'].prepare_init_reader(restored['plus_root'],restored['init_reference'],q0)
        classification=D['plus_contract'].catalog(Path(restored['plus_root'])/'libero/libero/benchmark/task_classification.json')
        mujoco=importlib.import_module('mujoco')
        # Resolve GL Python wrappers before the original reset seeds globals;
        # the later read-only renderer string check must not lazy-import them.
        importlib.import_module('OpenGL.GL')
        native={'wrapper_class':importlib.import_module('libero.libero.envs.env_wrapper').ControlEnv,
                'sim_class':importlib.import_module('robosuite.utils.binding_utils').MjSim,
                'camlight':lambda sim:mujoco.mj_camlight(sim.model._model,sim.data._data)}
        egl=importlib.import_module('robosuite.renderers.context.egl_context');egl_path=Path(egl.__file__);egl_raw=egl_path.read_bytes()
        need(len(egl_raw)==6063 and hashlib.sha256(egl_raw).hexdigest()=='fd5d6d8e4978477d0333f33315224e520245ebbedb5c27e50b0e13543893ffc1','exact_original_installed_robosuite_EGL_source')
        self.emit('REPLAY_ROBOSUITE_EGL_SOURCE',{'path':str(egl_path),'bytes':len(egl_raw),'sha256':hashlib.sha256(egl_raw).hexdigest(),
            'empty_CUDA_parser_avoided_by_science_child_unset':True,'GPU_requested':0,'backend':'egl'})
        # Actual manifest maps these paths to repair-v1 originals. Do not load
        # the older unfixed observation/collector files by their current names.
        self.S=load('state_pairing','study/continuous/state_pairing.py')
        collector_module=load('_continuous_collector','study/continuous/collector-v2/collector.py')
        producer=load('_continuous_producer','study/continuous/collector-v2/producer.py')
        camera_tap=load('_continuous_camera_tap','study/continuous/collector-v2/camera_tap.py')
        S=self.S;self.activity=N.Activity(self._record);self.activity.phase='recorded_replay_setup'
        try:
            parent,selection,clean,variant,profile=N.create_parent(root,frame,D=D,native=native,restored=restored,init_reader=reader,
                classification=classification,emit=self._record,activity=self.activity,check=check)
            self.parent=parent;expected=source.row['initial_selection']
            need(all(selection[k]==expected[k]for k in('unique_vector_rank','original_library_index','native_initial_vector','source_library')),'same_original_selected_init_vector')
            self.state={'env':parent,'observation':{}};self.clock={'query_index':0,'env_step_index':0,'absolute_env_step_index':0,'query_time_actual':np.asarray(parent.env.env.sim.data.time)}
            cursor=collector_module.B.IssuedSliceCursor(lambda:{k:self.clock[k]for k in('env_step_index','absolute_env_step_index')})
            import camera_sensor_pair as A
            sim=parent.env.env.sim;self.sim=sim;helper=D['_continuous_state_helpers'];physics=helper._adapter_module().physics_binding(sim)
            passive=[S.passive_witness_binding('physics.current',physics.snapshot)]
            optics=N.make_camera_parameters(sim,A,case_id=profile['case_id'],clean=clean,variant=variant,receipt=profile,
                camlight=native['camlight'],activity=lambda:self.activity.physical,passive=passive)
            tap=camera_tap.CameraPairTap(sim,{camera+'_image':parent.env.env._observables[camera+'_image']for camera in A.CAMERAS},optics,clock_state=physics.snapshot)
            binder_module=load('_continuous_observable_binding','study/continuous/native-pairing/observable-binding-v2/observable_binding.py')
            binder=binder_module.ObservableEventBinding(tap,observable_module=importlib.import_module('robosuite.utils.observables'),emit=self._small)
            self.stack.enter_context(self.activity.rollout(parent));self.stack.enter_context(tap.installed());self.stack.enter_context(binder.installed())
            groups,inventory=N.passive_groups(parent,helper,S,known_extra=[('paired_sensor_owner',tap),('observable_binding_owner',binder)])
            self.emit('REPLAY_ACTUAL_STATE_INVENTORY',{'inventory':inventory,'full_replay_qualified':False})
            optics.passive=tuple(groups)
            render=A._array_binding('render.parameters',sim.model,tuple(clean),selection=optics.camera,temporary=True)
            self.declared=S.DeclaredState([*groups,render,S.mapping_binding('root.clock',self.clock),cursor.binding,S.observation_binding(lambda:self.state['observation'])],
                source_id=N.sha(json.dumps(source_binding,sort_keys=True).encode()),step_attempts=lambda:self.activity.physical)
            plan={'root_trajectory_id':source.root_id,'max_policy_steps':root['official_max_policy_steps'],
                  'recorded_query_indices':list(range(23)),'tiers':{str(q):root['tiers'][str(q)]for q in range(23)}}
            self.storage=ReplayStorage()
            self.collector=collector_module.Collector(plan,self.declared,tap,cursor,self.clock,self._record,sensor_gate=binder.query_gate,
                sensor_return=binder.record_step_return,diagnose=self.forbid_neural,step_snapshot=self.step_snapshot,charge=self.forbid_neural,storage=self.storage)
            # The UNCHANGED original initializer emits exactly 10 zero/-1 dummy
            # commands. The counting wrapper has no state/RNG setter.
            actual_step=parent.step
            def counted(action):
                need(self.env_step_calls<120,'one_root_max120_envsteps');self.env_step_calls+=1;return actual_step(action)
            parent.step=counted
            self.activity.phase='recorded_replay_dummy'
            producer.initialize_after_exact_reset(self.state,collector=self.collector,check=check,rng_copy=D['canonical_step_capture'].rng_copy)
            need(self.env_step_calls==10,'exact_source_defined_dummy_count')
            from backend_probe import current_software_renderer
            self.emit('REPLAY_ACTUAL_ENV_GL',{'root_id':source.root_id,**current_software_renderer(allow_unset_cuda=True),'GPU_requested':0,'after_original_dummy_count':10})
            need(not any(n=='jax'or n.startswith(('openpi.models','openpi.policies'))for n in sys.modules),'no_neural_runtime_imported')
            self.activity.phase='recorded_replay_issued_commands'
        except BaseException:self.close();raise
    def step_snapshot(self):
        obs=self.state['observation']
        return {'sim_time':np.asarray(self.sim.data.time),'flat_physics':np.asarray(self.parent.env.get_sim_state()),
                'returned_eef_pos':obs.get('robot0_eef_pos'),'returned_eef_quat':obs.get('robot0_eef_quat'),
                'pose_source':'actual_returned_observation_not_resampled'}
    def query(self,q):
        self.latest={};self.collector.query_input(self.state['observation'],query=q,steps=5*q)
        return {'flat_physics':np.asarray(self.parent.env.get_sim_state()),'observation':copy.deepcopy(self.state['observation']),
                'paired_event':self.latest.get('PAIRED_INPUT')}
    def group_ids(self):
        # No extra ReferenceTransaction / RNG setter for this comparison.
        # Match that original transaction's exact hash domains using readers.
        result={b.name:self.S.R._identity(b.snapshot())for b in self.declared.bindings}
        result['rng.python_global']=self.S.R._identity(self.S.R.random.getstate())
        result['rng.numpy_global']=self.S.R._identity(self.S.R.np.random.get_state())
        return result
    def rng_id(self):return self.D['canonical_step_capture'].rng_id(self.D['canonical_step_capture'].rng_copy())
    def bind_recorded_actions(self,actions,q):
        need(actions.shape==(1,10,7)and actions.dtype==np.float64,'recorded_original_actions_ABI')
        need(0<=q<22,'only_original_q0_through_q21_chunks_may_bind')
        self.collector.cursor.bind_generated({'actions':actions,'raw_actions':self.source.parent_raw32(q)},query_index=q)
        self.collector.pending_query=None
        self.emit('RECORDED_COMMANDS_BOUND',{'query_index':q,'origin':'archived_parent_actions','model_calls':0,'issued_slots':[0,1,2,3,4]})
    def step(self,action,*,query,slot,steps):
        before=copy.deepcopy(self.step_snapshot())
        with self.collector.policy_step(slot=slot,action=action,query=query,steps=steps):
            obs,done=self.parent.step(action.tolist());self.collector.returned(obs,steps=steps+1,absolute=11+steps);self.state['observation']=obs
            success=self.parent.check_success();need(isinstance(success,(bool,np.bool_))and isinstance(done,(bool,np.bool_)),'native_bool_success_done')
            self.collector.sync(query=query,steps=steps+1,absolute=11+steps)
        self.collector.outcome(query=query,slot=slot,steps=steps+1,success=bool(success),done=bool(done))
        return {'before':before,'after':copy.deepcopy(self.step_snapshot()),'official_success':bool(success),'done':bool(done)}
    def query_done(self,q):self.collector.query_done(q)
    def close(self):
        if self.closed:return
        self.closed=True
        try:
            try:
                if hasattr(self,'collector'):
                    if self.storage.active_query is not None:self.storage.end_query(self.storage.active_query)
                    if not self.collector.rec.closed:
                        complete=self.collector.rec.next==len(self.collector.rec.rows)
                        self.collector.bridge.finish('window_complete'if complete else'root_failure')
            finally:self.stack.close()
        finally:
            if self.parent is not None:self.parent.close()
