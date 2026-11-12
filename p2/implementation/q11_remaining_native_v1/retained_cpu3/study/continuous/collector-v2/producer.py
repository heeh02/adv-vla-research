"""Narrow derivative of frozen collect.run_episode; same actual 5-slot loop.

Dependency modules are passed by root from the exact production package. This
module imports no environment/model and creates or resets no simulation.
"""
import copy
import numpy as np

PRODUCER_SHA='0f548d9c5af015981009ea13f7637336a8e46e00d8371e33d3d47258a5343f9b'
INITIALIZER_SHA='26e4c67ab3142a71edad074e65d97a02ab6f032007d5bda0f3c78112cb98b49e'


def initialize_after_exact_reset(state,*,collector,check,rng_copy):
    """Call ONLY after owner's exact reset/init receipt. No query-side render.

    Same ten dummy commands as E.initialize. The existing tap must already be
    installed; q0 (if planned) is armed before the actual tenth sensor cycle.
    Parent owner supplies state['env']; it binds state observation by reference.
    """
    if state.get('continuous_initialization_state')is not None:raise ValueError('exact_reset_initialization_may_run_once')
    state['continuous_initialization_state']='initializing'
    env=state['env'];collector.sync(query=0,steps=0,absolute=0)
    collector.emit('ROOT_RUNTIME_BINDING',{'actual_prompt':env.prompt,'environment_source':getattr(env,'source',None),
        'reset_and_initial_vector_owned_by_caller':True},{})
    try:
        for index in range(10):
            check()
            with collector.bridge.producing_step(next_n=0,next_absolute=index+1,startup=True):
                obs,done=env.step([0.0]*6+[-1.0])
                collector.returned(obs,steps=0,absolute=index+1)
                state['observation']=obs
                collector.sync(query=0,steps=0,absolute=index+1)
        state['rng']=rng_copy();state['physical']=np.array(env.env.get_sim_state(),copy=True)
        state['continuous_initialization_state']='complete'
        return state
    except BaseException as error:
        state['continuous_initialization_state']='failed'
        collector.finish({'status':'technical_missing','success':None,'phase':'dummy_initialization',
            'type':type(error).__name__,'policy_steps':0,'queries':0})
        raise



def run_episode(state,record,backend,store,prefix,check,collector,*,production,environments,orchestrator,
                PolicyAdapter,strict_bool,NonfiniteActionError,rng_copy,rng_id,rng_arrays,observation_identity,parent_noise_for):
    """Original parent sampler remains sole source of actually issued controls."""
    P,E,O=production,environments,orchestrator
    env,obs=state['env'],state['observation'];E.restore_rng(state['rng'])
    steps,query,success,done,trace=0,0,False,False,[]
    result={'status':'technical_missing','success':None,'observed_success':None,'policy_steps':0,'queries':0,
            'terminal_reason':None,'canonical_action_source':True,'original_F0_population':False,
            'noise_protocol':record['noise'],'error':None}
    phase='initialization';base_audit=P.rng_auditor(store,prefix+'/neural-calls');attempted_output={}
    def rng_audit(receipt,arrays):
        base_audit(receipt,arrays)
        O.need(receipt['rng_changed_during_call']['numpy']is False,'parent_raw_NumPy_change_forbidden')
    try:
        O.need(record['official_case']['max_policy_steps']==collector.plan['max_policy_steps'],'frozen_parent_horizon_mismatch')
        while steps<record['official_case']['max_policy_steps']:
            attempted_output.clear()
            check();phase='canonical_inference'
            collector.query_input(obs,query=query,steps=steps)
            noise=parent_noise_for(record,query)
            physical=np.array(env.env.get_sim_state(),copy=True)
            before={'observation':observation_identity(obs),'rng':rng_id(rng_copy())}
            branch=P.branch(obs,env.prompt,physical,record.get('goal_binding_id'))
            store.arrays(prefix+f'/query-{query:04d}-input.npz',{**obs,'noise':noise,'physical_sim_state':physical,**rng_arrays(rng_copy())},'actual_parent_wrapper_input')
            collector.charge_parent(query=query,noise_label=record['noise']['noise_label'])
            def parent_call(actual_branch,actual_noise):
                value=backend.baseline(actual_branch,actual_noise);attempted_output['value']=value;return value
            generated=P.invariant_call(parent_call,branch,noise,branch,noise,_rng_audit=rng_audit)
            store.arrays(prefix+f'/query-{query:04d}-actions.npz',{k:generated[k]for k in ('raw_actions','actions')},'canonical_native_sampler')
            PolicyAdapter.validate_output(generated,1)
            O.need(before=={'observation':observation_identity(obs),'rng':rng_id(rng_copy())},'canonical_policy_changed_environment_input_rng')
            phase='natural_same_state_diagnostic';collector.parent_generated(generated,noise,query=query)
            O.need(before=={'observation':observation_identity(obs),'rng':rng_id(rng_copy())},'diagnostic_changed_parent_input_rng')
            phase='canonical_control_step'
            for slot,action in enumerate(generated['actions'][0,:5]):
                check()
                if steps==record['official_case']['max_policy_steps']:break
                with collector.policy_step(slot=slot,action=action,query=query,steps=steps):
                    obs,done=env.step(action.tolist())
                    collector.returned(obs,steps=steps+1,absolute=11+steps)
                    state['observation']=obs
                    success=strict_bool(env.check_success(),'actual official success')
                    done=strict_bool(done,'actual official done')
                    steps+=1
                    collector.sync(query=query,steps=steps,absolute=10+steps)
                trace.append({'query':query,'slot':slot,'policy_step':steps,'done':done,'official_success':success})
                collector.outcome(query=query,slot=slot,steps=steps,success=success,done=done)
                if success or done:break
            collector.query_done(query)
            query+=1
            if success or done:break
        result.update(status='accepted_success'if success else'accepted_failure',success=success,observed_success=success,
            terminal_reason='official_success'if success else'environment_done'if done else'policy_budget_exhausted')
    except Exception as error:
        result.update(status='technical_missing',success=None,error={'type':type(error).__name__,'phase':phase},observed_success=success)
        if isinstance(error,NonfiniteActionError)and phase=='canonical_inference':
            result.update(status='accepted_model_failure',success=False,terminal_reason='nonfinite_canonical_action_on_validated_input')
        if phase=='canonical_inference'and 'value'in attempted_output:
            value=attempted_output['value']
            store.arrays(prefix+'/failed-parent-output.npz',{k:value[k]for k in('raw_actions','actions')if k in value},'failed_original_parent_call_output')
        if getattr(error,'arrays',None):store.arrays(prefix+'/failed-canonical-actions.npz',error.arrays,'failed_actual_canonical_output')
    finally:
        try:
            store.arrays(prefix+'/terminal.npz',{**obs,'physical_sim_state':np.asarray(env.env.get_sim_state()),**rng_arrays(rng_copy())},'actual_last_returned_wrapper_terminal')
            result['terminal_observation_physics_alignment']='verified_call_order'if result['status'].startswith('accepted_')else'unverified_after_exception'
        except Exception as error:result.update(status='technical_missing',success=None,terminal_capture_error_type=type(error).__name__)
        result.update(policy_steps=steps,queries=query,policy_trace=trace)
        try:env.close()
        except Exception as error:result.update(status='technical_missing',success=None,close_error_type=type(error).__name__)
        try:collector.finish(result)
        except Exception as error:result.update(status='technical_missing',success=None,collector_close_error_type=type(error).__name__)
    store.json(prefix+'/Y-v2.json',result);return result
