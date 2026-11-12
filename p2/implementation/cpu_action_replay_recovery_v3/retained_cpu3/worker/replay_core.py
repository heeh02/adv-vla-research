"""One bounded recorded-action replay; never a new reaction policy.

No simulator/model import. The native facade owns exact reset + original dummy
initializer and live collector/tap; source owns fully checked archived records.
"""
import copy,time
import numpy as np
from failure_frames import frames

ROOTS=('SDRU-CDEV-T00-R00','SDRU-CDEV-T01-R00','SDRU-CDEV-T00-R01','SDRU-CDEV-T01-R01')
ANCHORS=(0,11,16,22)
def need(ok,code):
    if not ok:raise ValueError(code)
def exact(a,b):
    a,b=np.asarray(a),np.asarray(b)
    equal=a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()
    delta=None
    if a.shape==b.shape and a.dtype.kind in 'biuf' and b.dtype.kind in 'biuf':delta=float(np.max(np.abs(a.astype(np.float64)-b.astype(np.float64))))if a.size else 0.
    return {'typed_exact':bool(equal),'actual_shape':list(a.shape),'reference_shape':list(b.shape),
            'actual_dtype':a.dtype.str,'reference_dtype':b.dtype.str,'max_abs_difference':delta}

def group_comparison(actual,expected):
    need(isinstance(expected,dict)and expected,'original_declared_group_identities_required')
    rows={}
    for key,value in expected.items():
        category=('mixed_environment'if key=='environment'else'mixed_observation'if key=='root.observation'
                  else'mixed_camera_observable'if key.startswith(('observable.agentview','observable.robot0_eye_in_hand'))
                  else'physics'if key=='physics.current'else'RNG'if key.startswith('rng.')else'controller'if key.startswith(('controller.','robot.'))else'proprio_or_clock')
        rows[key]={'category':category,'status':'UNAVAILABLE'if key not in actual else'EXACT'if actual[key]==value else'DIFFERENT',
                   'difference_attribution':'mixed_or_unknown'if category.startswith('mixed_')and actual.get(key)!=value else None,
                   'actual':actual.get(key),'reference':value}
    return rows

def validate_source(source):
    """Validate all actually-issued first5 BEFORE creating an environment."""
    for q in range(22):
        chunk=source.parent_actions(q)
        need(chunk.shape==(1,10,7)and chunk.dtype==np.float64 and np.isfinite(chunk).all(),'original_native_full10x7_parent_chunk')
        for slot in range(5):
            row=source.issued(q,slot)
            need(row['metadata']['query_index']==q and row['metadata']['action_position']==slot
                 and row['metadata']['policy_step']==5*q+slot+1 and row['metadata']['absolute_env_step_index']==11+5*q+slot,
                 'original_issued_clock_and_slot')
            need(exact(chunk[0,slot],row['values']['action'])['typed_exact'],'issued_command_matches_recorded_first5')
            outcome=source.outcome(q,slot)
            need(outcome['query_index']==q and outcome['action_position']==slot and outcome['policy_step']==5*q+slot+1,'original_outcome_clock')
    for q in range(23):
        audit=source.parent_audit(q)
        need(audit.get('input_unchanged')is True and audit.get('environment_rng_restored')is True
             and audit.get('rng_changed_during_call',{}).get('numpy')is False,'original_parent_input_RNG_preservation_required')
        proof=source.diagnostic(q)
        need(proof.get('before')==proof.get('after_restore') and not proof.get('forbidden_changes')
             and not proof.get('restore_errors') and not proof.get('observation_errors'),'original_diagnostics_passively_preserved_states')
    for q in ANCHORS:source.anchor(q)  # Require all declared targets, never a survivor subset.


def replay_one(root_id,source,runtime,*,emit,deadline_epoch):
    need(root_id in ROOTS,'declared_root')
    steps=0;query_rows=[];step_rows=[];failure=None;initial_rng_check=None;opened=False
    def check():need(time.time()<deadline_epoch,'absolute_CPU_science_deadline')
    try:
        check();validate_source(source);runtime.open(source,check=check);opened=True
        need(runtime.env_step_calls==10 and runtime.model_calls==0,'exact_original_ten_dummies_no_model')
        for q in range(23):
            check();live=runtime.query(q)
            if q==0:
                first=source.anchor(0)
                before=exact(live['flat_physics'],first['physical_sim_state'])
                need(before['typed_exact'],'post_dummy_q0_physics_mismatch_no_state_injection')
                initial_rng=source.initial_rng()
                initial_rng_check={'actual':runtime.rng_id(),'source':source.rng_id(initial_rng),
                                   'post_dummy_physics':before,'mode':'PRIMARY_NO_RNG_INJECTION','rng_restores':0}
                emit('REPLAY_INITIAL_RNG',initial_rng_check)
                need(initial_rng_check['actual']==initial_rng_check['source'],'post_dummy_q0_RNG_mismatch_no_injection')
            groups=group_comparison(runtime.group_ids(),source.diagnostic(q)['before'])
            row={'root_id':root_id,'query_index':q,'policy_steps':5*q,'absolute_env_steps':10+5*q,'groups':groups,
                 'RNG_was_injected_at_this_query':False,'anchor':q in ANCHORS}
            if q in ANCHORS:
                expected=source.anchor(q)
                row['physics']=exact(live['flat_physics'],expected['physical_sim_state'])
                row['observation']={key:exact(live['observation'][key],value)if key in live['observation']else{'typed_exact':False,'status':'UNAVAILABLE'}
                                    for key,value in expected['observation'].items()}
                row['source_receipts']=expected['source_receipts']
                image_keys={'agentview_image','robot0_eye_in_hand_image'}
                image_available=image_keys<=set(expected['observation']) and image_keys<=set(live['observation'])
                row['render_status']=('UNAVAILABLE'if not image_available else'AVAILABLE_EXACT'
                    if all(row['observation'][k]['typed_exact']for k in image_keys)else'AVAILABLE_DIFFERENT')
            query_rows.append(row);emit('REPLAY_QUERY_WITNESS',row)
            if q==22:break  # q22 observed; NONE of its new commands are issued.
            chunk=source.parent_actions(q);runtime.bind_recorded_actions(chunk,q)
            for slot,action in enumerate(chunk[0,:5]):
                check();need(steps<110,'policy110_step_cap')
                actual=runtime.step(action,query=q,slot=slot,steps=steps);steps+=1
                expected=source.issued(q,slot);outcome=source.outcome(q,slot)
                comparison={phase:{field:exact(actual[phase][field],expected['values'][phase][field])
                    for field in ('sim_time','flat_physics','returned_eef_pos','returned_eef_quat')}for phase in('before','after')}
                result={'root_id':root_id,'query_index':q,'slot':slot,'policy_step':steps,'absolute_env_step':10+steps,
                        'comparisons':comparison,'done_equal':actual['done']==outcome['done'],
                        'success_equal':actual['official_success']==outcome['official_success']}
                step_rows.append(result);emit('REPLAY_ISSUED_STEP_CHECK',result)
                need(result['done_equal']and result['success_equal'],'recorded_terminal_outcome_changed')
                need(not actual['done']and not actual['official_success'],'earlier_terminal_before_declared_q22')
            runtime.query_done(q)
        need(steps==110 and runtime.env_step_calls==120 and runtime.model_calls==0,'exact120_envsteps_0model')
    except BaseException as error:
        failure={'type':type(error).__name__,'code':str(error)[:180],'policy_steps_replayed':steps,'automatic_retry':False,'frames':frames(error)}
        emit('REPLAY_FAILURE',{'root_id':root_id,**failure})
    finally:
        if opened:
            try:runtime.close()
            except BaseException as error:
                failure={'type':type(error).__name__,'code':str(error)[:180],'phase':'native_close',
                         'policy_steps_replayed':steps,'automatic_retry':False,'frames':frames(error)}
                emit('REPLAY_CLOSE_FAILURE',{'root_id':root_id,**failure})
    all_physics=bool(len(step_rows)==110 and len(query_rows)==23 and all(
        d['typed_exact']for r in step_rows for p in r['comparisons'].values()for d in p.values()))
    all_physics=all_physics and all(r.get('physics',{'typed_exact':True})['typed_exact']for r in query_rows)
    # These are archived whole-group hashes. Mixed groups include clocks and
    # counters as well as pixels: an unequal hash cannot be attributed to images.
    groups_exact=bool(len(query_rows)==23 and all(r['status']=='EXACT'for q in query_rows for r in q['groups'].values()))
    nonimage=bool(len(query_rows)==23 and all(r['status']=='EXACT'for q in query_rows for r in q['groups'].values()if not r['category'].startswith('mixed_')))
    targets={q['query_index']:q for q in query_rows if q['anchor']}
    render_available=set(targets)==set(ANCHORS)and all(q.get('render_status','UNAVAILABLE').startswith('AVAILABLE_')for q in targets.values())
    pixels_exact=render_available and all(q['render_status']=='AVAILABLE_EXACT'for q in targets.values())
    physical_qualified=failure is None and all_physics and groups_exact and render_available
    result={'schema':'recorded-path-replay-root-v1','root_id':root_id,'status':'RECORDED_PHYSICAL_PATH_EXACT'if physical_qualified else'FAILED_OR_UNAVAILABLE',
        'physical_path_exact':all_physics,'declared_unmixed_nonimage_groups_exact':nonimage,'all_declared_groups_exact':groups_exact,
        'mixed_group_difference_attribution':'unavailable_from_archived_whole_hash','render_available':render_available,'image_bytes_exact':pixels_exact,
        'physical_only_qualification':physical_qualified,'full_state_restore_qualified':False,'new_reaction_policy_evaluated':False,
        'model_calls':0,'training_steps':0,'env_step_calls':runtime.env_step_calls,'policy_steps_replayed':steps,
        'initial_rng_comparison':initial_rng_check,'rng_restores':0,
        'rng_restores_scope':'archived_reference_injection_only_original_camera_transaction_retained',
        'query_denominator':23,'target_denominator':3,'query_rows':query_rows,'failure':failure}
    emit('REPLAY_ROOT_RESULT',result);return result


def run_campaign(plan,*,source_for_root,runtime_for_root,emit,deadline_epoch):
    need(plan.get('root_ids')==list(ROOTS)and plan.get('targets')==[11,16,22]
         and plan.get('env_steps_per_root_max')==120 and plan.get('model_calls')==plan.get('training_steps')==0,'fixed_four_root_no_model_candidate')
    need(plan.get('development_gate')=='physical_and_all_declared_groups_exact_render_available_pixels_separate','explicit_strict_declared_state_development_gate')
    results=[];development_pass=False
    for index,root_id in enumerate(ROOTS):
        if index and not development_pass:
            results.append({'root_id':root_id,'status':'NOT_ATTEMPTED_DEV_QUALIFICATION_FAILED','env_step_calls':0});continue
        try:result=replay_one(root_id,source_for_root(root_id),runtime_for_root(root_id),emit=emit,deadline_epoch=deadline_epoch)
        except BaseException as error:
            result={'root_id':root_id,'status':'FAILED_SOURCE_OR_RUNTIME_FACTORY','env_step_calls':0,'physical_only_qualification':False,
                    'failure':{'type':type(error).__name__,'code':str(error)[:160],'frames':frames(error)},'model_calls':0,'full_state_restore_qualified':False}
            emit('REPLAY_FACTORY_FAILURE',result)
        results.append(result)
        if index==0:development_pass=result['physical_only_qualification']
    return {'schema':'recorded-path-replay-campaign-v1','roots':results,'root_denominator':4,'targets_per_root':3,
        'development_qualification_passed':development_pass,'total_env_step_calls':sum(r['env_step_calls']for r in results),
        'model_calls':0,'training_steps':0,'full_state_restore_qualified':False,'new_reaction_policy_evaluated':False}
