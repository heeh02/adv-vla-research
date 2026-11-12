"""Canonical q0 and Y with one live Plus wrapper at any instant."""
import copy
import json
from pathlib import Path
import time
import numpy as np
import orchestrator as O
import production as P
import environments as E
from adapter import PolicyAdapter, observation_identity, strict_bool, NonfiniteActionError
from canonical_step_capture import rng_copy, rng_id, rng_arrays

VISUAL = {'Background Textures', 'Camera Viewpoints', 'Light Conditions', 'Sensor Noise'}


def isolated_startup(state, record, callback, store):
    """Technical compilation/qualification owns a private copy of global RNG.

    Restore both environmental streams even on failure. Never restore or mask
    a mutated physical state or observation; those still fail independently.
    """
    saved_rng = rng_copy()
    def capture():
        streams = rng_id(rng_copy())
        return {'numpy_rng': streams['numpy'], 'python_rng': streams['python'],
                'physics': O.array_id(state['env'].env.get_sim_state()),
                'observation': observation_identity(state['observation'])}
    before = capture()
    store.arrays('startup-isolation/before-rng.npz', rng_arrays(saved_rng), 'environment_rng_before_technical_startup')
    callback_error = None
    try:
        result = callback(state, record)
    except BaseException as error:
        callback_error = error
    finally:
        after_rng = rng_copy()
        try:
            after_call = capture()
        finally:
            # Restore even if a broken callback made passive inspection fail.
            # Physics and observations themselves are never restored here.
            E.restore_rng(saved_rng)
        after_restore = capture()
        store.arrays('startup-isolation/after-call-rng.npz', rng_arrays(after_rng), 'technical_startup_rng_before_restoration')
        store.arrays('startup-isolation/restored-rng.npz', rng_arrays(rng_copy()), 'environment_rng_after_restoration')
        changed = {key: before[key] != after_call[key] for key in before}
        restored = {key: before[key] == after_restore[key] for key in before}
        store.json('STARTUP-STATE-ISOLATION-v2.json', {
            'schema': 'plus-startup-state-isolation-v2',
            'before': before, 'after_callback_before_restore': after_call, 'after_rng_restore': after_restore,
            'before_component_sha256': {key: O.tree_id(value) for key, value in before.items()},
            'after_callback_component_sha256': {key: O.tree_id(value) for key, value in after_call.items()},
            'after_restore_component_sha256': {key: O.tree_id(value) for key, value in after_restore.items()},
            'changed_during_callback': changed, 'same_after_rng_restore': restored,
            'restored_fields': ['numpy_rng', 'python_rng'], 'physics_or_observation_restored': False,
            'callback_exception_type': type(callback_error).__name__ if callback_error else None,
            'previous_v1_failed_component': 'unmeasured_v1_did_not_record_component_differences'})
        O.need(restored['numpy_rng'] and restored['python_rng'], 'startup_environment_rng_restore_failed')
        O.need(not changed['physics'] and restored['physics'], 'startup_changed_actual_physics')
        O.need(not changed['observation'] and restored['observation'], 'startup_changed_actual_observation')
    if callback_error is not None:
        raise callback_error
    return result


def run_episode(state, record, backend, store, prefix, check, initial=None):
    env, obs = state['env'], state['observation']
    E.restore_rng(state['rng'])
    steps, query, success, done, trace = 0, 0, False, False, []
    result = {'status': 'technical_missing', 'success': None, 'observed_success': None,
              'policy_steps': 0, 'queries': 0, 'terminal_reason': None, 'canonical_action_source': True,
              'original_F0_population': False, 'noise_protocol': record['noise'], 'error': None}
    if 'clean_id' in state:
        result.update(clean_id=state['clean_id'], clean_execution_reused=False,
                      repeated_clean_identity_is_not_independent_sample=True)
    phase = 'initialization'
    rng_audit = P.rng_auditor(store, prefix + '/neural-calls')
    try:
        while steps < record['official_case']['max_policy_steps']:
            check(); phase = 'canonical_inference'
            noise = P.noise_for(record, query)
            physical = np.array(env.env.get_sim_state(), copy=True)
            before = {'observation': observation_identity(obs), 'rng': rng_id(rng_copy())}
            b = P.branch(obs, env.prompt, physical, record.get('goal_binding_id'))
            store.arrays(prefix + f'/query-{query:04d}-input.npz', {**obs, 'noise': noise,
                         'physical_sim_state': physical, **rng_arrays(rng_copy())}, 'actual_Y_wrapper_input')
            if query == 0 and initial is not None:
                O.need(initial['input_identity'] == O.request_binding(b) and initial['physical_identity'] == O.array_id(physical), 'M_to_Y_q0_actual_identity')
                generated = initial['actions']
            else:
                generated = P.invariant_call(backend.baseline, b, noise, b, noise, _rng_audit=rng_audit)
            store.arrays(prefix + f'/query-{query:04d}-actions.npz', {k: generated[k] for k in ('raw_actions', 'actions')}, 'canonical_native_sampler')
            PolicyAdapter.validate_output(generated, 1)
            O.need(before == {'observation': observation_identity(obs), 'rng': rng_id(rng_copy())}, 'canonical_policy_changed_environment_input_rng')
            phase = 'canonical_control_step'
            for slot, action in enumerate(generated['actions'][0, :5]):
                check()
                if steps == record['official_case']['max_policy_steps']: break
                obs, done = env.step(action.tolist())
                success = strict_bool(env.check_success(), 'actual official success')
                done = strict_bool(done, 'actual official done')
                steps += 1
                trace.append({'query': query, 'slot': slot, 'policy_step': steps, 'done': done, 'official_success': success})
                if success or done: break
            query += 1
            if success or done: break
        result.update(status='accepted_success' if success else 'accepted_failure', success=success,
                      observed_success=success, terminal_reason='official_success' if success else ('environment_done' if done else 'policy_budget_exhausted'))
    except Exception as error:
        result.update(status='technical_missing', success=None, error={'type': type(error).__name__, 'phase': phase}, observed_success=success)
        if isinstance(error, NonfiniteActionError) and phase == 'canonical_inference':
            result.update(status='accepted_model_failure', success=False, terminal_reason='nonfinite_canonical_action_on_validated_input')
        if getattr(error, 'arrays', None):
            store.arrays(prefix + '/failed-canonical-actions.npz', error.arrays, 'failed_actual_canonical_output')
    finally:
        try:
            store.arrays(prefix + '/terminal.npz', {**obs, 'physical_sim_state': np.asarray(env.env.get_sim_state()), **rng_arrays(rng_copy())}, 'actual_last_returned_wrapper_terminal')
            result['terminal_observation_physics_alignment'] = 'verified_call_order' if result['status'].startswith('accepted_') else 'unverified_after_exception'
        except Exception as error:
            result.update(status='technical_missing', success=None, terminal_capture_error_type=type(error).__name__)
        try: env.close()
        except Exception as error: result.update(status='technical_missing', success=None, close_error_type=type(error).__name__)
    result.update(policy_steps=steps, queries=query, policy_trace=trace)
    store.json(prefix + '/Y-v1.json', result)
    return result


def startup(real_state, record, backend, output, full_pp, *, synthetic=False):
    """One worker-wide numerical qualifier, identical actual input on both sides.

    This technical identity pair is NOT clean/variant data or confirmation N.
    """
    goal = 'technical_identity:' + real_state['env'].source['parsed_problem_sha256']
    b = P.branch(real_state['observation'], real_state['env'].prompt, real_state['physical'], goal)
    pair = {'pair_id': 'startup-identical-real-input', 'query': 0, 'environment_pair_status': 'externally_qualified',
            'pair_binding_id': 'identity_control_same_actual_wrapper_input', 'goal_binding_id': goal,
            'branches': {'reference': copy.deepcopy(b), 'variant': copy.deepcopy(b)}, 'noise': P.noise_for(record, 0)}
    original_require = backend.require_native_actions
    original_methods = {name: getattr(backend, name) for name in ('baseline', 'trajectory', 'prepare', 'capture')}
    rng_store = P.Store(Path(output).with_name(Path(output).name + '-rng-isolation'))
    audit = P.rng_auditor(rng_store, 'calls')
    def isolated(method):
        def call(*args, **kwargs):
            return P.invariant_call(method, pair, pair['noise'], *args, _rng_audit=audit, **kwargs)
        return call
    for name, method in original_methods.items():
        setattr(backend, name, isolated(method))
    backend.require_native_actions = lambda name, a, b: O.hard_exact(a, b, 'startup_T_C_raw_exact_only')
    try:
        report = O.run_pair(pair, backend, output, full_pp=full_pp, synthetic=synthetic)
    except O.ContractError as error:
        # run_pair checks C0/C1/C2 repeat, input and RNG invariants before these
        # exact-only T gates. No other error is eligible for this downgrade.
        if str(error) not in ('startup_T_C_raw_exact_only', 'baseline_recorder_actions'):
            raise
        return {'numerical_contract_completed': True, 'trace_qualified': False, 'canonical_repeat_passed': True,
                'trace_status': 'unqualified_T_C_exact_only', 'canonical_Y_permitted': True,
                'technical_identity_control_only': True, 'confirmation_samples': 0, 'upstream_failure': 'FAILED-v1.json'}
    finally:
        backend.require_native_actions = original_require
        for name, method in original_methods.items():
            setattr(backend, name, method)
    return {'numerical_contract_completed': report['status'] == ('synthetic_contract_PASS' if synthetic else 'candidate_replay_contract_complete_requires_root_review'),
            'trace_qualified': True, 'canonical_repeat_passed': True,
            'technical_identity_control_only': True, 'confirmation_samples': 0,
            'source_role': 'two_copies_of_the_same_real_q0_input', 'upstream_result': 'RESULT-v1.json'}


def collect_case(record, *, classification, init_reader, plus_root, runtime_manifest, backend, out,
                 startup_callback, check, binding, run_id, failure_location, synthetic=False):
    store = P.Store(out)
    small = {'schema': 'plus-analysis-case-v1', 'data_role': 'SYNTHETIC_SOFTWARE_FIXTURE' if synthetic else 'ACTUAL_METADATA_READSET',
             'identity': record['identity'], 'binding': {**binding, 'run_id': run_id, 'attempt_id': run_id + ':' + record['identity']['case_id']},
             'revision': 0, 'previous_sha256': None, 'analysis_status': 'pending',
             'M': {'status': 'pending', 'reason': 'not_measured'},
             'Y': {a: {'status': 'not_collected', 'success': None} for a in ('reference', 'treatment')}, 'metrics': []}
    active, phase, fatal = [], 'variant_initialize', None
    try:
        phase = 'official_base_prompt_binding'
        record = copy.deepcopy(record)
        actual_prompt, prompt_proof = E.matched_base_prompt(record, plus_root, runtime_manifest)
        record['frame_lexical_prompt_metadata'] = record.get('matched_prompt')
        record['matched_prompt'] = actual_prompt
        store.json('OFFICIAL-PROMPT-BINDING-v2.json', prompt_proof)
        store.json('CASE-PLAN-v1.json', record)
        phase = 'variant_initialize'
        variant_env = E.OfficialPlusEnvironment(classification, init_reader=init_reader)
        active.append(variant_env)
        var = E.initialize(variant_env, record['official_case'], store, 'treatment', check,
                           sensor_tap=record['identity']['family'] == 'Sensor Noise')
        variant_env.prompt = record['matched_prompt']
        var_q0 = {'observation': copy.deepcopy(var['observation']),
                  'physical': np.array(var['physical'], copy=True),
                  'prompt': str(variant_env.prompt)}
        store.json('TREATMENT-Q0-BINDING-v3.json', {
            'observation': observation_identity(var_q0['observation']),
            'physical': O.array_id(var_q0['physical']), 'prompt': var_q0['prompt'],
            'saved_before_treatment_Y': True, 'independent_from_closed_environment': True})
        # Capture q0 compiled identity only for an eligible paired reference.
        # Pair review later uses this saved snapshot, never a closed wrapper.
        eligible_reference = record.get('reference_contract') is not None and record['identity']['family'] in VISUAL
        vs = E.snapshot(var, store, 'treatment/model') if eligible_reference else None
        ref, ref_record, pair_gate = None, None, None
        if record.get('reference_contract') is None:
            small['M'] = {'status': 'blocked', 'reason': 'explicit_base_reference_source_missing'}
        elif record['identity']['family'] not in VISUAL:
            small['M'] = {'status': 'blocked', 'reason': 'semantic_or_physical_panel_separate_reference_initialization_contract_required'}
        phase = 'startup_numerical_gate'
        startup_status = isolated_startup(var, record, startup_callback, store)
        phase = 'variant_q0_recording'
        var_input = P.branch(var['observation'], variant_env.prompt, var['physical'], record.get('goal_binding_id'))
        if startup_status['trace_qualified']:
            var_record = P.record_branch(var_input, P.noise_for(record, 0), backend, store, 'M/treatment', full_pp=record['full_pp'], check=check)
        else:
            var_record = P.record_canonical_only(var_input, P.noise_for(record, 0), backend, store, 'M/treatment', check=check)
        small['metrics'].extend(P.metric_rows(var_record, 'treatment'))
        phase = 'treatment_canonical_Y'
        y = run_episode(var, record, backend, store, 'treatment/Y', check, initial=var_record)
        small['Y']['treatment'] = {k: y[k] for k in ('status', 'success')}
        O.need('close_error_type' not in y, 'variant_close_failed_before_reference')
        O.need(variant_env.env is None, 'variant_must_be_closed_before_reference_created')
        active.remove(variant_env)
        var.pop('env')  # No pair review may dereference the closed wrapper.
        del variant_env  # Release the wrapper reference before base construction.
        if eligible_reference:
            phase = 'reference_initialize_and_pair_qualification'
            try:
                reference_env = E.BaseReferenceEnvironment(classification, plus_root=plus_root,
                    reference=record['reference_contract'], init_reader=init_reader,
                    runtime_manifest=runtime_manifest, variant_preinit_snapshot=vs)
                active.append(reference_env)
                ref = E.initialize(reference_env, record['official_case'], store, 'reference', check)
                O.need(reference_env.prompt == record['matched_prompt'], 'actual_base_wrapper_prompt_differs_from_verified_BDDL_language')
                ref['clean_id'] = O.tree_id({'base_BDDL': reference_env.source['resolved_BDDL']['sha256'],
                    'base_scene': reference_env.source['scene']['sha256'], 'native_init0': reference_env.source['native_init0'],
                    'model': backend.runtime_identity['checkpoint'], 'prompt': reference_env.prompt, 'noise': record['noise'],
                    'wait_steps': 10, 'execute_prefix': 5, 'max_policy_steps': record['official_case']['max_policy_steps']})
                rs = E.snapshot(ref, store, 'reference/model')
                pair_gate = E.qualify_pair(ref, var_q0, rs, vs, record)
                store.json('PAIR-QUALIFICATION-v1.json', pair_gate)
                if not pair_gate['qualified']:
                    small['M'] = {'status': 'blocked', 'reason': 'actual_q0_pair_identity_or_physics_mismatch'}
            except Exception as error:
                store.json('REFERENCE-FAILURE-v1.json', failure_location(error, phase))
                small['M'] = {'status': 'technical_missing', 'reason': 'reference_source_init_or_runtime_qualification_failed'}
                ref = None
        store.json('MATCHED-PROMPT-v1.json', {'protocol': 'matched_goal_prompt', 'actual_policy_prompt': record['matched_prompt'],
                   'source': record['reference_contract']['base_bddl'] if record.get('reference_contract') else None,
                   'reference_wrapper_language_checked': ref is not None, 'original_F0_protocol_unchanged': True})
        if ref is not None and pair_gate['qualified']:
            phase = 'reference_q0_recording'
            ref_input = P.branch(ref['observation'], ref['env'].prompt, ref['physical'], record['goal_binding_id'])
            if var_record['report']['trace_qualified']:
                ref_record = P.record_branch(ref_input, P.noise_for(record, 0), backend, store, 'M/reference', full_pp=record['full_pp'], check=check)
            else:
                ref_record = P.record_canonical_only(ref_input, P.noise_for(record, 0), backend, store, 'M/reference', check=check)
            if var_record['report']['trace_qualified'] and ref_record['report']['trace_qualified']:
                small['M'] = {'status': 'complete', 'reason': None}
                small['metrics'].extend(P.metric_rows(ref_record, 'reference'))
                small['metrics'].extend(P.metric_rows(var_record, 'paired_difference', reference=ref_record))
            else:
                small['M'] = {'status': 'technical_missing', 'reason': 'T_C_exact_unqualified_canonical_Y_retained'}
        if ref is not None and record['collect_reference_Y']:
            phase = 'reference_canonical_Y'
            y = run_episode(ref, record, backend, store, 'reference/Y', check, initial=ref_record)
            small['Y']['reference'] = {k: y[k] for k in ('status', 'success')}
            small['reference_identity'] = {'clean_id': ref['clean_id'], 'execution_reused': False,
                'shared_statistical_cluster': record['identity']['task']}
        store.json('LATE50-v1.json', {'status': 'not_collected', 'reason': 'q0_primary_batch_late_own_clean_not_qualified',
                   'two_Y_branches_same_state_claim': False})
        small['analysis_status'] = 'complete'
    except Exception as error:
        fatal = failure_location(error, phase)
        store.json('CASE-FAILURE-v1.json', fatal)
        small['analysis_status'] = 'failed'
        if small['M']['status'] == 'pending': small['M'] = {'status': 'failed', 'reason': phase}
        for arm in small['Y']:
            if small['Y'][arm]['status'] == 'not_collected': small['Y'][arm] = {'status': 'technical_missing', 'success': None}
    finally:
        for env in active:
            try: env.close()
            except Exception: pass
    store.close_case(small)
    return {'case_id': record['identity']['case_id'], 'status': small['analysis_status'],
            'M': small['M'], 'Y': small['Y'], 'fatal': fatal, 'closed_marker': str(Path(out)/'ANALYSIS-CASE-v1.json')}
