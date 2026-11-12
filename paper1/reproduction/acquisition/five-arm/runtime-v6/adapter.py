"""Plus loop and batch delegate. No model loading, neural forward, or env rewrite."""
import copy
import hashlib
import time

import numpy as np

import plus_contract as C

IMAGE_KEYS = ('agentview_image', 'robot0_eye_in_hand_image')
STATE_SHAPES = {'robot0_eef_pos': (3,), 'robot0_eef_quat': (4,), 'robot0_gripper_qpos': (2,)}
DUMMY = [0.0]*6 + [-1.0]


class ActionValidationError(ValueError):
    def __init__(self, message, arrays):
        super().__init__(message)
        self.arrays = {k: np.array(v, copy=True) for k, v in arrays.items() if not np.asarray(v).dtype.hasobject}


class NonfiniteActionError(FloatingPointError):
    def __init__(self, arrays):
        super().__init__('nonfinite canonical action; unresolved outcome')
        self.arrays = {k: np.array(v, copy=True) for k, v in arrays.items() if not np.asarray(v).dtype.hasobject}


def array_identity(value):
    a = np.asarray(value)
    C.need(not a.dtype.hasobject, 'object arrays are prohibited')
    header = {'shape': list(a.shape), 'dtype': a.dtype.str, 'order': 'C'}
    return {**header, 'bytes': a.nbytes, 'sha256': hashlib.sha256(C.canonical(header)+b'\0'+a.tobytes(order='C')).hexdigest(),
            'hash_domain': 'plus-policy-native-array-v1'}


def observation_identity(obs):
    result = {}
    for key in IMAGE_KEYS:
        a = np.asarray(obs[key])
        C.need(a.shape == (256, 256, 3) and a.dtype == np.uint8, 'actual official dual-camera RGB ABI differs')
        result[key] = array_identity(a)
    for key, shape in STATE_SHAPES.items():
        a = np.asarray(obs[key])
        C.need(a.shape == shape and a.dtype.kind == 'f' and np.isfinite(a).all(), 'native numeric state ABI differs')
        result[key] = array_identity(a)
    return result


def noise_for(case, query):
    C.need(type(query) is int and query >= 0, 'nonnegative query required')
    # The old 10*suite+task packing COLLIDES for Plus ids >9. Keep independent
    # tuple components. Model identity is deliberately absent: paired models
    # receive exactly the same explicit noise on the same variant/query.
    seed = np.random.SeedSequence([20261008, 1, C.SUITES.index(case['suite']), case['classification_id'], 0, query, 0])
    return np.random.default_rng(seed).standard_normal((1, 10, 32)).astype(np.float32)


def sensor_rng_identity():
    name, keys, position, has_gauss, cached = np.random.get_state()
    return C.digest({'name': name, 'keys': array_identity(keys), 'position': position,
                     'has_gauss': has_gauss, 'cached_gaussian': cached})


class PolicyAdapter:
    """An already-loaded CampaignPolicy or compatible infer_batch callable.

    Supports native batch N. The caller owns model lifetime and batching; this
    class never instantiates a model, casts its parameters, or calls sample.
    """
    def __init__(self, loaded_policy):
        self.call = loaded_policy.infer_batch if hasattr(loaded_policy, 'infer_batch') else loaded_policy
        C.need(callable(self.call), 'existing infer_batch callable required')

    def infer_batch(self, observations, prompts, noise):
        C.need(len(observations) > 0 and len(observations) == len(prompts), 'nonempty aligned batch required')
        for obs, prompt in zip(observations, prompts):
            observation_identity(obs)
            C.need(type(prompt) is str and bool(prompt), 'official task.language required')
        noise = np.asarray(noise)
        C.need(noise.shape == (len(observations), 10, 32) and noise.dtype == np.float32 and np.isfinite(noise).all(), 'explicit native float32 noise ABI differs')
        # Canonical quaternion preprocessing clips in place: protect the real
        # official wrapper observation and any caller-owned RNG arrays.
        result = self.call(copy.deepcopy(observations), list(prompts), noise.copy(), capture_scalars=False)
        self.validate_output(result, len(observations))
        return result

    @staticmethod
    def validate_output(result, batch_size):
        arrays = {key: np.asarray(result[key]) for key in ('raw_actions', 'actions') if key in result}
        for name, tail, dtype in [('raw_actions', (10, 32), np.float32), ('actions', (10, 7), np.float64)]:
            a = np.asarray(result[name])
            if a.shape != (batch_size, *tail) or a.dtype != dtype:
                raise ActionValidationError('canonical raw/adapter action ABI differs', arrays)
            if not np.isfinite(a).all():
                raise NonfiniteActionError(arrays)

    def check_same_input(self, observations, prompts, noise, actual):
        """Q1 only: separate canonical call, same actual input/noise. No clean
        substitute and no requirement that two DIFFERENT variants act alike.
        """
        reference = self.call(copy.deepcopy(observations), list(prompts), noise.copy(), capture_scalars=False)
        self.validate_output(reference, len(observations))
        for key in ('raw_actions', 'actions'):
            a, b = np.asarray(actual[key]), np.asarray(reference[key])
            if a.dtype != b.dtype or a.shape != b.shape or a.tobytes() != b.tobytes():
                arrays = {f'{side}_{k}': value[k] for side, value in [('candidate', actual), ('reference', reference)] for k in ('raw_actions', 'actions')}
                raise ActionValidationError('same-input canonical parity differs', arrays)
        return {'status': 'PASS_exact_same_input_noise_bytes', 'canonical_forward_calls': 2,
                'independent_neural_implementation': False, 'different_variant_action_invariance_required': False}


def strict_bool(value, label):
    C.need(type(value) in (bool, np.bool_), label + ' must be an actual boolean; unknown cannot become failure')
    return bool(value)


def run_case(case, model, loaded_policy, env, *, purpose, deadline, protocol_mode='released_task_language', attempt=0, now=time.time, evidence=None):
    """One env with an ALREADY loaded policy; reusable by a persistent worker.

    ``evidence(name, arrays)`` may archive native arrays in cloud tmp/BOS. Q1 is
    exactly one distinct policy input, then at most five controls (terminal may
    stop sooner); two same-input forwards establish delegation parity. Formal
    mode is a reusable loop, not a CLI authorization to launch F0.
    """
    C.need(purpose in ('qualification', 'formal'), 'explicit analysis role required')
    C.need(protocol_mode in C.PROMPT_MODES and env.protocol_mode == protocol_mode, 'environment and requested prompt protocol differ')
    policy = loaded_policy if isinstance(loaded_policy, PolicyAdapter) else PolicyAdapter(loaded_policy)
    record = {'schema': 'plus-policy-outcome-v1', 'id': C.episode_id(case, model, protocol_mode), 'case': case, 'model': model,
              'protocol_mode': protocol_mode,
              'attempt': attempt, 'purpose': purpose, 'status': 'unresolved', 'success': None,
              'observed_task_success': None, 'policy_steps': 0, 'wait_steps': 0, 'queries': 0,
              'nonfinite_action_count': 0, 'exception': '', 'error_type': None, 'process_lifecycle': None,
              'input_indices': [], 'action_source': 'canonical_native_sampler', 'official_environment_source': None}
    def check():
        if now() >= deadline:
            raise TimeoutError('absolute case/science deadline')
    try:
        check()
        obs = env.reset(case)  # exact official task/get_task_init_states/variant wrapper
        record['official_environment_source'] = env.source
        for _ in range(10):
            check()
            obs, terminal = env.step(DUMMY)
            strict_bool(terminal, 'wait terminal')
            record['wait_steps'] += 1
        # Do not use reset/set_init_state or raw sim.render for policy input:
        # this is the 10th OFFICIAL wrapper.step obs, including Sensor Noise.
        maximum = 5 if purpose == 'qualification' else case['max_policy_steps']
        while record['policy_steps'] < maximum:
            check()
            before = observation_identity(obs)
            rng_before = sensor_rng_identity()
            noise = noise_for(case, record['queries'])
            record['input_indices'].append({'query': record['queries'], 'policy_step': record['policy_steps'],
                                           'source': 'official_wrapper_step_return', 'observation': before,
                                           'prompt': env.prompt, 'noise': array_identity(noise),
                                           'official_sensor_numpy_rng_unchanged': None, 'sensor_rng_sha256': rng_before})
            if evidence:
                arrays = {key: np.asarray(obs[key]) for key in (*IMAGE_KEYS, *STATE_SHAPES)}
                evidence(f"query-{record['queries']:04d}-input", {**arrays, 'noise': noise})
            result = policy.infer_batch([obs], [env.prompt], noise)
            C.need(observation_identity(obs) == before, 'policy changed official cached observation')
            if purpose == 'qualification':
                record['same_input_parity'] = policy.check_same_input([obs], [env.prompt], noise, result)
                C.need(observation_identity(obs) == before, 'parity changed official cached observation')
            C.need(sensor_rng_identity() == rng_before, 'policy changed global NumPy RNG consumed by official sensor noise')
            record['input_indices'][-1]['official_sensor_numpy_rng_unchanged'] = True
            if evidence:
                evidence(f"query-{record['queries']:04d}-output", {'raw_actions': result['raw_actions'], 'actions': result['actions']})
            record['queries'] += 1
            for action in result['actions'][0, :5]:
                check()
                if record['policy_steps'] == maximum:
                    break
                obs, terminal = env.step(action.tolist())
                record['policy_steps'] += 1
                done = strict_bool(terminal, 'official terminal')
                success = strict_bool(env.check_success(), 'official goal predicate')
                record['observed_task_success'] = success
                if done or success:
                    break
            if done or success:
                break
        record['status'] = 'qualified' if purpose == 'qualification' else 'complete'
        # A Q1 five-step truncation is never entered into the formal SR denominator.
        record['success'] = None if purpose == 'qualification' else record['observed_task_success']
    except Exception as error:
        if evidence and getattr(error, 'arrays', None):
            try:
                evidence('failure-action-bytes', error.arrays)
                record['failure_action_arrays_preserved'] = True
            except Exception as archive_error:
                record['failure_action_archive_error_type'] = type(archive_error).__name__
        record['status'] = 'deadline' if isinstance(error, TimeoutError) else 'infrastructure_error'
        if isinstance(error, FloatingPointError):
            record['status'] = 'nonfinite_action'
            record['nonfinite_action_count'] = 1
        record['success'] = None
        record['error_type'] = type(error).__name__
        # No arbitrary exception text/locals/credentials in durable records.
        record['exception'] = type(error).__name__
    finally:
        try:
            env.close()
        except Exception as error:
            record['status'], record['success'], record['error_type'] = 'infrastructure_error', None, type(error).__name__
            record['exception'] = type(error).__name__
    return record


def scorer_row(outcome, lifecycle):
    """Only a real parent-process receipt closes runtime fields for scoring."""
    C.need(outcome['purpose'] == 'formal' and outcome['status'] == 'complete' and type(outcome['success']) is bool, 'only complete formal outcomes enter scorer')
    C.need(set(lifecycle) == {'actual_exit_code', 'timed_out', 'outcome_sha256'}, 'explicit parent lifecycle fields required')
    C.need(type(lifecycle['actual_exit_code']) is int and lifecycle['actual_exit_code'] == 0
           and type(lifecycle['timed_out']) is bool and lifecycle['timed_out'] is False
           and lifecycle['outcome_sha256'] == C.digest(outcome), 'actual successful subprocess closure required')
    C.need(outcome['exception'] == '' and outcome['nonfinite_action_count'] == 0, 'unresolved runtime event blocks scorer')
    c = outcome['case']
    C.need(type(outcome['policy_steps']) is int and 0 < outcome['policy_steps'] <= c['max_policy_steps']
           and outcome['wait_steps'] == 10, 'formal wait/control counts differ')
    return {'task_suite': c['suite'], 'task_id': c['task_id'], 'task_name': c['task_name'], 'episode_idx': 0,
            'max_policy_steps': c['max_policy_steps'], 'policy_steps': outcome['policy_steps'],
            'success': outcome['success'], 'crashed': False, 'subprocess_timed_out': False,
            'nonfinite_action_count': 0, 'exception': '', 'model_variant_id': outcome['id'],
            'model_identity_sha256': C.digest(outcome['model']), 'protocol_mode': outcome['protocol_mode']}


def scorer_rows(outcomes, lifecycles, *, expected_cases, model, protocol_mode, classification):
    """Do not let a valid-looking mixed-model/mixed-prompt suite reach scorer."""
    C.model_identity(model)
    for case in expected_cases:
        C.validate_case(case, classification)
    expected = {C.episode_id(case, model, protocol_mode): case for case in expected_cases}
    C.need(len(expected) == len(expected_cases) and len(outcomes) == len(lifecycles) == len(expected), 'complete unique planned population required')
    seen, result = set(), []
    for outcome, life in zip(outcomes, lifecycles):
        eid = outcome['id']
        C.need(eid in expected and eid not in seen and outcome['case'] == expected[eid]
               and outcome['model'] == model and outcome['protocol_mode'] == protocol_mode,
               'wrong/duplicate/mixed model or prompt protocol outcome')
        seen.add(eid)
        result.append(scorer_row(outcome, life))
    return result
