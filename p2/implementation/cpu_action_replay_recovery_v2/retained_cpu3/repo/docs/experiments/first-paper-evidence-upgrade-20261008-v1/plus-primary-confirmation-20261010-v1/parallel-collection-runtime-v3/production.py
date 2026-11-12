"""Production recorder: reuse v2 operators, no repeated startup qualification."""
import copy
import hashlib
import json
import os
import random
from pathlib import Path
import numpy as np
import orchestrator as O
from canonical_step_capture import rng_copy, rng_id, rng_arrays, observation_ids


class Store(O.Writer):
    """Native payloads including full no-pickle RNG string/scalar arrays."""
    def arrays(self, name, values, role):
        data, schema = {}, {}
        for key, value in values.items():
            O.need('/' not in key and '..' not in key, 'flat_array_name')
            a = np.asarray(value)
            O.need(not a.dtype.hasobject, 'no_object_arrays')
            data[key] = a.view(np.uint16) if a.dtype.name == 'bfloat16' else a
            numeric = a.dtype.kind in 'biufc' or a.dtype.name == 'bfloat16'
            schema[key] = {**O.array_id(a), 'stored_dtype': data[key].dtype.str,
                           'finite': bool(np.isfinite(a).all()) if numeric else None}
        path = self.path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('xb') as stream:
            np.savez(stream, **data, __schema__=np.asarray(json.dumps({'role': role, 'fields': schema}, sort_keys=True)))
        row = {'path': name, 'bytes': path.stat().st_size, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'role': role, 'fields': schema}
        self.files.append(row)
        return row

    def close_case(self, small):
        """Final atomic marker; no writes to this case directory may follow."""
        self.json('NATIVE-FILES-v1.json', {'files': self.files, 'archive_closed': False})
        payload = json.dumps(small, sort_keys=True, indent=2, allow_nan=False).encode() + b'\n'
        O.need(len(payload) <= 2 * 1024**2, 'small_analysis_envelope_bound')
        temp = self.path / '.ANALYSIS-CASE-v1.tmp'
        with temp.open('xb') as stream:
            stream.write(payload); stream.flush(); os.fsync(stream.fileno())
        O.need(not (self.path / 'ANALYSIS-CASE-v1.json').exists(), 'immutable_case_marker')
        os.rename(temp, self.path / 'ANALYSIS-CASE-v1.json')


def noise_for(record, query):
    spec = record['noise']
    O.need(spec['protocol'] == 'plus-shared-policy-noise-v1', 'explicit_noise_protocol')
    O.need(type(spec['parent_family_id']) is str and spec['parent_family_id'], 'explicit_parent_family_noise_identity')
    O.need(type(spec['noise_label']) is int and spec['noise_label'] in (0, 1), 'explicit_noise_label')
    O.need(type(query) is int and query >= 0, 'query_clock')
    # Deliberately independent of process-global NumPy and sensor streams.
    text = f"plus-shared-policy-noise-v1|{spec['parent_family_id']}|{spec['noise_label']}|{query}"
    seed = int.from_bytes(hashlib.sha256(text.encode('utf8')).digest()[:8], 'little')
    return np.random.Generator(np.random.PCG64(seed)).standard_normal((1, 10, 32)).astype(np.float32)


def branch(observation, prompt, physical, goal):
    return {'query': 0, 'observation': copy.deepcopy(observation), 'prompt': prompt,
            'physical_state': np.array(physical, copy=True), 'goal_binding_id': goal,
            'passive_output': None, 'passive_output_status': 'counterfactual_not_previously_executed'}


def invariant_call(method, b, noise, *args, _rng_audit=None, **kwargs):
    before = O.tree_id({'branch': b, 'noise': noise})
    saved_rng = rng_copy(); before_rng = rng_id(saved_rng)
    error = None
    try:
        value = method(*args, **kwargs)
    except BaseException as caught:
        error = caught
    finally:
        after_rng_state = rng_copy(); after_rng = rng_id(after_rng_state)
        try:
            after = O.tree_id({'branch': b, 'noise': noise})
        finally:
            np.random.set_state(copy.deepcopy(saved_rng['numpy']))
            random.setstate(copy.deepcopy(saved_rng['python']))
        restored_rng = rng_id(rng_copy())
        changed = {key: before_rng[key] != after_rng[key] for key in before_rng}
        audit = {'schema': 'plus-neural-call-rng-isolation-v2', 'method': getattr(method, '__name__', type(method).__name__),
                 'input_binding_before': before, 'input_binding_after': after, 'input_unchanged': before == after,
                 'rng_before': before_rng, 'rng_after_call': after_rng, 'rng_after_restore': restored_rng,
                 'rng_changed_during_call': changed, 'environment_rng_restored': restored_rng == before_rng,
                 'exception_type': type(error).__name__ if error else None, 'explicit_model_noise_identity': O.array_id(noise)}
        if _rng_audit is not None:
            payload = ({**{'before_' + k: v for k, v in rng_arrays(saved_rng).items()},
                        **{'after_' + k: v for k, v in rng_arrays(after_rng_state).items()}} if any(changed.values()) else None)
            _rng_audit(audit, payload)
        O.need(before == after, 'production_input_mutated')
        O.need(restored_rng == before_rng, 'production_environment_rng_restore_failed')
    if error is not None:
        raise error
    return value


def rng_auditor(store, prefix):
    count = [0]
    def audit(receipt, arrays):
        number = count[0]; count[0] += 1
        stem = prefix + f'/rng-call-{number:03d}'
        receipt['raw_side_effect_states'] = (store.arrays(stem + '.npz', arrays,
            'neural_software_ambient_rng_side_effect_before_restore') if arrays is not None else None)
        store.json(stem + '.json', receipt)
    return audit


def record_branch(b, noise, backend, store, prefix, *, full_pp=False, check=lambda: None):
    """One C, one T, one prefix and three single C/O endpoints; no source rescans."""
    check()
    audit = rng_auditor(store, prefix)
    store.arrays(prefix + '/input.npz', {**{'observation_' + k: v for k, v in b['observation'].items()},
                 'physical_state': b['physical_state'], 'noise': noise}, 'q0_actual_or_externally_qualified_reference_input')
    base = invariant_call(backend.baseline, b, noise, b, noise, _rng_audit=audit)
    store.arrays(prefix + '/C-actions.npz', {k: base[k] for k in ('raw_actions', 'actions')}, O.ROLES['baseline'])
    O.action_check(base)
    check()
    trajectory = invariant_call(backend.trajectory, b, noise, b, noise, _rng_audit=audit)
    store.arrays(prefix + '/T-trajectory.npz', trajectory['snapshot'], O.ROLES['trajectory'])
    store.arrays(prefix + '/T-actions.npz', {k: trajectory[k] for k in ('raw_actions', 'actions')}, O.ROLES['trajectory'])
    O.action_check(trajectory); O.trajectory_check(trajectory, noise, backend.layout)
    if not all(O.exact(base[key], trajectory[key]) for key in ('raw_actions', 'actions')):
        # A diagnostic graph mismatch does not become a behavior failure. First
        # recheck C after the trace and retain the actual bad T payload above.
        repeated = invariant_call(backend.baseline, b, noise, b, noise, _rng_audit=audit)
        store.arrays(prefix + '/C-after-unqualified-T.npz', {k: repeated[k] for k in ('raw_actions', 'actions')}, O.ROLES['baseline'])
        O.action_check(repeated)
        for key in ('raw_actions', 'actions'): O.hard_exact(base[key], repeated[key], 'canonical_repeat_after_T_' + key)
        report = {'trace_qualified': False, 'reason': 'T_vs_C_final_exact_failed', 'canonical_repeat_after_T_exact': True,
                  'native_H_adopted': False, 'bad_T_arrays_preserved': True, 'canonical_Y_permitted': True}
        store.json(prefix + '/RECORD-v1.json', report)
        return {'actions': base, 'views': {}, 'report': report, 'input_identity': O.request_binding(b),
                'physical_identity': O.array_id(b['physical_state']), 'time': np.asarray(trajectory['snapshot']['time'])}
    p = invariant_call(backend.prepare, b, noise, b, full_pp=full_pp, _rng_audit=audit)
    store.arrays(prefix + '/prefix.npz', O.prefix_arrays(p, full_pp), p['role'])
    store.arrays(prefix + '/model-inputs.npz', p['model_inputs'], 'O_prefix_actual_consumed_inputs')
    n, cache_id = O.prepare_check(p, backend.layout, full_pp)
    O.need(p['request_binding'] == O.request_binding(b), 'production_prefix_request_binding')
    views = {'T_H': {}, 'PG_H': {}, 'PG_Z': {}, 'P': {}, 'W': {}}
    for i, layer in enumerate(O.LAYERS):
        views['PG_H'][layer, None] = np.asarray(p['pg']['hidden_input'][i, 0])
        views['PG_Z'][layer, None] = np.asarray(p['pg']['hidden_pre_attention_norm'][i, 0])
    captures = {}
    for j in O.CALLS:
        check()
        s = trajectory['snapshot']
        c = invariant_call(backend.capture, b, noise, p, 'C', s, j, _rng_audit=audit)
        store.arrays(prefix + f'/call-{j:02d}-C.npz', O.capture_arrays(c), c['role'])
        o = invariant_call(backend.capture, b, noise, p, 'O', s, j, _rng_audit=audit)
        store.arrays(prefix + f'/call-{j:02d}-O.npz', O.capture_arrays(o), o['role'])
        O.need(c['request_binding'] == o['request_binding'] == p['request_binding'], 'production_capture_request')
        diff = O.capture_check(c, o, p, s['x_t'][j], s['time'][j], backend.layout, cache_id)
        captures[str(j)] = {'C_vs_O': diff, 'O_vs_T_finite_differences': {
            'hidden': O.measure(s['hidden'][j], o['native_trace']['hidden_post_block']),
            'readout': O.measure(s['readout'][j], o['final_readout']),
            'velocity': O.measure(s['velocity'][j], o['velocity'])},
            'O_same_graph_control_exact': True, 'independent_O_vs_T_exact_required': False}
        for i, layer in enumerate(O.LAYERS):
            views['T_H'][layer, j] = np.asarray(s['hidden'][j, i, 0])
            views['P'][layer, j] = np.asarray(o['native_trace']['probabilities_AP_AA'][i, 0])
            views['W'][layer, j] = np.asarray(o['native_trace']['attention_output_post_WO_pre_gate'][i, 0])
    report = {'N': n, 'H_primary_role': 'T_native_H_after_exact_final_gate', 'T_prefix_cache_identity': None,
              'T_loop_time_weak_type': None, 'O_used_cache_identity': cache_id,
              'O_prefix_relationship_to_T': 'same_request_separate_prefill_T_cache_unexposed',
              'full_pp': full_pp, 'PG_H_Z': 'instrumented_prefix_replay', 'calls': captures,
              'generation_method_calls': 2, 'prefix_method_calls': 1, 'single_suffix_method_calls': 6,
              'source_rescans_during_production': 0, 'C_T_final_exact': True, 'trace_qualified': True}
    store.json(prefix + '/RECORD-v1.json', report)
    return {'actions': base, 'views': views, 'report': report, 'input_identity': O.request_binding(b),
            'physical_identity': O.array_id(b['physical_state']), 'time': np.asarray(trajectory['snapshot']['time'])}


def record_canonical_only(b, noise, backend, store, prefix, check=lambda: None):
    check()
    store.arrays(prefix + '/input.npz', {**{'observation_' + k: v for k, v in b['observation'].items()},
                 'physical_state': b['physical_state'], 'noise': noise}, 'q0_actual_input_trace_unqualified')
    base = invariant_call(backend.baseline, b, noise, b, noise, _rng_audit=rng_auditor(store, prefix))
    store.arrays(prefix + '/C-actions.npz', {k: base[k] for k in ('raw_actions', 'actions')}, O.ROLES['baseline'])
    O.action_check(base)
    report = {'trace_qualified': False, 'reason': 'worker_startup_T_C_exact_unqualified', 'native_H_adopted': False,
              'canonical_Y_permitted': True}
    store.json(prefix + '/RECORD-v1.json', report)
    return {'actions': base, 'views': {}, 'report': report, 'input_identity': O.request_binding(b),
            'physical_identity': O.array_id(b['physical_state']), 'time': None}


def metric_rows(recorded, arm, reference=None):
    specs = {'T_H': ('AE', 'post_block', 'T_native'), 'PG_H': ('PG', 'pre_block', 'instrumented_prefix_replay'),
             'PG_Z': ('PG', 'post_pre_attention_RMSNorm', 'instrumented_prefix_replay'),
             'P': ('AE', 'full_key_softmax_probability', 'instrumented_shadow_O'),
             'W': ('AE', 'post_WO_pre_gate', 'instrumented_shadow_O')}
    rows = []
    def add(name, value, expert, site, role, layer, j, scope, component, unit):
        a = np.asarray(value).astype(np.float32)
        rows.append({'metric': name, 'expert': expert, 'site': site, 'recording_role': role, 'layer': layer,
                     'query_index': 0, 'generation_call': j, 'flow_time': float(recorded['time'][j]) if j is not None else None,
                     'action_scope': scope, 'component': component, 'arm': arm,
                     'value': float(np.sqrt(np.mean(a*a, dtype=np.float64))), 'unit': unit, 'status': 'measured', 'reason': None})
    for field, values in recorded['views'].items():
        for (layer, j), value in values.items():
            if reference is not None:
                value = np.asarray(value).astype(np.float32) - np.asarray(reference['views'][field][layer, j]).astype(np.float32)
            add(field + ('_delta_rms' if reference is not None else '_rms'), value, *specs[field], layer, j,
                'all10' if specs[field][0] == 'AE' else None, 'all', 'probability' if field == 'P' else 'native_activation')
    a = np.asarray(recorded['actions']['actions'])[0]
    if reference is not None:
        a = a - np.asarray(reference['actions']['actions'])[0]
    for scope, sl in (('all10', slice(None)), ('first5', slice(0, 5))):
        for component, dims in (('translation', slice(0, 3)), ('rotation', slice(3, 6)), ('gripper', slice(6, 7))):
            add('action_delta_rms' if reference is not None else 'action_rms', a[sl, dims], 'action', 'adapter_output',
                'canonical_native_sampler', None, None, scope, component, 'adapter_command_' + component)
    return rows

