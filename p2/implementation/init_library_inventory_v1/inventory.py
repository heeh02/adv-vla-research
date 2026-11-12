"""Exact initialization identities; no task execution or independence inference."""
import hashlib
import json
import numpy as np


def need(ok, code):
    if not ok:
        raise ValueError(code)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def vector_identity(value):
    a = np.asarray(value)
    need(a.ndim == 1 and a.dtype.kind == 'f' and np.isfinite(a).all(), 'finite_native_float_vector')
    need(a.dtype.itemsize <= 8, 'lossless_float64_numeric_identity_required')
    header = {'shape': list(a.shape), 'dtype': a.dtype.str, 'order': 'C'}
    raw = a.tobytes(order='C')
    return {**header, 'bytes': a.nbytes,
            'sha256': hashlib.sha256(canonical(header) + b'\0' + raw).hexdigest(),
            'raw_sha256': hashlib.sha256(raw).hexdigest(), 'hash_domain': 'plus-policy-native-array-v1'}


def inventory(values, library_ref, reference_roots):
    need(0 < len(values) <= 1000, 'bounded_nonempty_library')
    typed, numeric, rows = {}, {}, []
    digest = hashlib.sha256(str(len(values)).encode() + b'\0')
    for i, value in enumerate(values):
        if hasattr(value, 'detach'):
            need(value.device.type == 'cpu', 'CPU_tensor_only')
            value = value.detach().numpy()
        a = np.asarray(value)
        ref = vector_identity(a)
        digest.update(a.dtype.str.encode() + b'\0' + repr(tuple(a.shape)).encode() + b'\0' + a.tobytes(order='C'))
        # A separate exact-value identity normalizes signed zero; no distance tolerance.
        v = a.astype('<f8', copy=True)
        v[v == 0] = 0.0
        numeric_hash = hashlib.sha256(canonical({'shape': list(v.shape), 'dtype': '<f8', 'zero_sign_normalized': True}) + b'\0' + v.tobytes()).hexdigest()
        first_typed = typed.setdefault(ref['sha256'], i)
        first_numeric = numeric.setdefault(numeric_hash, i)
        matches = []
        for r in reference_roots:
            s = r['selection']; source = s['source_library']
            if source['sha256'] == library_ref['sha256'] and s['native_initial_vector']['sha256'] == ref['sha256']:
                matches.append(r['root_id'])
        rows.append({'library_index': i, 'identity': ref, 'exact_value_sha256_zero_sign_normalized': numeric_hash,
                     'first_typed_equal_index': first_typed, 'first_numeric_equal_index': first_numeric,
                     'matches_reviewed_DEV_roots': matches,
                     'prospective_unseen_eligibility': 'NOT_ESTABLISHED'})
    cache_hash = digest.hexdigest()
    checks = []
    for r in reference_roots:
        s = r['selection']; source = s['source_library']
        if source['sha256'] != library_ref['sha256']:
            continue
        need(source['bytes'] == library_ref['bytes'] and 'official/' + source['path'] == library_ref['path'], 'reference_library_identity')
        i = s['original_library_index']
        need(type(i) is int and 0 <= i < len(rows), 'reference_library_index')
        actual = rows[i]['identity']; expected = s['native_initial_vector']
        need(all(actual[k] == expected[k] for k in expected), 'reference_native_vector_exact')
        need(cache_hash == s['original_init0_proof']['cached_states_native_typed_sha256'], 'whole_library_cached_identity_exact')
        checks.append({'root_id': r['root_id'], 'library_index': i, 'native_vector_identity_exact': True,
                       'whole_library_cached_identity_exact': True})
    return {'library': library_ref, 'rows': rows, 'row_count': len(rows),
            'typed_unique_rows': len(typed), 'exact_numeric_unique_rows': len(numeric),
            'cached_states_native_typed_sha256': cache_hash,
            'not_an_independence_or_unexposed_population_claim': True}, checks
