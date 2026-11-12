"""Pinned original-source adapters; no cloud/auth/model/environment operations."""
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
import importlib.util
import json
import sys
import numpy as np
import ablation as A

PINS = {
    'helper.py': '3b916618d1cd27f3bdb9918ad89a916a2ecac9a05cf8b6d3c53972bcf0897b75',
    'taskblind_prediction.py': 'f1ffaf7204d0ecb114397efd0fe9fc35a03ed967da8fe588d29f3539bc258436',
    'retained_prediction.py': '7b5c03d4ff364034a7bf196502124a8fb16d3d9241c107f649a6d91afba23114',
    'dev_reader.py': '4f7c6db238e5a43a534f8aa01dc4ba176bb7d981fa092035e35bfe37bad6d16c',
    'dev_worker.py': '6fdb9011035c48b698b126f16b822f66e8c484736146706e356285495de2f45d',
    'retained_replication.py': 'd3e3204863c20c160022f9ae2282e2f47567e41dda0b6b8b7454e962a034292c',
    'native_reader.py': '231e4814aa5466c1557753ab3d8f31a8478d2d8f0138111e96123e404cd64843',
}
QUAL_REFS = {
    'interface': {'path': 'PLOT-INTERFACE-v1.json', 'bytes': 12814, 'sha256': '8462f8363fa7f072dc9720e12ce7d26835b449fc375f8159cd2cb485ee2330d4'},
    'arrays': {'path': 'COMPACT-NUMERIC-v1.npz', 'bytes': 322368, 'sha256': '0d76f74286345b4e15b57cdcb200567c245f54ec9e77a279ba832eeb86a90af9'},
}


@contextmanager
def aliases(values):
    previous = {k: sys.modules.get(k) for k in values}
    sys.modules.update(values)
    try:
        yield
    finally:
        for k, v in previous.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def dependencies(paths):
    """Paths are original files, or byte-identical basename-mapped package files."""
    A.need(set(paths) == set(PINS), 'exact_seven_retained_source_files')
    for name, sha in PINS.items():
        A.need(A.digest(Path(paths[name]).read_bytes()) == sha, 'retained_source_SHA_' + name.replace('.', '_'))
    H = load('_context_ablation_original_helper', paths['helper.py'])
    P = H.P
    with aliases({'prediction': P}):
        R = load('_context_ablation_original_dev_reader', paths['dev_reader.py'])
        with aliases({'reader': R}):
            W = load('_context_ablation_original_dev_worker', paths['dev_worker.py'])
    N = load('_context_ablation_original_native_reader', paths['native_reader.py'])
    return SimpleNamespace(H=H, P=P, R=R, W=W, N=N)


def real_guard(synthetic):
    A.need(synthetic or sys.platform.startswith('linux'), 'real_neural_array_processing_cloud_Linux_only')


def row(D, model, query, effect, *, split, background, label_identity, proof):
    features = model.features(query, 'shared')
    A.need(features['current'] is not None and features['pooled'] is not None, 'defined_both_frozen_write_features')
    return A.Row((query.root_id, query.query_index, 'shared', background), split,
                 features['current'], features['pooled'], effect, not np.any(query.edit('shared')),
                 {'input_identity': query.fingerprint(), 'edit_identity': query.edit_identity('shared'),
                  'label_identity': label_identity, **proof})


def dev_query_row(D, model, group, *, source_run_id):
    """Source-level single-query seam, also exercised by real-codec fixtures."""
    P, R = D.P, D.R
    query, targets = R.build_query(group, model.plan)
    inputs = {(r['root_id'], r['query_index']): r['sha256'] for r in model.fit_inputs}
    labels = {tuple(r['key']): r for r in model.fit_labels}
    A.need(query.fingerprint() == inputs[query.root_id, query.query_index], 'reconstructed_DEV_query_matches_frozen_fit')
    for target in targets:
        old = labels[target.key]
        A.need(old['source_identity'] == target.source_identity and old['effect'] == P.array_id(target.effect), 'DEV_target_matches_original_fit_identity')
    target = next(t for t in targets if t.arm == 'shared')
    return row(D, model, query, target.effect, split='DEV', background='KT', label_identity=target.source_identity,
               proof={'source_run_id': source_run_id, 'original_readset_sha256': model.plan.readset_sha256})


def dev_rows(D, model, plan, read_exact, *, check=lambda: None, synthetic=False):
    """Original 695-object reader/gates, then ONLY 48 shared rows for fitting.

    read_exact(ref, callback) must preserve whole-object/archive verification;
    callback receives an exact temporary Path. No new cache is required.
    """
    real_guard(synthetic); model.verify(); P, R, W = D.P, D.R, D.W
    A.need(synthetic or model.identity == A.FIT_ID, 'actual_original_fit')
    refs = W.validate_plan(plan)
    A.need(P.sha(P.encoded(plan)) == model.plan.readset_sha256, 'original_DEV_readset_frozen_binding')
    groups, extras, calls, comparisons = {}, {}, [], []
    for ref in refs:
        check(); rec = read_exact(ref, lambda p: R.extract(p, ref)); ev = rec['event']; m = rec['metadata']
        if ev in ('CAMPAIGN_PLANNED', 'READ_ONLY_WO', 'RESULT'):
            A.need(ev not in extras, 'unique_DEV_global_event'); extras[ev] = rec
        elif ev == 'EXACT_COMPARISON':
            comparisons.append(rec)
        else:
            key = m['root_id'], m['query_index']; A.need(key in model.plan.keys(), 'original_DEV_query_only')
            slot = m['role'] if ev == 'CALL_OUTPUT' else 'label_' + m['arm'] if ev == 'ACTUAL_EFFECT' else ev
            group = groups.setdefault(key, {}); A.need(slot not in group, 'unique_DEV_query_event'); group[slot] = rec
            if ev == 'CALL_OUTPUT':
                calls.append(rec)
    gates = W.original_gates(extras['RESULT']['metadata'], comparisons, calls)
    planned = extras['CAMPAIGN_PLANNED']['metadata']
    A.need(planned['synthetic'] is False and P.sha(P.encoded(planned['plan'])) == model.plan.intervention_contract_sha256, 'original_DEV_intervention_contract')
    A.need(extras['READ_ONLY_WO']['values']['WO'].manifest() == model.wo.manifest(), 'original_DEV_WO_matches_frozen')
    output = []
    for key in model.plan.keys():
        check(); output.append(dev_query_row(D, model, groups[key], source_run_id=plan['source_run_id']))
        groups.pop(key)
    return output, {'original_gates': gates, 'reconstructed_original_targets': 192, 'selected_shared_targets': 48,
                    'read_objects': len(refs), 'read_body_bytes': sum(r['bytes'] for r in refs), 'new_fit_or_PCA_in_adapter': False}


def qualification(read_exact, refs, *, synthetic=False):
    """Reuse actual cpu-readout-v3 qualification, never infer it from finiteness.

    Both small artifacts stay cloud-side. The immutable original arrays provide
    the 192-key qualification/target table; only 120 shared keys are selected.
    """
    real_guard(synthetic)
    def checked(path, kind, fn):
        p = Path(path); raw = p.read_bytes(); expected = QUAL_REFS[kind]
        A.need(synthetic or (len(raw) == expected['bytes'] and A.digest(raw) == expected['sha256']), 'original_qualification_' + kind + '_SHA')
        return fn(p)
    doc = read_exact(refs['interface'], lambda p: checked(p, 'interface', lambda f: json.loads(f.read_bytes())))
    def arrays(p):
        with np.load(p, allow_pickle=False) as z:
            return {k: np.array(z[k], copy=True) for k in z.files}
    data = read_exact(refs['arrays'], lambda p: checked(p, 'arrays', arrays))
    A.need(doc['schema'] == 'task-heldout-compact-plot-v1' and doc['fit_identity'] == A.FIT_ID, 'original_qualification_fit')
    A.need(synthetic or all(doc['array_ref'][k] == QUAL_REFS['arrays'][k] for k in ('path', 'bytes', 'sha256')), 'qualification_array_reference')
    keys = [tuple(k) for k in doc['keys']]
    A.need(len(keys) == len(set(keys)) == 192 and set(A.CONF_KEYS) <= set(keys), 'original_192_and_selected120_keys')
    A.need(data['actual_effect'].shape == (192, 10, 7) and data['actual_effect'].dtype == np.float64, 'qualification_effect_ABI')
    for name in ('qualified', 'zero_actual_E', 'zero_actual_E_known'):
        A.need(data[name].shape == (192,) and data[name].dtype == bool, 'qualification_flags_ABI')
    return {key: {'qualified': bool(data['qualified'][i]), 'effect': data['actual_effect'][i],
                  'zero_actual_E': bool(data['zero_actual_E'][i]), 'zero_known': bool(data['zero_actual_E_known'][i]),
                  'qualification_source': dict(QUAL_REFS['arrays'])} for i, key in enumerate(keys) if key in A.CONF_KEYS}


def conf_rows(D, model, *, binding, common_records, edits, labels, seal_rows, qualified, synthetic=False):
    """One original query -> KT row, plus KC row only at its original anchors."""
    real_guard(synthetic); H = D.H; P = D.P; model.verify()
    A.need(synthetic or model.identity == A.FIT_ID, 'actual_original_fit')
    common_records, labels, seal_rows = list(common_records), list(labels), list(seal_rows)
    A.need(all(r.synthetic == synthetic for r in (binding, edits, *common_records, *labels)), 'no_mixed_real_synthetic_records')
    root, q = binding.metadata['root_id'], binding.metadata['query_index']
    common = H.common_roles(common_records, root, q, binding.metadata['source']['source_identity'])
    query = H.rebuild_query(binding, common, edits)
    chosen = {(r.metadata['root_id'], r.metadata['query_index'], r.metadata['arm'], r.metadata['reading_context']): r
              for r in labels if r.metadata['arm'] == 'shared'}
    expected = [k for k in A.CONF_KEYS if k[:2] == (root, q)]
    A.need(expected and len(chosen) == sum(r.metadata['arm'] == 'shared' for r in labels) and set(chosen) == set(expected), 'complete_unique_original_shared_labels')
    seals = {tuple(r['key']): r for r in seal_rows}
    A.need(len(seals) == len(seal_rows) and set(expected) <= set(seals), 'unique_original_sealed_targets')
    output = []
    for key in expected:
        label, sealed = chosen[key], seals[key]
        qv, base, edited = H.bind_target(query, common, edits, label, sealed)
        y = label.field('signed_adapter7_effect')
        a, b = edited.field('adapter7'), base.field('adapter7')
        A.need(y.dtype == a.dtype == b.dtype == 'float64' and y.shape == (10, 7) and a.shape == b.shape == (1, 10, 7), 'original_adapter_effect_ABI')
        A.need(H.same(y.storage, np.subtract(a.storage, b.storage, dtype=np.float64)[0]), 'label_equals_actual_endpoint_difference')
        old = qualified[key]
        A.need(old['qualified'] is True and old['zero_known'] is True and old['zero_actual_E'] == sealed['zero_actual_E'], 'original_target_qualified_and_zero_known')
        A.need(H.same(y.storage, old['effect']), 'target_matches_original_qualified_readout')
        output.append(row(D, model, qv, y.storage, split='CONF', background=key[3], label_identity=label.ref['sha256'],
                          proof={'source_CALL': H.ref(edited.ref), 'baseline_CALL': H.ref(base.ref), 'edit_ref': H.ref(edits.ref),
                                 'original_qualification': old['qualification_source']}))
    if len(output) == 2:
        A.need(np.array_equal(output[0].current[:9], output[1].current[:9]) and np.array_equal(output[0].pooled, output[1].pooled), 'KC_same_context_E_and_pooled_features')
    return output
