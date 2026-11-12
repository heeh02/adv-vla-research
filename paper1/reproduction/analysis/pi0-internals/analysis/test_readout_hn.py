"""Synthetic-only tests for readout_hn.py (no data, no cloud, no model).

  * POS=10 parity: readout_hn (legacy_math_hn) with the pi0.5 'old' bases reproduces the
    frozen producer readout.py (legacy_math) bit-for-bit on identical synthetic inputs.
  * POS=50: shapes/structure for pi0 (full=51200, new_response=800, flow=1600,
    prediction=1950, position_mean (50,1024)); old bases omitted; driver runs end to end.
"""
import hashlib
import importlib.util
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ORIG = Path("<repo>/docs/reviews/v7_19-plus-first-consolidation-20261017/"
            "current-basis-replay/producer-reference")

FAMILIES = ('Background Textures', 'Camera Viewpoints', 'Light Conditions', 'Sensor Noise')


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def ortho(dim, k, seed):
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.standard_normal((dim, k)))
    return q[:, :k]


def basis32(seed):
    b = np.zeros((1024, 32))
    b[:, :16] = ortho(1024, 16, seed)
    return b


def fold_of(component):
    FOLD_SEED = 'plus-shared-channel-whole-task-fold-v1'
    return int.from_bytes(hashlib.sha256((FOLD_SEED + '|' + component).encode()).digest()[:8], 'little') % 5


def comp_of(task):
    return hashlib.sha256(('component|' + task).encode()).hexdigest()


def make_dataset(POS, seed=7):
    held = 'Camera Viewpoints'; target_fold = 0
    other_fams = [f for f in FAMILIES if f != held]
    test_tasks, train_tasks = [], []
    i = 0
    while len(test_tasks) < 3 or len(train_tasks) < 6:
        t = f't{i}'; f = fold_of(comp_of(t))
        if f == target_fold and len(test_tasks) < 3:
            test_tasks.append(t)
        elif f != target_fold and len(train_tasks) < 6:
            train_tasks.append(t)
        i += 1
    rows, metadata = [], {}
    cid = 0
    def add(task, family):
        nonlocal cid
        case_id = f'case{cid:04d}'; cid += 1
        row = {'case_id': case_id, 'task_id': task, 'family': family,
               'clean_id': f'clean-{case_id}', 'row_id': f'var-{case_id}', 'noise_label': cid % 2,
               'suite': 'libero_10'}
        rows.append(row); metadata[case_id] = row
        return case_id
    for k, t in enumerate(test_tasks):
        add(t, held); add(t, held)                    # 2 cases per test task
    for k, t in enumerate(train_tasks):
        add(t, other_fams[k % len(other_fams)])        # 1 case per train task
    components = {r['task_id']: comp_of(r['task_id']) for r in rows}
    rng = np.random.default_rng(seed)
    n = len(rows)
    features = {
        'hidden_delta': 0.1 * rng.standard_normal((n, 3, POS, 1024)),
        'target': 0.1 * rng.standard_normal((n, POS * 32 + POS * 7)),
        'flow0_delta': 0.1 * rng.standard_normal((n, POS * 32)),
    }
    new = {'position_delta__basis32': basis32(1), 'position_clean__basis32': basis32(2),
           'curve_random32': basis32(3), 'position_mean': rng.standard_normal((POS, 1024))}
    old = {'delta_uncentered_PCA': ortho(1024, 16, 11),
           'same_rows_clean_uncentered_PCA': ortho(1024, 16, 12),
           'fixed_seed_random_orthogonal': ortho(1024, 16, 13)}
    allowed = {r['case_id'] for r in rows}
    return features, rows, components, new, old, metadata, allowed, held, target_fold


def build_support(M, rows, components, family, fold, layer):
    tr, te, _ = M.split_rows(rows, family, 'exploratory_unseen_task_cross_family', fold, components)
    tr_rows, te_rows = [rows[i] for i in tr], [rows[i] for i in te]
    return {'family': family, 'mode': 'exploratory_unseen_task_cross_family', 'fold': fold, 'layer': layer,
            'training_case_count': len(tr), 'test_case_count': len(te),
            'training_case_ids_sha256': M.case_set_sha(tr_rows),
            'test_case_ids_sha256': M.case_set_sha(te_rows)}, len(tr), len(te)


def test_pos10_parity():
    orig_math = load('legacy_math', ORIG / 'legacy_math.py')
    orig_readout = load('readout', ORIG / 'readout.py')
    Mhn = load('legacy_math_hn', HERE / 'legacy_math_hn.py')
    Rhn = load('readout_hn', HERE / 'readout_hn.py')
    Mhn.set_positions(10)
    feats, rows, comps, new, old, meta, allowed, fam, fold = make_dataset(10)
    layer = 8
    s_hn, ntr, nte = build_support(Mhn, rows, comps, fam, fold, layer)
    s_or, _, _ = build_support(orig_math, rows, comps, fam, fold, layer)
    assert s_hn == s_or, "support mismatch between ports"
    c_hn, g_hn, p_hn = Rhn.predict_split(feats, rows, comps, {'support': s_hn}, old, new)
    c_or, g_or, p_or = orig_readout.predict_split(feats, rows, comps, {'support': s_or}, old, new)
    assert len(c_hn) == len(c_or) and len(g_hn) == len(g_or), (len(c_hn), len(c_or))
    # compare BY KEY (predictor dict insertion order differs between ports; values must match)
    kc = lambda r: (r['model'], r['metric'], r['case_id'])
    dh, do = {kc(r): r for r in c_hn}, {kc(r): r for r in c_or}
    assert set(dh) == set(do), (len(dh), len(do))
    for k in dh:
        a, b = dh[k], do[k]
        assert np.isfinite(a['mse']) and np.isfinite(b['mse']), (k, a['mse'], b['mse'])
        assert abs(a['mse'] - b['mse']) <= 1e-12 + 1e-9 * abs(b['mse']), (k, a, b)
        assert abs(a['target_energy'] - b['target_energy']) <= 1e-12 + 1e-9 * abs(b['target_energy'])
    kg = lambda r: (r['model'], r['case_id'])
    gh, go = {kg(r): r for r in g_hn}, {kg(r): r for r in g_or}
    assert set(gh) == set(go)
    for k in gh:
        for f in ('sse', 'raw_energy', 'residual_energy'):
            assert abs(gh[k][f] - go[k][f]) <= 1e-12 + 1e-9 * abs(go[k][f]), (f, k)
    assert p_hn['feature_dimensions'] == p_or['feature_dimensions']
    assert p_hn['prediction_shape'] == p_or['prediction_shape'] == [nte, 390]
    # task/summary parity
    t_hn = Rhn.task_rows(c_hn, meta, comps, allowed); t_or = orig_readout.task_rows(c_or, meta, comps, allowed)
    e_hn, k_hn = Rhn.summary(t_hn); e_or, k_or = orig_readout.summary(t_or)
    assert len(e_hn) == len(e_or) and len(k_hn) == len(k_or), (len(e_hn), len(e_or), len(k_hn), len(k_or))
    ke = lambda r: (r['family'], r['layer'], r['metric'], r['model'])
    deh, deo = {ke(r): r for r in e_hn}, {ke(r): r for r in e_or}
    assert set(deh) == set(deo)
    for k in deh:
        assert abs(deh[k]['task_equal_mse'] - deo[k]['task_equal_mse']) <= 1e-12 + 1e-9 * abs(deo[k]['task_equal_mse']), k
    print(f"POS=10 parity PASS: {len(c_hn)} case-scores, {len(e_hn)} estimates, {len(k_hn)} contrasts match frozen readout.py")


def test_pos50_shapes():
    Mhn = load('legacy_math_hn', HERE / 'legacy_math_hn.py')
    Rhn = load('readout_hn', HERE / 'readout_hn.py')
    Mhn.set_positions(50)
    feats, rows, comps, new, old, meta, allowed, fam, fold = make_dataset(50)
    layer = 8
    s, ntr, nte = build_support(Mhn, rows, comps, fam, fold, layer)
    cases, geom, proof = Rhn.predict_split(feats, rows, comps, {'support': s}, None, new)  # old=None for pi0
    fd = proof['feature_dimensions']
    assert fd['full'] == 50 * 1024 == 51200, fd
    assert fd['new_response'] == 50 * 16 == 800, fd
    assert fd['flow'] == 50 * 32 == 1600, fd
    assert fd['magnitude'] == 50, fd
    assert proof['prediction_shape'] == [nte, 50 * 32 + 50 * 7] == [nte, 1950], proof['prediction_shape']
    assert proof['positions'] == 50
    models = {c['model'] for c in cases}
    assert 'new_response' in models and 'flow_plus_new' in models and not any('old' in m for m in models), models
    assert all(np.isfinite(c['mse']) and np.isfinite(c['target_energy']) for c in cases)
    tr = Rhn.task_rows(cases, meta, comps, allowed)
    est, contr = Rhn.summary(tr)
    assert est and all(np.isfinite(e['task_equal_mse']) for e in est)
    want = {('new_response', 'magnitude'), ('new_response', 'full'), ('new_response', 'new_clean'),
            ('new_response', 'new_random'), ('flow_plus_new', 'flow')}
    got = {(c['candidate'], c['baseline']) for c in contr}
    assert want <= got and not any(p[1] == 'old_response' for p in got), got  # old pairs absent for pi0
    gcases, gsumm = Rhn.geometry_summary(geom, meta, comps, allowed)
    assert gsumm and all(g['retention'] is not None for g in gsumm)
    assert {g['model'] for g in gsumm} == {'new_response', 'new_clean', 'new_random'}
    print(f"POS=50 shapes PASS: full=51200 proj=800 flow=1600 target=1950 mean=(50,1024); "
          f"{len(cases)} case-scores, {len(est)} estimates, {len(contr)} contrasts, old-bases omitted")


if __name__ == '__main__':
    test_pos10_parity()
    test_pos50_shapes()
    print("ALL READOUT PORT TESTS PASS")
