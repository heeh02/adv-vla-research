"""Regression test for the horizon-parameterized port.

POS=10 must reproduce the frozen producer legacy_math.py bit-for-bit on synthetic
data for every position-dependent function; POS=50 must run with correct shapes.
Run: <venv python> -I test_port.py   (numpy required; no cloud/model/creds).
"""
import importlib.util
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ORIG = Path('/private/tmp/claude-501/e1-pkg-extract/current-basis-replay/producer-reference/legacy_math.py')


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def orthob(seed):
    q, _ = np.linalg.qr(np.random.default_rng(seed).standard_normal((1024, 16)))
    return q


def main():
    np.seterr(all='ignore')
    sys.path.insert(0, str(HERE))
    port = load('legacy_math_hn', HERE / 'legacy_math_hn.py')
    orig = load('legacy_math_orig', ORIG)
    port.set_positions(10)
    rng = np.random.default_rng(0)
    checks = 0

    # 1. score_action_components identical at POS=10
    N = 7
    y = rng.standard_normal((N, 390)); p = rng.standard_normal((N, 390))
    so = orig.score_action_components(y, p); sp = port.score_action_components(y, p)
    assert set(so) == set(sp), (sorted(so), sorted(sp))
    for k in so:
        for f in ('mse', 'target_energy'):
            assert np.array_equal(so[k][f], sp[k][f]), k + '/' + f
    checks += 1

    # 2. project_j0_by_layer identical at POS=10
    h = rng.standard_normal((N, 3, 10, 1024)); b = orthob(1)
    for layer in (0, 8, 17):
        po = orig.project_j0_by_layer(h, b, layer); pp = port.project_j0_by_layer(h, b, layer)
        assert po.shape == (N, 160) == pp.shape
        assert np.array_equal(po, pp)
    checks += 1

    # 3. fit_weighted_affine_ridge identical predictions
    X = rng.standard_normal((20, 30)); Y = rng.standard_normal((20, 5)); w = rng.random(20) + 0.1
    Xt = rng.standard_normal((6, 30))
    fo = orig.fit_weighted_affine_ridge(X, Y, w, 0.01); fp = port.fit_weighted_affine_ridge(X, Y, w, 0.01)
    assert np.allclose(fo.predict(Xt), fp.predict(Xt), atol=0, rtol=0)
    checks += 1

    # 4. fit_block_ridge identical
    blocks = {'a': rng.standard_normal((20, 8)), 'b': rng.standard_normal((20, 12))}
    tblocks = {'a': rng.standard_normal((6, 8)), 'b': rng.standard_normal((6, 12))}
    bo = orig.fit_block_ridge(blocks, Y, w, 0.01); bp = port.fit_block_ridge(blocks, Y, w, 0.01)
    assert np.allclose(bo.predict(tblocks), bp.predict(tblocks), atol=0, rtol=0)
    checks += 1

    # 5. case_weights / task_components / fold_of / bootstrap_mean identical
    rows = [{'task_id': f't{i%5}', 'clean_id': f'c{i%7}', 'family': port.FAMILIES[i % 4],
             'row_id': f'v{i}', 'noise_label': i % 2, 'case_id': f'case{i}'} for i in range(30)]
    co = orig.task_components(rows); cp = port.task_components(rows)
    assert co == cp
    comp_rows = rows[:12]
    # make unique nested keys for case_weights
    cw_rows = [{'task_id': f't{i}', 'family': port.FAMILIES[i % 4], 'row_id': f'v{i}', 'noise_label': i % 2} for i in range(8)]
    comps = {f't{i}': f'comp{i}' for i in range(8)}
    assert np.array_equal(orig.case_weights(cw_rows, comps), port.case_weights(cw_rows, comps))
    assert orig.fold_of('comp3') == port.fold_of('comp3')
    vals = list(rng.standard_normal(14))
    assert orig.bootstrap_mean(vals, 'seed|x') == port.bootstrap_mean(vals, 'seed|x')
    checks += 1

    # 6. validate_native_pair runs at POS=10 (orig) and POS=50 (port)
    def make_pair(mod, pos, calls=10):
        t = np.sort(rng.random(calls))[::-1].astype(np.float32)
        x0 = rng.standard_normal((1, pos, 32)).astype(np.float32)
        def arm():
            xt = rng.standard_normal((calls, 1, pos, 32)).astype(np.float32); xt[0] = x0
            ra = rng.standard_normal((1, pos, 32)).astype(np.float32)
            return {'hidden': rng.standard_normal((calls, 3, 1, pos, 1024)).astype(np.float32),
                    'time': t, 'velocity': rng.standard_normal((calls, 1, pos, 32)).astype(np.float32),
                    'x_t': xt, 'call_indices': np.arange(calls), 'final_actions': ra, 'raw_actions': ra,
                    'actions': rng.standard_normal((1, pos, 7)).astype(np.float32)}
        a = arm(); b = dict(arm()); b['x_t'] = a['x_t'].copy(); b['time'] = a['time'].copy()
        return a, b
    row = {'suite': 'libero_10', 'family': 'Background Textures'}
    a, b = make_pair(orig, 10); assert orig.validate_native_pair(a, b, row) is True
    port.set_positions(50)
    a, b = make_pair(port, 50); assert port.validate_native_pair(a, b, row) is True
    checks += 1

    # 7. POS=50 shapes: project -> (N,800); score target dim 1950, keys include _all50
    h50 = rng.standard_normal((N, 3, 50, 1024))
    assert port.project_j0_by_layer(h50, orthob(2), 8).shape == (N, 800)
    y50 = rng.standard_normal((N, 1950)); p50 = rng.standard_normal((N, 1950))
    s50 = port.score_action_components(y50, p50)
    assert 'adapter7_translation_first5' in s50 and 'adapter7_gripper_all50' in s50
    assert s50['adapter7_translation_first5']['mse'].shape == (N,)
    checks += 1

    print('PASS', checks, 'check-groups; POS=10 reproduces frozen producer; POS=50 runs')


if __name__ == '__main__':
    main()
