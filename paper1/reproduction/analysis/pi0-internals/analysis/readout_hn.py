"""Horizon-parameterized port of current-basis-replay/producer-reference/readout.py.

The ONLY changes from the frozen pi0.5 producer driver are:
  * action-position count is M.POS (10 for pi0.5, 50 for pi0): full = POS*1024,
    basis-projected = POS*16, position_mean (POS,1024), prediction target POS*32+POS*7;
  * the pi0.5 "old" (uncentered-PCA) bases are OPTIONAL (absent for pi0); comparator
    pairs whose models are not produced are skipped;
  * the pi0.5-only `replicate()` stdlib score check is NOT ported (PROTOCOL §4).
Everything else — rank 16, ridge λ=0.01, weights, splits, bootstrap, centering, the
score/aggregation math — is byte-identical via legacy_math_hn. At POS=10 with old bases
this driver reproduces the frozen producer exactly (see test_readout_hn.py). Pure CPU
numpy; no cloud, model, or credentials.
"""
from collections import defaultdict
import hashlib
import numpy as np

import legacy_math_hn as M

MODE = 'exploratory_unseen_task_cross_family'
METRICS = tuple('adapter7_' + c + '_first5' for c in ('translation', 'rotation', 'gripper'))
OLD_NAMES = {'zero': 'zero', 'train_mean': 'train_mean', 'magnitude': 'magnitude',
             'full': 'full', 'flow': 'flow0', 'old_response': 'coeff_delta_uncentered_PCA',
             'old_clean': 'coeff_same_rows_clean_uncentered_PCA',
             'old_random': 'coeff_fixed_seed_random_orthogonal',
             'flow_plus_old': 'flow0_plus_delta_coeff'}
ALL_PAIRS = (('new_response', 'magnitude'), ('new_response', 'full'),
             ('new_response', 'new_clean'), ('new_response', 'new_random'),
             ('new_response', 'old_response'), ('flow_plus_new', 'flow'))
need = M.require


def basis(b):
    b = np.asarray(b, np.float64)
    need(b.shape == (1024, 16) and np.isfinite(b).all()
         and np.allclose(b.T @ b, np.eye(16), atol=1e-8, rtol=1e-8), 'archived_rank16_basis')
    return b


def predict_split(features, rows, components, item, old, new):
    """old may be None (pi0: no pi0.5 uncentered-PCA bases). POS from M.POS."""
    POS = M.POS
    s = item['support']
    tr, te, _ = M.split_rows(rows, s['family'], s['mode'], s['fold'], components)
    tr_rows, te_rows = [rows[i] for i in tr], [rows[i] for i in te]
    M.validate_split(tr_rows, te_rows, components, s)
    w = M.case_weights(tr_rows, components)
    h = features['hidden_delta'][:, M.LAYERS.index(s['layer'])].astype(np.float64)
    y = features['target']
    predictors = {'magnitude': np.linalg.norm(h, axis=-1), 'full': h.reshape(len(rows), POS * 1024),
                  'flow': features['flow0_delta']}
    bs = {'new_response': new['position_delta__basis32'][:, :16],
          'new_clean': new['position_clean__basis32'][:, :16],
          'new_random': new['curve_random32'][:, :16]}
    if old is not None:
        bs['old_response'] = old['delta_uncentered_PCA']
        bs['old_clean'] = old['same_rows_clean_uncentered_PCA']
        bs['old_random'] = old['fixed_seed_random_orthogonal']
    for name, b in bs.items():
        predictors[name] = (h @ basis(b)).reshape(len(rows), POS * 16)
    predictions = {'zero': np.zeros_like(y[te]), 'train_mean': np.tile(w @ y[tr], (len(te), 1))}
    transforms = {}
    for name, x in predictors.items():
        transforms[name] = M.fit_transform(x[tr], w)
        fit = M.fit_weighted_affine_ridge(x[tr], y[tr], w, .01, transform=transforms[name])
        predictions[name] = fit.predict(x[te])
    versions = ['new'] + (['old'] if old is not None else [])
    for version in versions:
        name = version + '_response'
        train_blocks = {'flow0': predictors['flow'][tr], 'delta_coeff': predictors[name][tr]}
        test_blocks = {'flow0': predictors['flow'][te], 'delta_coeff': predictors[name][te]}
        fit = M.fit_block_ridge(train_blocks, y[tr], w, .01,
                                transforms={'flow0': transforms['flow'], 'delta_coeff': transforms[name]})
        predictions['flow_plus_' + version] = fit.predict(test_blocks)
    context = {k: s[k] for k in ('family', 'layer', 'mode', 'fold')}
    cases = []
    for name, pred in predictions.items():
        scores = M.score_action_components(y[te], pred)
        for metric in METRICS:
            for i, row in enumerate(te_rows):
                cases.append({**context, 'model': name, 'metric': metric, 'case_id': row['case_id'],
                              'mse': float(scores[metric]['mse'][i]),
                              'target_energy': float(scores[metric]['target_energy'][i])})
    # All current dictionaries reconstruct the same observed residual.
    mu = np.asarray(new['position_mean'], np.float64)
    need(mu.shape == (POS, 1024) and np.isfinite(mu).all(), 'training_position_template')
    residual = h[te] - mu
    geometry = []
    for name in ('new_response', 'new_clean', 'new_random'):
        b = basis(bs[name]); err = residual - (residual @ b) @ b.T
        for i, row in enumerate(te_rows):
            geometry.append({**context, 'model': name, 'case_id': row['case_id'],
                             'sse': float(np.mean(np.sum(err[i] ** 2, axis=-1))),
                             'raw_energy': float(np.mean(np.sum(h[te[i]] ** 2, axis=-1))),
                             'residual_energy': float(np.mean(np.sum(residual[i] ** 2, axis=-1)))})
    proof = {'support': s, 'training_scales': {n: t.scale for n, t in transforms.items()},
             'feature_dimensions': {n: x.shape[1] for n, x in predictors.items()},
             'prediction_shape': [len(te), POS * 32 + POS * 7], 'lambda': .01, 'positions': POS}
    return cases, geometry, proof


def task_rows(cases, metadata, components, allowed):
    grouped = defaultdict(list)
    for row in cases:
        if row['case_id'] in allowed:
            key = (row['family'], row['layer'], row['metric'], row['model'],
                   components[metadata[row['case_id']]['task_id']])
            grouped[key].append(row)
    result = []
    for (family, layer, metric, model, task), values in sorted(grouped.items()):
        need(len({r['case_id'] for r in values}) == len(values), 'single_heldout_prediction_per_case')
        w = M.case_weights([metadata[r['case_id']] for r in values], components)
        result.append({'mode': MODE, 'family': family, 'layer': layer, 'metric': metric, 'model': model,
                       'task_component': task, 'case_count': len(values),
                       'mse': float(w @ [r['mse'] for r in values]),
                       'target_energy': float(w @ [r['target_energy'] for r in values])})
    return result


def summary(tasks):
    grouped = defaultdict(dict)
    for row in tasks:
        key = (row['family'], row['layer'], row['metric'], row['model'])
        need(row['task_component'] not in grouped[key], 'single_heldout_prediction_per_task')
        grouped[key][row['task_component']] = row
    estimates, contrasts = [], []
    for (family, layer, metric, model), by_task in sorted(grouped.items()):
        values = [by_task[t] for t in sorted(by_task)]
        seed_text = f'taskblock|{MODE}|{family}|{layer}'
        stats = M.bootstrap_mean([v['mse'] for v in values], seed_text)
        estimates.append({'family': family, 'layer': layer, 'metric': metric, 'model': model,
                          'case_count': sum(v['case_count'] for v in values),
                          'task_equal_mse': stats['mean'], 'interval_95': stats['interval_95'],
                          'task_count': len(values),
                          'task_equal_target_energy': float(np.mean([v['target_energy'] for v in values]))})
    present = {k[3] for k in grouped}
    pairs = [(c, b) for (c, b) in ALL_PAIRS if c in present and b in present]
    for family, layer, metric in sorted({k[:3] for k in grouped}):
        for candidate, baseline in pairs:
            key_c = (family, layer, metric, candidate)
            key_b = (family, layer, metric, baseline)
            if key_c not in grouped or key_b not in grouped:
                continue
            ca, ba = grouped[key_c], grouped[key_b]
            need(set(ca) == set(ba), 'paired_readout_task_support')
            ids = sorted(ca)
            c = np.array([ca[t]['mse'] for t in ids]); b = np.array([ba[t]['mse'] for t in ids])
            seed_text = f'taskblock|{MODE}|{family}|{layer}'
            stats = M.bootstrap_mean(b - c, seed_text)
            seed = int.from_bytes(hashlib.sha256(seed_text.encode()).digest()[:8], 'little')
            draw = np.random.default_rng(seed).integers(0, len(ids), (2000, len(ids)))
            den = b[draw].mean(1); num = (b - c)[draw].mean(1); valid = den > 0
            ci = [float(v) for v in np.quantile(num / den, [.025, .975])] if valid.all() else None
            contrasts.append({'family': family, 'layer': layer, 'metric': metric,
                              'candidate': candidate, 'baseline': baseline,
                              'difference_baseline_minus_candidate': stats,
                              'relative_error_reduction': float((b.mean() - c.mean()) / b.mean()) if b.mean() > 0 else None,
                              'relative_error_reduction_interval_95': ci,
                              'undefined_bootstrap_denominators': int((~valid).sum())})
    return estimates, contrasts


def geometry_summary(cases, metadata, components, allowed):
    groups = defaultdict(list)
    for row in cases:
        if row['case_id'] in allowed:
            groups[(row['family'], row['layer'], row['model'],
                    components[metadata[row['case_id']]['task_id']])].append(row)
    tasks = []
    for (family, layer, model, task), values in sorted(groups.items()):
        w = M.case_weights([metadata[r['case_id']] for r in values], components)
        sums = {k: float(w @ [r[k] for r in values]) for k in ('sse', 'raw_energy', 'residual_energy')}
        tasks.append({'family': family, 'layer': layer, 'model': model, 'task_component': task,
                      'case_count': len(values), **sums,
                      'residual_retention': 1 - sums['sse'] / sums['residual_energy'] if sums['residual_energy'] > 0 else None,
                      'raw_score': 1 - sums['sse'] / sums['raw_energy'] if sums['raw_energy'] > 0 else None})
    summaries = []
    for family, layer, model in sorted({(r['family'], r['layer'], r['model']) for r in tasks}):
        selected = [r for r in tasks if (r['family'], r['layer'], r['model']) == (family, layer, model)]
        values = [r['residual_retention'] for r in selected]
        stat = M.bootstrap_mean(values, f'taskblock|{MODE}|{family}|{layer}') if all(v is not None for v in values) else None
        summaries.append({'family': family, 'layer': layer, 'model': model,
                          'case_count': sum(r['case_count'] for r in selected),
                          'defined_tasks': sum(v is not None for v in values),
                          'planned_tasks': len(values), 'retention': stat})
    return tasks, summaries
