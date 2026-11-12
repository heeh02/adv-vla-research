"""Pure CPU readout mathematics. No cloud, model, credential, or file access.

Objective: sum_i w_i ||y_i - b - z_i W||^2 + lam ||W||^2,
where sum(w)=1. Each declared feature block has one training-only scalar RMS.
The intercept is unpenalized; the dual system is N by N, never D by D.
"""
from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json
import numpy as np

LAYERS = (0, 8, 17)
FAMILIES = ('Background Textures', 'Camera Viewpoints', 'Light Conditions', 'Sensor Noise')
SUITES = ('libero_10', 'libero_goal', 'libero_object', 'libero_spatial')
FOLD_SEED = 'plus-shared-channel-whole-task-fold-v1'


def require(ok, code):
    if not ok:
        raise ValueError(code)


def matrix(value, name='matrix'):
    a = np.asarray(value, np.float64)
    require(a.ndim == 2 and min(a.shape) > 0 and np.isfinite(a).all(), name + '_finite_2D')
    return a


def normalized_weights(weights, n):
    w = np.asarray(weights, np.float64)
    require(w.shape == (n,) and np.isfinite(w).all() and np.all(w > 0), 'positive_case_weights')
    return w / w.sum()


@dataclass(frozen=True)
class Transform:
    mean: np.ndarray
    scale: float

    def apply(self, value):
        a = matrix(value, 'predictor')
        require(a.shape[1] == len(self.mean), 'same_feature_dimension')
        return (a - self.mean) / self.scale


def fit_transform(X, weights):
    x = matrix(X, 'training_predictor')
    w = normalized_weights(weights, len(x))
    if np.all(x == x[0]):
        return Transform(mean=x[0].copy(), scale=1.0)
    mean = w @ x
    scale = float(np.sqrt(np.sum(w[:, None] * (x - mean) ** 2) / x.shape[1]))
    # Exact constant block remains a constant; no arbitrary small-signal cutoff.
    return Transform(mean=mean, scale=scale if scale > 0 else 1.0)


@dataclass(frozen=True)
class DualRidge:
    transform: Transform
    weighted_design: np.ndarray
    alpha: np.ndarray
    target_mean: np.ndarray
    lam: float

    def predict(self, X):
        return (self.transform.apply(X) @ self.weighted_design.T) @ self.alpha + self.target_mean


def fit_weighted_affine_ridge(X, Y, weights, lam=0.01, transform=None):
    x, y = matrix(X, 'training_predictor'), matrix(Y, 'training_target')
    require(len(x) == len(y) and np.isfinite(lam) and lam > 0, 'paired_positive_ridge')
    w = normalized_weights(weights, len(x))
    transform = fit_transform(x, w) if transform is None else transform
    z = transform.apply(x)
    # A supplied transform must be from these training rows, not test rows.
    require(np.allclose(w @ z, 0, atol=1e-10, rtol=0), 'training_centered_transform')
    mean_y = w @ y
    zw = np.sqrt(w[:, None]) * z
    yw = np.sqrt(w[:, None]) * (y - mean_y)
    gram = zw @ zw.T
    gram.flat[::len(gram) + 1] += float(lam)
    alpha = np.linalg.solve(gram, yw)
    require(np.isfinite(alpha).all(), 'finite_dual_solution')
    return DualRidge(transform, zw, alpha, mean_y, float(lam))


@dataclass(frozen=True)
class BlockRidge:
    names: tuple
    transforms: dict
    weighted_design: np.ndarray
    alpha: np.ndarray
    target_mean: np.ndarray
    lam: float

    def predict(self, blocks):
        require(set(blocks) == set(self.names), 'same_named_feature_blocks')
        z = np.concatenate([self.transforms[k].apply(blocks[k]) for k in self.names], axis=1)
        return (z @ self.weighted_design.T) @ self.alpha + self.target_mean


def fit_block_ridge(blocks, Y, weights, lam=0.01, transforms=None):
    require(bool(blocks), 'nonempty_named_feature_blocks')
    names = tuple(blocks)
    values = {k: matrix(blocks[k], k) for k in names}
    y = matrix(Y, 'training_target')
    require(all(len(v) == len(y) for v in values.values()) and np.isfinite(lam) and lam > 0,
            'paired_positive_block_ridge')
    w = normalized_weights(weights, len(y))
    transforms = ({k: fit_transform(values[k], w) for k in names}
                  if transforms is None else dict(transforms))
    require(set(transforms) == set(names), 'one_transform_per_named_block')
    z = np.concatenate([transforms[k].apply(values[k]) for k in names], axis=1)
    require(np.allclose(w @ z, 0, atol=1e-10, rtol=0), 'training_centered_block_transforms')
    zw = np.sqrt(w[:, None]) * z
    mean_y = w @ y
    gram = zw @ zw.T
    gram.flat[::len(gram) + 1] += float(lam)
    alpha = np.linalg.solve(gram, np.sqrt(w[:, None]) * (y - mean_y))
    require(np.isfinite(alpha).all(), 'finite_dual_block_solution')
    return BlockRidge(names, transforms, zw, alpha, mean_y, float(lam))


def task_components(rows):
    tasks = {r['task_id'] for r in rows}
    parent = {task: task for task in tasks}
    def find(task):
        while parent[task] != task:
            parent[task] = parent[parent[task]]
            task = parent[task]
        return task
    seen = {}
    for row in rows:
        clean, task = row['clean_id'], row['task_id']
        if clean in seen:
            a, b = find(task), find(seen[clean])
            if a != b:
                parent[max(a, b)] = min(a, b)
        else:
            seen[clean] = task
    grouped = defaultdict(list)
    for task in sorted(tasks):
        grouped[find(task)].append(task)
    return {task: hashlib.sha256(('component|' + '|'.join(grouped[find(task)])).encode()).hexdigest()
            for task in tasks}


def fold_of(component):
    return int.from_bytes(hashlib.sha256((FOLD_SEED + '|' + component).encode()).digest()[:8], 'little') % 5


def case_set_sha(rows):
    return hashlib.sha256(json.dumps(sorted(r['case_id'] for r in rows)).encode()).hexdigest()


def split_rows(rows, family, mode, fold=None, components=None):
    require(family in FAMILIES, 'declared_heldout_family')
    require(len({r['case_id'] for r in rows}) == len(rows), 'unique_case_ids')
    components = task_components(rows) if components is None else components
    if mode == 'known_task_cross_family':
        require(fold is None, 'known_task_fold_none')
        train = [i for i, r in enumerate(rows) if r['family'] != family]
        represented = {components[rows[i]['task_id']] for i in train}
        test = [i for i, r in enumerate(rows) if r['family'] == family and components[r['task_id']] in represented]
    else:
        require(mode == 'exploratory_unseen_task_cross_family' and fold in range(5), 'declared_unseen_task_fold')
        train = [i for i, r in enumerate(rows) if r['family'] != family and fold_of(components[r['task_id']]) != fold]
        test = [i for i, r in enumerate(rows) if r['family'] == family and fold_of(components[r['task_id']]) == fold]
    return train, test, components


def validate_split(train_rows, test_rows, components, support):
    require(train_rows and test_rows, 'available_train_and_test_support')
    require(all(r['family'] != support['family'] for r in train_rows)
            and all(r['family'] == support['family'] for r in test_rows), 'family_leakage')
    require(not ({r['case_id'] for r in train_rows} & {r['case_id'] for r in test_rows}), 'case_leakage')
    if support['mode'] == 'exploratory_unseen_task_cross_family':
        require(not ({components[r['task_id']] for r in train_rows} & {components[r['task_id']] for r in test_rows}), 'task_component_leakage')
        require(not ({r['clean_id'] for r in train_rows} & {r['clean_id'] for r in test_rows}), 'shared_clean_leakage')
        require(all(fold_of(components[r['task_id']]) != support['fold'] for r in train_rows)
                and all(fold_of(components[r['task_id']]) == support['fold'] for r in test_rows), 'frozen_fold_identity')
    for prefix, subset in (('training', train_rows), ('test', test_rows)):
        require(len(subset) == support[prefix + '_case_count']
                and case_set_sha(subset) == support[prefix + '_case_ids_sha256'], 'Q1_' + prefix + '_case_identity')
    return True


def case_weights(rows, components):
    require(bool(rows), 'nonempty_case_rows')
    keys = [(components[r['task_id']], r['family'], r['row_id'], str(r['noise_label'])) for r in rows]
    require(len(keys) == len(set(keys)), 'unique_component_family_variant_noise')
    counts = [defaultdict(set) for _ in range(4)]
    for key in keys:
        for depth in range(4):
            counts[depth][key[:depth]].add(key[depth])
    weights = np.asarray([np.prod([1 / len(counts[d][key[:d]]) for d in range(4)]) for key in keys])
    require(np.isclose(weights.sum(), 1, atol=1e-12, rtol=0), 'normalized_nested_case_weights')
    return weights


def validate_native_pair(reference, treatment, row):
    """Input is decoded, typed-SHA-checked native dictionaries, both full ABI.

    Member/field hashes and producer roles are a caller I/O responsibility.
    Validate full time/call/x0 identities before selecting j0.
    """
    require(row['suite'] in SUITES and row['family'] in FAMILIES, 'declared_suite_family')
    shapes = {'hidden': (10, 3, 1, 10, 1024), 'time': (10,), 'velocity': (10, 1, 10, 32),
              'x_t': (10, 1, 10, 32), 'call_indices': (10,), 'final_actions': (1, 10, 32),
              'raw_actions': (1, 10, 32), 'actions': (1, 10, 7)}
    for arm in (reference, treatment):
        for field, shape in shapes.items():
            a = np.asarray(arm[field])
            require(a.shape == shape and np.isfinite(a).all(), 'native_' + field + '_ABI')
        require(np.array_equal(arm['call_indices'], np.arange(10)), 'native_generation_call_order')
        require(np.array_equal(arm['final_actions'], arm['raw_actions']), 'T_final_equals_T_raw')
    require(np.array_equal(reference['time'], treatment['time']), 'paired_generation_times')
    require(np.all(np.diff(reference['time']) < 0), 'descending_generation_times')
    require(np.array_equal(reference['x_t'][0], treatment['x_t'][0]), 'same_j0_noise')
    return True


def project_j0_by_layer(hidden_delta, basis, layer):
    """Batch input is [N,3,10,1024] at j0; never pool native layer axes."""
    x, b = np.asarray(hidden_delta, np.float64), np.asarray(basis, np.float64)
    require(layer in LAYERS and x.ndim == 4 and x.shape[1:] == (3, 10, 1024)
            and np.isfinite(x).all(), 'j0_three_native_layers_ABI')
    require(b.shape == (1024, 16) and np.isfinite(b).all()
            and np.allclose(b.T @ b, np.eye(16), atol=1e-8, rtol=1e-8), 'Q1_same_layer_rank16_orthonormal_basis')
    return (x[:, LAYERS.index(layer)] @ b).reshape(len(x), 160)


def score_action_components(target, prediction):
    y, p = matrix(target, 'target'), matrix(prediction, 'prediction')
    require(y.shape == p.shape and y.shape[1] == 390, 'raw320_plus_adapter70_target_ABI')
    raw_y, raw_p = y[:, :320].reshape(-1, 10, 32), p[:, :320].reshape(-1, 10, 32)
    act_y, act_p = y[:, 320:].reshape(-1, 10, 7), p[:, 320:].reshape(-1, 10, 7)
    groups = {'raw32_all32': (raw_y, raw_p, slice(0, 32)), 'raw32_first7': (raw_y, raw_p, slice(0, 7)),
              'adapter7_translation': (act_y, act_p, slice(0, 3)),
              'adapter7_rotation': (act_y, act_p, slice(3, 6)),
              'adapter7_gripper': (act_y, act_p, slice(6, 7)),
              'adapter7_all7': (act_y, act_p, slice(0, 7))}
    result = {}
    for name, (yy, pp, dims) in groups.items():
        for count in (5, 10):
            actual = yy[:, :count, dims]
            error = pp[:, :count, dims] - actual
            result[name + ('_first5' if count == 5 else '_all10')] = {
                'mse': np.mean(error ** 2, axis=(1, 2)),
                'target_energy': np.mean(actual ** 2, axis=(1, 2))}
    return result


def aggregate_component_scores(scores, rows, components):
    require(bool(rows), 'nonempty_score_rows')
    result = []
    for component in sorted({components[r['task_id']] for r in rows}):
        ids = [i for i, row in enumerate(rows) if components[row['task_id']] == component]
        w = case_weights([rows[i] for i in ids], components)
        for metric, values in scores.items():
            mse = float(w @ values['mse'][ids])
            energy = float(w @ values['target_energy'][ids])
            result.append({'task_component': component, 'metric': metric, 'case_count': len(ids),
                           'mse': mse, 'target_energy': energy,
                           'relative_mse': mse / energy if energy > 0 else None})
    return result


def bootstrap_mean(values, seed_text, repeats=2000):
    a = np.asarray(values, np.float64)
    require(a.ndim == 1 and len(a) and np.isfinite(a).all(), 'finite_independent_task_values')
    if len(a) < 2:
        return {'task_count': len(a), 'mean': float(a.mean()), 'interval_95': None}
    seed = int.from_bytes(hashlib.sha256(seed_text.encode()).digest()[:8], 'little')
    rng = np.random.default_rng(seed)
    means = np.mean(a[rng.integers(0, len(a), size=(repeats, len(a)))], axis=1)
    return {'task_count': len(a), 'mean': float(a.mean()),
            'interval_95': [float(v) for v in np.quantile(means, [.025, .975])]}
