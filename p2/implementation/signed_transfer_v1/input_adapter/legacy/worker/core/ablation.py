"""One fixed shared-only DEV refit; post-hoc CONF evaluation. NumPy only.

This module receives source-qualified compact rows. Native decoding and source
joins live in sources.py; transport, deadlines and archival remain external.
"""
from dataclasses import dataclass
import hashlib
import json
import numpy as np

FIT_ID = '31ab045c8a96d4a1e5cdae7d8d41a62ca07056d6892a09ecc4ea65bf178a04e8'
METHODS = ('context9', 'context9_pooled4', 'context9_current4')
DEV_ROOTS = tuple(f'SDRU-CDEV-T0{t}-R0{r}' for t in range(2) for r in range(2))
DEV_KEYS = tuple((r, q, 'shared', 'KT') for r in DEV_ROOTS for q in range(11, 23))
CONF_ROOTS = tuple((f'SDRU-CCONF-T{t:02d}-V{v}-I00-P{p}', f'CCONF-T{t:02d}', p)
                  for t, v in ((6, '00700'), (7, '00774'), (8, '01029'), (9, '00827')) for p in ('C', 'T'))
CONF_KEYS = tuple((r, q, 'shared', bg) for r, _, _ in CONF_ROOTS for q in range(11, 23)
                  for bg in (('KT', 'KC') if q in (11, 16, 22) else ('KT',)))
DIMS = ('translation_command_x', 'translation_command_y', 'translation_command_z',
        'rotation_command_x', 'rotation_command_y', 'rotation_command_z', 'gripper_command')


def need(ok, code):
    if not ok:
        raise ValueError(code)


def encoded(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()


def digest(value):
    return hashlib.sha256(value).hexdigest()


def aid(value):
    a = np.asarray(value)
    return {'shape': list(a.shape), 'dtype': a.dtype.str, 'sha256': digest(a.tobytes())}


def frozen_array(value, shape):
    a = np.array(value, dtype=np.float64, copy=True)
    need(a.shape == shape and np.isfinite(a).all(), 'finite_compact_array_ABI')
    a.setflags(write=False)
    return a


@dataclass(frozen=True)
class Row:
    key: tuple
    split: str
    current: np.ndarray
    pooled: np.ndarray
    effect: np.ndarray
    zero_actual_E: bool
    provenance: dict

    def __post_init__(self):
        object.__setattr__(self, 'key', tuple(self.key))
        need(self.split in ('DEV', 'CONF') and self.key in (DEV_KEYS if self.split == 'DEV' else CONF_KEYS), 'fixed_shared_row_population')
        need(type(self.zero_actual_E) is bool, 'known_exactzero_flag')
        for name, shape in (('current', (13,)), ('pooled', (13,)), ('effect', (10, 7))):
            object.__setattr__(self, name, frozen_array(getattr(self, name), shape))
        need(np.array_equal(self.current[:9], self.pooled[:9]), 'identical_context9_across_readers')
        need(isinstance(self.provenance, dict) and all(isinstance(self.provenance.get(k), str) and len(self.provenance[k]) == 64
             for k in ('input_identity', 'edit_identity', 'label_identity')), 'source_qualified_compact_row_identities')
        object.__setattr__(self, 'provenance', json.loads(encoded(self.provenance)))

    def manifest(self):
        return {'key': list(self.key), 'split': self.split, 'zero_actual_E': self.zero_actual_E,
                'arrays': {n: aid(getattr(self, n)) for n in ('current', 'pooled', 'effect')}, 'provenance': self.provenance}

    def features(self, method):
        need(method in METHODS, 'fixed_three_methods')
        return self.current[:9] if method == 'context9' else self.pooled if method == 'context9_pooled4' else self.current


class FrozenScales:
    def __init__(self, model, *, synthetic=False):
        model.verify()
        need(synthetic or model.identity == FIT_ID, 'original_frozen_fit_required')
        self.fit_identity = model.identity
        self.synthetic = synthetic
        a, b = (model.regressions[n] for n in ('current', 'pooled'))
        for n in ('feature_mean', 'feature_scale'):
            need(np.array_equal(a[n], b[n]), 'original_current_pooled_common_scaling')
            setattr(self, n, frozen_array(a[n], (13,)))
        self.target_mean = frozen_array(model.target_mean, (70,))
        self.target_scale = frozen_array(model.target_scale, (70,))
        need((self.feature_scale > 0).all() and (self.target_scale > 0).all(), 'positive_original_scales')
        self.preprocessing = {n: aid(v) for n, v in model.arrays().items()
                              if n.startswith(('state_', 'first_flow_', 'write_')) or n == 'pooled_P'}

    def manifest(self):
        return {'fit_identity': self.fit_identity, 'synthetic': self.synthetic,
                'arrays': {n: aid(getattr(self, n)) for n in ('feature_mean', 'feature_scale', 'target_mean', 'target_scale')},
                'frozen_preprocessing': self.preprocessing}


class Fitted:
    def __init__(self, scales, rows, regressions):
        self.scales, self.regressions = scales, regressions
        self.fit_rows = [r.manifest() for r in rows]
        self.identity = digest(encoded(self.manifest()))

    def manifest(self):
        return {'schema': 'context-shared-ablation-fit-v1', 'role': 'POSTHOC_DEV_REFIT_CONF_DIAGNOSTIC',
                'preprocessing': self.scales.manifest(), 'fit_rows': self.fit_rows, 'methods': list(METHODS),
                'feature_count': dict(zip(METHODS, (9, 13, 13))), 'lambda': 1.0, 'query_weight': 1 / 48,
                'objective': 'mean_48 squared DEV-scaled residual + coefficient Frobenius squared',
                'intercept': 'unpenalized and reestimated on shared48', 'target_center_scale': 'original all192 frozen values',
                'zero_rule': 'original exactzero override applied identically to all three predictions',
                'PCA_or_scale_refit': False, 'old_coefficients_reused': False, 'old_scores_replaced': False,
                'regressions': {m: {k: aid(v) for k, v in r.items()} for m, r in self.regressions.items()}}

    def arrays(self):
        return {m + '_' + k: v for m, r in self.regressions.items() for k, v in r.items()}

    def verify(self):
        need(digest(encoded(self.manifest())) == self.identity, 'ablation_fit_immutable')

    def predict(self, row):
        self.verify()
        out = {}
        for method, reg in self.regressions.items():
            x = row.features(method); d = len(x)
            x = (x - self.scales.feature_mean[:d]) / self.scales.feature_scale[:d]
            y = ((x @ reg['coef'] + reg['intercept']) * self.scales.target_scale + self.scales.target_mean).reshape(10, 7)
            need(np.isfinite(y).all(), 'finite_refit_prediction')
            out[method] = np.zeros((10, 7)) if row.zero_actual_E else y
        return out


def fit(scales, rows):
    rows = list(rows); by = {r.key: r for r in rows}
    need(len(by) == len(rows) == 48 and set(by) == set(DEV_KEYS) and all(r.split == 'DEV' for r in rows), 'all48_shared_DEV_no_missing_duplicate_or_CONF')
    rows = [by[k] for k in DEV_KEYS]
    y = np.stack([r.effect.ravel() for r in rows])
    z = (y - scales.target_mean) / scales.target_scale
    zb = z.mean(axis=0); zc = z - zb
    regressions = {}
    for method in METHODS:
        raw = np.stack([r.features(method) for r in rows]); d = raw.shape[1]
        x = (raw - scales.feature_mean[:d]) / scales.feature_scale[:d]
        xb = x.mean(axis=0); xc = x - xb
        coef = np.linalg.solve(xc.T @ xc / 48 + np.eye(d), xc.T @ zc / 48)
        regressions[method] = {'coef': frozen_array(coef, (d, 70)), 'intercept': frozen_array(zb - xb @ coef, (70,))}
    return Fitted(scales, rows, regressions)


def evaluate(model, rows, *, unavailable=None):
    """KT96 primary, KC24 support. Missing remains NA; no survivor reweighting."""
    rows = list(rows); by = {r.key: r for r in rows}; unavailable = dict(unavailable or {})
    need(len(by) == len(rows) and set(by) <= set(CONF_KEYS) and all(r.split == 'CONF' for r in rows), 'only_unique_original_CONF_shared')
    need(set(unavailable) <= set(CONF_KEYS) and not (set(unavailable) & set(by)), 'missing_reasons_disjoint_from_observed')
    prediction = {}; coverage = []
    for key in CONF_KEYS:
        if key not in by:
            coverage.append({'key': list(key), 'status': 'UNAVAILABLE', 'reason': unavailable.get(key, 'not_supplied')})
        else:
            prediction[key] = model.predict(by[key])
            coverage.append({'key': list(key), 'status': 'QUALIFIED', 'zero_actual_E': by[key].zero_actual_E, 'provenance': by[key].provenance})
    root_scores = []
    for root, task, parent in CONF_ROOTS:
        for bg in ('KT', 'KC'):
            expected = [k for k in CONF_KEYS if k[0] == root and k[3] == bg]
            available = [k for k in expected if k in by]
            for method in METHODS:
                for window, end in (('first5', 5), ('full10', 10)):
                    for dim, name in enumerate(DIMS):
                        err = np.stack([prediction[k][method][:end, dim] - by[k].effect[:end, dim] for k in available]) if available else None
                        mse = None if err is None else float(np.mean(err ** 2))
                        scaled = None if err is None else float(np.mean((err / model.scales.target_scale.reshape(10, 7)[:end, dim]) ** 2))
                        full = len(available) == len(expected)
                        root_scores.append({'root_id': root, 'task_id': task, 'parent_condition': parent, 'K_background': bg,
                            'role': 'PRIMARY' if bg == 'KT' else 'SUPPORT_ANCHORS', 'method': method, 'window': window, 'dimension': name,
                            'planned_queries': len(expected), 'observed_queries': len(available), 'missing_queries': len(expected) - len(available),
                            'MSE': mse if full else None, 'DEV_scaled_MSE': scaled if full else None,
                            'available_only_MSE_not_population_estimate': mse if not full else None,
                            'available_only_scaled_MSE_not_population_estimate': scaled if not full else None})
    summary, increments = [], []
    for parent in ('C', 'T'):
        for bg in ('KT', 'KC'):
            for window in ('first5', 'full10'):
                for name in DIMS:
                    values = {}
                    for method in METHODS:
                        rr = [r for r in root_scores if (r['parent_condition'], r['K_background'], r['window'], r['dimension'], r['method']) == (parent, bg, window, name, method)]
                        need(len(rr) == 4, 'four_equal_task_roots_per_parent')
                        full = all(r['DEV_scaled_MSE'] is not None for r in rr)
                        value = float(np.mean([r['DEV_scaled_MSE'] for r in rr])) if full else None
                        values[method] = value
                        summary.append({'parent_condition': parent, 'K_background': bg, 'window': window, 'dimension': name,
                                        'method': method, 'task_weight': 0.25, 'planned_tasks': 4, 'complete_tasks': sum(r['MSE'] is not None for r in rr),
                                        'DEV_scaled_MSE': value, 'MSE': float(np.mean([r['MSE'] for r in rr])) if full else None,
                                        'query_independent_CI': False})
                    for stronger, weaker in (('context9_pooled4', 'context9'), ('context9_current4', 'context9'), ('context9_current4', 'context9_pooled4')):
                        a, b = values[stronger], values[weaker]
                        increments.append({'parent_condition': parent, 'K_background': bg, 'window': window, 'dimension': name,
                                           'comparison': stronger + '_vs_' + weaker, 'positive_means_lower_error': True,
                                           'DEV_scaled_MSE_reduction': None if a is None or b is None else b - a})
    return {'schema': 'context-shared-ablation-evaluation-v1', 'posthoc_not_new_blind_test': True,
            'fit_identity': model.identity, 'old_fit_identity': model.scales.fit_identity,
            'primary_planned_targets': 96, 'support_planned_targets': 24,
            'complete': len(by) == len(CONF_KEYS), 'coverage': coverage, 'root_scores': root_scores,
            'equal_task_scores': summary, 'paired_increments': increments,
            'predictions': [{'key': list(k), 'effect': by[k].effect.tolist(), 'predictions': {m: v.tolist() for m, v in prediction[k].items()}} for k in CONF_KEYS if k in by],
            'old_scores_replaced': False, 'new_model_calls': 0, 'new_env_steps': 0}
