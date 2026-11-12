"""DEV-only signed-transfer ridge; no IO, network, model, or CONF selection.

The adapter owns native source/clock qualification. This module never treats
FP64 forward-looking write proxies as actual native W, or CONF as blind data.
"""
from dataclasses import dataclass
import hashlib
import json
import math
import numpy as np


FAMILIES = ('c9', 'norm4', 'pooled4', 'current4', 'pooled_full',
            'current_full', 'headpos_signed', 'headpos_norms', 'headpos_signed_sum')
LAMBDAS = (0.1, 1.0, 10.0)
EXTRA_FIELDS = {'norm4': ('originalnorm4', (4,)), 'pooled4': ('pooled4', (4,)),
    'current4': ('current4', (4,)), 'pooled_full': ('pooled_gated_write', (10, 1024)),
    'current_full': ('current_gated_write', (10, 1024)),
    'headpos_signed': ('current_headpos_read', (8, 10, 256)),
    'headpos_norms': ('current_headpos_norms', (80,))}
DEV_ROOTS = tuple(f'SDRU-CDEV-T0{task}-R0{root}' for task in range(2) for root in range(2))
DEV_KEYS = tuple((root, q, 'shared', 'KT') for root in DEV_ROOTS for q in range(11, 23))
CONF_ROOTS = tuple((f'SDRU-CCONF-T{t:02d}-V{v}-I00-P{p}', f'T{t:02d}', p)
    for t, v in ((6, '00700'), (7, '00774'), (8, '01029'), (9, '00827')) for p in ('C', 'T'))
CONF_KEYS = tuple((r, q, 'shared', bg) for r, _, _ in CONF_ROOTS for q in range(11, 23)
    for bg in (('KT', 'KC') if q in (11, 16, 22) else ('KT',)))
WINDOWS = (('first5', 5), ('full10', 10))


def need(ok, message):
    if not ok:
        raise ValueError(message)


def array(value, shape=None):
    x = np.array(value, dtype=np.float64, copy=True)
    need((shape is None or x.shape == shape) and np.isfinite(x).all(), 'finite_array_shape')
    return x


def identity(x):
    x = np.ascontiguousarray(x)
    return {'shape': list(x.shape), 'dtype': x.dtype.str,
            'sha256': hashlib.sha256(x.tobytes()).hexdigest()}


def task_root_weights(tasks, roots):
    """Equal tasks, equal roots within task, equal rows within root; sum=1."""
    tasks, roots = list(tasks), list(roots)
    need(len(tasks) == len(roots) and len(tasks) > 0, 'nonempty_matching_group_labels')
    mapping = {}
    for task, root in zip(tasks, roots):
        need(isinstance(task, str) and isinstance(root, str), 'explicit_group_strings')
        need(root not in mapping or mapping[root] == task, 'one_task_per_root')
        mapping[root] = task
    unique_tasks = sorted(set(tasks))
    result = np.empty(len(tasks), np.float64)
    for task in unique_tasks:
        rr = sorted(r for r, t in mapping.items() if t == task)
        for root in rr:
            ii = [i for i, pair in enumerate(zip(tasks, roots)) if pair == (task, root)]
            result[ii] = 1.0 / (len(unique_tasks) * len(rr) * len(ii))
    need(np.isclose(result.sum(), 1.0, atol=1e-14, rtol=0), 'normalized_group_weights')
    return result


def weights(value, n):
    w = array(value, (n,))
    need((w > 0).all() and np.isclose(w.sum(), 1.0, atol=1e-12, rtol=0), 'positive_weights_sum_one')
    return w / w.sum()


@dataclass
class Scale:
    mean: np.ndarray
    std: np.ndarray
    constant: np.ndarray

    def apply(self, x):
        x = array(x)
        need(x.ndim == 2 and x.shape[1] == len(self.mean), 'feature_width')
        z = (x - self.mean) / self.std
        z[:, self.constant] = 0.0
        need(np.isfinite(z).all(), 'finite_scaled_features')
        return z

    def manifest(self):
        return {'mean': identity(self.mean), 'std': identity(self.std),
                'constant_columns': np.flatnonzero(self.constant).tolist(),
                'constant_columns_rule': 'DEV-exact-constant; discarded also at inference'}


def fit_scale(x, w, context_mean, context_scale):
    """First nine use frozen DEV192 scale; only extra columns use DEV48."""
    x = array(x)
    need(x.ndim == 2 and x.shape[1] >= 9, 'context9_then_extra_features')
    w = weights(w, len(x))
    cm, cs = array(context_mean, (9,)), array(context_scale, (9,))
    need((cs > 0).all(), 'positive_frozen_context_scale')
    extra = x[:, 9:]
    mu = w @ extra
    constant = np.all(extra == extra[:1], axis=0)
    variance = w @ ((extra - mu) ** 2)
    need((variance[~constant] > 0).all(), 'nonconstant_feature_variance_resolvable')
    sd = np.sqrt(variance)
    sd[constant] = 1.0  # computational divisor only; these columns become zero
    return Scale(np.r_[cm, mu], np.r_[cs, sd], np.r_[np.zeros(9, bool), constant])


class RidgeDesign:
    """Weighted centered thin SVD; objective sum(w*err²)+lambda*||B||²."""
    def __init__(self, x, w, *, check=lambda: None):
        self.x = array(x)
        need(self.x.ndim == 2 and self.x.shape[0] >= 2, 'ridge_design_matrix')
        self.w = weights(w, len(x)); self.mean = self.w @ self.x
        check()
        centered = self.x - self.mean
        centered[:, np.all(self.x == self.x[:1], axis=0)] = 0.0
        a = np.sqrt(self.w)[:, None] * centered
        self.u, self.s, self.vt = np.linalg.svd(a, full_matrices=False)
        check()
        self.rank_tolerance = (max(a.shape) * np.finfo(np.float64).eps * self.s[0]) if len(self.s) else 0.0
        self.active = self.s > self.rank_tolerance
        self.rank = int(self.active.sum())
        self.s = self.s[self.active]; self.u = self.u[:, self.active]; self.vt = self.vt[self.active]

    def edf(self, lam):
        need(np.isfinite(lam) and lam > 0, 'positive_finite_lambda')
        return float(1.0 + np.sum(self.s ** 2 / (self.s ** 2 + lam)))

    def match_edf(self, target):
        need(np.isfinite(target) and target >= 1, 'finite_total_edf_target')
        if self.rank == 0 and target == 1:
            return {'status': 'MATCHED', 'lambda': 1.0, 'target_edf': target, 'actual_edf': 1.0}
        if not 1 < target < 1 + self.rank:
            return {'status': 'UNATTAINABLE', 'lambda': None, 'target_edf': float(target),
                    'actual_edf': None, 'finite_positive_lambda_edf_open_interval': [1.0, float(1 + self.rank)]}
        lo, hi = 1.0, 1.0
        for _ in range(1074):
            if self.edf(lo) >= target: break
            lo *= 0.5
            need(lo > 0, 'edf_lower_bracket_underflow')
        for _ in range(1024):
            if self.edf(hi) <= target: break
            hi *= 2.0
            need(np.isfinite(hi), 'edf_upper_bracket_overflow')
        for _ in range(100):
            middle = math.exp((math.log(lo) + math.log(hi)) / 2)
            got = self.edf(middle)
            if abs(got - target) <= 1e-11 * max(1.0, target): break
            if got > target: lo = middle
            else: hi = middle
        need(abs(got - target) <= 1e-9 * max(1.0, target), 'edf_match_tolerance')
        return {'status': 'MATCHED', 'lambda': float(middle), 'target_edf': float(target), 'actual_edf': got}

    def fit(self, z, lam):
        z = array(z)
        need(z.ndim == 2 and len(z) == len(self.x), 'target_rows_match_design')
        self.edf(lam)
        mean = self.w @ z
        projected = self.u.T @ (np.sqrt(self.w)[:, None] * (z - mean))
        dual = (self.s / (self.s ** 2 + lam))[:, None] * projected
        return RidgeModel(self, dual, mean, float(lam))


@dataclass
class RidgeModel:
    design: RidgeDesign
    dual: np.ndarray
    target_mean: np.ndarray
    lam: float

    def predict(self, x):
        x = array(x)
        need(x.ndim == 2 and x.shape[1] == self.design.x.shape[1], 'predict_design_width')
        return ((x - self.design.mean) @ self.design.vt.T) @ self.dual + self.target_mean

    def manifest(self):
        return {'lambda': self.lam, 'features': self.design.x.shape[1], 'rank': self.design.rank,
                'rank_tolerance': float(self.design.rank_tolerance), 'total_edf': self.design.edf(self.lam),
                'intercept_edf': 1.0, 'coefficient_edf_per_target': self.design.edf(self.lam) - 1,
                'outputs': self.dual.shape[1], 'dual': identity(self.dual),
                'centered_target_mean': identity(self.target_mean)}


class Suite:
    def __init__(self, scales, models, unavailable, target_mean, target_scale, fit_manifest):
        self.scales, self.models, self.unavailable = scales, models, unavailable
        self.target_mean, self.target_scale = target_mean, target_scale
        self.fit_manifest = fit_manifest

    def manifest(self):
        return {**self.fit_manifest, 'models': {k: v.manifest() for k, v in self.models.items()},
                'unavailable': self.unavailable,
                'feature_scaling': {k: v.manifest() for k, v in self.scales.items()},
                'target_mean': identity(self.target_mean), 'target_scale': identity(self.target_scale),
                'CONF_used_to_fit_scale_lambda_or_model': False,
                'edf_equal_does_not_equal_hypothesis_class': True}

    def arrays(self):
        """Optional reproducible compact fit payload; no duplicate full X."""
        result = {'target_mean': self.target_mean, 'target_scale': self.target_scale}
        for family, scale in self.scales.items():
            for name in ('mean', 'std', 'constant'):
                result[family + '_scale_' + name] = getattr(scale, name)
            design = self.models[family + '@lambda1'].design
            for name in ('mean', 's', 'vt', 'w'):
                result[family + '_design_' + name] = getattr(design, name)
        for name, model in self.models.items():
            result[name + '_dual'] = model.dual
            result[name + '_target_mean'] = model.target_mean
        return result

    def predict(self, features, *, zero_actual_E=None):
        need(set(features) == set(FAMILIES), 'complete_feature_family_roster')
        n = len(features['c9']); prepared = {}
        for family in FAMILIES:
            prepared[family] = self.scales[family].apply(features[family])
            need(len(prepared[family]) == n, 'same_prediction_rows_all_families')
        zero = np.zeros(n, bool) if zero_actual_E is None else np.asarray(zero_actual_E)
        need(zero.shape == (n,) and zero.dtype == bool, 'exactzero_edit_boolean_flags')
        result = {'zero': np.zeros((n, 10, 7), np.float64)}
        for name, model in self.models.items():
            family = name.split('@')[0]
            out = (model.predict(prepared[family]) * self.target_scale + self.target_mean).reshape(n, 10, 7)
            out[zero] = 0.0
            need(np.isfinite(out).all(), 'finite_predictions')
            result[name] = out
        return result

    def predict_rows(self, rows):
        return self.predict(features_from_rows(rows), zero_actual_E=np.array([r['zero_actual_E'] for r in rows], bool))


def fit_suite(dev_features, dev_targets, w, frozen, *, fit_provenance=None, check=lambda: None):
    """Low-level DEV arrays only. There is deliberately no evaluation argument."""
    need(set(dev_features) == set(FAMILIES), 'complete_DEV_feature_families')
    y = array(dev_targets); need(y.ndim == 3 and y.shape[1:] == (10, 7), 'signed_target_10x7')
    w = weights(w, len(y))
    tm = array(frozen['target_mean']).reshape(70); ts = array(frozen['target_scale']).reshape(70)
    need((ts > 0).all(), 'positive_original_target_scale')
    cm = array(frozen['feature_mean']).reshape(13)[:9]; cs = array(frozen['feature_scale']).reshape(13)[:9]
    z = (y.reshape(len(y), 70) - tm) / ts
    scales, designs, models, unavailable = {}, {}, {}, {}
    common = array(dev_features['c9'], (len(y), 9))
    for family in FAMILIES:
        check(); x = array(dev_features[family])
        need(x.ndim == 2 and len(x) == len(y) and np.array_equal(x[:, :9], common), 'same_context9_all_families')
        scales[family] = fit_scale(x, w, cm, cs)
        designs[family] = RidgeDesign(scales[family].apply(x), w, check=check)
    anchor = designs['current4'].edf(1.0)
    for family, design in designs.items():
        for lam in LAMBDAS:
            check(); models[f'{family}@lambda{lam:g}'] = design.fit(z, lam)
        match = design.match_edf(anchor); name = family + '@edf_match'
        if match['status'] == 'MATCHED': models[name] = design.fit(z, match['lambda'])
        else: unavailable[name] = match
    return Suite(scales, models, unavailable, tm, ts,
        {'schema': 'signed-transfer-ridge-fit-v1', 'training_rows': len(y),
         'objective': 'sum(w * DEV_sigma_scaled_residual_squared) + lambda * coefficient_Frobenius_squared',
         'weights': identity(w), 'target': identity(y),
         'training_features': {k: identity(v) for k, v in dev_features.items()},
         'training_provenance': json.loads(json.dumps(fit_provenance)), 'fixed_lambdas': list(LAMBDAS),
         'primary_lambda': 1.0, 'lambda_selected_using_labels': False,
         'edf_anchor': {'model': 'current4@lambda1', 'total_edf': anchor},
         'context_scale_source': 'original frozen DEV192', 'extra_scale_source': 'weighted DEV48 only',
         'target_scale_source': 'original frozen DEV192 per-position per-dimension',
         'intercept': 'unpenalized, fitted separately per model',
         'edf_scope': 'linear ridge smoother before exactzero-edit prediction override',
         'full_write_semantics': 'FP64 gated algebraic proxy; not actual edited native W',
         'information_cutoff': 'unedited natural j5 through v5; before persistent j5:9 edited branch',
         'posthoc_CONF_transfer_not_blind': True})


def features_from_rows(rows):
    rows = list(rows); need(bool(rows), 'nonempty_rows')
    context = np.stack([array(r['context9'], (9,)) for r in rows])
    result = {'c9': context}
    for family, (field, shape) in EXTRA_FIELDS.items():
        extra = np.stack([array(r[field], shape).ravel() for r in rows])
        result[family] = np.c_[context, extra]
    result['headpos_signed_sum'] = np.c_[context, np.stack([
        array(r['current_headpos_read'], (8, 10, 256)).sum(axis=(0, 1)) for r in rows])]
    for row in rows:
        read = array(row['current_headpos_read'], (8, 10, 256))
        norms = array(row['current_headpos_norms'], (80,))
        need(np.allclose(norms, np.linalg.norm(read, axis=-1).ravel(), rtol=2e-12, atol=0), 'norm80_exact_head_position_order')
        need(type(row['zero_actual_E']) is bool, 'explicit_zero_actual_E_flag')
    return result


def fit_rows(rows, frozen, *, check=lambda: None):
    """Production wrapper rejects CONF, missing, duplicate, and nonshared rows."""
    rows = list(rows); keys = [tuple(r['key']) for r in rows]
    need(len(keys) == 48 and len(set(keys)) == 48 and set(keys) == set(DEV_KEYS) and
         all(r['split'] == 'DEV' for r in rows), 'exact_DEV48_shared_KT_only')
    by = dict(zip(keys, rows)); rows = [by[k] for k in DEV_KEYS]
    y = np.stack([array(r['effect'], (10, 7)) for r in rows])
    need(all(not r['zero_actual_E'] or not np.any(r['effect']) for r in rows), 'zero_edit_target_exact_noop')
    w = task_root_weights([k[0].split('-')[2] for k in DEV_KEYS], [k[0] for k in DEV_KEYS])
    return fit_suite(features_from_rows(rows), y, w, frozen, check=check,
        fit_provenance=[{'key': list(r['key']), 'zero_actual_E': r['zero_actual_E'],
                         'provenance': r['provenance']} for r in rows])


def score_predictions(predictions, actual, target_scale):
    """Return per-query MSE grids [query,window,raw-or-DEV_sigma,dimension]."""
    y = array(actual); need(y.ndim == 3 and y.shape[1:] == (10, 7), 'score_signed_10x7')
    sd = array(target_scale).reshape(10, 7); need((sd > 0).all(), 'positive_score_scale')
    names = list(predictions); errors = []
    for name in names:
        err = array(predictions[name], y.shape) - y
        errors.append(np.stack([np.stack([np.mean(err[:, :end] ** 2, axis=1),
            np.mean((err[:, :end] / sd[:end]) ** 2, axis=1)], axis=1) for _, end in WINDOWS], axis=1))
    target = np.stack([np.stack([np.mean(y[:, :end] ** 2, axis=1),
        np.mean((y[:, :end] / sd[:end]) ** 2, axis=1)], axis=1) for _, end in WINDOWS], axis=1)
    return {'model_names': names, 'windows': [w for w, _ in WINDOWS], 'scales': ['raw', 'DEV_sigma'],
            'mse': np.stack(errors), 'target_mse': target}


def _aggregate(scores, keys, backgrounds):
    groups = []; lookup = {root: (task, parent) for root, task, parent in CONF_ROOTS}
    for bg in backgrounds:
        for parent in ('C', 'T'):
            tasks = sorted({t for _, t, p in CONF_ROOTS if p == parent})
            selections = [('population', 'all', [i for i, k in enumerate(keys) if k[3] == bg and lookup[k[0]][1] == parent])]
            for task in tasks:
                ii = [i for i, k in enumerate(keys) if k[3] == bg and lookup[k[0]] == (task, parent)]
                selections.append(('task', task, ii))
            for root, task, pp in CONF_ROOTS:
                if pp == parent: selections.append(('root', root, [i for i, k in enumerate(keys) if k[3] == bg and k[0] == root]))
            for level, label, ii in selections:
                need(bool(ii), 'nonempty_declared_evaluation_group')
                w = task_root_weights([lookup[keys[i][0]][0] for i in ii], [keys[i][0] for i in ii])
                mse = np.einsum('i,miwsd->mwsd', w, scores['mse'][:, ii])
                target = np.einsum('i,iwsd->wsd', w, scores['target_mse'][ii])
                ratio = np.divide(mse, target[None], out=np.full_like(mse, np.nan), where=target[None] > 0)
                groups.append({'level': level, 'label': label, 'parent': parent, 'background': bg,
                    'planned_rows': len(ii), 'observed_rows': len(ii), 'mse': mse,
                    'target_mse': target, 'relative_to_zero': ratio, 'ratio_defined': target > 0})
    return groups


def evaluate_rows(suite, rows):
    """Full fixed support only; KT96 and KC24 never pooled, C/T separate."""
    rows = list(rows); keys = [tuple(r['key']) for r in rows]
    need(len(keys) == len(set(keys)) == len(CONF_KEYS) and set(keys) == set(CONF_KEYS) and
         all(r['split'] == 'CONF' for r in rows), 'complete_CONF96_KT_and24_KC_no_survivor_reweighting')
    by = dict(zip(keys, rows)); rows = [by[k] for k in CONF_KEYS]
    pred = suite.predict_rows(rows); actual = np.stack([array(r['effect'], (10, 7)) for r in rows])
    scores = score_predictions(pred, actual, suite.target_scale)
    groups = _aggregate(scores, CONF_KEYS, ('KT', 'KC'))
    index = {key: i for i, key in enumerate(CONF_KEYS)}
    kc = [i for i, key in enumerate(CONF_KEYS) if key[3] == 'KC']
    kt = [index[(*CONF_KEYS[i][:3], 'KT')] for i in kc]
    for a, b in zip(kt, kc):
        for field in ('context9', 'originalnorm4', 'pooled4', 'pooled_gated_write'):
            need(np.array_equal(rows[a][field], rows[b][field]), 'paired_own_K_same_context_edit_and_pooled_features')
        need(rows[a]['zero_actual_E'] == rows[b]['zero_actual_E'], 'paired_same_zero_edit_flag')
    interaction_pred = {name: value[kc] - value[kt] for name, value in pred.items()}
    interaction_actual = actual[kc] - actual[kt]
    interaction_scores = score_predictions(interaction_pred, interaction_actual, suite.target_scale)
    interaction_keys = [(*CONF_KEYS[i][:3], 'KC_minus_KT') for i in kc]
    interaction = {'keys': [list(k) for k in interaction_keys], 'predictions': interaction_pred,
        'actual': interaction_actual, 'query_mse': interaction_scores['mse'],
        'query_target_mse': interaction_scores['target_mse'],
        'groups': _aggregate(interaction_scores, interaction_keys, ('KC_minus_KT',)),
        'scale_scope': 'same original DEV target sigma; not interaction-estimated',
        'auxiliary_only': True, 'no_refit_or_lambda_selection': True}
    return {'schema': 'signed-transfer-evaluation-v1', 'model_names': scores['model_names'],
            'windows': scores['windows'], 'scales': scores['scales'], 'keys': [list(k) for k in CONF_KEYS],
            'predictions': pred, 'query_mse': scores['mse'], 'query_target_mse': scores['target_mse'],
            'groups': groups, 'paired_interaction': interaction, 'unavailable_models': suite.unavailable,
            'posthoc_CONF_transfer_not_blind': True, 'positions_and_dims_are_not_independent_samples': True}
