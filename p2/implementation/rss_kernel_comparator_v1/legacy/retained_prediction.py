"""Pure CPU candidate: frozen input features -> fit-root ridge -> sealed holdout.

No files, cloud, model imports, native edits, or real-label access. The external
controller must enforce the fresh-GET seal before any heldout intervention call.
"""
from dataclasses import dataclass
import hashlib
import json
from types import MappingProxyType
import numpy as np

ARMS = ('shared', 'random_matched', 'distance_matched', 'full_v_restore')
READERS = ('current', 'task_mean', 'pooled_mean')
METHODS = (*READERS, 'norm_only', 'first_flow')
FEATURE_SLICES = {
    'main':{'joint_state_rank4':[0,4],'explicit_first_flow_rank4':[4,8],'task':[8,10],'time':[10,11],'primary':[11,15]},
    'first_flow':{'explicit_first_flow_rank4':[0,4],'task':[4,6],'time':[6,7],'magnitude':[7,11]}}
FLOW = tuple(side+'_'+site for side in ('C', 'T') for site in ('x0', 'v0', 'x5', 'v5'))
RANK = 4
LAMBDA = 1.0
UNITS = ('translation_command_x', 'translation_command_y', 'translation_command_z',
         'rotation_command_x', 'rotation_command_y', 'rotation_command_z', 'gripper_command')


def need(ok, code):
    if not ok: raise ValueError(code)


def encoded(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)+'\n').encode()


def sha(raw): return hashlib.sha256(raw).hexdigest()
def is_sha(v): return type(v) is str and len(v) == 64 and set(v) <= set('0123456789abcdef')
def array_id(a):
    return {'shape':list(a.shape), 'dtype':a.dtype.str, 'sha256':sha(a.tobytes(order='C'))}


def readonly(a):
    a = np.array(a, copy=True); a.setflags(write=False); return a


@dataclass(frozen=True)
class Native:
    """Already source-verified leaf; BF16 storage is uint16, not integer values.

    payload_sha256 is the raw storage hash, not the producer's domain-prefixed
    typed hash. The caller preserves original receipt + field + selection proof.
    """
    storage: np.ndarray
    logical_dtype: str
    payload_sha256: str

    def __post_init__(self):
        object.__setattr__(self, 'storage', readonly(self.storage))
        self.check()

    def check(self):
        a = self.storage
        need(not a.dtype.hasobject and is_sha(self.payload_sha256) and sha(a.tobytes()) == self.payload_sha256, 'native_payload_identity')
        expected = 'uint16' if self.logical_dtype == 'bfloat16' else self.logical_dtype
        need(self.logical_dtype in ('bfloat16', 'float32', 'float64', 'int32', 'bool') and a.dtype.name == expected, 'explicit_native_storage_dtype')

    def value(self):
        self.check()
        a = self.storage
        if self.logical_dtype == 'bfloat16': a = (a.astype(np.uint32) << np.uint32(16)).view(np.float32)
        need(np.isfinite(a).all(), 'finite_native_leaf')
        return a

    def manifest(self):
        self.check(); return {**array_id(self.storage), 'logical_dtype':self.logical_dtype}


@dataclass(frozen=True)
class Root:
    root_id: str
    task_condition: str
    split: str


class Plan:
    def __init__(self, roots, *, native_source_sha256, intervention_contract_sha256):
        self.roots = {r.root_id:r for r in roots}
        need(len(roots) == len(self.roots) == 4, 'four_roots_no_duplication')
        self.fit = tuple(r for r in roots if r.split == 'fit')
        self.held = tuple(r for r in roots if r.split == 'heldout')
        need(len(self.fit) == len(self.held) == 2 and len({r.task_condition for r in self.fit}) == 2, 'two_fit_two_heldout')
        need({r.task_condition for r in self.fit} == {r.task_condition for r in self.held}, 'same_task_condition_internal_holdout')
        need(all(r.root_id.endswith('-R00' if r.split == 'fit' else '-R01') for r in roots), 'whole_root_split')
        need(is_sha(native_source_sha256) and is_sha(intervention_contract_sha256), 'explicit_source_and_operation_contract')
        self.native_source_sha256, self.intervention_contract_sha256 = native_source_sha256, intervention_contract_sha256
        self.identity = sha(encoded(self.manifest()))

    def manifest(self):
        return {'schema':'effect-prediction-plan-candidate-v1', 'roots':[vars(r) for r in sorted(self.roots.values(), key=lambda x:x.root_id)],
            'query_indices':list(range(11,23)), 'arms':list(ARMS), 'layer_index':8, 'schedule':[5,6,7,8,9],
            'noise_role':'diagnostic_fixed', 'noise_label':0, 'rank':4, 'ridge_lambda':1.0,
            'fit_weight':'root 1/2; query 1/12; arm 1/4; candidate pending root execution freeze',
            'native_source_sha256':self.native_source_sha256, 'intervention_contract_sha256':self.intervention_contract_sha256}

    def keys(self, split):
        return tuple((r.root_id,q) for r in sorted(self.roots.values(),key=lambda r:r.root_id) if r.split == split for q in range(11,23))


@dataclass(frozen=True)
class Query:
    root_id: str
    query_index: int
    env_step_index: int
    source_identity: str
    p_native: Native       # [1,1,8,10,N+10] at natural T L9/j5
    mask: Native           # [1,10,N+10]
    prefix_valid: Native   # [1,N]
    prefix_positions: Native
    suffix_positions: Native
    clean_v: Native        # [1,N,1,256]
    treated_v: Native
    gate: Native           # [1,1,1024], same natural j5 receiver
    flow: dict             # exactly FLOW, [1,10,32], natural only
    state: Native          # [1,32] original consumed normalized state
    native_times: Native   # [2] j0/j5 actual FP32, not nominal replacements
    time_weak_type: Native  # [2] bool
    modified_v: dict       # exactly ARMS -> actual BF16 Vmod

    def __post_init__(self):
        object.__setattr__(self,'flow',MappingProxyType(dict(self.flow)))
        object.__setattr__(self,'modified_v',MappingProxyType(dict(self.modified_v)))

    def validate(self, plan, split=None):
        need(self.root_id in plan.roots and (split is None or plan.roots[self.root_id].split == split), 'query_root_role')
        need(type(self.query_index) is int and 11 <= self.query_index <= 22 and is_sha(self.source_identity), 'frozen_query_identity')
        need(type(self.env_step_index) is int and self.env_step_index == 5*self.query_index, 'actual_query_policy_step_clock')
        n = self.treated_v.storage.shape[1]
        need(0 < n <= 968, 'bounded_prefix_tokens')
        required = ((self.p_native, 'bfloat16', (1,1,8,10,n+10)), (self.mask, 'bool', (1,10,n+10)),
            (self.prefix_valid,'bool',(1,n)), (self.prefix_positions,'int32',(1,n)), (self.suffix_positions,'int32',(1,10)),
            (self.clean_v,'bfloat16',(1,n,1,256)), (self.treated_v,'bfloat16',(1,n,1,256)),
            (self.state,'float32',(1,32)), (self.native_times,'float32',(2,)), (self.time_weak_type,'bool',(2,)))
        for field, dtype, shape in required:
            need(isinstance(field, Native) and field.logical_dtype == dtype and field.storage.shape == shape, 'actual_query_leaf_ABI')
            field.value()
        need(self.gate.logical_dtype in ('bfloat16','float32') and self.gate.storage.shape == (1,1,1024), 'native_gate_ABI')
        self.gate.value()
        need(set(self.flow) == set(FLOW) and set(self.modified_v) == set(ARMS), 'exact_natural_fields_and_four_arms')
        for key, field in self.flow.items():
            need(field.storage.shape == (1,10,32) and field.logical_dtype in (('float32',) if key.endswith(('x0','x5')) else ('bfloat16','float32')), 'natural_flow_ABI')
            field.value()
        need(np.array_equal(self.flow['C_x0'].value(), self.flow['T_x0'].value()), 'same_actual_initial_noise_C_T')
        valid = self.prefix_valid.value()[0]
        need(valid.any() and np.array_equal(self.prefix_positions.value()[0], np.cumsum(valid,dtype=np.int32)-1), 'actual_prefix_address_support')
        need(np.array_equal(self.suffix_positions.value()[0], int(valid.sum())+np.arange(10,dtype=np.int32)), 'actual_suffix_positions')
        mask, p = self.mask.value()[0], self.p_native.value()[0,0]
        need(np.array_equal(mask[:,:n], np.broadcast_to(valid,(10,n))) and mask.any(-1).all(), 'full_receiver_mask')
        need((p >= 0).all() and (p[:,~mask] == 0).all() and (p.sum(-1) > 0).all(), 'native_P_mask_and_defined_rows')
        for field in self.modified_v.values():
            need(isinstance(field,Native) and field.logical_dtype == 'bfloat16' and field.storage.shape == (1,n,1,256), 'actual_BF16_modified_V')
            field.value()
            need(np.array_equal(field.storage[:,~valid], self.treated_v.storage[:,~valid]), 'invalid_prefix_unchanged')
        return n

    def fingerprint(self):
        values = {k:v.manifest() for k,v in vars(self).items() if isinstance(v,Native)}
        values.update(flow={k:v.manifest() for k,v in self.flow.items()}, modified_v={k:v.manifest() for k,v in self.modified_v.items()})
        return sha(encoded({'root_id':self.root_id, 'query_index':self.query_index, 'env_step_index':self.env_step_index,
                           'source_identity':self.source_identity, 'arrays':values}))

    def edit(self, arm):
        a = self.modified_v[arm].value().astype(np.float32)
        e = np.subtract(a, self.treated_v.value(), dtype=np.float32)[0,:,0]
        need(np.isfinite(e).all(), 'finite_actual_FP32_edit')
        return e

    def edit_identity(self, arm):
        return sha(encoded({'original':self.treated_v.manifest(), 'modified':self.modified_v[arm].manifest(), 'arm':arm}))

    def state_vector(self):
        return np.concatenate([self.flow[k].value().ravel().astype(np.float64) for k in FLOW]+[self.state.value().ravel().astype(np.float64)])

    def first_flow_vector(self):
        return np.subtract(self.flow['T_v0'].value().astype(np.float32),self.flow['C_v0'].value().astype(np.float32),dtype=np.float32).ravel().astype(np.float64)


def local_write(p, edit, wo, gate):
    """Float64 feature algebra after exact native decoding, not a native replay.

    p: [8,10,N+10], edit: [N,256], WO: native [8,256,1024].
    All heads share ONE KV V; suffix mass stays in the full-softmax denominator.
    """
    need(isinstance(wo,Native) and wo.logical_dtype in ('bfloat16','float32') and wo.storage.shape == (8,256,1024), 'actual_AE_WO_native_ABI')
    n = len(edit)
    need(p.shape == (8,10,n+10) and gate.storage.shape == (1,1,1024), 'write_feature_shape')
    value = np.einsum('hpn,nd->phd',p[:,:,:n].astype(np.float64),edit.astype(np.float64))
    value = np.einsum('phd,hdf->pf',value,wo.value().astype(np.float64)) * gate.value()[0,0].astype(np.float64)
    need(value.shape == (10,1024) and np.isfinite(value).all(), 'finite_signed_write_feature')
    return value.ravel()  # ONE query/arm vector, not 10 independent rows


def masked_mean(p, mask):
    value = p * mask[None]
    mass = value.sum(-1,keepdims=True)
    if np.any(mass == 0): return None  # no epsilon and no changing token support
    return value / mass


def norms(query, arm):
    use = query.prefix_valid.value()[0]
    e = query.edit(arm).astype(np.float64)
    d = np.subtract(query.treated_v.value(),query.clean_v.value(),dtype=np.float32)[0,:,0].astype(np.float64)
    positions = np.arange(len(use),dtype=np.float64)/max(len(use)-1,1)
    energy = np.sum(e*e,axis=-1)
    # Unnormalized fixed address first moment: exactly zero for a zero edit.
    return np.asarray([np.linalg.norm(e[use]),np.linalg.norm(d[use]),np.linalg.norm((d+e)[use]),np.sum(positions[use]*energy[use])])


@dataclass(frozen=True)
class Target:
    root_id: str
    query_index: int
    arm: str
    edit_identity: str
    input_identity: str
    source_identity: str
    effect: np.ndarray  # [10,7], FP64 difference of actual native adapter7s

    def __post_init__(self):
        a = readonly(self.effect)
        need(a.shape == (10,7) and a.dtype == np.float64 and np.isfinite(a).all(), 'raw_adapter_effect_10x7_FP64')
        need(all(is_sha(x) for x in (self.edit_identity,self.input_identity,self.source_identity)), 'actual_label_identities')
        object.__setattr__(self,'effect',a)

    @property
    def key(self): return (self.root_id,self.query_index,self.arm)


def target_from_endpoints(query, arm, baseline_adapter7, modified_adapter7, *, source_identity):
    """Called by the owner only after the permitted actual endpoint is available."""
    need(arm in ARMS, 'declared_target_arm')
    for field in (baseline_adapter7,modified_adapter7):
        need(isinstance(field,Native) and field.logical_dtype=='float64' and field.storage.shape==(1,10,7), 'native_adapter7_endpoint_ABI')
    effect=np.subtract(modified_adapter7.value(),baseline_adapter7.value(),dtype=np.float64)[0]
    result=Target(query.root_id,query.query_index,arm,query.edit_identity(arm),query.fingerprint(),source_identity,effect)
    check_target(result,query,arm)
    return result


def targets_by_key(targets):
    values = {t.key:t for t in targets}
    need(len(values) == len(targets), 'one_target_per_query_arm')
    return values


def check_target(t, query, arm):
    need(t.key == (query.root_id,query.query_index,arm) and t.edit_identity == query.edit_identity(arm)
         and t.input_identity == query.fingerprint(), 'target_matches_same_actual_edit_and_inputs')
    if not np.any(query.edit(arm)):
        need(not np.any(t.effect), 'zero_edit_nonzero_target_requires_native_gate')


class PCA4:
    def __init__(self, x, weights):
        need(x.ndim == 2 and x.shape[0] >= 4 and x.shape[1] >= 4 and np.isfinite(x).all(), 'fit_matrix_rank4_shape')
        w = np.asarray(weights,np.float64); need(w.shape == (len(x),) and (w > 0).all() and np.isclose(w.sum(),1), 'normalized_fit_weights')
        mean=np.sum(w[:,None]*x,axis=0)
        self.mean = readonly(np.where(np.all(x==x[0],axis=0),x[0],mean))
        _, s, vt = np.linalg.svd(np.sqrt(w)[:,None]*(x-self.mean),full_matrices=False)
        b = vt[:4].T.copy()
        for j in range(4):
            if b[np.argmax(np.abs(b[:,j])),j] < 0: b[:,j] *= -1
        self.basis, self.singular_values = readonly(b), readonly(s[:4])
        self.scale = readonly(np.where(s[:4] == 0,1,s[:4]))

    def apply(self, x): return ((x-self.mean)@self.basis)/self.scale


def scale_fit(x,w):
    mean = np.sum(w[:,None]*x,axis=0)
    constant=np.all(x==x[0],axis=0)
    mean=np.where(constant,x[0],mean)
    std = np.sqrt(np.sum(w[:,None]*(x-mean)**2,axis=0))
    std=np.where(constant,0,std)
    return readonly(mean), readonly(np.where(std == 0,1,std))


class Fitted:
    def __init__(self, plan, wo):
        self.plan, self.wo = plan, wo

    def arrays(self):
        out = {'write_mean':self.write.mean,'write_basis':self.write.basis,'write_scale':self.write.scale,
            'write_singular_values':self.write.singular_values,'state_mean':self.state.mean,'state_basis':self.state.basis,
            'state_scale':self.state.scale,'state_singular_values':self.state.singular_values,'pooled_P':self.pooled,
            'first_flow_mean':self.first_flow.mean,'first_flow_basis':self.first_flow.basis,'first_flow_scale':self.first_flow.scale,
            'first_flow_singular_values':self.first_flow.singular_values,
            'target_mean':self.target_mean,'target_scale':self.target_scale,'target_zero_variance':self.target_zero_variance}
        for i,(_,p) in enumerate(sorted(self.task.items())): out['task_P_'+str(i)] = p
        for method in METHODS:
            for key in ('feature_mean','feature_scale','coef','intercept'): out[method+'_'+key] = self.regressions[method][key]
        return out

    def manifest(self):
        return {'schema':'effect-prediction-fit-candidate-v1','plan':self.plan.manifest(),'plan_sha256':self.plan.identity,
            'methods':list(METHODS),'task_P_order':sorted(self.task),'rank':4,'ridge_lambda':1.0,
            'ridge_objective':'sum(normalized equal-root/query/arm weights * scaled squared residual) + lambda * coefficient Frobenius squared',
            'write_basis_population':'R00 only; current/task_mean/pooled_mean each weight 1/3; each query/arm 1/96',
            'state_basis_population':'R00 unique query only, 1/24; no arm replication',
            'first_flow_basis_population':'natural FP32(T_v0)-FP32(C_v0), flattened320; 24 R00 unique queries at 1/24, no arm replication',
            'feature_count_per_method':{k:11 if k=='first_flow' else 15 for k in METHODS},
            'feature_slices':FEATURE_SLICES,'unpenalized_intercept_per_method':True,'primary_feature_count':4,
            'reading_centers_scales':'one joint write PCA and scale; no per-reader recentering or rescaling',
            'shared_context':'joint-state rank4 + explicit first-flow rank4 + 2 task indicators + n/520',
            'first_flow_baseline_scope':'explicit rank4 delta-v0 + task/time + four magnitudes; not exhaustive first-flow information',
            'target_layout':[10,7],'target_units':list(UNITS),'target_scaling':'per slot/dimension fit-only weighted mean and SD; exact zero SD -> 1; invert before scoring',
            'WO':self.wo.manifest(),'fit_input_identities':self.fit_inputs,'fit_label_identities':self.fit_labels,
            'arrays':{k:array_id(a) for k,a in self.arrays().items()},'actual_native_precision_qualification':False}

    def verify(self): need(sha(encoded(self.manifest())) == self.identity, 'frozen_fit_mutation')

    def features(self, q, arm):
        first_context=np.concatenate([self.first_flow.apply(q.first_flow_vector()),
            np.asarray([float(self.plan.roots[q.root_id].task_condition == t) for t in sorted(self.task)]),np.asarray([q.env_step_index/520])])
        context=np.concatenate([self.state.apply(q.state_vector()),first_context])
        p = q.p_native.value()[0,0].astype(np.float64); mask = q.mask.value()[0]
        probabilities = {'current':p,'task_mean':masked_mean(self.task[self.plan.roots[q.root_id].task_condition],mask),
                         'pooled_mean':masked_mean(self.pooled,mask)}
        amplitude=norms(q,arm)
        result = {'norm_only':np.concatenate([context,amplitude]),'first_flow':np.concatenate([first_context,amplitude])}
        for name, value in probabilities.items():
            result[name] = None if value is None else np.concatenate([context,self.write.apply(local_write(value,q.edit(arm),self.wo,q.gate))])
        return result

    def predict(self,q,arm):
        q.validate(self.plan,'heldout'); self.verify()
        zero = not np.any(q.edit(arm)); result = {}
        for method,x in self.features(q,arm).items():
            if zero: result[method] = np.zeros((10,7),np.float64); continue
            if x is None: result[method] = None; continue
            reg = self.regressions[method]
            z = ((x-reg['feature_mean'])/reg['feature_scale'])@reg['coef'] + reg['intercept']
            y = (z*self.target_scale+self.target_mean).reshape(10,7)
            need(np.isfinite(y).all(), 'finite_raw_unit_prediction'); result[method] = y
        return result


def fit(plan, fit_queries, fit_targets, wo):
    """R00 only. Incomplete frozen support is unavailable, never reweighted."""
    by = {(q.root_id,q.query_index):q for q in fit_queries}
    need(len(by) == len(fit_queries), 'unique_fit_query')
    expected = set(plan.keys('fit')); need(set(by) <= expected, 'fit_cannot_consume_R01')
    ts = targets_by_key(fit_targets); target_keys = {(*key,a) for key in expected for a in ARMS}
    need(set(ts) <= target_keys, 'fit_labels_cannot_include_holdout_or_new_arm')
    if set(by) != expected or set(ts) != target_keys:
        return {'status':'UNAVAILABLE_FIT','missing_queries':sorted(expected-set(by)),
                'missing_targets':sorted(target_keys-set(ts)),'fit_expected_queries':24,'fit_expected_targets':96,'survivors_reweighted':False}
    qs = [by[k] for k in sorted(expected)]
    n = qs[0].validate(plan,'fit')
    need(all(q.validate(plan,'fit') == n for q in qs), 'same_padded_key_axis')
    m = Fitted(plan,wo)
    m.task = {t:readonly(np.mean([q.p_native.value()[0,0].astype(np.float64) for q in qs if plan.roots[q.root_id].task_condition == t],axis=0))
              for t in sorted({r.task_condition for r in plan.fit})}
    m.pooled = readonly(np.mean(list(m.task.values()),axis=0))
    m.state = PCA4(np.stack([q.state_vector() for q in qs]),np.full(24,1/24))
    m.first_flow = PCA4(np.stack([q.first_flow_vector() for q in qs]),np.full(24,1/24))
    writes=[]
    for q in qs:
        ps = (q.p_native.value()[0,0].astype(np.float64),masked_mean(m.task[plan.roots[q.root_id].task_condition],q.mask.value()[0]),
              masked_mean(m.pooled,q.mask.value()[0]))
        need(all(p is not None for p in ps), 'fit_fixed_P_row_unavailable')
        for arm in ARMS:
            check_target(ts[(q.root_id,q.query_index,arm)],q,arm)
            for p in ps: writes.append(local_write(p,q.edit(arm),wo,q.gate))
    m.write = PCA4(np.stack(writes),np.full(288,1/288)); del writes
    features={k:[] for k in METHODS}; targets=[]
    for q in qs:
        for arm in ARMS:
            for method,x in m.features(q,arm).items():
                width=11 if method=='first_flow' else 15
                need(x is not None and x.shape == (width,), 'frozen_15_main_11_first_flow_width'); features[method].append(x)
            targets.append(ts[(q.root_id,q.query_index,arm)].effect.ravel())
    y=np.stack(targets); w=np.full(96,1/96)
    m.target_mean,m.target_scale=scale_fit(y,w); z=(y-m.target_mean)/m.target_scale
    m.target_zero_variance=readonly(np.all(y==y[0],axis=0))
    m.regressions={}
    # Context is identical; the three reading methods keep the SAME joint
    # write center/scale. A second per-reader standardization would violate it.
    context_mu,context_sd=scale_fit(np.stack(features['current'])[:,:11],w)
    norm_mu,norm_sd=scale_fit(np.stack(features['norm_only'])[:,11:],w)
    for method in METHODS:
        cm,cs=(context_mu[4:],context_sd[4:]) if method=='first_flow' else (context_mu,context_sd)
        magnitude=method in ('norm_only','first_flow')
        mu=readonly(np.concatenate([cm,norm_mu if magnitude else np.zeros(4)]))
        sd=readonly(np.concatenate([cs,norm_sd if magnitude else np.ones(4)]))
        x=(np.stack(features[method])-mu)/sd
        # Center only inside the intercept-solving identity; it is NOT an
        # extra learned feature transform. Each method has one equally
        # unpenalized intercept (its fitted numerical value can differ).
        xbar=np.sum(w[:,None]*x,axis=0); zbar=np.sum(w[:,None]*z,axis=0)
        xc=x-xbar; zc=z-zbar
        coef=np.linalg.solve(xc.T@(w[:,None]*xc)+np.eye(xc.shape[1])*LAMBDA,xc.T@(w[:,None]*zc))
        intercept=zbar-xbar@coef
        need(np.isfinite(coef).all(), 'finite_ridge_coefficients')
        m.regressions[method]={'feature_mean':mu,'feature_scale':sd,'coef':readonly(coef),'intercept':readonly(intercept)}
    m.fit_inputs=[{'root_id':q.root_id,'query_index':q.query_index,'sha256':q.fingerprint()} for q in qs]
    m.fit_labels=[{'key':list(k),'source_identity':ts[k].source_identity,'effect':array_id(ts[k].effect)} for k in sorted(ts)]
    m.identity=sha(encoded(m.manifest())); return m


@dataclass(frozen=True)
class Seal:
    payload: bytes
    sha256: str
    def verify(self):
        need(type(self.payload) is bytes and sha(self.payload) == self.sha256, 'immutable_prediction_seal')
        return json.loads(self.payload)


def seal_all_holdout(model, queries, *, unavailable=None):
    """No labels argument. All 24 query opportunities, each four arms, in ONE seal."""
    model.verify(); by={(q.root_id,q.query_index):q for q in queries}; missing={} if unavailable is None else dict(unavailable)
    expected=set(model.plan.keys('heldout'))
    need(len(by)==len(queries) and not(set(by)&set(missing)) and set(by)|set(missing)==expected, 'all24_heldout_before_any_labels')
    rows=[]
    for key in sorted(expected):
        if key in missing:
            need(type(missing[key]) is str and bool(missing[key]), 'explicit_missing_reason')
            rows.extend({'key':[*key,arm],'available':False,'reason':missing[key],'predictions':{name:None for name in METHODS}} for arm in ARMS)
            continue
        q=by[key]; q.validate(model.plan,'heldout')
        for arm in ARMS:
            values=model.predict(q,arm)
            rows.append({'key':[*key,arm],'available':True,'input_identity':q.fingerprint(),'edit_identity':q.edit_identity(arm),
                'zero_actual_edit':not np.any(q.edit(arm)),
                'predictions':{k:None if v is None else v.tolist() for k,v in values.items()},
                'unavailable_methods':{k:'fixed_probability_mask_has_zero_mass' for k,v in values.items() if v is None}})
    doc={'schema':'all-heldout-effect-predictions-candidate-v1','plan_sha256':model.plan.identity,'fit_sha256':model.identity,
         'query_count':24,'query_arm_count':96,'target_layout':[10,7],'raw_command_units':list(UNITS),'rows':rows,
         'no_labels_read_by_this_function':True,'controller_must_block_first_heldout_intervention_until_fresh_GET':True,
         'plan':model.plan.manifest()}
    payload=encoded(doc); return Seal(payload,sha(payload))


def score_after_seal(seal, receipt, label_reader):
    """Invoke label_reader ONCE, only after exact archived seal proof validation.

    The pure module validates evidence structure/identity. Authentic cloud
    verification and ordering of GPU calls remain the root controller's job.
    """
    doc=seal.verify()
    need(doc.get('schema')=='all-heldout-effect-predictions-candidate-v1' and doc.get('query_count')==24
         and doc.get('query_arm_count')==96 and sha(encoded(doc['plan']))==doc['plan_sha256'], 'sealed_plan_and_counts')
    need(receipt.get('method')=='fresh_GET' and receipt.get('bytes')==len(seal.payload)
         and receipt.get('sha256')==seal.sha256 and receipt.get('fit_sha256')==doc['fit_sha256']
         and isinstance(receipt.get('uri'),str) and receipt['uri'].startswith('bos://'), 'remote_prediction_seal_required_before_label_access')
    need(callable(label_reader), 'deferred_heldout_label_reader')
    rows=doc['rows']; need(len(rows)==96 and len({tuple(r['key']) for r in rows})==96, 'sealed_full_heldout_coverage')
    held=[r['root_id'] for r in doc['plan']['roots'] if r['split']=='heldout']
    need(len(held)==2 and {tuple(r['key']) for r in rows}=={(r,q,a) for r in held for q in range(11,23) for a in ARMS}
         and all(set(r['predictions'])==set(METHODS) for r in rows), 'sealed_exact_heldout_keys_and_methods')
    allowed={tuple(r['key']) for r in rows if r['available']}
    targets=targets_by_key(label_reader(tuple(sorted(allowed))))
    need(set(targets)<=allowed,'labels_only_for_sealed_prediction_keys')
    errors=[]; missing=[]; unavailable_predictions=[]
    for row in rows:
        key=tuple(row['key']); t=targets.get(key)
        if not row['available'] or t is None:
            missing.append({'key':list(key),'reason':row.get('reason','actual_target_missing')}); continue
        need(t.edit_identity==row['edit_identity'] and t.input_identity==row['input_identity'], 'heldout_label_same_sealed_edit_inputs')
        if row['zero_actual_edit']: need(not np.any(t.effect),'zero_edit_nonzero_target_requires_native_gate')
        for method,value in row['predictions'].items():
            if value is None:
                unavailable_predictions.append({'key':list(key),'method':method,'reason':row['unavailable_methods'][method]}); continue
            pred=np.asarray(value,np.float64); need(pred.shape==(10,7) and np.isfinite(pred).all(),'sealed_raw_prediction_shape')
            for name,end in (('proposed_first5',5),('full10',10)):
                delta=pred[:end]-t.effect[:end]
                errors.append({'key':list(key),'method':method,'window':name,
                    'MSE_by_command_dimension':np.mean(delta*delta,axis=0).tolist(),
                    'MAE_by_command_dimension':np.mean(np.abs(delta),axis=0).tolist(),
                    'raw_units':list(UNITS),'positions_are_not_independent_samples':True})
    # Complete-root means only. Missing results are not renormalized away.
    summaries=[]
    for root in sorted({r['key'][0] for r in rows}):
        for method in METHODS:
            for window in ('proposed_first5','full10'):
                selected=[e for e in errors if e['key'][0]==root and e['method']==method and e['window']==window]
                complete=len(selected)==48
                sums=np.sum([e['MSE_by_command_dimension'] for e in selected],axis=0) if selected else np.zeros(7)
                summaries.append({'root_id':root,'method':method,'window':window,'expected_query_arms':48,'observed_query_arms':len(selected),
                    'complete':complete,'MSE_by_command_dimension':(sums/48).tolist() if complete else None,
                    'partial_numerator_divided_by_frozen_denominator_not_comparable':None if complete else (sums/48).tolist()})
    equal_root=[]
    for method in METHODS:
        for window in ('proposed_first5','full10'):
            roots=[r for r in summaries if r['method']==method and r['window']==window]
            complete=len(roots)==2 and all(r['complete'] for r in roots)
            equal_root.append({'method':method,'window':window,'root_ids':[r['root_id'] for r in roots],'root_weights':[.5,.5],
                'complete':complete,'MSE_by_command_dimension':np.mean([r['MSE_by_command_dimension'] for r in roots],axis=0).tolist() if complete else None,
                'no_survivor_root_reweighting':True})
    return {'schema':'sealed-heldout-effect-score-candidate-v1','seal_sha256':seal.sha256,'missing':missing,
            'unavailable_predictions':unavailable_predictions,'errors':errors,'roots':summaries,'equal_root':equal_root,
            'new_fit':False,'confidence_intervals':False,'behavior_claim':False,'mixed_unit_scalar_loss_reported':False}


class Predictor:
    """Small science-loop facade; source arrays and cloud persistence stay outside."""
    def __init__(self, plan, wo_native):
        self.plan,self.wo=plan,wo_native
        self.model=None; self.seal=None; self._fit_called=False; self._predict_called=False

    def fit(self, fit_queries, fit_targets):
        need(not self._fit_called, 'one_predeclared_fit_no_label_driven_retry')
        self._fit_called=True
        result=fit(self.plan,fit_queries,fit_targets,self.wo)
        if isinstance(result,Fitted): self.model=result
        return {'status':'FIT_FROZEN','fit_sha256':result.identity} if self.model is not None else result

    def export_frozen(self):
        need(self.model is not None, 'complete_fit_before_export')
        self.model.verify()
        return {'identity':self.model.identity,'metadata':self.model.manifest(),'arrays':self.model.arrays()}

    def predict_all(self, held_queries, *, unavailable=None):
        need(self.model is not None and not self._predict_called, 'one_all_heldout_prediction_seal')
        self._predict_called=True
        self.seal=seal_all_holdout(self.model,held_queries,unavailable=unavailable)
        return self.seal

    def score(self, receipt, label_reader):
        need(self.seal is not None, 'all_heldout_predictions_before_score')
        return score_after_seal(self.seal,receipt,label_reader)
