"""Task-blind DEV refit and label-free future prediction. Pure NumPy, no IO/model.

Retained helpers are byte-pinned; old task-conditioned fit/predict are never used.
"""
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
import importlib.util
import sys
import numpy as np

_RETAINED_SHA = '7b5c03d4ff364034a7bf196502124a8fb16d3d9241c107f649a6d91afba23114'
def _retained():
    import hashlib
    p=Path(__file__).with_name('retained_prediction.py')
    if not p.is_file(): p=Path(__file__).resolve().parent.parent/'effect-prediction-v1/prediction.py'
    if hashlib.sha256(p.read_bytes()).hexdigest()!=_RETAINED_SHA: raise ValueError('retained_prediction_source_SHA')
    name='_task_blind_retained_prediction_v1'
    if name in sys.modules: return sys.modules[name]
    spec=importlib.util.spec_from_file_location(name,p); m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
L=_retained()
Native, Target, Seal = L.Native, L.Target, L.Seal
need, sha, encoded, is_sha, array_id, readonly = L.need,L.sha,L.encoded,L.is_sha,L.array_id,L.readonly
local_write, masked_mean, norms, PCA4, scale_fit = L.local_write,L.masked_mean,L.norms,L.PCA4,L.scale_fit
check_target, targets_by_key, target_from_endpoints = L.check_target,L.targets_by_key,L.target_from_endpoints
ARMS,FLOW,UNITS=L.ARMS,L.FLOW,L.UNITS
METHODS=('current','pooled','norm')
ROOTS=tuple(f'SDRU-CDEV-T0{t}-R0{r}' for t in range(2) for r in range(2))
FEATURE_SLICES={'joint_state_rank4':[0,4],'first_flow_rank4':[4,8],'time':[8,9],'primary':[9,13]}

class Plan:
    def __init__(self, *, readset_sha256, native_source_sha256, intervention_contract_sha256):
        need(all(is_sha(v) for v in (readset_sha256,native_source_sha256,intervention_contract_sha256)), 'source_contract_identities')
        self.readset_sha256,self.native_source_sha256,self.intervention_contract_sha256=readset_sha256,native_source_sha256,intervention_contract_sha256
        self.roots={r:SimpleNamespace(split='DEV') for r in ROOTS};self.identity=sha(encoded(self.manifest()))
    def keys(self):return tuple((r,q) for r in ROOTS for q in range(11,23))
    def manifest(self):
        return {'schema':'task-blind-dev-plan-v1','roots':list(ROOTS),'query_indices':list(range(11,23)),
            'arms':list(ARMS),'role':'all_48_queries_and_192_targets_DEV_including_revealed_R01',
            'readset_sha256':self.readset_sha256,'native_source_sha256':self.native_source_sha256,
            'intervention_contract_sha256':self.intervention_contract_sha256,'rank':4,'ridge_lambda':1.0,
            'query_weight':1/48,'target_weight':1/192,'write_reader_weight':0.5,'camera_dictionary_refit':False}

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
        need(set(self.flow) == set(FLOW) and bool(self.modified_v) and set(self.modified_v) <= set(ARMS), 'exact_natural_fields_and_declared_arms')
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


class Fitted:
    def __init__(self,plan,wo):self.plan,self.wo=plan,wo
    def arrays(self):
        out={'pooled_P':self.pooled,'WO_native_storage':self.wo.storage,
            'target_mean':self.target_mean,'target_scale':self.target_scale,'target_zero_variance':self.target_zero_variance}
        for name in ('state','first_flow','write'):
            for field in ('mean','basis','scale','singular_values'):out[name+'_'+field]=getattr(getattr(self,name),field)
        for name in METHODS:
            for field,v in self.regressions[name].items():out[name+'_'+field]=v
        return out
    def manifest(self):
        return {'schema':'task-blind-frozen-model-v1','plan':self.plan.manifest(),'plan_sha256':self.plan.identity,
            'methods':list(METHODS),'feature_count':{k:13 for k in METHODS},'feature_slices':FEATURE_SLICES,
            'ridge_lambda':1.,'unpenalized_intercept':True,
            'objective':'sum(1/192 * scaled_squared_residual) + coefficient_Frobenius_squared',
            'state_and_firstflow_population':'48 unique DEV queries each 1/48',
            'write_PCA_population':'current and pooled each 1/2; all 192 DEV query-arms each 1/192',
            'pooled_P_semantics':'full-key address mean of 48 natural T P; apply query mask and normalize full key axis',
            'task_features_or_task_lookup':False,'target_layout':[10,7],'target_units':list(UNITS),
            'native_implementation_equivalence_claim':False,'behavior_claim':False,
            'WO':self.wo.manifest(),'fit_inputs':self.fit_inputs,'fit_labels':self.fit_labels,
            'arrays':{k:array_id(v) for k,v in self.arrays().items()}}
    def verify(self):need(sha(encoded(self.manifest()))==self.identity,'frozen_fit_mutation')
    def features(self,q,arm):
        need(arm in q.modified_v,'actual_requested_edit_required')
        need(q.p_native.storage.shape[2:]==self.pooled.shape,'frozen_full_key_address_ABI')
        context=np.concatenate([self.state.apply(q.state_vector()),self.first_flow.apply(q.first_flow_vector()),[q.env_step_index/520]])
        ps={'current':q.p_native.value()[0,0].astype(np.float64),'pooled':masked_mean(self.pooled,q.mask.value()[0])}
        out={'norm':np.concatenate([context,norms(q,arm)])}
        for name,p in ps.items():out[name]=None if p is None else np.concatenate([context,self.write.apply(local_write(p,q.edit(arm),self.wo,q.gate))])
        return out
    def predict(self,q,arm):
        self.verify();q.validate(SimpleNamespace(roots={q.root_id:None}))
        zero=not np.any(q.edit(arm));out={}
        for name,x in self.features(q,arm).items():
            if zero:out[name]=np.zeros((10,7),np.float64);continue
            if x is None:out[name]=None;continue
            r=self.regressions[name];z=((x-r['feature_mean'])/r['feature_scale'])@r['coef']+r['intercept']
            out[name]=(z*self.target_scale+self.target_mean).reshape(10,7)
            need(np.isfinite(out[name]).all(),'finite_prediction')
        return out


def fit(plan,queries,targets,wo):
    by={(q.root_id,q.query_index):q for q in queries};ts=targets_by_key(targets);expected=set(plan.keys())
    need(len(by)==len(queries) and set(by)<=expected,'only_declared_DEV_queries')
    targetkeys={(*k,a) for k in expected for a in ARMS};need(set(ts)<=targetkeys,'only_declared_DEV_targets')
    if set(by)!=expected or set(ts)!=targetkeys:
        return {'status':'UNAVAILABLE_FIT','missing_queries':sorted(expected-set(by)),'missing_targets':sorted(targetkeys-set(ts)),
                'fit_expected_queries':48,'fit_expected_targets':192,'survivors_reweighted':False}
    qs=[by[k] for k in sorted(expected)];n=qs[0].validate(plan)
    need(all(q.validate(plan)==n and set(q.modified_v)==set(ARMS) for q in qs),'all_DEV_native_queries_four_actual_edits')
    m=Fitted(plan,wo)
    m.pooled=readonly(np.mean([q.p_native.value()[0,0].astype(np.float64) for q in qs],axis=0))
    m.state=PCA4(np.stack([q.state_vector() for q in qs]),np.full(48,1/48))
    m.first_flow=PCA4(np.stack([q.first_flow_vector() for q in qs]),np.full(48,1/48))
    writes=[]
    for q in qs:
        ps=(q.p_native.value()[0,0].astype(np.float64),masked_mean(m.pooled,q.mask.value()[0]))
        need(all(p is not None for p in ps),'fit_pooled_P_row_unavailable')
        for arm in ARMS:
            check_target(ts[(q.root_id,q.query_index,arm)],q,arm)
            for p in ps:writes.append(local_write(p,q.edit(arm),wo,q.gate))
    m.write=PCA4(np.stack(writes),np.full(384,1/384));del writes
    features={k:[] for k in METHODS};ys=[]
    for q in qs:
        for arm in ARMS:
            for name,x in m.features(q,arm).items():
                need(x is not None and x.shape==(13,),'thirteen_features');features[name].append(x)
            ys.append(ts[(q.root_id,q.query_index,arm)].effect.ravel())
    y=np.stack(ys);w=np.full(192,1/192)
    m.target_mean,m.target_scale=scale_fit(y,w);m.target_zero_variance=readonly(np.all(y==y[0],axis=0))
    z=(y-m.target_mean)/m.target_scale
    cm,cs=scale_fit(np.stack(features['current'])[:,:9],w);nm,ns=scale_fit(np.stack(features['norm'])[:,9:],w)
    m.regressions={}
    for name in METHODS:
        mu=readonly(np.concatenate([cm,nm if name=='norm' else np.zeros(4)]))
        sd=readonly(np.concatenate([cs,ns if name=='norm' else np.ones(4)]))
        x=(np.stack(features[name])-mu)/sd;xb=np.sum(w[:,None]*x,axis=0);zb=np.sum(w[:,None]*z,axis=0)
        xc=x-xb;zc=z-zb
        coef=np.linalg.solve(xc.T@(w[:,None]*xc)+np.eye(13),xc.T@(w[:,None]*zc))
        need(np.isfinite(coef).all(),'finite_ridge')
        m.regressions[name]={'feature_mean':mu,'feature_scale':sd,'coef':readonly(coef),'intercept':readonly(zb-xb@coef)}
    m.fit_inputs=[{'root_id':q.root_id,'query_index':q.query_index,'sha256':q.fingerprint()} for q in qs]
    m.fit_labels=[{'key':list(k),'source_identity':ts[k].source_identity,'effect':array_id(ts[k].effect)} for k in sorted(ts)]
    m.identity=sha(encoded(m.manifest()));return m


def load_frozen(manifest,arrays,*,expected_identity):
    need(sha(encoded(manifest))==expected_identity and manifest['schema']=='task-blind-frozen-model-v1','frozen_manifest_SHA')
    need(set(arrays)==set(manifest['arrays']),'complete_frozen_arrays')
    for k,a in arrays.items():need(array_id(a)==manifest['arrays'][k],'frozen_array_SHA')
    p=manifest['plan'];plan=Plan(**{k:p[k] for k in ('readset_sha256','native_source_sha256','intervention_contract_sha256')})
    need(plan.manifest()==p,'fixed_taskblind_model_definition')
    w=manifest['WO'];wo=Native(arrays['WO_native_storage'],w['logical_dtype'],w['sha256'])
    m=Fitted(plan,wo);m.pooled=readonly(arrays['pooled_P'])
    for name in ('state','first_flow','write'):
        obj=object.__new__(PCA4)
        for f in ('mean','basis','scale','singular_values'):setattr(obj,f,readonly(arrays[name+'_'+f]))
        setattr(m,name,obj)
    for f in ('target_mean','target_scale','target_zero_variance'):setattr(m,f,readonly(arrays[f]))
    m.regressions={name:{f:readonly(arrays[name+'_'+f]) for f in ('feature_mean','feature_scale','coef','intercept')} for name in METHODS}
    m.fit_inputs,m.fit_labels=manifest['fit_inputs'],manifest['fit_labels'];m.identity=expected_identity;m.verify();return m


def seal_predictions(model,population,queries,*,unavailable=None):
    """Population is predeclared root/query/arm keys, with no task labels or targets.

    Query may carry only the shared actual Vmod. Missing opportunities remain.
    Authentic remote archival is external; scoring below requires its receipt.
    """
    model.verify();keys=[tuple(k) for k in population]
    need(len(keys)==len(set(keys)) and bool(keys),'unique_predeclared_prediction_population')
    need(all(type(r)is str and bool(r) and type(q)is int and 11<=q<=22 and a in ARMS for r,q,a in keys),'prediction_population_ABI')
    expected={(r,q) for r,q,a in keys};by={(q.root_id,q.query_index):q for q in queries};missing=dict(unavailable or {})
    need(len(by)==len(queries) and not(set(by)&set(missing)) and set(by)|set(missing)==expected,'all_planned_queries_in_one_seal')
    rows=[]
    for root,qidx,arm in sorted(keys):
        key=(root,qidx)
        if key in missing:
            need(type(missing[key])is str and bool(missing[key]),'missing_reason')
            rows.append({'key':[root,qidx,arm],'available':False,'reason':missing[key],'predictions':{m:None for m in METHODS}});continue
        q=by[key];pred=model.predict(q,arm)
        rows.append({'key':[root,qidx,arm],'available':True,'input_identity':q.fingerprint(),'edit_identity':q.edit_identity(arm),
            'zero_actual_edit':not np.any(q.edit(arm)),'predictions':{m:None if a is None else a.tolist() for m,a in pred.items()}})
    doc={'schema':'task-blind-prediction-seal-v1','fit_sha256':model.identity,'population':sorted([list(k) for k in keys]),
         'population_sha256':sha(encoded(sorted([list(k) for k in keys]))),'methods':list(METHODS),'rows':rows,
         'no_target_access':True,'no_task_code_or_lookup':True,'no_refit':True,'behavior_claim':False}
    payload=encoded(doc);return Seal(payload,sha(payload))


def score_after_seal(seal,receipt,label_reader):
    d=seal.verify()
    need(d['schema']=='task-blind-prediction-seal-v1' and sha(encoded(d['population']))==d['population_sha256'],'prediction_population_identity')
    need(receipt.get('method')=='fresh_GET' and receipt.get('bytes')==len(seal.payload) and receipt.get('sha256')==seal.sha256
         and receipt.get('fit_sha256')==d['fit_sha256'] and str(receipt.get('uri','')).startswith('bos://'),'fresh_GET_before_label_access')
    allowed={tuple(r['key']) for r in d['rows'] if r['available']};targets=targets_by_key(label_reader(tuple(sorted(allowed))))
    need(set(targets)<=allowed,'labels_only_for_sealed_keys');rows=[];missing=[]
    for row in d['rows']:
        key=tuple(row['key']);t=targets.get(key)
        if not row['available'] or t is None:missing.append({'key':list(key),'reason':row.get('reason','target_missing')});continue
        need(t.edit_identity==row['edit_identity'] and t.input_identity==row['input_identity'],'same_sealed_inputs_edit')
        if row['zero_actual_edit']:need(not np.any(t.effect),'zero_edit_nonzero_target')
        for m,values in row['predictions'].items():
            if values is None:continue
            pred=np.asarray(values,np.float64);need(pred.shape==(10,7) and np.isfinite(pred).all(),'prediction_shape')
            for window,end in (('first5',5),('full10',10)):
                delta=pred[:end]-t.effect[:end]
                rows.append({'key':list(key),'method':m,'window':window,'MSE_by_dimension':np.mean(delta*delta,axis=0).tolist()})
    return {'schema':'task-blind-sealed-score-v1','rows':rows,'missing':missing,'expected_targets':len(d['population']),
        'observed_targets':len(targets),'survivors_reweighted':False,'root_aggregation_not_automatically_inferred':True,
        'mixed_units_scalar':False,'behavior_claim':False}
