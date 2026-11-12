"""Deterministic host BF16 payloads. No labels, model call or basis fitting."""
from dataclasses import dataclass
import hashlib,json
import numpy as np
import ml_dtypes

ARMS=('shared','random_matched','distance_matched','full_v_restore')
POLICY={'schema':'per-token-native-BF16-dose-match-v1','bisection_iterations':20,'bracket_expansions':4,
        'uniform_remaining':'actual_D_plus_E_R','response_lambda':1.0,'match_support':'receiver_valid_prefix_tokens'}
def need(ok,code):
    if not ok:raise ValueError(code)
def aid(a):
    a=np.asarray(a);h={'shape':list(a.shape),'logical_dtype':a.dtype.name,'stored_dtype':a.dtype.str,'bytes':a.nbytes}
    return {**h,'sha256':hashlib.sha256(json.dumps(h,sort_keys=True,separators=(',',':')).encode()+b'\0'+a.tobytes()).hexdigest()}
def sha(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
@dataclass(frozen=True)
class Bases:
    response:np.ndarray
    random:np.ndarray
    source_identity:str
    source_proof:dict
    layer_index:int=8
    def validate(self):
        need(self.layer_index==8 and len(self.source_identity)==64,'frozen_L9_basis_identity')
        for a in(self.response,self.random):
            need(a.shape==(256,4)and a.dtype==np.float64 and np.isfinite(a).all(),'frozen_basis_FP64_256x4')
            need(np.allclose(a.T@a,np.eye(4),rtol=0,atol=1e-10),'frozen_basis_orthonormal')
        return self

@dataclass(frozen=True)
class EditBundle:
    values:dict
    actual_E:dict
    report:dict
    identity:str

def construct(clean,treated,valid,bases,policy):
    need(policy==POLICY,'explicit_frozen_dose_policy_no_defaults')
    bases.validate();c,t=np.asarray(clean),np.asarray(treated);mask=np.asarray(valid)
    need(c.shape==t.shape and c.ndim==4 and c.shape[0]==1 and c.shape[2:]==(1,256)
         and c.dtype.name==t.dtype.name=='bfloat16','native_L9_C_T_V')
    need(mask.shape==(1,t.shape[1])and mask.dtype==bool and mask.any(),'actual_prefix_support')
    use=mask[0];old=t[0,:,0].astype(np.float64);d=np.subtract(t.astype(np.float32),c.astype(np.float32),dtype=np.float32)[0,:,0].astype(np.float64)
    need(np.isfinite(old).all()and np.isfinite(d).all(),'finite_original_V')
    q=(d@bases.response)@bases.response.T;q[~use]=0
    def payload(delta):
        value=(old+delta).astype(ml_dtypes.bfloat16)
        value=np.where(delta==0,t[0,:,0],value).astype(ml_dtypes.bfloat16);value[~use]=t[0,~use,0]
        need(np.isfinite(value.astype(np.float32)).all(),'finite_BF16_candidate')
        return value[None,:,None,:]
    def e(value):return np.subtract(value.astype(np.float32),t.astype(np.float32),dtype=np.float32)
    values={'shared':payload(-q)};response=e(values['shared']);target=np.linalg.norm(response[0,:,0].astype(np.float64),axis=-1)
    direction=(d@bases.random)@bases.random.T;direction[~use]=0
    dn=np.linalg.norm(direction,axis=-1);active=use&(target>0)&(dn>0)
    lo=np.zeros(len(use));hi=np.divide(target,dn,out=np.zeros_like(target),where=active)
    best=np.array(t,copy=True);best_norm=np.zeros(len(use));best_scale=np.zeros(len(use));best_error=target.copy();evaluations=0
    def evaluate(scale):
        nonlocal best,best_norm,best_scale,best_error,evaluations
        candidate=payload(-direction*scale[:,None]);actual=np.linalg.norm(e(candidate)[0,:,0].astype(np.float64),axis=-1)
        error=np.abs(actual-target);choose=active&((error<best_error)|((error==best_error)&(scale<best_scale)))
        best[:,choose]=candidate[:,choose];best_norm[choose]=actual[choose];best_scale[choose]=scale[choose];best_error[choose]=error[choose]
        evaluations+=1;return actual
    norms=evaluate(hi)
    for _ in range(4):
        expand=active&(norms<target);lo=np.where(expand,hi,lo);hi=np.where(expand,hi*2,hi);norms=evaluate(hi)
    bracketed=active&(norms>=target)
    for _ in range(20):
        mid=(lo+hi)/2;norms=evaluate(mid);lower=bracketed&(norms<target)
        lo=np.where(lower,mid,lo);hi=np.where(bracketed&~lower,mid,hi)
    values['random_matched']=best
    before=float(np.linalg.norm(d[use]));remaining=float(np.linalg.norm((d+response[0,:,0].astype(np.float64))[use]))
    remaining_scale=1.0 if before==0 else remaining/before
    removed_fraction=1.0-remaining_scale
    values['distance_matched']=payload(-removed_fraction*d)
    full=np.array(c,copy=True);full[:,~use]=t[:,~use];values['full_v_restore']=full
    actual={a:e(v)for a,v in values.items()}
    stats={a:{'Vmod':aid(values[a]),'actual_E':aid(actual[a]),'actual_norm':float(np.linalg.norm(actual[a][0,use,0].astype(np.float64))),
              'actual_remaining':float(np.linalg.norm((d+actual[a][0,:,0].astype(np.float64))[use])),
              'zero_edit':bool(not np.any(actual[a])),'invalid_prefix_preserved':bool(np.array_equal(values[a][:,~use].view(np.uint16),t[:,~use].view(np.uint16)))}for a in ARMS}
    report={'schema':'native_L9_actual_payloads-v1','basis_identity':bases.source_identity,'policy':policy,'arms':stats,
      'random_match':{'candidate_evaluations':evaluations,'token_target_norm':target,'token_actual_norm':best_norm,'token_residual':best_norm-target,
        'token_scale':best_scale,'zero_target':use&(target==0),'zero_direction_nonzero_target':use&(target>0)&(dn==0),
        'unbracketed':active&~bracketed,'global_target_norm':float(np.linalg.norm(target[use])),
        'global_actual_norm':float(np.linalg.norm(best_norm[use])),'exact_norm_match_not_asserted':True},
      'uniform':{'remaining_scale':remaining_scale,'removed_fraction':removed_fraction,'is_shrink':0<=remaining_scale<=1,'target_actual_remaining':remaining,'actual_remaining':stats['distance_matched']['actual_remaining'],
                 'remaining_residual':stats['distance_matched']['actual_remaining']-remaining},'response_lambda':1.0}
    for a in (*values.values(),*actual.values()):a.setflags(write=False)
    return EditBundle(values,actual,report,sha({'basis':bases.source_identity,'policy':policy,'arms':stats}))
