"""Strict grouped DEV persistent-effect diagnostics. NumPy, no IO/model/cloud."""
import hashlib
import json
import math
import numpy as np

ARMS=('shared','random_matched','distance_matched','full_v_restore')
FAMILIES=('zero','arm_mean','context','current4','pooled','current_full','rbf')
LEARNED=FAMILIES[2:]
SIMPLE=('zero','arm_mean','context','current4','pooled')
LAMBDAS=(.1,1.,10.)
ROOTS=tuple(f'SDRU-CDEV-T0{t}-R0{r}'for t in range(2)for r in range(2))
QUERY_KEYS=tuple((r,q)for r in ROOTS for q in range(11,23))


def need(ok,code):
    if not ok:raise ValueError(code)


def arr(x,shape=None):
    x=np.asarray(x,dtype=np.float64)
    need((shape is None or x.shape==shape)and np.isfinite(x).all(),'finite_array_shape')
    return x


def aid(x):
    x=np.ascontiguousarray(x)
    return {'shape':list(x.shape),'dtype':x.dtype.str,'sha256':hashlib.sha256(x.tobytes()).hexdigest()}


def serial(x):
    if isinstance(x,np.ndarray):return serial(x.tolist())
    if isinstance(x,np.generic):return serial(x.item())
    if isinstance(x,float):
        need(not math.isinf(x),'infinite_output_is_not_undefined')
        return None if math.isnan(x)else x
    if isinstance(x,dict):return {str(k):serial(v)for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [serial(v)for v in x]
    return x


def group_weights(tasks,clusters,queries=None,arms=None):
    """Equal task/cluster/query/arm weights; no duplicated-query sample bonus."""
    n=len(tasks);need(n and len(clusters)==n,'nonempty_weight_groups')
    if queries is None:queries=list(range(n))
    if arms is None:arms=['single']*n
    keys=list(zip(tasks,clusters,queries,arms));need(len(keys)==len(set(keys)),'unique_weight_rows')
    result=np.zeros(n);tt=sorted(set(tasks))
    for t in tt:
        cc=sorted({c for task,c,_,_ in keys if task==t})
        for c in cc:
            qq=sorted({q for task,cl,q,_ in keys if (task,cl)==(t,c)})
            for q in qq:
                ii=[i for i,k in enumerate(keys)if k[:3]==(t,c,q)]
                result[ii]=1/(len(tt)*len(cc)*len(qq)*len(ii))
    need(np.isclose(result.sum(),1,rtol=0,atol=1e-12),'normalized_group_weights')
    return result


def exact_mean(x,w):
    x=arr(x);m=w@x
    return np.where(np.all(x==x[:1],axis=0),x[0],m)


class Block:
    def __init__(self,x,w):
        x=arr(x);self.mean=exact_mean(x,w)
        self.scale=float(np.sqrt(w@np.sum((x-self.mean)**2,axis=1)))
        need(np.isfinite(self.mean).all()and math.isfinite(self.scale),'finite_block_RMS')
        self.constant=self.scale==0
    def apply(self,x):
        x=arr(x)
        return np.zeros_like(x)if self.constant else(x-self.mean)/self.scale
    def manifest(self):return {'mean':aid(self.mean),'RMS':self.scale,'constant':self.constant,'metric':'one centered Frobenius RMS scalar, not coordinate whitening'}


class PCA:
    def __init__(self,x,w,rank=4,whiten=False):
        self.mean=exact_mean(x,w);center=arr(x)-self.mean
        need(np.isfinite(center).all(),'finite_centered_PCA_input')
        _,s,vt=np.linalg.svd(np.sqrt(w)[:,None]*center,full_matrices=False)
        need(np.isfinite(s).all()and np.isfinite(vt).all(),'finite_PCA_solution')
        tol=max(center.shape)*np.finfo(float).eps*(s[0]if len(s)else 0)
        count=min(rank,int(np.count_nonzero(s>tol)))
        self.basis=vt[:count].T.copy();self.s=s[:count].copy();self.whiten=whiten
        for j in range(count):
            if self.basis[np.argmax(np.abs(self.basis[:,j])),j]<0:self.basis[:,j]*=-1
    def apply(self,x):
        z=(arr(x)-self.mean)@self.basis
        return z/self.s if self.whiten and len(self.s)else z
    def manifest(self):return {'mean':aid(self.mean),'basis':aid(self.basis),'singular_values':self.s.tolist(),'whiten':self.whiten,'rank':len(self.s)}


def local_write(p,e,wo,gate):
    n=len(e)
    need(p.shape[2]==n+10,'full_key_denominator_retained')
    value=np.einsum('hpn,nd->phd',p[:,:,:n],e)
    return (np.einsum('phd,hdf->pf',value,wo)*gate).ravel()


class Dataset:
    def __init__(self,d,check,synthetic):
        self.raw=d;self.check=check;self.arms=tuple(d['arms']);self.groups=d['groups']
        need(d['schedule']=='persistent_j5_to_j9','actual_persistent_not_pulse')
        need(self.arms==ARMS and len(self.groups)==4,'actual_four_arms_four_init_groups')
        self.registry={g['root_id']:g for g in self.groups}
        need(len(self.registry)==4 and len({g['cluster_id']for g in self.groups})==4,'four_distinct_verified_clusters')
        identities={(g['task_id'],g['library_sha256'],g['init_index'],g['init_state_sha256'])for g in self.groups}
        need(len(identities)==4 and len({(g['task_id'],g['init_state_sha256'])for g in self.groups})==4,'four_distinct_typed_initialization_identities')
        self.clusters=tuple(sorted(g['cluster_id']for g in self.groups));self.tasks={g['cluster_id']:g['task_id']for g in self.groups}
        need(len(set(self.tasks.values()))==2 and all(list(self.tasks.values()).count(t)==2 for t in set(self.tasks.values())),'two_tasks_two_clusters_each')
        self.q=sorted(d['queries'],key=lambda q:(q['root_id'],q['query_index']))
        keys=[(q['root_id'],q['query_index'])for q in self.q]
        need(len(keys)==len(set(keys))and set(q['root_id']for q in self.q)==set(self.registry),'unique_complete_query_roots')
        if not synthetic:need(set(keys)==set(QUERY_KEYS),'all48_original_DEV_queries')
        self.wo=arr(d['wo']);self.p_shape=None
        self.y=[];self.current=[];self.row_meta=[];self.qindex=[];self.armindex=[];self.zero=[];self.dose=[]
        for i,q in enumerate(self.q):
            check();g=self.registry[q['root_id']]
            need(q['cluster_id']==g['cluster_id']and q['task_id']==g['task_id'],'query_initialization_cluster_binding')
            need(q.get('split','DEV')=='DEV','DEV_only_no_CONF_input')
            need(set(q['edits'])==set(q['targets'])==set(self.arms),'complete_declared_arms_per_query')
            p=arr(q['p']);mask=np.asarray(q['mask']);gate=arr(q['gate']);n=p.shape[2]-10
            need(mask.dtype==bool and mask.shape==p.shape[1:]and n>0,'full_native_mask_shape')
            need((p>=0).all()and(p[:,~mask]==0).all()and(p.sum(-1)>0).all(),'native_P_defined_masked')
            if self.p_shape is None:self.p_shape=p.shape
            need(p.shape==self.p_shape,'fixed_token_layout_for_training_pool')
            if not synthetic:
                need(p.shape[:2]==(8,10)and self.wo.shape==(8,256,1024)and gate.shape==(1024,),'native_write_ABI')
                arr(q['state_vector'],(2592,));arr(q['first_flow_delta'],(320,))
                need(q['time_scalar']==5*q['query_index']/520,'original_policy_clock')
            for ai,arm in enumerate(self.arms):
                e=arr(q['edits'][arm]);y=arr(q['targets'][arm],(10,7))
                need(e.shape==(n,self.wo.shape[1]),'actual_edit_shape')
                zero=not np.any(e);need(not zero or not np.any(y),'zero_edit_true_target_exact_noop')
                dose=float(np.linalg.norm(e));need(math.isfinite(dose),'finite_actual_edit_dose')
                self.current.append(local_write(p,e,self.wo,gate));self.y.append(y)
                self.row_meta.append((q['root_id'],q['query_index'],arm,q['cluster_id'],q['task_id']))
                self.qindex.append(i);self.armindex.append(ai);self.zero.append(zero);self.dose.append(dose)
        self.current=np.stack(self.current);self.y=np.stack(self.y);self.qindex=np.array(self.qindex);self.armindex=np.array(self.armindex);self.zero=np.array(self.zero,bool);self.dose=np.array(self.dose)
        self.z=np.eye(len(self.arms))[self.armindex];self.folds={};self.fit_count=0;self.fit_attempt_count=0
    def row_indices(self,clusters):return np.array([i for i,r in enumerate(self.row_meta)if r[3]in clusters],int)
    def query_indices(self,clusters):return np.array([i for i,q in enumerate(self.q)if q['cluster_id']in clusters],int)
    def weights(self,indices):
        rows=[self.row_meta[i]for i in indices]
        return group_weights([r[4]for r in rows],[r[3]for r in rows],[(r[0],r[1])for r in rows],[r[2]for r in rows])
    def qweights(self,indices):
        qs=[self.q[i]for i in indices]
        return group_weights([q['task_id']for q in qs],[q['cluster_id']for q in qs],[(q['root_id'],q['query_index'])for q in qs])
    def fold(self,clusters):
        key=tuple(sorted(clusters))
        if key not in self.folds:self.folds[key]=Fold(self,key)
        return self.folds[key]


class Kernel:
    def __init__(self,x,indices,w,z,rbf=False):
        self.x=arr(x);self.indices=indices;self.w=w;self.z=z;self.rbf=rbf;xt=x[indices]
        dot=x@xt.T;train=dot[indices]
        need(np.isfinite(dot).all(),'finite_kernel_dot_product')
        self.bandwidth=None
        if rbf:
            distances=np.maximum(np.sum(x*x,axis=1)[:,None]+np.sum(xt*xt,axis=1)[None]-2*dot,0.)
            d=distances[indices];self.bandwidth=float(np.median(d[np.triu_indices(len(d),1)]))
            need(np.isfinite(distances).all()and math.isfinite(self.bandwidth),'finite_kernel_distance')
            need(self.bandwidth>0,'RBF_bandwidth_UNAVAILABLE_zero_training_median')
            self.cross=np.exp(-distances/(2*self.bandwidth));train=self.cross[indices]
        else:self.cross=dot
        sw=np.sqrt(w);zw=sw[:,None]*z[indices];self.zw=zw
        self.ztz=zw.T@zw;need(np.linalg.matrix_rank(self.ztz)==z.shape[1],'all_training_arms_present')
        self.projection=zw@np.linalg.solve(self.ztz,zw.T);self.residual=np.eye(len(w))-self.projection
        self.weighted=sw[:,None]*train*sw[None]
        centered=self.residual@self.weighted@self.residual;centered=(centered+centered.T)/2
        eigen,u=np.linalg.eigh(centered);tol=1e-10*max(1.,float(np.max(np.abs(eigen))))
        need(np.isfinite(eigen).all()and np.isfinite(u).all(),'finite_kernel_eigensystem')
        need(eigen.min()>=-tol,'kernel_PSD_beyond_roundoff')
        self.eigen=np.maximum(eigen,0);self.u=u;self.rank=int(np.count_nonzero(self.eigen>max(1.,self.eigen.max())*len(w)*np.finfo(float).eps))
    def predict(self,target,lam):
        need(lam in LAMBDAS,'bounded_lambda_grid')
        yw=np.sqrt(self.w)[:,None]*target
        theta=self.u@((self.u.T@(self.residual@yw))/(self.eigen[:,None]+lam))
        theta=self.residual@theta
        intercept=np.linalg.solve(self.ztz,self.zw.T@(yw-self.weighted@theta))
        prediction=self.z@intercept+self.cross@(np.sqrt(self.w)[:,None]*theta)
        need(np.isfinite(theta).all()and np.isfinite(intercept).all()and np.isfinite(prediction).all(),'finite_kernel_predictions')
        edf=self.z.shape[1]+float(np.sum(self.eigen/(self.eigen+lam)))
        return (prediction,{'lambda':lam,'total_EDF':edf,'unpenalized_arm_intercepts':self.z.shape[1],
            'centered_kernel_rank':self.rank,'features':self.x.shape[1],
            'RBF_median_squared_distance':self.bandwidth,'EDF_scope':'fitted smoother before common known-zero override'},
            {'theta':theta,'arm_intercepts':intercept})


class Fold:
    def __init__(self,data,clusters):
        self.data=data;self.clusters=clusters;self.indices=data.row_indices(clusters);self.w=data.weights(self.indices)
        qi=data.query_indices(clusters);qw=data.qweights(qi);queries=data.q
        state=np.stack([arr(q['state_vector'])for q in queries]);delta=np.stack([arr(q['first_flow_delta'])for q in queries])
        self.state=PCA(state[qi],qw,whiten=True);self.delta=PCA(delta[qi],qw,whiten=True)
        c=np.c_[self.state.apply(state),self.delta.apply(delta),[q['time_scalar']for q in queries]]
        self.cmean=exact_mean(c[qi],qw);cs=np.sqrt(qw@((c[qi]-self.cmean)**2));self.cconstant=np.all(c[qi]==c[qi][:1],axis=0)
        need(np.isfinite(cs).all()and np.isfinite(self.cmean).all(),'finite_context_scale')
        self.cscale=np.where(self.cconstant,1.,cs);need((self.cscale>0).all(),'fold_context_positive_scale')
        c=(c-self.cmean)/self.cscale;c[:,self.cconstant]=0
        dose=data.dose[:,None];self.dose_mean=float(exact_mean(dose[self.indices],self.w)[0])
        self.dose_std=float(np.sqrt(self.w@((dose[self.indices,0]-self.dose_mean)**2)))
        self.dose_constant=bool(np.all(dose[self.indices]==dose[self.indices][:1]))
        need(math.isfinite(self.dose_std),'finite_training_dose_scale')
        if self.dose_constant:self.dose_std=0.;dose=np.zeros_like(dose)
        else:
            need(self.dose_std>0,'nonconstant_dose_scale_resolvable');dose=(dose-self.dose_mean)/self.dose_std
        rowc=np.c_[c[data.qindex],dose];self.context_width=rowc.shape[1]
        self.context_block=Block(rowc[self.indices],self.w);rowc=self.context_block.apply(rowc)
        context=(rowc[:,None,:]*data.z[:,:,None]).reshape(len(data.y),-1)
        self.current_block=Block(data.current[self.indices],self.w);current=self.current_block.apply(data.current)
        self.write_pca=PCA(current[self.indices],self.w,whiten=False)
        self.pool=sum(qw[j]*arr(queries[i]['p'])for j,i in enumerate(qi));pooled=[]
        for q in queries:
            data.check();p=self.pool*np.asarray(q['mask'])[None];mass=p.sum(-1,keepdims=True)
            need((mass>0).all(),'fold_pooled_full_key_mass_defined')
            p=p/mass
            for arm in data.arms:pooled.append(local_write(p,arr(q['edits'][arm]),data.wo,arr(q['gate'])))
        pooled=np.stack(pooled);self.pooled_block=Block(pooled[self.indices],self.w);pooled=self.pooled_block.apply(pooled)
        self.x={'context':context,'current4':np.c_[context,self.write_pca.apply(current)],
            'pooled':np.c_[context,pooled],'current_full':np.c_[context,current]}
        self.x['rbf']=self.x['current_full'];self.current_normalized=current;self.context_design=context
        y=data.y[self.indices].reshape(len(self.indices),70);self.target_mean=exact_mean(y,self.w)
        self.target_std=np.sqrt(self.w@((y-self.target_mean)**2));constant=np.all(y==y[:1],axis=0)
        need(np.isfinite(self.target_std).all()and np.isfinite(self.target_mean).all(),'finite_target_scale')
        self.target_std[constant]=0.;self.divisor=np.where(constant,1.,self.target_std)
        self.y=(y-self.target_mean)/self.divisor;self.kernels={};self.cache={};self.unavailable={}
    def predict(self,family,lam=1.):
        key=(family,lam)
        if key in self.cache:return self.cache[key]
        self.data.check();d=self.data;d.fit_attempt_count+=family!='zero'
        if family=='zero':p=np.zeros_like(d.y);info={'family':family,'total_EDF':0,'lambda':None}
        elif family=='arm_mean':
            means=np.stack([np.average(d.y[self.indices][d.armindex[self.indices]==a],axis=0,
                weights=self.w[d.armindex[self.indices]==a])for a in range(len(d.arms))])
            p=means[d.armindex];info={'family':family,'total_EDF':len(d.arms),'lambda':None}
        else:
            if family not in self.kernels:
                try:self.kernels[family]=Kernel(self.x[family],self.indices,self.w,d.z,rbf=family=='rbf')
                except ValueError as e:
                    if str(e).startswith('RBF_bandwidth_UNAVAILABLE'):
                        self.unavailable[family]=str(e);self.cache[key]=(None,{'family':family,'status':'UNAVAILABLE','reason':str(e)});return self.cache[key]
                    raise
            scaled,info,_=self.kernels[family].predict(self.y,lam)
            p=(scaled*self.divisor+self.target_mean).reshape(-1,10,7);info['family']=family
        d.fit_count+=family!='zero'
        need(np.isfinite(p).all(),'finite_predictions_before_zero_override')
        p=np.array(p,copy=True);p[d.zero]=0.;need(np.isfinite(p).all(),'finite_predictions')
        self.cache[key]=(p,info);return p,info
    def manifest(self):
        return {'training_clusters':list(self.clusters),'training_rows':len(self.indices),
            'training_keys':[list(self.data.row_meta[i][:3])for i in self.indices],
            'unique_training_queries':len(self.data.query_indices(self.clusters)),
            'state_PCA':self.state.manifest(),'delta_v0_PCA':self.delta.manifest(),
            'context_mean':aid(self.cmean),'context_scale':aid(self.cscale),
            'context_width_before_arm_interactions':self.context_width,
            'known_edit_dose':{'definition':'Frobenius norm of actual FP32-decoded edit E',
                'training_mean':self.dose_mean,'training_std':self.dose_std,'constant':self.dose_constant,
                'fit_population':'training rows only; equal task/root/query/arm weights',
                'used_by':'every learning family, appended to fold-c9 before shared block RMS and arm interaction'},
            'context_RMS':self.context_block.manifest(),'current_RMS':self.current_block.manifest(),
            'pooled_RMS':self.pooled_block.manifest(),'write_PCA4':self.write_pca.manifest(),
            'pooled_P':aid(self.pool),'target_mean':aid(self.target_mean),'target_std':aid(self.target_std),
            'target_constant_outputs':np.flatnonzero(self.target_std==0).tolist(),
            'no_external_frozen_PCA_or_pool':True,'unavailable':self.unavailable}


def losses(pred,y,scale):
    if pred is None:return None
    error=(pred-y)**2;out=np.empty((len(y),2,2,7))
    need(np.isfinite(error).all(),'finite_raw_squared_error')
    sd=np.asarray(scale).reshape(-1,10,7)
    need(np.isfinite(sd).all()and(sd>=0).all(),'finite_nonnegative_score_scale')
    if len(sd)==1:sd=np.broadcast_to(sd,y.shape)
    for wi,end in enumerate((5,10)):
        out[:,wi,0]=error[:,:end].mean(1)
        normalized=np.divide(error[:,:end],sd[:,:end]**2,out=np.full_like(error[:,:end],np.nan),where=sd[:,:end]>0)
        out[:,wi,1]=normalized.mean(1)
        defined=np.all(sd[:,:end]>0,axis=1)
        need(np.isfinite(out[:,wi,0]).all()and np.isfinite(out[:,wi,1][defined]).all(), 'finite_defined_scores')
    return out


def tune_score(fold,pred,test_indices):
    if pred is None:return None
    ls=losses(pred[test_indices],fold.data.y[test_indices],fold.target_std)
    v=ls[:,0,1]
    if not np.isfinite(v).all():return None
    return float(fold.data.weights(test_indices)@v.mean(1))


def tune_contrast_score(fold,pred,test_indices):
    if pred is None:return None
    d=fold.data;indices=set(int(i)for i in test_indices);qs=sorted(set(int(d.qindex[i])for i in indices))
    by={(r[0],r[1],r[2]):i for i,r in enumerate(d.row_meta)}
    shared=[by[(d.q[q]['root_id'],d.q[q]['query_index'],'shared')]for q in qs]
    random=[by[(d.q[q]['root_id'],d.q[q]['query_index'],'random_matched')]for q in qs]
    need(set(shared+random)<=indices,'validation_contrast_complete_same_query')
    y=d.y[shared]-d.y[random];p=pred[shared]-pred[random]
    value=losses(p,y,fold.target_std)[:,0,1]
    if not np.isfinite(value).all():return None
    return float(d.qweights(qs)@value.mean(1))


def choose(records):
    valid=[r for r in records if r['score']is not None]
    if len(valid)!=len(records)or not valid:return None
    return min(valid,key=lambda r:(r['score'],-r['lambda']))['lambda']


def nested_choices(data,available):
    bymodel={f:[]for f in FAMILIES};details=[]
    for held in available:
        training=[c for c in available if c!=held];fold=data.fold(training);ii=data.row_indices([held])
        for family in FAMILIES:
            for lam in LAMBDAS if family in LEARNED else(1.,):
                pred,info=fold.predict(family,lam);score=tune_score(fold,pred,ii)
                row={'validation_cluster':held,'validation_task':data.tasks[held],'training_clusters':training,
                     'family':family,'lambda':lam,'score':score,
                     'contrast_score':tune_contrast_score(fold,pred,ii),'diagnostics':info}
                details.append(row);bymodel[family].append(row)
    selected={};candidates={};contrast_selected={};contrast_candidates={}
    for family,rows in bymodel.items():
        grid=[];contrast_grid=[]
        for lam in LAMBDAS if family in LEARNED else(1.,):
            rr=[r for r in rows if r['lambda']==lam]
            w=group_weights([r['validation_task']for r in rr],[r['validation_cluster']for r in rr])
            score=None if any(r['score']is None for r in rr)else float(w@np.array([r['score']for r in rr]))
            grid.append({'lambda':lam,'score':score})
            score=None if any(r['contrast_score']is None for r in rr)else float(w@np.array([r['contrast_score']for r in rr]))
            contrast_grid.append({'lambda':lam,'score':score})
        candidates[family]=grid;selected[family]=choose(grid)
        contrast_candidates[family]=contrast_grid;contrast_selected[family]=choose(contrast_grid)
    simple=[(next(r['score']for r in candidates[f]if r['lambda']==selected[f]),SIMPLE.index(f),f)
        for f in SIMPLE if selected[f]is not None]
    best=min(simple)[2]if len(simple)==len(SIMPLE)else None
    csimple=[(next(r['score']for r in contrast_candidates[f]if r['lambda']==contrast_selected[f]),SIMPLE.index(f),f)
        for f in SIMPLE if contrast_selected[f]is not None]
    cbest=min(csimple)[2]if len(csimple)==len(SIMPLE)else None
    return selected,best,{'inner_folds':details,'candidate_scores':candidates,'selected_lambda':selected,
        'selected_simple':best,'simple_selection_criterion':'inner all-arm first5 standardized score only',
        'contrast_candidate_scores':contrast_candidates,'contrast_selected_lambda':contrast_selected,
        'contrast_selected_simple':cbest,'contrast_selection_criterion':'inner same-query persistent shared-minus-random first5 standardized score; same fitted candidates, no extra fits or lambda grid'}


def orthogonal_check(data,fold,family,lam):
    x=fold.current_normalized;u=np.arange(1,x.shape[1]+1,dtype=float);u/=np.linalg.norm(u)
    rotated=x-2*(x@u)[:,None]*u[None,:]
    design=np.c_[fold.context_design,rotated];k=Kernel(design,fold.indices,fold.w,data.z,rbf=family=='rbf')
    pred,info,_=k.predict(fold.y,lam);pred=(pred*fold.divisor+fold.target_mean).reshape(-1,10,7);pred[data.zero]=0
    original,old=fold.predict(family,lam);data.fit_count+=1;data.fit_attempt_count+=1
    error=float(np.max(np.abs(pred-original)/np.maximum(1.,np.abs(original))))
    gram_error=float(np.max(np.abs(k.cross-fold.kernels[family].cross)))
    scaled_gram_error=float(np.max(np.abs(k.cross-fold.kernels[family].cross)/np.maximum(1.,np.abs(fold.kernels[family].cross))))
    need(error<=1e-8 and scaled_gram_error<=1e-9,'orthogonal_Frobenius_geometry_QA')
    return {'family':family,'lambda':lam,'maximum_scaled_prediction_error':error,
        'maximum_kernel_error':gram_error,'EDF_difference':abs(info['total_EDF']-old['total_EDF']),
        'maximum_scaled_kernel_error':scaled_gram_error,
        'fixed_transform':'one Householder reflection with u proportional to coordinate1..D',
        'passed':True,'not_a_new_scientific_direction_search':True}


def score_groups(data,predictions,scales):
    result={};raw={}
    for name,p in predictions.items():
        ls=losses(p,data.y,scales);raw[name]=ls;groups=[]
        for cluster in data.clusters:
            ii=data.row_indices([cluster])
            for arm in (*data.arms,'equal_arm'):
                selected=ii if arm=='equal_arm'else np.array([i for i in ii if data.row_meta[i][2]==arm])
                groups.append({'cluster':cluster,'task':data.tasks[cluster],'arm':arm,'rows':len(selected),
                    'MSE':None if ls is None else np.einsum('i,iwsd->wsd',data.weights(selected),ls[selected])})
        result[name]=groups
    return result,raw


def contrast(data,predictions,scales):
    index={(r[0],r[1],r[2]):i for i,r in enumerate(data.row_meta)}
    shared=np.array([index[(q['root_id'],q['query_index'],'shared')]for q in data.q])
    random=np.array([index[(q['root_id'],q['query_index'],'random_matched')]for q in data.q])
    target=data.y[shared]-data.y[random];sd=scales[shared]
    out={};arrays={'contrast_actual':target,'contrast_scale':sd};root_scores={}
    for name,p in predictions.items():
        if p is None:out[name]=None;continue
        value=p[shared]-p[random];ls=losses(value,target,sd);arrays['contrast_prediction__'+name]=value
        arrays['contrast_query_losses__'+name]=ls;groups=[]
        for c in data.clusters:
            qi=data.query_indices([c]);root=ls[qi].mean(0)
            groups.append({'cluster':c,'task':data.tasks[c],'queries':len(qi),'MSE':root})
        out[name]=groups;root_scores[name]=np.stack([g['MSE']for g in groups])
    return {'schedule':'persistent_j5_to_j9','contrast':'shared-minus-random_matched',
        'snapshots':len(data.q),'root_scores':out,
        'normalization':'outer-training-fold all-arm target sigma; not contrast-specific, random across folds',
        'pulse_result':False},arrays,root_scores


def precision_sensitivity(data,root_scores):
    result=[]
    candidate='contrast_nested__current_full'
    if candidate not in root_scores:return {'status':'UNAVAILABLE','reason':'nested_full_missing','pulse_precision':None}
    for baseline in ('contrast_nested__zero','contrast_nested__best_simple'):
        if baseline not in root_scores:continue
        a,b=root_scores[candidate],root_scores[baseline]
        # Predeclared first5 standardized seven-dimension diagnostic only.
        loss_a=a[:,0,1].mean(1);loss_b=b[:,0,1].mean(1);diff=loss_b-loss_a
        if not(np.isfinite(diff).all()and np.isfinite(loss_b).all()):
            result.append({'baseline':baseline,'status':'UNAVAILABLE_CONSTANT_SCALE'});continue
        denom=float(loss_b.mean());variance={}
        for task in sorted(set(data.tasks.values())):
            values=np.array([diff[i]for i,c in enumerate(data.clusters)if data.tasks[c]==task])
            variance[task]=float(np.var(values,ddof=1))
        scenarios=[]
        for inflation in (1.,2.,4.):
            for n in (2,4,8,16,32):
                se=math.sqrt(sum(variance.values())/n)/len(variance)*inflation
                scenarios.append({'SD_multiplier':inflation,'roots_per_task':n,'total_clusters':n*len(variance),
                    'SE_absolute_MSE_difference':se,'normal_1_96_halfwidth_relative_to_baseline':1.96*se/denom if denom>0 else None})
        required=[]
        for inflation in (1.,2.,4.):
            for halfwidth in (.10,.05):
                raw=None if denom<=0 else (1.96*inflation)**2*sum(variance.values())/(len(variance)**2*(halfwidth*denom)**2)
                n=None if raw is None else max(2,int(math.ceil(raw)))
                required.append({'SD_multiplier':inflation,'relative_halfwidth':halfwidth,'assumed_roots_per_task':n,
                    'within_planning_cap32_per_task':None if n is None else n<=32})
        result.append({'baseline':baseline,'status':'PERSISTENT_PROXY_SENSITIVITY_ONLY',
            'root_loss_difference_baseline_minus_full':diff,'task_variance':variance,
            'degenerate_empirical_variance':all(v==0 for v in variance.values()),
            'baseline_mean_loss':denom,'relative_mean_improvement':float(diff.mean()/denom)if denom>0 else None,
            'scenarios':scenarios,'hypothetical_required_roots':required})
    return {'comparisons':result,'pulse_variance':None,'pulse_10percent_margin_precision':None,
        'power_guarantee':False,'confidence_interval_claim':False,
        'assumptions':'Hypothetical independent roots, fixed two-task distribution and variance stability; not established by first-two deterministic library states or overlapping CV training folds.',
        'normalizer_randomness':'Each OOF root uses its own train-fold target scale; root loss differences include this variation.',
        'selection_scope':'Full and strongest-simple each use inner persistent contrast losses; neither uses held-out contrast labels; endpoint-selected track also retained separately'}


def run(dataset,check=lambda:None,synthetic=False):
    data=Dataset(dataset,check,synthetic);n=len(data.y);predictions={};fold_receipts=[];orth=[];curve={}
    scales=np.full_like(data.y,np.nan)
    for held in data.clusters:
        check();available=[c for c in data.clusters if c!=held];fold=data.fold(available);test=data.row_indices([held])
        scales[test]=fold.target_std.reshape(10,7)
        selected,best,selection=nested_choices(data,available);receipt={'held_out_cluster':held,'selection':selection,'model_diagnostics':{}}
        for family in FAMILIES:
            for lam in LAMBDAS if family in LEARNED else(1.,):
                key=f'fixed__{family}__{lam:g}';p,info=fold.predict(family,lam)
                predictions.setdefault(key,np.full_like(data.y,np.nan))
                if p is not None:predictions[key][test]=p[test]
                receipt['model_diagnostics'][key]=info
            key='nested__'+family;predictions.setdefault(key,np.full_like(data.y,np.nan))
            if selected[family]is not None:
                p,info=fold.predict(family,selected[family]);predictions[key][test]=p[test]
                receipt['model_diagnostics'][key]=info
        predictions.setdefault('nested__best_simple',np.full_like(data.y,np.nan))
        if best is not None:predictions['nested__best_simple'][test]=fold.predict(best,selected[best])[0][test]
        for family in FAMILIES:
            name='contrast_nested__'+family;predictions.setdefault(name,np.full_like(data.y,np.nan))
            lam=selection['contrast_selected_lambda'][family]
            if lam is not None:
                p,info=fold.predict(family,lam);predictions[name][test]=p[test];receipt['model_diagnostics'][name]=info
        predictions.setdefault('contrast_nested__best_simple',np.full_like(data.y,np.nan))
        cbest=selection['contrast_selected_simple']
        if cbest is not None:
            lam=selection['contrast_selected_lambda'][cbest]
            predictions['contrast_nested__best_simple'][test]=fold.predict(cbest,lam)[0][test]
        for family in ('current_full','rbf'):
            if selected[family]is not None:orth.append({'held_out_cluster':held,**orthogonal_check(data,fold,family,selected[family])})
        balanced=[]
        for task in sorted(set(data.tasks.values())):
            candidates=[c for c in available if data.tasks[c]==task]
            balanced.append(min(candidates,key=lambda c:hashlib.sha256(('rss-root-curve-v1:'+c).encode()).hexdigest()))
        receipt['learning_curve_training']={'two_balanced':sorted(balanced),'three_all_available':available}
        for label,training in (('two_balanced',balanced),('three_all_available',available)):
            ff=data.fold(training)
            for family in FAMILIES:
                key=label+'__'+family;curve.setdefault(key,np.full_like(data.y,np.nan))
                p,_=ff.predict(family,1.)
                if p is not None:curve[key][test]=p[test]
            curve.setdefault(label+'__target_scale',np.full_like(data.y,np.nan))[test]=ff.target_std.reshape(10,7)
        fold_receipts.append(receipt)
    unavailable={}
    for name,p in predictions.items():
        if not np.isfinite(p).all():unavailable[name]='one_or_more_fold_predictions_unavailable'
    usable={k:v for k,v in predictions.items()if k not in unavailable}
    grouped,loss_arrays=score_groups(data,usable,scales)
    con,conarrays,conroots=contrast(data,usable,scales)
    final_lambda,final_simple,final_selection=nested_choices(data,list(data.clusters));final=data.fold(data.clusters)
    final_info={}
    for family in FAMILIES:
        if final_lambda[family]is not None:
            _,info=final.predict(family,final_lambda[family]);final_info[family]=info
    curve_scores={}
    for label in ('two_balanced','three_all_available'):
        pp={f:curve[label+'__'+f]for f in FAMILIES if np.isfinite(curve[label+'__'+f]).all()}
        curve_scores[label]=score_groups(data,pp,curve[label+'__target_scale'])[0]
    summary={'schema':'rss-persistent-root-math-result-v1','analysis_complete':True,'synthetic':synthetic,
        'scope':'Historical four-arm persistent estimation diagnostic; no dual-pulse primary result',
        'schedule':dataset['schedule'],'arms':list(data.arms),'queries':len(data.q),'rows':n,
        'groups':dataset['groups'],'independence':dataset['independence'],
        'known_zero_edits':int(data.zero.sum()),'known_zero_override':'all models, all folds; no rows dropped',
        'inner_objective':'equal task/cluster/query/arm mean first5 train-fold-standardized seven-coordinate MSE',
        'fixed_lambdas':list(LAMBDAS),'lambda_tie_rule':'largest lambda on exact score tie',
        'folds':fold_receipts,'preprocessing':[f.manifest()for f in data.folds.values()],
        'OOF_root_arm_scores':grouped,'unavailable_predictions':unavailable,
        'paired_persistent_contrast':con,'precision_sensitivity':precision_sensitivity(data,conroots),
        'root_learning_curve':{'fixed_lambda':1.,'scores':curve_scores,'two_cluster_point':'one cluster per task, stable hash order',
            'three_cluster_point':'all outer-training clusters; unequal counts1/2 but equal task weighting',
            'four_cluster_OOF_point':None,'scale_caution':'Each budget fits its own target normalizer; raw dimension-wise errors remain necessary for cross-budget interpretation.'},
        'orthogonal_QA':orth,'final_DEV_lock':{'selection':final_selection,'selected_simple':final_simple,'diagnostics':final_info,
            'CONF_used':False,'descriptive_CONF_evaluation_executed':False,
            'released_executable_predictor':False,'scope':'locked DEV configuration/diagnostics; not a deployed or exported prediction service'},
        'new_fits':data.fit_count,'fit_attempts':data.fit_attempt_count,'model_calls':0,'environment_steps':0,'policy_training_updates':0,
        'new_fits_scope':'successful estimator fits including arm means and orthogonal QA; preprocessing sets counted separately',
        'preprocessing_fit_sets':len(data.folds),'PCA_fit_count':3*len(data.folds),
        'outer_estimand':'new initialization within the two already represented tasks; NOT held-task validation',
        'inner_support_caution':'An inner fold can train only on the other task; tuning mixes those supports.',
        'old_frozen_results_replaced':False,'new_blind_test':False}
    arrays={'actual_targets':data.y,'OOF_target_scale':scales,'zero_actual_E':data.zero,
        'row_keys':np.array([[str(v)for v in r[:3]]for r in data.row_meta]),
        **{'prediction__'+k:v for k,v in predictions.items()},**{'query_loss__'+k:v for k,v in loss_arrays.items()},
        **{'curve__'+k:v for k,v in curve.items()},**conarrays}
    # Preserve every candidate's existing prediction cache for independent
    # inner-selection replay, without computing any additional fit.
    summary['fold_prediction_cache']=[]
    for fi,fold in enumerate(data.folds.values()):
        arrays[f'fold_{fi:02d}__target_scale']=fold.target_std
        arrays[f'fold_{fi:02d}__training_indices']=fold.indices
        for (family,lam),(prediction,info)in fold.cache.items():
            key=f'fold_{fi:02d}__{family}__lambda{lam:g}'
            if prediction is not None:arrays[key]=prediction
            summary['fold_prediction_cache'].append({'array':key if prediction is not None else None,
                'training_clusters':list(fold.clusters),'family':family,'lambda':lam,'diagnostics':info,
                'row_order':'same global row_keys, including both train and validation rows'})
    summary['final_DEV_lock']['contrast_configs_locked_without_additional_full_DEV_fits']=True
    return {'summary':serial(summary),'arrays':arrays}
