"""Independent cache-score audit. No feature recovery, fitting, or model calls."""
import hashlib
import math
import numpy as np

FAMILIES = ('c9','norm4','pooled4','current4','pooled_full','current_full',
            'headpos_signed','headpos_norms','headpos_signed_sum')
ROOTS = tuple((f'SDRU-CCONF-T{t:02d}-V{v}-I00-P{p}',f'T{t:02d}',p)
    for t,v in ((6,'00700'),(7,'00774'),(8,'01029'),(9,'00827')) for p in ('C','T'))
KEYS = tuple((r,q,'shared',bg) for r,_,_ in ROOTS for q in range(11,23)
    for bg in (('KT','KC') if q in (11,16,22) else ('KT',)))
DEV_KEYS = tuple((f'SDRU-CDEV-T0{t}-R0{r}',q,'shared','KT')
    for t in range(2) for r in range(2) for q in range(11,23))
PAIRS = (('current_full','current4'),('current_full','pooled_full'),
         ('current_full','c9'),('current_full','zero'),
         ('headpos_signed','headpos_norms'),('headpos_signed','headpos_signed_sum'),
         ('headpos_signed','current_full'))
F2_PAIRS = tuple(pair for i,pair in enumerate(PAIRS) if i in (0,1,2,4,5))
DIMENSIONS = ('Translation x','Translation y','Translation z',
              'Rotation x','Rotation y','Rotation z','Gripper')


def need(value, code):
    if not value: raise ValueError(code)


def aid(value):
    a=np.ascontiguousarray(value)
    return {'shape':list(a.shape),'dtype':a.dtype.str,'sha256':hashlib.sha256(a.tobytes()).hexdigest()}


def numbers(value, shape=None, *, allow_undefined=False):
    a=np.asarray(value,dtype=np.float64)
    need((shape is None or a.shape==shape) and not np.isinf(a).any() and
         (allow_undefined or np.isfinite(a).all()),'finite_cache_array_shape')
    return a


class Comparator:
    def __init__(self): self.cells=0;self.maximum_absolute=0.;self.maximum_scaled=0.
    def equal(self,actual,expected,code):
        a,b=numbers(actual,allow_undefined=True),numbers(expected,allow_undefined=True)
        need(a.shape==b.shape and np.array_equal(np.isnan(a),np.isnan(b)),code+'_shape_or_undefined')
        mask=np.isfinite(a);diff=np.abs(a[mask]-b[mask]);scaled=diff/np.maximum(1.,np.abs(b[mask]))
        self.cells+=a.size
        if diff.size:
            self.maximum_absolute=max(self.maximum_absolute,float(diff.max()))
            self.maximum_scaled=max(self.maximum_scaled,float(scaled.max()))
            need(float(scaled.max())<=5e-12,code+'_numeric_mismatch')


def score(predicted,actual,sigma):
    """[model,row,window,scale,dimension], from original signed predictions."""
    diff=predicted-actual[None]
    mse=np.empty((len(predicted),len(actual),2,2,7),np.float64)
    target=np.empty((len(actual),2,2,7),np.float64)
    for w,end in enumerate((5,10)):
        mse[:,:,w,0]=np.square(diff[:,:,:end]).mean(axis=2)
        mse[:,:,w,1]=np.square(diff[:,:,:end]/sigma[None,None,:end]).mean(axis=2)
        target[:,w,0]=np.square(actual[:,:end]).mean(axis=1)
        target[:,w,1]=np.square(actual[:,:end]/sigma[None,:end]).mean(axis=1)
    return mse,target


def aggregate(query_mse,query_target,keys,backgrounds):
    lookup={root:(task,parent)for root,task,parent in ROOTS};result=[]
    for bg in backgrounds:
        for parent in ('C','T'):
            byroot={}
            for root,task,pp in ROOTS:
                if pp!=parent:continue
                indices=[i for i,k in enumerate(keys)if k[0]==root and k[3]==bg]
                need(len(indices)==(12 if bg=='KT'else 3),'complete_queries_per_root')
                byroot[root]=(query_mse[:,indices].mean(axis=1),query_target[indices].mean(axis=0),len(indices))
            groups=[('population','all',list(byroot))]
            groups += [('task',task,[root]) for root,task,pp in ROOTS if pp==parent]
            groups += [('root',root,[root]) for root in byroot]
            for level,label,roots in groups:
                mse=np.stack([byroot[r][0]for r in roots]).mean(axis=0)
                target=np.stack([byroot[r][1]for r in roots]).mean(axis=0)
                ratio=np.divide(mse,target[None],out=np.full_like(mse,np.nan),where=target[None]>0)
                count=sum(byroot[r][2]for r in roots)
                result.append(dict(level=level,label=label,parent=parent,background=bg,
                    planned_rows=count,observed_rows=count,mse=mse,target_mse=target,
                    ratio_defined=target>0,relative_to_zero=ratio))
    return result


def groupkey(g):return tuple(g[k]for k in ('level','label','parent','background'))


def check_groups(checker,stored,recomputed,code):
    old={groupkey(g):g for g in stored};new={groupkey(g):g for g in recomputed}
    need(len(old)==len(stored)and old.keys()==new.keys(),code+'_exact_group_roster')
    for key,g in new.items():
        for field in ('planned_rows','observed_rows'):need(old[key][field]==g[field],code+'_complete_support')
        for field in ('mse','target_mse','relative_to_zero','ratio_defined'):
            checker.equal(old[key][field],g[field],code+'_'+field)


def paired_comparisons(groups,names):
    index={n:i for i,n in enumerate(names)};out=[]
    for mode in ('lambda1','edf_match'):
        for fa,fb in PAIRS:
            a=fa+'@'+mode;b='zero'if fb=='zero'else fb+'@'+mode
            for g in groups:
                if g['level']!='population':continue
                tasks=[t for t in groups if t['level']=='task'and t['parent']==g['parent']and t['background']==g['background']]
                need(len(tasks)==4,'all_four_tasks_for_pair')
                delta=np.stack([t['mse'][index[b]]-t['mse'][index[a]]for t in tasks])
                out.append(dict(stronger=a,weaker=b,parent=g['parent'],background=g['background'],
                    status='DEFINED',MSE_reduction=g['mse'][index[b]]-g['mse'][index[a]],
                    task_order=[t['label']for t in tasks],task_MSE_reduction=delta,
                    positive_task_count=(delta>0).sum(0),negative_task_count=(delta<0).sum(0),
                    ties=(delta==0).sum(0),task_count=4,positive_means_lower_error=True,
                    not_independent_coordinate_tests=True))
    return out


def verify(result,evaluation,predictions,fit_arrays,fit,*,synthetic=False,check=lambda:None):
    check();c=Comparator()
    need(result['status']=='SIGNED_TRANSFER_COMPLETE'and result['analysis_complete']is True and
         result['input_qualified']is True and result['synthetic']is synthetic,'qualified_producer_result')
    need(result['support']=={'DEV_shared_KT':48,'CONF_shared_KT':96,'CONF_shared_KC':24,
         'CONF_tasks':4,'CONF_roots':8,'heads_positions_not_independent_samples':True},'complete_original_support')
    need(result['input_objects_verified']==1500 and result['new_model_calls']==result['new_env_steps']==0 and
         result['old192_results_replaced']is False and result['posthoc_CONF_not_new_blind_test']is True,'producer_scope')
    names=result['model_names'];expected={'zero'}|{f+'@'+m for f in FAMILIES for m in ('lambda0.1','lambda1','lambda10','edf_match')}
    need(len(names)==len(set(names))==37 and set(names)==expected and names==evaluation['model_names'], 'all37_predictors')
    need(evaluation['windows']==result['windows']==['first5','full10']and
         evaluation['scales']==result['scales']==['raw','DEV_sigma'],'window_scale_order')
    need([tuple(k)for k in evaluation['keys']]==list(KEYS),'exact_CONF_key_order')
    need([tuple(r['key'])for r in fit['training_provenance']]==list(DEV_KEYS),'exact_DEV_key_order')
    need(result['unavailable_models']==fit['unavailable']=={},'no_omitted_fits')
    need(result['model_diagnostics']==fit['models']and result['edf_anchor']==fit['edf_anchor'],'same_fit_metadata')
    need(aid(fit_arrays['target_mean'])==fit['target_mean']and aid(fit_arrays['target_scale'])==fit['target_scale'],'target_scale_manifest_SHA')
    sigma=numbers(fit_arrays['target_scale'],(70,)).reshape(10,7);need((sigma>0).all(),'positive_original_DEV_sigma')
    for family in FAMILIES:
        for field,key in (('mean','mean'),('std','std')):
            need(aid(fit_arrays[family+'_scale_'+field])==fit['feature_scaling'][family][key],'feature_scale_SHA')
        constant=np.asarray(fit_arrays[family+'_scale_constant']);need(constant.dtype==bool,'constant_mask_dtype')
        need(np.flatnonzero(constant).tolist()==fit['feature_scaling'][family]['constant_columns'],'constant_mask_indices')
    anchor=fit['edf_anchor']['total_edf'];max_df=0.
    for name,m in fit['models'].items():
        check();family=name.split('@')[0];s=numbers(fit_arrays[family+'_design_s'])
        need(len(s)==m['rank']and 0<=m['rank']<=47 and m['outputs']==70 and m['lambda']>0,'rank_lambda_shape')
        need(np.asarray(fit_arrays[family+'_design_vt']).shape==(m['rank'],m['features']),'right_basis_shape')
        need(aid(fit_arrays[name+'_dual'])==m['dual']and aid(fit_arrays[name+'_target_mean'])==m['centered_target_mean'],'fit_dual_intercept_SHA')
        got=1+sum(float(x*x/(x*x+m['lambda']))for x in s);c.equal(got,m['total_edf'],'recomputed_EDF')
        if name.endswith('@edf_match'):max_df=max(max_df,abs(got-anchor));need(abs(got-anchor)<1e-8,'matched_EDF_target')
    roster=set(names)|{'DEV__'+n for n in names}|{'INTERACTION__'+n for n in names}|{'DEV_actual','CONF_actual','INTERACTION_actual'}
    need(set(predictions)==roster,'exact_prediction_cache_roster')
    actual=numbers(predictions['CONF_actual'],(120,10,7));dev_actual=numbers(predictions['DEV_actual'],(48,10,7))
    need(aid(dev_actual)==fit['target'],'DEV_target_identity')
    pred=np.stack([numbers(predictions[n],(120,10,7))for n in names])
    devpred=np.stack([numbers(predictions['DEV__'+n],(48,10,7))for n in names])
    need(not np.any(pred[names.index('zero')])and not np.any(devpred[names.index('zero')]),'zero_predictor_is_exact_zero')
    qm,qt=score(pred,actual,sigma);dm,dt=score(devpred,dev_actual,sigma)
    c.equal(evaluation['query_mse'],qm,'CONF_query_MSE');c.equal(evaluation['query_target_mse'],qt,'CONF_target_MSE')
    c.equal(evaluation['DEV_training_score']['mse'],dm,'DEV_query_MSE')
    c.equal(evaluation['DEV_training_score']['target_mse'],dt,'DEV_target_MSE')
    c.equal(result['DEV_training_mean_MSE'],dm.mean(axis=1),'DEV_fit_score')
    groups=aggregate(qm,qt,KEYS,('KT','KC'));check_groups(c,evaluation['groups'],groups,'all_CONF_groups')
    check_groups(c,result['population_scores'],[g for g in groups if g['level']=='population'],'main_population')
    ix={k:i for i,k in enumerate(KEYS)};kc=[i for i,k in enumerate(KEYS)if k[3]=='KC'];kt=[ix[(*KEYS[i][:3],'KT')]for i in kc]
    ia=actual[kc]-actual[kt];ip=pred[:,kc]-pred[:,kt];ikeys=[(*KEYS[i][:3],'KC_minus_KT')for i in kc]
    inter=evaluation['paired_interaction'];need([tuple(k)for k in inter['keys']]==ikeys,'interaction_keys')
    c.equal(predictions['INTERACTION_actual'],ia,'interaction_actual_difference');c.equal(inter['actual'],ia,'interaction_saved_actual')
    for i,name in enumerate(names):c.equal(predictions['INTERACTION__'+name],ip[i],'interaction_prediction_difference')
    im,it=score(ip,ia,sigma);c.equal(inter['query_mse'],im,'interaction_query_MSE');c.equal(inter['query_target_mse'],it,'interaction_target_MSE')
    ig=aggregate(im,it,ikeys,('KC_minus_KT',));check_groups(c,inter['groups'],ig,'interaction_groups')
    check_groups(c,result['same_query_interaction']['population_scores'],[g for g in ig if g['level']=='population'],'main_interaction')
    pairs=paired_comparisons(groups,names)
    for stored in (evaluation['paired_comparisons'],result['paired_comparisons']):
        key=lambda r:tuple(r[k]for k in ('stronger','weaker','parent','background'))
        old={key(r):r for r in stored};need(len(old)==len(stored)==56,'all56_paired_comparisons')
        for p in pairs:
            r=old[key(p)];need(r['task_order']==p['task_order']and r['task_count']==4,'task_order_for_comparison')
            for field in ('MSE_reduction','task_MSE_reduction','positive_task_count','negative_task_count','ties'):
                c.equal(r[field],p[field],'paired_'+field)
    # Four task/parent cells each contain one root: combine only after equality.
    merged=[]
    for g in groups+ig:
        if g['level']=='task':
            root=next(r for r,t,p in ROOTS if t==g['label']and p==g['parent'])
            other=next(x for x in groups+ig if groupkey(x)==('root',root,g['parent'],g['background']))
            for field in ('mse','target_mse','relative_to_zero','ratio_defined'):
                c.equal(g[field],other[field],'merged_task_root_'+field)
            merged.append({'task':g['label'],'root':root,'parent':g['parent'],'background':g['background']})
    need(len(merged)==24,'exact24_duplicate_task_root_groups')
    legacy=result['legacy_reproduction']
    need(legacy['passed']is True and legacy['comparisons']==1680 and
         math.isfinite(legacy['maximum_scaled_absolute_error'])and
         legacy['maximum_scaled_absolute_error']<=legacy['tolerance']and
         legacy['old_scores_replaced']is False,'inherited_old_QA_receipt')
    qa={'schema':'signed-figures-cache-QA-v1','passed':True,'synthetic':synthetic,
        'source_job':'redacted-b76c1f9b9012','independent_current_score_recomputation':True,
        'checked_numeric_cells':c.cells,'maximum_absolute_error':c.maximum_absolute,
        'maximum_scaled_error':c.maximum_scaled,'tolerance':5e-12,'maximum_EDF_anchor_error':max_df,
        'all37_predictors':True,'DEV_rows':48,'CONF_KT_rows':96,'CONF_KC_rows':24,
        'CONF_groups':36,'interaction_groups':18,'merged_task_root_pairs':merged,
        'unique_atlas_pages':30,'CSV_retains_all54_groups':True,
        'legacy_1680_receipt_checked':True,'legacy_1680_independently_reexecuted':False,
        'old_QA':legacy,'fits_retrained':0,'original_1500_inputs_read':False,
        'feature_cache_read':False,'prediction_recomputed_from_features':False,
        'limitation':'Five caches verify scores, fit-array identities and EDF; they do not rerun original-source QA or model fitting.'}
    return {'names':names,'groups':groups,'interaction_groups':ig,'pairs':pairs,
            'DEV_mean_mse':dm.mean(axis=1),'diagnostics':fit['models'],
            'edf_anchor':anchor,'qa':qa,'synthetic':synthetic}
