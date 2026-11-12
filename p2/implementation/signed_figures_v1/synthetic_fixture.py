"""Explicit SYNTHETIC cache fixture; never a production input or fit."""
from pathlib import Path
import argparse
import numpy as np
import audit as A


def make():
    rng=np.random.default_rng(20261020)
    names=['zero']+[f+'@'+m for f in A.FAMILIES for m in ('lambda0.1','lambda1','lambda10','edf_match')]
    sigma=np.linspace(.3,2.1,70);mean=np.zeros(70)
    actual=rng.normal(size=(120,10,7));devactual=rng.normal(size=(48,10,7))
    actual[:,:,6]=0;devactual[:,:,6]=0  # Explicit undefined-denominator cells.
    arrays={'target_mean':mean,'target_scale':sigma};models={};scales={}
    dims=dict(c9=9,norm4=13,pooled4=13,current4=13,pooled_full=10249,
        current_full=10249,headpos_signed=20489,headpos_norms=89,headpos_signed_sum=265)
    base=np.array([.4,.7,1.1,1.8,2.7]);anchor_factor=1.3
    anchor=1+sum((base*anchor_factor)**2/((base*anchor_factor)**2+1))
    for i,family in enumerate(A.FAMILIES):
        d=dims[family];factor=1+i/10;s=base*factor
        arrays[family+'_scale_mean']=np.zeros(d);arrays[family+'_scale_std']=np.ones(d)
        arrays[family+'_scale_constant']=np.zeros(d,bool)
        scales[family]={'mean':A.aid(arrays[family+'_scale_mean']),
            'std':A.aid(arrays[family+'_scale_std']),'constant_columns':[]}
        arrays[family+'_design_s']=s;arrays[family+'_design_vt']=np.zeros((5,d))
        for mode in ('lambda0.1','lambda1','lambda10','edf_match'):
            name=family+'@'+mode;lam=(factor/anchor_factor)**2 if mode=='edf_match'else float(mode[6:])
            arrays[name+'_dual']=rng.normal(size=(5,70));arrays[name+'_target_mean']=np.zeros(70)
            models[name]={'rank':5,'features':d,'outputs':70,'lambda':lam,
                'total_edf':float(1+sum(s*s/(s*s+lam))),'dual':A.aid(arrays[name+'_dual']),
                'centered_target_mean':A.aid(arrays[name+'_target_mean'])}
    predictions={'CONF_actual':actual,'DEV_actual':devactual}
    for j,name in enumerate(names):
        if name=='zero':p=np.zeros_like(actual);q=np.zeros_like(devactual)
        else:
            # All masks are fixed before rendering, with positive and negative cases.
            amplitude=.45+(j%8)*.25
            p=actual*.25+rng.normal(size=actual.shape)*amplitude
            q=devactual*.8+rng.normal(size=devactual.shape)*(.05+(j%4)*.03)
            p[:24,:,0]*=1.4;p[24:48,:,2]*=.7
        predictions[name]=p;predictions['DEV__'+name]=q
    sigma2=sigma.reshape(10,7)
    pred=np.stack([predictions[n]for n in names]);devpred=np.stack([predictions['DEV__'+n]for n in names])
    qm,qt=A.score(pred,actual,sigma2);dm,dt=A.score(devpred,devactual,sigma2)
    groups=A.aggregate(qm,qt,A.KEYS,('KT','KC'));pairs=A.paired_comparisons(groups,names)
    index={k:i for i,k in enumerate(A.KEYS)};kc=[i for i,k in enumerate(A.KEYS)if k[3]=='KC'];kt=[index[(*A.KEYS[i][:3],'KT')]for i in kc]
    ip=pred[:,kc]-pred[:,kt];ia=actual[kc]-actual[kt]
    predictions['INTERACTION_actual']=ia
    for i,name in enumerate(names):predictions['INTERACTION__'+name]=ip[i]
    ikeys=[(*A.KEYS[i][:3],'KC_minus_KT')for i in kc];im,it=A.score(ip,ia,sigma2)
    ig=A.aggregate(im,it,ikeys,('KC_minus_KT',))
    fit={'target_mean':A.aid(mean),'target_scale':A.aid(sigma),'target':A.aid(devactual),
        'training_provenance':[{'key':list(k)}for k in A.DEV_KEYS],
        'feature_scaling':scales,'models':models,'unavailable':{},
        'edf_anchor':{'model':'current4@lambda1','total_edf':float(anchor)}}
    evaluation={'keys':[list(k)for k in A.KEYS],'model_names':names,'windows':['first5','full10'],
        'scales':['raw','DEV_sigma'],'query_mse':qm,'query_target_mse':qt,'groups':groups,
        'DEV_training_score':{'mse':dm,'target_mse':dt},'paired_comparisons':pairs,
        'paired_interaction':{'keys':[list(k)for k in ikeys],'actual':ia,'query_mse':im,'query_target_mse':it,'groups':ig}}
    result={'status':'SIGNED_TRANSFER_COMPLETE','analysis_complete':True,'input_qualified':True,'synthetic':True,
        'support':{'DEV_shared_KT':48,'CONF_shared_KT':96,'CONF_shared_KC':24,'CONF_tasks':4,
                   'CONF_roots':8,'heads_positions_not_independent_samples':True},
        'input_objects_verified':1500,'new_model_calls':0,'new_env_steps':0,'old192_results_replaced':False,
        'posthoc_CONF_not_new_blind_test':True,'model_names':names,'windows':['first5','full10'],
        'scales':['raw','DEV_sigma'],'unavailable_models':{},'model_diagnostics':models,'edf_anchor':fit['edf_anchor'],
        'DEV_training_mean_MSE':dm.mean(axis=1),'population_scores':[g for g in groups if g['level']=='population'],
        'same_query_interaction':{'population_scores':[g for g in ig if g['level']=='population']},
        'paired_comparisons':pairs,'legacy_reproduction':{'passed':True,'comparisons':1680,
            'maximum_scaled_absolute_error':0.,'tolerance':1e-10,'old_scores_replaced':False,'synthetic':True}}
    return result,evaluation,predictions,arrays,fit


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True);a=parser.parse_args()
    a.out.mkdir(parents=True,exist_ok=False)
    data=A.verify(*make(),synthetic=True)
    import renderer,adapter
    rendered=renderer.render(data,a.out)
    adapter.write(a.out/'SYNTHETIC-QA.json',data['qa'])
    adapter.write(a.out/'SYNTHETIC-RENDER.json',rendered)
    print('SYNTHETIC COMPLETE',a.out,len(rendered['atlas_pages']), 'atlas pages')


if __name__=='__main__':main()
