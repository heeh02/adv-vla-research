import copy
import unittest
import numpy as np
import core as C


def fixture():
    rng=np.random.default_rng(872);wo=rng.normal(size=(2,3,5));queries=[];groups=[]
    output=rng.normal(size=(3,70))
    for ri,root in enumerate(C.ROOTS):
        task='T'+str(ri//2);cluster='verified-init-'+str(ri)
        groups.append(dict(root_id=root,cluster_id=cluster,task_id=task,
            library_sha256=str(ri//2)*64,init_index=ri%2,init_state_sha256=str(ri+2)*64,source_ref={'synthetic':True}))
        for qidx in (11,12,13):
            state=rng.normal(size=8);delta=rng.normal(size=6);p=rng.uniform(.01,1,size=(2,10,13));p/=p.sum(-1,keepdims=True)
            edits={};targets={}
            for ai,arm in enumerate(C.ARMS):
                e=rng.normal(size=(3,3))*(ai+1)/4;edits[arm]=e
                latent=np.array([state[0]+e.sum(),state[1]+e[0,0],delta[0]+ai])
                targets[arm]=(latent@output).reshape(10,7)
            queries.append(dict(root_id=root,cluster_id=cluster,task_id=task,query_index=qidx,
                source_identity='a'*64,state_vector=state,first_flow_delta=delta,time_scalar=5*qidx/520,
                p=p,mask=np.ones((10,13),bool),gate=rng.uniform(.1,1,size=5),edits=edits,targets=targets,
                provenance={'synthetic':True},split='DEV'))
    return dict(schedule='persistent_j5_to_j9',arms=list(C.ARMS),wo=wo,groups=groups,queries=queries,
                independence={'status':'SYNTHETIC_FOUR_DISTINCT_GROUPS','iid_claim':False})


class Tests(unittest.TestCase):
    def test_weighted_kernel_ridge_matches_unpenalized_arm_primal(self):
        rng=np.random.default_rng(6);x=rng.normal(size=(24,6));y=rng.normal(size=(24,3));arms=np.arange(24)%4;z=np.eye(4)[arms]
        w=np.arange(1,25,dtype=float);w/=w.sum();k=C.Kernel(x,np.arange(24),w,z)
        pred,info,_=k.predict(y,.1)
        xm=np.stack([np.average(x[arms==a],axis=0,weights=w[arms==a])for a in range(4)])
        ym=np.stack([np.average(y[arms==a],axis=0,weights=w[arms==a])for a in range(4)])
        xc=x-xm[arms];yc=y-ym[arms]
        beta=np.linalg.solve(xc.T@(w[:,None]*xc)+.1*np.eye(6),xc.T@(w[:,None]*yc))
        expected=(x-xm[arms])@beta+ym[arms]
        np.testing.assert_allclose(pred,expected,rtol=1e-10,atol=1e-10)
        self.assertGreaterEqual(info['total_EDF'],4)

    def test_block_RMS_is_orthogonal_but_coordinate_scaling_is_not(self):
        rng=np.random.default_rng(5);x=rng.normal(size=(25,5))*np.array([1,2,5,9,12]);y=rng.normal(size=(25,2));w=np.full(20,.05)
        u=np.arange(1.,6.);u/=np.linalg.norm(u);rot=x-2*(x@u)[:,None]*u
        z=np.ones((25,1));idx=np.arange(20)
        a=C.Block(x[idx],w).apply(x);b=C.Block(rot[idx],w).apply(rot)
        for rbf in (False,True):
            pa=C.Kernel(a,idx,w,z,rbf).predict(y[idx],1.)[0]
            pb=C.Kernel(b,idx,w,z,rbf).predict(y[idx],1.)[0]
            np.testing.assert_allclose(pa,pb,atol=1e-11,rtol=1e-11)
        a=(x-x[idx].mean(0))/x[idx].std(0);b=(rot-rot[idx].mean(0))/rot[idx].std(0)
        pa=C.Kernel(a,idx,w,z).predict(y[idx],1.)[0];pb=C.Kernel(b,idx,w,z).predict(y[idx],1.)[0]
        self.assertGreater(np.max(np.abs(pa-pb)),1e-3)

    def test_pooled_write_retains_suffix_denominator(self):
        p=np.full((1,10,12),1/12);e=np.ones((2,1));wo=np.ones((1,1,1));gate=np.ones(1)
        value=C.local_write(p,e,wo,gate)
        np.testing.assert_allclose(value,1/6)
        self.assertFalse(np.allclose(value,1))

    def test_training_pool_ignores_heldout_P_and_bandwidth_ignores_test(self):
        d=fixture();data=C.Dataset(d,lambda:None,True);train=data.clusters[:3];fold=data.fold(train)
        d2=copy.deepcopy(d)
        for q in d2['queries']:
            if q['cluster_id']==data.clusters[3]:
                q['p'][:]=1/q['p'].shape[-1]
        other=C.Dataset(d2,lambda:None,True).fold(train)
        np.testing.assert_array_equal(fold.pool,other.pool)
        rng=np.random.default_rng(2);x=rng.normal(size=(10,3));z=np.ones((10,1));idx=np.arange(8);w=np.full(8,1/8)
        a=C.Kernel(x,idx,w,z,True);x[8:]*=10000;b=C.Kernel(x,idx,w,z,True)
        self.assertEqual(a.bandwidth,b.bandwidth)
        with self.assertRaisesRegex(ValueError,'UNAVAILABLE'):
            C.Kernel(np.zeros((10,3)),idx,w,z,True)

    def test_heldout_dose_does_not_change_training_preprocessing(self):
        raw=fixture();first=C.Dataset(raw,lambda:None,True);training=first.clusters[:3];a=first.fold(training)
        changed=copy.deepcopy(raw)
        for q in changed['queries']:
            if q['cluster_id']==first.clusters[3]:
                for arm in C.ARMS:q['edits'][arm]*=50
        b=C.Dataset(changed,lambda:None,True).fold(training)
        self.assertEqual(a.manifest(),b.manifest())
        self.assertEqual(a.context_width,10)
        self.assertEqual(a.x['context'].shape[1],40)

    def test_nonfinite_model_outputs_fail_and_only_sigma_zero_is_undefined(self):
        with self.assertRaisesRegex(ValueError,'infinite_output'):
            C.serial(float('inf'))
        x=np.array([[1.,2.],[3.,4.],[2.,1.],[4.,3.]])
        k=C.Kernel(x,np.arange(4),np.full(4,.25),np.ones((4,1)))
        with self.assertRaisesRegex(ValueError,'finite_kernel_predictions'):
            k.predict(np.full((4,2),np.nan),1.)
        result=C.losses(np.ones((2,10,7)),np.zeros((2,10,7)),np.zeros(70))
        self.assertTrue(np.isfinite(result[:,:,0]).all());self.assertTrue(np.isnan(result[:,:,1]).all())

    def test_arm_mean_strong_contrast_and_root_balancing(self):
        data=C.Dataset(fixture(),lambda:None,True);fold=data.fold(data.clusters[:3]);pred,_=fold.predict('arm_mean')
        self.assertGreater(np.max(np.abs(pred[0]-pred[1])),.1)
        w=C.group_weights(['A','A','A','B'],['a','a','b','c'],['q0','q1','q0','q0'],['s']*4)
        np.testing.assert_allclose(w,[.125,.125,.25,.5])

    def test_endpoint_and_contrast_selection_rankings_can_differ(self):
        d=C.Dataset(fixture(),lambda:None,True);fold=d.fold(d.clusters[:3]);ii=d.row_indices([d.clusters[3]])
        common=d.y.copy()+10
        opposite=d.y.copy();opposite[d.armindex==0]+=1;opposite[d.armindex==1]-=1
        self.assertGreater(C.tune_score(fold,common,ii),C.tune_score(fold,opposite,ii))
        self.assertLess(C.tune_contrast_score(fold,common,ii),C.tune_contrast_score(fold,opposite,ii))

    def test_nested_outer_labels_never_change_transform_selection_or_predictions(self):
        raw=fixture();d=C.Dataset(raw,lambda:None,True);held=d.clusters[0];train=d.clusters[1:]
        first,best,selection=C.nested_choices(d,list(train));f=d.fold(train);p=f.predict('current_full',first['current_full'])[0]
        modified=copy.deepcopy(raw)
        for q in modified['queries']:
            if q['cluster_id']==held:
                for a in C.ARMS:q['targets'][a]*=-1000
        other=C.Dataset(modified,lambda:None,True);second,best2,selection2=C.nested_choices(other,list(train));g=other.fold(train)
        self.assertEqual(first,second);self.assertEqual(best,best2);self.assertEqual(selection,selection2)
        self.assertEqual(f.manifest(),g.manifest())
        np.testing.assert_array_equal(p,g.predict('current_full',second['current_full'])[0])
        for r in selection['inner_folds']:
            self.assertNotIn(held,r['training_clusters']);self.assertNotIn(r['validation_cluster'],r['training_clusters'])

    def test_missing_arms_conf_or_duplicate_init_rejected(self):
        d=fixture();del d['queries'][0]['targets']['random_matched']
        with self.assertRaisesRegex(ValueError,'complete_declared_arms'):C.Dataset(d,lambda:None,True)
        d=fixture();d['queries'][0]['split']='CONF'
        with self.assertRaisesRegex(ValueError,'DEV_only'):C.Dataset(d,lambda:None,True)
        d=fixture();d['groups'][1]['init_state_sha256']=d['groups'][0]['init_state_sha256']
        with self.assertRaisesRegex(ValueError,'distinct_typed'):C.Dataset(d,lambda:None,True)

    def test_complete_run_contrast_curve_precision_and_zero_override(self):
        d=fixture();d['queries'][0]['edits']['shared'][:]=0;d['queries'][0]['targets']['shared'][:]=0
        result=C.run(d,synthetic=True);s=result['summary'];arrays=result['arrays']
        self.assertTrue(s['analysis_complete']);self.assertEqual(s['known_zero_edits'],1)
        for k,v in arrays.items():
            if k.startswith('prediction__'):np.testing.assert_array_equal(v[0],0)
        self.assertEqual(s['paired_persistent_contrast']['snapshots'],12)
        self.assertFalse(s['paired_persistent_contrast']['pulse_result'])
        self.assertIsNone(s['precision_sensitivity']['pulse_variance'])
        self.assertIsNone(s['root_learning_curve']['four_cluster_OOF_point'])
        for r in s['folds']:
            a=r['learning_curve_training']['two_balanced'];b=r['learning_curve_training']['three_all_available']
            self.assertEqual(len(a),2);self.assertEqual(len(b),3);self.assertTrue(set(a)<set(b));self.assertNotIn(r['held_out_cluster'],b)
        self.assertTrue(all(r['passed']for r in s['orthogonal_QA']))
        scenarios=s['precision_sensitivity']['comparisons'][0]['scenarios']
        self.assertEqual({r['SD_multiplier']for r in scenarios},{1.,2.,4.})
        self.assertEqual({r['roots_per_task']for r in scenarios},{2,4,8,16,32})
        np.testing.assert_array_equal(arrays['contrast_actual'][0],-d['queries'][0]['targets']['random_matched'])


if __name__=='__main__':unittest.main()
