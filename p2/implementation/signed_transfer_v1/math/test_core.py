import json
import unittest
import numpy as np
import core as C


def fixture(n=24):
    rng = np.random.default_rng(83)
    context = rng.normal(size=(n, 9))
    features = {name: np.c_[context, rng.normal(size=(n, 4))] for name in C.FAMILIES}
    features['c9'] = context
    frozen = {'feature_mean': np.arange(13) / 20, 'feature_scale': np.arange(13) / 10 + 1,
              'target_mean': np.zeros(70), 'target_scale': np.arange(70) / 70 + 1}
    y = rng.normal(size=(n, 10, 7))
    return features, y, np.full(n, 1 / n), frozen


class Tests(unittest.TestCase):
    def test_weighted_svd_equals_independent_normal_equations(self):
        rng = np.random.default_rng(6)
        x, y = rng.normal(size=(23, 7)), rng.normal(size=(23, 3))
        w = np.arange(1, 24, dtype=float); w /= w.sum()
        lam = .7; d = C.RidgeDesign(x, w); model = d.fit(y, lam)
        xc, yc = x - w @ x, y - w @ y
        gram = xc.T @ (w[:, None] * xc)
        beta = np.linalg.solve(gram + lam * np.eye(7), xc.T @ (w[:, None] * yc))
        np.testing.assert_allclose(model.predict(x), xc @ beta + w @ y, rtol=2e-13, atol=2e-13)
        expected_df = 1 + np.trace(np.linalg.solve(gram + lam * np.eye(7), gram))
        self.assertAlmostEqual(d.edf(lam), expected_df, places=12)

    def test_unpenalized_intercept_constant_columns_and_rank(self):
        x = np.full((12, 20), .3); y = np.full((12, 4), 19.5)
        d = C.RidgeDesign(x, np.full(12, 1 / 12))
        self.assertEqual(d.rank, 0)
        np.testing.assert_allclose(d.fit(y, 1e8).predict(x), y)
        self.assertEqual(d.edf(.1), 1)
        self.assertEqual(d.match_edf(2)['status'], 'UNATTAINABLE')
        self.assertEqual(d.match_edf(1)['status'], 'MATCHED')

    def test_edf_matches_capacity_but_not_feature_count(self):
        rng = np.random.default_rng(3)
        a = rng.normal(size=(30, 4)); b = rng.normal(size=(30, 100))
        w = np.full(30, 1 / 30)
        da, db = C.RidgeDesign(a, w), C.RidgeDesign(b, w)
        target = da.edf(1)
        match = db.match_edf(target)
        self.assertEqual(match['status'], 'MATCHED')
        self.assertAlmostEqual(match['actual_edf'], target, places=9)
        self.assertNotEqual(db.edf(1), target)
        low = C.RidgeDesign(a[:, :1], w)
        self.assertEqual(low.match_edf(target)['status'], 'UNATTAINABLE')

    def test_task_root_weights_do_not_reward_more_queries(self):
        w = C.task_root_weights(['A', 'A', 'A', 'B'], ['a1', 'a1', 'a2', 'b1'])
        np.testing.assert_allclose(w, [.125, .125, .25, .5])
        with self.assertRaises(ValueError): C.task_root_weights(['A', 'B'], ['same', 'same'])

    def test_original_c9_scaling_extra_dev_scaling_and_constant_drop(self):
        x = np.c_[np.arange(54).reshape(6, 9), np.arange(6), np.full(6, 100.)]
        w = np.full(6, 1/6)
        scale = C.fit_scale(x, w, np.full(9, 3.), np.full(9, 2.))
        got = scale.apply(x)
        np.testing.assert_array_equal(got[:, :9], (x[:, :9] - 3) / 2)
        self.assertAlmostEqual(float(w @ got[:, 9]), 0, places=12)
        self.assertAlmostEqual(float(w @ (got[:, 9] ** 2)), 1, places=12)
        future = x.copy(); future[:, 10] = -10000
        np.testing.assert_array_equal(scale.apply(future)[:, 10], 0)
        self.assertEqual(scale.manifest()['constant_columns'], [10])

    def test_CONF_label_changes_cannot_change_fit_scale_lambda_or_predictions(self):
        f, y, w, frozen = fixture()
        suite = C.fit_suite(f, y, w, frozen)
        before = json.dumps(suite.manifest(), sort_keys=True)
        predicted = suite.predict(f)
        # Evaluation targets are never arguments to fitting, scaling or EDF.
        score1 = C.score_predictions(predicted, y, frozen['target_scale'])
        score2 = C.score_predictions(predicted, -1000 * y, frozen['target_scale'])
        self.assertFalse(np.array_equal(score1['mse'], score2['mse']))
        self.assertEqual(before, json.dumps(suite.manifest(), sort_keys=True))
        again = suite.predict(f)
        for name in predicted: np.testing.assert_array_equal(predicted[name], again[name])
        self.assertEqual(suite.manifest()['fixed_lambdas'], [.1, 1., 10.])
        self.assertFalse(suite.manifest()['CONF_used_to_fit_scale_lambda_or_model'])

    def test_signed_head_position_cancellation_norm_and_sum_counterexample(self):
        # Two patterns have equal norm80 and signed sum256, but opposite labels.
        amplitude = np.array([-2., -1., 1., 2.])
        full = np.c_[amplitude, -amplitude]
        norms = np.abs(full); collapsed = full.sum(1, keepdims=True)
        y = amplitude[:, None]; w = np.full(4, .25)
        signed = C.RidgeDesign(full, w).fit(y, .1).predict(full)
        norm = C.RidgeDesign(norms, w).fit(y, .1).predict(norms)
        summed = C.RidgeDesign(collapsed, w).fit(y, .1).predict(collapsed)
        self.assertLess(np.mean((signed-y)**2), .002)
        np.testing.assert_allclose(norm, 0, atol=1e-14)
        np.testing.assert_array_equal(summed, 0)

    def test_no_transfer_is_possible_even_with_complete_signed_features(self):
        # A task reverses the mapping. Failure is not proof X lacks information.
        x = np.arange(-10., 11.)[:, None]; w = np.full(len(x), 1/len(x))
        model = C.RidgeDesign(x, w).fit(x, .1)
        pred = model.predict(x)
        self.assertLess(np.mean((pred-x)**2), .001)
        self.assertGreater(np.mean((pred+x)**2), np.mean(x**2))

    def test_target_scoring_keeps_windows_dimensions_and_zero_denominators(self):
        target = np.zeros((2, 10, 7)); pred = target.copy(); pred[:, :5, 0] = 2
        sd = np.ones(70); sd.reshape(10,7)[:5,0] = 2
        score = C.score_predictions({'test':pred}, target, sd)
        self.assertEqual(score['mse'].shape, (1,2,2,2,7))
        np.testing.assert_array_equal(score['target_mse'], 0)
        self.assertEqual(score['mse'][0,0,0,0,0], 4)
        self.assertEqual(score['mse'][0,0,1,0,0], 2)
        self.assertEqual(score['mse'][0,0,0,1,0], 1)
        np.testing.assert_array_equal(score['mse'][0,:,:,:,1:], 0)

    def test_zero_edit_override_and_production_DEV_rejects_CONF(self):
        f, y, w, frozen = fixture()
        suite = C.fit_suite(f,y,w,frozen)
        pred = suite.predict(f, zero_actual_E=np.ones(len(y),bool))
        for p in pred.values(): np.testing.assert_array_equal(p, 0)
        with self.assertRaisesRegex(ValueError, 'exact_DEV48'):
            C.fit_rows([{'key': ('fake', 11, 'shared', 'KT'), 'split':'CONF'}], frozen)

    def test_fixed_production_shapes_complete_support_and_paired_errors(self):
        def row(key, split):
            x = (key[1] - 16) / 5
            context = np.zeros(9); context[0] = x
            write = np.zeros((10,1024)); write[0,0] = x; write[1,0] = -x
            head = np.zeros((8,10,256)); head[0,0,0] = x; head[1,0,0] = -x
            extra = np.array([x*x, np.sin(x), np.cos(x), 0])
            target = np.zeros((10,7)); target[:,0] = x
            return dict(key=key, split=split, context9=context, current4=extra,
                pooled4=extra*.5, originalnorm4=np.abs(extra), current_gated_write=write,
                pooled_gated_write=write*.5, current_headpos_read=head,
                current_headpos_norms=np.linalg.norm(head,axis=-1).ravel(),
                effect=target, zero_actual_E=False, provenance={'synthetic':True})
        frozen = {'feature_mean':np.zeros(13), 'feature_scale':np.ones(13),
                  'target_mean':np.zeros(70), 'target_scale':np.ones(70)}
        dev = [row(k,'DEV') for k in C.DEV_KEYS]
        conf = [row(k,'CONF') for k in C.CONF_KEYS]
        suite = C.fit_rows(dev,frozen)
        result = C.evaluate_rows(suite,conf)
        self.assertEqual(len(result['groups']),36)
        self.assertEqual(len(result['keys']),120)
        self.assertEqual(len(result['paired_interaction']['keys']),24)
        self.assertEqual(len(result['paired_interaction']['groups']),18)
        np.testing.assert_array_equal(result['paired_interaction']['actual'],0)
        for value in result['paired_interaction']['predictions'].values():np.testing.assert_array_equal(value,0)
        self.assertEqual(suite.models['current_full@lambda1'].design.x.shape,(48,10249))
        self.assertEqual(suite.models['headpos_signed@lambda1'].design.x.shape,(48,20489))
        self.assertEqual(suite.models['headpos_signed_sum@lambda1'].design.x.shape,(48,265))
        zero_index = result['model_names'].index('zero')
        np.testing.assert_array_equal(result['query_mse'][zero_index],result['query_target_mse'])
        for group in result['groups']:
            self.assertTrue(np.isnan(group['relative_to_zero'][:,:,:,1:]).all())
            np.testing.assert_array_equal(group['ratio_defined'][:,:,1:],False)
        before=json.dumps(suite.manifest(),sort_keys=True)
        predictions=suite.predict_rows(conf)
        for item in conf: item['effect'][:]=10000
        for name,pred in suite.predict_rows(conf).items():np.testing.assert_array_equal(pred,predictions[name])
        self.assertEqual(before,json.dumps(suite.manifest(),sort_keys=True))
        bad=[dict(r) for r in conf]
        kc=next(r for r in bad if r['key'][3]=='KC')
        kc['context9']=kc['context9'].copy();kc['context9'][0]+=1
        with self.assertRaisesRegex(ValueError,'paired_own_K'):
            C.evaluate_rows(suite,bad)
        with self.assertRaisesRegex(ValueError,'complete_CONF'):
            C.evaluate_rows(suite,conf[:-1])


if __name__ == '__main__': unittest.main()
