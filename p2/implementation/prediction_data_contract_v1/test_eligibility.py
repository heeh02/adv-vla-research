import copy
import unittest
from eligibility import check, check_split, REQUIRED_T_FIELDS, ORIGINS


class ScientificBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.case = {'roles': ['baseline_none_1', 'shared_pulse', 'random_pulse'],
                     'ownN_pairing_verified': True, 'fresh_C_AE_observed': False,
                     'features': [{'field': f, 'origin': ORIGINS[f]} for f in sorted(REQUIRED_T_FIELDS)],
                     'label_exposure': 'already_analyzed', 'initialization_cluster': 'task/init0'}

    def result(self, **kw):
        return check(self.case, target=kw.get('target', 'shared_minus_random_pulse'),
                     information_set=kw.get('information_set', 'T_only_shadow'),
                     cutoff=kw.get('cutoff', 'after_unedited_j5_before_edited_output'),
                     prospective=kw.get('prospective', False))

    def test_allowed_retrospective_does_not_authorize_compute(self):
        r = self.result()
        self.assertTrue(r['eligible_for_requested_scientific_use'])
        self.assertFalse(r['execution_authorized'])

    def test_persistent_cannot_fill_missing_random_pulse(self):
        self.case['roles'] = ['baseline_none_1', 'shared_pulse', 'shared_persistent']
        self.assertFalse(self.result()['eligible_for_requested_scientific_use'])
        self.assertTrue(self.result(target='shared_pulse')['eligible_for_requested_scientific_use'])

    def test_edited_native_write_excluded_by_unedited_shadow_contract(self):
        self.case['features'].append({'field': 'native_delta_H', 'origin': 'edited_S'})
        self.assertFalse(self.result()['eligible_for_requested_scientific_use'])

    def test_future_natural_endpoint_is_also_leakage(self):
        self.case['features'].append({'field': 'x10', 'origin': 'ownN'})
        self.assertFalse(self.result()['eligible_for_requested_scientific_use'])

    def test_v5_not_available_at_pre_setter_cutoff(self):
        self.assertFalse(self.result(cutoff='single_forward_before_L9_setter')['eligible_for_requested_scientific_use'])

    def test_parent_C_does_not_supply_fresh_C_AE(self):
        self.case['parent_branch'] = 'PC'
        self.assertFalse(self.result(information_set='fresh_CT_shadow')['eligible_for_requested_scientific_use'])

    def test_hybrid_old_C_is_not_same_graph(self):
        self.case['features'].append({'field': 'v0', 'origin': 'old_G_C'})
        self.assertFalse(self.result()['eligible_for_requested_scientific_use'])

    def test_repartitioning_does_not_make_exposed_labels_prospective(self):
        self.case['split_name'] = 'brand_new_test'
        self.assertFalse(self.result(prospective=True)['eligible_for_requested_scientific_use'])

    def test_C_T_parents_share_initialization_fold(self):
        other = copy.deepcopy(self.case)
        other['parent_branch'] = 'PT'
        self.case['parent_branch'] = 'PC'
        self.assertFalse(check_split([self.case], [other])['group_disjoint'])

    def test_unknown_pairing_or_exposure_is_not_qualified(self):
        self.case['ownN_pairing_verified'] = None
        self.case['label_exposure'] = 'unknown'
        self.assertFalse(self.result()['eligible_for_requested_scientific_use'])
        self.assertFalse(self.result(prospective=True)['eligible_for_requested_scientific_use'])

    def test_empty_features_cannot_be_qualified(self):
        self.case['features'] = []
        self.assertFalse(self.result()['eligible_for_requested_scientific_use'])

    def test_payload_tag_cannot_launder_attention_origin(self):
        next(f for f in self.case['features'] if f['field'] == 'P_j5')['origin'] = 'known_predeclared_payload'
        self.assertFalse(self.result()['eligible_for_requested_scientific_use'])

    def test_CT_requires_actual_C_feature_roster_not_only_boolean(self):
        self.case['fresh_C_AE_observed'] = True
        self.assertFalse(self.result(information_set='fresh_CT_shadow')['eligible_for_requested_scientific_use'])


if __name__ == '__main__':
    unittest.main()
