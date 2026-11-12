"""Metadata-only eligibility for an unedited-shadow feature contract.

This does not adjudicate a different post-edit/pre-endpoint prediction problem.
Such a design needs its own cutoff, features and matched baselines. No execution
is authorized here; metadata booleans do not replace source/body validation.
"""

BASELINE_FIELDS = {'x0', 'v0', 'x5', 'v5', 'P_j5', 'gate_j5', 'receiver_input_j5',
                   'mask', 'positions', 'state', 'actual_E', 'WO', 'query_clock'}
REQUIRED_T_FIELDS = BASELINE_FIELDS - {'receiver_input_j5'}
C_FIELDS = {'C_x0', 'C_v0', 'C_x5', 'C_v5'}
ORIGINS = {name: 'ownN' for name in BASELINE_FIELDS}
ORIGINS.update({'state': 'original_input', 'query_clock': 'original_input',
                'actual_E': 'known_predeclared_payload', 'WO': 'known_predeclared_payload'})
ORIGINS.update({name: 'fresh_C_ownN' for name in C_FIELDS})


def check(case, *, target, information_set, cutoff, prospective=False):
    errors = []
    if target not in {'shared_pulse', 'random_pulse', 'shared_minus_random_pulse'}:
        errors.append('unsupported_target_requires_separate_contract')
    required = {'baseline_none_1', 'shared_pulse'}
    if target == 'random_pulse':
        required = {'baseline_none_1', 'random_pulse'}
    elif target == 'shared_minus_random_pulse':
        required.add('random_pulse')
    missing = sorted(required - set(case['roles']))
    if missing:
        errors.append('missing_roles:' + ','.join(missing))
    if case.get('ownN_pairing_verified') is not True:
        errors.append('ownN_pairing_unproven')
    if information_set not in {'T_only_shadow', 'fresh_CT_shadow'}:
        errors.append('information_set_not_declared')
    if information_set == 'fresh_CT_shadow' and case.get('fresh_C_AE_observed') is not True:
        errors.append('fresh_C_AE_not_observed')
    if cutoff not in {'after_unedited_j5_before_edited_output', 'single_forward_before_L9_setter'}:
        errors.append('cutoff_not_declared')
    names = [f['field'] for f in case['features']]
    required_fields = set(REQUIRED_T_FIELDS)
    if information_set == 'fresh_CT_shadow':
        required_fields |= C_FIELDS
    if cutoff == 'single_forward_before_L9_setter':
        required_fields -= {'v5', 'C_v5'}
    if required_fields - set(names):
        errors.append('missing_declared_feature_fields:' + ','.join(sorted(required_fields - set(names))))
    if len(names) != len(set(names)):
        errors.append('duplicate_feature_fields')
    for feature in case['features']:
        if feature['field'] not in ORIGINS:
            errors.append('field_after_cutoff_or_not_whitelisted:' + feature['field'])
        elif feature['origin'] != ORIGINS[feature['field']]:
            errors.append('wrong_or_posttreatment_field_origin:' + feature['field'])
        if feature['field'] in C_FIELDS and information_set == 'T_only_shadow':
            errors.append('C_field_not_in_T_only_information_set')
        if feature['field'] in {'v5', 'C_v5'} and cutoff == 'single_forward_before_L9_setter':
            errors.append('v5_requires_completed_unedited_shadow_call')
    if prospective:
        if case.get('label_exposure') != 'unrevealed':
            errors.append('labels_already_exposed_or_unknown')
        if case.get('preregistration_before_label_exposure_verified') is not True:
            errors.append('no_verified_prospective_freeze')
        if case.get('independent_new_unit_verified') is not True:
            errors.append('no_verified_new_independent_unit')
    return {'eligible_for_requested_scientific_use': not errors, 'reasons': errors,
            'metadata_only': True, 'fresh_body_validation_required': True,
            'execution_authorized': False}


def check_split(train, test):
    """Keep same-initialization parents, queries and arms together."""
    left = {r['initialization_cluster'] for r in train}
    right = {r['initialization_cluster'] for r in test}
    overlap = sorted(left & right)
    return {'group_disjoint': not overlap, 'overlapping_clusters': overlap,
            'new_prospective_evidence': False}
