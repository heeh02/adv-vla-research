"""Consume the experimental PI's exact frame; never select on outcomes."""
import hashlib
import json
from pathlib import Path


def _atoms(value):
    if isinstance(value, str):
        return {value.lower()}
    if isinstance(value, list):
        result = set()
        for item in value:
            result.update(_atoms(item))
        return result
    return set()


def admit_source(source):
    """Narrow runtime-backed exceptions to symbolic enrichment, with warnings kept."""
    proof = {'admitted': False, 'original_status': source.get('status'),
             'original_reason': source.get('reason'),
             'unresolved_references': source.get('unresolved_references', []),
             'basis': None}
    if not source.get('source_ref') or source.get('actual_file_verified') is not True:
        return proof
    if source.get('status') == 'parsed':
        proof.update(admitted=True, basis='original_parser_complete')
        return proof
    if source.get('status') != 'unresolved' or source.get('reason') != 'unknown_symbol_predicate_or_section':
        return proof
    if source.get('unknown_predicates') != [] or source.get('unknown_sections') != []:
        return proof
    declarations = source.get('declarations') or {}
    objects, fixtures, regions = (declarations.get(key) or {} for key in ('objects','fixtures','regions'))
    if not all(isinstance(x, dict) for x in (objects, fixtures, regions)):
        return proof
    if not declarations.get('language') or not declarations.get('goal') or not source.get('semantic_goal_tree_SHA'):
        return proof
    entities = set(objects) | set(fixtures)
    referenced = _atoms(declarations.get('goal')) | _atoms(declarations.get('init'))
    issues = proof['unresolved_references']
    if not isinstance(issues, list) or not issues or not all(isinstance(issue, dict) for issue in issues):
        return proof
    referenced.update(issue.get('symbol') for issue in issues if issue.get('where') == ':obj_of_interest')
    for issue in issues:
        reason, symbol, where = issue.get('reason'), issue.get('symbol'), issue.get('where')
        if reason == 'undeclared_interest' and where == ':obj_of_interest':
            region = regions.get(symbol)
            if not isinstance(region, dict) or region.get('target_symbol') not in entities:
                return proof
        elif reason == 'undeclared_region_parent' and isinstance(where, str) and where.startswith('region:'):
            region_name = where.removeprefix('region:')
            region = regions.get(region_name)
            if not isinstance(region, dict) or region.get('target_symbol') != symbol:
                return proof
            if region_name in referenced or symbol in referenced:
                return proof
        else:
            return proof
    proof.update(admitted=True, basis='declared_interest_region_or_unreferenced_orphan_region_only')
    return proof


def read_frame(args, shard, C, enrich, output, classification):
    raw = Path(args.frame).read_bytes()
    C.need(hashlib.sha256(raw).hexdigest() == shard['frame_sha256'], 'frozen frame identity')
    frame = json.loads(raw)
    mraw = Path(args.runtime_file_manifest).read_bytes()
    expected = frame['runtime_file_manifest']
    C.need(len(mraw) == expected['bytes'] and hashlib.sha256(mraw).hexdigest() == expected['sha256'], 'original runtime member manifest identity')
    manifest = {}
    for line in mraw.splitlines():
        if not line.strip(): continue
        row = json.loads(line); C.need(row['path'] not in manifest, 'unique runtime members'); manifest[row['path']] = row
    # SourceCache paths include official/...; use the already restored parent.
    cache = enrich.SourceCache(Path(args.plus_root).resolve().parent, manifest, manifest_identity=expected)
    rows = enrich.enrich_rows(frame, cache)
    enrich.write_results(output / 'frame-enrichment', frame, enrich.identity(raw), cache, rows,
                         scope='same_worker_original_runtime_all_frame_once_no_network')
    idx, parsed = {r['row_id']: r for r in frame['rows']}, {r['row_id']: r for r in rows}
    chosen = shard['case_rows']
    C.need(len(chosen) == len(set(chosen)) and all(x in idx for x in chosen), 'frozen unique selected frame IDs')
    result = []
    for row_id in chosen:
        row, enriched = idx[row_id], parsed[row_id]
        source = cache.get(row['resource_routes']['clean_BDDL']['member'])
        variant_source = cache.get(row['resource_routes']['variant_BDDL']['member'])
        admission = {'clean': admit_source(source), 'variant': admit_source(variant_source)}
        source_ref = source.get('source_ref')
        prompt = source.get('declarations', {}).get('language')
        # Unresolved candidates remain exact planned rows and fail closed later.
        eligible = admission['clean']['admitted'] and admission['variant']['admitted']
        goal_id = 'goal:' + source['semantic_goal_tree_SHA'] if eligible else None
        contract = None
        if eligible:
            contract = {'id': row_id + ':actual-base-source', 'goal_binding_id': goal_id,
                        'base_bddl': {'relative_to_plus_root': source_ref['member'].removeprefix('official/'),
                                      'bytes': source_ref['bytes'], 'sha256': source_ref['sha256']}}
        # C.make_case independently validates the official registry identity.
        case = C.make_case(row['suite'], row['official_task_id_zero_based'], classification)
        C.need(case['classification_id'] == row['official_classification_id_one_based'] and case['task_name'] == row['official_variant_name'], 'frame official identity')
        repeats = shard['noise_repeats']
        C.need(repeats == [0, 1], 'frozen_two_noise_repeats')
        for repeat in repeats:
            identity = {'population_id': shard['population_id'], 'suite': row['suite'], 'family': row['official_category'],
                        'task': row['canonical_parent_task_candidate']['family_id'], 'case_id': row_id + ':n' + str(repeat),
                        'pair_id': row_id + ':r' + str(repeat)}
            result.append({'identity': identity, 'official_case': case, 'goal_binding_id': goal_id,
                       'reference_contract': contract, 'matched_prompt': prompt, 'semantic_enrichment': enriched,
                       'source_admission_v4': admission,
                       'exposure': row['exposure'], 'difficulty': row['official_difficulty_level'],
                       'task_identity_status': 'canonical_parent_candidate_until_runtime_goal_entity_checks',
                       'noise': {'parent_family_id': row['canonical_parent_task_candidate']['family_id'], 'noise_label': repeat,
                                 'protocol': 'plus-shared-policy-noise-v1'},
                       'full_pp': row_id in shard.get('full_pp_case_rows', []) and repeat == 0, 'collect_reference_Y': True, 'late50': False})
    return result, manifest
