"""Consume the experimental PI's exact frame; never select on outcomes."""
import hashlib
import json
from pathlib import Path


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
        source_ref = source.get('source_ref')
        prompt = source.get('declarations', {}).get('language')
        # Unresolved candidates remain exact planned rows and fail closed later.
        goal_id = 'goal:' + source.get('semantic_goal_tree_SHA', 'unresolved') if source['status'] == 'parsed' else None
        contract = None
        if source_ref and source['status'] == 'parsed':
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
                       'exposure': row['exposure'], 'difficulty': row['official_difficulty_level'],
                       'task_identity_status': 'canonical_parent_candidate_until_runtime_goal_entity_checks',
                       'noise': {'parent_family_id': row['canonical_parent_task_candidate']['family_id'], 'noise_label': repeat,
                                 'protocol': 'plus-shared-policy-noise-v1'},
                       'full_pp': row_id in shard.get('full_pp_case_rows', []) and repeat == 0, 'collect_reference_Y': True, 'late50': False})
    return result, manifest
