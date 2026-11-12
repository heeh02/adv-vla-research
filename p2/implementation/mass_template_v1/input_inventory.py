"""Metadata-only input inventory. No arrays, cloud SDK, or network calls."""
from pathlib import Path
import argparse
import gzip
import hashlib
import json
import tarfile

HERE = Path(__file__).resolve().parent
REL = Path('docs/experiments/state-dependent-response-use-20261013-v1/continuous/task-heldout-confirmation-v1')
FIELDS = ('raw_gram_upper', 'masses', 'corner_defined', 'projection_gram_upper', 'group_corner_defined')


def identity(raw):
    return {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


def build(research_root):
    h = Path(research_root) / REL
    old = h / 'root/natural-reading-use-cpu-v1'
    rb = old / 'run-v1/readback'
    result_raw = (rb / 'summary/RESULT-v1.json').read_bytes()
    result = json.loads(result_raw)
    delivery_raw = (rb / 'summary/DELIVERY-v1.json').read_bytes()
    delivery = json.loads(delivery_raw)
    assert identity(result_raw) == {k: delivery['result'][k] for k in ('bytes', 'sha256')}
    assert delivery['scientific_complete'] is result['scientific_complete'] is True
    cloud = {r['relative']: r for r in result['cloud_objects']}
    metadata = {}
    for local_name, relative in (
        ('PARENT-RESULT.json', 'summary/RESULT-v1.json'),
        ('PARENT-MANIFEST.json', 'science/MANIFEST-v1.json'),
        ('PARENT-SOURCE.json', 'science/analysis/SOURCE-v1.json'),
        ('PARENT-SUMMARY.json', 'science/analysis/SUMMARY-v1.json'),
    ):
        path = rb / relative
        raw = path.read_bytes()
        ref = delivery['result'] if relative.startswith('summary/') else cloud[relative]
        assert identity(raw) == {k: ref[k] for k in ('bytes', 'sha256')}
        metadata[local_name] = {'local_source_path': str(path), 'parent_relative': relative,
                                **{k: ref[k] for k in ('uri', 'bytes', 'sha256')}}
    source = json.loads((rb / 'science/analysis/SOURCE-v1.json').read_bytes())
    manifest = json.loads((rb / 'science/MANIFEST-v1.json').read_bytes())
    entries = {r['path']: r for r in manifest['files']}
    queries = []
    for query in source['queries']:
        ref = query['array_file']
        rel = 'science/analysis/' + ref['path']
        assert all(ref[k] == cloud[rel][k] == entries['analysis/' + ref['path']][k]
                   for k in ('bytes', 'sha256'))
        queries.append({k: query[k] for k in ('root_id', 'task_id', 'query_index', 'source_identity')} |
                       {'path': ref['path'], **{k: cloud[rel][k] for k in ('uri', 'bytes', 'sha256')}})
    assert len(queries) == 24 and len({(q['root_id'], q['query_index']) for q in queries}) == 24
    raw_per_query = {sum(q['arrays'][f]['bytes'] for f in FIELDS) for q in source['queries']}
    assert len(raw_per_query) == 1
    descriptor = old / 'prepared-v1/SOURCE_PACKAGE-v1.json'
    producer = json.loads(descriptor.read_bytes())
    package = h / 'root/natural-control-cpu-v3/prepared-v1/SOURCE_PACKAGE-v1.tar.gz'
    with tarfile.open(package) as archive:
        readraw = gzip.decompress(archive.extractfile('metadata/READSET-v1.json.gz').read())
        cdev = json.loads(readraw)
    wpath = h / 'model-aligned-full-response-view-v1/frozen-v-transfer-v1/protocol/V-PROJECTION-WEIGHT-SOURCE-v1.json'
    weights = json.loads(wpath.read_bytes())
    stats = {'query_NPZ_objects': 24, 'query_NPZ_bytes': sum(q['bytes'] for q in queries),
             'metadata_objects': len(metadata), 'metadata_bytes': sum(r['bytes'] for r in metadata.values()),
             'selected_decoded_bytes_per_query': raw_per_query.pop(),
             'maximum_query_file_bytes': max(q['bytes'] for q in queries),
             'input_reads_per_query': 1, 'array_decodes_per_selected_field': 1,
             'derived_raw_array_bytes_per_query': 1628118,
             'derived_raw_array_bytes_all24': 39074832,
             'derived_raw_size_evidence': 'Actual ABI synthetic fixture only; exact ndarray shape arithmetic, not a real-output compression estimate.',
             'planned_comparisons': 432, 'planned_primary_comparisons': 24,
             'new_model_calls': 0, 'new_env_steps': 0, 'new_fit': False}
    stats['total_input_body_bytes'] = stats['query_NPZ_bytes'] + stats['metadata_bytes']
    stats['selected_decoded_bytes_total'] = stats['selected_decoded_bytes_per_query'] * 24
    return {
        'schema': 'PAPER2-saved-Gram-input-inventory-v1', 'status': 'PREPARED_NOT_SUBMITTED',
        'source_run_id': result['run_id'], 'synthetic': False,
        'selection': 'All 24 original CONF anchors, unchanged C/T x layers0/8/17 x flow0-5/5-9/0-9; posthoc reuse, not a new holdout.',
        'purpose': 'New global/per-position/per-head-position prefix-mass baselines from existing four-corner Grams; no refactorization or signed-vector reconstruction.',
        'metadata': metadata, 'queries': queries, 'selected_fields': list(FIELDS),
        'field_abi': {f: {k: source['queries'][0]['arrays'][f][k] for k in ('shape', 'dtype', 'bytes')} for f in FIELDS},
        'budgets': stats,
        'limits': {'output_bytes': 64 * 1024**2, 'output_raw_array_bytes': 48 * 1024**2,
                   'single_output_bytes': 4 * 1024**2, 'metadata_file_bytes': 2 * 1024**2,
                   'maximum_science_seconds': 900, 'CPU_suggested': 4, 'GPU': 0, 'memory_GiB_suggested': 8},
        'parent_producer': {'descriptor_local_path': str(descriptor), 'descriptor_identity': identity(descriptor.read_bytes()),
                            'source_manifest': producer['source_manifest'], 'source_package': producer['source_package'],
                            'delivery_local_path': str(rb / 'summary/DELIVERY-v1.json'), 'delivery_identity': identity(delivery_raw)},
        'deferred_CDEV_full_support': {'status': 'NOT_SELECTED_NO_RAW_RESTORE',
            'source_package_path': str(package), 'source_package': identity(package.read_bytes()),
            'readset_tar_member': 'metadata/READSET-v1.json.gz', 'readset_uncompressed': identity(readraw),
            'queries_by_root': {r['root_id']: len(r['queries']) for r in cdev['roots']},
            'planned_queries': cdev['planned_queries'], 'objects': len(cdev['objects']),
            'input_body_bytes': cdev['input_body_bytes'], 'query_input_bytes': cdev['query_input_bytes'],
            'maximum_query_cache_bytes': cdev['maximum_query_cache_bytes'],
            'note': 'Exact existing BOS member URI/bytes/SHA list stays in the pinned parent readset; no local full-index copy.'},
        'WO_boundary': {'status': 'NOT_SELECTED', 'three_layer_payload_obtained': False,
            'existing_L9_WO_only': 'Known AE L9 attn_vec_einsum_1/w[8], shape8x256x1024, original member3327532 bytes, SHA d1c9d6de3ed53425a390e98bfc608754cf97717bdb65b9d4afe26437866becd3.',
            'source_path': str(wpath), 'source_identity': identity(wpath.read_bytes()),
            'model_parameter_tree_sha256': weights['model_match']['loaded_parameter_tree_sha256'],
            'note': 'Observed native W is not WO. No native-W inference or weights admitted by this inventory.'},
        'activities': {'actual_array_reads': 0, 'cloud_calls': 0, 'creates': 0, 'stops': 0, 'permissions_writes': 0},
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--research-root', required=True)
    parser.add_argument('--out', type=Path, default=HERE / 'SOURCE-INPUT-INVENTORY.json')
    args = parser.parse_args()
    if args.out.resolve().parent != HERE:
        raise ValueError('inventory_output_owned_directory_only')
    value = build(args.research_root)
    args.out.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps(value['budgets']))
