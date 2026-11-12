"""Cloud CPU archive/library inventory. No model or environment invocation."""
import argparse
import io
import json
import math
from pathlib import Path
import sys
import time
import traceback
import archive_reader as A
from extract import extract
from inventory import inventory

HERE = Path(__file__).resolve().parent
RESULT = 'init-library-inventory-result.json'


def run(stage, out, manifest_sha, deadline):
    A.need(sys.platform.startswith('linux') and math.isfinite(deadline), 'cloud_CPU_only')
    stage, out = Path(stage), Path(out)
    A.need(not out.exists(), 'fresh_output_directory'); out.mkdir(parents=True)
    result = {'schema': 'paper2-init-library-inventory-result-v1', 'status': 'FAILED_PRESERVED',
              'analysis_complete': False, 'input_qualified': False, 'synthetic': False,
              'GPU_count': 0, 'new_model_calls': 0, 'new_env_steps': 0, 'new_fits': 0,
              'training_updates': 0, 'JPFS_new_bytes': 0, 'runtime_imports': 0,
              'native_runtime_imports': 0, 'environment_constructions': 0}
    def check():
        A.need(time.time() < deadline, 'science_absolute_deadline')
    def write(name, value):
        raw = A.encoded(value); A.need(len(raw) < 2 * 1024**2, 'bounded_JSON_output')
        with (out / name).open('xb') as f:
            f.write(raw)
    started = time.monotonic(); phase = 'source'
    try:
        raw = (HERE / 'SOURCE-MANIFEST.json').read_bytes()
        A.need(A.fid(raw)['sha256'] == manifest_sha, 'source_manifest_SHA')
        manifest = A.json_read(raw)
        A.need(manifest['schema'] == 'paper2-init-library-inventory-source-v1', 'source_schema')
        for ref in manifest['files']:
            p = HERE / ref['path']; A.relative(ref['path'])
            A.need(p.is_file() and not p.is_symlink() and A.fid(p.read_bytes()) == {k: ref[k] for k in ('bytes', 'sha256')}, 'source_member_SHA')
        ir = (HERE / 'INPUTS.json').read_bytes(); inputs = A.json_read(ir)
        libraries = A.json_read((HERE / 'LIBRARY-ROSTER.json').read_bytes())['libraries']
        reference = A.json_read((HERE / 'REFERENCE-ROOTS.json').read_bytes())
        A.need(len(libraries) == 10 and len(reference['roots']) == 4, 'fixed_libraries_and_reference_roots')
        A.need(inputs['input_object_count'] == 15 and inputs['input_body_bytes'] == 786530837 and len(inputs['archives']) == 2, 'fixed_existing_capsules')
        result.update(source_manifest=A.fid(raw), input_manifest=A.fid(ir))
        phase = 'verify_archives'; blobs, proofs = extract(stage, inputs, libraries, check)
        # These are the existing, whole-SHA-qualified original numeric libraries.
        # Preserve the original default-loader incompatibility; do not retry it.
        phase = 'CPU_library_deserialization'
        import torch
        import numpy as np
        original_load = torch.load
        inventories, matches = [], []
        for lib in libraries:
            check(); payload = blobs[lib['path']]
            A.need(A.fid(payload) == {k: lib[k] for k in ('bytes', 'sha256')}, 'library_SHA_before_deserialization')
            values = original_load(io.BytesIO(payload), map_location='cpu', weights_only=False)
            inv, checked = inventory(values, lib, reference['roots'])
            inventories.append(inv); matches.extend(checked)
            A.need(torch.load is original_load, 'global_torch_load_unchanged')
        A.need(len(matches) == 4 and len({r['root_id'] for r in matches}) == 4, 'all4_reference_roots_matched')
        imported = [n for n in sys.modules if n.split('.')[0] in {'openpi', 'jax', 'jaxlib', 'libero', 'robosuite', 'mujoco', 'mujoco_py'}]
        result['native_runtime_imports'] = result['runtime_imports'] = len(imported)
        A.need(not imported, 'no_model_or_environment_runtime_imports')
        write('LIBRARY-INVENTORY.json', {'schema': 'paper2-initialization-library-inventory-v1', 'libraries': inventories,
            'arrays_exported': False, 'new_task_or_initialization_selected': False,
            'unique_rows_do_not_establish_independence_or_unexposed_status': True})
        write('EXCLUSION-MATCHES.json', {'schema': 'paper2-reviewed-DEV-root-exclusions-v1', 'matched_references': matches,
            'scope': reference['exclusion_scope'], 'unmatched_rows_not_certified_unused': True})
        write('QA.json', {'pass': True, 'archive_proofs': proofs, 'matched_reference_roots': 4,
            'all10_library_members_whole_SHA_verified_before_load': True, 'global_torch_load_unchanged': True,
            'array_values_not_in_outputs': True, 'new_unseen_or_independent_population_claim': False,
            'torch': torch.__version__, 'numpy': np.__version__, 'torch_CPU_loads': 10})
        write('PROTOCOL.json', A.json_read((HERE / 'PROTOCOL.json').read_bytes()))
        check()
        result.update(status='INIT_LIBRARY_INVENTORY_COMPLETE', analysis_complete=True, input_qualified=True,
            input_objects_verified=15, input_body_bytes=786530837,
            counts={'libraries': 10, 'archives': 2, 'reference_roots': 4},
            library_rows_observed=sum(r['row_count'] for r in inventories),
            typed_unique_rows_total=sum(r['typed_unique_rows'] for r in inventories),
            exact_numeric_unique_rows_total=sum(r['exact_numeric_unique_rows'] for r in inventories),
            output_files=[{'path': p.name, **A.fid(p.read_bytes())} for p in out.iterdir() if p.is_file()])
    except Exception as e:
        result['failure'] = {'phase': phase, 'type': type(e).__name__, 'code': str(e)[:160],
            'frames': [{'file': Path(f.filename).name, 'line': f.lineno, 'function': f.name} for f in traceback.extract_tb(e.__traceback__)[-7:]]}
    result['elapsed_seconds'] = time.monotonic() - started
    write(RESULT, result)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--stage', required=True); p.add_argument('--out', required=True)
    p.add_argument('--source-manifest-sha256', required=True); p.add_argument('--deadline-epoch', type=float, required=True)
    a = p.parse_args(); r = run(a.stage, a.out, a.source_manifest_sha256, a.deadline_epoch)
    print(json.dumps({'status': r['status'], 'analysis_complete': r['analysis_complete']}))
    raise SystemExit(0 if r['analysis_complete'] else 2)
