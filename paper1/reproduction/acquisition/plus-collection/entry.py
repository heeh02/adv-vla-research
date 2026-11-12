"""One already-admitted GPU worker: one model load, one startup, many cases."""
import argparse
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import signal
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
Q0 = {'bytes': 2819, 'sha256': '27707f9feb721fe305ae25e24566e09b85e5cc3c5c22051ae401d013e1701148'}


def need(ok, code):
    if not ok: raise ValueError(code)


def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream: json.dump(value, stream, indent=2, allow_nan=False); stream.write('\n')


def fail(error, phase):
    return {'error_type': type(error).__name__, 'phase': phase, 'exception_text_omitted': True, 'locals_omitted': True,
            'traceback_frames': [{'source_file': Path(f.f_code.co_filename).name, 'function': f.f_code.co_name, 'line': line}
                                 for f, line in list(traceback.walk_tb(error.__traceback__))[-64:]]}


def module(name, directory):
    expected = (Path(directory)/(name+'.py')).resolve(strict=True)
    if name in sys.modules:
        need(Path(sys.modules[name].__file__).resolve() == expected, 'foreign_same_name_module')
    sys.path.insert(0, str(expected.parent))
    result = importlib.import_module(name)
    need(Path(result.__file__).resolve() == expected, 'module_origin')
    return result


def run(args):
    signal.pthread_sigmask(signal.SIG_UNBLOCK, {signal.SIGTERM, signal.SIGINT, signal.SIGALRM})
    need(sys.platform.startswith('linux'), 'cloud_linux_only')
    need(os.environ.get('MUJOCO_GL') == os.environ.get('PYOPENGL_PLATFORM') == 'egl', 'qualified_EGL_required')
    need(math.isfinite(args.deadline_epoch) and time.time() < args.deadline_epoch, 'absolute_deadline')
    out = Path(args.out).resolve()
    need(out.is_relative_to(Path('/tmp/Jinwen/adv-vla')) and not out.exists(), 'unique_owned_output')
    out.mkdir(parents=True)
    phase = 'source_gate'
    try:
        root = next(p for p in HERE.parents if (p/'AGENTS.md').is_file())
        source = json.loads(Path(args.source_manifest).read_text())
        for row in source['files']:
            path = root/row['path']; raw = path.read_bytes()
            need(len(raw) == row['bytes'] and hashlib.sha256(raw).hexdigest() == row['sha256'], 'frozen_source_byte_identity')
        shard = json.loads(Path(args.shard).read_text())
        need(shard['schema'] == 'plus-parallel-collection-shard-v1' and shard['protocol_mode'] == 'matched_goal_prompt', 'matched_goal_prompt_shard')
        selection = shard['selection']
        need(selection['status'] == 'frozen_selected_rows' and selection['outcomes_used'] is False, 'selected_not_candidate_rows_required')
        selected_raw = json.dumps(shard['case_rows'], sort_keys=True, separators=(',', ':')).encode()
        need(hashlib.sha256(selected_raw).hexdigest() == selection['selected_row_ids_sha256'], 'selected_rows_identity')
        save(out/'SHARD-v1.json', shard)
        sys.path.insert(0, str(Path(args.environment_root).resolve()/'python'))
        C = module('plus_contract', args.adapter_dir)
        runtime = module('runtime', args.adapter_dir)
        for name in ('adapter', 'official_environment', 'trusted_init', 'q0_consumer', 'official_namespace'):
            module(name, args.adapter_dir)
        module('canonical_step_capture', args.observer_dir)
        module('runtime_snapshot', args.render_snapshot_dir)
        O = module('orchestrator', args.recorder_dir)
        bridge = module('runtime_bridge', args.recorder_dir)
        enricher = module('bulk_enrich', args.frame_enricher_dir)
        frame_module = module('frame', HERE)
        P = module('production', HERE); collector = module('collect', HERE)
        phase = 'official_runtime_and_frame'
        bindings = runtime.verify_sources()
        q0 = runtime.validate_q0_gate(args.q0_gate, Q0)
        official = runtime.prepare_official(args.plus_root, args.config_dir, bindings)
        init_reader = runtime.prepare_init_reader(args.plus_root, args.init_reference, q0)
        classification = C.catalog(Path(args.plus_root)/'libero/libero/benchmark/task_classification.json')
        records, runtime_manifest = frame_module.read_frame(args, shard, C, enricher, out, classification)
        phase = 'load_one_original_model'
        policy, model_receipt = runtime.load_policy(shard['model'], args.checkpoint_root, args.checkpoint_metadata, args.checkpoint_manifest)
        campaign = getattr(policy.call, '__self__', None)
        need(campaign is not None, 'already_loaded_campaign_bound_method')
        runtime_identity = {'checkpoint': shard['model'], 'model_config': model_receipt,
            'input_transform': {'frozen_repository_source_gate': True, 'actual_encoded_inputs_saved_each_q0': True},
            'output_transform': {'frozen_repository_source_gate': True, 'actual_raw32_adapter7_saved': True},
            'tokenizer': {'private_cache_prepared_by_parent': True, 'actual_token_IDs_and_masks_saved': True,
                          'separate_asset_identity': shard.get('tokenizer_identity')},
            'normalization': {'checkpoint_gate': model_receipt['checkpoint']},
            'runtime': {'python': sys.version, 'official': official}, 'worker': {'run_id': shard['run_id']}}
        backend = bridge.ExistingPolicyBackend(campaign, runtime_identity=runtime_identity)
        # The immutable source package is byte checked at startup. Keep the old
        # qualifier's repeated logical identity checks without re-reading files.
        frozen_source_id = backend.verify_sources()
        backend.verify_sources = lambda: frozen_source_id
        save(out/'RUNTIME-v1.json', {'model': model_receipt, 'official': official, 'runtime_identity': runtime_identity,
             'model_loads': 1, 'source_verification_policy': 'startup_bytes_then_immutable_binding',
             'planned_cases': len(records), 'independent_task_count_claim': None})
        gate = {'passed': False}
        def check():
            if time.time() >= args.deadline_epoch: raise TimeoutError('collection_absolute_deadline')
        def qualify(state, record):
            if gate['passed']: return gate
            check()
            report = collector.startup(state, record, backend, out/'startup', bool(shard['startup']['full_pp']))
            need(report['numerical_contract_completed'], 'startup_numerical_gate_not_completed')
            save(out/'STARTUP-v1.json', report)
            gate.update(passed=True, trace_qualified=report['trace_qualified'])
            return gate
        results = []
        phase = 'sequential_frozen_cases'
        for index, record in enumerate(records):
            check()
            result = collector.collect_case(record, classification=classification, init_reader=init_reader,
                plus_root=args.plus_root, runtime_manifest=runtime_manifest, backend=backend,
                out=out/'cases'/f'{index:05d}', startup_callback=qualify, check=check,
                binding=shard['binding'], run_id=shard['run_id'], failure_location=fail)
            results.append(result)
            # Any neural/endpoint failure stops this worker; never retry to pass.
            if result['fatal'] is not None: break
        save(out/'WORKER_RESULT-v1.json', {'schema': 'plus-parallel-collection-worker-result-v1',
            'run_id': shard['run_id'], 'model_loads': 1, 'startup_gate_passed': gate['passed'], 'case_results': results,
            'planned_cases': len(records), 'not_started_case_ids': [r['identity']['case_id'] for r in records[len(results):]],
            'BOS_archive_closed': False, 'provider_release_proven': False, 'archive_owner': 'parent_incremental_case_archive'})
        return 0 if gate['passed'] and len(results) == len(records) and not any(r['fatal'] for r in results) else 2
    except Exception as error:
        save(out/'WORKER_FAILURE-v1.json', fail(error, phase))
        return 1


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--execute-collection', action='store_true')
    for name in ('shard','source-manifest','adapter-dir','recorder-dir','observer-dir','render-snapshot-dir',
                 'frame','frame-enricher-dir','runtime-file-manifest','plus-root','environment-root','config-dir',
                 'q0-gate','init-reference','checkpoint-root','checkpoint-metadata','checkpoint-manifest','out'):
        p.add_argument('--'+name)
    p.add_argument('--deadline-epoch', type=float)
    args = p.parse_args()
    if not args.execute_collection:
        print(json.dumps({'status': 'plan_only', 'cloud_calls': 0, 'model_loads': 0})); return 0
    for key, value in vars(args).items():
        if key not in ('checkpoint_manifest', 'execute_collection'): need(value is not None, 'required_argument:'+key)
    return run(args)


if __name__ == '__main__': raise SystemExit(main())
