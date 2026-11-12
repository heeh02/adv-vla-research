"""Cloud-CPU-only analysis of fixed, previously verified scalar caches."""
import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback
import audit as A
import shape

HERE = Path(__file__).resolve().parent
RESULT = 'pvp-shape-result.json'
OUTPUTS = (RESULT, 'SHAPE-VALUES.json', 'QA.json', 'DECISION.json',
           'SHAPE-SUPPORT.csv', 'PROTOCOL.json')


def identity(raw):
    return {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


def write(path, value):
    with path.open('xb') as f:
        f.write((json.dumps(value, sort_keys=True, indent=2, allow_nan=False)+'\n').encode())


def load(stage, source_sha, check):
    A.need(sys.platform.startswith('linux'), 'real_cache_analysis_only_on_cloud_Linux')
    raw = (HERE/'SOURCE-MANIFEST.json').read_bytes()
    A.need(identity(raw)['sha256'] == source_sha, 'externally_bound_source_manifest')
    source = json.loads(raw)
    A.need(source['schema'] == 'paper2-pvp-shape-source-v1', 'source_schema')
    for row in source['files']:
        p = HERE/row['path']
        A.need(p.is_file() and not p.is_symlink() and p.resolve().is_relative_to(HERE),
               'regular_source_member')
        A.need(identity(p.read_bytes()) == {k: row[k] for k in ('bytes', 'sha256')},
               'source_member_SHA')
    manifest = json.loads((HERE/'INPUTS.json').read_bytes())
    refs = manifest['input_objects']
    A.need(len(refs) == 3 and {r['path'] for r in refs} ==
           {'FULL-CACHED-SUPPORT.json.gz', 'QA.json', 'conf24-figures-result.json'},
           'exact_three_cached_inputs')
    A.need({p.name for p in stage.iterdir()} == {r['path'] for r in refs},
           'only_declared_stage_inputs')
    cache = {}
    for row in refs:
        p = stage/row['path']
        A.need(p.is_file() and not p.is_symlink() and p.stat().st_size <= 2*1024**2,
               'bounded_regular_cache')
        raw = p.read_bytes()
        A.need(identity(raw) == {k: row[k] for k in ('bytes', 'sha256')}, 'input_SHA')
        if p.suffix == '.gz':
            with gzip.open(p, 'rb') as f:
                raw = f.read(16*1024**2+1)
            A.need(len(raw) <= 16*1024**2, 'bounded_decompressed_cache')
        cache[p.name] = json.loads(raw)
        check()
    producer = cache['conf24-figures-result.json']
    qa = cache['QA.json']
    support = cache['FULL-CACHED-SUPPORT.json.gz']
    A.need(producer['schema'] == 'paper2-conf24-figures-result-v1'
           and producer['status'] == 'CONF24_FIGURES_COMPLETE'
           and producer['analysis_complete'] and producer['input_qualified']
           and producer['source_manifest']['sha256'] ==
           '12d031d7419b4f563ea91e1fde4ef14ded72ba8393b53cad75e0b562bcbc48dd',
           'exact_successful_cache_producer')
    A.need(all(v.get('synthetic') is False for v in (producer, qa, support)),
           'actual_scope_only')
    A.need(qa['status'] == 'PASS' and qa['snapshot_count'] == 24 and
           qa['all_snapshot_dimension_window_cells'] == 336,
           'prior_complete_scalar_QA')
    A.need(support['schema'] == 'paper2-conf24-all-cached-support-v1',
           'cached_support_schema')
    produced = {r['path']: r for r in producer['outputs']}
    for row in refs:
        if row['path'] == 'conf24-figures-result.json':
            continue
        A.need({k: produced[row['path']][k] for k in ('bytes', 'sha256')} ==
               {k: row[k] for k in ('bytes', 'sha256')}, 'producer_output_binding')
    A.need(sum(r['bytes'] for r in refs) == manifest['input_body_bytes'] == 294209,
           'exact_input_byte_total')
    return support['tables'], manifest, identity((HERE/'SOURCE-MANIFEST.json').read_bytes())


def run(stage, out, source_sha, deadline):
    out.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    result = dict(schema='paper2-pvp-shape-result-v1', status='INCOMPLETE',
                  input_qualified=False, analysis_complete=False, synthetic=False,
                  new_model_calls=0, new_env_steps=0, training_updates=0,
                  new_predictive_fits=0, GPU_count=0, JPFS_new_bytes=0,
                  analytic_projection_pairs=0, prediction_model_claim=False)
    phase = 'source_and_inputs'
    def check():
        A.need(time.time() < deadline, 'absolute_science_deadline')
    try:
        check()
        tables, manifest, source = load(stage, source_sha, check)
        result.update(input_qualified=True, input_objects_verified=3,
                      input_body_bytes=294209, source_manifest=source,
                      inputs_manifest=identity((HERE/'INPUTS.json').read_bytes()))
        phase = 'analytic_response_projection'
        rows = shape.analyze(tables)
        check()
        summary = shape.summarize(rows)
        statuses = {s: sum(r['status'] == s for r in rows)
                    for s in sorted({r['status'] for r in rows})}
        invalid = sum(n for s, n in statuses.items() if s.startswith('invalid'))
        qa = dict(schema='paper2-pvp-shape-QA-v1', synthetic=False,
                  passed=invalid == 0, source_rows=1008, projection_pairs=336,
                  status_counts=statuses, invalid_cells=invalid,
                  moment_clamped=sum(r['moment_clamped'] for r in rows),
                  rounding_sensitive=sum(r['rounding_sensitive'] for r in rows),
                  fixed_moment_tolerance=shape.MOMENT_TOL,
                  tolerance_is_not_native_noise_or_scientific_significance=True,
                  no_undefined_denominator_replaced=True, source_input_refs=manifest['input_objects'])
        values = dict(schema='paper2-pvp-shape-values-v1', synthetic=False,
                      source_input_refs=manifest['input_objects'], cells=rows,
                      by_axis_window=summary, independent_sample_count_not_claimed=True)
        decision = dict(
            schema='paper2-pvp-shape-decision-v1', synthetic=False,
            question='Does a constant signed gain per coordinate explain its candidate-position effect vector?',
            status_counts=statuses, by_axis_window=summary,
            interpretation='Residual fractions describe effect-sequence shape after allowing an in-sample signed scale; a negative gain alone is not shape deformation.',
            alternatives_retained=['constant nonnegative gain', 'constant signed gain',
                                   'zero or numerically unresolved response'],
            not_identified=['native repetition noise', 'causal feedback versus continued exposure',
                            'controller clipping', 'gripper events', 'physical motion', 'task success',
                            'new task generalization', 'held-out prediction'],
            unit='command_adapter_units, analyzed separately per coordinate',
            selection='all frozen24 snapshots, seven coordinates, first5 and full10; no outcome selection',
            hypothesis_status='retrospective descriptive diagnostic, not new confirmation',
        )
        for name, value in [('SHAPE-VALUES.json', values), ('QA.json', qa),
                            ('DECISION.json', decision)]:
            write(out/name, value)
        keys = sorted(set().union(*(r.keys() for r in rows)))
        with (out/'SHAPE-SUPPORT.csv').open('x', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)
        write(out/'PROTOCOL.json', json.loads((HERE/'PROTOCOL.json').read_bytes()))
        A.need({p.name for p in out.iterdir()} == set(OUTPUTS)-{RESULT}, 'exact_output_roster')
        A.need(all(p.stat().st_size <= 2*1024**2 for p in out.iterdir()), 'small_output_budget')
        check()
        result.update(status='PVP_SHAPE_COMPLETE', analysis_complete=True,
                      analytic_projection_pairs=336,
                      defined_projection_pairs=statuses.get('defined', 0),
                      analytic_scale_candidates_per_pair=2, QA_passed=invalid == 0,
                      outputs=[{'path': p.name, **identity(p.read_bytes())}
                               for p in sorted(out.iterdir())])
    except BaseException as e:
        result.update(status='FAILED_PRESERVED', failure={'phase': phase, 'type': type(e).__name__,
                      'code': str(e)[:160] if isinstance(e, ValueError) else None,
                      'frames': [{'file': Path(x.filename).name, 'line': x.lineno, 'function': x.name}
                                 for x in traceback.extract_tb(e.__traceback__)]})
    result['elapsed_seconds'] = time.monotonic()-start
    write(out/RESULT, result)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--stage', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--source-manifest-sha256', required=True)
    p.add_argument('--deadline-epoch', type=float, required=True)
    args = p.parse_args()
    result = run(args.stage, args.out, args.source_manifest_sha256, args.deadline_epoch)
    print(json.dumps({'status': result['status'], 'analysis_complete': result['analysis_complete']}))
    raise SystemExit(0 if result['analysis_complete'] else 2)
