"""Linux-only cache consumer. All cloud transport and submission belongs to root."""
from pathlib import Path
import argparse
import gzip
import hashlib
import json
import math
import sys
import time
import traceback
import audit as A

HERE = Path(__file__).resolve().parent
RESULT = 'conf24-figures-result.json'
CORE = 'F3-NATIVE-PVP'
PAPER_INPUT_NAMES = ('FIGURE-VALUES.json', 'QA.json', 'signed-figures-result.json')
PAPER_OUTPUT_NAMES = tuple(f'{name}.{ext}' for name in ('F1-OVERVIEW', 'F2-CAPACITY-TRANSFER')
                           for ext in ('pdf', 'svg', 'png')) + (
                               'PAPER-PANEL-VALUES.json', 'PAPER-PANEL-QA.json')
OUTPUT_NAMES = tuple(CORE+'.'+ext for ext in ('pdf', 'svg', 'png')) + (
    'SUPPORT-ATLAS.pdf', 'FULL-COMMAND-SUPPORT.csv.gz', 'FULL-CACHED-SUPPORT.json.gz',
    'FIGURE-VALUES.json', 'CAPTIONS.json', 'QA.json', 'RENDER-MANIFEST.json', RESULT,
) + tuple(f'ATLAS-{p:02d}.{ext}' for p in range(1, 10) for ext in ('svg', 'png')) + PAPER_OUTPUT_NAMES


def ident(path):
    h = hashlib.sha256()
    count = 0
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk); count += len(chunk)
    return {'bytes': count, 'sha256': h.hexdigest()}


def serial(value):
    if isinstance(value, float):
        A.need(math.isfinite(value), 'no_nonfinite_JSON')
    if isinstance(value, dict):
        return {str(k): serial(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [serial(v) for v in value]
    return value


def write(path, value, *, compressed=False):
    raw = (json.dumps(serial(value), sort_keys=True, separators=(',', ':'),
                      allow_nan=False)+'\n').encode()
    if compressed:
        with Path(path).open('xb') as f:
            with gzip.GzipFile(fileobj=f, mode='wb', mtime=0) as zipped:
                zipped.write(raw)
    else:
        with Path(path).open('xb') as f:
            f.write(raw)
    return ident(path)


def load_tables(stage):
    tables = {kind: [] for kind in A.HEADERS}
    for task in A.TASKS:
        for kind in ('FLOW-SUMMARY', 'MECHANISM-SUMMARY', 'READING_EXACT', 'WINDOWS'):
            rows = A.parse_csv(stage/f'{task}-PVP-{kind}-v1.csv', kind)
            A.need(all(r['task_id'] == task for r in rows), 'filename_task_binding')
            tables[kind].extend(rows)
    tables['COVERAGE'] = A.parse_csv(stage/'PVP-COVERAGE-v1.csv', 'COVERAGE')
    return tables


def run(stage, out, source_sha, deadline):
    # Deliberately before output initialization, manifest or any input-body read.
    A.need(sys.platform.startswith('linux'), 'real_CONF24_cached_numerics_cloud_Linux_only')
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    A.need(not out.is_symlink() and not any(out.iterdir()), 'new_empty_owned_output')
    result = {'schema': 'paper2-conf24-figures-result-v1', 'status': 'INCOMPLETE',
              'analysis_complete': False, 'render_complete': False, 'synthetic': False,
              'input_qualified': False, 'new_model_calls': 0, 'new_env_steps': 0,
              'new_fits': 0, 'training_updates': 0, 'GPU_count': 0, 'JPFS_new_bytes': 0,
              'original100objects_reopened': False,
              'native_qualification_reexecuted': False, 'new_blind_holdout': False}
    started = time.monotonic(); phase = 'source'

    def check():
        A.need(time.time() < deadline, 'absolute_science_deadline')

    try:
        check()
        ref = ident(HERE/'SOURCE-MANIFEST.json')
        A.need(ref['sha256'] == source_sha, 'externally_bound_source_manifest')
        source = json.loads((HERE/'SOURCE-MANIFEST.json').read_bytes())
        A.need(source['schema'] == 'paper2-conf24-figures-source-v1', 'source_schema')
        for row in source['files']:
            path = HERE/row['path']
            A.need(path.is_file() and not path.is_symlink() and
                   path.resolve().is_relative_to(HERE), 'owned_source_member')
            A.need(ident(path) == {k: row[k] for k in ('bytes', 'sha256')}, 'source_member_SHA')
        manifest = json.loads((HERE/'INPUTS.json').read_bytes())
        contract = json.loads((HERE/'FIGURE-CONTRACT.json').read_bytes())
        refs = manifest['input_objects']
        input_names = set(A.INPUT_NAMES) | set(PAPER_INPUT_NAMES)
        A.need(len(refs) == 21 and {r['path'] for r in refs} == input_names,
               'exact21_cached_objects')
        A.need(sum(r['bytes'] for r in refs) == manifest['input_body_bytes'] and
               manifest['input_body_bytes'] <= 4*1024**2, 'bounded_exact_input_bytes')
        stage = Path(stage).resolve(strict=True)
        result.update(source_manifest=ref, inputs_manifest=ident(HERE/'INPUTS.json'),
                      figure_contract=ident(HERE/'FIGURE-CONTRACT.json'), scope=contract['scope'])
        phase = 'input_binding'
        A.need({p.name for p in stage.iterdir()} == input_names, 'only_declared_stage_inputs')
        for row in refs:
            p = stage/row['path']
            A.need(p.is_file() and not p.is_symlink(), 'regular_staged_cache')
            A.need(ident(p) == {k: row[k] for k in ('bytes', 'sha256')}, 'whole_cache_bytes_SHA')
            check()
        producer = json.loads((stage/'RESULT-v1.json').read_bytes())
        bound = {r['path']: r for r in producer['output_files']}
        for row in refs:
            if row['path'] == 'RESULT-v1.json' or row['path'] in PAPER_INPUT_NAMES:
                continue
            A.need(row['path'] in bound and
                   {k: bound[row['path']][k] for k in ('bytes', 'sha256')} ==
                   {k: row[k] for k in ('bytes', 'sha256')}, 'producer_output_binding')
        phase = 'strict_scalar_QA'
        tables = load_tables(stage)
        data = A.verify(tables, producer)
        data['qa']['input_objects'] = refs
        write(out/'QA.json', data['qa'])
        result.update(PVP_input_qualified=True, input_objects_verified=21,
                      input_body_bytes=manifest['input_body_bytes'], PVP_QA_passed=True)
        check(); phase = 'render'
        import renderer
        rendered = renderer.render(data, out, check)
        check(); phase = 'paper_panels'
        import paper_panels
        paper = paper_panels.render(stage, out, check)
        A.need(set(paper['outputs']) == set(PAPER_OUTPUT_NAMES) and
               len(paper['outputs']) == len(PAPER_OUTPUT_NAMES), 'exact8_paper_panel_outputs')
        result.update(input_qualified=True, QA_passed=True)
        write(out/'FULL-CACHED-SUPPORT.json.gz',
              {'schema': 'paper2-conf24-all-cached-support-v1', 'synthetic': False,
               'input_objects': refs, 'tables': tables}, compressed=True)
        write(out/'FIGURE-VALUES.json',
              {'schema': 'paper2-conf24-figure-values-v1', 'synthetic': False,
               'input_objects': refs, **rendered['main_values']})
        write(out/'CAPTIONS.json', rendered['captions'])
        render_manifest = {
            'schema': 'paper2-conf24-render-manifest-v1', 'synthetic': False,
            'source_manifest': ref, 'inputs_manifest': result['inputs_manifest'],
            'figure_contract': result['figure_contract'], 'QA': ident(out/'QA.json'),
            'core_layouts': rendered['core_layouts'], 'atlas_pages': rendered['atlas_pages'],
            'paper_panel_layouts': paper['layouts'], 'paper_panel_numerical_QA': paper['numerical_QA'],
            'atlas_page_count': 9, 'command_support_rows': 336,
            'ratio_axis': 'shared log2, full finite positive range; zero and NA separate', 'minimum_font_points': 7,
            'captions': ident(out/'CAPTIONS.json'), 'figure_values': ident(out/'FIGURE-VALUES.json'),
            'outputs': []}
        for p in sorted(out.iterdir()):
            A.need(p.name in OUTPUT_NAMES and p.is_file() and not p.is_symlink(),
                   'declared_regular_output')
            record = {'path': p.name, **ident(p)}
            A.need(record['bytes'] <= 8*1024**2, 'per_output8MiB')
            render_manifest['outputs'].append(record)
        write(out/'RENDER-MANIFEST.json', render_manifest)
        A.need({p.name for p in out.iterdir()} == set(OUTPUT_NAMES)-{RESULT},
               'complete_exact_output_roster')
        A.need(sum(p.stat().st_size for p in out.iterdir())+2*1024**2 <= 32*1024**2,
               'total_output_budget32MiB')
        check()
        result.update(status='CONF24_FIGURES_COMPLETE', analysis_complete=True,
                      render_complete=True, main_figures=3, atlas_pages=9,
                      command_support_rows=336, output_names=list(OUTPUT_NAMES),
                      outputs=[{'path': p.name, **ident(p)} for p in sorted(out.iterdir())])
    except BaseException as error:
        result.update(status='FAILED_PRESERVED', analysis_complete=False, render_complete=False,
                      failure={'phase': phase, 'type': type(error).__name__,
                               'code': str(error) if isinstance(error, (ValueError, RuntimeError)) and
                               len(str(error)) < 180 and all(c.isalnum() or c in '_-' for c in str(error)) else None,
                               'frames': [{'file': Path(f.filename).name, 'function': f.name, 'line': f.lineno}
                                          for f in traceback.extract_tb(error.__traceback__)[-10:]]})
    result['elapsed_seconds'] = time.monotonic()-started
    write(out/RESULT, result)
    return result


def main():
    p = argparse.ArgumentParser()
    for name in ('stage', 'out', 'source-manifest-sha256', 'deadline-epoch'):
        p.add_argument('--'+name, required=True)
    a = p.parse_args()
    result = run(a.stage, a.out, a.source_manifest_sha256, float(a.deadline_epoch))
    print(json.dumps({'status': result['status'], 'analysis_complete': result['analysis_complete'],
                      'failure': result.get('failure')}), flush=True)
    return 0 if result['analysis_complete'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
