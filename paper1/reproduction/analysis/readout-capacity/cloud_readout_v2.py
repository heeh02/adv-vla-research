"""E5 readout-capacity cloud worker (CPU, GPU=0, no model).

Mirrors the proven plus-centered-command-readout worker contract
(run(source, local, descriptor, writer, writer_namespace); numpy; cloud_commands
transport; writer.put), but runs the FROZEN, locally-validated E5 stack:
legacy_math (identical SHA d179b8b4...) + e5_readout_math + e5_worker.fit_split,
fitting three readout families per split (affine reference, rfridge D=2048 RBF-RFF
with the frozen per-split seed, 1-hidden-layer MLP) on identical inputs/splits/
targets/weights, then e5_worker.summarize + sign_change_counts, and the affine
reproduction gate against the frozen replay expected.json.

Source tensors stay in cloud RAM/scratch; only small JSON (RESULTS/HEADLINE/CONTRASTS,
per-split checkpoints, FAILED marker) is written to the E5 output prefix. Credentials
arrive via the STS the launcher injects (consumed by cloud_commands); none are read,
printed or written here.
"""
import gzip, io, json, os, sys, time
from pathlib import Path
import numpy as np

# Module caps the launcher reads (W.SPEC_CAP etc.), mirroring the readout worker.
SPEC_CAP = 4 * 1024**2
CASE_CAP = 2 * 1024**2
SMALL_CAP = 2 * 1024**2
CASE_TOTAL_CAP = 8 * 1024**2
DETAIL_CAP = 16 * 1024**2
BASE = 'archive://'
MODE = 'exploratory_unseen_task_cross_family'
READOUTS = ('affine', 'rfridge', 'mlp')


def need(ok, code):
    if not ok:
        raise RuntimeError(code)


def ident(b):
    import hashlib
    return {'bytes': len(b), 'sha256': hashlib.sha256(b).hexdigest()}


def encoded(d):
    return json.dumps(d, sort_keys=True, separators=(',', ':'), allow_nan=False).encode() + b'\n'


def validate_spec(spec):
    need(spec.get('schema') == 'e5-readout-capacity-input-v1', 'e5_input_schema')
    need(len(spec['fits']) == 60 and all(f['support']['mode'] == MODE for f in spec['fits']),
         'fixed60_joint_splits')
    need(len(spec['rows']) == len(spec['case_ids']) == 536 and len(set(spec['case_ids'])) == 536,
         'fixed536_cases')
    need([r['case_id'] for r in spec['rows']] == spec['case_ids'], 'ordered_cases')
    iss = spec.get('issued_ids')
    need(isinstance(iss, list) and len(iss) == 404 and len(set(iss)) == 404
         and set(iss) <= set(spec['case_ids']), 'frozen_404_issued_ids')
    refs = {}
    values = [spec['features'], spec['draws'], spec['expected']]
    for f in spec['fits']:
        values.extend((f['old_basis'], f['centered_fit']))
    import re
    for r in values:
        need(set(r) == {'uri', 'bytes', 'sha256'} and r['uri'].startswith(BASE) and '?' not in r['uri']
             and isinstance(r['bytes'], int) and 0 < r['bytes'] <= 32 * 1024**2
             and re.fullmatch('[a-f0-9]{64}', r['sha256']), 'exact_bounded_source')
        need(r['uri'] not in refs or refs[r['uri']] == r, 'consistent_duplicate_ref')
        refs[r['uri']] = r
    need(spec['read_budget'] == {'exact_GET_count': len(refs),
                                 'exact_bytes': sum(r['bytes'] for r in refs.values())}
         and spec['read_budget']['exact_bytes'] < 256 * 1024**2, 'frozen_read_budget')
    return spec


def _threads():
    """Best-effort raise BLAS threads for the <=3h deadline; context manager."""
    try:
        from threadpoolctl import threadpool_limits
        n = int(os.environ.get('E5_BLAS_THREADS', '8'))
        return threadpool_limits(limits=n)
    except Exception:
        import contextlib
        return contextlib.nullcontext()


def run(source, local, descriptor, writer, writer_namespace):
    import e5_worker as EW
    import e5_readout_math as ERM
    local = Path(local)
    caps = descriptor['caps']
    need(sys.platform.startswith('linux') and os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU_only')
    need(descriptor['resources'].get('GPU', 0) == 0, 'GPU_zero')
    need(local.resolve().is_relative_to('/tmp'), 'scratch_under_tmp')

    # The launcher stages legacy_math.py next to this worker module (on sys.path).
    M = ERM.load_legacy(Path(__file__).resolve().parent / 'legacy_math.py')

    result = {'schema': 'e5-readout-capacity-CPU-result-v1', 'run_id': descriptor['run_id'],
              'source_sha256': descriptor.get('source_sha256'), 'status': 'FAILED_PRESERVED',
              'failure': None, 'GPU': 0, 'model_calls': 0, 'environment_steps': 0,
              'readouts': list(READOUTS), 'completed_readout_splits': 0,
              'physical_GETs': 0, 'physical_input_bytes': 0, 'files': {}}

    import cloud_commands
    transport = cloud_commands.byte_loader(writer, writer_namespace)
    spec_ref = descriptor['inputs']['spec_ref']
    allowed_refs = {spec_ref['uri']: {k: spec_ref[k] for k in ('uri', 'bytes', 'sha256')}}
    seen = set()

    def fetch(r):
        need(allowed_refs.get(r['uri']) == r and r['uri'] not in seen, 'once_only_input')
        need(result['physical_GETs'] < caps['physical_GETs']
             and result['physical_input_bytes'] + r['bytes'] <= caps['input_bytes'], 'read_cap')
        result['physical_GETs'] += 1
        raw = transport(r, r['bytes'])
        need(ident(raw) == {k: r[k] for k in ('bytes', 'sha256')}, 'exact_SHA_bytes')
        seen.add(r['uri'])
        result['physical_input_bytes'] += len(raw)
        return raw

    def put(name, data, cap, gz=False):
        raw = encoded(data)
        if gz:
            raw = gzip.compress(raw, mtime=0)
        need(len(raw) <= cap, 'bounded_output:' + name)
        p = local / name.replace('/', '_')
        p.write_bytes(raw)
        ref = writer.put(p, 'summary/' + name)
        result['files'][name] = ref
        return ref

    phase = 'input'
    all_task_rows = []
    try:
        spec = validate_spec(json.loads(fetch(spec_ref)))
        # register every declared ref as fetch-allowed, by exact identity
        for f in spec['fits']:
            for key in ('old_basis', 'centered_fit'):
                allowed_refs[f[key]['uri']] = f[key]
        for key in ('features', 'draws', 'expected'):
            allowed_refs[spec[key]['uri']] = spec[key]

        phase = 'features'
        with np.load(io.BytesIO(fetch(spec['features'])), allow_pickle=False) as z:
            need(set(z.files) == {'case_ids', 'hidden_delta', 'flow0_delta', 'target'}
                 and z['case_ids'].tolist() == spec['case_ids'], 'feature_ABI_identity')
            feats = {'hidden_delta': z['hidden_delta'], 'flow0_delta': z['flow0_delta'], 'target': z['target']}
        need(feats['hidden_delta'].shape == (536, 3, 10, 1024) and feats['flow0_delta'].shape == (536, 320)
             and feats['target'].shape == (536, 390)
             and all(np.isfinite(v).all() for v in feats.values()), 'full_finite_feature_shapes')

        phase = 'draws_expected'
        draws = json.loads(gzip.decompress(fetch(spec['draws'])))['draws']
        exp_raw = fetch(spec['expected'])
        exp_path = local / 'expected.json'
        exp_path.write_bytes(exp_raw)

        rows = spec['rows']
        components = M.task_components(rows)
        allowed = EW.common_components(M, rows, components)
        need(len(allowed) == 28, 'common28_components')
        issued = set(spec['issued_ids'])   # v3: reference-matched 404 same-404 eval population
        hyper = {'readouts': list(READOUTS)}

        with _threads():
            for i, f in enumerate(spec['fits']):
                phase = 'readout_split_' + str(i)
                with np.load(io.BytesIO(fetch(f['old_basis'])), allow_pickle=False) as z:
                    old = {k: z[k] for k in z.files}
                with np.load(io.BytesIO(fetch(f['centered_fit'])), allow_pickle=False) as z:
                    new = {k: z[k] for k in z.files}
                bases = {'position_delta__basis32': np.asarray(new['position_delta__basis32'], np.float64),
                         'position_clean__basis32': np.asarray(new['position_clean__basis32'], np.float64),
                         'curve_random32': np.asarray(new['curve_random32'], np.float64),
                         'delta_uncentered_PCA': np.asarray(old['delta_uncentered_PCA'], np.float64)}
                item = {'support': f['support']}
                tr = EW.fit_split(M, feats, rows, components, item, bases, hyper, allowed=allowed, allowed_cases=issued)
                all_task_rows.extend(tr)
                result['completed_readout_splits'] = i + 1
                s = f['support']
                put('checkpoints/split-%02d.json' % i,
                    {'split': s, 'n_task_rows': len(tr), 'task_rows': tr}, DETAIL_CAP)
                if (i + 1) % 5 == 0:
                    print(json.dumps({'event': 'completed_readout_splits', 'count': i + 1, 'total': 60}), flush=True)

            need(seen == set(allowed_refs), 'complete_fixed_inputs')
            phase = 'summarize'
            estimates, contrasts = EW.summarize(M, all_task_rows, draws)
            scc = EW.sign_change_counts(contrasts)
            gate = EW.affine_gate(M, {'estimates': estimates, 'contrasts': contrasts}, str(exp_path))

        need(len(estimates) == 864 and len(contrasts) == 648,
             'full_grid_864_estimates_648_contrasts')  # 3 readouts x (4 fam x 3 layers x 3 metrics) x {8 models / 6 pairs}
        result.update(status='COMPLETE', primary_cases=404, primary_tasks=28,
                      affine_reproduction_gate=gate, sign_change_counts=scc,
                      n_task_rows=len(all_task_rows),
                      inference='exploratory pointwise95 conditional on fixed bases and fitted readouts; '
                                'capacity control on readout family only; not causal or outcome evidence')
        put('RESULTS-v1.json', {'estimates': estimates, 'contrasts': contrasts,
                                'affine_reproduction_gate': gate, 'sign_change_counts': scc,
                                'readouts': list(READOUTS), 'splits_used': len(spec['fits']),
                                'primary_cases': 404, 'primary_tasks': 28,
                                'run_id': descriptor['run_id']}, DETAIL_CAP)
        put('HEADLINE-v1.json', _headline(estimates, contrasts, gate, scc), SMALL_CAP)
    except BaseException as e:
        import re as _re
        msg = str(e)
        result['failure'] = {'phase': phase, 'exception_type': type(e).__name__,
                             'contract_code': msg if _re.fullmatch('[A-Za-z0-9_:.-]{1,200}', msg) else None,
                             'msg': msg[:300], 'automatic_retry': False}
    result['epoch'] = time.time()
    raw = encoded(result)
    p = local / 'RESULT-v1.json'
    p.write_bytes(raw)
    writer.put(p, 'summary/RESULT-v1.json')
    if result['status'] != 'COMPLETE':
        try:
            fp = local / 'FAILED.json'
            fp.write_bytes(encoded({'status': 'WORKER_FAILED', **(result['failure'] or {}),
                                    'run_id': descriptor['run_id'], 'epoch': time.time(), 'GPU': 0}))
            writer.put(fp, 'summary/FAILED-v1.json')
        except Exception:
            pass
    print(json.dumps({'status': result['status'], 'failure': result['failure'],
                      'completed_readout_splits': result['completed_readout_splits']}), flush=True)
    return result


def _headline(estimates, contrasts, gate, scc):
    fams = ('Background Textures', 'Camera Viewpoints', 'Light Conditions', 'Sensor Noise')
    idx = {(c['readout'], c['family'], c['layer'], c['metric'], c['candidate'], c['baseline']): c
           for c in contrasts}
    out = {'affine_gate': gate,
           'sign_change_counts': {ro: {k: scc[ro][k] for k in
                                  ('contrasts_compared', 'point_sign_changes', 'ci_category_changes')}
                                  for ro in ('rfridge', 'mlp')},
           'l9_translation_new_response_vs_new_clean': {},
           'l18_translation_new_response_vs_new_clean': {},
           'l18_rotation_new_response_vs_new_clean': {}}
    for layer, key in ((8, 'l9_translation_new_response_vs_new_clean'),
                       (17, 'l18_translation_new_response_vs_new_clean')):
        for fam in fams:
            out[key][fam] = {}
            for ro in READOUTS:
                c = idx.get((ro, fam, layer, 'adapter7_translation_first5', 'new_response', 'new_clean'))
                out[key][fam][ro] = None if c is None else {
                    'rer': c.get('relative_error_reduction'),
                    'ci': c.get('relative_error_reduction_interval_95'), 'ci_sign': c.get('rel_ci_sign')}
    for fam in fams:
        out['l18_rotation_new_response_vs_new_clean'][fam] = {}
        for ro in READOUTS:
            c = idx.get((ro, fam, 17, 'adapter7_rotation_first5', 'new_response', 'new_clean'))
            out['l18_rotation_new_response_vs_new_clean'][fam][ro] = None if c is None else c.get('rel_ci_sign')
    return out
