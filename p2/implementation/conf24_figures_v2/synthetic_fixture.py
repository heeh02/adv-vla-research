"""Explicitly synthetic scalar fixtures. Never locates or reads real PVP caches."""
from pathlib import Path
from itertools import product
import argparse
import csv
import json
import math
import audit as A


def base(snapshot, contrast):
    s = dict(zip(A.IDENTITY, snapshot))
    return {**s, 'policy_env_step': 5*s['query_index'],
            'absolute_env_step': 10+5*s['query_index'], 'graph': 'Gstar',
            'contrast': contrast}


def stats(rms, positions, width):
    n = positions*width
    return {'action_window': 'first5' if positions == 5 else 'full10',
            'action_positions': positions, 'coordinates_in_window': n,
            'signed_mean': -.1*rms, 'RMS_difference': rms, 'maxabs_difference': 2*rms,
            'difference_energy': rms*rms*n, 'reference_energy': 1., 'candidate_energy': 2.,
            'all_exactzero_difference': rms == 0, 'different_coordinates': n//2 if rms else 0,
            'summary_not_independent_samples': True}


def fixture():
    tables = {kind: [] for kind in A.HEADERS}
    for si, s in enumerate(A.SNAPSHOTS):
        tables['COVERAGE'].append({**dict(zip(A.IDENTITY, s)), 'status': 'COMPLETE',
                                  'same_j5_and_x6': True, 'same_j6_L1_L9_QP': True,
                                  'actual_E_exactzero': False})
        for c, j, l in product(A.CONTRASTS, A.FLOWS, A.LAYERS):
            early = c == 'persistent_minus_pulse' and (j == 5 or (j == 6 and l in (1, 9)))
            tables['READING_EXACT'].append({
                **base(s, c), 'flow_index': j, 'reference_flow_time': (10-j)/10,
                'candidate_flow_time': (10-j)/10, 'layer_index': l-1, 'layer_label': l,
                **{k: bool(early or k == 'attention_gate_typed_exact') for k in A.EXACT}})
        for j, l, field, window in product(A.FLOWS, A.LAYERS, A.FIELDS, A.WINDOWS):
            zero = j == 5 or (j == 6 and l == 1) or (
                j == 6 and l == 9 and field == 'receiver_residual_input')
            rms = 0. if zero else .003*(si+1)*(j-5)*(l/9)
            tables['MECHANISM-SUMMARY'].append({
                **base(s, 'persistent_minus_pulse'), 'field': field, 'flow_index': j,
                'reference_flow_time': (10-j)/10, 'candidate_flow_time': (10-j)/10,
                'coordinates': 'AE_1024_native', 'layer_index': l-1, 'layer_label': l,
                **stats(rms, 5 if window == 'first5' else 10, 1024)})
        for c, field in product(A.CONTRASTS, ('x', 'velocity_native')):
            for j in range(11 if field == 'x' else 10):
                zero = c == 'persistent_minus_pulse' and j < (7 if field == 'x' else 6)
                rms = 0. if zero else .002*(si+1)*j
                tables['FLOW-SUMMARY'].append({
                    **base(s, c), 'field': field, 'flow_index': j,
                    'reference_flow_time': (10-j)/10, 'candidate_flow_time': (10-j)/10,
                    'coordinates': 'native_32_not_physical_units', **stats(rms, 10, 32)})
        for window, (di, dim) in product(A.WINDOWS, enumerate(A.DIMENSIONS)):
            n = 5 if window == 'first5' else 10
            magnitude = .001*(si+1)*(di+1)
            pulse = [magnitude*(1.+p/10) for p in range(n)]
            factor = (.4, 1., 2.5)[(si+di) % 3]
            sign = -1 if (si+di) % 4 == 0 else 1
            persistent = [v*factor*sign for v in pulse]
            if si == 0 and di == 0: pulse = [0.]*n
            if si == 1 and di == 1: persistent = [0.]*n
            if si == 2 and di == 2: pulse = persistent = [0.]*n
            if si == 3 and di == 3:
                pulse = [-magnitude, magnitude]+[0.]*(n-2)
            for c, values in zip(A.CONTRASTS, (pulse, persistent,
                                              [p-a for p, a in zip(persistent, pulse)])):
                tables['WINDOWS'].append({
                    **base(s, c), 'window': window, 'command_dimension': dim,
                    'mean_signed': sum(values)/n, 'meanabs': sum(abs(v) for v in values)/n,
                    'RMS': math.sqrt(sum(v*v for v in values)/n),
                    'maxabs': max(abs(v) for v in values),
                    'exactzero_slots': sum(v == 0 for v in values), 'positions': n,
                    'unit': 'command_adapter_units', 'proposed_not_issued': True})
    producer = {'schema': 'PVP-existing-Gstar-CPU-readout-v1', 'complete': True,
                'synthetic': True, 'planned_queries': 24, 'completed_queries': 24,
                'errors': [], 'new_model_calls': 0, 'new_environment_steps': 0,
                'new_fit': False, 'old_R_arrays_used': False,
                'behavior_or_equal_dose_timing_claim': False,
                'coverage': tables['COVERAGE']}
    return tables, producer


def roundtrip(tables, directory):
    """Exercise the real parser on synthetic CSVs, never on archived paths."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    decoded = {}
    for kind, rows in tables.items():
        path = directory/(kind+'.csv')
        with path.open('w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=A.HEADERS[kind])
            w.writeheader(); w.writerows(rows)
        decoded[kind] = A.parse_csv(path, kind)
    return decoded


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', required=True)
    args = p.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=False)
    tables, producer = fixture()
    data = A.verify(roundtrip(tables, out/'synthetic-inputs'), producer, synthetic=True)
    import renderer
    rendered = renderer.render(data, out)
    (out/'SYNTHETIC-QA.json').write_text(json.dumps({
        'synthetic': True, 'scientific_result': False, 'QA': data['qa'],
        'core_layouts': rendered['core_layouts'], 'atlas_pages': rendered['atlas_pages']},
        indent=2)+'\n')
    print(json.dumps({'synthetic': True, 'atlas_pages': len(rendered['atlas_pages']),
                      'path': str(out)}))


if __name__ == '__main__':
    main()
