"""Recompute the paper's central numerical comparisons using only the Python standard library."""
import argparse
import csv
import gzip
import hashlib
import json
import math
import statistics
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / 'reproduce/inputs'
PAPER = ROOT / 'paper1/current'
sys.path.insert(0, str(PAPER / 'figure_src/scripts'))
sys.path.insert(0, str(ROOT / 'paper1/reproduction'))
import fig4_data
import fig5_data
import rebuild_task_ordering

MEAN = statistics.mean
ARMS = {'N_C', 'N_T', 'R_resp', 'R_clean', 'R_rand', 'R_full'}


def check(condition, message):
    if not condition:
        raise ValueError(message)


def geometry():
    path = INPUT / 'Current_basis_geometry_task_scores.csv.gz'
    scores = defaultdict(dict)
    with gzip.open(path, 'rt', newline='') as stream:
        for row in csv.DictReader(stream):
            key = (row['family'], row['layer'], row['model'])
            task = row['task_component']
            check(task not in scores[key], f'duplicate task in {key}')
            retention = 1 - float(row['sse']) / float(row['residual_energy'])
            check(math.isclose(retention, float(row['residual_retention']), abs_tol=1e-12), str(key))
            scores[key][task] = retention
    check(len(scores) == 36 and all(len(v) == 28 for v in scores.values()), 'Figure 3 support')
    cells = [dict(family=f, layer=int(l)+1, basis=b, tasks=len(v), percent=100*MEAN(v.values()))
             for (f, l, b), v in sorted(scores.items())]
    ranges = {}
    for method in ('new_response', 'new_clean', 'new_random'):
        points = [row['percent'] for row in cells if row['basis'] == method]
        ranges[method] = [min(points), max(points)]
    reference = PAPER / 'latex/analysis_archive/section4-task-ordering/three_layer_task_checks.csv'
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        for source in INPUT.glob('Current_basis_*.csv.gz'):
            (work / source.stem).write_bytes(gzip.decompress(source.read_bytes()))
        generated = work / 'task-order.csv'
        rebuild_task_ordering.build(work, generated)
        check(generated.read_bytes() == reference.read_bytes(), 'G1 task table differs from manuscript file')
    check([round(x, 1) for x in ranges['new_response']] == [36.5, 83.6], 'Abstract retention')
    check(round(ranges['new_clean'][1], 1) == 10.2, 'Abstract clean control')
    return dict(task_cells=cells, percent_range=ranges, task_table_sha256=hashlib.sha256(reference.read_bytes()).hexdigest())


def restoration():
    anchors = defaultdict(dict)
    with gzip.open(INPUT / 'CASE-BEHAVIOR.jsonl.gz', 'rt') as stream:
        for line in stream:
            row = json.loads(line)
            key = tuple(row[k] for k in ('family','task_id','row_id','init_index','noise_label'))
            check(row['status'] == 'COMPLETE' and row['arm'] not in anchors[key], 'duplicate/incomplete arm')
            anchors[key][row['arm']] = row
    check(len(anchors) == 216 and all(set(v) == ARMS for v in anchors.values()), 'six-arm support')
    settings = defaultdict(list)
    outcome_reference = {(r['family'], r['task_id'], r['row_id'], int(r['init_index']),
                          int(r['noise_label']), r['arm']): int(r['success'])
                         for r in fig4_data._read('restoration_case_outcomes.csv')}
    check(len(outcome_reference) == 1296, 'outcome table support')
    for (family, task, variant, init, noise), arms in anchors.items():
        clean = arms['N_C']['issued']['first_query_issued_prefix7']
        for arm, row in arms.items():
            commands = row['issued']['first_query_issued_prefix7']
            check(len(commands) == 5 and all(len(c) == 7 for c in commands), 'command shape')
            error = math.sqrt(math.fsum((cmd[k]-ref[k])**2 for cmd,ref in zip(commands,clean) for k in range(3)))
            check(row['success'] == outcome_reference[family,task,variant,init,noise,arm], 'outcome differs')
            settings[family,task,arm,variant,init].append(error)
    tasks = defaultdict(list)
    for (family,task,arm,variant,init), errors in settings.items():
        check(len(errors) == 2, 'missing noise pair')
        tasks[family,task,arm].append(MEAN(errors))
    table = fig4_data._read('Figure4_Task_Values.csv')
    measured = {(r['family'],r['task_id'],r['arm']): float(r['initial_command_error']) for r in table}
    check(set(measured) == {k for k in tasks if k[2] != 'N_C'} or set(measured)==set(tasks), 'task table support')
    worst = max(abs(MEAN(tasks[key])-value) for key,value in measured.items())
    check(worst < 1e-9, f'task error mismatch {worst}')
    families = fig4_data.build(verbose=False)
    for family, expected in [('Camera',(24,1.85)),('Sensor',(5,3.33))]:
        d = families[family]
        diff = d['rows']['N_T']
        reduction = 100 * diff['reduction'] / d['arms']['N_T']['error']
        check((round(reduction),round(diff['success_gain'],2)) == expected,f'Figure 1b: {family}')
        d['relative_error_reduction_percent'] = reduction
    return dict(anchors=len(anchors),task_error_max_abs_difference=worst,families=families)


def outcomes():
    results = fig5_data.build(verbose=False)
    sensor = results['panels']['S_pi0']
    bg = results['panels']['B_pi05']
    check((round(sensor['ratio'],2),round(sensor['net'],1)) == (0.58,-14.3),'Table 3 Sensor')
    check((round(bg['lost'],2),round(bg['gained'],2)) == (3.57,3.57),'Table 3 Background')
    pairs = results['pairs']
    losses = [p for p in pairs if p['cell']=='SF']
    check((len(pairs),len(losses),sum(p['ratio']<1 for p in losses)) == (110,18,12),'Figure 5c support')
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('paper-results.json'))
    args = parser.parse_args()
    check(sys.flags.optimize == 0, 'Run Python without -O; estimator cross-checks use assertions.')
    for item in json.loads((INPUT/'provenance.json').read_text()):
        check(hashlib.sha256((INPUT/item['local']).read_bytes()).hexdigest()==item['local_sha256'],item['local'])
    g = geometry()
    r = restoration()
    o = outcomes()
    args.output.write_text(json.dumps(dict(geometry=g,restoration=r,behavior=o),indent=2,allow_nan=False)+'\n')
    lo,hi = g['percent_range']['new_response']
    print(f'Fig. 3   Response retention: {lo:.1f}–{hi:.1f}%; 112 manuscript task rows matched.')
    for family,d in r['families'].items():
        print(f"Fig. 4   {family}: command error reduced by {d['relative_error_reduction_percent']:.1f}%; "
              f"success change {d['rows']['N_T']['success_gain']:+.2f} pp.")
    p = o['panels']['S_pi0']
    print(f"Fig. 5   Sensor Noise (π0): early motion {p['ratio']:.3f}×; success {p['net']:+.1f} pp.")
    print(f'Wrote grouped estimates, uncertainty and matched-pair details to {args.output}')


if __name__ == '__main__':
    main()
