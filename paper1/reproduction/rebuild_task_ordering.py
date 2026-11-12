"""Build the November Section 4 task-order table from the retained task score CSVs."""
import argparse
import csv
from pathlib import Path

LAYERS = (0, 8, 17)


def scores(path, column, method, metric=None):
    result = {}
    with path.open(newline='') as stream:
        for row in csv.DictReader(stream):
            if row['model'] != method:
                continue
            if metric and (row['metric'] != metric or row['mode'] != 'exploratory_unseen_task_cross_family'):
                continue
            key = (row['family'], row['task_component'], int(row['layer']))
            if key in result:
                raise ValueError(f'Duplicate score in {path.name}: {key}')
            result[key] = float(row[column])
    return result


def build(tables, output):
    g = scores(tables / 'Current_basis_geometry_task_scores.csv', 'residual_retention', 'new_response')
    rpath = tables / 'Current_basis_readout_task_scores.csv'
    metric = 'adapter7_translation_first5'
    m = scores(rpath, 'mse', 'new_response', metric)
    z = scores(rpath, 'mse', 'zero', metric)
    if not g or g.keys() != m.keys() or m.keys() != z.keys():
        raise ValueError('Geometry, readout and zero predictor have different supports.')
    rows = []
    for family, task in sorted({key[:2] for key in g}):
        keys = [(family, task, layer) for layer in LAYERS]
        retention = [g[key] for key in keys]
        mse = [m[key] for key in keys]
        baseline = z[keys[0]]
        if any(z[key] != baseline for key in keys):
            raise ValueError(f'Zero predictor varies across layers for {family}/{task}')
        rows.append([family, task, *retention, *mse, baseline,
                     retention[0] > max(retention[1:]) and mse[2] < min(mse[:2]),
                     retention[0] > retention[1] > retention[2] and mse[0] > mse[1] > mse[2],
                     mse[2] < baseline])
    with output.open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['family', 'task_component', 'g_L1', 'g_L9', 'g_L18', 'MSE_L1',
                         'MSE_L9', 'MSE_L18', 'zero_mse', 'joint_endpoints',
                         'full_strict_order', 'L18_beats_zero'])
        writer.writerows(rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('tables', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    build(args.tables, args.output)
