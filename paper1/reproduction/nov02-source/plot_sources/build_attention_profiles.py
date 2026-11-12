"""Aggregate archived P/W/native-H scalars on the original matched 404 pairs.

This deterministic display build performs no inference, fitting or resampling.
Relative responses are averaged per case, not divided after aggregation.
"""
from pathlib import Path
from collections import defaultdict
import csv
import gzip
import hashlib
import json
import math

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'data/PWH-JOIN-v1.csv.gz'
EXPECTED_SHA = 'c2ec6807eab9576f768e1fcabd2b94922ff8558c6aa56d027a38b5fa18de6d96'
assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == EXPECTED_SHA
with gzip.open(SOURCE, 'rt') as stream:
    all_rows = list(csv.DictReader(stream))
assert len(all_rows) == 4824
rows = [r for r in all_rows if r['joint404_derived'] == 'True']
assert len(rows) == 3636 and len({r['case_id'] for r in rows}) == 404
fields = ['p_full_key_tv', 'w_delta_rms', 'w_reference_rms', 'w_relative_l2',
          'ae_delta_rms', 'ae_reference_rms', 'ae_relative_l2']
groups = defaultdict(list)
for r in rows:
    if int(r['layer_index']) == 17:
        groups[(r['family'], int(r['generation_call']))].append(r)
profiles = []
for (family, step), records in sorted(groups.items()):
    point = dict(family=family, generation_call=step, layer_index=17,
                 case_count=len(records),
                 variant_count=len({r['row_id'] for r in records}),
                 task_count=len({r['task'] for r in records}))
    assert point['task_count'] == 28
    for field in fields:
        variants = defaultdict(list)
        for r in records:
            value = float(r[field])
            assert r[field + '_status'] == 'measured' and math.isfinite(value)
            variants[(r['task'], r['row_id'])].append((r['noise_label'], value))
        tasks = defaultdict(list)
        for (task, variant), values in variants.items():
            assert len(values) == 2 and {n for n, _ in values} == {'0', '1'}
            tasks[task].append(sum(v for _, v in values) / 2)
        point[field] = sum(sum(v) / len(v) for v in tasks.values()) / len(tasks)
    profiles.append(point)

old = json.loads((ROOT / 'data/r11_result_sources.json').read_text())['attention_generation']
keys = {'O_shadow_P_full_key_TV': 'p_full_key_tv',
        'O_shadow_pre_gate_W__relative_l2': 'w_relative_l2'}
lookup = {(r['family'], r['generation_call']): r for r in profiles}
errors = [abs(float(r['equal_task_mean']) -
              lookup[(r['family'], int(r['generation_call']))][keys[r['metric']]])
          for r in old]
assert len(errors) == 24 and max(errors) < 1e-14
result = dict(
    source_path='data/PWH-JOIN-v1.csv.gz', source_sha256=EXPECTED_SHA,
    source_archive='print-supplement-v15/data/PWH-JOIN-v1.csv.gz; historical v7.30 author-review archive',
    support='joint404_derived: 404 pairs, 202 variants, 28 tasks',
    aggregation='mean noises within variant, variants within task, then equal task mean',
    relative_values='average original per-case relative values directly',
    original_profiles_reproduced=24, max_absolute_difference=max(errors),
    measurement_roles=dict(P='diagnostic replay full-key probability TV',
                           W='diagnostic replay projected attention output before gating',
                           H='native policy-trajectory post-block action-expert hidden state'),
    profiles=profiles,
    fidelity_scope='No native P/W fields; these data do not measure native-vs-replay P/W error.')
(ROOT / 'data/r11_attention_profiles.json').write_text(json.dumps(result, indent=2) + '\n')

names = [('Background Textures', 'Background'), ('Camera Viewpoints', 'Camera'),
         ('Light Conditions', 'Light'), ('Sensor Noise', 'Sensor')]
lines = [r'\begin{table*}[t]\centering\small',
         r'\begin{tabular}{lrrr}\toprule',
         r'Family & Replayed $W$: raw RMS & Native $H$: raw RMS & Native $H$: relative $L_2$\\',
         r' & $j=0\!\rightarrow9$ & $j=0\!\rightarrow9$ & $j=0\!\rightarrow9$\\\midrule']
for family, label in names:
    first, last = lookup[(family, 0)], lookup[(family, 9)]
    cells = [f'${first[key]:.3f}\\!\\rightarrow{last[key]:.3f}$'
             for key in ['w_delta_rms', 'ae_delta_rms', 'ae_relative_l2']]
    lines.append(label + ' & ' + ' & '.join(cells) + r'\\')
lines += [r'\bottomrule\end{tabular}',
          r'\caption{\textbf{Generation trends differ between the projected attention output and the native post-block state.} L18 of the first policy query on the same 404 matched LIBERO-Plus pairs: 96/102/96/110 pairs across Background/Camera/Light/Sensor, each on 28 tasks. Entries are task-equal means, averaging noises within settings and settings within tasks. Raw responses are per-case Frobenius norms divided by the square root of tensor size; relative responses divide by the corresponding clean tensor norm before averaging. $W$ is diagnostic replay before gating; $H$ is recorded natively after the block.}',
          r'\label{tab:attention-generation-sites}', r'\end{table*}']
(ROOT.parent / 'sections/attention_generation_sites_table.tex').write_text('\n'.join(lines) + '\n')
print('Aggregated original 404-pair attention profiles and native-H table; 24 existing points reproduced.')
