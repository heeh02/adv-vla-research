"""Format existing archived estimates; no model fitting or new statistics."""
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]
DATA = json.loads((ROOT / 'plot_sources/data/r11_evidence_tables.json').read_text())
FAMILIES = [('Background Textures', 'Background'), ('Camera Viewpoints', 'Camera'),
            ('Light Conditions', 'Light'), ('Sensor Noise', 'Sensor Noise')]
LAYERS = [(0, 1), (8, 9), (17, 18)]


def row_values(records):
    return [r['value'] for r in records]


affine = {(v['layer'], v['family'], v['model']): v['task_equal_mse']
          for v in row_values(DATA['affine_translation'])}
capacity = {(v['readout'], v['layer'], v['family'], v['model']): v['task_equal_mse']
            for v in row_values(DATA['capacity_translation'])}


def mse(value):
    return f'{1000 * value:.3f}'


def interval(pair):
    return '$[' + ','.join('0.0' if abs(x) < 0.05 else f'{x:+.1f}' for x in pair) + ']$'


def write(name, text):
    (ROOT / 'sections' / name).write_text(text.rstrip() + '\n')


lines = [r'\begin{table*}[ht]\centering\small',
         r'\setlength{\tabcolsep}{3.3pt}',
         r'\begin{tabular}{llrrrrrrrr}\toprule',
         r'Layer & Family & Response & Clean & Random & Full $H$ & First flow & No-change & \shortstack{Uncentered\\response} & \shortstack{Uncentered\\clean}\\\midrule']
for k, (index, label) in enumerate(LAYERS):
    if k:
        lines.append(r'\midrule')
    for family, display in FAMILIES:
        vals = [affine[index, family, model] for model in
                ['new_response', 'new_clean', 'new_random', 'full', 'flow', 'zero', 'old_response', 'old_clean']]
        lines.append(' & '.join([f'L{label}', display] + [mse(v) for v in vals]) + r'\\')
lines += [r'\bottomrule\end{tabular}',
          r'\caption{\textbf{Absolute translation error places the representation comparisons on a common scale.} Task-equal first-five translation MSE $\times10^3$ in squared adapter-command units; lower is better. All columns use the same 404 held-out cases on 28 tasks. Response, Clean and Random use position-centered rank-16 dictionaries; the last two columns use uncentered fits. Full $H$ uses the complete hidden response, and First flow uses the clean--perturbed difference in the initial predicted velocity. First flow and No-change are the same comparators at every layer. These point estimates complement the paired readout intervals in main Fig.~\ref*{M-fig:token-structure}b and Fig.~\ref{fig:full-readout-controls}.}',
          r'\label{tab:current-basis-control}', r'\end{table*}']
write('current_basis_control_table.tex', '\n'.join(lines))

lines = [r'\begin{table*}[t]\centering\small',
         r'\setlength{\tabcolsep}{5pt}',
         r'\begin{tabular}{llrrrrrrr}\toprule',
         r' & & \multicolumn{2}{c}{Affine} & \multicolumn{2}{c}{RFF ridge} & \multicolumn{2}{c}{MLP} & \\',
         r'Layer & Family & Response & Clean & Response & Clean & Response & Clean & No-change\\\midrule']
for k, (index, label) in enumerate(LAYERS):
    if k:
        lines.append(r'\midrule')
    for family, display in FAMILIES:
        vals = [capacity[readout, index, family, model]
                for readout in ['affine', 'rfridge', 'mlp'] for model in ['new_response', 'new_clean']]
        vals.append(affine[index, family, 'zero'])
        lines.append(' & '.join([f'L{label}', display] + [mse(v) for v in vals]) + r'\\')
lines += [r'\bottomrule\end{tabular}',
          r'\caption{\textbf{Estimator-dependent dictionary rankings accompany different absolute errors.} First-five translation MSE $\times10^3$, task-equal on the same 404 matched cases across 28 tasks; lower is better. Response and Clean use the same position-centered rank-16 dictionaries in every estimator. RFF ridge denotes random Fourier feature ridge readout. The No-change column is shared by all estimators. Table~\ref{tab:readout-capacity-full} gives the corresponding response-over-clean error reductions and original pointwise 95\% intervals, including rotation and gripper.}',
          r'\label{tab:readout-capacity}', r'\end{table*}']
write('readout_capacity_absolute_table.tex', '\n'.join(lines))

lines = [r'\begin{table*}[t]\centering\small',
         r'\setlength{\tabcolsep}{5pt}',
         r'\begin{tabular}{llrrrr}\toprule',
         r'Condition & Comparator & \shortstack{Planned-outcome\\bounds} & \shortstack{Available-case\\point} & 95\% interval & 99.44\% interval\\\midrule']
last_condition = None
for row in DATA['ablation_nine_contrasts_pp']:
    condition = row['condition']
    if last_condition is not None and condition != last_condition:
        lines.append(r'\midrule')
    display = {'clean': 'Clean', 'camera_plus15': 'Camera', 'headlight_off': 'Headlight'}[condition]
    comparator = {'D': 'Unedited ($D$)', 'Q': 'Clean-matched ($Q$)', 'R': 'Random mean ($R$)'}[row['code']]
    point = f"${row['available_case_point_pp']:+.1f}$"
    lines.append(' & '.join([display, comparator, interval(row['identification_bounds_pp']), point,
                             interval(row['available_case_pointwise95_pp']),
                             interval(row['available_case_bonferroni_interval_pp'])]) + r'\\')
    last_condition = condition
lines += [r'\bottomrule\end{tabular}',
          r'\caption{\textbf{Success contrasts distinguish the cost of an edit from its cost relative to other edits.} Every row is Response-large minus the named comparator, in percentage points. Planned-outcome bounds enumerate the 111 missing outcomes on the original 6,048-episode plan. Available-case points and intervals use the 5,937 known outcomes; the intervals are the archived post-hoc whole-task bootstrap results (16 tasks, 2,000 draws). The 99.44\% intervals apply Bonferroni adjustment across the nine contrasts. Missing-outcome bounds and sampling intervals describe different uncertainty sources. Random mean averages the three random bases.}',
          r'\label{tab:supp-native-contrasts}', r'\end{table*}']
write('native_contrasts_table.tex', '\n'.join(lines))
print('Formatted three tables from archived estimates and intervals.')
