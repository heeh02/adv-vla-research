"""Publication core and complete scalar support atlas, without fit or selection."""
from pathlib import Path
import csv
import gzip
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.text import Text
import audit as A

CORE = 'F3-NATIVE-PVP'
DIM_LABELS = ('Trans. x', 'Trans. y', 'Trans. z', 'Rot. x', 'Rot. y', 'Rot. z', 'Gripper')
STYLE = {'font.family': 'DejaVu Sans', 'font.size': 8, 'axes.titlesize': 9,
         'axes.labelsize': 8, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
         'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
         'savefig.facecolor': 'white', 'figure.facecolor': 'white'}
BOUNDARY = ('Historical CONF snapshots; 4 tasks × paired C/T parents × q11/16/22. '
            'T receiver; same G* N baseline; diagnostic noise.\n'
            'Candidate commands; no executed motion, equal-dose, pure-feedback or independent-holdout claim.')
CAPTION = (
    'Native continuation and candidate-command effects in all 24 archived CONF snapshots. '
    '(A) P (persistent j5-j9) and S (single pulse j5) share j5 and x6. The matrices give '
    'the number of snapshots with different post-WO/pre-gate outputs (upper) and different '
    'native Q/P storage (lower), out of 24, for P-S. Separate Q|P counts are printed if unequal. '
    'Post-WO/pre-gate output is not a gated residual write. Later differences mix renewed '
    'replacement and changed receivers. (B,C) All 24 snapshots in each of seven command '
    'dimensions are plotted: RMS(P-N)/RMS(S-N), separately for first 5 and all 10 candidate '
    'positions, on a common logarithmic axis. The dashed line is one. Deterministic vertical '
    'jitter encodes the fixed snapshot order, not uncertainty. Orange denotes strictly opposite '
    'signed WINDOW means of P-N and S-N, not per-position sign reversals; blue denotes same '
    'sign or a zero mean. All finite positive ratios are in view. Exact zero ratios use < and '
    'zero-denominator undefined ratios use x in the separate 0/NA lane, when present. '
    'N is the same G* baseline. C/T denote paired visited-state parent trajectories, not '
    'different receivers or independent initializations. Queries q11/16/22 correspond to '
    'policy steps 55/80/110; windows overlap. Archived diagnostic noise is not asserted '
    'to equal parent execution noise. This is historical descriptive evidence, not new '
    'holdout, equal-dose timing, pure feedback, physical movement or task success.'
)


def number(value):
    if value is None or not math.isfinite(value):
        return 'NA'
    if value == 0:
        return '0'
    return f'{value:.2g}' if abs(value) < .01 or abs(value) >= 100 else f'{value:.2f}'


def label(snapshot):
    task, _, parent, q = snapshot
    return f'{task[-3:]} {parent} q{q}'


def cells(data, window):
    idx = {A.snapshot(r)+(r['window'], r['command_dimension']): r
           for r in data['command_cells']}
    return [[idx[s+(window, d)] for d in A.DIMENSIONS] for s in A.SNAPSHOTS]



def layout_check(fig, name, *, min_font=7):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    problems = []
    actual_min = float('inf')
    for text in fig.findobj(Text):
        if not text.get_visible() or not text.get_text():
            continue
        actual_min = min(actual_min, text.get_fontsize())
        b = text.get_window_extent(renderer)
        if b.x0 < -1 or b.y0 < -1 or b.x1 > fig.bbox.width+1 or b.y1 > fig.bbox.height+1:
            problems.append(text.get_text()[:100])
    for ax in fig.axes:
        b = ax.get_window_extent(renderer)
        if b.x0 < -1 or b.y0 < -1 or b.x1 > fig.bbox.width+1 or b.y1 > fig.bbox.height+1:
            problems.append('axes outside canvas')
    A.need(not problems and actual_min >= min_font, 'readable_layout_' + name)
    return {'name': name, 'width_inches': fig.get_figwidth(),
            'height_inches': fig.get_figheight(), 'minimum_font_points': actual_min,
            'text_and_axes_inside_canvas': True, 'problems': problems}


def save(fig, out, name, layouts):
    layouts.append(layout_check(fig, name, min_font=7.5 if name == CORE else 7))
    for ext in ('pdf', 'svg', 'png'):
        fig.savefig(Path(out)/(name+'.'+ext), dpi=300)
    plt.close(fig)


def core(data, out):
    """180 mm paper panel; all states visible, identities jittered without selection."""
    layouts = []
    values = [c['RMS_ratio'] for c in data['command_cells']
              if c['ratio_defined'] and c['RMS_ratio'] > 0]
    logs = [math.log2(v) for v in values]
    lo = min(-1., math.floor(min(logs))) if logs else -1.
    hi = max(1., math.ceil(max(logs))) if logs else 1.
    # A separate, labelled lane retains exact zero and undefined ratios.
    lane = lo-.18*(hi-lo)
    with plt.rc_context({**STYLE, 'font.size': 7.5, 'axes.titlesize': 8,
                         'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5,
                         'pdf.use14corefonts': False}):
        fig = plt.figure(figsize=(180/25.4, 80/25.4))
        if data['synthetic']:
            fig.text(.5, .99, 'SYNTHETIC', ha='center', va='top', fontsize=8)
        idx = {(c['flow_index'], c['layer_label']): c for c in data['separation_counts']}
        matrices = fig.add_gridspec(2, 1, left=.065, right=.325, bottom=.20, top=.85, hspace=.66)
        for mi, kind in enumerate(('output', 'reads')):
            ax = fig.add_subplot(matrices[mi, 0])
            vals = []
            for layer in A.LAYERS:
                vals.append([24-idx[j, layer][A.FIELDS[1]+'_zero'] if kind == 'output'
                             else max(24-idx[j, layer][A.EXACT[0]], 24-idx[j, layer][A.EXACT[1]])
                             for j in A.FLOWS])
            ax.imshow(vals, cmap='Blues', vmin=0, vmax=24, aspect='auto')
            for li, layer in enumerate(A.LAYERS):
                for ji, j in enumerate(A.FLOWS):
                    q = 24-idx[j, layer][A.EXACT[0]]
                    p = 24-idx[j, layer][A.EXACT[1]]
                    label_ = f'{vals[li][ji]}/24' if kind == 'output' or q == p else f'{q}|{p}/24'
                    ax.text(ji, li, label_, ha='center', va='center', fontsize=7.5,
                            color='white' if vals[li][ji] > 12 else '#151515')
            ax.set_yticks(range(3), [f'L{l}' for l in A.LAYERS])
            ax.set_xticks(range(5), [f'j{j}' for j in A.FLOWS])
            ax.tick_params(length=0)
            ax.set_title('Projected read differs' if kind == 'output'
                         else 'Q / P differ', fontsize=7.5, pad=4)
        fig.text(.025, .935, 'A  Native separation', fontsize=8.5, weight='bold')
        fig.text(.065, .075, 'P-S contrast; same j5 and x6', fontsize=7.5)
        positive_color, opposite_color = '#2671A3', '#C35328'
        counts = []
        for wi, window in enumerate(A.WINDOWS):
            ax = fig.add_axes([.425+wi*.30, .20, .245, .65])
            counts.append(data['command_counts'][wi])
            for si, s in enumerate(A.SNAPSHOTS):
                jitter = (si-11.5)*.024
                for di, c in enumerate(cells(data, window)[si]):
                    r = c['RMS_ratio']
                    x = math.log2(r) if r is not None and r > 0 else lane
                    marker = 'o' if r is not None and r > 0 else ('<' if r == 0 else 'x')
                    ax.scatter(x, di+jitter, s=9, marker=marker, linewidths=.6,
                               color=opposite_color if c['signed_window_means_opposite'] else positive_color,
                               alpha=.85, zorder=3)
            ax.axvline(0, color='#666666', lw=.7, ls='--')
            ax.set_xlim(lane-.06*(hi-lo), hi+.06*(hi-lo)); ax.set_ylim(6.55, -.55)
            ticks = list(range(int(lo), int(hi)+1, max(1, int(math.ceil((hi-lo)/4)))))
            if 0 not in ticks: ticks.append(0)
            ticks = sorted(set(ticks))
            ax.set_xticks(ticks, [f'{2.**v:g}' for v in ticks])
            ax.set_yticks(range(7), ('Tx', 'Ty', 'Tz', 'Rx', 'Ry', 'Rz', 'Grip'))
            ax.grid(axis='y', color='#dddddd', lw=.4)
            ax.set_axisbelow(True)
            ax.spines[['top', 'right']].set_visible(False)
            ax.tick_params(length=2)
            if any(c['RMS_ratio'] is None or c['RMS_ratio'] == 0
                   for row in cells(data, window) for c in row):
                ax.text(lane, -.85, '0/NA', ha='center', fontsize=7.5)
            fig.text(.385+wi*.30, .935, f"{'BC'[wi]}  {'First 5' if wi == 0 else 'Full 10'} positions",
                     fontsize=8.5, weight='bold')
            ax.set_xlabel('RMS(P-N) / RMS(S-N)', fontsize=7.5, labelpad=4)
        handles = [
            plt.Line2D([], [], marker='o', linestyle='none', markersize=3.5,
                       color=positive_color, label='Same / zero mean sign'),
            plt.Line2D([], [], marker='o', linestyle='none', markersize=3.5,
                       color=opposite_color, label='Opposite mean signs'),
        ]
        fig.legend(handles=handles, loc='lower left', bbox_to_anchor=(.43, .005),
                   frameon=False, fontsize=7.5, handletextpad=.5, labelspacing=.3,
                   borderaxespad=0)
        save(fig, out, CORE, layouts)
    return layouts

def table_page(pdf, data, out, page, title, rows, columns, values, *, footer,
               width=14., height=10.5, header=''):
    """All displayed numeric values are annotated, without value-based selection."""
    fig, ax = plt.subplots(figsize=(width, height))
    fig.subplots_adjust(left=.15, right=.98, bottom=.12, top=.86)
    prefix = 'SYNTHETIC — ' if data['synthetic'] else ''
    fig.text(.03, .975, prefix+title, fontsize=12, weight='bold', va='top')
    fig.text(.03, .93, header, fontsize=8, va='top')
    ax.axis('off')
    tab = ax.table(cellText=values, rowLabels=rows, colLabels=columns,
                   cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
    tab.auto_set_font_size(False); tab.set_fontsize(8)
    for (r, c), cell in tab.get_celld().items():
        cell.set_edgecolor('#dddddd'); cell.set_linewidth(.4)
        if r == 0: cell.set_facecolor('#ebebeb')
    fig.text(.03, .07, footer, fontsize=8, va='top', linespacing=1.5)
    layout = layout_check(fig, f'atlas-{page}')
    layout.update(page=page, title=title, rows=len(rows), columns=len(columns),
                  printed_numeric_cells=len(rows)*len(columns))
    pdf.savefig(fig)
    # Every page is also available as SVG/PNG for direct visual QA and reuse.
    for ext in ('svg', 'png'):
        fig.savefig(Path(out)/f'ATLAS-{page:02d}.{ext}', dpi=100)
    plt.close(fig)
    return layout


def atlas(data, out, check):
    """Eight parent-root pages plus one global native-equality count page."""
    layouts = []
    columns = ['S-N RMS', 'P-N RMS', 'P/S RMS', 'S-N mean', 'P-N mean']
    metric_keys = ('pulse_RMS', 'persistent_RMS', 'RMS_ratio',
                   'pulse_mean_signed', 'persistent_mean_signed')
    with plt.rc_context(STYLE), PdfPages(Path(out)/'SUPPORT-ATLAS.pdf') as pdf:
        for page, root in enumerate(A.ROOTS, 1):
            check()
            snapshots = [s for s in A.SNAPSHOTS if s[:3] == root]
            fig = plt.figure(figsize=(17, 20))
            prefix = 'SYNTHETIC - ' if data['synthetic'] else ''
            fig.text(.03, .98, prefix+root[1], fontsize=14, weight='bold', va='top')
            fig.text(.03, .958, 'All q11/16/22; full scalar support. Values rounded here; lossless precision is in the CSV / JSON.', fontsize=9)
            for wi, window in enumerate(A.WINDOWS):
                ax = fig.add_axes([.10+wi*.47, .655, .39, .265]); ax.axis('off')
                rows = [c for c in data['command_cells'] if A.snapshot(c) in snapshots and c['window'] == window]
                values = [[number(c[k])+(' *' if k == 'RMS_ratio' and c['signed_window_means_opposite'] else '')
                           for k in metric_keys] for c in rows]
                tab = ax.table(cellText=values, rowLabels=[f"q{c['query_index']} {DIM_LABELS[A.DIMENSIONS.index(c['command_dimension'])]}" for c in rows],
                               colLabels=columns, cellLoc='center', bbox=[0, 0, 1, 1])
                tab.auto_set_font_size(False); tab.set_fontsize(8)
                for (r, c), cell in tab.get_celld().items():
                    cell.set_edgecolor('#dddddd'); cell.set_linewidth(.4)
                    if r == 0: cell.set_facecolor('#ebebeb')
                ax.set_title(window+' candidate commands', fontsize=10, pad=10)
            fig.text(.03, .632, '* Opposite signed WINDOW means; 0 is exact zero; NA is a zero pulse RMS denominator. No pooled command units.', fontsize=8)
            grid = fig.add_gridspec(2, 4, left=.07, right=.98, bottom=.30, top=.60, hspace=.30, wspace=.30)
            row_keys = [(s, l) for s in snapshots for l in A.LAYERS]
            for wi, window in enumerate(A.WINDOWS):
                for fi, field in enumerate(A.FIELDS):
                    ax = fig.add_subplot(grid[wi, fi]); ax.axis('off')
                    vals = [[number(data['mechanism'][s+('persistent_minus_pulse', field, j, l, window)]['RMS_difference'])
                             for j in A.FLOWS] for s, l in row_keys]
                    tab = ax.table(cellText=vals, rowLabels=[f'q{s[3]} L{l}' for s, l in row_keys],
                                   colLabels=[f'j{j}' for j in A.FLOWS], cellLoc='center', bbox=[0, 0, 1, 1])
                    tab.auto_set_font_size(False); tab.set_fontsize(8)
                    for cell in tab.get_celld().values(): cell.set_edgecolor('#dddddd'); cell.set_linewidth(.4)
                    short = ('Input residual', 'Post-WO / pre-gate', 'After attn. residual', 'Post-block')[fi]
                    ax.set_title(window+' | '+short+' RMS', fontsize=9, pad=10)
            # All three contrasts and both native flow variables, three queries each.
            for fi, field in enumerate(('x', 'velocity_native')):
                ax = fig.add_axes([.08+fi*.32, .11, .27, .14])
                for ci, contrast in enumerate(A.CONTRASTS):
                    for qi, s in enumerate(snapshots):
                        js = list(range(11 if field == 'x' else 10))
                        ys = [data['flow'][s+(contrast, field, j, 'full10')]['RMS_difference'] for j in js]
                        ax.plot(js, ys, color=('#2671A3', '#C35328', '#23836B')[ci],
                                ls=('-', '--', ':')[qi], lw=.9)
                ax.set_title('Native '+field+' | all 3 contrasts', fontsize=9)
                ax.set_xlabel('State k' if field == 'x' else 'Flow call j', fontsize=8)
                ax.set_ylabel('Full10 RMS', fontsize=8); ax.grid(lw=.3, color='#dddddd')
            ax = fig.add_axes([.75, .11, .23, .14]); ax.axis('off')
            vals = [[''.join('=' if data['reading'][s+('persistent_minus_pulse', j, l)][k] else '!' for k in A.EXACT)
                     for j in A.FLOWS] for s, l in row_keys]
            tab = ax.table(cellText=vals, rowLabels=[f'q{s[3]} L{l}' for s, l in row_keys],
                           colLabels=[f'j{j}' for j in A.FLOWS], cellLoc='center', bbox=[0, 0, 1, 1])
            tab.auto_set_font_size(False); tab.set_fontsize(8)
            for cell in tab.get_celld().values(): cell.set_edgecolor('#dddddd'); cell.set_linewidth(.4)
            ax.set_title('P-S typed equality: Q,P,gate,K,V', fontsize=9, pad=10)
            fig.text(.03, .062, 'Flow colors: blue S-N; orange P-N; green P-S. Line styles: solid q11, dashed q16, dotted q22. No fit.\n'
                     'Equality strings: = exact, ! different, in Q/P/gate/suffix-K/suffix-V order. Neural summaries are P-S; all 3 contrasts remain in cached CSV/JSON.\n'+BOUNDARY,
                     fontsize=8, va='top', linespacing=1.6)
            name = f'ATLAS-{page:02d}'; info = layout_check(fig, name)
            info.update(page=page, root_id=root[1], snapshots=[list(s) for s in snapshots],
                        command_cells=42, neural_RMS_cells=360, exact_flags=225, flow_traces=18)
            layouts.append(info); pdf.savefig(fig)
            for ext in ('svg', 'png'): fig.savefig(Path(out)/(name+'.'+ext), dpi=100)
            plt.close(fig)
        rows = []
        vals = []
        for key in A.EXACT:
            for l in A.LAYERS:
                rows.append(key.replace('_typed_exact', '')+f' L{l}')
                vals.append([str(next(c for c in data['separation_counts'] if c['flow_index'] == j and c['layer_label'] == l)[key])+'/24'
                             for j in A.FLOWS])
        for field in A.FIELDS:
            for l in A.LAYERS:
                rows.append(field+f' L{l}')
                vals.append([str(next(c for c in data['separation_counts'] if c['flow_index'] == j and c['layer_label'] == l)[field+'_zero'])+'/24'
                             for j in A.FLOWS])
        layouts.append(table_page(pdf, data, out, 9, 'All 24 snapshots | P-S equality and zero-difference counts',
            rows, [f'j{j}' for j in A.FLOWS], vals,
            header='First 15 rows: native typed exact equality. Last 12 rows: decoded-coordinate zero difference (full10).',
            footer='All snapshots retained; Q/P equality is not a magnitude threshold. Full precision, counts, energies and separate windows in JSON.\n'+BOUNDARY,
            width=18, height=13))
    A.need(len(layouts) == 9, 'all9_atlas_pages')
    return layouts

def write_support(data, out):
    path = Path(out)/'FULL-COMMAND-SUPPORT.csv.gz'
    with gzip.open(path, 'wt', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(data['command_cells'][0]))
        writer.writeheader(); writer.writerows(data['command_cells'])
    return len(data['command_cells'])


def render(data, out, check=lambda: None):
    layouts = core(data, out); check()
    pages = atlas(data, out, check)
    count = write_support(data, out)
    return {'core_layouts': layouts, 'atlas_pages': pages, 'command_support_rows': count,
            'captions': {CORE: CAPTION},
            'main_values': {'separation_counts': data['separation_counts'],
                            'command_counts': data['command_counts'],
                            'command_cells': data['command_cells'],
                            'ratio_axis': 'common log2; all finite positive ratios within range; zero and NA separate lane'}}
