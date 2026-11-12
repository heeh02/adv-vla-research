"""Redraw native-test and readout figures plus retained behavioral support.

Run: python plot_sources/build_results.py [basis|cfr|function_tests|behavior|attention_support|sensor_case_view ...]
No fitting, bootstrap, model execution, or case selection is performed here.
Original coordinates come from data/figure_data.json. Restored comparisons and
aggregation display points come from data/restored_evidence.json; the companion
collector documents their archived sources and verifies both rank correlations.
"""
from pathlib import Path
import json
import os
import sys
import tempfile

import numpy as np
os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'advvla-results-mpl'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from matplotlib.ticker import ScalarFormatter, NullLocator


ROOT = Path(__file__).resolve().parent
OUT = ROOT.parent / 'figures'
PREVIEW = ROOT / 'previews'
OUT.mkdir(exist_ok=True)
PREVIEW.mkdir(exist_ok=True)
D = json.loads((ROOT / 'data/figure_data.json').read_text())


def restored():
    return json.loads((ROOT / 'data/restored_evidence.json').read_text())

FAMILIES = ['Background Textures', 'Camera Viewpoints', 'Light Conditions', 'Sensor Noise']
SHORT = dict(zip(FAMILIES, ['Background', 'Camera', 'Light', 'Sensor']))
COLORS = dict(zip(FAMILIES, ['#188B80', '#2968B2', '#B57B16', '#92519B']))
INK = '#243247'
MUTED = '#647383'
GREY = '#9AA6B2'
GRID = '#E8EDF1'
LINE = '#BFC9D2'
GEOMETRY = '#007F86'
ATTENTION = '#7C5CBF'
BEHAVIOR = '#C36435'
LAYER_COLORS = ['#647B96', '#C06642']


def configure():
    plt.rcParams.update({
        'font.family': 'DejaVu Sans', 'font.size': 8.0,
        'axes.labelsize': 8.0, 'axes.titlesize': 9.0,
        'axes.titleweight': 'bold', 'xtick.labelsize': 7.8,
        'ytick.labelsize': 7.8, 'legend.fontsize': 7.8,
        'text.color': INK, 'axes.labelcolor': INK,
        'axes.edgecolor': LINE, 'axes.linewidth': .6,
        'xtick.color': INK, 'ytick.color': INK,
        'xtick.major.size': 2.5, 'ytick.major.size': 0,
        'lines.linewidth': 1.35, 'pdf.fonttype': 42,
        'ps.fonttype': 42, 'svg.fonttype': 'none',
        'svg.hashsalt': 'advvla-results-20261029',
        'figure.facecolor': 'white', 'axes.facecolor': 'white',
        'savefig.facecolor': 'white',
    })


def select(rows, **criteria):
    values = [r for r in rows if all(r[k] == v for k, v in criteria.items())]
    assert len(values) == 1, (criteria, len(values))
    return values[0]


def save(fig, name):
    temporary = OUT / f'{name}.results.tmp.pdf'
    fig.savefig(temporary, metadata={'Title': name, 'Author': 'Anonymous Authors',
                                    'CreationDate': None, 'ModDate': None})
    assert temporary.stat().st_size > 1000
    temporary.replace(OUT / f'{name}.pdf')
    fig.savefig(OUT / f'{name}.svg', metadata={'Date': None, 'Creator': 'Anonymous Authors'})
    fig.savefig(PREVIEW / f'{name}.png', dpi=200)
    plt.close(fig)


def style(ax, grid='y', left=False):
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines['left'].set_visible(left)
    if grid:
        ax.grid(axis=grid, color=GRID, linewidth=.65)
    ax.set_axisbelow(True)


def heading(fig, x, y, letter, title, color, fontsize=9.0):
    fig.text(x, y, letter, color=color, fontsize=10.2, fontweight='bold', va='center')
    fig.text(x + .027, y, title, color=INK, fontsize=fontsize, fontweight='bold', va='center')


def interval(ax, value, low, high, y, color, marker='o', hollow=False, ms=4.4):
    ax.plot([low, high], [y, y], color=color, lw=1.3, solid_capstyle='round', zorder=3)
    for x in [low, high]:
        ax.plot([x, x], [y-.075, y+.075], color=color, lw=.8, zorder=3)
    ax.plot(value, y, marker=marker, ms=ms, ls='', color=color,
            markerfacecolor='white' if hollow else color, markeredgewidth=.95, zorder=4)


def family_legend(fig, y, anchor=.51, ncol=4):
    handles = [Line2D([], [], color=COLORS[f], marker='o', ms=4,
                      lw=1.5, label=SHORT[f]) for f in FAMILIES]
    return fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(anchor, y),
                      ncol=ncol, frameon=False, columnspacing=1.7,
                      handlelength=1.5, handletextpad=.5)


def basis():
    configure()
    evidence = restored()
    fig = plt.figure(figsize=(7, 3.98))
    boxes = [(.066, .626, .350, .263), (.640, .626, .347, .263),
             (.066, .198, .350, .263)]
    a, b, c = [fig.add_axes(box) for box in boxes]
    d = fig.add_axes([.640, .198, .347, .263])
    headings = [
        ('a', 'Shared geometry', 'Residual energy retained (%)', .974, .925),
        ('b', 'Command readout', 'Translation MSE / no change (log)', .974, .925),
        ('c', 'Response over clean', 'Translation-error reduction (%)', .548, .498),
    ]
    for ax, (letter, title, subtitle, title_y, sub_y) in zip([a, b, c], headings):
        p = ax.get_position()
        heading(fig, p.x0, title_y, letter, title, GEOMETRY)
        fig.text(p.x0, sub_y, subtitle, fontsize=7.9, color=MUTED)
        ax.set_xticks([0, 1, 2], ['L1', 'L9', 'L18'])
        ax.set_xlim(-.17, 2.17)
        style(ax, 'y')
    a.set_ylim(0, 95)
    a.set_yticks([0, 25, 50, 75])
    b.set_yscale('log')
    b.set_ylim(.16, 8.8)
    b.set_yticks([.2, .5, 1, 2, 5])
    b.yaxis.set_major_formatter(ScalarFormatter())
    b.yaxis.set_minor_locator(NullLocator())
    b.axhspan(.16, 1, color='#F1F7F6', zorder=0)
    b.axhline(1, color=MUTED, lw=.9, ls=(0, (3, 2)), zorder=2)
    b.text(.04, .07, 'No change = 1', transform=b.transAxes, fontsize=7.3,
           color=MUTED, va='bottom')
    c.set_ylim(-62, 90)
    c.set_yticks([-50, 0, 50])
    c.axhline(0, color=MUTED, lw=.9, ls=(0, (3, 2)), zorder=2)

    # Small symmetric horizontal offsets keep every task interval visible.
    # Layer labels remain categorical; connecting segments are visual guides.
    offsets = [-.075, -.025, .025, .075]
    for f, off in zip(FAMILIES, offsets):
        col = COLORS[f]
        xx = np.array([0, 1, 2]) + off
        for model, marker, ls, alpha, lw in [
                ('new_response', 'o', '-', 1, 1.55),
                ('new_clean', 'D', (0, (2, 2)), .72, .9),
                ('new_random', 'x', '', .65, .8)]:
            values = [select(D['basis']['geometry'], family=f, layer=l, model=model)['retention']
                      for l in [0, 8, 17]]
            yy = np.array([r['mean'] * 100 for r in values])
            lo = np.array([r['interval_95'][0] * 100 for r in values])
            hi = np.array([r['interval_95'][1] * 100 for r in values])
            color = GREY if model == 'new_random' else col
            a.errorbar(xx, yy, yerr=[yy-lo, hi-yy], color=color, fmt=marker, ls=ls,
                       lw=lw, ms=4 if model == 'new_response' else 3.4,
                       mfc='white' if model == 'new_clean' else color,
                       mew=.8, ecolor=color, elinewidth=.8, capsize=1.6,
                       alpha=alpha, zorder=4 if model == 'new_response' else 3)
            if model == 'new_random':
                continue
            rr = [select(D['basis']['readouts'], family=f, layer=l, model=model)
                  for l in [0, 8, 17]]
            ratios = [r['task_equal_mse']/r['task_equal_target_energy'] for r in rr]
            b.plot(xx, ratios, color=col, marker=marker, ls=ls,
                   lw=lw, ms=4 if model == 'new_response' else 3.4,
                   mfc='white' if model == 'new_clean' else col,
                   mew=.8, alpha=alpha, zorder=4 if model == 'new_response' else 3)

        rr = [select(D['basis']['contrasts'], family=f, layer=l) for l in [0, 8, 17]]
        yy = np.array([r['relative_error_reduction']*100 for r in rr])
        lo = np.array([r['relative_error_reduction_interval_95'][0]*100 for r in rr])
        hi = np.array([r['relative_error_reduction_interval_95'][1]*100 for r in rr])
        c.errorbar(xx, yy, yerr=[yy-lo, hi-yy], color=col, fmt='o', ls='-',
                   lw=1.3, ms=4, ecolor=col, elinewidth=.8, capsize=1.6, zorder=4)

    c.set_xlabel('Action-expert layer', labelpad=2)
    method_handles = [
        Line2D([], [], color=INK, marker='o', ms=4, lw=1.5, label='Response'),
        Line2D([], [], color=INK, marker='D', mfc='white', ms=3.7,
               ls=(0, (2, 2)), lw=.9, label='Centered clean'),
        Line2D([], [], color=GREY, marker='x', ls='', ms=4, label='Random (a)'),
    ]
    fig.legend(handles=method_handles, loc='center', bbox_to_anchor=(.507, .760),
               ncol=1, frameon=False, handlelength=1.3, columnspacing=1.0,
               handletextpad=.45, fontsize=7.4, labelspacing=.75,
               title='Basis (a,b)', title_fontsize=7.5)

    # Original paired contrasts against the full-hidden and first-flow readouts.
    # Both coordinates and intervals are read directly from expected.json.
    heading(fig, .640, .548, 'd', 'Stronger readouts at L18', GEOMETRY)
    fig.text(.640, .498, 'Translation-error reduction (%)', fontsize=7.8, color=MUTED)
    for i, fam in enumerate(FAMILIES):
        for offset, cand, base, marker, hollow in [
            (-.15, 'new_response', 'full', 'o', False),
            (.15, 'flow_plus_new', 'flow', '^', True),
        ]:
            r = select(evidence['strongest_readouts']['contrasts'], family=fam,
                       candidate=cand, baseline=base)
            interval(d, 100*r['relative_error_reduction'],
                     100*r['relative_error_reduction_interval_95'][0],
                     100*r['relative_error_reduction_interval_95'][1],
                     i+offset, COLORS[fam], marker=marker, hollow=hollow, ms=4.0)
    d.set_ylim(3.55, -.55)
    d.set_yticks(range(4), [SHORT[f] for f in FAMILIES])
    d.set_xlim(-104, 38); d.set_xticks([-100, -50, 0])
    d.axvline(0, color=MUTED, lw=.9, ls=(0, (3, 2)))
    style(d, 'x')
    comparison_handles = [
        Line2D([], [], color=INK, marker='o', ls='', ms=4,
               label='Response vs. full hidden response'),
        Line2D([], [], color=INK, marker='^', mfc='white', ls='', ms=4,
               label='First-flow + response vs. first-flow'),
    ]
    fig.legend(handles=comparison_handles, loc='lower right', bbox_to_anchor=(.985, .047),
               frameon=False, ncol=1, fontsize=7.4, handletextpad=.45,
               handlelength=1.0, labelspacing=.55)
    family_legend(fig, -.003, anchor=.37)
    save(fig, 'same_basis_translation')


def cfr():
    configure()
    absolute = json.loads((ROOT / 'data/cfr_arm_means.json').read_text())
    fig = plt.figure(figsize=(7, 3.70))
    # The operation is common to the absolute means and paired endpoint contrasts.
    top = fig.add_axes([.01, .890, .98, .105])
    top.set_xlim(0, 1); top.set_ylim(0, 1); top.axis('off')
    blocks = [
        (.005, .255, 'Same-state clean reference', r'$H_C$ and perturbed $H_T$'),
        (.294, .412, 'L9 · rank-16 natural projection', r"$H'=H_T-(H_T-H_C)BB^\top$"),
        (.740, .255, 'Native continuation', 'All 10 generation calls'),
    ]
    for x, w, title, body in blocks:
        top.add_patch(FancyBboxPatch((x, .05), w, .91,
                      boxstyle='round,pad=0.005,rounding_size=0.045',
                      facecolor='#F2F8F8', edgecolor='#B8D6D5', lw=.7))
        top.text(x+w/2, .68, title, ha='center', va='center', fontsize=7.7,
                 fontweight='bold', color=GEOMETRY)
        top.text(x+w/2, .29, body, ha='center', va='center', fontsize=8.0)
    for x1, x2 in [(.265, .290), (.711, .736)]:
        top.add_patch(FancyArrowPatch((x1, .5), (x2, .5), arrowstyle='-|>',
                                     mutation_scale=8, lw=.8, color=GEOMETRY))

    heading(fig, .013, .850, 'a', 'Absolute first-query translation error', GEOMETRY)
    fig.text(.655, .845, 'Task-equal means · adapter units', fontsize=7.6, color=MUTED)
    arms = ['N_C', 'N_T', 'R_resp', 'R_clean', 'R_rand', 'R_full']
    arm_labels = [r'Native clean $N_C$', r'Unedited $N_T$', r'Response $R_{\rm resp}$',
                  r'Clean basis $R_{\rm clean}$', r'Random $R_{\rm rand}$', r'Full-site $R_{\rm full}$']
    for left, fam in [(.170, 'Camera Viewpoints'), (.670, 'Sensor Noise')]:
        ax = fig.add_axes([left, .535, .285, .235])
        ratio = select(absolute['descriptive_ratios'], family=fam)['relative_total_error_reduction']
        label = SHORT[fam] + r' · response vs. $N_T$: ' + f'−{100*ratio:.1f}%'
        fig.text(left, .802, label, color=COLORS[fam], fontsize=7.8, fontweight='bold')
        for y, arm in enumerate(arms):
            r = select(absolute['means'], family=fam, arm=arm)
            value = r['mean_error']
            if arm == 'R_resp':
                face, edge, lw = COLORS[fam], COLORS[fam], .7
            elif arm == 'R_full':
                face, edge, lw = 'white', COLORS[fam], 1.05
            else:
                face, edge, lw = '#D7DFE6', '#D7DFE6', .5
            ax.barh(y, value, height=.54, color=face, edgecolor=edge, linewidth=lw, zorder=3)
            if arm == 'N_C':
                ax.plot(0, y, marker='o', mfc='white', mec=MUTED, ms=3.5, mew=.8, zorder=4)
            ax.text(value+.012, y, f'{value:.4f}', ha='left', va='center', fontsize=7.4,
                    color=COLORS[fam] if arm in ['R_resp', 'R_full'] else INK,
                    fontweight='bold' if arm == 'R_resp' else 'normal')
        ax.set_ylim(5.6, -.6)
        ax.set_xlim(-.015, .55)
        ax.set_xticks([0, .2, .4])
        ax.set_yticks(range(6), arm_labels)
        ax.tick_params(axis='y', labelsize=7.5, pad=3)
        style(ax, 'x')

    # Original six planned contrasts remain complete for each endpoint. Their
    # six-comparison simultaneous intervals are not recomputed or reinterpreted.
    b = fig.add_axes([.225, .115, .365, .255])
    c = fig.add_axes([.682, .115, .307, .255], sharey=b)
    heading(fig, .225, .447, 'b', 'Command contrasts', GEOMETRY)
    heading(fig, .682, .447, 'c', 'Task-success contrasts', GEOMETRY)
    fig.text(.225, .404, r'Key secondary · error reduction / frozen $S_f$', fontsize=7.5, color=MUTED)
    fig.text(.682, .404, 'Primary · success difference (pp)', fontsize=7.5, color=MUTED)
    rows = [(f, rhs) for f in ['Camera Viewpoints', 'Sensor Noise']
            for rhs in ['N_T', 'R_clean', 'R_rand']]
    yy = [0, 1, 2, 3.5, 4.5, 5.5]
    names = {'N_T': 'vs. unedited', 'R_clean': 'vs. clean basis', 'R_rand': 'vs. random basis'}
    for y, (fam, rhs) in zip(yy, rows):
        col = COLORS[fam]
        for ax, endpoint, scale in [(b, 'q0', 1), (c, 'success', 100)]:
            r = select(D['cfr'][endpoint], family=fam, rhs=rhs)
            interval(ax, r['point']*scale, r['sampling_interval'][0]*scale,
                     r['sampling_interval'][1]*scale, y, col, ms=4.0)
    for ax in [b, c]:
        ax.set_ylim(6.1, -.6)
        style(ax, 'x')
        ax.axhline(2.75, color=GRID, lw=.8)
        ax.axvline(0, color=MUTED, lw=.9, ls=(0, (3, 2)))
        ax.set_yticks(yy)
    b.set_yticklabels([names[r] for _, r in rows], fontsize=7.6)
    c.tick_params(labelleft=False)
    b.set_xlim(-.025, .535); b.set_xticks([0, .1, .2, .3, .4, .5])
    c.set_xlim(-15, 15); c.set_xticks([-10, 0, 10])
    for y, label, fam in [(1, 'Camera\n27 tasks', 'Camera Viewpoints'),
                          (4.5, 'Sensor\n30 tasks', 'Sensor Noise')]:
        yfig = fig.transFigure.inverted().transform(b.transData.transform((0, y)))[1]
        fig.text(.013, yfig, label, fontsize=7.8, fontweight='bold', color=COLORS[fam],
                 va='center', linespacing=1.35)
    fig.text(.505, .024, '(b,c) Positive favors response restoration · simultaneous 95% task intervals by endpoint',
             ha='center', fontsize=7.5, color=MUTED)
    save(fig, 'r13_cfr')


def attention():
    configure()
    fig = plt.figure(figsize=(7, 2.65))
    # Separate scale-bearing subplots prevent conditional % from being conflated
    # with percentage points of the full-key probability allocation.
    fig.text(.013, .948, 'FIXED STATE', color=ATTENTION, fontsize=7.5, fontweight='bold')
    fig.text(.389, .948, 'DIAGNOSTIC REPLAY · BRANCH-SPECIFIC STATES',
             color=ATTENTION, fontsize=7.5, fontweight='bold')
    heading(fig, .013, .892, 'a', 'Regional ranking', ATTENTION)
    heading(fig, .401, .892, 'b', 'Probability difference', ATTENTION, fontsize=8.7)
    heading(fig, .728, .892, 'c', 'Projected output', ATTENTION, fontsize=8.7)
    fig.text(.013, .818, 'Same wrist pixels · 36 tasks', color=MUTED, fontsize=7.7)
    fig.text(.401, .818, 'Layer 18 · full-key TV (%)', color=MUTED, fontsize=7.7)
    fig.text(.728, .818, r'Layer 18 · relative $W$ response', color=MUTED, fontsize=7.7)
    a1 = fig.add_axes([.035, .291, .121, .435])
    a2 = fig.add_axes([.227, .291, .121, .435])
    values = [D['wrist']['conditional_tv_percent'], D['wrist']['regional_l1_pp']]
    for ax, vals, ylim, ticks in [(a1, values[0], 32, [0, 15, 30]),
                                  (a2, values[1], 16, [0, 5, 10, 15])]:
        ax.plot([0, 1], vals, color='#ACB6C3', lw=1.7, zorder=2)
        for x, val, col in zip([0, 1], vals, LAYER_COLORS):
            ax.plot(x, val, 'o', color=col, ms=5.3, zorder=4)
            ax.text(x, val + ylim*.075, f'{val:.1f}', ha='center', va='bottom',
                    fontsize=8.0, fontweight='bold', color=col)
        ax.set_xlim(-.25, 1.25); ax.set_ylim(0, ylim)
        ax.set_xticks([0, 1], ['L9', 'L18']); ax.set_yticks(ticks)
        ax.tick_params(axis='y', labelsize=7.0, pad=2)
        style(ax, 'y')
    fig.text(.096, .192, 'Conditional\nTV (%)', ha='center', va='top', fontsize=7.9, linespacing=1.3)
    fig.text(.288, .192, 'Raw regional\nL1 (pp)', ha='center', va='top', fontsize=7.9, linespacing=1.3)
    # A subtle division marks two different experiments, rather than one pipeline.
    fig.add_artist(Line2D([.371, .371], [.14, .78], transform=fig.transFigure,
                          color=LINE, lw=.65))
    b = fig.add_axes([.423, .291, .232, .435])
    c = fig.add_axes([.752, .291, .232, .435])
    for ax, metric, scale in [(b, 'O_shadow_P_full_key_TV', 100),
                              (c, 'O_shadow_pre_gate_W__relative_l2', 1)]:
        for f in FAMILIES:
            rr = sorted([r for r in D['attention'] if r['family'] == f and r['metric'] == metric],
                        key=lambda r: int(r['generation_call']))
            assert len(rr) == 3
            ax.plot([int(r['generation_call']) for r in rr],
                    [float(r['equal_task_mean'])*scale for r in rr], color=COLORS[f],
                    marker='o', ms=4.2, lw=1.6)
        ax.set_xlim(-.4, 9.4); ax.set_xticks([0, 5, 9])
        ax.set_xlabel('Generation call j', labelpad=4)
        style(ax, 'y')
    b.set_ylim(0, 20); b.set_yticks([0, 5, 10, 15, 20])
    c.set_ylim(0, .5); c.set_yticks([0, .1, .2, .3, .4, .5])
    family_legend(fig, .006, anchor=.64)
    save(fig, 'r13_attention')


def attention_support():
    """Preserve and regenerate the former main attention figure for supplement."""
    import shutil
    attention()
    for extension in ['pdf', 'svg']:
        shutil.copyfile(OUT / f'r13_attention.{extension}', OUT / f'r3_attention_support.{extension}')
    shutil.copyfile(PREVIEW / 'r13_attention.png', PREVIEW / 'r3_attention_support.png')


def function_tests():
    """Existing DEV local-norm contrasts and every CONF case outcome pair."""
    configure()
    evidence = json.loads((ROOT / 'data/function_tests.json').read_text())
    fig = plt.figure(figsize=(7, 3.12))
    fig.text(.013, .959, 'DEV SAVED QUERIES', fontsize=7.4, fontweight='bold', color=GEOMETRY)
    fig.text(.455, .959, 'CONFIRMATION · ALL FIRST-QUERY CASES',
             fontsize=7.4, fontweight='bold', color=GEOMETRY)
    heading(fig, .013, .893, 'a', 'Response vs. clean', GEOMETRY, fontsize=9.0)
    heading(fig, .455, .893, 'b', 'Camera · 96 cases', GEOMETRY, fontsize=8.8)
    heading(fig, .777, .893, 'c', 'Sensor · 120 cases', GEOMETRY, fontsize=8.8)
    fig.text(.013, .823, 'Queries 0, 2, 5, 10', fontsize=7.8, color=MUTED)
    fig.text(.455, .823, r'Outcome: unedited $N_T$ → restored $R_{\rm resp}$',
             fontsize=7.7, color=MUTED)
    fig.add_artist(Line2D([.377, .377], [.11, .94], transform=fig.transFigure,
                          color=LINE, lw=.7))

    a = fig.add_axes([.111, .307, .235, .433])
    for center, family in [(1, 'Camera Viewpoints'), (0, 'Sensor Noise')]:
        for offset, scheme, marker, hollow in [
                (.16, 'natural_projection_alpha_1', 'o', False),
                (-.16, 'common_prescribed_local_norm', 's', True)]:
            r = select(evidence['development']['comparisons'], family=family, scheme=scheme)
            interval(a, r['point'], r['original_interval'][0], r['original_interval'][1],
                     center+offset, COLORS[family], marker=marker, hollow=hollow, ms=4.6)
    a.set_ylim(-.55, 1.55)
    a.set_yticks([1, 0], ['Camera\n5 tasks', 'Sensor\n9 tasks'])
    a.tick_params(axis='y', labelsize=7.7, pad=3)
    a.set_xlim(-.028, .205); a.set_xticks([0, .1, .2])
    a.axvline(0, color=MUTED, lw=.95, ls=(0, (3, 2)))
    a.set_xlabel(r'Error reduction / $S_f$', fontsize=8.0, labelpad=4)
    style(a, 'x')
    scheme_handles = [
        Line2D([], [], color=INK, marker='o', ms=4.3, lw=1.2, label='Natural projection'),
        Line2D([], [], color=INK, marker='s', mfc='white', ms=4.3, lw=1.2,
               label='Common prescribed local norm'),
    ]
    fig.legend(handles=scheme_handles, loc='lower center', bbox_to_anchor=(.191, .026),
               frameon=False, ncol=1, fontsize=7.5, handlelength=1.6,
               handletextpad=.5, labelspacing=.6)
    fig.text(.013, .175, 'Original intervals', fontsize=7.4, color=MUTED)

    # The y coordinate is categorical. A fixed base-2 sequence supplies only
    # display offsets; horizontal gains and the complete case population remain exact.
    def offset(index):
        value, factor = 0.0, .5
        while index:
            value += (index % 2) * factor
            factor *= .5
            index //= 2
        return (value-.5)*.54

    outcomes = [('11', 'SS', 3, '#8A99A9', 'o', 10),
                ('10', 'SF', 2, '#C36435', '^', 22),
                ('01', 'FS', 1, '#188B80', 's', 18),
                ('00', 'FF', 0, '#66788A', 'x', 15)]
    case_total = 0
    for left, family, expected_n in [(.455, 'Camera Viewpoints', 96),
                                     (.777, 'Sensor Noise', 120)]:
        ax = fig.add_axes([left, .307, .211, .433])
        cases = [r for r in evidence['confirmation']['cases'] if r['family'] == family]
        assert len(cases) == expected_n
        case_total += len(cases)
        tick_labels = []
        for code, label, y, color, marker, size in outcomes:
            selected = sorted([r for r in cases if r['outcome_code'] == code], key=lambda r:r['case_id'])
            tick_labels.append(label + f' · {len(selected)}')
            x = [r['translation_error_gain_over_S_f'] for r in selected]
            yy = [y + offset(i+1) for i in range(len(selected))]
            if marker == 'x':
                ax.scatter(x, yy, color=color, marker=marker, s=size, linewidths=.8,
                           alpha=.8, zorder=3)
            else:
                ax.scatter(x, yy, facecolors=color, edgecolors='white', marker=marker,
                           s=size, linewidths=.3, alpha=.85, zorder=4 if code in ['10', '01'] else 3)
        ax.set_yticks([3, 2, 1, 0], tick_labels)
        ax.tick_params(axis='y', labelsize=7.5, pad=3)
        ax.set_ylim(-.5, 3.5)
        ax.set_xlim(-.26, 1.62)
        ax.set_xticks([0, .5, 1, 1.5])
        ax.tick_params(axis='x', labelsize=7.5)
        ax.axvline(0, color=MUTED, lw=.95, ls=(0, (3, 2)))
        for y in [.5, 1.5, 2.5]:
            ax.axhline(y, color=GRID, lw=.6, zorder=0)
        style(ax, 'x')
        if family == 'Camera Viewpoints':
            failed = [r for r in cases if r['outcome_code'] == '10']
            positive = sum(r['translation_error_gain_over_S_f'] > 0 for r in failed)
            assert (positive, len(failed)) == (3, 5)
            ax.text(.92, 2.11, '3/5 SF\ngain > 0', color='#A8532B', fontsize=7.1,
                    va='center', linespacing=1.35,
                    bbox={'facecolor':'white', 'edgecolor':'none', 'pad':.8})
    assert case_total == 216
    fig.text(.722, .191, r'First-query translation gain $(E^{N_T}-E^{R_{\rm resp}})/S_f$',
             fontsize=7.9, ha='center')
    fig.text(.722, .106, 'S: success · F: failure · each point is one case',
             fontsize=7.5, ha='center', color=MUTED)
    fig.text(.722, .049, 'Positive gain = smaller error to native clean commands',
             fontsize=7.4, ha='center', color=MUTED)
    save(fig, 'r3_function_tests')


def _behavior_forests(fig, a, b):
    """The unchanged eight-model/family primary LIBERO-Plus comparisons."""
    rows = [(m, f) for m in ['pi05_libero', 'pi0_libero'] for f in FAMILIES]
    labels = []
    for y, (m, f) in enumerate(rows):
        hollow = m == 'pi0_libero'
        col = COLORS[f]
        label = 'Bkg' if f == 'Background Textures' else SHORT[f]
        labels.append((r'$\pi_0$' if hollow else r'$\pi_{0.5}$') + '  ' + label)
        motion = select(D['motion'], model=m, family=f)
        interval(a, float(motion['ratio_of_task_means']), float(motion['ratio_ci95_low']),
                 float(motion['ratio_ci95_high']), y, col, marker='s' if hollow else 'o', hollow=hollow)
        outcome = select(D['outcomes'], model=m, family=f)
        if outcome['point'] is not None:
            interval(b, outcome['point']*100, outcome['ci98_75_low']*100,
                     outcome['ci98_75_high']*100, y, col,
                     marker='s' if hollow else 'o', hollow=hollow)
        else:
            lo, hi = outcome['missing_bound_low']*100, outcome['missing_bound_high']*100
            b.add_patch(Rectangle((lo, y-.145), hi-lo, .29, facecolor='white',
                                  edgecolor=col, hatch='////', lw=.8, zorder=4))
    for ax in [a, b]:
        ax.set_ylim(7.6, -.6)
        ax.axhspan(6.55, 7.45, color='#F4EDF6', zorder=0)
        ax.axhline(3.5, color=LINE, lw=.65)
        style(ax, 'x')
    a.set_yticks(range(8), labels)
    b.tick_params(labelleft=False)
    a.set_xlim(0, 3.8); a.set_xticks([0, 1, 2, 3])
    b.set_xlim(-43, 13); b.set_xticks([-40, -20, 0])
    a.axvline(1, color=MUTED, ls=(0, (3, 2)), lw=.9)
    b.axvline(0, color=MUTED, ls=(0, (3, 2)), lw=.9)
    a.set_xlabel('Visual / clean resampling', labelpad=4)
    b.set_xlabel('Success-rate change (pp)', labelpad=4)
    # Exact focal estimates are attached to the existing Sensor row.
    a.text(1.48, 7, '0.580×', color=COLORS['Sensor Noise'], fontsize=7.8,
            va='center', fontweight='bold')
    b.text(-14.2857, 6.57, '−14.3', color=COLORS['Sensor Noise'], fontsize=7.7,
            va='bottom', ha='center', fontweight='bold')

    model_handles = [Line2D([], [], marker='o', color=INK, ls='', ms=4, label=r'$\pi_{0.5}$'),
                     Line2D([], [], marker='s', color=INK, mfc='white', ls='', ms=4, label=r'$\pi_0$'),
                     Rectangle((0, 0), 1, 1, facecolor='white', edgecolor=COLORS['Light Conditions'],
                               hatch='////', label='Missing-outcome bounds')]
    return model_handles


def _sensor_scatter(c):
    """All 110 archived Sensor cases, retained as a separate optional figure."""
    c.fill_between([0, 20], [0, 0], [0, 20], color='#F7F2F8', zorder=0)
    c.plot([0, 20], [0, 20], color=MUTED, lw=.85, ls=(0, (3, 2)), zorder=2)
    styles = {'SS': ('#A0ACB8', 'o', 15, False), 'FF': ('#637286', 'x', 20, True),
              'FS': ('#188B80', 's', 22, True), 'SF': ('#C36435', '^', 30, True)}
    for cell in ['SS', 'FF', 'FS', 'SF']:
        rr = [r for r in D['sensor_cases'] if r['cell'] == cell]
        color, marker, size, filled = styles[cell]
        if marker == 'x':
            c.scatter([r['rr5_mm'] for r in rr], [r['tr5_mm'] for r in rr],
                      s=size, marker=marker, color=color, linewidths=.9, alpha=.9,
                      label=cell, zorder=4)
        else:
            c.scatter([r['rr5_mm'] for r in rr], [r['tr5_mm'] for r in rr],
                      s=size, marker=marker, facecolors=color if filled else 'none',
                      edgecolors=color, linewidths=.75, alpha=.9,
                      label=cell, zorder=5 if cell == 'SF' else 3)
    assert len(D['sensor_cases']) == 110
    c.set_xlim(-.4, 20); c.set_ylim(-.4, 20)
    c.set_xticks([0, 5, 10, 15, 20]); c.set_yticks([0, 5, 10, 15, 20])
    c.set_aspect('equal', adjustable='box')
    style(c, 'both', True)
    c.set_xlabel('Clean resampling (mm)', labelpad=4)
    c.set_ylabel('Visual separation (mm)', labelpad=2)
    harm_cases = [r for r in D['sensor_cases'] if r['cell'] == 'SF']
    below = sum(r['tr5_mm'] < r['rr5_mm'] for r in harm_cases)
    assert (below, len(harm_cases)) == (12, 18)
    c.text(.7, 19.2, f'{below} of {len(harm_cases)} harm pairs\nbelow clean resampling',
            color=BEHAVIOR, fontsize=7.7, va='top', linespacing=1.45,
            bbox={'facecolor': 'white', 'edgecolor': 'none', 'pad': 1})
    handles, labels = c.get_legend_handles_labels()
    lookup = dict(zip(labels, handles))
    return [lookup[k] for k in ['SS', 'SF', 'FS', 'FF']]


def sensor_case_view():
    configure()
    fig = plt.figure(figsize=(3.45, 3.45))
    c = fig.add_axes([.175, .225, .79, .65])
    handles = _sensor_scatter(c)
    fig.text(.175, .964, r'$\pi_0$ Sensor: all paired executions', fontsize=9, fontweight='bold')
    fig.text(.175, .918, '110 pairs · 55 variants · 28 tasks', fontsize=7.8, color=MUTED)
    fig.legend(handles, ['SS', 'Harm: S→F', 'Recovery: F→S', 'FF'],
               ncol=2, loc='lower center', bbox_to_anchor=(.55, .013), frameon=False,
               fontsize=7.6, handlelength=.8, handletextpad=.45, columnspacing=1.0,
               labelspacing=.6)
    save(fig, 'r2_sensor_case_view')


def behavior():
    configure()
    evidence = restored()['aggregation']
    fig = plt.figure(figsize=(7, 3.32))
    a = fig.add_axes([.137, .260, .223, .545])
    b = fig.add_axes([.441, .260, .208, .545], sharey=a)
    fig.text(.137, .967, 'LIBERO-PLUS', color=BEHAVIOR, fontsize=7.3, fontweight='bold')
    fig.text(.749, .967, 'ORIGINAL LIBERO', color=BEHAVIOR, fontsize=7.3, fontweight='bold')
    heading(fig, .137, .930, 'a', 'Early separation', BEHAVIOR, fontsize=8.8)
    heading(fig, .441, .930, 'b', 'Task outcomes', BEHAVIOR, fontsize=8.8)
    heading(fig, .749, .930, 'c', 'Aggregation', BEHAVIOR, fontsize=8.8)
    fig.text(.137, .862, 'End-effector · control 5', color=MUTED, fontsize=7.7)
    fig.text(.441, .862, 'Paired success change', color=MUTED, fontsize=7.7)
    fig.text(.749, .865, r'$\pi_{0.5}$ · 2,344 paired records', color=MUTED, fontsize=7.6)
    fig.text(.749, .821, '40 tasks · two observations', color=MUTED, fontsize=7.6)
    handles = _behavior_forests(fig, a, b)
    fig.text(.013, .072, '(a,b)', color=MUTED, fontsize=7.5)
    fig.legend(handles=handles, loc='lower left', bbox_to_anchor=(.064, .041),
               ncol=3, frameon=False, fontsize=7.6, columnspacing=1.2,
               handlelength=1.0, handletextpad=.5)

    # Separate population and measurements: these are two-observation planned
    # translation features and signed paired success, not EEF or issued commands.
    fig.add_artist(Line2D([.691, .691], [.12, .955], transform=fig.transFigure,
                          color=LINE, lw=.7))
    specifications = [
        ('condition', .750, .706, [.752, .493, .233, .174],
         '10 condition means', BEHAVIOR, 22, .95),
        ('task_condition', .410, .363, [.752, .166, .233, .174],
         r'400 task $\times$ condition means', '#657A8E', 6, .55),
    ]
    for key, title_y, rho_y, box, title, color, size, alpha in specifications:
        r = evidence[key]
        points = r['points']
        assert len(points) == r['n'] == (10 if key == 'condition' else 400)
        ax = fig.add_axes(box)
        ax.scatter([p['action_rank_percentile'] for p in points],
                   [p['delta_sr_rank_percentile'] for p in points],
                   s=size, color=color, edgecolors='none', alpha=alpha, zorder=3)
        ax.set_xlim(0, 100); ax.set_ylim(0, 100)
        ax.set_xticks([0, 50, 100]); ax.set_yticks([0, 50, 100])
        ax.tick_params(labelsize=7.0, pad=2)
        style(ax, 'both', True)
        fig.text(.749, title_y, title, fontsize=7.8, fontweight='bold', color=color)
        lo, hi = r['interval_95']
        rho = f"ρ = {r['rho']:.3f} [{lo:.3f}, {hi:+.3f}]".replace('-', '−')
        fig.text(.749, rho_y, rho, fontsize=7.4, color=INK)
    fig.text(.711, .422, 'Success-change rank (%)', ha='center', va='center',
             rotation=90, fontsize=7.4)
    fig.text(.869, .066, 'Planned-translation rank (%)', ha='center', fontsize=7.4)
    save(fig, 'r13_behavior')


if __name__ == '__main__':
    available = dict(basis=basis, cfr=cfr, attention=attention, attention_support=attention_support,
                     function_tests=function_tests, behavior=behavior,
                     sensor_case_view=sensor_case_view)
    names = sys.argv[1:] or ['basis', 'cfr', 'function_tests', 'behavior']
    for name in names:
        available[name]()
    print('Redrawn from frozen estimates: ' + ', '.join(names))
