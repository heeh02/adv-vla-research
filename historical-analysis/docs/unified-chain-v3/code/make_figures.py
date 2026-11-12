"""Figures for P1/P2/P3 (reads data/p1_pairs.csv.gz, data/p3_units.csv.gz, reports/*.json)."""
import json
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, '..', 'data')
R = os.path.join(HERE, '..', 'reports')
F = os.path.join(HERE, '..', 'figures')
os.makedirs(F, exist_ok=True)
mpl.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 7, 'axes.linewidth': 0.6, 'pdf.fonttype': 42,
                     'axes.spines.top': False, 'axes.spines.right': False})
INK, MUTED = '#1f2a33', '#6b7782'
ORDER = ['matched_global', 'surface_warm', 'surface_cool', 'headlight_off', 'camera_plus15', 'camera_minus15',
         'camera_plus15_surface_warm', 'camera_minus15_surface_cool', 'camera_plus30', 'camera_minus30']
LABEL = {'matched_global': 'matched dim', 'surface_warm': 'warm surf', 'surface_cool': 'cool surf', 'headlight_off': 'headlight off',
         'camera_plus15': '+15°', 'camera_minus15': '−15°', 'camera_plus15_surface_warm': '+15° & warm',
         'camera_minus15_surface_cool': '−15° & cool', 'camera_plus30': '+30°', 'camera_minus30': '−30°'}


def color(c):
    if '30' in c:
        return '#1b4f8a'
    if 'camera' in c and 'surface' in c:
        return '#e08a2c'
    if 'camera' in c:
        return '#3d8fd1'
    if c in ('headlight_off', 'matched_global'):
        return '#2a9d78'
    return '#8a63b8'


def save(fig, name):
    fig.savefig(os.path.join(F, name + '.pdf'))
    fig.savefig(os.path.join(F, name + '.png'), dpi=200)
    plt.close(fig)


def fig_chain_factors(df):
    """Per-condition medians (with IQR) of each multiplicative factor, normalized to the across-condition median."""
    facs = [('Rtot', 'readout energy'), ('Teff', 'transfer gain'), ('Keff', 'effective steps'), ('c', 'step coherence'),
            ('alpha', 'adapter factor'), ('P', 'planned-cmd energy')]
    fig, axes = plt.subplots(1, len(facs), figsize=(7.0, 2.4), sharey=True)
    for ax, (k, lab) in zip(axes, facs):
        ref = df.groupby('cond')[k].median().median()
        for i, c in enumerate(ORDER):
            v = df[df.cond == c][k] / ref
            q1, m, q3 = np.percentile(v, [25, 50, 75])
            ax.plot([q1, q3], [i, i], color=color(c), lw=1.2, alpha=0.6)
            ax.scatter(m, i, s=14, color=color(c), zorder=3)
        ax.set_xscale('log')
        ax.axvline(1, color=MUTED, lw=0.5, ls='--')
        ax.set_title(lab, fontsize=7)
        ax.set_xlim(0.008, 12)
    axes[0].set_yticks(range(len(ORDER)))
    axes[0].set_yticklabels([LABEL[c] for c in ORDER])
    fig.suptitle('Readout→planned-command chain per condition (median, IQR; ÷ cross-condition median, log scale)\n'
                 'Condition centres of the middle four factors are stable; per-pair spread is not (e.g. transfer gain p5–p95 0.010–0.062).',
                 fontsize=6.5, x=0.02, ha='left')
    fig.subplots_adjust(left=0.12, right=0.99, bottom=0.12, top=0.80, wspace=0.15)
    save(fig, 'F1_chain_factors')


def fig_temporal(df, p2):
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.3))
    tk = np.stack([df[f'V_{k}'] / df[f'R_{k}'] for k in range(10)], 1)
    vk = df[[f'V_{k}' for k in range(10)]].values
    share = vk / vk.sum(1, keepdims=True)
    ax = axes[0]
    for fam, cs, col in [('camera', [c for c in ORDER if 'camera' in c], '#3d8fd1'), ('lighting', ['headlight_off', 'matched_global'], '#2a9d78'),
                         ('surface', ['surface_warm', 'surface_cool'], '#8a63b8')]:
        sel = df.cond.isin(cs).values
        ax.plot(range(10), np.median(tk[sel], 0), color=col, lw=1.2, label=fam)
    ax.set_yscale('log')
    ax.set_xlabel('denoising step k')
    ax.set_title('a  effective transfer gain (median)\nT_k = |Δv_eff,k|²/|Δr_k|²,  Δv_eff,k = Δx_k/Δt', loc='left', fontsize=6.5)
    ax.legend(frameon=False, fontsize=6)
    ax = axes[1]
    med = np.median(share, 0)
    ax.plot(range(10), med, color=INK, lw=1.2, marker='o', ms=2.5)
    ax.fill_between(range(10), np.percentile(share, 25, 0), np.percentile(share, 75, 0), color=MUTED, alpha=0.25, lw=0)
    ax.axhline(0.1, color=MUTED, ls='--', lw=0.5)
    ax.text(0.2, 0.104, 'uniform = 0.1', fontsize=5.5, color=MUTED)
    ax.annotate(f'last step {med[-1]:.2f}', (9, med[-1]), (5.2, 0.27), fontsize=5.5, arrowprops=dict(arrowstyle='-', lw=0.5, color=MUTED))
    ax.set_ylim(0, 0.32)
    ax.set_xlabel('denoising step k')
    ax.set_title('b  share of update energy per step\n(median, IQR)', loc='left', fontsize=6.5)
    ax = axes[2]
    for i, c in enumerate(ORDER):
        v = df[df.cond == c].c
        q1, m, q3 = np.percentile(v, [25, 50, 75])
        ax.plot([q1, q3], [i, i], color=color(c), lw=1.2, alpha=0.6)
        ax.scatter(m, i, s=14, color=color(c), zorder=3)
        ax.text(1.005, i, f'{(v < 0.9).mean():.0%}', va='center', fontsize=5.3, color=MUTED)
    ax.axvline(0.9, color=MUTED, ls='--', lw=0.5)
    ax.set_yticks(range(len(ORDER)))
    ax.set_yticklabels([LABEL[c] for c in ORDER], fontsize=6)
    ax.set_xlim(0.4, 1.0)
    ax.set_xlabel('step coherence c = |Σu|/Σ|u|')
    ax.set_title('c  step coherence (median, IQR)\nright: share of pairs with c < 0.9', loc='left', fontsize=6.5)
    fig.subplots_adjust(left=0.07, right=0.95, bottom=0.18, top=0.84, wspace=0.5)
    save(fig, 'F2_temporal')


def fig_prediction(p1):
    d = p1['D2']
    lofo = p1['D3_leave_family_out']
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.3), gridspec_kw=dict(width_ratios=[1.5, 1]))
    labels = ['held-out tasks (5-fold): A vs B', 'held-out camera family: A vs B′', 'held-out lighting family: A vs B′', 'held-out surface family: A vs B′']
    a = [d['A'], lofo['camera']['R2_A'], lofo['lighting']['R2_A'], lofo['surface']['R2_A']]
    b = [d['B'], lofo['camera']['R2_Bprime'], lofo['lighting']['R2_Bprime'], lofo['surface']['R2_Bprime']]
    y = np.arange(4)
    ax.barh(y + 0.18, a, height=0.34, color='#3d8fd1', label='A: total readout energy only')
    ax.barh(y - 0.18, b, height=0.34, color='#e08a2c',
            label='B: A + training-fold condition-mean profile (T_eff, c, K_eff);  B′: profile of unseen family extrapolated\n'
                  'from its conditions\' mean log readout energy (unlabelled test inputs; no test command energy used)')
    for yi, v in zip(y, a):
        ax.text(v + 0.01, yi + 0.18, f'{v:.3f}', va='center', fontsize=5.5)
    for yi, v in zip(y, b):
        ax.text(v + 0.01, yi - 0.18, f'{v:.3f}', va='center', fontsize=5.5)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.12)
    ax.set_xlabel('out-of-fold R², log planned-command energy (same generation)', fontsize=6)
    ax.legend(frameon=False, fontsize=5.3, loc='upper left', bbox_to_anchor=(-0.42, -0.3), ncol=1)
    ax.set_title('a  prediction', loc='left', fontsize=6.5)
    ax2.errorbar([d['B_minus_A']], [0], xerr=[[d['B_minus_A'] - d['B_minus_A_ci'][0]], [d['B_minus_A_ci'][1] - d['B_minus_A']]],
                 fmt='o', color='#e08a2c', ms=4, capsize=2, lw=1)
    ax2.axvline(0, color=MUTED, lw=0.6)
    ax2.axvline(0.05, color=MUTED, lw=0.6, ls='--')
    ax2.text(0.049, 0.42, 'prereg.\nthreshold 0.05', fontsize=5.5, color=MUTED, ha='right')
    ax2.text(0.004, -0.3, f"{d['B_minus_A']:.4f} [{d['B_minus_A_ci'][0]:.4f}, {d['B_minus_A_ci'][1]:.4f}]", fontsize=5.5, ha='left')
    ax2.set_ylim(-0.6, 0.6)
    ax2.set_yticks([])
    ax2.set_xlim(-0.005, 0.06)
    ax2.set_xlabel('B − A (task-cluster bootstrap 95% CI)')
    ax2.set_title('b  held-out tasks: B − A', loc='left', fontsize=6.5)
    ax2.spines['left'].set_visible(False)
    fig.subplots_adjust(left=0.25, right=0.98, bottom=0.46, top=0.9, wspace=0.12)
    save(fig, 'F3_prediction')


def fig_direction(U, p3, p3all):
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.5), gridspec_kw=dict(width_ratios=[1.15, 1, 1.25]))
    ax = axes[0]
    d1, d1a, d1f = p3['D1'], p3all['D1'], p3['D1_sensitivity_floor_10.6mm']
    rows = [('primary\n3 anchors, 29.8 mm', d1['cos_DO_median'], d1['ci'], f"{d1['n_units']} units, {d1['n_tasks']} tasks", '#1b4f8a'),
            ('all anchors\n(DEV-3), 29.8 mm', d1a['cos_DO_median'], d1a['ci'], f"{d1a['n_units']} units, {d1a['n_tasks']} tasks", MUTED),
            ('3 anchors\n10.6 mm floor', d1f['cos_DO_median'], d1f['ci'], f"{d1f['n_units']} units, {d1f['n_tasks']} tasks", MUTED)]
    for i, (lab, m, ci, n, col) in enumerate(rows):
        ax.plot(ci, [i, i], color=col, lw=1.5)
        ax.scatter(m, i, color=col, s=18, zorder=3)
        ax.text(-0.95, i + 0.3, n, fontsize=5, color=MUTED, ha='left')
    ax.axvline(0, color=MUTED, lw=0.6)
    ax.set_yticks(range(3))
    ax.set_yticklabels([r[0] for r in rows], fontsize=5.5)
    ax.set_xlim(-1, 1)
    ax.set_xlabel('median cos(D_sys, O_sys), camera units')
    ax.set_ylim(2.6, -0.5)
    ax.set_title('a  command deviation vs placement offset\n(successful seeds only; primary not met)', loc='left', fontsize=6.5)
    ax = axes[1]
    mir = p3['D2_mirror']
    keys = [('prereg46:camera_plus15_vs_camera_minus15', '±15°, 46 prereg. roots'), ('prereg46:camera_plus30_vs_camera_minus30', '±30°, 46 prereg. roots'),
            ('all72:camera_plus15_vs_camera_minus15', '±15°, all 72 roots'), ('all72:camera_plus30_vs_camera_minus30', '±30°, all 72 roots')]
    for i, (k, lab) in enumerate(keys):
        v = mir[k]
        col = '#1b4f8a' if k.startswith('prereg') else MUTED
        ax.plot(v['ci'], [i, i], color=col, lw=1.5)
        ax.scatter(v['cos_median'], i, color=col, s=18, zorder=3)
    ax.axvline(0, color=MUTED, lw=0.6)
    ax.set_yticks(range(4))
    ax.set_yticklabels([k[1] for k in keys], fontsize=5.5)
    ax.invert_yaxis()
    ax.set_xlim(-1, 1)
    ax.set_xlabel('median cos(D_sys(+θ), D_sys(−θ))')
    ax.set_title('b  left vs right rotation: same direction\n(not mirrored; geometry not excluded)', loc='left', fontsize=6.5)
    ax = axes[2]
    conds = ['headlight_off', 'matched_global', 'surface_warm', 'surface_cool', 'camera_plus15', 'camera_minus15', 'camera_plus30', 'camera_minus30']
    M = np.full((len(conds), len(conds)), np.nan)
    piv = {c: U[U.cond == c].set_index('root')[['D_x', 'D_y']] for c in conds}
    for i, a in enumerate(conds):
        for j, b in enumerate(conds):
            com = piv[a].index.intersection(piv[b].index)
            A, B = piv[a].loc[com].values.astype(float), piv[b].loc[com].values.astype(float)
            na, nb = np.linalg.norm(A, axis=1), np.linalg.norm(B, axis=1)
            ok = (na > 0) & (nb > 0)
            M[i, j] = np.median((A[ok] * B[ok]).sum(1) / na[ok] / nb[ok])
    im = ax.imshow(M, cmap='RdBu_r', vmin=-1, vmax=1)
    ax.set_xticks(range(len(conds)))
    ax.set_yticks(range(len(conds)))
    ax.set_xticklabels([LABEL[c] for c in conds], rotation=60, ha='right', fontsize=5.5)
    ax.set_yticklabels([LABEL[c] for c in conds], fontsize=5.5)
    for i in range(len(conds)):
        for j in range(len(conds)):
            ax.text(j, i, f'{M[i, j]:.1f}', ha='center', va='center', fontsize=4.8, color='white' if abs(M[i, j]) > 0.6 else INK)
    ax.set_title('c  same-scene direction similarity\n(median cos, 72 roots; descriptive)', loc='left', fontsize=6.5)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    fig.subplots_adjust(left=0.13, right=0.95, bottom=0.27, top=0.83, wspace=0.85)
    save(fig, 'F4_direction')


if __name__ == '__main__':
    df = pd.read_csv(os.path.join(D, 'p1_pairs.csv.gz'))
    p1 = json.load(open(os.path.join(R, 'P1-RESULTS.json')))
    p2 = json.load(open(os.path.join(R, 'P2-RESULTS.json')))
    p3 = json.load(open(os.path.join(R, 'P3-RESULTS.json')))
    p3all = json.load(open(os.path.join(R, 'P3-RESULTS-allanchors.json')))
    U = pd.read_csv(os.path.join(D, 'p3_units.csv.gz'))
    fig_chain_factors(df)
    fig_temporal(df, p2)
    fig_prediction(p1)
    fig_direction(U, p3, p3all)
    print('ok')
