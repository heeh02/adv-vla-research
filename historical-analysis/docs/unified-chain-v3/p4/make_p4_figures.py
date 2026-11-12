"""P4 main figures (editable source). Inputs: results/analysis/P4-RESULTS.json, results/analysis/tables/p4_profiles.csv.gz,
results/analysis-v2/P4-RESULTS-v2.json. Outputs: ../figures/P4_F1_gain_decomposition.{pdf,png,svg},
../figures/P4_F2_fair_prediction.{pdf,png,svg} and ../figures/P4_plot_data.json (every plotted number).
"""
import csv
import gzip
import io
import json
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, 'results')
FIG = os.path.join(HERE, '..', 'figures')
os.makedirs(FIG, exist_ok=True)
mpl.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 7, 'axes.linewidth': 0.6, 'pdf.fonttype': 42, 'svg.fonttype': 'none',
                     'axes.spines.top': False, 'axes.spines.right': False, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6})
INK, MUTED, GRID = '#1f2a33', '#6b7782', '#d5d9de'
C_T, C_RHO, C_G, C_NU, C_CLEAN = INK, '#2f6db5', '#9aa1a8', '#d9822b', '#7d8790'
C_BASE, C_BEST, C_READ = '#b9c0c7', '#1f7a6d', '#2f6db5'
ORDER = ['camera_plus30', 'camera_minus30', 'camera_plus15', 'camera_minus15', 'camera_plus15_surface_warm',
         'camera_minus15_surface_cool', 'headlight_off', 'surface_cool', 'surface_warm', 'matched_global']
LABEL = {'matched_global': 'matched dim', 'surface_warm': 'warm surface', 'surface_cool': 'cool surface', 'headlight_off': 'headlight off',
         'camera_plus15': 'camera +15°', 'camera_minus15': 'camera −15°', 'camera_plus15_surface_warm': '+15° & warm',
         'camera_minus15_surface_cool': '−15° & cool', 'camera_plus30': 'camera +30°', 'camera_minus30': 'camera −30°'}


def save(fig, name):
    for ext in ('pdf', 'png', 'svg'):
        fig.savefig(os.path.join(FIG, f'{name}.{ext}'), dpi=220 if ext == 'png' else None)
    plt.close(fig)


def profiles():
    rows = list(csv.reader(io.StringIO(gzip.decompress(open(os.path.join(RES, 'analysis', 'tables', 'p4_profiles.csv.gz'), 'rb').read()).decode())))
    out = {}
    for r in rows[1:]:
        out[(r[0], r[1])] = np.array([float(x) for x in r[2:]])
    return out


def shared_rho():
    """PH-3: condition-balanced readable fraction at tau 0/5/9 for the shared-state counterfactual and the free path."""
    import glob
    import pandas as pd
    files = sorted(glob.glob(os.path.join(RES, 'shared-g*', 'shared_scalars.csv.gz')))
    if not files:
        return None
    d = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    d['task'] = d.root.str.rsplit('-', n=1).str[0]
    out = {}
    for tag, name in (('sh', 'shared'), ('fr', 'free')):
        vals = []
        for j in range(3):
            v = np.log(d[f'{tag}_A_{j}'] / d[f'{tag}_R_{j}'])
            t = pd.DataFrame({'v': v, 'task': d.task, 'cond': d.cond})
            vals.append(float(100 * np.exp(t.groupby(['cond', 'task']).v.mean().groupby('cond').mean().mean())))
        out[name] = vals
    return out


def fig1(R, P, pdata):
    A = R['A']
    k = np.arange(10)
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.6))
    axes = axes.ravel()
    # (a) exact decomposition along the steps, relative to step 0
    ax = axes[0]
    for q, col, lab, lw in [('logT', C_T, 'T (effective gain)', 1.6), ('logrho', C_RHO, 'ρ readable fraction', 1.2),
                            ('logg', C_G, 'g in-subspace gain', 1.0), ('lognu', C_NU, 'ν linear-reference dev.', 1.0)]:
        y = P[('all', q)] - P[('all', q)][0]
        ax.plot(k, y, color=col, lw=lw, label=lab, ls='--' if q == 'logrho' else '-')
        pdata.setdefault('F1a', {})[q] = y.tolist()
    ax.axhline(0, color=GRID, lw=0.6)
    ax.set_xlabel('denoising step k')
    ax.set_ylabel('ln X_k − ln X_0 (condition-balanced mean)')
    ax.set_title('a  exact decomposition T = ρ·g·ν along the ten denoising steps', loc='left', fontsize=7)
    ax.legend(frameon=False, fontsize=6, loc='center left', bbox_to_anchor=(0.02, 0.42))
    ax.set_xticks([0, 3, 6, 9])
    # (b) per-condition first->last contributions
    ax = axes[1]
    y = np.arange(len(ORDER))[::-1]
    bc = A['by_condition']
    for yi, c in zip(y, ORDER):
        rho, g, nu, T = bc[c]['rho'], bc[c]['g'], bc[c]['nu'], bc[c]['T']
        ax.barh(yi, rho, height=0.62, color=C_RHO, lw=0)
        ax.barh(yi, nu, left=rho + min(g, 0), height=0.62, color=C_NU, lw=0)
        ax.plot([T], [yi], marker='|', color=INK, ms=7, mew=1.2)
    ax.set_yticks(y)
    ax.set_yticklabels([LABEL[c] for c in ORDER], fontsize=6)
    ax.set_xlim(-2.1, 0.05)
    ax.axvline(0, color=GRID, lw=0.6)
    pr = A['primary']
    ax.set_xlabel('Δ ln, step 0 → 9')
    ax.set_title(f"b  step 0→9 by condition (tick = Δln T)\n    ρ (blue) {100 * pr['share']['rho']:.1f}% "
                 f"[{100 * pr['share_ci']['rho'][0]:.1f}, {100 * pr['share_ci']['rho'][1]:.1f}], ν (orange) {100 * pr['share']['nu']:.1f}%, g ≈ 0",
                 loc='left', fontsize=6.8)
    pdata['F1b'] = {c: bc[c] for c in ORDER}
    # (c) readable fraction: perturbation response vs clean readout
    ax = axes[2]
    rho_d = 100 * np.exp(P[('all', 'logrho')])
    rho_c = 100 * np.exp(P[('all', 'logrho_clean')])
    ci = np.array(A['logrho_profile_ci'], dtype=float)
    ax.fill_between(k, 100 * np.exp(ci[:, 0]), 100 * np.exp(ci[:, 1]), color=C_RHO, alpha=0.2, lw=0)
    ax.plot(k, rho_d, color=C_RHO, lw=1.4, label='Δr (perturbation response)')
    ax.plot(k, rho_c, color=C_CLEAN, lw=1.2, ls='--', label='r (clean readout)')
    pdata['F1c'] = {'rho_delta_pct': rho_d.tolist(), 'rho_clean_pct': rho_c.tolist()}
    sh = shared_rho()
    if sh is not None:
        ax.plot([0, 5, 9], sh['shared'], color=C_RHO, lw=0.9, ls=(0, (2, 1.5)), marker='o', ms=3.2, mfc='white', mew=0.8,
                label='Δr at shared x_t, τ 0/5/9 (counterfactual, post hoc)')
        pdata['F1c']['rho_shared_state_pct_tau0_5_9'] = sh['shared']
        pdata['F1c']['rho_free_same_tau_pct'] = sh['free']
    ax.axhline(100 * 7 / 1024, color=MUTED, lw=0.6, ls=':')
    ax.text(9.2, 100 * 7 / 1024 * 1.15, 'isotropic 7/1024', fontsize=5.6, color=MUTED, ha='right')
    ax.legend(frameon=False, fontsize=5.6, loc='lower left', bbox_to_anchor=(0.0, 0.11))
    ax.set_yscale('log')
    ax.set_ylim(0.5, 40)
    ax.set_yticks([1, 3, 10, 30])
    ax.set_yticklabels(['1%', '3%', '10%', '30%'])
    ax.set_xticks([0, 3, 6, 9])
    ax.set_xlabel('denoising step k')
    ax.set_ylabel('readable fraction of energy')
    ax.set_title(f'c  readable fraction: response falls {rho_d[0] / rho_d[-1]:.1f}×, clean readout {rho_c[0] / rho_c[-1]:.1f}×', loc='left', fontsize=7)
    # (d) where the readout response energy goes
    ax = axes[3]
    for q, col, lab in [('logN', C_G, 'null-space energy'), ('logA', C_RHO, 'readable energy'), ('logV', C_T, 'effective update energy')]:
        yy = P[('all', q)] - P[('all', q)][0]
        ax.plot(k, np.exp(yy), color=col, lw=1.3 if q != 'logV' else 1.0, ls='-' if q != 'logV' else ':', label=lab)
        pdata.setdefault('F1d', {})[q] = np.exp(yy).tolist()
    ax.set_yscale('log')
    ax.set_xticks([0, 3, 6, 9])
    ax.set_xlabel('denoising step k')
    ax.set_ylabel('energy relative to step 0')
    ax.set_title('d  response grows in the null space;\n    update energy tracks the readable part', loc='left', fontsize=7)
    ax.legend(frameon=False, fontsize=6, loc='upper left')
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.08, top=0.93, wspace=0.42, hspace=0.42)
    save(fig, 'P4_F1_gain_decomposition')


def fig2(R, R2, pdata):
    B, C = R['B'], R['C']
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.9), gridspec_kw=dict(width_ratios=[1.0, 1.15]))
    axes = axes.ravel()
    # (a) out-of-fold EEF by model
    ax = axes[0]
    models = [('Z', 'zero'), ('M0', 'cell mean (M0)'), ('M1', 'magnitude (M1)'), ('M2', '9·u₀, no fit (M2)'), ('M2g', 'β·u₀ (M2g)'),
              ('M2L', 'linear on u₀ (M2L)'), ('M3', '+ signed readout (M3)'), ('M3n', 'readout only (M3n)'), ('M3f', '+ full Δr₀ PCs (M3f)')]
    y = np.arange(len(models))[::-1]
    for yi, (m, lab) in zip(y, models):
        v = B['metrics'][m]
        col = C_BEST if m == 'M2' else C_READ if m.startswith('M3') else C_BASE
        ax.barh(yi, max(v['EEF'], -0.45), height=0.62, color=col, lw=0)
        ax.plot(v['EEF_condition_balanced'], yi, marker='o', ms=2.8, color=INK, mew=0)
        ax.text(max(v['EEF'], 0) + 0.03, yi, f"{v['EEF']:.3f}", va='center', fontsize=5.8)
    ax.set_yticks(y)
    ax.set_yticklabels([l for _, l in models], fontsize=6.2)
    ax.set_xlim(-0.45, 1.0)
    ax.axvline(0, color=GRID, lw=0.6)
    ax.set_xlabel('out-of-fold EEF of the remaining 9-step update')
    ax.set_title('a  task-held-out prediction (bar pooled, dot condition-balanced)', loc='left', fontsize=6.8)
    pdata['F2a'] = {m: B['metrics'][m] for m, _ in models}
    # (b) increments with 95% task-bootstrap CI
    ax = axes[1]
    pc = B['primary_contrast_M3_minus_M2L']
    rows = [('M3 − M2L (prereg. primary)', pc['pooled'], pc['pooled_ci'], C_READ, 'o'),
            ('M3 − M2L, condition-balanced', pc['condition_balanced'], pc['condition_balanced_ci'], C_READ, 'o'),
            ('M3 − M2 (no-fit 9·u₀)', B['M3_minus_each_baseline']['M2']['pooled'], B['M3_minus_each_baseline']['M2']['pooled_ci'], C_READ, 'o')]
    for fam in ('camera', 'lighting', 'surface'):
        v = B['leave_family_out'][fam]
        rows.append((f'held-out {fam} family: M3 − M2L', v['M3_minus_M2L'], v['ci'], C_READ, 'o'))
    if R2 and 'PH1_M2_plus_readout' in R2:
        ph = R2['PH1_M2_plus_readout']
        rows.append(('post hoc: (M2 + readout) − M2', ph['delta'], ph['ci'], MUTED, 'D'))
    yy = np.arange(len(rows))[::-1]
    for yi, (lab, v, ci, col, mk) in zip(yy, rows):
        ax.plot(ci, [yi, yi], color=col, lw=1.4)
        ax.plot(v, yi, marker=mk, color=col, ms=3.6, mfc='white' if mk == 'D' else col)
    ax.axvline(0, color=INK, lw=0.6)
    ax.axvline(0.02, color=MUTED, lw=0.6, ls='--')
    ax.text(0.019, yy[-1] - 0.45, 'prereg.\nthreshold', fontsize=5.6, color=MUTED, va='top', ha='right')
    ax.set_ylim(yy[-1] - 1.0, yy[0] + 0.6)
    ax.set_yticks(yy)
    ax.set_yticklabels([r[0] for r in rows], fontsize=6.2)
    ax.set_xlim(-0.05, 0.035)
    ax.set_xlabel('Δ EEF (95% task-bootstrap CI)')
    ax.set_title('b  increments from signed readout', loc='left', fontsize=6.8)
    pdata['F2b'] = [{'label': r[0], 'value': r[1], 'ci': r[2]} for r in rows]
    # (c) cross-seed transfer by anchor class (q0 = same visited state; others = different states)
    ax = axes[2]
    R3 = pdata.get('_R3')
    classes = ['q0', 'lm1', 'lm2', 'later']
    Cs = {'q0': C['q0']}
    if R2:
        Cs.update({c: R2['C'][c] for c in classes[1:] if 'metrics' in R2['C'].get(c, {})})
    series = [('C1', 'C1 other seeds’ command mean', C_BEST, None), ('C1s', 'C1s fitted command mean (post hoc)', 'white', C_BEST),
              ('C2', 'C2 readout instead', C_READ, None), ('C3', 'C3 = C1 + readout', '#8fb3e0', None)]
    wbar = 0.19
    for j, (m, lab, fc, ec) in enumerate(series):
        xs, vs = [], []
        for i, c in enumerate(classes):
            if c not in Cs:
                continue
            if m == 'C1s':
                if not R3 or c not in R3:
                    continue
                v = R3[c]['EEF']['C1s']
            else:
                v = Cs[c]['metrics'][m]['EEF']
            xs.append(i + (j - 1.5) * wbar); vs.append(v)
        if xs:
            ax.bar(xs, vs, width=wbar * 0.92, color=fc, edgecolor=ec or fc, lw=0.8 if ec else 0, label=lab)
    ax.axhline(0, color=INK, lw=0.6)
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(['q0\n(same state)', 'lm1', 'lm2', 'later'], fontsize=6.2)
    ax.set_ylim(-0.2, 1.62)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_ylabel('out-of-fold EEF, target-seed command')
    ax.legend(frameon=False, fontsize=5.4, loc='upper center', ncol=2, columnspacing=0.8, handlelength=1.2)
    ax.set_title('c  cross-seed transfer (C0 magnitude baseline < 0, not shown)', loc='left', fontsize=6.8)
    pdata['F2c'] = {c: {m: (Cs[c]['metrics'][m]['EEF'] if c in Cs else None) for m in ('C0', 'C1', 'C2', 'C3')} for c in classes}
    if R3:
        for c in classes:
            if c in R3:
                pdata['F2c'][c]['C1s_posthoc'] = R3[c]['EEF']['C1s']
    # (d) by condition: where readout helps or hurts
    ax = axes[3]
    bc = B['by_condition']
    yy = np.arange(len(ORDER))[::-1]
    for yi, c in zip(yy, ORDER):
        ax.plot(bc[c]['M3'] - bc[c]['M2L'], yi, marker='o', ms=3.4, color=C_READ, ls='none', label='M3 − M2L' if c == ORDER[0] else None)
        ax.plot(bc[c]['M3'] - bc[c]['M2'], yi, marker='o', ms=3.4, color=C_BEST, mfc='white', ls='none', label='M3 − M2' if c == ORDER[0] else None)
    ax.axvline(0, color=INK, lw=0.6)
    ax.axvline(0.02, color=MUTED, lw=0.6, ls='--')
    ax.set_yticks(yy)
    ax.set_yticklabels([LABEL[c] for c in ORDER], fontsize=6.2)
    ax.set_xlabel('Δ EEF within condition (task-held-out)')
    ax.set_title('d  by condition:  ● M3 − M2L   ○ M3 − M2', loc='left', fontsize=6.8)
    pdata['F2d'] = {c: {'M3_minus_M2L': bc[c]['M3'] - bc[c]['M2L'], 'M3_minus_M2': bc[c]['M3'] - bc[c]['M2']} for c in ORDER}
    fig.subplots_adjust(left=0.17, right=0.98, bottom=0.1, top=0.93, wspace=0.75, hspace=0.55)
    save(fig, 'P4_F2_fair_prediction')


if __name__ == '__main__':
    R = json.load(open(os.path.join(RES, 'analysis', 'P4-RESULTS.json')))
    p2 = os.path.join(RES, 'analysis-v2', 'P4-RESULTS-v2.json')
    R2 = json.load(open(p2)) if os.path.exists(p2) else None
    P = profiles()
    p3 = os.path.join(RES, 'analysis-v3', 'P4-RESULTS-v3.json')
    R3 = json.load(open(p3))['PH2'] if os.path.exists(p3) else None
    pdata = {'sources': {'v1': 'results/analysis/P4-RESULTS.json', 'v2': 'results/analysis-v2/P4-RESULTS-v2.json' if R2 else None,
                         'v3': 'results/analysis-v3/P4-RESULTS-v3.json' if R3 else None}, '_R3': R3}
    fig1(R, P, pdata)
    fig2(R, R2, pdata)
    pdata.pop('_R3', None)
    json.dump(pdata, open(os.path.join(FIG, 'P4_plot_data.json'), 'w'), indent=1, default=float)
    print('ok')
