"""Figure 5 (independent executions of the unedited policies), built around one question:
does more early motion mean more lost success?
  a  early motion (x) against success change (y), one mark per perturbation family and policy
  b  early motion of every task, and successes lost and gained (zero net change can hide flips)
  c  pi0 under Sensor Noise: every clean-perturbed pair, grouped by paired outcome
Data: every value is recomputed from raw/Five_arm_cases_flat.csv by fig5_data.build(), which also checks the family
values against the archived estimate tables. Light has unrecorded outcomes: its marks and bars use recorded outcomes
only and carry no success interval."""
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import figstyle as S
from figstyle import *
import fig5_data

apply()
DATA = fig5_data.build(); P = DATA['panels']                # P[family_policy]: ratio, net change, lost, gained, task ratios
FAM = {'B': 'Background', 'C': 'Camera', 'L': 'Light†', 'S': 'Sensor Noise'}
K = '#222222'; GRID = '#9a9a9a'; CI = '#8c8c8c'
HF = 204
F = Fig(HF); T = F.text; cv = F.cv


def title(x, y, letter, text):
    T(x, y, letter, size=9.5, weight='bold', color=INK); T(x + 11, y + 0.3, text, size=8.2, weight='bold', color=INK)


def mk(p5):
    return dict(marker='o' if p5 else 's', s=30 if p5 else 24, fc=K if p5 else '#ffffff', ec=K, lw=1.0)


def style(ax):
    for sp in ('top', 'right'): ax.spines[sp].set_visible(False)
    for sp in ('left', 'bottom'): ax.spines[sp].set_color(K); ax.spines[sp].set_linewidth(0.7)
    ax.tick_params(colors=K, width=0.7, length=2.6, labelsize=7.0)


# =================================================================== a: motion against success
title(1, 7, 'a', 'Small early motion can accompany lost success')
aA = F.axes(34, 20, 196, 156); style(aA)
aA.set_xscale('log'); aA.set_xlim(0.14, 4.6); aA.set_ylim(-43, 13)
aA.add_patch(Rectangle((0.14, -43), 1 - 0.14, 43, fc='#fbeeea', ec='none', zorder=0))
aA.axvline(1, color=GRID, lw=0.7, ls=(0, (2, 1.6)), zorder=1); aA.axhline(0, color=GRID, lw=0.7, ls=(0, (2, 1.6)), zorder=1)
aA.text(0.152, -40.6, 'less motion than a clean\nrerun, yet lower success', fontsize=7.0, color=K, va='bottom', ha='left', linespacing=1.15)
for fam in 'BCLS':                                              # link the two policies of a family
    pts = []
    for pol in ('pi05', 'pi0'):
        d = P[f'{fam}_{pol}']; pts.append((d['ratio'], d['net']))
    aA.plot([pts[0][0], pts[1][0]], [pts[0][1], pts[1][1]], color='#cfcfcf', lw=2.2, zorder=1.5, solid_capstyle='round')
for r, d in P.items():
    p5 = r.endswith('pi05')
    aA.plot([d['ratio_lo'], d['ratio_hi']], [d['net']] * 2, color=CI, lw=0.8, zorder=3)
    if d['net_lo'] is not None:                                 # Light: recorded outcomes only, so no success interval
        aA.plot([d['ratio']] * 2, [d['net_lo'], d['net_hi']], color=CI, lw=0.8, zorder=3)
    aA.scatter([d['ratio']], [d['net']], zorder=4, **mk(p5))
LAB = {'C': (1.66, -25.6, 'Camera', 'center'), 'B': (0.43, -7.6, 'Background', 'right'), 'L': (0.2, 7.0, 'Light†', 'center'), 'S': (0.152, -13.4, 'Sensor Noise', 'left')}
for fam, (x, y, s_, ha) in LAB.items(): aA.text(x, y, s_, fontsize=7.4, color=K, ha=ha, va='center')
aA.text(0.152, -17.6, f"{P['S_pi0']['ratio']:.2f}×, {P['S_pi0']['net']:.1f} pp".replace('-', '−'), fontsize=7.4, fontweight='bold', color=K, ha='left', va='center')
aA.set_xticks([0.25, 0.5, 1, 2, 4]); aA.set_xticklabels(['0.25', '0.5', '1×', '2', '4']); aA.minorticks_off()
aA.set_yticks([-40, -20, 0]); aA.set_yticklabels(['−40', '−20', '0'])
T(34 + 98, 196, 'Motion ratio after five controls (1× = clean rerun)', size=7.4, color=K, ha='center')
T(7, 20 + 78, 'Success change (pp)', size=7.4, color=K, ha='center', rotation=90)
cv.scatter([172], [26], zorder=5, **mk(True)); T(178, 26, r'$\pi_{0.5}$', size=7.8, color=K)
cv.scatter([200], [26], zorder=5, **mk(False)); T(206, 26, r'$\pi_0$', size=7.8, color=K)

# =================================================================== b: every task, and outcome flips
BX = 262.0
title(BX, 7, 'b', 'Motion ratio per task; successes lost and gained (%)')
rows = ['B_pi05', 'B_pi0', 'C_pi05', 'C_pi0', 'L_pi05', 'L_pi0', 'S_pi05', 'S_pi0']
RP, FG, Y0 = 8.0, 3.0, 21.0
ypt = {r: Y0 + (i // 2) * (2 * RP + FG) + (i % 2) * RP + RP / 2 for i, r in enumerate(rows)}; Y1 = Y0 + 8 * RP + 3 * FG
aT = F.axes(BX + 64, Y0, 84, Y1 - Y0); aT.set_ylim(Y1, Y0); style(aT); aT.spines['left'].set_visible(False); aT.set_yticks([])
aB = F.axes(BX + 160, Y0, 60, Y1 - Y0); aB.set_ylim(Y1, Y0); style(aB); aB.spines['left'].set_visible(False); aB.set_yticks([])
rng = np.random.default_rng(5); ntask = 0
for r in rows:
    y = ypt[r]; p5 = r.endswith('pi05'); h = RP * 0.6; d = P[r]
    v = np.array([t[0] for t in d['task_points']]); ntask += len(v)
    aT.scatter(v, y + rng.uniform(-2.0, 2.0, len(v)), s=3.2, color='#9fa6ad', lw=0, zorder=2)
    aT.plot([d['ratio_lo'], d['ratio_hi']], [y, y], color=K, lw=0.9, zorder=3); aT.scatter([d['ratio']], [y], zorder=4, **{**mk(p5), 's': 15 if p5 else 12})
    l, g = d['lost'], d['gained']; aB.barh(y, -l, height=h, color=LOST, lw=0); aB.barh(y, g, height=h, color=GAIN, lw=0)
    if r == 'B_pi05':
        aB.text(-l - 1.2, y, f'{l:.2f}', ha='right', va='center', fontsize=7.0, color=K); aB.text(g + 1.2, y, f'{g:.2f}', ha='left', va='center', fontsize=7.0, color=K)
    cv.scatter([BX + 57], [y], zorder=5, **{**mk(p5), 's': 15 if p5 else 12})
    if p5: T(BX + 1, y + RP / 2, FAM[r[0]], size=7.4, color=K)
print('task points drawn:', ntask)
aT.set_xscale('log'); aT.set_xlim(0.007, 11); aT.axvline(1, color=GRID, lw=0.7, ls=(0, (2, 1.6)), zorder=1)
aT.set_xticks([0.01, 0.1, 1, 10]); aT.set_xticklabels(['0.01', '0.1', '1×', '10']); aT.minorticks_off()
aB.set_xlim(-28, 12); aB.axvline(0, color=K, lw=0.7); aB.set_xticks([-25, 0, 10]); aB.set_xticklabels(['25', '0', '10'])
T(BX + 64 + 42, Y1 + 15.5, 'motion ratio per task', size=7.0, color=K, ha='center')
T(BX + 160 + 60 * 14 / 40, Y1 + 15.5, 'lost', size=7.0, color=LOST, ha='center'); T(BX + 160 + 60 * 34 / 40, Y1 + 15.5, 'gained', size=7.0, color=GAIN, ha='center')

# =================================================================== c: pairs by outcome
YC = Y1 + 33
title(BX, YC, 'c', r'$\pi_0$, Sensor Noise: every pair, by outcome')
yd = {'SS': 0.0, 'SF': 1.25, 'FS': 2.1, 'FF': 2.85}; DY, DH = YC + 10, HF - (YC + 10) - 22
aD = F.axes(BX + 52, DY, 142, DH); aD.set_ylim(3.35, -0.85); style(aD); aD.spines['left'].set_visible(False); aD.set_yticks([])
for c, lab, col, sz in (('SS', 'S → S', '#9fa6ad', 4.0), ('SF', 'S → F', LOST, 12), ('FS', 'F → S', GAIN, 12), ('FF', 'F → F', K, 12)):
    v = np.sort(np.array([q['ratio'] for q in DATA['pairs'] if q['cell'] == c])); lv = np.log10(v); off = np.zeros(len(v)); done = []
    for i in range(len(v)):
        cand = 0.0; k = 0; step = 0.14 if c == 'SS' else 0.2
        while any(abs(lv[i] - lv[j]) < (0.03 if c == 'SS' else 0.05) and abs(cand - off[j]) < step * 0.9 for j in done):
            k += 1; cand = ((k + 1) // 2) * step * (1 if k % 2 else -1)
        off[i] = cand; done.append(i)
    aD.scatter(v, yd[c] + off, s=sz, color=col, lw=0.25, ec='#ffffff', zorder=3)
    n = len(v); nb = int((v < 1).sum()); Y = DY + DH * (yd[c] + 0.85) / (3.35 + 0.85)
    T(BX + 1, Y, lab, size=7.4, color=K); T(BX + 30, Y, f'{n}', size=7.0, color=K)
    T(W_PT - 1, Y, f'{nb} of {n}', size=7.0, color=K, ha='right', weight='bold' if c == 'SF' else 'normal')
aD.set_xscale('log'); aD.set_xlim(0.02, 5.5); aD.axvline(1, color=GRID, lw=0.7, ls=(0, (2, 1.6)), zorder=1)
aD.set_xticks([0.03, 0.1, 0.3, 1, 3]); aD.set_xticklabels(['0.03', '0.1', '0.3', '1×', '3']); aD.minorticks_off()
T(W_PT - 1, YC + 11, 'below 1×', size=7.0, color=K, ha='right'); T(BX + 30, YC + 11, 'n', size=7.0, color=K)
T(BX + 52 + 71, HF - 4.5, 'Motion ratio of each pair after five controls', size=7.0, color=K, ha='center')

out = sys.argv[1] if len(sys.argv) > 1 else 'wire/fig5_new'
F.save(out)
print('saved', out)
