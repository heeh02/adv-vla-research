"""Figure 4 (restoration test): each row compares restoration with another run of the same cases.
First row: the unedited perturbed run (the main comparison, also summarized in Figure 1b). Then the two controls with
16 directions (random, clean-feature) and replacement of the whole L9 state.
  a  reduction in initial-command error, arm minus response restoration (mean, simultaneous 95% interval, tasks)
  b  task-success gain, response restoration minus arm (percentage points, simultaneous 95% interval)
Dots and intervals are drawn as in Figure 1b. Full L9 is outside the six prespecified comparisons: open mark, no interval.
All values come from fig4_data.build() (raw/restoration/)."""
import sys
import numpy as np
from matplotlib.patches import Circle
import figstyle as S
from figstyle import *
import fig4_data

apply()
D = fig4_data.build()
K = '#222222'
ROWS = (('N_T', 'Perturbed, unedited'), ('R_rand', 'Random directions'), ('R_clean', 'Clean-feature directions'), ('R_full', 'Whole L9 state'))
Y0, DYR = 35.0, 24.0                       # first row, row pitch
YA = Y0 + (len(ROWS) - 1) * DYR + 14.0     # axis
HF = YA + 12.5
F = S.Fig(HF); T = F.text; cv = F.cv


def title(x, letter, text):
    T(x, 7, letter, size=FS['letter'], weight='bold'); T(x + 11, 7.3, text, size=FS['title'], weight='bold')


def pack(xs, sx, sep=2.3, step=1.9, limit=1):
    """Vertical offsets (pt) so that neighbouring task dots do not overlap; x positions are exact."""
    order = np.argsort(xs); off = np.zeros(len(xs)); done = []
    for i in order:
        k = 0; cand = 0.0
        while any(abs(sx(xs[i]) - sx(xs[j])) < sep and abs(cand - off[j]) < step * 0.95 for j in done) and k < 2 * limit:
            k += 1; cand = ((k + 1) // 2) * step * (1 if k % 2 else -1)
        off[i] = cand; done.append(i)
    return off


def forest(x0, x1, fam, lo, hi, ticks, key, ci_key, fmt, dots):
    sx = lambda v: x0 + (v - lo) / (hi - lo) * (x1 - x0)
    T((x0 + x1) / 2, 19.5, 'Camera' if fam == 'Camera' else 'Sensor Noise', size=7.4, color=K, ha='center')
    F.line([(x0, YA), (x1, YA)], color=K, lw=0.7, z=3)
    for v, lab in ticks:
        F.line([(sx(v), YA), (sx(v), YA + 2.6)], color=K, lw=0.7, z=3)
        T(sx(v), YA + 7.8, lab, size=FS['tick'], color=K, ha='center')
    F.line([(sx(0), Y0 - 9), (sx(0), YA)], color='#7a7a7a', lw=0.7, ls=(0, (2, 1.6)), z=2.8)
    for k, (arm, _) in enumerate(ROWS):
        r = D[fam]['rows'][arm]; y = Y0 + k * DYR; m = r[key]
        if dots:
            xs = np.array(r['task_reduction']); assert xs.min() > lo and xs.max() < hi, (fam, arm, xs.min(), xs.max())
            cv.scatter([sx(v) for v in xs], y + 6.3 + pack(xs, sx), s=3.6, color=ACC, alpha=0.32, lw=0, zorder=3)   # tasks, as in Figure 3
        if r[ci_key] is not None:                              # one of the six prespecified comparisons
            F.line([(sx(r[ci_key][0]), y), (sx(r[ci_key][1]), y)], color=ACC, lw=1.3, z=3.6)
            cv.add_patch(Circle((sx(m), y), 2.6, fc=ACC, ec='#ffffff', lw=0.7, zorder=4))
        else:                                                   # reference arm: no interval
            cv.add_patch(Circle((sx(m), y), 2.3, fc='#ffffff', ec=ACC, lw=1.0, zorder=4))
        T(sx(m), y - 6.9, fmt(m), size=FS['tick'], color=K, ha='center', weight='bold', zorder=5, bbox=dict(fc='#ffffff', ec='none', pad=0.5))


LX = 1.0
T(LX, 19.5, 'Restoration vs.', size=FS['tick'], color=SUB)
for k, (_, lab) in enumerate(ROWS): T(LX, Y0 + k * DYR + 0.2, lab, size=7.4, color=K)
AX = 104.0
title(AX, 'a', 'Command-error reduction')
signed = lambda v: f'{v:+.4f}'.replace('-', '−').replace('+', '') if v > 0 else f'{v:.4f}'.replace('-', '−')
forest(AX + 4, AX + 142, 'Camera', -0.06, 0.45, ((0, '0'), (0.2, '0.2'), (0.4, '0.4')), 'reduction', 'reduction_ci', signed, True)
forest(AX + 156, AX + 262, 'Sensor', -0.03, 0.08, ((0, '0'), (0.04, '0.04'), (0.08, '0.08')), 'reduction', 'reduction_ci', signed, True)
BX = 384.0
title(BX, 'b', 'Task-success gain (pp)')
pp = lambda v: f'{v:+.2f}'.replace('-', '−')
forest(BX + 4, BX + 52, 'Camera', -14, 14, ((-10, '−10'), (0, '0'), (10, '10')), 'success_gain', 'success_ci', pp, False)
forest(BX + 62, BX + 110, 'Sensor', -14, 14, ((-10, '−10'), (0, '0'), (10, '10')), 'success_gain', 'success_ci', pp, False)

import os
os.makedirs('figs_v3', exist_ok=True)
out = sys.argv[1] if len(sys.argv) > 1 else 'figs_v3/figure4_restoration'
F.save(out)
print('saved', out, 'height', HF)
