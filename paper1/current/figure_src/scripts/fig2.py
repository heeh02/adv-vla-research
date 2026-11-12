"""Figure 2 (time and protocol), same visual idiom as Figure 1: pastel blocks, thin dark outlines, black arrows,
blue = what this paper adds. Four small diagrams:
  a  one query: ten generation steps, then five controls and a new observation
  b  recording: clean and perturbed branches share the initial noise and then evolve separately
  c  restoration: at every generation step a clean reference is computed on the same plan sample and time
  d  execution study: five runs per setting from one matched state
All content follows Sec. 3.1-3.2 and Sec. 4.3 of the paper; nothing here is data."""
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mi
from matplotlib.patches import Circle, Rectangle, PathPatch
from matplotlib.path import Path
from figstyle import *

apply()
HF = 203
f = Fig(HF)
R = f.fig.canvas.get_renderer()
K = '#222222'
C_VLM, C_AE, C_REF = '#fdf0c0', '#fcd9b8', '#fef0e2'
FS_B = 7.8


def T(x, y, s, size=FS_B, color=K, ha='left', weight='normal', **k):
    t = f.text(x, y, s, size=size, color=color, ha=ha, weight=weight, **k)
    return t.get_window_extent(R).x1 * 72 / f.fig.dpi


def block(x, y, w, h, label, fc, size=FS_B, r=3.2, ec=K, ls='-'):
    f.box(x, y, w, h, fc=fc, ec=ec, lw=0.75, r=r, z=3, ls=ls)
    if label: T(x + w / 2, y + h / 2 + 0.2, label, size=size, ha='center')


def arr(pts, color=K, lw=0.9, ls='-', hs=4.4, z=3.4, r=4.0):
    f.arrow(pts, color=color, lw=lw, ls=ls, hs=hs, z=z, r=r)


def title(x, y, letter, text):
    T(x, y, letter, size=9.5, weight='bold', color=INK)
    T(x + 11, y + 0.3, text, size=8.2, weight='bold', color=INK)


def dots(x, y):
    for dx in (0, 4, 8):
        f.cv.add_patch(Circle((x + dx, y), 0.75, fc=K, ec='none', zorder=3.4))


def frame(x, y, s, path):
    f.cv.imshow(mi.imread(path), extent=(x, x + s, y + s, y), zorder=2, aspect='auto', interpolation='lanczos')
    f.cv.set_xlim(0, W_PT); f.cv.set_ylim(f.H, 0)
    f.cv.add_patch(Rectangle((x, y), s, s, fc='none', ec=K, lw=0.75, zorder=2.5))


BW_, BH_ = 30.0, 18.0          # one action-expert pass at generation step j

# =================================================================== a: one query
title(1, 7, 'a', 'One query: ten generation steps, then five controls')
yc = 50.0
frame(1, yc - 20, 40, 'extract/frame_0.png')
arr([(41, yc), (54, yc)])
block(54, yc - 10, 48, 20, 'VLM prefix', C_VLM)
arr([(102, yc), (120, yc)])
block(120, yc - 10, 58, 20, 'Action expert', C_AE)
arr([(168, yc - 10), (168, yc - 23), (130, yc - 23), (130, yc - 10)])
T(149, yc - 28.5, '×10', ha='center')
T(126, yc + 24.5, r'$x_0$', size=8.4, ha='center'); arr([(126, yc + 19.5), (126, yc + 10)])
arr([(178, yc), (190, yc)])
S_, G_ = 4.3, 0.75
for i in range(10):
    f.box(190 + i * (S_ + G_), yc - S_ / 2, S_, S_, fc=('#f3a968' if i < 5 else '#fdebd9'), ec=K, lw=0.5, r=0.9, z=3)
xm = 190 + 2.5 * (S_ + G_) - G_ / 2
arr([(xm, yc + 4.2), (xm, yc + 36), (21, yc + 36), (21, yc + 20.6)])
T(111, yc + 43, 'first 5 commands executed, then a new observation', size=7.0, ha='center')

# =================================================================== b: recording
BX = 256.0
title(BX, 7, 'b', 'Recording: same noise, two branches')
y1, y2 = 36.0, 80.0
T(BX + 3, (y1 + y2) / 2, r'$x_0$', size=8.4)
for y in (y1, y2):
    arr([(BX + 14, (y1 + y2) / 2), (BX + 21, (y1 + y2) / 2), (BX + 21, y), (BX + 31, y)])
XS = (BX + 31, BX + 75, BX + 136)
for y in (y1, y2):
    for x, lab in zip(XS, (r'$j=0$', r'$j=1$', r'$j=9$')):
        block(x, y - BH_ / 2, BW_, BH_, lab, C_AE)
    arr([(XS[0] + BW_, y), (XS[1], y)])
    f.line([(XS[1] + BW_, y), (XS[1] + BW_ + 7, y)], color=K, lw=0.9, z=3.4)
    dots(XS[1] + BW_ + 11, y)
    arr([(XS[2] - 9, y), (XS[2], y)])
    arr([(XS[2] + BW_, y), (XS[2] + BW_ + 13, y)])
T(XS[2] + BW_ + 17, y1 + 0.2, 'clean'); T(XS[2] + BW_ + 17, y2 + 0.2, 'perturbed')
for x in XS:
    xc = x + BW_ / 2
    f.line([(xc, y1 + BH_ / 2), (xc, y2 - BH_ / 2)], color=ACC, lw=1.1, z=3.2)
    f.cv.add_patch(Circle((xc, (y1 + y2) / 2), 2.7, fc=ACC, ec='none', zorder=4))
T(XS[1] + BW_ / 2 + 6, (y1 + y2) / 2 + 0.2, r'$\Delta H_{\ell,j}$', size=8.4)

# =================================================================== c: restoration
Y0 = 113.0
title(1, Y0, 'c', 'Restoration: clean reference on the same plan, every step')
yr, ym = Y0 + 31, Y0 + 73
T(3, ym, r'$x_0$', size=8.4)
XC = (34.0, 92.0, 166.0)
prev = 15.0
for i, (x, lab) in enumerate(zip(XC, (r'$j=0$', r'$j=1$', r'$j=9$'))):
    block(x, ym - BH_ / 2, BW_, BH_, lab, C_AE)
    block(x, yr - BH_ / 2, BW_, BH_, lab, C_REF)
    jx = x - 10
    if i < 2: arr([(prev, ym), (x, ym)])
    else: arr([(x - 14, ym), (x, ym)])
    f.cv.add_patch(Circle((jx, ym), 1.3, fc=K, ec='none', zorder=3.6))
    arr([(jx, ym), (jx, yr), (x, yr)])
    arr([(x + BW_ / 2, yr + BH_ / 2), (x + BW_ / 2, ym - BH_ / 2)], color=ACC, lw=1.1, hs=4.6, z=3.6)
    prev = x + BW_
f.line([(XC[1] + BW_, ym), (XC[1] + BW_ + 8, ym)], color=K, lw=0.9, z=3.4)
dots(XC[1] + BW_ + 12, ym)
arr([(XC[2] + BW_, ym), (XC[2] + BW_ + 13, ym)])
T(XC[2] + BW_ + 17, ym + 0.2, 'perturbed'); T(XC[2] + BW_ + 17, yr + 0.2, 'clean')
T(XC[1] + BW_ / 2 + 6, (yr + ym) / 2 + 0.2, 'restore at L9', size=7.0)

# =================================================================== d: execution study
title(BX, Y0, 'd', 'Paired executions: five runs from one state')
x0, xe, ys = BX + 10, BX + 128, Y0 + 40
ends = {r'$T_0$': -27.0, r"$R_0,\,R_0'$": -5.0, r'$R_1$': 11.0, r'$T_1$': 31.0}


def run(dy, ls, lw=1.0, col=K):
    f.cv.add_patch(PathPatch(Path([(x0, ys), (x0 + 48, ys + 0.12 * dy), (xe, ys + dy)], [Path.MOVETO, Path.CURVE3, Path.CURVE3]), fc='none', ec=col, lw=lw, ls=ls, zorder=3.3, capstyle='butt'))


for lab, dy in ends.items():
    run(dy, (0, (3.2, 1.8)) if lab.startswith('$T') else '-')
    T(xe + 4, ys + dy + 0.2, lab, size=8.2)
run(-8.2, '-', lw=0.7, col='#8a8a8a')                          # the same-noise repeat R'_0 stays close to R_0
f.cv.add_patch(Circle((x0, ys), 2.2, fc=K, ec='none', zorder=4))
ya = Y0 + 80
f.line([(x0, ya), (xe, ya)], color=K, lw=0.7, z=3)
for frac, lab in ((0.22, '5'), (0.93, '60')):
    xx = x0 + frac * (xe - x0)
    f.line([(xx, ya), (xx, ya + 2.6)], color=K, lw=0.7, z=3)
    f.line([(xx, ys - 30), (xx, ya)], color='#9a9a9a', lw=0.6, ls=(0, (1.2, 1.8)), z=2.5)
    T(xx, ya + 7.6, lab, size=7.0, ha='center')
T(xe + 5, ya + 0.2, 'controls', size=7.0)
bx = xe + 40


def brace(ya_, yb_, lab):
    f.line([(bx - 3, ya_ + 1), (bx, ya_ + 1), (bx, yb_ - 1), (bx - 3, yb_ - 1)], color=K, lw=0.8, z=3.4)
    T(bx + 4, (ya_ + yb_) / 2 + 0.2, lab, size=7.0)


brace(ys - 27, ys - 5, 'perturbation')
brace(ys - 5, ys + 11, 'clean rerun')
brace(ys + 11, ys + 31, 'perturbation')

out = sys.argv[1] if len(sys.argv) > 1 else 'wire/fig2_new'
f.save(out)
print('saved', out)
