"""Figure 1, TEASER (a: architecture in the classic idiom, b: restoration result): a plain architecture diagram in the classic idiom (grey containers, pastel blocks, thin dark
outlines, black arrows, labels only on blocks). No numbers and no explanatory text: the only things added to the
frozen pi0.5 model are the three response reads and the restoration at L9, in the paper's accent colour."""
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mi
from matplotlib.patches import Circle, Rectangle, FancyBboxPatch
from figstyle import *

apply()
HF = 196                      # height of the diagram itself
DY = 14.0                     # title row above it
f = Fig(HF + DY); f.cv.set_ylim(HF, -DY)
TT = 7.0 - DY                 # baseline of the title row
R = f.fig.canvas.get_renderer()
K = '#222222'                 # outline / arrows
GREY = '#f2f2f2'              # containers
C_VIT, C_LIN, C_EMB = '#d8ecd2', '#dddaf3', '#fad6dd'
C_VLM, C_AE = '#fdf0c0', '#fcd9b8'
C_TOK = {'lin': '#dddaf3', 'lin2': '#efedfa', 'txt': '#fad6dd', 'act': '#fcd9b8'}   # a token takes the colour of the block that emits it
FS_B = 7.8


def T(x, y, s, size=FS_B, color=K, ha='left', weight='normal', **k):
    t = f.text(x, y, s, size=size, color=color, ha=ha, weight=weight, **k)
    return t.get_window_extent(R).x1 * 72 / f.fig.dpi


def block(x, y, w, h, label, fc, size=FS_B, r=3.2):
    f.box(x, y, w, h, fc=fc, ec=K, lw=0.75, r=r, z=3)
    if label: T(x + w / 2, y + h / 2 + 0.2, label, size=size, ha='center')


def arr(pts, color=K, lw=0.9, ls='-', hs=4.4, z=3.4, r=4.0):
    f.arrow(pts, color=color, lw=lw, ls=ls, hs=hs, z=z, r=r)


def tokens(xc, y, kinds, s=8.0, g=2.0):
    w = len(kinds) * s + (len(kinds) - 1) * g; x0 = xc - w / 2
    for i, k in enumerate(kinds):
        f.box(x0 + i * (s + g), y, s, s, fc=C_TOK[k], ec=K, lw=0.6, r=1.8, z=3)
    return x0, x0 + w


def frame(x, y, s, path):
    f.cv.imshow(mi.imread(path), extent=(x, x + s, y + s, y), zorder=2, aspect='auto', interpolation='lanczos')
    f.cv.set_xlim(0, W_PT); f.cv.set_ylim(HF, -DY)
    f.cv.add_patch(Rectangle((x, y), s, s, fc='none', ec=K, lw=0.75, zorder=2.5))


import csv
# ------------------------------------------------------------------ a: inputs and encoders
T(1, TT, 'a', size=9.5, weight='bold', color=INK)
x = T(12, TT + 0.3, 'Reading and restoring visual responses', size=8.2, weight='bold', color=INK)
x = T(x + 3.5, TT + 0.3, 'in frozen', size=8.2, color=INK); x = T(x + 2.6, TT + 0.3, r'$\pi_{0.5}$', size=8.8, color=INK)
print('a title ends at', round(x, 1))
FSZ = 56
T(1, 5.0, 'clean'); frame(1, 10, FSZ, 'extract/frame_0.png')
T(1, 81.5, 'perturbed'); frame(1, 87, FSZ, 'extract/frame_1.png')
block(72, 65.5, 30, 22, 'ViT', C_VIT)
arr([(57, 38), (64, 38), (64, 71.5), (72, 71.5)])
arr([(57, 115), (64, 115), (64, 81.5), (72, 81.5)])
block(114, 65.5, 33, 22, 'Linear', C_LIN)
arr([(102, 76.5), (114, 76.5)])
T(1, 183, 'Instruction')
block(72, 172, 46, 22, 'Embedding', C_EMB)
arr([(43, 183), (72, 183)])

# ------------------------------------------------------------------ two stacks
YS = {'L18': 35.0, 'L9': 83.0, 'L1': 118.0}; BH = 17.0
VX, VW = 155.0, 94.0
AX, AW = 270.0, 68.0
CB = 143.0                                                  # container bottom
for x, w in ((VX, VW), (AX, AW)):
    f.box(x, 20, w, CB - 20, fc=GREY, ec=K, lw=0.75, r=6, z=1.5)
T(VX + VW / 2, 27.5, 'VLM prefix', ha='center')
T(AX + AW / 2, 27.5, 'Action expert', ha='center')
for L, y in YS.items():
    block(VX + 9, y, VW - 18, BH, L, C_VLM)
    block(AX + 8, y, AW - 16, BH, L, C_AE)
    arr([(VX + VW - 9, y + BH / 2), (AX + 8, y + BH / 2)])


def skip(xc, y_from, y_to):                                 # elided layers: dashed shaft, solid head
    f.line([(xc, y_from), (xc, y_to + 6.2)], color=K, lw=0.9, ls=(0, (1.6, 1.5)), z=3.3)
    arr([(xc, y_to + 6.4), (xc, y_to)])


XV, XA = VX + VW / 2, AX + AW / 2
NY = 72.0                                                   # the restoration acts on the state leaving L9
skip(XV, YS['L1'], YS['L9'] + BH); skip(XV, YS['L9'], YS['L18'] + BH)
skip(XA, YS['L1'], YS['L9'] + BH); skip(XA, NY - 4.4, YS['L18'] + BH)
arr([(XA, YS['L9']), (XA, NY + 4.5)], hs=4.0)
T((VX + VW + AX) / 2, YS['L18'] + BH / 2 - 6.5, 'K/V', ha='center')

# prefix tokens: one outlined group per source, each fed by the block that produces it, each feeding the VLM
TY = 158.0
GI, GT = (160.0, 44.0), (210.0, 34.0)
for (gx, gw), kinds in ((GI, ['lin', 'lin', 'lin2', 'lin2']), (GT, ['txt'] * 3)):
    f.box(gx, TY - 3, gw, 14, fc='#ffffff', ec=K, lw=0.75, r=3.4, z=2.8)
    tokens(gx + gw / 2, TY, kinds)
    arr([(gx + gw / 2, TY - 3), (gx + gw / 2, CB)])
arr([(130.5, 87.5), (130.5, TY + 4), (GI[0], TY + 4)])
arr([(118, 183), (GT[0] + GT[1] / 2, 183), (GT[0] + GT[1] / 2, TY + 11)])
a0, a1 = tokens(AX + AW / 2, TY, ['act'] * 5)
arr([(AX + AW / 2, TY - 1.5), (AX + AW / 2, CB)])
T(a1 + 5, TY + 4.3, 'Noisy actions')
tokens(AX + AW / 2, 1.5, ['act'] * 5)
arr([(AX + AW / 2, 20), (AX + AW / 2, 10.2)])
T(a1 + 5, 5.8, 'Commands')

# ------------------------------------------------------------------ what this paper adds
# three reads of the response (out of the model), and one closed loop at L9:
# response -> its part along 16 shared directions -> subtracted from the state that leaves L9
bx = AX + AW - 8
PX = 394.0                                                   # panel b box starts here
for L, y in YS.items():
    yr = y + BH / 2 + (4.0 if L == 'L9' else 0.0)
    arr([(bx, yr), (bx + 21, yr)], color=ACC, lw=1.1, hs=4.6, z=3.6)
    T(bx + 24, yr + 0.2, r'$\Delta H_{%s}$' % L[1:], size=8.4, color=K)
y9 = YS['L9'] + BH / 2 + 4.0
OX, OW, OH = 344.0, 42.0, 19.2
f.box(OX, NY - OH / 2, OW, OH, fc=ACC_T, ec=ACC, lw=0.9, r=3.2, z=3)
T(OX + OW / 2, NY - 4.1, '16 shared', size=7.0, ha='center'); T(OX + OW / 2, NY + 4.3, 'directions', size=7.0, ha='center')
arr([(OX + OW / 2, y9 - 5.6), (OX + OW / 2, NY + OH / 2)], color=ACC, lw=1.1, hs=4.6, z=3.6)
arr([(OX, NY), (XA + 4.6, NY)], color=ACC, lw=1.1, hs=4.6, z=3.6)
f.cv.add_patch(Circle((XA, NY), 4.4, fc='#ffffff', ec=ACC, lw=1.0, zorder=4))
f.line([(XA - 2.3, NY), (XA + 2.3, NY)], color=ACC, lw=1.2, z=4.2)
T((XA + 4.6 + AX + AW) / 2, NY - 6.2, 'restore', size=7.0, ha='center')

# ------------------------------------------------------------------ b: what restoration does (same data as Figure 4)
# one box, same height as the two model containers, tinted in the accent so it reads as the result of the blue loop
rows = {(r['family'], r['comparator']): r for r in csv.DictReader(open('data/fig4_summary_recomputed.csv'))}
BW = W_PT - 0.6 - PX
BB = 194.0
f.box(PX, 1.5, BW, BB - 1.5, fc='#f6f9fd', ec=ACC, lw=0.8, r=6, z=1.5)
T(PX + 1, TT, 'b', size=9.5, weight='bold', color=INK)
x = T(PX + 11, TT + 0.3, 'Effect of restoration', size=8.2, weight='bold', color=INK); print('b title ends at', round(x, 1), 'limit', W_PT - 0.6)
TX0 = PX + 6.5
AX0, AX1 = PX + 40, PX + BW - 8


def forest(t, title, lo, hi, ticks, key, fmt):
    T(TX0, t, title, size=7.6, color=INK)
    sx = lambda v: AX0 + (v - lo) / (hi - lo) * (AX1 - AX0)
    ya = t + 48
    f.line([(AX0, ya), (AX1, ya)], color=K, lw=0.7, z=3)
    for v, lab in ticks:
        f.line([(sx(v), ya), (sx(v), ya + 2.6)], color=K, lw=0.7, z=3)
        T(sx(v), ya + 7.8, lab, size=7.0, color=K, ha='center')
    f.line([(sx(0), t + 9), (sx(0), ya)], color='#7a7a7a', lw=0.7, ls=(0, (2, 1.6)), z=2.8)
    for k, (fam, label) in enumerate((('Camera', ('Camera',)), ('Sensor', ('Sensor', 'Noise')))):
        r = rows[(fam, 'Unedited')]; y = t + 20 + k * 17
        m, a, b = (float(r[c]) for c in key)
        if len(label) == 1:
            T(TX0, y + 0.2, label[0], size=7.0, color=K)
        else:                                                   # the family name on two lines: the panel is too narrow for one
            T(TX0, y - 3.5, label[0], size=7.0, color=K); T(TX0, y + 4.1, label[1], size=7.0, color=K)
        f.line([(sx(a), y), (sx(b), y)], color=ACC, lw=1.3, z=3.6)
        f.cv.add_patch(Circle((sx(m), y), 2.6, fc=ACC, ec='#ffffff', lw=0.7, zorder=4))
        f.text(sx(m), y - 6.9, fmt(m), size=7.0, color=K, ha='center', weight='bold', zorder=5,
               bbox=dict(fc='#f6f9fd', ec='none', pad=0.5))


T(TX0, 13.0, 'vs. the unedited run', size=7.0, color=SUB)
forest(27.5, 'Command-error reduction', -0.02, 0.17, ((0, '0'), (0.1, '0.1')), ('mean_of_dots', 'ci_lo', 'ci_hi'), lambda v: f'{v:.4f}')
forest(112.5, 'Task-success gain (pp)', -12, 14, ((-10, '−10'), (0, '0'), (10, '10')), ('succ_gain_pp', 'succ_lo', 'succ_hi'), lambda v: f'+{v:.2f}')

out = sys.argv[1] if len(sys.argv) > 1 else 'wire/fig1_teaser'
f.save(out)
print('saved', out)
