"""One fixed bridge-audit figure. Source numbers enter only through the cloud adapter."""
from pathlib import Path
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.text import Text
import analysis as A

NAME = 'F1-BRIDGE-ERROR'
STYLE = {'font.family': 'DejaVu Sans', 'font.size': 8, 'axes.labelsize': 7.5,
         'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5, 'pdf.fonttype': 42,
         'ps.fonttype': 42, 'svg.fonttype': 'none', 'figure.facecolor': 'white',
         'savefig.facecolor': 'white'}
CAPTION = (
    'Exploratory existing-data bridge audit of eight fixed DEV snapshots from two tasks and '
    'four initialization clusters, with q11/q22 repeated within each cluster. A: each point is '
    'the RMS old-G/new-Gstar discrepancy divided by the OLD reference tensor RMS, in percent. '
    'v0 and v5 use full native 10x32 vector-field tensors, including non-adapter7 coordinates; '
    'P, head-position read and full-write proxy each use their own tensor denominator. Head '
    'and full features are FP64 algebraic reconstructions, not native BF16 intervention writes. '
    'B: per controller-input coordinate, RMS[(Pnew-Nnew)-(Pold-Nold)] / RMS(Snew-Nnew), with '
    'first5 and full10 candidate positions separately shown. Each row retains all eight '
    'snapshots; vertical offsets follow fixed snapshot identity, not uncertainty. The unity '
    'line is a pulse-magnitude reference, not an equivalence, physical or significance '
    'threshold. Different fields and panels have different denominators and must not be '
    'interpreted as a causal amplification chain. Exact-zero ratios and undefined '
    'zero-denominator ratios use distinct < and x markers in a separate 0/NA lane when '
    'present. The full support table retains all values and the alternative S-R denominator. '
    'Strict <1/=1/>1 counts use unrounded source scalars; near-one rows are separately listed '
    'only for display-rounding sensitivity. Fresh C context, whole-graph equivalence, '
    'prediction transfer and physical motion are not measured. No single BF16, compiler or '
    'hardware cause is identified. Historical cache results retain their original domain; '
    'the internally qualified new Gstar S/R/P comparisons also remain valid.'
)


def limits(values):
    pos = [v for v in values if v is not None and v > 0]
    lo = min(math.floor(math.log10(min(pos))) if pos else -1, 0)
    hi = max(math.ceil(math.log10(max(pos))) if pos else 1, 0)
    if hi == lo: hi += 1
    special = any(v is None or v == 0 for v in values)
    lane = lo-.22*(hi-lo)
    return lo, hi, lane, special


def points(ax, groups, bounds, *, labels, unity=False):
    lo, hi, lane, special = bounds
    points_out = []
    for yi, values in enumerate(groups):
        for si, value in enumerate(values):
            x = math.log10(value) if value is not None and value > 0 else lane
            marker = 'o' if value is not None and value > 0 else ('<' if value == 0 else 'x')
            ax.scatter(x, yi+(si-3.5)*.068, marker=marker, s=13, color='#236C94',
                       linewidths=.8, zorder=3)
            points_out.append({'row': yi, 'snapshot': list(A.KEYS[si]), 'value': value,
                               'zero': value == 0, 'undefined': value is None})
    if unity: ax.axvline(0, color='#777777', ls='--', lw=.7)
    start = lane if special else lo
    ax.set_xlim(start-.06*(hi-lo), hi+.06*(hi-lo))
    ax.set_ylim(len(groups)-.5, -.5)
    ax.set_yticks(range(len(groups)), labels)
    step = max(1, math.ceil((hi-lo)/4))
    ticks = sorted(set(range(lo, hi+1, step)) | ({0} if lo <= 0 <= hi else set()))
    ax.set_xticks(ticks, [f'{10.**v:g}' for v in ticks])
    ax.tick_params(length=2, pad=3)
    ax.grid(axis='y', color='#dddddd', lw=.4)
    ax.spines[['top', 'right']].set_visible(False)
    if special: ax.text(lane, -.8, '0/NA', ha='center', fontsize=7.5)
    return points_out


def render(data, out):
    field_values = []
    for name in A.FIELDS_A:
        idx = data['fields'] if name in A.FIELDS else data['features']
        field_values.append([None if idx[k+(name,)]['relative_RMS_to_reference'] is None
                             else 100.*idx[k+(name,)]['relative_RMS_to_reference'] for k in A.KEYS])
    pulse = {w: [[data['endpoints'][k+('persistent_effect_own_baselines', w, d)]
                   ['residual_RMS_over_new_S_RMS'] for k in A.KEYS] for d in A.DIMS] for w in A.WINDOWS}
    both = [v for w in A.WINDOWS for row in pulse[w] for v in row]
    with plt.rc_context(STYLE):
        fig = plt.figure(figsize=(180/25.4, 84/25.4))
        if data['synthetic']:
            fig.text(.02, .99, 'SYNTHETIC', fontsize=7.5, va='top', color='#A54628')
        fig.text(.025, .91, 'A  Measured readout shift', fontsize=8.5, weight='bold')
        fig.text(.455, .91, 'B  Persistent-effect discrepancy / pulse', fontsize=8.5, weight='bold')
        ax = fig.add_axes([.14, .23, .235, .58])
        shown = {'A': points(ax, field_values, limits([v for r in field_values for v in r]),
                            labels=('v0 (10×32)', 'v5 (10×32)', 'P at j5', 'Head read', 'Full write'))}
        ax.set_xlabel('Relative RMS (%)', labelpad=6)
        for wi, window in enumerate(A.WINDOWS):
            ax = fig.add_axes([.50+wi*.275, .23, .205, .58])
            shown[window] = points(ax, pulse[window], limits(both),
                                   labels=('Tx', 'Ty', 'Tz', 'Rx', 'Ry', 'Rz', 'Grip'), unity=True)
            ax.set_title('First 5' if wi == 0 else 'Full 10', fontsize=8, pad=6)
            ax.set_xlabel('Discrepancy / S-N RMS', labelpad=6)
        fig.text(.025, .075, '8 snapshots / 4 roots; repeated queries. Each row has its own denominator.', fontsize=7.5)
        fig.text(.025, .027, 'Dashed 1: pulse-magnitude reference, not an acceptance threshold.', fontsize=7.5)
        fig.canvas.draw()
        rr = fig.canvas.get_renderer(); problems=[]; minimum=100.
        for text in fig.findobj(Text):
            if not text.get_visible() or not text.get_text(): continue
            minimum=min(minimum, text.get_fontsize()); b=text.get_window_extent(rr)
            if b.x0 < -1 or b.y0 < -1 or b.x1 > fig.bbox.width+1 or b.y1 > fig.bbox.height+1:
                problems.append(text.get_text())
        A.need(minimum >= 7.5 and not problems, 'figure_text_bbox_and_minimum_font')
        for ext in ('pdf', 'svg', 'png'): fig.savefig(Path(out)/(NAME+'.'+ext), dpi=300)
        plt.close(fig)
    return {'caption': CAPTION, 'plotted_values': shown,
            'layout': {'width_mm': 180, 'height_mm': 84, 'minimum_font_points': minimum,
                       'text_bbox': 'PASS', 'PNG_dpi': 300,
                       'A_points': 40, 'B_points': 112,
                       'different_denominators_not_causal_amplification': True}}
