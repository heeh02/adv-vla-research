"""Redraw the teaser and measurement architecture from frozen sources.

No model, fit, bootstrap or synthetic observation is used. Camera PNGs are shown
without photometric changes. The teaser displays individual archived means and
existing intervals; arrows in the architecture describe computation, not a
cross-cohort causal identification result.
"""
from pathlib import Path
import json
import os
import tempfile
import hashlib
os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'advvla-overview-refinement'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parent
OUT = ROOT.parent / 'figures'
OUT.mkdir(exist_ok=True)
D = json.loads((ROOT / 'data/figure_data.json').read_text())
C = json.loads((ROOT / 'data/overview_context.json').read_text())
AP = json.loads((ROOT / 'data/r11_attention_profiles.json').read_text())
WIDTH = 6.875
INK = '#243247'
MUTED = '#64748B'
LINE = '#CDD6DF'
GRID = '#E5EAF0'
RESPONSE = '#0F766E'
CLEAN = '#64748B'
RANDOM = '#94A3B8'
PREFIX = '#3677A0'
ATTN = '#8062A5'
EXPERT = '#97633A'
OUTCOME = '#344454'
plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 8,
    'text.color': INK, 'axes.labelcolor': INK, 'xtick.color': MUTED,
    'ytick.color': INK, 'pdf.fonttype': 42, 'ps.fonttype': 42,
    'svg.fonttype': 'none', 'svg.hashsalt': 'reader-overview-20261031-v1',
    'figure.facecolor': 'white', 'savefig.facecolor': 'white',
    'axes.linewidth': .6, 'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5,
})


def canvas(height):
    fig = plt.figure(figsize=(WIDTH, height))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, WIDTH), ylim=(0, height))
    ax.axis('off')
    return fig, ax


def text(ax, x, y, s, size=8, color=INK, weight='normal', ha='left', va='center', **kw):
    return ax.text(x, y, s, fontsize=size, color=color, weight=weight,
                   ha=ha, va=va, linespacing=1.2, **kw)


def box(ax, x, y, w, h, fc='white', ec=LINE, lw=.75, r=.035):
    p = FancyBboxPatch((x, y), w, h,
                      boxstyle=f'round,pad=0,rounding_size={r}',
                      linewidth=lw, facecolor=fc, edgecolor=ec)
    ax.add_patch(p)
    return p


def arrow(ax, pts, color=MUTED, lw=.8, style='-', head=6):
    if len(pts) > 2:
        ax.plot(*zip(*pts[:-1]), color=color, lw=lw, ls=style,
                solid_capstyle='round', solid_joinstyle='round')
    ax.add_patch(FancyArrowPatch(pts[-2], pts[-1], arrowstyle='-|>',
                 mutation_scale=head, linewidth=lw, color=color,
                 linestyle=style, shrinkA=0, shrinkB=0))


def probe(ax, x, y, color, filled=False, r=.029):
    ax.add_patch(Circle((x, y), r, edgecolor=color,
                       facecolor=color if filled else 'white', lw=1.0, zorder=8))


def image_at(fig, filename, x, y, size):
    a = fig.add_axes([x / WIDTH, y / fig.get_figheight(),
                      size / WIDTH, size / fig.get_figheight()])
    a.imshow(plt.imread(ROOT / 'assets' / filename), interpolation='none')
    a.axis('off')


def chart(fig, rect):
    x, y, w, h = rect
    a = fig.add_axes([x / WIDTH, y / fig.get_figheight(),
                      w / WIDTH, h / fig.get_figheight()])
    for s in ['top', 'right', 'left']:
        a.spines[s].set_visible(False)
    a.spines['bottom'].set_color(LINE)
    a.tick_params(length=2.5, width=.6, pad=2)
    return a


def save(fig, name):
    fig.canvas.draw()
    # Report technical font range and bounds; visual review remains mandatory.
    renderer = fig.canvas.get_renderer()
    outside = []
    sizes = []
    for t in fig.findobj(matplotlib.text.Text):
        if not t.get_visible() or not t.get_text():
            continue
        sizes.append(float(t.get_fontsize()))
        b = t.get_window_extent(renderer)
        f = fig.bbox
        if b.x0 < f.x0-1 or b.x1 > f.x1+1 or b.y0 < f.y0-1 or b.y1 > f.y1+1:
            outside.append(t.get_text())
    for ext in ['pdf', 'svg', 'png']:
        meta = {'Title': name, 'Author': 'Anonymous Authors', 'CreationDate': None,
                'ModDate': None} if ext == 'pdf' else {'Date': None, 'Creator': 'Anonymous Authors'} if ext == 'svg' else None
        fig.savefig(OUT / f'{name}.{ext}', dpi=240, metadata=meta)
    qa = {'width_in': WIDTH, 'height_in': float(fig.get_figheight()),
          'min_font_pt': min(sizes), 'max_font_pt': max(sizes),
          'text_outside_canvas': outside}
    (OUT / f'{name}-layout.json').write_text(json.dumps(qa, indent=2))
    plt.close(fig)
    return qa


def overview():
    # Two independent empirical contrasts: one visual input triptych, two outcome scales.
    fig, ax = canvas(2.52)
    text(ax,.04,2.405,'(a) Large command changes, equal success counts',8.2,weight='bold')
    text(ax,3.65,2.405,'(b) Small motion differences, lower success',8.2,weight='bold')
    text(ax,.04,2.235,r'$\pi_{0.5}$ · LIBERO · 80 initial states on 40 tasks',7.15,color=MUTED)
    text(ax,3.65,2.235,r'$\pi_0$ · LIBERO-Plus · Sensor Noise',7.15,color=MUTED)
    ax.plot([3.48,3.48],[.16,2.12],color=LINE,lw=.5)
    for i,(x,lab) in enumerate([(.12,'Clean'),(1.25,'Headlight off'),(2.38,'Matched dimming')]):
        text(ax,x+.43,2.085,lab,7.45,ha='center')
        image_at(fig,f'input_{i}.png',x,1.17,.86)
    text(ax,1.69,1.065,'77/80 successes in each condition',7.9,weight='bold',ha='center')
    text(ax,1.69,.905,'Equal counts; different successful cases',7.05,color=MUTED,ha='center')
    text(ax,.04,.735,'Command change',7.6,weight='bold')
    text(ax,3.26,.735,'5.12×',9.0,color=RESPONSE,weight='bold',ha='right')
    # Quoted rounded means. Ratio is archived from unrounded means.
    means=[.0570,.0111]
    a=chart(fig,[.94,.295,2.32,.34])
    a.barh([1,0],means,height=.47,color=[RESPONSE,'#A4ADB7'],zorder=3)
    a.set(xlim=(0,.071),ylim=(-.55,1.55),yticks=[1,0],yticklabels=['Headlight off','Matched dimming'],xticks=[0,.03,.06],xticklabels=['0','0.03','0.06'])
    a.tick_params(axis='y',length=0,pad=3,labelsize=7.05)
    a.tick_params(axis='x',labelsize=7.0)
    a.xaxis.grid(True,color=GRID,lw=.5,zorder=0)
    for y,v in zip([1,0],means):a.text(v+.0014,y,f'{v:.4f}',fontsize=7.0,color=INK if y else MUTED,va='center')
    text(ax,1.69,.055,'First five translation commands (RMS)',7.15,ha='center')

    motion=next(r for r in D['motion'] if r['model']=='pi0_libero' and r['family']=='Sensor Noise')
    outcome=next(r for r in D['outcomes'] if r['model']=='pi0_libero' and r['family']=='Sensor Noise')
    text(ax,3.65,2.005,'End-effector separation after 5 controls',7.65,weight='bold')
    text(ax,3.65,1.835,'Relative to clean resampling',7.05,color=MUTED)
    b=chart(fig,[3.79,1.295,2.88,.40])
    v,lo,hi=[float(motion[k]) for k in ['ratio_of_task_means','ratio_ci95_low','ratio_ci95_high']]
    b.errorbar(v,.40,xerr=[[v-lo],[hi-v]],fmt='o',color=RESPONSE,ms=4.0,elinewidth=1.0,capsize=2.5,capthick=.85,zorder=4)
    b.axvline(1,color=MUTED,lw=.7,ls=(0,(3,2)))
    b.set(xlim=(0,1.3),ylim=(0,1),yticks=[],xticks=[0,.5,1],xticklabels=['0','0.5','1'])
    b.text(v,.82,f'{v:.3f}×',ha='center',fontsize=8.2,weight='bold',color=RESPONSE)
    b.text(1,1.18,'Resampling = 1',ha='center',fontsize=7.0,color=MUTED)
    text(ax,3.65,.905,'Success change, perturbed − clean',7.65,weight='bold')
    text(ax,3.65,.735,'Percentage points',7.05,color=MUTED)
    b=chart(fig,[3.79,.295,2.88,.37])
    v,lo,hi=[100*outcome[k] for k in ['point','ci98_75_low','ci98_75_high']]
    b.errorbar(v,.40,xerr=[[v-lo],[hi-v]],fmt='o',color='#C65C31',ms=4.0,elinewidth=1.0,capsize=2.5,capthick=.85,zorder=4)
    b.axvline(0,color=MUTED,lw=.7,ls=(0,(3,2)))
    b.set(xlim=(-35,8),ylim=(0,1),yticks=[],xticks=[-30,-20,-10,0])
    b.text(v,.80,f'{v:.1f}',ha='center',fontsize=8.2,weight='bold',color='#C65C31')
    b.text(0,1.18,'No change = 0',ha='center',fontsize=7.0,color=MUTED)
    return save(fig,'fig2_teaser')


def architecture():
    # Native print-point layout follows the old project r11 architecture.
    # Every represented layer keeps its own prefix/cache/attention/expert row.
    fig=plt.figure(figsize=(WIDTH,3.15))
    ax=fig.add_axes([0,0,1,1]);ax.set(xlim=(0,495),ylim=(234,0));ax.axis('off')
    colors=dict(ink=INK,muted=MUTED,prefix='#2D6687',prefix_fill='#EDF4F8',
                expert='#995827',expert_fill='#FCF2E8',attention='#765790',
                attention_fill='#F3EEF7',execution='#357765',execution_fill='#EFF6F2',restore='#9C3E64',border=LINE)
    labels=[]
    def tx(x,y,s,size=7.7,color='ink',weight='normal',ha='center',back=False):
        t=ax.text(x,y,s,ha=ha,va='center',fontsize=size,weight=weight,color=colors.get(color,color),zorder=8)
        if back:t.set_bbox(dict(facecolor='white',edgecolor='none',pad=.25))
        labels.append(t);return t
    def bx(x,y,w,h,color='border',fill=None,lw=.65):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0,rounding_size=2',fc=fill or colors.get(color+'_fill','white'),ec=colors.get(color,color),lw=lw,zorder=2))
    def ar(pts,color='muted',lw=.7,dotted=False,head=True):
        col=colors.get(color,color);ls=(0,(1.4,2.0)) if dotted else '-'
        for i,(p,q) in enumerate(zip(pts[:-1],pts[1:])):
            if i==len(pts)-2 and head:ax.add_patch(FancyArrowPatch(p,q,arrowstyle='-|>',mutation_scale=5.3,shrinkA=0,shrinkB=0,lw=lw,linestyle=ls,color=col,zorder=3))
            else:ax.plot([p[0],q[0]],[p[1],q[1]],color=col,lw=lw,linestyle=ls,zorder=3)
    def dt(x,y,color='prefix',native=False,r=1.7):
        ax.add_patch(Circle((x,y),r,facecolor=colors[color] if native else 'white',edgecolor=colors[color],lw=.8,zorder=7))
    tx(5,8,'Paired inputs',8.4,weight='medium',ha='left')
    tx(144,8,r'$\pi_{0.5}$ VLM prefix',8.4,'prefix','medium')
    tx(257,8,'Joint attention',8.4,'attention','medium')
    tx(358,8,'Action expert',8.4,'expert','medium')
    tx(454,8,'Generation',8.4,'expert','medium')
    tx(144,21,'2048 channels',7.5,'muted')
    tx(257,21,'prefix + action keys',7.5,'muted')
    tx(358,21,r'1024 · native $H_{\mathrm{AE}}$',7.5,'muted')
    tx(454,21,r'$j=0,\ldots,9$',7.5,'expert')
    for i,(x,lab) in enumerate([(5,'Clean'),(53,'Perturbed')]):
        ax.imshow(plt.imread(ROOT/'assets'/f'input_{i}.png'),extent=(x,x+44,78,34),interpolation='none',zorder=2)
        tx(x+22,86,lab,7.5)
    tx(51,97,'Same frozen policy',7.2,'muted')
    # Visual and language inputs share one compact group, with distinct ports.
    bx(5,112,91,43,'border','white')
    tx(50.5,124,'Main + wrist images',7.6,'prefix')
    tx(50.5,142,'Same instruction',7.6)
    tx(5,174,'Matched start',7.5,weight='medium',ha='left')
    tx(5,186,'State · instruction · noise',7.0,ha='left')

    PX,PW,AX,AW,EX,EW=108,72,226,61,324,70
    for L,y in [(18,60),(9,103),(1,146)]:
        bx(PX,y,PW,29,'prefix')
        tx(PX+7,y+8,f'L{L}',8.0,'prefix','medium',ha='left')
        bx(PX+5,y+17,49,10,'prefix','white',.45);tx(PX+29.5,y+22,'RMSNorm',7.2,'muted')
        dt(144,y+32);tx(137,y+34,r'$H_{\mathrm{VLM}}$',7.1,'prefix',ha='right',back=True)
        ar([(144,y+32),(144,y+29)],'prefix',head=False)
        dt(172,y+22);tx(172,y+12,'Z',7.6,'prefix')
        ar([(162,y+22),(170,y+22)],'prefix',head=False)
        for dx,dy in [(0,0),(2,-2)]:ax.add_patch(Rectangle((196+dx,y+16+dy),11,11,fc='white',ec=colors['prefix'],lw=.65,zorder=4))
        tx(204,y+7,'K/V',7.5,'prefix');dt(208,y+24);tx(216,y+26,'V',7.5,'prefix')
        ar([(174,y+22),(196,y+22)],'prefix');ar([(209,y+19),(AX,y+19)],'prefix')
        bx(AX,y,AW,30,'attention');tx(AX+AW/2,y+8,'Mix + project',7.4,'muted')
        dt(AX,y+23,'attention');tx(AX+AW/2,y+23,r'$P$',7.7,'attention')
        ar([(AX+AW/2,y+17),(AX+AW/2,y+13)],'attention',head=True)
        bx(EX,y,EW,30,'expert');tx(EX+5,y+8,f'L{L}',7.8,'expert','medium',ha='left');tx(EX+45,y+8,'Gate + FFN',7.3,'muted')
        tx(EX+EW/2,y+23,'Pre-norm',7.6,'muted')
        ar([(EX,y+23),(AX+AW,y+23)],'expert');tx(305,y+23,'Q/K/V',7.1,'expert',back=True)
        ar([(AX+AW,y+8),(EX,y+8)],'attention');dt(298,y+8,'attention');tx(307,y+8,'W',7.6,'attention',back=True)
        dt(358,y-6,'expert',True)
        tx(366,y-6,r'$H_{\rm AE}$',7.4,'expert',ha='left',back=True)
        if L==9:
            # The reference is from the current state, not the initial clean image.
            ax.add_patch(Rectangle((354.5,y-9.5),7,7,facecolor='#FBF2F6',
                                   edgecolor=colors['restore'],lw=.85,zorder=6))
            tx(350,y-6,'Same-state clean → restore',7.1,'restore',ha='right',back=True)
            ar([(351,y-6),(354.5,y-6)],'restore',lw=.7,head=False)
    for hi,lo in [(89,103),(132,146)]:
        ar([(144,lo),(144,hi+3)],'prefix',dotted=True)
        ar([(358,lo),(358,hi+1)],'expert',dotted=True)
    tx(204,39,'K/V cache',7.4,'prefix');tx(204,49,'fixed per query',7.0,'prefix')
    # Grouped image / wrist / language prefix tokens.
    for i,(lab,col) in enumerate([('main','prefix'),('wrist','execution'),('text','muted')]):
        xx=108+i*24
        ax.add_patch(Rectangle((xx,185),24,14,fc=colors.get(col+'_fill','#F1F3F4'),ec='white',lw=.5,zorder=4))
        tx(xx+12,192,lab,7.5,col)
    ar([(96,124),(100,124),(100,189),(108,189)],'prefix')
    ar([(96,142),(103,142),(103,196),(108,196)],'muted')
    ar([(144,185),(144,178)],'prefix')
    bx(324,185,70,16,'expert','white');tx(358,193,r'$10\times1024$',7.5,'expert')
    ar([(358,185),(358,176)],'expert')
    bx(247,185,68,16,'expert','white');tx(281,193,r'$x_j:\ 10\times32$',7.5,'expert')
    ar([(315,193),(324,193)],'expert');tx(358,211,r'time input: $t_j$',7.1,'expert')
    bx(324,34,70,17,'expert','white');tx(358,42.5,r'Norm + head $v_j$',7.4)
    ar([(358,60),(358,51)],'expert')
    bx(418,33,73,31,'expert');tx(454.5,42,'10 Euler steps',7.8,weight='medium');tx(454.5,55,r'$x_{j+1}=x_j-0.1v_j$',7.2)
    ar([(394,42.5),(418,42.5)],'expert')
    ar([(418,58),(406,58),(406,221),(281,221),(281,201)],'expert')
    tx(398,222,r'$j<9$',7.1,'expert',ha='right',back=True)
    ar([(454.5,64),(454.5,78)],'expert');tx(462,71,r'$j=9$',7.1,'expert',ha='left',back=True)
    bx(418,78,73,28,'execution');tx(454.5,86,'Final plan',7.8,weight='medium');tx(454.5,98,'10 positions × 32',7.4)
    ar([(454.5,106),(454.5,116)],'execution')
    bx(418,116,73,35,'execution');tx(454.5,124,'Unnormalize',7.6);tx(454.5,135,'7 control coords',7.5);tx(454.5,145,r'Issue $\leq5$',7.8,weight='medium')
    ar([(454.5,151),(454.5,162)],'execution')
    bx(418,162,73,31,'execution');tx(454.5,170,'Execute',7.8,weight='medium');tx(454.5,181,'EEF after 5 / 60',7.2);tx(454.5,189,'controls',7.1,'execution')
    ar([(454.5,193),(454.5,203)],'execution')
    bx(418,203,73,21,'execution','white');tx(454.5,210,'Task outcome',7.5,weight='medium');tx(454.5,219,'success / failure',7.3)
    # Physical feedback comes from execution, not from the terminal outcome.
    ar([(418,176),(412,176),(412,230),(2,230),(2,124),(5,124)],'execution',.75)
    tx(190,230,'New observation → replan',7.4,'execution',back=True)
    dt(112,210,'prefix');tx(118,210,'Prefix replay',7.1,'prefix',ha='left')
    dt(210,210,'attention');tx(216,210,'P/W replay',7.1,'attention',ha='left')
    qa=save(fig,'fig1_model_measurements')
    return qa



if __name__ == '__main__':
    qa={'fig2_teaser':overview(),'fig1_model_measurements':architecture()}
    print(json.dumps(qa,indent=2))
