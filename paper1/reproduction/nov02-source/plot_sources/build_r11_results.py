"""Render the original R11 result sequence from frozen, bundled scalar inputs.

    python3 plot_sources/build_r11_results.py
    python3 plot_sources/build_r11_results.py basis attention motion outcomes wrist

The narrative and measurement identities are those of R11. Estimates are read
from archived scalar results or deterministically recomputed from issued commands.
No experiment, fit or bootstrap is run. Motion retains all originally qualified cases;
missing planned rows remain in data/r11_result_sources.json.
"""
from pathlib import Path
import json
import sys
import shutil

import numpy as np
import build_results as common
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle, Patch
from matplotlib.ticker import FuncFormatter, NullLocator

plt = common.plt
ROOT = Path(__file__).resolve().parent
OUT = ROOT.parent / 'figures'
PREVIEW = ROOT / 'previews'
D = json.loads((ROOT / 'data/r11_result_sources.json').read_text())
FAMILIES = common.FAMILIES
COLORS = common.COLORS
SHORT = common.SHORT
INK, MUTED, GRID, LINE = common.INK, common.MUTED, common.GRID, common.LINE
TEAL, PURPLE, ORANGE = common.GEOMETRY, common.ATTENTION, common.BEHAVIOR


def setup():
    common.configure()
    plt.rcParams.update({'svg.hashsalt': 'advvla-r11-faithful-20261029',
                         'xtick.labelsize': 7.2, 'ytick.labelsize': 7.2,
                         'font.size': 7.8})


def save(fig, name):
    common.save(fig, name)
    shutil.copyfile(PREVIEW / (name + '.png'), OUT / (name + '.png'))


def heading(fig, x, y, letter, title, color=TEAL, size=8.5):
    fig.text(x, y, letter, color=color, fontsize=9.3, weight='bold', va='center')
    fig.text(x + .024, y, title, color=INK, fontsize=size, weight='bold', va='center')


def basis():
    # Recording comparisons and native-edit confirmation are separate populations.
    setup()
    fig = plt.figure(figsize=(7, 3.50))
    fig.text(.145, .98, 'Recording study: 404 pairs on 28 tasks', fontsize=8.0, weight='bold', va='top')
    fig.text(.755, .98, 'Native edits: confirmation', fontsize=7.6, weight='bold', va='top')
    for lo, hi in [(.145,.705),(.755,.995)]:
        fig.add_artist(Line2D([lo,hi],[.937,.937],transform=fig.transFigure,color=LINE,lw=.65))
    fig.add_artist(Line2D([.735,.735],[.15,.94],transform=fig.transFigure,color=LINE,lw=.6))
    boxes = [[.145, .181, .170, .519], [.360, .181, .155, .519],
             [.565, .181, .135, .519]]
    a,b,c = [fig.add_axes(p) for p in boxes]
    for ax,letter,title in zip([a,b,c],'abc',
            ['Equal-rank bases','Response vs. clean','No change = 1']):
        heading(fig,ax.get_position().x0,.895,letter,title,TEAL,7.3)
        ax.set_ylim(13.15,-.65);ax.set_yticks([]);common.style(ax,'x')
        for sep in [3.88,8.62]:ax.axhline(sep,color=GRID,lw=.8)
    handles=[Line2D([],[],color=INK,marker='o',ls='',ms=3.4,label='Response'),
             Line2D([],[],color=INK,marker='o',mfc='white',ls='',ms=3.4,label='Clean'),
             Line2D([],[],color='#98A7B6',marker='x',ls='',ms=3.5,label='Random')]
    fig.legend(handles=handles,loc='upper left',bbox_to_anchor=(.140,.856),
               ncol=1,frameon=False,fontsize=6.9,handletextpad=.45,labelspacing=.22,
               handlelength=.8,borderaxespad=0)
    fig.text(.360,.831,'Highest retention: L1\nLowest affine error: L18',fontsize=6.5,color=MUTED,
             va='top',linespacing=1.25)
    fig.legend(handles=handles[:2],loc='upper left',bbox_to_anchor=(.560,.856),
               ncol=1,frameon=False,fontsize=6.9,handletextpad=.45,labelspacing=.22,
               handlelength=.8,borderaxespad=0)
    source=common.D['basis']
    for layer,base_y,label in [(0,0,'L1'),(8,4.75,'L9'),(17,9.5,'L18')]:
        fy=fig.transFigure.inverted().transform(a.transData.transform((0,base_y+1.5)))[1]
        fig.text(.006,fy,label,fontsize=8.1,weight='bold',va='center')
        for k,fam in enumerate(FAMILIES):
            y=base_y+k;col=COLORS[fam]
            fy=fig.transFigure.inverted().transform(a.transData.transform((0,y)))[1]
            fig.text(.135,fy,SHORT[fam],ha='right',va='center',fontsize=7.1,color=INK)
            for model,off,marker,hollow,color in [
                    ('new_response',-.12,'o',False,col),
                    ('new_clean',.05,'o',True,col),
                    ('new_random',.20,'x',False,'#98A7B6')]:
                r=common.select(source['geometry'],family=fam,layer=layer,model=model)['retention']
                common.interval(a,100*r['mean'],100*r['interval_95'][0],100*r['interval_95'][1],
                                y+off,color,marker=marker,hollow=hollow,ms=3.2)
            r=common.select(source['contrasts'],family=fam,layer=layer)
            common.interval(b,100*r['relative_error_reduction'],100*r['relative_error_reduction_interval_95'][0],
                            100*r['relative_error_reduction_interval_95'][1],y,col,ms=3.3)
            for model,off,hollow in [('new_response',-.13,False),('new_clean',.13,True)]:
                r=common.select(source['readouts'],family=fam,layer=layer,model=model)
                c.plot(r['task_equal_mse']/r['task_equal_target_energy'],y+off,marker='o',ls='',
                       ms=3.4,color=col,mfc='white' if hollow else col,mew=.85)
    a.set_xlim(-2,100);a.set_xticks([0,25,50,75,100])
    b.set_xlim(-80,105);b.set_xticks([-50,0,50,100]);b.axvline(0,color=MUTED,ls=(0,(3,2)),lw=.7)
    c.set_xscale('log');c.set_xlim(.15,9);c.set_xticks([.25,1,4],['.25','1','4'])
    c.xaxis.set_minor_locator(NullLocator());c.axvspan(.15,1,color='#F1F7F6',zorder=0)
    c.axvline(1,color=MUTED,ls=(0,(3,2)),lw=.9)
    for ax,text in zip([a,b,c],['Residual energy\nretained (%)','Translation-error\nreduction (%)',
                               'Translation MSE /\nno-change MSE']):
        ax.set_xlabel(text,fontsize=6.9,labelpad=3)
    for ax, label in zip([a,b,c], ['Higher: more retained', 'Higher: less error', 'Lower: less error']):
        fig.text(ax.get_position().x0+ax.get_position().width/2,.025,label,ha='center',color=MUTED,fontsize=6.15)

    heading(fig,.755,.895,'d','Issued-command change',TEAL,7.2)
    direction = json.loads((ROOT/'command_direction_audit/SUMMARY.json').read_text())
    rows = {(r['family'],r['arm']):r for r in direction['CONF']['translation_family_arm_means']}
    dh = [Line2D([],[],color=INK,marker='>',ls='',ms=3.6,label='Toward clean'),
          Line2D([],[],color=INK,marker='D',mfc='white',ls='',ms=3.3,label='Orthogonal norm')]
    fig.legend(handles=dh,loc='upper left',bbox_to_anchor=(.755,.855),ncol=1,
               frameon=False,fontsize=6.6,handletextpad=.4,labelspacing=.3,
               handlelength=.8,borderaxespad=0)
    for fam, rect, title_y, xmax, ticks, ticklabels in [
            ('Camera Viewpoints',[.815,.495,.180,.205],.737,.130,[0,.06,.12],['0','.06','.12']),
            ('Sensor Noise',[.815,.181,.180,.205],.423,.021,[0,.01,.02],['0','.01','.02'])]:
        ax=fig.add_axes(rect)
        col=COLORS[fam]
        title='Camera · 27 tasks' if fam=='Camera Viewpoints' else 'Sensor · 30 tasks'
        fig.text(.755,title_y,title,fontsize=6.9,weight='bold',color=col,va='center')
        for y,arm in enumerate(['R_resp','R_clean','R_rand','R_full']):
            m=rows[(fam,arm)]['metrics']
            ax.plot(m['signed_along_clean']['value'],y-.12,marker='>',ls='',ms=3.8,color=col)
            ax.plot(m['orthogonal_norm']['value'],y+.12,marker='D',ls='',ms=3.3,
                    color=col,mfc='white',mew=.85)
        ax.set_yticks(range(4),['Response','Clean','Random',r'Full $H$'])
        ax.set_ylim(3.55,-.55);ax.set_xlim(-.025*xmax,xmax)
        ax.set_xticks(ticks,ticklabels);ax.tick_params(labelsize=6.5,pad=1.5)
        ax.axvline(0,color=MUTED,lw=.65)
        common.style(ax,'x')
    fig.text(.875,.091,'Translation adapter units',ha='center',fontsize=6.5,color=INK)
    fig.text(.875,.025,'Separate family scales · point means',ha='center',fontsize=6.0,color=MUTED)
    save(fig,'same_basis_translation')
    readout_controls()


def readout_controls():
    """Keep every original full-hidden and first-flow translation contrast."""
    setup()
    fig=plt.figure(figsize=(7,3.4))
    axes=[fig.add_axes([.15,.20,.33,.66]),fig.add_axes([.65,.20,.33,.66])]
    for ax,title,candidate,baseline in [
            (axes[0],'Response versus\nfull hidden state','new_response','full'),
            (axes[1],'First flow + response\nversus first flow','flow_plus_new','flow')]:
        ax.set_title(title,fontsize=8.6,pad=9,weight='bold',linespacing=1.1)
        for layer,base_y,label in [(0,0,'L1'),(8,4.75,'L9'),(17,9.5,'L18')]:
            for k,fam in enumerate(FAMILIES):
                row=common.select(D['basis_full_reference_contrasts'],family=fam,
                                  layer=layer,candidate=candidate,baseline=baseline)
                common.interval(ax,100*row['relative_error_reduction'],
                                100*row['relative_error_reduction_interval_95'][0],
                                100*row['relative_error_reduction_interval_95'][1],
                                base_y+k,COLORS[fam],ms=4)
        ys=[base_y+k for base_y in [0,4.75,9.5] for k in range(4)]
        ax.set_yticks(ys,[SHORT[f] for _ in range(3) for f in FAMILIES])
        ax.set_ylim(13.15,-.65);ax.set_xlim(-130,70);ax.set_xticks([-100,-50,0,50])
        ax.axvline(0,color=MUTED,lw=.8,ls=(0,(3,2)))
        for sep in [3.88,8.62]:ax.axhline(sep,color=GRID,lw=.8)
        common.style(ax,'x')
        for y,label in [(1.5,'L1'),(6.25,'L9'),(11,'L18')]:
            ax.text(-.20,y,label,transform=ax.get_yaxis_transform(),va='center',
                    ha='right',fontsize=8.0,weight='bold')
        ax.set_xlabel('Translation-error reduction (%)',fontsize=8)
    fig.text(.5,.025,'404 paired recordings · same fitted readouts · pointwise 95% task intervals',
             ha='center',fontsize=7.6,color=MUTED)
    save(fig,'supp_full_readout_controls')


def attention():
    style = common
    style.configure()
    plt.rcParams.update({'font.size': 7.5, 'xtick.labelsize': 7.1, 'ytick.labelsize': 7.1,
                         'svg.hashsalt': 'advvla-attention-h-preview-20261030'})
    profiles = json.loads((ROOT / 'data/r11_attention_profiles.json').read_text())
    old = D
    families = style.FAMILIES
    markers = ['o', 's', '^', 'D']
    assert len(profiles['profiles']) == 12
    assert {r['task_count'] for r in profiles['profiles']} == {28}
    assert sum(r['case_count'] for r in profiles['profiles'] if r['generation_call'] == 0) == 404
    lookup = {(r['family'], r['generation_call']): r for r in profiles['profiles']}
    keymap = {'O_shadow_P_full_key_TV': 'p_full_key_tv',
              'O_shadow_pre_gate_W__relative_l2': 'w_relative_l2'}
    old_errors = [abs(float(r['equal_task_mean']) -
                      lookup[(r['family'], int(r['generation_call']))][keymap[r['metric']]])
                  for r in old['attention_generation']]
    assert len(old_errors) == 24 and max(old_errors) < 1e-14

    # Three equal-sized generation panels; command associations receive extra width.
    fig = plt.figure(figsize=(7, 2.35))
    boxes = [[.045, .280, .165, .470], [.269, .280, .165, .470],
             [.493, .280, .165, .470], [.734, .280, .255, .470]]
    axes = [fig.add_axes(p) for p in boxes]
    headings = [('a', r'Probability $P$'), ('b', r'Attention output $W$'),
                ('c', r'Post-block state $H$'), ('d', 'Command association')]
    subtitles = ['Replay: full-key TV (%)', 'Replay: difference RMS',
                 'Native: difference RMS', r'Issued translation: rank $\rho$']
    for ax, (letter, title), subtitle in zip(axes, headings, subtitles):
        x = ax.get_position().x0
        fig.text(x-.024, .962, letter, color=style.ATTENTION, fontsize=9.2,
                 weight='bold', va='center')
        fig.text(x, .962, title, color=style.INK, fontsize=7.8,
                 weight='bold', va='center')
        fig.text(x, .837, subtitle, color=style.MUTED, fontsize=7.2, va='center')

    metrics = [('p_full_key_tv', 100), ('w_delta_rms', 1), ('ae_delta_rms', 1)]
    for ax, (metric, scale) in zip(axes[:3], metrics):
        for fam, mark in zip(families, markers):
            rows = sorted((r for r in profiles['profiles'] if r['family'] == fam),
                          key=lambda r: r['generation_call'])
            assert [r['generation_call'] for r in rows] == [0, 5, 9]
            ax.plot([0, 5, 9], [r[metric]*scale for r in rows],
                    color=style.COLORS[fam], marker=mark, ms=3.5, lw=1.35, mew=.65)
        ax.set_xlim(-.6, 9.6)
        ax.set_xticks([0, 5, 9])
        ax.set_xlabel('Generation step', labelpad=3, fontsize=7.0)
        style.style(ax, 'y')
    axes[0].set_ylim(0, 22); axes[0].set_yticks([0, 10, 20])
    axes[1].set_ylim(0, 2.8); axes[1].set_yticks([0, 1, 2])
    axes[2].set_ylim(0, 6.2); axes[2].set_yticks([0, 3, 6])

    ax = axes[3]
    signals = ['PG_H_main_relative_l2', 'P_full_key_TV',
               'W_pre_gate_relative_l2', 'AE_native_relative_l2']
    assert set(signals) == {r['signal'] for r in old['attention_association']}
    for fam, offset, mark in zip(families, [-.18, -.06, .06, .18], markers):
        rows = [style.select(old['attention_association'], family=fam, signal=s) for s in signals]
        y = np.array([float(r['rho_issued']) for r in rows])
        lo = np.array([float(r['rho_issued_low']) for r in rows])
        hi = np.array([float(r['rho_issued_high']) for r in rows])
        ax.errorbar(np.arange(4)+offset, y, yerr=[y-lo, hi-y], fmt=mark,
                    color=style.COLORS[fam], ms=3.2, elinewidth=.75, capsize=1.2, mew=.65)
    ax.set_xticks(range(4), ['Prefix H', 'P', 'W', 'Expert H'])
    ax.tick_params(axis='x', labelsize=6.7)
    ax.set_xlim(-.4, 3.4); ax.set_ylim(-.55, .9); ax.set_yticks([-.5, 0, .5])
    ax.set_xlabel('First generation step', fontsize=7.0, labelpad=3)
    ax.axhline(0, color=style.MUTED, ls=(0, (3, 2)), lw=.7)
    style.style(ax, 'y')

    handles = [Line2D([], [], color=style.COLORS[f], marker=m, ms=3.5, lw=1.2,
                      label=style.SHORT[f]) for f, m in zip(families, markers)]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.51, .006), ncol=4,
               frameon=False, columnspacing=1.9, fontsize=7.6,
               handlelength=1.4, handletextpad=.5)
    save(fig, 'fig4_attention')


def wrist():
    setup()
    fig = plt.figure(figsize=(3.375, 1.16))
    axes = [fig.add_axes([x,.25,.22,.44]) for x in [.095,.425,.755]]
    vals = [D['wrist']['conditional_tv_percent'], D['wrist']['mass_weighted_tv_pp'],
            D['wrist']['regional_l1_pp']]
    for ax, values, max_y, ticks in zip(axes, vals, [32,7.0,17], [[0,15,30],[0,3,6],[0,8,16]]):
        ax.plot([0, 1], values, color='#A5B3C0', lw=1.1)
        for x, v, col in zip([0, 1], values, ['#526D86', ORANGE]):
            ax.scatter([x], [v], c=col, s=18, zorder=3)
            ax.text(x, v + .08*max_y, f'{v:.2f}', ha='left' if x==0 else 'center', color=col,
                    fontsize=6.9, weight='bold')
        ax.set_ylim(0, max_y); ax.set_xlim(-.2, 1.2)
        ax.set_xticks([0, 1], ['L9', 'L18']); ax.set_yticks(ticks)
        ax.tick_params(labelsize=6.5,pad=1.5)
        common.style(ax, 'y')
    for x,title in zip([.205,.535,.865],['Conditional TV\n(%)','Mass-weighted TV\n(pp)',r'Regional $L_1$'+'\n(pp)']):
        fig.text(x,.98,title,ha='center',fontsize=7.0,weight='bold',color=PURPLE,va='top',linespacing=1.2)
    fig.text(.5,.02,'Same wrist pixels · 36 tasks',ha='center',fontsize=7.0,color=MUTED)
    save(fig, 'attention_rank_reversal_table')


def motion():
    setup()
    fig = plt.figure(figsize=(7, 2.37))
    pairs = [('pi05_libero', '5'), ('pi05_libero', '60'),
             ('pi0_libero', '5'), ('pi0_libero', '60')]
    positions = [.064, .305, .555, .796]
    outcome_colors = {'SS': '#72899D', 'SF': '#CF673B', 'FS': '#15968B', 'FF': '#8B65A5'}
    outcome_markers = {'SS': 'o', 'SF': '^', 'FS': 's', 'FF': 'x'}
    fig.text(.236, .985, r'$\pi_{0.5}$ · 516 / 536 pairs', ha='center', va='top', fontsize=9, weight='bold')
    fig.text(.733, .985, r'$\pi_0$ · 526 / 536 pairs', ha='center', va='top', fontsize=9, weight='bold')
    for i, (left, (model, step)) in enumerate(zip(positions, pairs)):
        ax = fig.add_axes([left, .284, .186, .492])
        rr = [r for r in D['motion_points'] if r['model'] == model and r['step'] == step
              and r['included'] == 'True']
        assert len(rr) == (516 if model == 'pi05_libero' else 526)
        for category in ['SS', 'FF', 'FS', 'SF']:
            rows = [r for r in rr if r['Y'] == category]
            ax.scatter([float(r['q0_norm']) for r in rows], [float(r['eef_mm']) for r in rows],
                       s=7.2 if category == 'SS' else 12, c=outcome_colors[category],
                       marker=outcome_markers[category],
                       alpha=.4 if category == 'SS' else .75,
                       linewidths=.55 if category == 'FF' else 0, rasterized=False)
        ax.set_xscale('symlog', linthresh=.01, linscale=.75)
        ax.set_yscale('symlog', linthresh=1, linscale=.75)
        ax.set_xlim(0, 1.6); ax.set_ylim(0, 440)
        ax.set_xticks([0, .01, .1, 1], ['0', '.01', '.1', '1'])
        ax.set_yticks([0, 1, 10, 100], ['0', '1', '10', '100'])
        ax.xaxis.set_minor_locator(NullLocator()); ax.yaxis.set_minor_locator(NullLocator())
        common.style(ax, 'both')
        if i == 0:
            ax.set_ylabel('End-effector separation (mm)', fontsize=7.6, labelpad=3)
        else:
            ax.tick_params(axis='y', labelleft=False)
        rho = D['motion_correlations_display'][model][step]['rho']
        ax.text(.96, .06, rf'$\rho={rho:.3f}$', transform=ax.transAxes, ha='right',
                fontsize=7.6, color=INK, bbox={'facecolor':'white','edgecolor':'none','alpha':.9,'pad':1})
        fig.text(left+.093, .841, f'After {step} controls', ha='center', fontsize=8.0,
                 color=TEAL if step == '5' else ORANGE, weight='bold')
    fig.text(.515, .145, 'Initial first-five translation-command shift (norm, adapter units)',
             ha='center', fontsize=7.7)
    labels = {'SS':'Joint success', 'SF':'Harm', 'FS':'Recovery', 'FF':'Joint failure'}
    handles = [Line2D([], [], marker=outcome_markers[k], color=outcome_colors[k], ls='',
                      ms=4, label=labels[k]) for k in ['SS','SF','FS','FF']]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.51, -.015),
               ncol=4, frameon=False, fontsize=7.3, handletextpad=.4, columnspacing=1.8)
    save(fig, 'fig5_command_motion')


def _number(value):
    return None if value in ['', None] else float(value)


def outcome_row(model, family, metric):
    return common.select(D['outcome_metrics'], model=model, family=family, scope='common28', metric=metric)


def _bound_or_point(ax, row, y, color, marker='o', size=3.8, ci=False, scale=100, hollow=False):
    point = _number(row['point'])
    if point is not None:
        if ci:
            lo, hi = float(row['ci98_75_lower']), float(row['ci98_75_upper'])
            common.interval(ax, point*scale, lo*scale, hi*scale, y, color,
                            marker=marker, hollow=hollow, ms=size)
        else:
            ax.plot(point*scale, y, marker=marker, ls='', ms=size, color=color,
                    mfc='white' if hollow else color, mew=.8, zorder=3)
    else:
        lo, hi = float(row['bound_lower'])*scale, float(row['bound_upper'])*scale
        ax.add_patch(Rectangle((lo, y-.085), hi-lo, .17, facecolor='white',
                               edgecolor=color, lw=.7, hatch='////', zorder=3))


def outcomes():
    setup()
    fig = plt.figure(figsize=(7, 2.86))
    boxes = [[.151, .30, .177, .50], [.364, .30, .181, .50],
             [.584, .30, .184, .50], [.807, .30, .181, .50]]
    axes = [fig.add_axes(p) for p in boxes]
    titles = [('a','Early motion'),('b','Success change'),('c','Outcome flips'),('d','Harm / recovery')]
    subtitles = ['Visual / clean resampling', 'Success difference (pp)', 'Cases changing outcome (%)', 'Harm / recovery (%)']
    for ax, (letter,title), subtitle in zip(axes,titles,subtitles):
        heading(fig, ax.get_position().x0, .912, letter, title, ORANGE, 8.0)
        ax.set_xlabel(subtitle, fontsize=7.1, labelpad=3)
        ax.set_ylim(7.8, -.6); ax.set_yticks([]); common.style(ax,'x')
        for sep in [1.55,3.65,5.75]:ax.axhline(sep,color=GRID,lw=.75)
        ax.axhspan(4.77,5.39,color='#FCF0E9',zorder=0)
    a,b,c,d=axes
    families=['Background Textures','Camera Viewpoints','Sensor Noise','Light Conditions']
    ys=[0,.95,2.1,3.05,4.2,5.15,6.3,7.25]
    for i,fam in enumerate(families):
        fig_y=fig.transFigure.inverted().transform(a.transData.transform((0,(ys[2*i]+ys[2*i+1])/2)))[1]
        fig.text(.003,fig_y,SHORT[fam],ha='left',va='center',fontsize=7.6,weight='bold')
        for j,model in enumerate(['pi05_libero','pi0_libero']):
            y=ys[2*i+j]; marker='o' if j==0 else 's'; hollow=j==1
            fy=fig.transFigure.inverted().transform(a.transData.transform((0,y)))[1]
            fig.text(.115,fy,r'$\pi_{0.5}$' if j==0 else r'$\pi_0$',ha='center',va='center',fontsize=7.4,color=MUTED)
            mr=common.select(D['motion_reference'],model=model,family=fam)
            common.interval(a,float(mr['ratio_of_task_means']),float(mr['ratio_ci95_low']),
                            float(mr['ratio_ci95_high']),y,INK,marker=marker,hollow=hollow,ms=3.7)
            _bound_or_point(b,outcome_row(model,fam,'risk_difference'),y,INK,marker,ci=True,hollow=hollow)
            for metric,offset,col,mk,hollow2 in [('tr_flip',0,INK,marker,hollow),
                                               ('rr_policy_flip',-.18,'#98A7B6','^',False),
                                               ('rr_repeat_flip',.18,'#98A7B6','D',True)]:
                _bound_or_point(c,outcome_row(model,fam,metric),y+offset,col,mk,size=3.3,hollow=hollow2)
            for metric,offset,col in [('harm',-.15,ORANGE),('recovery',.15,TEAL)]:
                row=outcome_row(model,fam,metric);point=_number(row['point'])
                if point is not None:
                    d.barh(y+offset,point*100,height=.22,color=col,zorder=3)
                else:
                    lo,hi=float(row['bound_lower'])*100,float(row['bound_upper'])*100
                    d.add_patch(Rectangle((lo,y+offset-.11),hi-lo,.22,facecolor='white',edgecolor=col,lw=.7,hatch='////',zorder=3))
    a.set_xlim(0,4.2);a.set_xticks([0,1,2,3,4]);a.axvline(1,color=MUTED,ls=(0,(3,2)),lw=.7)
    b.set_xlim(-43,16);b.set_xticks([-40,-20,0]);b.axvline(0,color=MUTED,lw=.7)
    c.set_xlim(-1,36);c.set_xticks([0,10,20,30])
    d.set_xlim(0,28);d.set_xticks([0,10,20])
    a.text(1.22,5.15,'0.580',color=ORANGE,fontsize=7.3,weight='bold',va='center')
    b.text(-42,5.15,'−14.3',color=ORANGE,fontsize=6.7,weight='bold',va='center')
    handles=[Line2D([],[],marker='o',ls='',color=INK,ms=3.4,label=r'$\pi_{0.5}$ visual'),
             Line2D([],[],marker='s',ls='',color=INK,mfc='white',ms=3.4,label=r'$\pi_0$ visual'),
             Line2D([],[],marker='^',ls='',color='#98A7B6',ms=3.5,label='Clean resampling'),
             Line2D([],[],marker='D',ls='',color='#98A7B6',mfc='white',ms=3.4,label='Clean repeat'),
             Patch(facecolor=ORANGE,label='Harm'),Patch(facecolor=TEAL,label='Recovery'),
             Patch(facecolor='white',edgecolor=MUTED,hatch='////',label='Missing-outcome bounds')]
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.015),ncol=4,
               frameon=False,fontsize=7.0,handletextpad=.45,columnspacing=1.25,
               handlelength=1.0,labelspacing=.65)
    save(fig,'fig6_outcomes')


# Main result figures use the reader-led renderer; old support renderers remain available.
from reader_result_figures import basis, attention, motion, outcomes, spatial

if __name__ == '__main__':
    functions={'basis':basis,'attention':attention,'motion':motion,'outcomes':outcomes,'spatial':spatial}
    if sys.argv[1:] == ['wrist']:
        wrist(); raise SystemExit(0)
    for name in (sys.argv[1:] or functions):
        if name not in functions:raise SystemExit('Unknown figure: '+name)
        functions[name]()
        print('Rendered',name)
