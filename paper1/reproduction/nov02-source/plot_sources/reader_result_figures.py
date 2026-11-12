"""Publication-size result figures from unchanged bundled scalar estimates.

No fitting, bootstrap, model calls or cohort selection. Descriptive point means
remain point means. Spatial RGB tiles are losslessly extracted from the original
PDF; their numeric statistic and common color scale are unchanged.
"""
from pathlib import Path
import hashlib
import json
import shutil
import subprocess

import numpy as np
from PIL import Image
import build_results as original
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle, Patch, Polygon
from matplotlib.colors import AsinhNorm
from matplotlib.cm import ScalarMappable
from matplotlib.ticker import NullLocator

plt = original.plt
ROOT = Path(__file__).resolve().parent
OUT = ROOT.parent / 'figures'
QA = ROOT.parent.parent / 'qa/result_figures'
QA.mkdir(parents=True, exist_ok=True)
D = json.loads((ROOT / 'data/r11_result_sources.json').read_text())
B = json.loads((ROOT / 'data/figure_data.json').read_text())['basis']
FAMILIES = original.FAMILIES
SHORT = original.SHORT
INK = '#243247'
MUTED = '#667483'
GRID = '#E6EBEF'
LINE = '#C9D1D9'
BASIS = {'new_response': '#0F766E', 'new_clean': '#64748B',
         'new_random': '#94A3B8', 'R_resp': '#0F766E',
         'R_clean': '#64748B', 'R_rand': '#94A3B8', 'R_full': '#62798F'}
FAMILY_COLORS = dict(zip(FAMILIES, ['#188B80', '#2968B2', '#B57B16', '#92519B']))
HARM, RECOVERY = '#CE6338', '#188B80'
WIDTH = 6.875


def setup():
    original.configure()
    plt.rcParams.update({'font.size': 8, 'axes.labelsize': 8,
                         'xtick.labelsize': 7.8, 'ytick.labelsize': 7.8,
                         'legend.fontsize': 7.8,
                         'svg.hashsalt': 'paper1-r11-visual-hierarchy-20261102'})


def style(ax, grid='x', left=False):
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines['left'].set_visible(left)
    ax.spines['bottom'].set_color(LINE)
    ax.set_axisbelow(True)
    if grid:
        ax.grid(axis=grid, color=GRID, linewidth=.55)
    ax.tick_params(pad=2)


def interval(ax, value, low, high, y, color, marker='o', hollow=False, ms=3.8):
    ax.plot([low, high], [y, y], color=color, lw=1.05, zorder=3)
    ax.plot([low, low], [y-.065, y+.065], color=color, lw=.7, zorder=3)
    ax.plot([high, high], [y-.065, y+.065], color=color, lw=.7, zorder=3)
    ax.plot(value, y, marker=marker, ls='', ms=ms, color=color,
            mfc='white' if hollow else color, mew=.85, zorder=4)


def title(fig, x, y, letter, words, size=8.5):
    fig.text(x, y, f'({letter}) {words}', ha='left', va='top',
             weight='bold', fontsize=size, color=INK)


def save(fig, name, audit):
    fig.canvas.draw()
    # Report labels extending outside the canvas, not just saved PDF dimensions.
    renderer = fig.canvas.get_renderer()
    bounds = fig.bbox
    outside = []
    for artist in fig.findobj(match=lambda x: hasattr(x, 'get_text')):
        if not artist.get_visible() or not artist.get_text():
            continue
        box = artist.get_window_extent(renderer)
        if box.width and box.height and (box.x0 < -1 or box.y0 < -1 or
                                         box.x1 > bounds.x1+1 or box.y1 > bounds.y1+1):
            outside.append(artist.get_text())
    audit.update({'figure': name, 'size_inches': list(fig.get_size_inches()),
                  'labels_outside_canvas': outside,
                  'new_fit_or_bootstrap': False})
    (QA / (name+'.json')).write_text(json.dumps(audit, ensure_ascii=False, indent=2))
    fig.savefig(OUT/(name+'.pdf'), metadata={'Title': name, 'Author': 'Anonymous Authors',
                                            'CreationDate': None, 'ModDate': None})
    fig.savefig(OUT/(name+'.svg'), metadata={'Date': None, 'Creator': 'Anonymous Authors'})
    fig.savefig(OUT/(name+'.png'), dpi=250)
    (ROOT/'previews').mkdir(exist_ok=True)
    shutil.copyfile(OUT/(name+'.png'), ROOT/'previews'/(name+'.png'))
    plt.close(fig)


def basis():
    setup()
    W,H=WIDTH,3.90
    fig=plt.figure(figsize=(W,H))
    axes=[]
    for x,w in [(1.04,1.57),(3.08,1.50),(5.12,1.51)]:
        ax=fig.add_axes([x/W,1.77/H,w/W,1.70/H]);axes.append(ax)
        ax.set_ylim(13.15,-.65);ax.set_yticks([]);style(ax)
        for y in [3.88,8.62]:ax.axhline(y,color='#DDE3E7',lw=.55)
    a,b,c=axes
    for ax,label in zip(axes,['(a) Reconstruction','(b) Prediction gain','(c) Prediction error']):
        fig.text(ax.get_position().x0,3.80/H,label,fontsize=8.8,weight='medium',va='top',color=INK)
    handles=[Line2D([],[],marker='o',ls='',color=BASIS['new_response'],ms=3.8,label='Response'),
             Line2D([],[],marker='o',ls='',color=BASIS['new_clean'],mfc='white',ms=3.8,label='Clean'),
             Line2D([],[],marker='x',ls='',color=BASIS['new_random'],ms=3.8,label='Random')]
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.54,3.63/H),ncol=3,frameon=False,
               fontsize=7.6,handletextpad=.35,handlelength=.9,columnspacing=1.8,borderaxespad=0)
    plotted={'retention':[],'readout_gain':[],'readout_error':[]}
    for layer,base,label in [(0,0,'L1'),(8,4.75,'L9'),(17,9.5,'L18')]:
        yfig=fig.transFigure.inverted().transform(a.transData.transform((0,base+1.5)))[1]
        fig.text(.025,yfig,label,fontsize=8.3,weight='medium',va='center',color=INK)
        for k,fam in enumerate(FAMILIES):
            y=base+k;yf=fig.transFigure.inverted().transform(a.transData.transform((0,y)))[1]
            fig.text(.975/W,yf,SHORT[fam],fontsize=7.5,ha='right',va='center',color=INK)
            for model,off,mark,hollow in [('new_response',-.19,'o',False),('new_clean',0,'o',True),('new_random',.19,'x',False)]:
                r=original.select(B['geometry'],family=fam,layer=layer,model=model)['retention']
                interval(a,100*r['mean'],100*r['interval_95'][0],100*r['interval_95'][1],y+off,BASIS[model],mark,hollow,3.0)
                plotted['retention'].append({'family':fam,'layer':layer,'basis':model,**r})
            r=original.select(B['contrasts'],family=fam,layer=layer)
            interval(b,100*r['relative_error_reduction'],100*r['relative_error_reduction_interval_95'][0],100*r['relative_error_reduction_interval_95'][1],y,BASIS['new_response'],ms=3.3)
            plotted['readout_gain'].append(r)
            for model,off,hollow in [('new_response',-.12,False),('new_clean',.12,True)]:
                r=original.select(B['readouts'],family=fam,layer=layer,model=model)
                v=r['task_equal_mse']/r['task_equal_target_energy']
                c.plot(v,y+off,marker='o',ls='',ms=3.3,color=BASIS[model],mfc='white' if hollow else BASIS[model],mew=.8)
                plotted['readout_error'].append({'family':fam,'layer':layer,'basis':model,'ratio':v})
    a.set_xlim(-2,100);a.set_xticks([0,50,100])
    b.set_xlim(-80,105);b.set_xticks([-50,0,50,100]);b.axvline(0,color=MUTED,lw=.7,ls=(0,(3,2)))
    c.set_xscale('log');c.set_xlim(.15,9);c.set_xticks([.25,1,4],['.25','1','4']);c.xaxis.set_minor_locator(NullLocator())
    c.axvspan(.15,1,color='#F2F6F6',zorder=0);c.axvline(1,color=MUTED,lw=.7,ls=(0,(3,2)))
    for ax,lab in zip(axes,['Residual energy retained (%)','Error reduction vs. clean (%)','MSE / zero-change MSE']):
        ax.set_xlabel(lab,fontsize=7.5,labelpad=4);ax.tick_params(labelsize=7.4)
    # A single quiet rule separates the independent intervention from paired recordings.
    fig.add_artist(Line2D([.025,.975],[1.32/H,1.32/H],transform=fig.transFigure,color='#D2DAE0',lw=.6))
    fig.text(.025,1.19/H,'(d) Independent native restoration',fontsize=8.8,weight='medium',va='center',color=INK)
    fig.legend(handles=[Line2D([],[],color=INK,lw=.8,marker='>',ms=3,label='Toward clean'),
                        Line2D([],[],color=INK,ls='',marker='D',mfc='white',ms=3.3,label='Orthogonal norm')],
               loc='center right',bbox_to_anchor=(.975,1.19/H),ncol=2,frameon=False,fontsize=7.5,
               handletextpad=.4,handlelength=1.1,columnspacing=1.3,borderaxespad=0)
    direction=json.loads((ROOT/'command_direction_audit/SUMMARY.json').read_text())
    lookup={(r['family'],r['arm']):r for r in direction['CONF']['translation_family_arm_means']}
    restoration=[]
    for fam,x,w,xmax,ticks,lab in [
            ('Camera Viewpoints',1.04,2.16,.132,[0,.06,.12],'Camera · 27 tasks'),
            ('Sensor Noise',4.49,2.14,.021,[0,.01,.02],'Sensor · 30 tasks')]:
        ax=fig.add_axes([x/W,.34/H,w/W,.54/H]);ax.set_ylim(3.55,-.55);ax.set_xlim(0,xmax)
        ax.set_yticks(range(4),['Response','Clean','Random',r'Full $H$']);ax.set_xticks(ticks)
        ax.set_xticklabels(['0']+[f'{v:g}' for v in ticks[1:]])
        style(ax);ax.tick_params(labelsize=7.3,pad=2)
        ax.axvline(0,color=LINE,lw=.65)
        fig.text(x/W,1.00/H,lab,fontsize=7.8,va='center',color=INK)
        full=lookup[(fam,'R_full')]['metrics']['signed_along_clean']['value']
        ax.axvline(full,color=BASIS['R_full'],lw=.65,ls=(0,(2,2)),alpha=.65)
        for y,arm in enumerate(['R_resp','R_clean','R_rand','R_full']):
            v=lookup[(fam,arm)]['metrics'];along=v['signed_along_clean']['value'];orth=v['orthogonal_norm']['value'];yy=y-.10
            ax.plot([0,along],[yy,yy],color=BASIS[arm],lw=.85,solid_capstyle='butt',zorder=2)
            start=ax.transData.transform((0,yy));end=ax.transData.transform((along,yy));d=abs(end[0]-start[0])*72/fig.dpi
            if d>=4:
                hp=3.2*fig.dpi/72;hw=1.45*fig.dpi/72
                vtx=np.array([end,end+[-hp,hw],end+[-hp,-hw]])
                ax.add_patch(Polygon(ax.transData.inverted().transform(vtx),closed=True,facecolor=BASIS[arm],edgecolor='none',linewidth=0,zorder=4,clip_on=False))
            else:ax.plot(along,yy,marker='o',ms=2.3,ls='',color=BASIS[arm],mew=0,zorder=4,clip_on=False)
            ax.plot(orth,y+.15,marker='D',ls='',ms=3,color=BASIS[arm],mfc='white',mew=.8,zorder=4)
            restoration.append({'family':fam,'arm':arm,'along':along,'orthogonal':orth,'full_reference':full})
    fig.text(.54,.052/H,'Translation-command units · independent horizontal scales',ha='center',fontsize=7.3,color=MUTED)
    fig.canvas.draw()
    assert [len(plotted[k]) for k in plotted]==[36,12,24]
    assert len(restoration)==8
    save(fig,'same_basis_translation',{'source':'figure_data.json and command_direction_audit/SUMMARY.json',
                                      'plotted':plotted,'restoration':restoration,
                                      'layout':{'arrangement':'matched reconstruction/readout row above independent native-restoration row',
                                                'native_families':'side-by-side with independent horizontal scales',
                                                'readout_error_axis':'logarithmic'},
                                      'direction_intervals_available':False,
                                      'ratios_not_claimed_as_functional_capacity':True})


def attention():
    setup();W,H=WIDTH,3.02
    profiles=json.loads((ROOT/'data/r11_attention_profiles.json').read_text());values=profiles['profiles']
    fig=plt.figure(figsize=(W,H))
    left=.43;middle=2.65;fw=1.45
    boxes=[(left,1.80,fw,.78),(left,.55,fw,.78),(middle,1.80,fw,.78),(middle,.55,fw,.78)]
    axes=[fig.add_axes([x/W,y/H,w/W,h/H]) for x,y,w,h in boxes]
    for x,y,t in [(left,2.92,r'(a) Probability $P$'),(left,1.59,r'(b) Output $W$'),(middle,2.92,r'(c) Native state $H_{\mathrm{AE}}$'),(4.88,2.92,'(d) Command association')]:
        fig.text(x/W,y/H,t,fontsize=8.3,weight='medium',va='top',color=INK)
    metrics=[('p_full_key_tv',100),('w_delta_rms',1),('ae_delta_rms',1),('ae_relative_l2',1)]
    markers=['o','s','^','D']
    for ax,(metric,scale) in zip(axes,metrics):
        for fam,mk in zip(FAMILIES,markers):
            rows=sorted((r for r in values if r['family']==fam),key=lambda r:r['generation_call'])
            assert [r['generation_call'] for r in rows]==[0,5,9]
            ax.plot([0,5,9],[r[metric]*scale for r in rows],color=FAMILY_COLORS[fam],marker=mk,ms=3.0,lw=.95,mew=.5)
        ax.set_xlim(-.6,9.6);ax.set_xticks([0,5,9]);style(ax,'y');ax.tick_params(labelsize=7.3)
    for ax,ylims,ticks,ylabel in zip(axes,[(0,22),(0,2.8),(0,6.2),(0,.25)],[[0,10,20],[0,1,2],[0,3,6],[0,.1,.2]],['TV (%)','Difference RMS','Difference RMS','Relative difference']):
        ax.set_ylim(*ylims);ax.set_yticks(ticks);ax.set_ylabel(ylabel,fontsize=7.3,labelpad=3)
    axes[0].tick_params(labelbottom=False);axes[2].tick_params(labelbottom=False)
    fig.text(middle/W,1.55/H,'Clean RMS: 46.8 → 16.6',fontsize=7.2,color=MUTED)
    fig.text((left+fw/2)/W,.28/H,'Generation step',ha='center',fontsize=7.5,color=INK)
    fig.text((middle+fw/2)/W,.28/H,'Generation step',ha='center',fontsize=7.5,color=INK)
    fig.text((left+fw)/W,2.70/H,'Replay',ha='right',fontsize=7.0,color=MUTED)
    fig.text((left+fw)/W,1.41/H,'Replay',ha='right',fontsize=7.0,color=MUTED)
    # A horizontal forest plot keeps the supporting association readable without
    # giving it the same visual structure as the generation trajectories.
    ax=fig.add_axes([5.10/W,.58/H,1.52/W,2.00/H])
    signals=['PG_H_main_relative_l2','P_full_key_TV','W_pre_gate_relative_l2','AE_native_relative_l2']
    associations=[]
    for fam,off,mk in zip(FAMILIES,[-.21,-.07,.07,.21],markers):
        rows=[original.select(D['attention_association'],family=fam,signal=s) for s in signals]
        v=np.array([float(r['rho_issued']) for r in rows]);lo=np.array([float(r['rho_issued_low']) for r in rows]);hi=np.array([float(r['rho_issued_high']) for r in rows])
        ax.errorbar(v,np.arange(4)+off,xerr=[v-lo,hi-v],fmt=mk,color=FAMILY_COLORS[fam],ms=3.0,elinewidth=.65,capsize=1.2,mew=.6)
        associations.extend(rows)
    ax.set_yticks(range(4),[r'$H_{\mathrm{VLM}}$',r'$P$',r'$W$',r'$H_{\mathrm{AE}}$']);ax.set_ylim(3.5,-.5)
    ax.set_xlim(-.55,.9);ax.set_xticks([-.5,0,.5]);ax.axvline(0,color=MUTED,ls=(0,(3,2)),lw=.65);style(ax,'x')
    ax.tick_params(labelsize=7.4);ax.set_xlabel(r'Rank correlation $\rho$',fontsize=7.4,labelpad=3)
    fig.text(5.86/W,2.70/H,'Step 0 · 95% intervals',ha='center',fontsize=7.0,color=MUTED)
    handles=[Line2D([],[],color=FAMILY_COLORS[f],marker=mk,ms=3.2,lw=.95,label=SHORT[f]) for f,mk in zip(FAMILIES,markers)]
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,-.012),ncol=4,frameon=False,fontsize=7.5,columnspacing=1.6,handlelength=1.2,handletextpad=.4)
    assert len(associations)==16 and len(values)==12
    save(fig,'fig4_attention',{'source':'r11_attention_profiles.json and r11_result_sources.json',
                              'profiles':values,'association_rows':associations,
                              'generation_intervals_available':False,
                              'relative_metric':'mean per-case relative L2, copied from frozen profile; not a ratio of plotted means',
                              'layout':'paired P/W and native H_AE raw/relative columns; supporting horizontal command intervals',
                              'state_notation':'H_VLM prefix input; H_AE native post-block'})


def motion():
    setup()
    fig=plt.figure(figsize=(WIDTH,2.30))
    colors={'SS':'#8A969F','SF':HARM,'FS':RECOVERY,'FF':'#75657F'}
    markers={'SS':'o','SF':'^','FS':'s','FF':'x'}
    pairs=[('pi05_libero','5'),('pi05_libero','60'),('pi0_libero','5'),('pi0_libero','60')]
    positions=[.066,.305,.555,.794]
    counts=[]
    for i,(x,(model,step)) in enumerate(zip(positions,pairs)):
        ax=fig.add_axes([x,.292,.187,.545]);rr=[r for r in D['motion_points'] if r['model']==model and r['step']==step and r['included']=='True']
        assert len(rr)==(516 if model=='pi05_libero' else 526)
        assert {r['Y'] for r in rr}==set(markers)
        for cat in ['SS','FF','FS','SF']:
            data=[r for r in rr if r['Y']==cat]
            options={'s':8,'alpha':.42,'linewidths':0,'zorder':2} if cat=='SS' else (
                {'s':8,'alpha':.48,'linewidths':.5,'zorder':3} if cat=='FF' else
                {'s':12,'alpha':.78,'linewidths':.28,'edgecolors':'white','zorder':5})
            ax.scatter([float(r['q0_norm']) for r in data],[float(r['eef_mm']) for r in data],
                       c=colors[cat],marker=markers[cat],rasterized=False,**options)
        ax.set_xscale('symlog',linthresh=.01,linscale=.75);ax.set_yscale('symlog',linthresh=1,linscale=.75)
        ax.set_xlim(0,1.6);ax.set_ylim(0,440);ax.set_xticks([0,.01,.1,1],['0','.01','.1','1'])
        ax.set_yticks([0,1,10,100],['0','1','10','100']);ax.xaxis.set_minor_locator(NullLocator());ax.yaxis.set_minor_locator(NullLocator());style(ax,'both')
        if i==0:ax.set_ylabel('End-effector separation (mm)',fontsize=7.9,labelpad=3)
        else:ax.tick_params(axis='y',labelleft=False)
        rho=D['motion_correlations_display'][model][step]['rho']
        ax.text(.96,.04,rf'$\rho={rho:.3f}$',transform=ax.transAxes,ha='right',fontsize=7.8,
                bbox={'facecolor':'white','edgecolor':'none','alpha':.9,'pad':1})
        policy=r'$\pi_{0.5}$' if model=='pi05_libero' else r'$\pi_0$'
        fig.text(x+.093,.935,f'({chr(97+i)}) '+policy+f' · {step} controls',ha='center',va='top',fontsize=8.0,color=INK,weight='medium')
        assert all(0<=float(r['q0_norm'])<=1.6 and 0<=float(r['eef_mm'])<=440 for r in rr)
        counts.append({'model':model,'step':step,'points':len(rr),'rho':rho,'counts':{k:sum(r['Y']==k for r in rr) for k in markers}})
    fig.text(.515,.155,'Initial five-command translation shift (norm, adapter units)',ha='center',fontsize=7.5)
    handles=[Line2D([],[],marker=markers[k],color=colors[k],ls='',ms=4.5,label=v) for k,v in
             [('SS','Joint success'),('SF','Harm'),('FS','Recovery'),('FF','Joint failure')]]
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.51,-.010),ncol=4,frameon=False,
               fontsize=7.5,columnspacing=1.5,handletextpad=.4)
    assert sum(x['points'] for x in counts)==2084
    save(fig,'fig5_command_motion',{'source':'r11_result_sources.json','panel_counts':counts,
                                   'all_qualified_points_retained':True,'axes_ranges_preserved':True,
                                   'new_fitted_trend':False,'presentation_changes_only':True})



def number(x):return None if x in ['',None] else float(x)
def outcome_row(model,family,metric):return original.select(D['outcome_metrics'],model=model,family=family,scope='common28',metric=metric)


def bound_or_point(ax,row,y,color,marker='o',hollow=False,ci=False,size=3.8):
    p=number(row['point'])
    if p is None:
        lo,hi=float(row['bound_lower'])*100,float(row['bound_upper'])*100
        ax.add_patch(Rectangle((lo,y-.075),hi-lo,.15,facecolor='white',edgecolor=color,lw=.75,hatch='////',zorder=3))
    elif ci:
        interval(ax,p*100,float(row['ci98_75_lower'])*100,float(row['ci98_75_upper'])*100,y,color,marker,hollow,size)
    else:
        ax.plot(p*100,y,marker=marker,ls='',ms=size,color=color,mfc='white' if hollow else color,mew=.85,zorder=4)


def outcomes():
    setup()
    fig=plt.figure(figsize=(WIDTH,2.75))
    axes=[fig.add_axes(p) for p in [[.151,.282,.170,.578],[.366,.282,.177,.578],
                                  [.589,.282,.173,.578],[.809,.282,.180,.578]]]
    for ax,letter,words,label in zip(axes,'abcd',['Early motion','Success change','Outcome flips','Harm / recovery'],
                                    ['Visual / resampling','Success difference (pp)','Changed outcomes (%)','Harm ← (%) → Recovery']):
        fig.text(ax.get_position().x0,.970,f'({letter}) {words}',fontsize=8.1,weight='medium',va='top',color=INK)
        ax.set_xlabel(label,fontsize=7.5,labelpad=4);ax.set_ylim(7.8,-.6);ax.set_yticks([]);style(ax)
        for sep in [1.55,3.65,5.75]:ax.axhline(sep,color=GRID,lw=.75)
        ax.axhspan(4.77,5.39,color='#FCF7F3',zorder=0)
    a,b,c,d=axes
    a.axvspan(0,1,color='#F5F6F6',zorder=0)
    families=[FAMILIES[0],FAMILIES[1],FAMILIES[3],FAMILIES[2]]
    ys=[0,.95,2.1,3.05,4.2,5.15,6.3,7.25]
    used=[];motion_rows=[]
    for i,fam in enumerate(families):
        fy=fig.transFigure.inverted().transform(a.transData.transform((0,(ys[2*i]+ys[2*i+1])/2)))[1]
        fig.text(.003,fy,SHORT[fam],ha='left',va='center',fontsize=7.6,weight='medium')
        for j,model in enumerate(['pi05_libero','pi0_libero']):
            y=ys[2*i+j];mk='o' if j==0 else 's';hollow=j==1
            fy=fig.transFigure.inverted().transform(a.transData.transform((0,y)))[1]
            fig.text(.123,fy,r'$\pi_{0.5}$' if j==0 else r'$\pi_0$',ha='center',va='center',fontsize=7.8,color=MUTED)
            mr=original.select(D['motion_reference'],model=model,family=fam)
            interval(a,float(mr['ratio_of_task_means']),float(mr['ratio_ci95_low']),float(mr['ratio_ci95_high']),y,INK,mk,hollow,3.8);motion_rows.append(mr)
            risk=outcome_row(model,fam,'risk_difference');bound_or_point(b,risk,y,INK,mk,hollow,True);used.append(risk)
            visual=outcome_row(model,fam,'tr_flip');resamp=outcome_row(model,fam,'rr_policy_flip');repeat=outcome_row(model,fam,'rr_repeat_flip')
            if number(visual['point']) is not None and number(resamp['point']) is not None:
                c.plot([float(resamp['point'])*100,float(visual['point'])*100],[y,y],color='#B2BCC4',lw=.85,zorder=2)
                voff=roff=0
            else:voff,roff=-.12,.12
            bound_or_point(c,resamp,y+roff,'#9AA8B4','^',False,False,3.9)
            bound_or_point(c,visual,y+voff,INK,mk,hollow,False,3.7)
            bound_or_point(c,repeat,y+.27,'#9AA8B4','D',True,False,3.1)
            used.extend([visual,resamp,repeat])
            for metric,color,sign in [('harm',HARM,-1),('recovery',RECOVERY,1)]:
                row=outcome_row(model,fam,metric);p=number(row['point']);used.append(row)
                if p is not None:d.barh(y,sign*p*100,height=.38,color=color,zorder=3)
                else:
                    lo,hi=float(row['bound_lower'])*100,float(row['bound_upper'])*100
                    start=lo if sign==1 else -hi
                    d.add_patch(Rectangle((start,y-.19),hi-lo,.38,facecolor='white',edgecolor=color,lw=.7,hatch='////',zorder=3))
    a.set_xlim(0,4.2);a.set_xticks([0,1,2,3,4]);a.axvline(1,color=MUTED,ls=(0,(3,2)),lw=.65)
    b.set_xlim(-43,16);b.set_xticks([-40,-20,0]);b.axvline(0,color=MUTED,lw=.65)
    c.set_xlim(-1,36);c.set_xticks([0,10,20,30])
    d.set_xlim(-28,18);d.set_xticks([-20,-10,0,10],['20','10','0','10']);d.axvline(0,color=MUTED,lw=.65)
    a.annotate('0.580×',xy=(.580,5.15),xytext=(1.42,5.15),ha='left',va='center',color=HARM,fontsize=7.3,weight='normal',
               arrowprops={'arrowstyle':'-','color':HARM,'lw':.7},bbox={'facecolor':'#FCF7F3','edgecolor':'none','pad':.3})
    b.text(2,5.15,'−14.3',ha='left',va='center',color=HARM,fontsize=7.3,weight='normal')
    d.annotate('3.57% each',xy=(-1,0),xytext=(-26,-.36),ha='left',va='center',fontsize=7.0,color=MUTED,
               arrowprops={'arrowstyle':'-','color':MUTED,'lw':.5})
    handles=[Line2D([],[],marker='o',ls='',color=INK,ms=4,label=r'$\pi_{0.5}$ visual'),
             Line2D([],[],marker='s',ls='',color=INK,mfc='white',ms=4,label=r'$\pi_0$ visual'),
             Line2D([],[],marker='^',ls='',color='#9AA8B4',ms=4,label='Clean resampling'),
             Line2D([],[],marker='D',ls='',color='#9AA8B4',mfc='white',ms=3.5,label='Clean repeat'),
             Patch(facecolor=HARM,label='Harm'),Patch(facecolor=RECOVERY,label='Recovery'),
             Patch(facecolor='white',edgecolor=MUTED,hatch='////',label='Missing-outcome bounds')]
    # Place each unchanged key beside the panels that use it.
    key_style=dict(frameon=False,fontsize=7.3,handlelength=1.05,
                   handletextpad=.45,labelspacing=.40,borderaxespad=0,borderpad=0)
    fig.legend(handles=handles[:2],loc='upper left',bbox_to_anchor=(.151,.145),
               ncol=2,columnspacing=1.5,**key_style)
    fig.legend(handles=handles[2:4],loc='upper left',bbox_to_anchor=(.589,.145),
               ncol=1,**key_style)
    fig.legend(handles=handles[4:6],loc='upper left',bbox_to_anchor=(.809,.145),
               ncol=1,**key_style)
    fig.legend(handles=handles[6:],loc='upper left',bbox_to_anchor=(.151,.070),
               ncol=1,**key_style)
    assert len(motion_rows)==8 and len(used)==48
    save(fig,'fig6_outcomes',{'source':'r11_result_sources.json','motion_rows':motion_rows,'outcome_rows':used,
                            'sampling_intervals':'motion: existing95%; success: existing98.75% or missing-data bounds',
                            'flip_connections':'descriptive levels only; no difference inference',
                            'harm_display':'negative x is a display direction; tick labels show positive event percentages',
                            'reference_region_not_safety_zone':True})



def spatial():
    setup();W,H=WIDTH,2.12
    p=ROOT/'spatial_source'
    images=[np.asarray(Image.open(p/f'image-{i:03}.png').convert('RGB')) for i in range(8)]
    assert np.array_equal(images[1],images[3])
    fig=plt.figure(figsize=(W,H))
    def imax(x,y,w,h):return fig.add_axes([x/W,y/H,w/W,h/H])
    for x,title in [(.035,'(a) Inputs'),(1.64,'(b) Wrist attention change'),(5.64,'(c) L18 on wrist')]:
        fig.text(x/W,2.03/H,title,fontsize=8.4,weight='medium',va='top',color=INK)
    for index,x in [(0,.16),(2,.82)]:
        ax=imax(x,1.03,.56,.56);ax.imshow(images[index],interpolation='none');ax.axis('off')
    for x,label in [(.44,'Reference'),(1.10,'Main +15°')]:
        fig.text(x/W,1.71/H,label,ha='center',fontsize=7.5,color=INK)
    ax=imax(.47,.30,.60,.60);ax.imshow(images[1],interpolation='none');ax.axis('off')
    fig.text(.77/W,.15/H,'Shared wrist input',ha='center',fontsize=7.5,color=MUTED)
    for i,(x,label) in enumerate([(1.65,'L1'),(2.80,'L9'),(3.95,'L18')]):
        ax=imax(x,.50,1.00,1.00);ax.imshow(images[4+i],interpolation='nearest',origin='upper',extent=(-.5,15.5,15.5,-.5))
        ax.set_xticks([0,15]);ax.set_yticks([0,15] if i==0 else []);ax.tick_params(labelsize=7.0,pad=1,length=2)
        for spine in ax.spines.values():spine.set_visible(False)
        fig.text((x+.5)/W,1.68/H,label,ha='center',fontsize=8.0,color=INK)
    cbax=imax(5.05,.50,.055,1.00)
    norm=AsinhNorm(linear_width=.0009429195779375732,vmin=0,vmax=.060346852988004684)
    cb=fig.colorbar(ScalarMappable(norm=norm,cmap='magma'),cax=cbax,ticks=[0,.001,.01,.06])
    cb.ax.set_yticklabels(['0','.001','.01','.06']);cb.ax.tick_params(labelsize=7.0,pad=2,length=2);cb.outline.set_linewidth(.4)
    fig.text(3.30/W,.15/H,r'$\langle |\Delta\widetilde{P}_{\mathrm{wrist}}|\rangle$ · 16 × 16 · shared scale',ha='center',fontsize=7.4,color=MUTED)
    ax=imax(5.65,.445,1.11,1.11)
    ax.imshow(images[1],interpolation='none',origin='upper',extent=(-.5,15.5,15.5,-.5))
    ax.imshow(images[6],interpolation='nearest',origin='upper',extent=(-.5,15.5,15.5,-.5),alpha=.28)
    ax.axis('off');fig.text(6.205/W,.25/H,'Spatial guide',ha='center',fontsize=7.3,color=MUTED)
    save(fig,'attention_spatial',{'source_pdf_sha256':hashlib.sha256((p/'original_attention_spatial.pdf').read_bytes()).hexdigest(),
                                  'source_tiles':[{'file':f'image-{i:03}.png','sha256':hashlib.sha256((p/f'image-{i:03}.png').read_bytes()).hexdigest(),'shape':list(x.shape)} for i,x in enumerate(images)],
                                  'wrist_pixels_equal':True,'grid_statistic':'original mean absolute conditional redistribution; no reconstructed numeric array',
                                  'orientation':'same square field; origin top left; no crop, rotation or flip',
                                  'color_norm':{'type':'asinh','linear_width':.0009429195779375732,'vmin':0,'vmax':.060346852988004684},
                                  'overlay':'original L18 color tile, alpha0.28, location guide only; colorbar applies only to unblended grids',
                                  'L1_preserved':'actual near-zero grid retained; no exact-zero assertion',
                                  'input_layout':'two main-camera inputs and one explicitly shared unchanged wrist input'})
