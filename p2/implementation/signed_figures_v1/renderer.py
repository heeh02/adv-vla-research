"""Publication figures from verified scalar caches. English labels; CPU only."""
from pathlib import Path
import csv
import gzip
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import TwoSlopeNorm
from matplotlib.text import Text
import audit as A

FIGURE_IDS=('F1-CAPACITY-TRANSFER','F2-LOCAL-STRUCTURE','F3-ROOT-COUNTEREXAMPLES','F4-PAIRED-INTERACTION')
LABELS={'c9':'Context only','norm4':'Original norm / address (4)',
 'pooled4':'Pooled write PCA (4)','current4':'Current write PCA (4)',
 'pooled_full':'Pooled signed write','current_full':'Current signed write',
 'headpos_signed':'Head-position signed','headpos_norms':'Head-position norms (80)',
 'headpos_signed_sum':'Signed head-position sum'}
SHORT_DIMS=('Trans. x','Trans. y','Trans. z','Rot. x','Rot. y','Rot. z','Gripper')
STYLE={'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':11,'axes.labelsize':9,
 'xtick.labelsize':8,'ytick.labelsize':8,'pdf.fonttype':42,'ps.fonttype':42,
 'svg.fonttype':'none','savefig.facecolor':'white','figure.facecolor':'white'}
CAPTIONS={
 FIGURE_IDS[0]:{'title':'Capacity and transfer error',
  'caption':'KT96 transfer: each cell is task-balanced MSE divided by the zero-effect predictor MSE, separately for seven command dimensions. Left: fixed lambda=1; right: DEV training EDF matched to current-write PCA4. EDF includes the intercept and is not equality of hypothesis classes. The lower table gives the minimum and maximum of seven separate DEV training error ratios; it is a range, not a cross-unit score. C/T denote parent trajectories. Baseline information includes unedited j5 through v5. CONF is already seen; no new blind test.'},
 FIGURE_IDS[1]:{'title':'Local value of signed and positional structure',
  'caption':'KT96, EDF-matched comparisons. Each cell is MSE(method A)/MSE(method B); below it, n/4 counts tasks with strictly lower error for A. All seven dimensions and both parent populations and position windows are shown. Counts are descriptive, not independent significance tests. Comparisons retain failures as well as improvements; relative improvement does not imply beating zero.'},
 FIGURE_IDS[2]:{'title':'Every root: complete-write transfer counterexamples',
  'caption':'All eight CONF roots, both overlapping position windows and all seven dimensions. Cells show EDF-matched current-full-write MSE divided by zero-predictor MSE using the original DEV per-position scale. No roots or dimensions were selected by outcome. These are candidate endpoint commands, not issued actions or task outcomes.'},
 FIGURE_IDS[3]:{'title':'Same-query K-background interaction prediction',
  'caption':'Auxiliary KC-minus-KT contrast at all 24 shared-edit anchors, separately for C/T parent populations and both windows. Cells compare predicted-interaction MSE with zero-interaction MSE. Scaling uses the same original DEV target sigma, not an interaction-estimated scale. Context, original-norm and pooled models predict an exactly zero interaction by construction; their ratio is one when target energy is positive. No new interaction fit or mechanism attribution is made.'}}


def population(data,parent,bg='KT'):
    source=data['interaction_groups']if bg=='KC_minus_KT'else data['groups']
    return next(g for g in source if g['level']=='population'and g['parent']==parent and g['background']==bg)


def ratio(a,b):
    a,b=np.asarray(a,float),np.asarray(b,float)
    return np.divide(a,b,out=np.full(np.broadcast_shapes(a.shape,b.shape),np.nan),where=b>0)


def label_number(x):
    if not np.isfinite(x):return 'NA'
    if x==0:return '0'
    if x<.01:return f'{x:.2g}'
    if x>=100:return f'{x:.1f}'if x<1000 else f'{x:.2g}'
    return f'{x:.2f}'


def heat(ax,values,rows,*,counts=None,fontsize=8,limits=(-1.,4.),show_y=True):
    values=np.asarray(values,float);finite=np.isfinite(values)
    log=np.full_like(values,np.nan);positive=finite&(values>0)
    log[positive]=np.log2(values[positive]);log[finite&(values==0)]=limits[0]
    cmap=plt.get_cmap('RdBu_r').copy();cmap.set_bad('#dddddd')
    im=ax.imshow(np.ma.masked_invalid(log),aspect='auto',cmap=cmap,
        norm=TwoSlopeNorm(vmin=limits[0],vcenter=0.,vmax=limits[1]),interpolation='nearest')
    ax.set_xticks(range(7),SHORT_DIMS);ax.tick_params(axis='x',length=0,pad=5)
    ax.set_yticks(range(len(rows)),rows if show_y else ['']*len(rows));ax.tick_params(axis='y',length=0,pad=6)
    ax.set_xticks(np.arange(-.5,7,1),minor=True);ax.set_yticks(np.arange(-.5,len(rows),1),minor=True)
    ax.grid(which='minor',color='white',linewidth=.65);ax.tick_params(which='minor',length=0)
    for i in range(values.shape[0]):
        for j in range(7):
            text=label_number(values[i,j])
            if counts is not None:text+='\n'+str(int(counts[i,j]))+'/4'
            color='white'if np.isfinite(log[i,j])and(log[i,j]>limits[1]*.70 or log[i,j]<limits[0]*.75)else'#151515'
            t=ax.text(j,i,text,ha='center',va='center',fontsize=fontsize,color=color,linespacing=1.2)
            t._figure_cell=(ax,i,j,values.shape)
    return im


def colorbar(fig,im,axes,*,limits,description='MSE ratio; 1 = equal error'):
    bar=fig.colorbar(im,ax=axes,location='right',fraction=.025,pad=.025,shrink=.88)
    ticks=[k for k in range(int(np.ceil(limits[0])),int(np.floor(limits[1]))+1)]
    if 0 not in ticks:ticks.append(0)
    bar.set_ticks(sorted(ticks),labels=[f'{2**k:g}'for k in sorted(ticks)])
    bar.ax.tick_params(labelsize=8);bar.set_label(description,fontsize=9)


def title_footer(fig,title,synthetic,footer):
    fig.suptitle(('SYNTHETIC — 'if synthetic else '')+title,fontsize=16,y=.975,weight='bold')
    fig.text(.5,.018,footer,ha='center',va='bottom',fontsize=8,linespacing=1.5)


def layout_check(fig,name):
    fig.canvas.draw();renderer=fig.canvas.get_renderer();bounds=fig.bbox;problems=[];min_font=100.;min_numeric=100.;texts=0;cells=0
    for ax in fig.axes:
        box=ax.get_window_extent(renderer)
        if box.x0<bounds.x0-1 or box.x1>bounds.x1+1 or box.y0<bounds.y0-1 or box.y1>bounds.y1+1:
            problems.append({'kind':'axes_outside_figure'})
    for text in fig.findobj(Text):
        if not text.get_visible()or not text.get_text():continue
        texts+=1;min_font=min(min_font,text.get_fontsize());b=text.get_window_extent(renderer)
        if b.x0<bounds.x0-1 or b.x1>bounds.x1+1 or b.y0<bounds.y0-1 or b.y1>bounds.y1+1:
            problems.append({'kind':'outside_figure','text':text.get_text()[:80]})
        cell=getattr(text,'_figure_cell',None)
        if cell is not None:
            min_numeric=min(min_numeric,text.get_fontsize())
            ax,i,j,shape=cell;left,bottom=ax.transData.transform((j-.5,i+.5));right,top=ax.transData.transform((j+.5,i-.5));cells+=1
            if b.width>(right-left)*.97 or b.height>(top-bottom)*.97:
                problems.append({'kind':'cell_text_overflow','text':text.get_text(),'row':i,'column':j})
    main=name in FIGURE_IDS
    A.need(min_font>=(8 if main else 7)and(not cells or min_numeric>=(9 if main else 7))and not problems,'readable_layout_'+name)
    return {'name':name,'width_inches':float(fig.get_figwidth()),'height_inches':float(fig.get_figheight()),
            'text_objects':texts,'numeric_cells':cells,'minimum_font_points':float(min_font),
            'minimum_numeric_font_points':float(min_numeric)if cells else None,
            'text_and_axes_bbox_check':'PASS','problems':problems}


def save(fig,out,name,layouts):
    layouts.append(layout_check(fig,name))
    for extension in ('png','pdf','svg'):
        fig.savefig(Path(out)/(name+'.'+extension),dpi=150)
    plt.close(fig)


def main_figures(data,out,check):
    index={n:i for i,n in enumerate(data['names'])};synthetic=data['synthetic'];values={};layouts=[]
    with plt.rc_context(STYLE):
        # F1: preserve all parent/dimension cells; show capacity beside each row.
        fig,axes=plt.subplots(3,2,figsize=(10.5,8.2),gridspec_kw={'height_ratios':[3.,3.,2.4]})
        fig.subplots_adjust(left=.25,right=.92,top=.90,bottom=.13,hspace=.55,wspace=.15)
        f1={};heat_axes=list(axes[:2].ravel())
        for pi,parent in enumerate(('C','T')):
            g=population(data,parent)
            for mi,mode in enumerate(('lambda1','edf_match')):
                selected=[f+'@'+mode for f in A.FAMILIES]
                cells=np.stack([g['relative_to_zero'][index[n],1,1]for n in selected]);f1[parent+'_'+mode]=cells.tolist()
                rows=[LABELS[f]+f"  [EDF {data['diagnostics'][n]['total_edf']:.2f}]"for f,n in zip(A.FAMILIES,selected)]
                im=heat(axes[pi,mi],cells,rows,fontsize=9,show_y=mi==0)
                axes[pi,mi].set_title(f"Parent {parent} | "+('fixed λ = 1'if mi==0 else f"matched EDF = {data['edf_anchor']:.2f}"))
        colorbar(fig,im,heat_axes,limits=(-1,4),description='Transfer MSE / zero MSE (log2 color)')
        zero=data['DEV_mean_mse'][index['zero'],1,1]
        for mi,mode in enumerate(('lambda1','edf_match')):
            ax=axes[2,mi];ax.axis('off');table=[];ranges={}
            for f in A.FAMILIES:
                r=ratio(data['DEV_mean_mse'][index[f+'@'+mode],1,1],zero)
                finite=r[np.isfinite(r)];text='NA'if not len(finite)else f'{finite.min():.3g} — {finite.max():.3g}'
                table.append([LABELS[f],text]);ranges[f]={'ratios':r.tolist(),'defined_dimensions':len(finite)}
            tab=ax.table(cellText=table,colLabels=['DEV dimension range','MSE / zero'],cellLoc='left',colWidths=[.63,.37],bbox=[0,0,1,1])
            tab.auto_set_font_size(False);tab.set_fontsize(9)
            for cell in tab.get_celld().values():cell.set_edgecolor('#dddddd')
            f1['DEV_'+mode]=ranges
        title_footer(fig,'F1  Capacity and transfer error',synthetic,
            'KT96 • Full 10 candidate positions • Original DEV σ • Four tasks equally weighted within each parent population\n'
            'Blue: lower error; red: higher. Color range 0.5–16; numbers not clipped. NA: undefined. DEV range uses defined dimensions, not a mean.')
        save(fig,out,FIGURE_IDS[0],layouts);values[FIGURE_IDS[0]]=f1;check()
        # F2: five fixed comparisons; no outcome-dependent selection.
        fig,axes=plt.subplots(2,2,figsize=(10.5,7.5))
        fig.subplots_adjust(left=.26,right=.92,top=.88,bottom=.16,hspace=.50,wspace=.20)
        f2={}
        rows=['Complete write / PCA4','Current / pooled complete write','Complete write / context',
              'Head-position signed / norms','Head-position signed / signed sum']
        for pi,parent in enumerate(('C','T')):
            g=population(data,parent)
            for wi,window in enumerate(('First 5','Full 10')):
                cells=[];counts=[]
                for fa,fb in A.F2_PAIRS:
                    a,b=fa+'@edf_match',fb+'@edf_match'
                    cells.append(ratio(g['mse'][index[a],wi,1],g['mse'][index[b],wi,1]))
                    p=next(x for x in data['pairs']if x['parent']==parent and x['background']=='KT'and x['stronger']==a and x['weaker']==b)
                    counts.append(p['positive_task_count'][wi,1])
                im=heat(axes[pi,wi],cells,rows,counts=np.array(counts),fontsize=9,limits=(-1.5,1.5),show_y=wi==0)
                axes[pi,wi].set_title(f'Parent {parent} | {window} positions')
                f2[parent+'_'+window]={'ratio':np.asarray(cells).tolist(),'tasks_improved_out_of4':np.asarray(counts).tolist()}
        colorbar(fig,im,list(axes.ravel()),limits=(-1.5,1.5),description='MSE(A) / MSE(B); lower is better for A')
        title_footer(fig,'F2  Local value of signed and positional structure',synthetic,
            'KT96 • Original DEV σ • All models matched to the same DEV EDF\n'
            'Each cell: error ratio, then tasks improved / 4. Descriptive counts, not significance tests.\n'
            'Color range 0.35–2.83; printed numbers are not clipped. NA: undefined.')
        save(fig,out,FIGURE_IDS[1],layouts);values[FIGURE_IDS[1]]=f2;check()
        # F3: every root, including all counterexamples.
        fig,axes=plt.subplots(1,2,figsize=(10.5,5.2));fig.subplots_adjust(left=.13,right=.91,top=.82,bottom=.24,wspace=.17)
        ordered=sorted(A.ROOTS,key=lambda r:(r[1],r[2]));labels=[task+' / '+parent for _,task,parent in ordered];f3={}
        for wi,window in enumerate(('First 5','Full 10')):
            cells=np.stack([next(g for g in data['groups']if g['level']=='root'and g['label']==r and g['background']=='KT')['relative_to_zero'][index['current_full@edf_match'],wi,1]for r,_,_ in ordered])
            im=heat(axes[wi],cells,labels,fontsize=9,limits=(-1,3),show_y=wi==0);axes[wi].set_title(window+' positions');f3[window]=cells.tolist()
        colorbar(fig,im,list(axes),limits=(-1,3),description='MSE / zero MSE (log2 color)')
        title_footer(fig,'F3  Every root: complete-write transfer counterexamples',synthetic,
            'KT96 • Current complete signed write • Matched EDF • Original DEV σ\n'
            'All 8 roots and dimensions; 12 queries per root. Candidate positions were not executed.\n'
            'Color range 0.5–8; printed numbers are not clipped. NA: undefined.')
        save(fig,out,FIGURE_IDS[2],layouts);values[FIGURE_IDS[2]]=f3;check()
        # F4: the predeclared same-query interaction, not primary training support.
        fig,axes=plt.subplots(2,2,figsize=(10.5,6));fig.subplots_adjust(left=.22,right=.91,top=.84,bottom=.23,hspace=.64,wspace=.18)
        selected=('current4','current_full','headpos_signed');f4={}
        for pi,parent in enumerate(('C','T')):
            g=population(data,parent,'KC_minus_KT')
            for wi,window in enumerate(('First 5','Full 10')):
                cells=np.stack([g['relative_to_zero'][index[f+'@edf_match'],wi,1]for f in selected])
                im=heat(axes[pi,wi],cells,[LABELS[f]for f in selected],fontsize=9,limits=(-1,2),show_y=wi==0)
                axes[pi,wi].set_title(f'Parent {parent} | {window} positions');f4[parent+'_'+window]=cells.tolist()
        colorbar(fig,im,list(axes.ravel()),limits=(-1,2),description='Interaction MSE / zero-interaction MSE')
        title_footer(fig,'F4  Same-query K-background interaction prediction',synthetic,
            'Auxiliary KC−KT24 • Matched EDF • Same original DEV σ (not interaction-estimated)\n'
            'Context / original-norm / pooled models predict zero interaction structurally; their defined ratio is 1.\n'
            'No new fit. Color range 0.5–4; printed numbers are not clipped. NA: undefined.')
        save(fig,out,FIGURE_IDS[3],layouts);values[FIGURE_IDS[3]]=f4;check()
    return values,layouts


def atlas_groups(data):
    groups=[]
    for bg in ('KT','KC','KC_minus_KT'):
        for parent in ('C','T'):
            source=data['interaction_groups']if bg=='KC_minus_KT'else data['groups']
            for task in ('T06','T07','T08','T09','all'):
                level='population'if task=='all'else'task'
                g=next(g for g in source if (g['background'],g['parent'],g['level'],g['label'])==(bg,parent,level,task))
                root=None if task=='all'else next(r for r,t,p in A.ROOTS if t==task and p==parent)
                groups.append((g,root))
    return groups


def atlas(data,out,check):
    layouts=[];thumbs=[];groups=atlas_groups(data);A.need(len(groups)==30,'all30_unique_atlas_pages')
    with plt.rc_context(STYLE),PdfPages(Path(out)/'SUPPLEMENT-ATLAS.pdf')as pdf:
        for page,(g,root)in enumerate(groups,1):
            check();fig,axes=plt.subplots(2,2,figsize=(18,22),dpi=70)
            fig.subplots_adjust(left=.205,right=.935,top=.93,bottom=.07,hspace=.16,wspace=.13)
            for wi,window in enumerate(('First 5','Full 10')):
                for si,scale in enumerate(('Raw command units','Original DEV σ')):
                    im=heat(axes[wi,si],g['relative_to_zero'][:,wi,si],data['names'],fontsize=7,limits=(-2,5),show_y=si==0)
                    axes[wi,si].tick_params(axis='y',labelsize=7)
                    axes[wi,si].set_title(window+' | '+scale,fontsize=11)
            colorbar(fig,im,list(axes.ravel()),limits=(-2,5),description='MSE / zero MSE (log2 color)')
            label=g['label']+' / parent '+g['parent'];caption=f"Page {page}/30 • {g['background']} • {label} • n={g['observed_rows']} queries"
            if root:caption+='\nTask/root scores verified identical: '+root
            else:caption+='\nFour tasks equally weighted; root details appear on preceding pages.'
            title_footer(fig,caption,data['synthetic'],
                'All 37 predictors • Both windows and scales • No lambda selection • NA denotes an undefined ratio\n'
                'Color clipped to 0.25–32; printed ratios are not clipped. Parent, dimensions and overlapping positions are not independent tests.')
            layout=layout_check(fig,'atlas-'+str(page));layout.update(page=page,caption=caption,
                group_key=list(A.groupkey(g)),merged_root=root,all37_predictors=True,all_values_printed=True)
            layouts.append(layout);pdf.savefig(fig,dpi=110)
            pixels=np.asarray(fig.canvas.buffer_rgba())[::4,::4,:3].copy();thumbs.append(pixels);plt.close(fig)
    with plt.rc_context(STYLE):
        for sheet in range(2):
            fig,axes=plt.subplots(3,5,figsize=(15,12));fig.subplots_adjust(left=.01,right=.99,bottom=.02,top=.95,wspace=.03,hspace=.09)
            for j,ax in enumerate(axes.ravel()):
                page=sheet*15+j;ax.imshow(thumbs[page]);ax.set_xticks([]);ax.set_yticks([]);ax.axis('off');ax.set_title(f'Page {page+1}',fontsize=9)
            fig.suptitle(('SYNTHETIC — 'if data['synthetic']else'')+f'Complete atlas contact sheet {sheet+1}/2',fontsize=14)
            name=f'ATLAS-CONTACT-{sheet+1:02d}';layout_check(fig,name);fig.savefig(Path(out)/(name+'.png'),dpi=110);plt.close(fig)
    return layouts


def write_csv(data,path):
    with gzip.open(path,'wt',newline='',encoding='utf-8')as f:
        writer=csv.writer(f);writer.writerow(['level','label','parent','background','queries','model','window','scale','dimension','MSE','zero_MSE','MSE_over_zero','ratio_defined'])
        rows=0
        for g in data['groups']+data['interaction_groups']:
            for mi,model in enumerate(data['names']):
                for wi,window in enumerate(('first5','full10')):
                    for si,scale in enumerate(('raw','DEV_sigma')):
                        for dim,name in enumerate(A.DIMENSIONS):
                            value=g['relative_to_zero'][mi,wi,si,dim];defined=bool(g['ratio_defined'][wi,si,dim])
                            writer.writerow([g['level'],g['label'],g['parent'],g['background'],g['observed_rows'],model,window,scale,name,
                                repr(float(g['mse'][mi,wi,si,dim])),repr(float(g['target_mse'][wi,si,dim])),repr(float(value))if defined else'',defined]);rows+=1
    A.need(rows==54*37*2*2*7,'full_csv_complete_roster')
    return rows


def render(data,out,*,check=lambda:None):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    values,layouts=main_figures(data,out,check)
    pages=atlas(data,out,check);rows=write_csv(data,out/'FULL-SCORES.csv.gz')
    return {'figures':values,'layouts':layouts,'atlas_pages':pages,'CSV_rows':rows,
            'captions':CAPTIONS,'all_atlas_values':'FULL-SCORES.csv.gz','synthetic':data['synthetic']}
