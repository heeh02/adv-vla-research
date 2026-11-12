#!/usr/bin/env python3
"""Render regenerated D1 estimates; never re-estimate or select scientific rows."""
import argparse
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

FAMILIES=('Camera Viewpoints','Sensor Noise','Background Textures','Light Conditions')
SHORT={'Camera Viewpoints':'Camera','Sensor Noise':'Sensor Noise','Background Textures':'Background','Light Conditions':'Light'}
MODELS=('pi05_libero','pi0_libero')
COLORS=('#24647b','#b3602b')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('output',type=Path);a=ap.parse_args();source=a.output/'RESULTS.json';data=json.loads(source.read_text())
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'svg.fonttype':'none','pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False,'axes.spines.left':False})
    fig,axes=plt.subplots(1,3,figsize=(11.6,4),gridspec_kw={'width_ratios':[1,1.1,1.2]})
    for col,ax in enumerate(axes):
        ax.set_yticks(range(4),[SHORT[f] if col==0 else '' for f in FAMILIES]);ax.set_ylim(3.6,-.6);ax.tick_params(axis='y',length=0);ax.grid(axis='x',color='#e4e7e9',lw=.6);ax.set_axisbelow(True)
    for fi,family in enumerate(FAMILIES):
        for mi,model in enumerate(MODELS):
            panel=next(p for p in data['panels'] if p['model']==model and p['scope']=='common28' and p['family']==family);y=fi+(-.13 if mi==0 else .13);color=COLORS[mi]
            for ci,name in [(0,'tr_flip'),(1,'risk_difference')]:
                v=panel['outcomes']['metrics'][name];ax=axes[ci];interval=v['ci95'] if ci==0 else v['ci98_75']
                if v['point'] is not None:
                    value=100*v['point'];ax.errorbar(value,y,xerr=[[value-100*interval[0]],[100*interval[1]-value]],fmt='o' if mi==0 else 'D',color=color,ms=4,capsize=2,lw=1)
                else:
                    lo,hi=v['identification_bounds'];ax.add_patch(Rectangle((100*lo,y-.07),100*(hi-lo),.14,facecolor='none',edgecolor=color,hatch='////',lw=.8))
            motion=next(m for m in panel['motion'] if m['clock']==5 and m['metric']=='position_distance' and m['reference']=='R0_R1');v=motion['point']['ratio_of_task_means'];ci=motion['ci95']['ratio_of_task_means']
            if v is not None:
                axes[2].errorbar(v,y,xerr=[[v-ci[0]],[ci[1]-v]],fmt='o' if mi==0 else 'D',color=color,ms=4,capsize=2,lw=1,label=r'$\pi_{0.5}$' if fi==0 and mi==0 else r'$\pi_0$' if fi==0 else None)
                axes[2].text(4.02,y,f'{motion["joint_variants"]}/{motion["planned_variants"]}',color=color,fontsize=8,va='center')
    axes[0].set_title('a  Paired outcome changes',loc='left',weight='bold');axes[0].set_xlabel('Flip rate (%) · 95% CI');axes[0].set_xlim(-1,50)
    axes[1].set_title('b  Net task cost',loc='left',weight='bold');axes[1].set_xlabel('Success change (pp) · 98.75% CI');axes[1].set_xlim(-45,18);axes[1].axvline(0,color='#7b8186',lw=.8,ls='--')
    axes[2].set_title('c  Early motion vs. sampling',loc='left',weight='bold');axes[2].set_xlabel('Step 5 EEF ratio · 95% CI');axes[2].set_xlim(0,4.5);axes[2].axvline(1,color='#7b8186',lw=.8,ls='--');axes[2].legend(frameon=False,loc='lower right',ncol=2);axes[2].text(4.02,-.48,'n / plan',fontsize=8,color='#555')
    fig.suptitle('Recomputed five-arm protocol: fixed outcomes, matched motion support',x=.07,ha='left',fontsize=12,weight='bold')
    fig.text(.07,.035,'Common 28 tasks, 202 planned variants per model. Hatched Light ranges are marginal identification bounds, not CIs.\nMotion uses the ratio of task-equal means on the same eligible cases; its support does not replace the outcome denominator.',fontsize=8.4,va='bottom')
    fig.subplots_adjust(left=.12,right=.98,top=.79,bottom=.25,wspace=.22)
    outputs=[]
    for ext in ('svg','png','pdf'):
        path=a.output/('D1-PRIMARY.'+ext);fig.savefig(path,dpi=180,metadata={'Creator':'D1 derived-record pipeline'} if ext!='png' else None);outputs.append({'path':path.name,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    plt.close(fig)
    (a.output/'FIGURE-RECEIPT.json').write_text(json.dumps({'source':'RESULTS.json','source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'render_code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'matplotlib':matplotlib.__version__,'outputs':outputs,'new_estimates':False},indent=2)+'\n')
if __name__=='__main__':main()
