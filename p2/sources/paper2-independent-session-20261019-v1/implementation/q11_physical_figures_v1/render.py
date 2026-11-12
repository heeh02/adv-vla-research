"""Publication figures from all observed points; never selects an effect maximum."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

COLORS = {"S-N": "#1875A1", "R-N": "#D75A30", "S-R": "#77539C"}


def style():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.2,
        "axes.titlesize": 8.2, "axes.labelsize": 7.2, "xtick.labelsize": 6.7,
        "ytick.labelsize": 6.7, "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": .6, "lines.linewidth": 1.25, "pdf.fonttype": 42,
        "svg.fonttype": "none", "savefig.facecolor": "white"})


def finish(fig, out, name):
    for ext in ("pdf", "svg", "png"):
        fig.savefig(Path(out)/(name+'.'+ext), dpi=240, metadata={"Creator": "Matplotlib"} if ext=='pdf' else None)
    plt.close(fig)


def render(records, out):
    style()
    groups = {name: [r for r in records if r["contrast"] == name] for name in COLORS}
    x = np.array([r["elapsed_seconds"] for r in groups["S-N"]])*1000
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.5))
    fig.subplots_adjust(left=.075, right=.99, bottom=.25, top=.74, wspace=.54)
    for name in ("S-N", "R-N"):
        rows = groups[name]
        axes[0].plot(x, [r['eef_position_distance_m']*1e6 for r in rows], '-o', color=COLORS[name], markersize=3)
        axes[1].plot(x, [r['eef_orientation_geodesic_rad']*1e6 for r in rows], '-o', color=COLORS[name], markersize=3)
        for finger, line in enumerate(('-', '--')):
            axes[2].plot(x, [r['gripper_qpos_delta_original_units'][finger]*1e6 for r in rows],
                         line, color=COLORS[name], marker='o' if finger==0 else 's', markersize=2.7)
    axes[3].plot(x, [r['reference_contact_count'] for r in groups['S-N']], '-s', color='#454545',
                 markersize=4, markerfacecolor='white', label='N', zorder=1)
    axes[3].plot(x, [r['candidate_contact_count'] for r in groups['S-N']], '-x', color=COLORS['S-N'],
                 markersize=4, label='S', zorder=3)
    axes[3].plot(x, [r['candidate_contact_count'] for r in groups['R-N']], ':o', color=COLORS['R-N'],
                 markersize=2.7, label='R', zorder=2)
    for ax, title, ylabel in zip(axes, ['a  Position', 'b  Orientation', 'c  Finger state', 'd  Contacts'],
        [r'Distance ($\mu$m)', r'Geodesic distance ($\mu$rad)', 'Joint difference\n($10^{-6}$ sim units)', 'Contact count']):
        ax.set_title(title, loc='left', pad=7)
        ax.set_ylabel(ylabel, labelpad=3)
        ax.set_xlabel('Elapsed time (ms)', labelpad=3)
        ax.set_xticks([50, 150, 250])
        ax.grid(axis='y', linewidth=.35, color='#d9dfe3')
        ax.set_axisbelow(True)
    axes[0].set_ylim(bottom=0);axes[1].set_ylim(bottom=0)
    axes[2].axhline(0, color='#666666', linewidth=.5)
    axes[3].set_yticks([27,28,29]);axes[3].set_ylim(26.7,29.4)
    axes[3].legend(frameon=False, fontsize=6, ncol=3, loc='upper right', handlelength=.9, columnspacing=.55)
    legend=[Line2D([],[],color=COLORS[n],marker='o',markersize=3,label=n) for n in ('S-N','R-N')]
    legend += [Line2D([],[],color='#555555',linestyle=s,label='Finger '+str(i+1)) for i,s in enumerate(('-','--'))]
    fig.legend(handles=legend,loc='upper center',bbox_to_anchor=(.5,.95),ncol=4,frameon=False,fontsize=7,
               columnspacing=1.3,handlelength=1.8)
    fig.text(.5,.98,'One state, five commands | identical start and exact N repeat',ha='center',va='top',fontsize=8)
    fig.text(.5,.025,'Direct gripper targets match at every step; joint positions need not. No task-benefit claim.',
             ha='center',va='bottom',fontsize=6.6,color='#333333')
    finish(fig,out,'fig_physical_response')

    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.9))
    fig.subplots_adjust(left=.09,right=.985,bottom=.13,top=.86,hspace=.65,wspace=.52)
    titles=['Position distance','Orientation distance','Contact-count difference',
            'Signed EEF coordinates','Finger 1 state difference','Finger 2 state difference']
    ylabels=[r'$\mu$m',r'$\mu$rad','Candidate minus reference',r'$\mu$m',r'$10^{-6}$ sim units',r'$10^{-6}$ sim units']
    for name,rows in groups.items():
        col=COLORS[name]
        for ax,values in [(axes[0,0],[r['eef_position_distance_m']*1e6 for r in rows]),
                          (axes[0,1],[r['eef_orientation_geodesic_rad']*1e6 for r in rows]),
                          (axes[0,2],[r['candidate_contact_count']-r['reference_contact_count'] for r in rows])]:
            ax.plot(x,values,'-o',color=col,markersize=3)
        for coordinate,line in enumerate(('-','--',':')):
            axes[1,0].plot(x,[r['eef_delta_xyz_m'][coordinate]*1e6 for r in rows],line,color=col)
        for finger in (0,1):
            axes[1,finger+1].plot(x,[r['gripper_qpos_delta_original_units'][finger]*1e6 for r in rows],
                                 '-o',color=col,markersize=3)
    for ax,title,yl in zip(axes.flat,titles,ylabels):
        ax.set_title(title,loc='left');ax.set_ylabel(yl);ax.set_xlabel('Elapsed time (ms)')
        ax.set_xticks([50,100,150,200,250]);ax.grid(axis='y',color='#d9dfe3',linewidth=.4)
        ax.set_axisbelow(True);ax.axhline(0,color='#888888',linewidth=.4)
    axes[0,2].set_yticks([-1,0,1])
    handles=[Line2D([],[],color=COLORS[n],marker='o',label=n,markersize=3)for n in COLORS]
    handles += [Line2D([],[],color='#555555',linestyle=s,label=axis)for s,axis in zip(('-','--',':'),('x','y','z'))]
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,.98),ncol=6,frameon=False,fontsize=7)
    fig.text(.5,.02,'All 15 pair-time observations; one DEV state. Lines connect samples, not fitted dynamics.',
             ha='center',fontsize=7)
    finish(fig,out,'fig_physical_support')
