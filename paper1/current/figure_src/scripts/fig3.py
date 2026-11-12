"""Figure 3. Reading question: do 16 shared directions capture held-out responses, and does the layer that captures
most response energy also give the best fixed affine command prediction?
Data: data/fig3_task_scores.csv (per-task, response basis; shipped with the LaTeX package) and data/fig3_summary.csv."""
import csv, numpy as np
import figstyle as S; S.apply()
T_=list(csv.DictReader(open('data/fig3_task_scores.csv'))); Sm=list(csv.DictReader(open('data/fig3_summary.csv')))
fams=['Background','Camera','Light','Sensor Noise']; layers=['L1','L9','L18']
F=S.Fig(183); T=F.text; cv=F.cv
def swarm(v,log=False,step=0.052,width=0.24):
    v=np.log10(v) if log else np.asarray(v,float); o=np.argsort(v); x=np.zeros(len(v)); span=v.max()-v.min()+1e-9; done=[]
    for i in o:                                   # deterministic beeswarm on the non-data axis only
        c=0.0;k=0
        while any(abs(v[i]-v[j])<span*0.05 and abs(c-x[j])<step*0.95 for j in done):
            k+=1; c=((k+1)//2)*step*(1 if k%2 else -1)
            if abs(c)>width: break
        x[i]=c; done.append(i)
    return x
X0,CW,GAP=36,108.5,8; YA,HA,YB,HB=30,58,112,60
F.panel(0,7,'a','Response energy retained'); F.panel(0,101,'b','Command prediction from the same coordinates')
# legend (top right)
def tw(t):                                        # drawn width of a text artist, in points
    F.fig.canvas.draw(); return t.get_window_extent().width*72/F.fig.dpi
lx=132; GAPL=11
t=None
cv.scatter([lx],[7],s=4,color=S.ACC,alpha=0.35,lw=0,zorder=3); t=T(lx+4,7,'task',size=7.0,color=S.K); lx+=4+tw(t)+GAPL
cv.scatter([lx],[7],s=18,color=S.ACC,ec='white',lw=0.5,zorder=3); t=T(lx+5,7,'shared directions, mean (95% CI)',size=7.0,color=S.K); lx+=5+tw(t)+GAPL
cv.scatter([lx],[7],s=11,marker='D',fc='white',ec=S.CLEANC,lw=0.9,zorder=3); t=T(lx+5,7,'clean-feature directions',size=7.0,color=S.K); lx+=5+tw(t)+GAPL
cv.scatter([lx],[7],s=13,marker='x',color=S.RAND,lw=1.0,zorder=3); t=T(lx+5,7,'random directions',size=7.0,color=S.K); lx+=5+tw(t)+GAPL
cv.plot([lx-2,lx+10],[7,7],color=S.REF,lw=0.7,ls=S.DASH); t=T(lx+14,7,'no change',size=7.0,color=S.K); lx+=14+tw(t)
print('legend ends at',round(lx,1),'limit',S.W_PT-1); assert lx<S.W_PT-1
for c,fam in enumerate(fams):
    x=X0+c*(CW+GAP); a=F.axes(x,YA,CW,HA); b=F.axes(x,YB,CW,HB)
    T(x+CW/2,21,fam,size=7.4,color=S.K,ha='center')
    for i,L in enumerate(layers):
        r=[t for t in T_ if t['family']==fam and t['layer']==L]; s=[m for m in Sm if m['family']==fam and m['layer']==L][0]
        e=np.array([float(t['energy_retained_pct']) for t in r]); m=np.array([float(t['mse_over_family_mean_nochange']) for t in r])
        xo,xm=i-0.22,i+0.10
        a.scatter(xo+swarm(e),e,s=4.2,color=S.ACC,alpha=0.32,lw=0,zorder=2)
        mu,lo,hi=float(s['energy_mean']),float(s['energy_ci_lo_boot']),float(s['energy_ci_hi_boot'])
        a.plot([xm,xm],[lo,hi],color=S.ACC,lw=1.0,zorder=3); a.plot([xm-0.045,xm+0.045],[lo,lo],color=S.ACC,lw=1.0,zorder=3); a.plot([xm-0.045,xm+0.045],[hi,hi],color=S.ACC,lw=1.0,zorder=3)
        a.scatter([xm],[mu],s=20,color=S.ACC,ec='white',lw=0.5,zorder=4)
        a.scatter([i+0.27],[float(s['energy_clean'])],s=12,marker='D',fc='white',ec=S.CLEANC,lw=0.9,zorder=3); a.scatter([i+0.43],[float(s['energy_random'])],s=13,marker='x',color=S.RAND,lw=1.0,zorder=3)
        b.scatter(xo+swarm(m,log=True),m,s=4.2,color=S.ACC,alpha=0.32,lw=0,zorder=2)
        b.scatter([xm],[float(s['mse_ratio_response'])],s=20,color=S.ACC,ec='white',lw=0.5,zorder=4)
        b.scatter([i+0.27],[float(s['mse_ratio_clean'])],s=12,marker='D',fc='white',ec=S.CLEANC,lw=0.9,zorder=3); b.scatter([i+0.43],[float(s['mse_ratio_random'])],s=13,marker='x',color=S.RAND,lw=1.0,zorder=3)
    a.set_ylim(-4,100); a.set_yticks([0,25,50,75,100]); b.set_yscale('log'); b.set_ylim(2e-3,25); b.set_yticks([1e-2,1e-1,1,10]); b.minorticks_off()
    b.axhline(1,color=S.REF,lw=0.7,ls=S.DASH,zorder=1); b.axhspan(2e-3,1,color=S.BAND,lw=0,zorder=0)
    for ax in (a,b):
        ax.set_xlim(-0.6,2.65); ax.set_xticks([0,1,2]); ax.tick_params(axis='x',length=0); ax.grid(axis='y',color=S.RULE,lw=0.5,zorder=0); ax.set_axisbelow(True)
        ax.spines['left'].set_visible(c==0)
        if c: ax.tick_params(axis='y',length=0,labelleft=False)
    a.set_xticklabels([]); b.set_xticklabels(layers)
    if c==0:
        b.set_yticklabels(['0.01','0.1','1','10']); a.set_ylabel('Energy (%)',labelpad=2); b.set_ylabel('MSE / no change',labelpad=2)
        b.text(-0.52,0.0027,'better than no change',fontsize=7.0,color=S.K,va='bottom',ha='left')
F.save('figs_v3/figure3_geometry')
