"""Turn recovered Figure-4 marks into tidy CSVs and recompute summary statistics.
Inputs are coordinates recovered from the authors' vector figure (see extract_fig4_v9.py)."""
import json, csv, sys, numpy as np
R=json.load(open('extract/fig4_v9_raw_marks.json'))
ax={'Camera':dict(x0=107.15,k=(225.27-107.15)/0.4,rows={'Unedited':44.25,'Clean PCA':55.2,'Random':66.14}),
    'Sensor':dict(x0=121.91,k=(240.03-121.91)/0.08,rows={'Unedited':116.08,'Clean PCA':127.02,'Random':137.96})}
succ=dict(x0=395.65,k=(444.87-395.65)/10.0)
rows=[]
for fam,a in ax.items():
    for x,y in R['dots']:
        best=min(a['rows'],key=lambda r:abs(a['rows'][r]+3.7-y))
        allr={**ax['Camera']['rows'],**ax['Sensor']['rows']}
        if y<30: continue
        if abs(a['rows'][best]+3.7-y)<=5.6: rows.append((fam,best,(x-a['x0'])/a['k']))
with open('data/fig4_task_contrasts_recovered.csv','w',newline='') as f:
    w=csv.writer(f); w.writerow(['family','comparator','task_rank','control_minus_response_error']); 
    for fam in ax:
        for comp in ax[fam]['rows']:
            v=sorted(r[2] for r in rows if r[0]==fam and r[1]==comp)
            for i,x in enumerate(v): w.writerow([fam,comp,i+1,f'{x:.5f}'])
summ=[]
for fam in ax:
    for comp in ax[fam]['rows']:
        v=np.array([r[2] for r in rows if r[0]==fam and r[1]==comp])
        mean_mark=[h for h in R['hl'] if h[0].startswith('MEAN') and abs(h[3]-ax[fam]['rows'][comp])<0.1 and h[1]<300][0]
        ci=[h for h in R['hl'] if not h[0].startswith('MEAN') and abs(h[3]-ax[fam]['rows'][comp])<0.1 and h[2]<300 and h[5]==1.0][0]
        sm=[h for h in R['hl'] if h[0].startswith('MEAN') and abs(h[3]-ax[fam]['rows'][comp])<0.1 and h[1]>300][0]
        sci=[h for h in R['hl'] if not h[0].startswith('MEAN') and abs(h[3]-ax[fam]['rows'][comp])<0.1 and h[1]>300 and h[5]==1.0][0]
        s=np.sort(v)[::-1]; pos=v[v>0].sum(); tot=v.sum()
        summ.append(dict(family=fam,comparator=comp,n_tasks=len(v),mean_of_dots=v.mean(),drawn_mean=(mean_mark[1]-ax[fam]['x0'])/ax[fam]['k'],
            ci_lo=(ci[1]-ax[fam]['x0'])/ax[fam]['k'],ci_hi=(ci[2]-ax[fam]['x0'])/ax[fam]['k'],median=np.median(v),n_pos=int((v>0).sum()),n_neg=int((v<0).sum()),
            top3_share_of_net=s[:3].sum()/tot,top4_share_of_net=s[:4].sum()/tot,min=v.min(),max=v.max(),
            succ_gain_pp=(sm[1]-succ['x0'])/succ['k'],succ_lo=(sci[1]-succ['x0'])/succ['k'],succ_hi=(sci[2]-succ['x0'])/succ['k']))
keys=list(summ[0]); 
with open('data/fig4_summary_recomputed.csv','w',newline='') as f:
    w=csv.DictWriter(f,keys); w.writeheader(); [w.writerow({k:(f'{v:.5f}' if isinstance(v,float) else v) for k,v in r.items()}) for r in summ]
for r in summ: print(r['family'][:3],r['comparator'][:8].ljust(8),r['n_tasks'],'mean %.4f drawn %.4f CI[%.4f,%.4f] med %.5f pos %d neg %d top3 %.3f top4 %.3f | succ %+.2f [%.2f,%.2f]'%(r['mean_of_dots'],r['drawn_mean'],r['ci_lo'],r['ci_hi'],r['median'],r['n_pos'],r['n_neg'],r['top3_share_of_net'],r['top4_share_of_net'],r['succ_gain_pp'],r['succ_lo'],r['succ_hi']))
# full-L9 diamonds were not parsed as dots; heatmap -> per-case outcomes
cells=R['cells']; arms=['Native clean','Unedited','Clean PCA','Random','Response','Full L9']
ysort=sorted(set(round(c[1],1) for c in cells))
seps=sorted(v[1] for v in R['vl'] if v[0].startswith('rgb(100%') and v[5]==0.32)
out=[]
for fam,(lo,hi) in {'Camera':(90,270),'Sensor':(276,499)}.items():
    xs=sorted(set(round(c[0],2) for c in cells if lo<c[0]<hi)); sp=[s for s in seps if lo<s<hi]
    task=[sum(1 for s in sp if s<x)+1 for x in xs]
    grid={}
    for c in cells:
        if lo<c[0]<hi: grid[(round(c[0],2),round(c[1],1))]=c[2]
    for j,x in enumerate(xs):
        out.append([fam,j+1,task[j]]+[1-grid[(x,y)] for y in ysort])
with open('data/fig4_paired_outcomes_recovered.csv','w',newline='') as f:
    w=csv.writer(f); w.writerow(['family','case_col','task_block']+[a+'_success' for a in arms]); w.writerows(out)
O=np.array([[r[1],r[2]]+r[3:] for r in out if True],dtype=float); fams=np.array([r[0] for r in out])
for fam in ['Camera','Sensor']:
    A=O[fams==fam]; t=A[:,1]; S=A[:,2:]
    nt=len(set(t)); sizes=sorted(set(np.bincount(t.astype(int))[1:]))
    te=[np.mean([S[t==k,i].mean() for k in sorted(set(t))])*100 for i in range(6)]
    raw=S.sum(0)
    print(fam,'cases',len(A),'tasks',nt,'cases/task',sizes,'| task-equal %:',' '.join(f'{a[:6]} {v:.2f}' for a,v in zip(arms,te)),'| raw counts',raw.astype(int).tolist())
    for b,name in [(1,'Unedited'),(2,'Clean PCA'),(3,'Random'),(5,'Full L9→? (vs Unedited)')]:
        if b==5: base,new=S[:,1],S[:,5]; lab='Unedited→Full L9'
        else: base,new=S[:,b],S[:,4]; lab=f'{name}→Response'
        fs=(base==0)&(new==1); sf=(base==1)&(new==0)
        tfs=np.mean([fs[t==k].mean() for k in sorted(set(t))])*100; tsf=np.mean([sf[t==k].mean() for k in sorted(set(t))])*100
        print(f'   {lab}: F→S {fs.sum()} cases ({tfs:.2f}% task-equal, {len(set(t[fs]))} tasks), S→F {sf.sum()} ({tsf:.2f}%, {len(set(t[sf]))} tasks), net {tfs-tsf:+.2f} pp')
