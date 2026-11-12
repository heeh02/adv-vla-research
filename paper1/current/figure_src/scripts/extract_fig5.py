"""Recover Figure 5 data from the authors' matplotlib SVG (Figure5_Refined (1).svg).
Provenance: drawn-mark coordinates, not raw records."""
import re, bisect, collections, json, sys, csv
import numpy as np
s=open(sys.argv[1]).read()
pos=[(m.start(),m.group(1)) for m in re.finditer(r'<g id="([A-Za-z0-9_]+)"',s)]; starts=[p[0] for p in pos]
uses=collections.defaultdict(list)
for m in re.finditer(r'<use xlink:href="#([^"]+)" x="([-\d.]+)" y="([-\d.]+)" style="([^"]*)"',s):
    uses[pos[bisect.bisect(starts,m.start())-1][1]].append((float(m.group(2)),float(m.group(3)),m.group(4)))
A=lambda x:10**((x-180.2)/((221.7-97.2)/3))          # panel a: ratio (log)
B=lambda x:(x-347.7)/((347.7-263.2)/40)              # panel b: pp
C=lambda x:(x-465.7)/((486.0-465.7)/10)              # panel c: % (negative = lost side)
D=lambda x:10**((x-322.5)/(322.5-186.8))             # panel d: ratio (log)
rows=['B_pi05','B_pi0','C_pi05','C_pi0','S_pi05','S_pi0','L_pi05','L_pi0']
out={}
# (a) task dots
T=[]
for i,r in enumerate(rows):
    for x,y,_ in uses[f'PathCollection_{i+1}']: T.append((r,A(x)))
# (a) means + CI: line2d_9..24 caps, 25..32 means
caps=[uses[f'line2d_{k}'][0][0] for k in range(9,25)]; means=[uses[f'line2d_{k}'][0][0] for k in range(25,33)]
out['a']={r:dict(ratio=A(means[i]),lo=A(caps[2*i]),hi=A(caps[2*i+1]),n_task_points=sum(1 for t in T if t[0]==r)) for i,r in enumerate(rows)}
caps=[uses[f'line2d_{k}'][0][0] for k in range(41,53)]; means=[uses[f'line2d_{k}'][0][0] for k in range(53,59)]
out['b']={r:dict(change_pp=B(means[i]),lo=B(caps[2*i]),hi=B(caps[2*i+1])) for i,r in enumerate(rows[:6])}
# patches: rectangles
P=[]
for m in re.finditer(r'<g id="(patch_\d+)">\s*<path d="([^"]*)"[^>]*style="([^"]*)"',s):
    v=[float(t) for t in re.findall(r'-?\d+\.?\d*',m.group(2))]; xs=v[0::2]; ys=v[1::2]
    P.append((m.group(1),min(xs),max(xs),min(ys),max(ys),m.group(3)[:70]))
out['patches']=P
cls=['SS','SF','FS','FF']; Dp=[]
for i,c in enumerate(cls):
    for x,y,_ in uses[f'PathCollection_{9+i}']: Dp.append((c,D(x)))
json.dump(out,open('extract/fig5_marks.json','w'),indent=0)
with open('data/fig5a_task_motion_ratio_recovered.csv','w',newline='') as f:
    w=csv.writer(f); w.writerow(['family_policy','task_ratio_control5']); [w.writerow([r,f'{v:.5g}']) for r,v in T]
with open('data/fig5d_pairs_recovered.csv','w',newline='') as f:
    w=csv.writer(f); w.writerow(['outcome_class','pair_ratio_control5']); [w.writerow([c,f'{v:.6g}']) for c,v in Dp]
print('task points',len(T),'pairs',len(Dp))
for r in rows: print(r,{k:round(v,3) for k,v in out['a'][r].items()}, {k:round(v,2) for k,v in out['b'].get(r,{}).items()})
for p in P:
    if p[3]>30 and p[4]<120: print(p[0],'x[%.1f,%.1f] y[%.1f,%.1f]'%p[1:5],p[5][:48], 'b:[%.2f,%.2f] c:[%.2f,%.2f]'%(B(p[1]),B(p[2]),C(p[1]),C(p[2])))
d=dict((c,np.array([v for k,v in Dp if k==c])) for c in cls)
sf,ss=d['SF'],d['SS']
wins=sum((a>b)+0.5*(a==b) for a in sf for b in ss)
print('AUROC SF>SS: %g/%d = %.6f'%(wins,len(sf)*len(ss),wins/(len(sf)*len(ss))))
for c in cls: print(c,len(d[c]),'below 1:',int((d[c]<1).sum()),'min %.4f max %.3f'%(d[c].min(),d[c].max()))
near=np.sort(ss[(ss>0.74)&(ss<0.79)]); print('SS in 0.74-0.79:',len(near),near.round(4).tolist())
