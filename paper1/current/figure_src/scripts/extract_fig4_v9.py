"""Recover Figure 4 data from the authors' vector PDF (v9, converted with pdftocairo -svg).
Provenance: coordinates of drawn marks, NOT raw experiment records. Precision ~1e-4 command units."""
import re, sys, json, csv, collections
import numpy as np
src, outdir = sys.argv[1], sys.argv[2]
s = open(src).read(); body = s[s.index('</defs>'):]
M = (0.99759, -0.99759, 0.607229, 247.801)  # matplotlib->page transform used by cairo for stroked paths
def nums(d): return [float(x) for x in re.findall(r'-?\d+\.?\d*(?:e-?\d+)?', d)]
dots=[]; cells=[]; hl=[]; vl=[]
for p in re.findall(r'<path ([^>]*?)/>', body):
    fill=re.search(r'fill="([^"]*)"',p).group(1) if re.search(r'fill="([^"]*)"',p) else None
    stroke=re.search(r'stroke="([^"]*)"',p).group(1) if re.search(r'stroke="([^"]*)"',p) else None
    d=re.search(r'\bd="([^"]*)"',p).group(1); v=nums(d); xs=np.array(v[0::2]); ys=np.array(v[1::2])
    tr=re.search(r'transform="matrix\(([^)]*)\)"',p)
    if tr:
        a,b,c,dd,e,f=[float(t) for t in tr.group(1).split(',')]; xs=a*xs+e; ys=dd*ys+f
    ntok=len(d.split())
    if fill and fill.startswith('rgb(47.05') and ntok==63: dots.append((xs.min()/2+xs.max()/2, ys.min()/2+ys.max()/2))
    elif fill and ntok==15 and (fill.startswith('rgb(88.6') or fill.startswith('rgb(58.4')):
        cells.append((xs.min()/2+xs.max()/2, ys.min()/2+ys.max()/2, int(fill.startswith('rgb(58.4'))))
    elif stroke and fill=='none' and ntok==6:
        (hl if abs(ys[0]-ys[1])<1e-6 else vl).append((stroke, xs.min(), xs.max(), ys.min(), ys.max(), float(re.search(r'stroke-width="([^"]*)"',p).group(1))))
    elif fill and stroke and ntok in (63,16) and (fill.startswith('rgb(15.29') or fill.startswith('rgb(50.19')):
        hl.append(('MEAN'+fill, xs.min()/2+xs.max()/2, xs.min()/2+xs.max()/2, ys.min()/2+ys.max()/2, ys.min()/2+ys.max()/2, 0))
json.dump(dict(dots=dots,cells=cells,hl=hl,vl=vl), open(outdir+'/fig4_v9_raw_marks.json','w'))
print(len(dots),len(cells),len(hl),len(vl))
ys=sorted(set(round(d[1]) for d in dots)); print('dot y range', min(d[1] for d in dots), max(d[1] for d in dots))
print(sorted(collections.Counter(round(c[1],1) for c in cells).items()))
for h in sorted(hl,key=lambda t:(t[3],t[1])): print('H',h[0][:22],[round(t,2) for t in h[1:]])
for v in sorted(vl,key=lambda t:(t[3],t[1])): print('V',v[0][:22],[round(t,2) for t in v[1:]])
