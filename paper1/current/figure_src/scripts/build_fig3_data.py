"""Figure 3 data. Response-basis per-task values: analysis_archive CSV shipped in the LaTeX package (real saved task scores).
Clean/random energy means and the drawn response intervals: recovered from the authors' earlier vector Figure 3.
Clean/random/no-change MSE: Supplement absolute-error table (sections/current_basis_control_table.tex)."""
import json, csv, re, numpy as np
M=json.load(open('extract/fig3_marks.json'))
E=lambda y:(89.3-y)/((89.3-29.6)/100)+0.07  # +0.07: offset calibrated on the 12 response means vs. the CSV
axes=sorted({k.split('|')[0] for k in M},key=lambda a:int(a.split('_')[1]))
fam_names=['Background','Camera','Light','Sensor Noise']; layers=['L1','L9','L18']
top=[a for a in axes if any(('PathCollection' in k and k.startswith(a+'|') and max(u[1] for u in v)<95) for k,v in M.items())]
rec={}
for fam,a in zip(fam_names,top):
    marks=[(k,v[0]) for k,v in M.items() if k.startswith(a+'|line2d') and v[0][1]<91.5]
    for col in ['#69747c','#ac714f','#235a78']:
        pts=sorted([(m[0],m[1]) for k,m in marks if col in m[2]])
        xs=sorted({round(p[0],1) for p in pts})
        for L,x in zip(layers,xs):
            ys=[p[1] for p in pts if round(p[0],1)==x]
            name={'#69747c':'random','#ac714f':'clean','#235a78':'response'}[col]
            vals=sorted(E(y) for y in ys)
            rec[(fam,L,name)]=(vals[0],float(np.median(vals)),vals[-1])
rows=list(csv.DictReader(open('../latex/analysis_archive/section4-task-ordering/three_layer_task_checks.csv')))
fmap={'Background Textures':'Background','Camera Viewpoints':'Camera','Light Conditions':'Light','Sensor Noise':'Sensor Noise'}
tab={}
for line in open('../latex/sections/current_basis_control_table.tex'):
    m=re.match(r'(L\d+) & ([A-Za-z ]+) & ([\d.]+) & ([\d.]+) & ([\d.]+) & ([\d.]+) & ([\d.]+) & ([\d.]+)',line)
    if m: tab[(m.group(2).strip(),m.group(1))]=dict(response=float(m.group(3)),clean=float(m.group(4)),random=float(m.group(5)),nochange=float(m.group(8)))
rng=np.random.default_rng(20261110); out=[]
print('family layer | resp mean(csv) drawn  CI drawn        CI boot(mine)   | clean rand energy | MSE ratio resp(csv) table  clean  random')
for fam in fam_names:
    R=[r for r in rows if fmap[r['family']]==fam]; z=np.array([float(r['zero_mse']) for r in R])
    for L in layers:
        g=np.array([float(r['g_'+L]) for r in R])*100; mse=np.array([float(r['MSE_'+L]) for r in R])
        idx=rng.integers(0,len(g),(10000,len(g))); bm=g[idx].mean(1); lo,hi=np.percentile(bm,[2.5,97.5])
        t=tab[(fam,L)]; d=rec[(fam,L,'response')]
        o=dict(family=fam,layer=L,n_tasks=len(g),energy_mean=g.mean(),energy_ci_lo_drawn=d[0],energy_ci_hi_drawn=d[2],energy_ci_lo_boot=lo,energy_ci_hi_boot=hi,
               energy_clean=rec[(fam,L,'clean')][1],energy_random=rec[(fam,L,'random')][1],
               mse_ratio_response=mse.mean()/z.mean(),mse_ratio_clean=t['clean']/t['nochange'],mse_ratio_random=t['random']/t['nochange'],
               mse_ratio_response_table=t['response']/t['nochange'],tasks_below_mean_nochange=int((mse/z.mean()<1).sum()),tasks_beating_own_nochange=int((mse<z).sum()))
        out.append(o)
        print(f"{fam[:6]:6s} {L:3s} | {g.mean():6.2f} {d[1]:6.2f} [{d[0]:.1f},{d[2]:.1f}] [{lo:.1f},{hi:.1f}] | {o['energy_clean']:5.2f} {o['energy_random']:5.2f} | {o['mse_ratio_response']:.3f} {o['mse_ratio_response_table']:.3f} {o['mse_ratio_clean']:.3f} {o['mse_ratio_random']:.3f} | <1: {o['tasks_below_mean_nochange']}/28 own: {o['tasks_beating_own_nochange']}/28")
with open('data/fig3_summary.csv','w',newline='') as f:
    w=csv.DictWriter(f,list(out[0])); w.writeheader(); [w.writerow({k:(f'{v:.5g}' if isinstance(v,float) else v) for k,v in o.items()}) for o in out]
with open('data/fig3_task_scores.csv','w',newline='') as f:
    w=csv.writer(f); w.writerow(['family','task','layer','energy_retained_pct','mse','own_nochange_mse','mse_over_family_mean_nochange'])
    for fam in fam_names:
        R=[r for r in rows if fmap[r['family']]==fam]; zm=np.mean([float(r['zero_mse']) for r in R])
        for r in R:
            for L in layers: w.writerow([fam,r['task_component'],L,f"{float(r['g_'+L])*100:.4f}",r['MSE_'+L],r['zero_mse'],f"{float(r['MSE_'+L])/zm:.5g}"])
allc=[o['energy_clean'] for o in out]; allr=[o['energy_random'] for o in out]; alle=[o['energy_mean'] for o in out]
print('ranges: response %.1f-%.1f clean %.1f-%.1f random %.1f-%.1f'%(min(alle),max(alle),min(allc),max(allc),min(allr),max(allr)))
