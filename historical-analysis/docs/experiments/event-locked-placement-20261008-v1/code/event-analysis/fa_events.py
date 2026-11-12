import pickle,glob,numpy as np,pandas as pd
def blocks(p):
    out=[]; i=10
    while i+7<=len(p):
        q=p[i+3:i+7]
        if abs(np.linalg.norm(q)-1)<2e-3 and 0.3<p[i+2]<1.6: out.append(i); i+=7
        else: i+=1
    return out
LIFT=0.03
rows=[]; names={}
for f in sorted(glob.glob('traj/shard-*.pkl')):
    o=pickle.load(open(f,'rb'))
    for ep,v in o['task'].items(): names[ep[0]]=v.get('task_name')
    byroot={}
    for ep in o['state']: byroot.setdefault(ep[0],[]).append(ep)
    for root,eps in byroot.items():
        cl=[e for e in eps if e[2]=='clean' and o['result'].get(e,{}).get('success')]
        if not cl: continue
        p0=o['state'][cl[0]][0]['phys'].astype(float); B=blocks(p0)
        disp={b:[] for b in B}; fin={b:[] for b in B}; lifted={b:[] for b in B}
        for e in cl:
            P=np.stack([d['phys'].astype(float) for d in o['state'][e]])
            if P.shape[1]!=len(p0): continue
            for b in B:
                disp[b].append(np.linalg.norm(P[-1,b:b+3]-P[0,b:b+3])); fin[b].append(P[-1,b:b+3]); lifted[b].append((P[:,b+2]-P[0,b+2]).max())
        # manipulated = moved >3cm in clean success; 'picked' objects also lifted >3cm
        man=[b for b in B if np.median(disp[b])>0.03]
        if not man: continue
        ref={b:np.median(np.stack(fin[b]),0) for b in man}; picked={b:np.median(lifted[b])>LIFT for b in man}
        for e in eps:
            st=o['state'][e]; res=o['result'].get(e,{}); a=o['exec'].get(e)
            if a is None or not st: continue
            P=np.stack([d['phys'].astype(float) for d in st]); steps=np.array([d['step'] for d in st])
            if P.shape[1]!=len(p0): continue
            g=np.sign(a[:,6]); g[g==0]=-1
            closes=sorted(list(np.where((g[1:]>0)&(g[:-1]<0))[0]+1)+([0] if g[0]>0 else []))
            for b in man:
                z=P[:,b+2]-P[0,b+2]; xy=P[:,b:b+2]
                lift_t=steps[np.argmax(z>LIFT)] if (z>LIFT).any() else np.nan
                # attempts before first lift
                att_before=sum(1 for c in closes if np.isnan(lift_t) or c<=lift_t)
                # drops: after lifting, falls back near initial height away from ref
                fe=np.linalg.norm(P[-1,b:b+3]-ref[b]); fexy=np.linalg.norm(P[-1,b:b+2]-ref[b][:2]); dz=P[-1,b+2]-ref[b][2]; dxy=P[-1,b:b+2]-ref[b][:2]
                rows.append(dict(root=root,noise=e[1],cond=e[2],obj=b,picked_type=picked[b],success=res.get('success'),steps=res.get('steps'),
                   n_close=len(closes),lift_t=lift_t,ever_lift=bool((z>LIFT).any()),attempts_to_lift=att_before,
                   final_err=fe,final_err_xy=fexy,final_dz=dz,dx=dxy[0],dy=dxy[1],grip_end=float(g[-1]),eef_obj_end=float(np.linalg.norm(st[-1]['eef']-P[-1,b:b+3])),moved=float(np.linalg.norm(P[-1,b:b+3]-P[0,b:b+3])),max_lift=float(z.max()),task_name=names.get(root)))
O=pd.DataFrame(rows); O['task_id']=O.root.str.rsplit('-',n=1).str[0]
O.to_pickle('fa_events.pkl'); print(O.shape, O.root.nunique())
