import pickle,glob,numpy as np,pandas as pd
def blocks(p):
    out=[]; i=10
    while i+7<=len(p):
        q=p[i+3:i+7]
        if abs(np.linalg.norm(q)-1)<2e-3 and 0.3<p[i+2]<1.6: out.append(i); i+=7
        else: i+=1
    return out
rows=[]
for f in sorted(glob.glob('traj/shard-*.pkl')):
    o=pickle.load(open(f,'rb'))
    byroot={}
    for ep in o['state']: byroot.setdefault(ep[0],[]).append(ep)
    for root,eps in byroot.items():
        # reference: clean successful episodes of this root -> manipulated objects
        cl=[e for e in eps if e[2]=='clean' and o['result'].get(e,{}).get('success')]
        if not cl: continue
        p0=o['state'][cl[0]][0]['phys'].astype(float); B=blocks(p0)
        disp={b:[] for b in B}; final={b:[] for b in B}
        for e in cl:
            s0=o['state'][e][0]['phys'].astype(float); s1=o['state'][e][-1]['phys'].astype(float)
            if len(s1)!=len(p0): continue
            for b in B: disp[b].append(np.linalg.norm(s1[b:b+3]-s0[b:b+3])); final[b].append(s1[b:b+3])
        moved=[b for b in B if np.median(disp[b])>0.03]
        if not moved: continue
        ref_final={b:np.median(np.stack(final[b]),0) for b in moved}
        for e in eps:
            st=o['state'][e]; res=o['result'].get(e,{})
            P=np.stack([d['phys'].astype(float) for d in st]); 
            if P.shape[1]!=len(p0): continue
            for b in moved:
                z0=P[0,b+2]; xyz=P[:,b:b+3]
                rows.append(dict(root=root,noise=e[1],cond=e[2],obj=b,success=res.get('success'),
                    lift_max=float((xyz[:,2]-z0).max()),moved=float(np.linalg.norm(xyz[-1]-xyz[0])),
                    final_err=float(np.linalg.norm(xyz[-1]-ref_final[b])),final_err_xy=float(np.linalg.norm(xyz[-1,:2]-ref_final[b][:2])),
                    final_dz=float(xyz[-1,2]-ref_final[b][2])))
O=pd.DataFrame(rows); O.to_pickle('fa_objects.pkl'); print(O.shape, O.root.nunique(), O.groupby('root').obj.nunique().value_counts().to_dict())
