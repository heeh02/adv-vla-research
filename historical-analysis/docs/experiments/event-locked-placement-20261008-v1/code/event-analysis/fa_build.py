import pickle,glob,numpy as np,pandas as pd
rows=[];names={}
for f in sorted(glob.glob('traj/shard-*.pkl')):
    o=pickle.load(open(f,'rb'))
    for ep,v in o['task'].items(): names[ep[0]]=v
    for ep,a in o['exec'].items():
        root,noise,cond,att=ep
        st=o['state'].get(ep,[]); res=o['result'].get(ep,{})
        g=a[:,6]; s=np.sign(g); s[s==0]=-1
        closes=list(np.where((s[1:]>0)&(s[:-1]<0))[0]+1)+([0] if s[0]>0 else [])
        closes=sorted(closes); opens=list(np.where((s[1:]<0)&(s[:-1]>0))[0]+1)
        steps=np.array([d['step'] for d in st]); E=np.stack([d['eef'] for d in st]) if st else np.zeros((0,3))
        def eef_at(t):
            if len(steps)==0: return np.full(3,np.nan)
            return np.array([np.interp(t,steps,E[:,k]) for k in range(3)])
        fc=closes[0] if closes else np.nan
        path=float(np.linalg.norm(np.diff(E,axis=0),axis=1).sum()) if len(E)>1 else np.nan
        dE=np.linalg.norm(np.diff(E,axis=0),axis=1) if len(E)>1 else np.array([])
        rows.append(dict(root=root,noise=noise,cond=cond,attempt=att,success=res.get('success'),steps=res.get('steps',len(a)),n_close=len(closes),n_open=len(opens),
            first_close=fc,eef_close=eef_at(fc) if closes else np.full(3,np.nan),eef_end=E[-1] if len(E) else np.full(3,np.nan),
            path=path,stall_frac=float((dE<0.002).mean()) if len(dE) else np.nan,n_states=len(st),
            closed_frac=float((s>0).mean())))
D=pd.DataFrame(rows)
D['task_name']=D.root.map(lambda r: names.get(r,{}).get('task_name'))
D['suite']=D.root.str.split('-').str[0]; D['task_id']=D.root.str.rsplit('-',n=1).str[0]
# keep one attempt per (root,noise,cond): highest attempt with result
D=D.sort_values('attempt').groupby(['root','noise','cond']).tail(1)
D.to_pickle('fa_episodes.pkl'); print(D.shape, D.cond.value_counts().to_dict()); print(D.success.value_counts(dropna=False).to_dict())
