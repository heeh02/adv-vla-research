"""CPU-only held-out action readout using the exact Q1 paired source plus T-actions."""
from __future__ import annotations
from collections import defaultdict
import argparse, csv, hashlib, io, json, pathlib, time, zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from bos_stdlib import BOSWriter
import reader
import readset_contract as R

RUN='jinwen-adv-vla-plus-action-readout-20261010-v1'
OUTPUT='redacted-4d65b032b46b'
FAMILIES=('Background Textures','Camera Viewpoints','Light Conditions','Sensor Noise')
LAYERS=(0,1,2); CALL=0; DIM_RAW=32; DIM_ADAPTER=7

def need(ok,code):
    if not ok: raise RuntimeError(code)
def sha(b):return hashlib.sha256(b).hexdigest()
def write_json(p,x):pathlib.Path(p).write_text(json.dumps(x,sort_keys=True,indent=2,allow_nan=False)+'\n')

def read_action(case,member):
    raw=case.member(member)
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        names=set(z.namelist());need(names=={'raw_actions.npy','actions.npy','__schema__.npy'},'T_actions_inventory')
        out={}
        for key in ('raw_actions','actions'):
            with z.open(key+'.npy') as f:out[key]=np.lib.format.read_array(f,allow_pickle=False)
    raw32=np.asarray(out['raw_actions'],np.float32);adapter=np.asarray(out['actions'],np.float64)
    need(raw32.shape==(1,10,32) and adapter.shape==(1,10,7) and np.isfinite(raw32).all() and np.isfinite(adapter).all(),'T_actions_ABI')
    return raw32[0],adapter[0]

def load_pair(row,signer,deadline):
    members=row['trajectory_members'];case=reader.CaseReader(signer,row,members,deadline)
    hidden={};actions={};proof={}
    for arm in ('reference','treatment'):
        traj=f'M/{arm}/T-trajectory.npz';act=f'M/{arm}/T-actions.npz'
        h=np.asarray(case.fields(traj,('hidden',))['hidden'],np.float32)
        need(h.shape==(10,3,1,10,1024) and np.isfinite(h).all(),'T_hidden_ABI')
        raw32,adapter=read_action(case,act)
        hidden[arm]=h[:,:,0,:,:];actions[arm]={'raw32':raw32,'adapter7':adapter}
        proof[arm]={'hidden_member':case.native[traj],'action_member':case.native[act]}
    need(np.array_equal(hidden['reference'][...,:].shape,hidden['treatment'].shape),'paired_hidden_shape')
    return {'row':row,'hidden_delta':hidden['treatment']-hidden['reference'],
            'hidden_clean':hidden['reference'],'action_delta_raw32':actions['treatment']['raw32']-actions['reference']['raw32'],
            'action_delta_adapter7':actions['treatment']['adapter7']-actions['reference']['adapter7'],
            'proof':proof}

def task_component(rows):
    return {r['task_id']:r['task_id'] for r in rows}

def basis_fit(X,kind,seed):
    # X is [N,1024] and all bases are rank 16.
    if kind=='delta_pca':
        _,_,vh=np.linalg.svd(X,full_matrices=False);return vh[:16].T
    if kind=='clean_pca':
        _,_,vh=np.linalg.svd(X,full_matrices=False);return vh[:16].T
    if kind=='coords':
        ids=np.argsort(np.sum(X*X,axis=0))[::-1][:16];B=np.zeros((1024,16));B[ids,np.arange(16)]=1;return B
    rng=np.random.default_rng(seed);Q,_=np.linalg.qr(rng.normal(size=(1024,16)));return Q[:,:16]

def standardize_fit(X):
    mu=X.mean(0);sd=X.std(0);sd=np.where(sd<1e-8,1,sd);return mu,sd

def ridge_fit(X,Y,lam):
    mu,sd=standardize_fit(X);Z=(X-mu)/sd
    A=Z.T@Z+lam*np.eye(Z.shape[1]);W=np.linalg.solve(A,Z.T@Y)
    return lambda V:((V-mu)/sd)@W

def aggregate_task(rows):
    by=defaultdict(list)
    for r in rows:by[r['task_id']].append(r['mse'])
    vals=np.array([np.mean(v) for v in by.values()],float)
    return {'task_count':len(vals),'case_count':len(rows),'task_equal_mse':float(vals.mean()),'case_equal_mse':float(np.mean([r['mse']for r in rows]))}

def evaluate(cases,family,action_key):
    train=[c for c in cases if c['row']['family']!=family];test=[c for c in cases if c['row']['family']==family]
    Ytrain=[];Ytest=[];dtrain=[];dtest=[];clean_train=[];clean_test=[];scalar_train=[];scalar_test=[];meta=[]
    for c in train:
        d=c['hidden_delta'][CALL];cl=c['hidden_clean'][CALL]
        Ytrain.append(c['action_delta_'+action_key]);dtrain.append(d);clean_train.append(cl);scalar_train.append(np.linalg.norm(d,axis=-1))
    for c in test:
        d=c['hidden_delta'][CALL];cl=c['hidden_clean'][CALL]
        Ytest.append(c['action_delta_'+action_key]);dtest.append(d);clean_test.append(cl);scalar_test.append(np.linalg.norm(d,axis=-1));meta.append(c['row'])
    Ytr=np.asarray(Ytrain);Yte=np.asarray(Ytest);dt=np.asarray(dtrain);de=np.asarray(dtest);ct=np.asarray(clean_train);ce=np.asarray(clean_test);st=np.asarray(scalar_train);se=np.asarray(scalar_test)
    # Each action target has 10 positions and dim outputs; fit one readout to flattened action sequence.
    ytr=Ytr.reshape(len(Ytr),-1);yte=Yte.reshape(len(Yte),-1)
    bases={'delta_pca':basis_fit(dt.reshape(-1,1024),'delta_pca',0),
           'clean_pca':basis_fit(ct.reshape(-1,1024),'clean_pca',1),
           'coords':basis_fit(dt.reshape(-1,1024),'coords',2),
           'random':basis_fit(dt.reshape(-1,1024),'random',3)}
    feats={}
    for name,B in bases.items():feats['rank16_'+name+'_train']=np.asarray([d@B for d in dt]).reshape(len(dt),-1);feats['rank16_'+name+'_test']=np.asarray([d@B for d in de]).reshape(len(de),-1)
    feats['magnitude_train']=st;feats['magnitude_test']=se
    # full response is a declared upper-bound, while rank16/magnitude are matched-capacity candidates.
    feats['full_train']=dt.reshape(len(dt),-1);feats['full_test']=de.reshape(len(de),-1)
    output=[]
    for name in ('rank16_delta_pca','rank16_clean_pca','rank16_coords','rank16_random','magnitude','full'):
        pred=ridge_fit(feats[name+'_train'],ytr,1e-2)(feats[name+'_test'])
        for i,row in enumerate(meta):
            err=float(np.mean((pred[i]-yte[i])**2));output.append({'family':family,'action':action_key,'model':name,'case_id':row['case_id'],'task_id':row['task_id'],'mse':err})
    summaries=[]
    for name in ('rank16_delta_pca','rank16_clean_pca','rank16_coords','rank16_random','magnitude','full'):
        summaries.append({'family':family,'action':action_key,'model':name,**aggregate_task([r for r in output if r['model']==name])})
    return summaries,output

def main():
    p=argparse.ArgumentParser();p.add_argument('--execute-cloud',action='store_true');p.add_argument('--root');p.add_argument('--deadline-epoch',type=float);a=p.parse_args()
    if not a.execute_cloud:print(json.dumps({'status':'dry_action_readout_candidate','GPU':0,'model_calls':0}));return 0
    root=pathlib.Path(a.root).resolve();need(root.is_relative_to(pathlib.Path('/tmp/Jinwen/adv-vla')),'private_action_root');meta=root/'package/metadata';req=json.loads((meta/'REQUEST-v1.json').read_bytes());need(req['planned_case_count']==536 and req['resources']=={'GPU':0,'CPU':32,'memory_GiB':128},'action_request')
    readset=json.loads((meta/'FINAL_READSET-v1.json').read_bytes());validation=R.validate(readset,(meta/'CASE-PLAN-v1.csv').read_bytes());rows=validation['eligible_rows'];writer=BOSWriter(OUTPUT,a.deadline_epoch-90);cases=[];fails=[]
    with ThreadPoolExecutor(max_workers=8) as pool:
        fs={pool.submit(load_pair,row,writer,a.deadline_epoch-180):row for row in rows}
        for f in as_completed(fs):
            row=fs[f]
            try:cases.append(f.result())
            except BaseException as e:fails.append({'case_id':row['case_id'],'error_type':type(e).__name__})
    need(len(cases)==536,'all_536_action_members')
    summaries=[];detail=[]
    for family in FAMILIES:
        for action in ('raw32','adapter7'):
            s,d=evaluate(cases,family,action);summaries+=s;detail+=d
    out=root/'summary';out.mkdir(exist_ok=False);import csv
    with (out/'ACTION-SUMMARY-v1.csv').open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(summaries[0]));w.writeheader();w.writerows(summaries)
    with gzip.open(out/'ACTION-CASE-v1.csv.gz','wt',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(detail[0]));w.writeheader();w.writerows(detail)
    result={'schema':'plus-action-readout-result-v1','run_id':RUN,'cases':len(cases),'technical_failures':fails,'actions':['raw32','adapter7'],'fit_call':0,'held_out_unit':'visual_family','models':['rank16_delta_pca','rank16_clean_pca','rank16_coords','rank16_random','magnitude','full'],'target':'paired treatment-minus-reference action sequence','GPU':0,'model_calls':0,'action_member_proofs':[c['proof'] for c in sorted(cases,key=lambda x:x['row']['case_id'])]}
    write_json(out/'RESULT-v1.json',result);refs={'ACTION-SUMMARY-v1.csv':writer.put(out/'ACTION-SUMMARY-v1.csv','summary/ACTION-SUMMARY-v1.csv'),'ACTION-CASE-v1.csv.gz':writer.put(out/'ACTION-CASE-v1.csv.gz','summary/ACTION-CASE-v1.csv.gz'),'RESULT-v1.json':writer.put(out/'RESULT-v1.json','summary/RESULT-v1.json')};delivery={'schema':'plus-action-readout-delivery-v1','run_id':RUN,'cases':len(cases),'GPU':0,'model_calls':0,'files':refs};write_json(out/'DELIVERY-v1.json',delivery);refs['DELIVERY-v1.json']=writer.put(out/'DELIVERY-v1.json','summary/DELIVERY-v1.json');print(json.dumps({'status':'published_action_readout','cases':len(cases),'failures':len(fails),'files':refs}));return 0
if __name__=='__main__':raise SystemExit(main())
