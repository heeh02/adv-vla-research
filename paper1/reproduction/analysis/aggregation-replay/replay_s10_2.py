#!/usr/bin/env python3
"""Portable replay of the original two-observation scalar estimands.

This uses archived scalar records and the exact saved task draw matrix. It never
loads a checkpoint, fetches data, executes a simulator or substitutes later labels.
"""
from pathlib import Path
import argparse,hashlib,json,sys,platform,math
import numpy as np
import pandas as pd
import scipy
import original_estimators as est

HERE=Path(__file__).resolve().parent

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def need(ok,why):
 if not ok:raise ValueError(why)
def write_json(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def compare(actual,expected,keys,fields):
 j=actual.merge(expected,on=keys,suffixes=('_actual','_expected'),validate='one_to_one');need(len(j)==len(actual)==len(expected),'expected_support_changed');maxerr=0.
 for col in fields:
  err=np.abs(j[col+'_actual'].to_numpy()-j[col+'_expected'].to_numpy());maxerr=max(maxerr,float(err.max()));need(np.isfinite(err).all()and np.max(err)<1e-12,'published_value_mismatch_'+col)
 return {'rows':len(j),'maximum_absolute_error':maxerr}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);args=ap.parse_args();out=args.out.resolve();need(not out.exists(),'output_must_be_new');out.mkdir(parents=True)
 manifest=json.loads((HERE/'INPUT-MANIFEST.json').read_text())
 for x in manifest['files']:
  p=HERE/x['path'];need(p.stat().st_size==x['bytes']and sha(p)==x['sha256'],'input_identity_'+x['path'])
 d=pd.read_csv(HERE/'data/episode-scalars.csv.gz',dtype={'pair_status':'string','treatment_result_sha256':'string','clean_result_sha256':'string','source_version':'string'})
 need(len(d)==2400 and d.episode_id.nunique()==2400,'full_planned_population');need(not d.duplicated(['task_id','root_id','noise_label','condition']).any(),'duplicate_execution_key')
 need(d.task_id.nunique()==40 and d.root_id.nunique()==80,'task_root_support')
 need(set(d.condition)==set(manifest['conditions']),'condition_support');need(set(d.noise_label)=={0,1,2},'noise_support')
 for _,g in d.groupby('clean_reference_id'):
  for f in ['task_id','root_id','noise_label','clean_result_sha256','clean_success']:
   need(g[f].fillna('MISSING').nunique()==1,'clean_reference_identity_conflict_'+f)
 known=d.pair_status.isin(['11','10','01','00']);k=d[known].copy();need(len(k)==2344,'complete_pair_support')
 for r in k.itertuples():
  need(r.pair_status==str(int(r.treatment_success))+str(int(r.clean_success)),'outcome_cell_mismatch')
  need(len(r.treatment_result_sha256)==64 and len(r.clean_result_sha256)==64,'missing_original_result_identity')
  need(r.treatment_state_sha256==r.clean_state_sha256 and r.treatment_noise_sha256==r.clean_noise_sha256,'initial_state_or_noise_pairing_conflict')
 for metric in ['hidden','main_TV','action']:
  fs=['slot1_'+metric,'slot2_'+metric];need(np.isfinite(k[fs].to_numpy()).all(),'missing_paired_feature')
  need(d.loc[~known,fs].isna().all().all(),'unknown_support_feature_unexpected')
  k['average_'+metric]=(k[fs[0]]+k[fs[1]])/2
 k['delta_sr']=k.treatment_success-k.clean_success
 cols=['average_hidden','average_main_TV','average_action','delta_sr'];cond=k.groupby('condition')[cols].mean().reset_index();cell=k.groupby(['task_id','condition'])[cols].mean().reset_index();need(len(cond)==10 and len(cell)==400,'aggregate_support')
 cond.to_csv(out/'condition-means.csv',index=False);cell.to_csv(out/'task-condition-means.csv',index=False)
 points=[]
 for level,t in [('condition',cond),('task_condition',cell),('episode',k)]:
  groups={'none':[]}
  if level=='episode':groups['task_and_condition']=[pd.factorize(t.task_id)[0],pd.factorize(t.condition)[0]]
  for control,codes in groups.items():
   for pair,rho in est.correlations(t[cols].to_numpy(),codes).items():points.append(dict(slot='average',level=level,control=control,pair=pair,rho=rho,n=len(t)))
 points=pd.DataFrame(points);points.to_csv(out/'correlations.csv',index=False)
 cp=compare(points,pd.read_csv(HERE/'data/expected-correlations.csv'),['slot','level','control','pair','n'],['rho'])
 auc=[];z=k[k.pair_status.isin(['01','11'])]
 for metric in ['hidden','main_TV','action']:
  pieces=[]
  for name,g in z.groupby('condition'):
   v,n=est.auc((g.pair_status=='01').astype(int),g['average_'+metric]);pieces.append((v,n));auc.append(dict(metric=metric,condition=name,auc=v,n=len(g),harm=int((g.pair_status=='01').sum()),comparison_pairs=n))
  pieces=np.asarray(pieces);auc.append(dict(metric=metric,condition='WITHIN_CONDITION_PAIR_WEIGHTED',auc=float(np.average(pieces[:,0],weights=pieces[:,1])),n=len(z),harm=int((z.pair_status=='01').sum()),comparison_pairs=int(pieces[:,1].sum())))
 auc=pd.DataFrame(auc);auc.to_csv(out/'aucs.csv',index=False);ac=compare(auc,pd.read_csv(HERE/'data/expected-aucs.csv'),['metric','condition','n','harm','comparison_pairs'],['auc'])
 tasks=sorted(k.task_id.unique());draws=pd.read_csv(HERE/'data/task-draws.csv.gz').to_numpy(dtype=int);need(draws.shape==(1000,40),'draw_shape');need(np.array_equal(draws,np.random.default_rng(20261008).integers(0,40,(1000,40))),'original_PCG64_draw_matrix_mismatch')
 bytask={t:np.flatnonzero(k.task_id.to_numpy()==t)for t in tasks};bycell={t:np.flatnonzero(cell.task_id.to_numpy()==t)for t in tasks};cc=pd.Categorical(k.condition,categories=manifest['conditions']).codes;tc=pd.Categorical(k.task_id,categories=tasks).codes;arr=k[cols].to_numpy();ca=cell[cols].to_numpy();boot=[]
 for b,chosen in enumerate(draws):
  ix=np.concatenate([bytask[tasks[i]]for i in chosen]);ci=np.concatenate([bycell[tasks[i]]for i in chosen]);x=arr[ix];c=cc[ix];t=tc[ix];n=np.bincount(c,minlength=10);means=np.stack([np.bincount(c,weights=x[:,p],minlength=10)/n for p in range(4)],axis=1)
  for level,control,xx,codes in [('condition','none',means,[]),('task_condition','none',ca[ci],[]),('episode','none',x,[]),('episode','task_and_condition',x,[t,c])]:
   for pair,rho in est.correlations(xx,codes).items():boot.append(dict(replicate=b,slot='average',level=level,control=control,pair=pair,rho=rho))
 boot=pd.DataFrame(boot);boot.to_csv(out/'bootstrap-replicates.csv.gz',index=False,compression={'method':'gzip','mtime':0})
 cis=boot.groupby(['slot','level','control','pair']).rho.agg(valid_reps='count',lo=lambda x:x.quantile(.025),hi=lambda x:x.quantile(.975)).reset_index();cis.to_csv(out/'intervals.csv',index=False);bc=compare(cis,pd.read_csv(HERE/'data/expected-intervals.csv'),['slot','level','control','pair','valid_reps'],['lo','hi'])
 report={'status':'VERIFIED_ARCHIVED_SCALAR_REPLAY','scope':'original complete2344pairedscalarpopulation with56missingplannedpairs; no raw tensor or policy rerun','planned':2400,'paired':2344,'missing':56,'tasks':40,'conditions':10,'task_condition_cells':400,'shared_clean_episodes':int(k.clean_reference_id.nunique()),'outcome_cells_treatment_clean':{str(a):int(b)for a,b in k.pair_status.value_counts().items()},'clean_success_pairs':len(z),'correlation_check':cp,'AUC_check':ac,'interval_check':bc,'bootstrap_draws':len(draws),'bootstrap_statistic_rows':len(boot),'RNG':'originalPCG64seed20261008 savedmatrix checked','new_labels_substituted':False,'identity_check':'same saved originalresult/source IDs; archivedscalarjoin isnot freshraw-resultparsing','python':sys.version,'numpy':np.__version__,'pandas':pd.__version__,'scipy':scipy.__version__,'code_sha256':sha(Path(__file__)),'estimator_sha256':sha(HERE/'original_estimators.py'),'input_manifest_sha256':sha(HERE/'INPUT-MANIFEST.json')}
 write_json(out/'RESULT.json',report);print(json.dumps(report,indent=2))
if __name__=='__main__':main()
