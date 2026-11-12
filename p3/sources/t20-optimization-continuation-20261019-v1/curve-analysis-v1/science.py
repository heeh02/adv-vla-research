"""20k->30k existing-data curves; real statistics/rendering only on Linux CPU.

Infra restores each exact input once with If-Match+size and whole SHA. This
module has no network/model/training/simulator operations. Tiny synthetic
fixtures may test reducers on Mac; real arrays and figures may not.
"""
import csv
import gzip
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
import platform

LABELS=('floor-s0','floor-s1','reheat-s0','reheat-s1')
CONDITIONS=('C0_Hbase','C15_Hbase','C0_Hoff','C15_Hoff')
PROBE_METRICS=tuple(f'H{h}_D{d}' for h in (5,10) for d in (7,25,32))
COLORS={'floor':'#276DA5','reheat':'#D27530'}
WINDOW=201

def need(ok,code):
    if not ok:raise ValueError(code)

def finite(value):
    need(type(value) in (int,float) and math.isfinite(value),'finite_scalar')
    return float(value)

def load(path):
    with Path(path).open() as stream:return json.load(stream,parse_constant=lambda _:(_ for _ in ()).throw(ValueError('nonfinite_JSON')))

def dump(path,value):
    Path(path).write_text(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n')

def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()

def table(path,rows,fields=None):
    if not rows:return
    with Path(path).open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields or list(rows[0]),extrasaction='raise')
        writer.writeheader();writer.writerows(rows)

def read_updates(item):
    rows=[];seen=set();label=item['run_label'];branch=item['branch'];seed=item['seed']
    with Path(item['local_path']).open() as stream:
        for line_no,line in enumerate(stream,1):
            need(line.endswith('\n'),'updates_incomplete_line')
            value=json.loads(line);step=value['effective_update'];stock=value['actual_stock_step']
            need(type(step) is int and type(stock) is int and step==stock and 20001<=step<=30000,'actual_step_identity')
            need(step not in seen and (not rows or step==rows[-1]['effective_update']+1),'updates_duplicate_or_gap')
            seen.add(step);info=value['stock_info'];ids=value['row_ids']
            need(len(ids)==32,'native_batch32')
            row=dict(run_label=label,branch=branch,seed=seed,effective_update=step,loss_parameter_step=step-1,
                     epoch=value['epoch'],loss=finite(info['loss']),LR_before_update=finite(value['LR_before_update']),
                     grad_norm=finite(info['grad_norm']),blocked_update_seconds=finite(value['blocked_update_seconds']),
                     input_row_ids_sha256=hashlib.sha256(canonical(ids)).hexdigest(),source_line=line_no)
            need(min(row['loss'],row['LR_before_update'],row['grad_norm'])>=0,'nonnegative_training_metrics')
            rows.append(row)
    need(rows and rows[0]['effective_update']==20001,'continuation_starts20001')
    return rows

def rolling(values,window=WINDOW):
    import numpy as np
    values=np.asarray(values,dtype=float);n=len(values)
    med=np.full(n,np.nan);low=np.full(n,np.nan);high=np.full(n,np.nan)
    if n>=window:
        views=np.lib.stride_tricks.sliding_window_view(values,window)
        med[window-1:]=np.median(views,axis=1)
        low[window-1:],high[window-1:]=np.quantile(views,[.25,.75],axis=1)
    return med,low,high

def stats(values):
    import numpy as np
    a=np.asarray(values,dtype=float)
    return dict(n=len(a),mean=float(a.mean()),median=float(np.median(a)),q25=float(np.quantile(a,.25)),
                q75=float(np.quantile(a,.75)),min=float(a.min()),max=float(a.max()))

def internal_record(item):
    value=load(item['local_path']);step=value['effective_update'];aggregate=value['aggregate']
    need(value['schema']=='cul-EMA-fixed-internal-loss-v1' and value['parameter_view']=='EMA','original_internal_schema')
    need(step in range(20000,30001,1000) and value['missing_states']==[],'internal_step_support')
    need(aggregate['states']==800 and aggregate['source_demos']==400 and aggregate['tasks']==40,'original_internal_800_400_40')
    need(len(value['anchor_scores'])==800 and len(aggregate['demo_records'])==400 and len(aggregate['task_records'])==40,'internal_count_consistency')
    role='restoration_check' if item['name'].startswith('RESTORE_CHECK') else 'original20k' if item.get('role')=='parent_internal20k' else 'new_observation'
    rows=[]
    for condition in CONDITIONS:
        tasks=[finite(t['mean_loss'][condition]) for t in aggregate['task_records'].values()]
        mean=sum(tasks)/40
        need(math.isclose(mean,float(aggregate['macro_task_mean_loss'][condition]),rel_tol=1e-10,abs_tol=1e-12),'internal_macro_recompute')
        rows.append(dict(run_label=item['run_label'],branch=item.get('branch'),seed=item['seed'],step=step,
            condition=condition,parameter_view='EMA',metric='H10_D32',mean=mean,
            states=800,demos=400,tasks=40,measurement_role=role,
            task_values={k:finite(v['mean_loss'][condition]) for k,v in aggregate['task_records'].items()}))
    return value,rows

def probe_record(item):
    value=load(item['local_path'])
    need(value['schema']=='t20-fixed-train-probe-scores-v1' and value['is_success_rate'] is False,'fixed_probe_schema')
    step=value['effective_update'];need(step in (20000,25000,30000),'fixed_probe_steps')
    need(value['seed']==item['seed'] and value['branch']==item['branch'],'probe_run_seed_branch')
    records=value['records'];need(value['planned_scores']==1600 and value['completed_scores']==len(records) and not value['missing'],'complete_fixed_probe_1600')
    need(len(records)==1600,'fixed800x2views')
    keys=set()
    for row in records:
        key=(row['physical_anchor_id'],row['parameter_view'])
        need(key not in keys and row['parameter_view'] in ('online','EMA') and row['role'] in ('C_old','T_added'),'probe_unique_key')
        keys.add(key)
        need(row['effective_update']==step and row['condition']=='C0_Hbase' and row['split']=='train' and row['draw']==0,'fixed_probe_identity')
        for metric in PROBE_METRICS:need(finite(row[metric])>=0,'probe_nonnegative')
        for horizon in (5,10):
            expected=(7*row[f'H{horizon}_D7']+25*row[f'H{horizon}_D25'])/32
            need(math.isclose(expected,row[f'H{horizon}_D32'],rel_tol=2e-5,abs_tol=1e-8),'D32_weighted_identity')
    return value

def probe_summaries(probes):
    rows=[];task_rows=[];changes=[];index={}
    for (label,step),value in sorted(probes.items()):
        branch,seed=label.rsplit('-s',1);seed=int(seed)
        grouped=defaultdict(list)
        for record in value['records']:
            grouped[(record['role'],record['parameter_view'])].append(record)
            index[(label,step,record['physical_anchor_id'],record['parameter_view'])]=record
        for (role,view),records in sorted(grouped.items()):
            need(len(records)==400 and len({r['task_id'] for r in records})==40,'probe_role400_task40')
            for metric in PROBE_METRICS:
                bytask=defaultdict(list)
                for r in records:bytask[r['task_id']].append(finite(r[metric]))
                taskmeans={t:sum(v)/len(v) for t,v in bytask.items()}
                rows.append(dict(run_label=label,branch=branch,seed=seed,step=step,role=role,view=view,metric=metric,
                    states=len(records),demos=len({r['source_demo_id'] for r in records}),tasks=len(bytask),
                    task_macro_mean=sum(taskmeans.values())/len(taskmeans),state_median=stats([r[metric] for r in records])['median']))
                for task,mean in taskmeans.items():
                    task_rows.append(dict(run_label=label,step=step,role=role,view=view,metric=metric,task_id=task,n_states=len(bytask[task]),mean=mean))
                if step==20000:continue
                differences=[];taskdiff=defaultdict(list)
                for r in records:
                    baseline=index.get((label,20000,r['physical_anchor_id'],view))
                    need(baseline is not None,'fixed_probe_parent_key_present')
                    for field in ('task_id','source_demo_id','source_j','role','key_seed','draw','flow_time','condition'):
                        need(baseline[field]==r[field],'probe_paired_identity:'+field)
                    delta=r[metric]-baseline[metric];differences.append(delta);taskdiff[r['task_id']].append(delta)
                taskdelta=[sum(v)/len(v) for v in taskdiff.values()]
                changes.append(dict(run_label=label,step=step,role=role,view=view,metric=metric,
                    paired_states=len(differences),paired_tasks=len(taskdelta),task_macro_delta=sum(taskdelta)/len(taskdelta),
                    tasks_improved=sum(v<0 for v in taskdelta),tasks_worsened=sum(v>0 for v in taskdelta),tasks_equal=sum(v==0 for v in taskdelta),
                    states_improved=sum(v<0 for v in differences),states_worsened=sum(v>0 for v in differences)))
    return rows,task_rows,changes

def branch_contrasts(probes):
    result=[]
    for seed in (0,1):
        for step in (20000,25000,30000):
            a=probes.get((f'floor-s{seed}',step));b=probes.get((f'reheat-s{seed}',step))
            if a is None or b is None:continue
            left={(r['physical_anchor_id'],r['parameter_view']):r for r in a['records']}
            right={(r['physical_anchor_id'],r['parameter_view']):r for r in b['records']}
            need(left.keys()==right.keys(),'branch_matched_probe_population')
            for key in left:
                for field in ('task_id','source_demo_id','source_j','role','key_seed','draw','flow_time'):
                    need(left[key][field]==right[key][field],'branch_pair_field:'+field)
            for role in ('C_old','T_added'):
                for view in ('online','EMA'):
                    keys=[k for k,r in left.items() if r['role']==role and k[1]==view]
                    for metric in PROBE_METRICS:
                        task=defaultdict(list)
                        for key in keys:task[left[key]['task_id']].append(right[key][metric]-left[key][metric])
                        delta=[sum(v)/len(v) for v in task.values()]
                        result.append(dict(seed=seed,step=step,role=role,view=view,metric=metric,
                            contrast='reheat_minus_floor',paired_states=len(keys),paired_tasks=len(delta),
                            task_macro_delta=sum(delta)/len(delta),tasks_reheat_lower=sum(v<0 for v in delta),
                            tasks_reheat_higher=sum(v>0 for v in delta),tasks_equal=sum(v==0 for v in delta)))
    return result

def figure_setup():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
                         'axes.spines.right':False,'figure.facecolor':'white','savefig.facecolor':'white'})
    return plt

def save(fig,path):
    fig.savefig(str(path)+'.png',dpi=170,bbox_inches='tight')
    fig.savefig(str(path)+'.svg',bbox_inches='tight')

def render_training(updates,out):
    import numpy as np
    plt=figure_setup();fig,axes=plt.subplots(3,2,figsize=(13,9),sharex=True)
    for seed in (0,1):
        for branch in ('floor','reheat'):
            label=f'{branch}-s{seed}';rows=updates.get(label,[])
            if not rows:
                axes[0,seed].text(.02,.95 if branch=='floor' else .88,label+': UPDATES not archived',transform=axes[0,seed].transAxes,va='top',color=COLORS[branch]);continue
            x=np.array([r['effective_update'] for r in rows])
            for i,metric in enumerate(('loss','LR_before_update','grad_norm')):
                y=np.array([r[metric] for r in rows]);axis=axes[i,seed]
                if metric=='LR_before_update':axis.plot(x,y,color=COLORS[branch],label=branch,lw=1.8)
                else:
                    axis.scatter(x,y,s=1,alpha=.10,color=COLORS[branch],rasterized=True)
                    median,q25,q75=rolling(y)
                    axis.fill_between(x,q25,q75,color=COLORS[branch],alpha=.14)
                    axis.plot(x,median,color=COLORS[branch],lw=1.8,label=f'{branch}, n={len(rows):,}')
            axes[0,seed].set_title(f'Original training seed {seed}')
        for i,name in enumerate(('Online batch flow MSE (H10, D32)','Learning rate before update','Gradient L2 norm before clipping')):
            axes[i,seed].set_ylabel(name);axes[i,seed].grid(alpha=.18);axes[i,seed].legend(fontsize=8)
            axes[i,seed].set_xlim(20000,30000)
        axes[2,seed].set_xlabel('Global effective optimizer update')
    fig.suptitle('T20 → T30 optimization: every batch and fixed 201-update trailing median/IQR',fontsize=14)
    fig.text(.5,.015,'B32 = 16 physical starts × 2 clean rows; H10 × D32 normalized flow velocity MSE. Loss measured before labeled update.\nIQR is batch dispersion, not confidence; online batches change across epochs and are not a generalization metric.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.06,1,.96));save(fig,out/'F01-online-training');plt.close(fig)

def render_internal(rows,out):
    import numpy as np
    plt=figure_setup();fig,axes=plt.subplots(2,2,figsize=(12,8),sharex=True)
    for axis,condition in zip(axes.flat,CONDITIONS):
        for label in LABELS:
            selected=sorted([r for r in rows if r['run_label']==label and r['condition']==condition and r['measurement_role']!='original20k'],key=lambda r:r['step'])
            branch,seed=label.rsplit('-s',1)
            bystep={r['step']:r for r in selected};steps=list(range(20000,30001,1000))
            if selected:axis.plot(steps,[bystep[s]['mean'] if s in bystep else np.nan for s in steps],marker='o',ms=3,
                color=COLORS[branch],ls='-' if seed=='0' else '--',label=label)
        axis.set_title(condition);axis.set_ylabel('EMA H10 × D32 flow MSE (task macro mean)');axis.grid(alpha=.18);axis.legend(fontsize=8)
        axis.set_xlim(20000,30000)
    for axis in axes[-1]:axis.set_xlabel('Global optimizer step; 20k = restoration check')
    fig.suptitle('Original fixed internal validation: 800 states / 400 held-out demos / 40 tasks',fontsize=14)
    fig.text(.5,.015,'Four condition panels share fixed internal-monitor keys. No new model calls. Missing checkpoints remain gaps; this is loss, not success rate.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.05,1,.96));save(fig,out/'F02-internal-EMA-validation');plt.close(fig)

def render_probe(rows,out):
    import numpy as np
    plt=figure_setup()
    for view in ('online','EMA'):
        for role in ('C_old','T_added'):
            fig,axes=plt.subplots(3,2,figsize=(12,9),sharex=True)
            for col,seed in enumerate((0,1)):
                for rowidx,dims in enumerate((7,25,32)):
                    axis=axes[rowidx,col]
                    for branch in ('floor','reheat'):
                        for horizon in (5,10):
                            metric=f'H{horizon}_D{dims}';selected=sorted([r for r in rows if r['run_label']==f'{branch}-s{seed}' and r['view']==view and r['role']==role and r['metric']==metric],key=lambda r:r['step'])
                            bystep={r['step']:r for r in selected};steps=(20000,25000,30000)
                            if selected:axis.plot(steps,[bystep[s]['task_macro_mean'] if s in bystep else np.nan for s in steps],marker='o',color=COLORS[branch],
                                ls='-' if horizon==10 else '--',label=f'{branch}, H{horizon}')
                    axis.set_ylabel(f'D{dims} '+('actual' if dims==7 else 'padding' if dims==25 else 'all')+' flow MSE');axis.grid(alpha=.18);axis.legend(fontsize=8)
                    axis.set_xticks((20000,25000,30000));axis.set_xlim(19500,30500)
                axes[0,col].set_title(f'Seed {seed}');axes[2,col].set_xlabel('Checkpoint step')
            fig.suptitle(f'Fixed train probe · {view} parameters · {role} support',fontsize=14)
            fig.text(.5,.012,'400 fixed states / 200 training demos / 40 tasks per role; fixed key per state shared across branches, views and checkpoints.\nH5 is first five expert-flow positions, not generated/issued actions. Equal task means; no held-out generalization or success-rate claim.',ha='center',fontsize=9)
            fig.tight_layout(rect=(0,.07,1,.96));save(fig,out/f'F03-fixed-probe-{view}-{role}');plt.close(fig)

def analyze(files,request,out):
    need(platform.system()=='Linux','real_statistics_and_rendering_Baidu_CPU_only')
    out=Path(out);out.mkdir(exist_ok=True)
    updates={};internal=[];probes={};aliases=[];raw_internal={};integrity=[];missing=[];cached_sources=[];controls={}
    caches=[item for item in files if item.get('role')=='analysis_cache']
    need(len(caches)<=1,'one_prior_analysis_cache')
    if caches:
        with gzip.open(caches[0]['local_path'],'rt') as stream:cache=json.load(stream)
        need(cache['schema']=='T30-curves-reusable-numerics-v1','prior_cache_schema')
        updates=cache['updates'];internal=cache['internal'];integrity=cache['integrity'];cached_sources=cache['sources'];controls=cache.get('controls',{})
        probes={(row['run_label'],row['step']):row['value'] for row in cache['probes']}
        raw_internal={tuple(row['key']):row['value'] for row in cache['raw_internal']}
    for item in files:
        if item.get('role')=='analysis_cache':continue
        name=item.get('name',Path(item['local_path']).name);label=item['run_label']
        item={**item,'name':name}
        if name=='UPDATES-v1.jsonl':
            need(label not in updates,'one_update_log_per_run');updates[label]=read_updates(item)
        elif name=='ALLOCATION_RESULT-v1.json':
            value=load(item['local_path'])
            need(value['seed']==item['seed'] and value['branch']==item['branch'] and value['arm']=='T','control_branch_seed')
            controls[label]={k:value.get(k) for k in ('run_id','allocation_id','status','effective_updates','actual_stock_step','verified_checkpoint_steps_this_allocation')}
        elif name.startswith(('RESTORE_CHECK_INTERNAL_EMA','INTERNAL_EMA')):
            value,rows=internal_record(item);internal.extend(rows)
            raw_internal[(label,value['effective_update'],rows[0]['measurement_role'])]=value
        elif name.startswith('FIXED_TRAIN_PROBE-step'):
            if name.endswith('-ALIAS.json'):aliases.append((item,load(item['local_path'])))
            else:
                value=probe_record(item);key=(label,value['effective_update']);need(key not in probes,'unique_probe_checkpoint');probes[key]=value
    # A completed 20k alias is an existing measurement, not a second sample.
    by_uri={r['uri']:r for r in files if 'uri' in r}
    for item,alias in aliases:
        ref=alias['source'];source=by_uri.get(ref.get('uri'))
        need(source is not None,'fixed20k_alias_source_must_be_explicit_input')
        value=load(source['local_path']);need(value['effective_update']==20000 and value['seed']==item['seed'] and value['parent_commit']==alias['parent_commit'],'exact20k_alias_state')
        need(value['base_selection_manifest_sha256']==alias['base_selection_manifest_sha256'],'exact20k_alias_population')
        probes[(item['run_label'],20000)]={**value,'branch':item['branch'],'measurement_role':'shared20k_alias_no_new_measurement'}
        integrity.append(dict(kind='fixed20k_alias',run_label=item['run_label'],source_uri=ref.get('uri'),new_observations=0))
    training_summaries=[];window_rows=[];paired=[]
    for label in LABELS:
        rows=updates.get(label)
        if rows:
            for lo,hi in ((20001,21000),(24001,25000),(29001,30000)):
                selected=[r for r in rows if lo<=r['effective_update']<=hi]
                for metric in ('loss','grad_norm','LR_before_update'):
                    if selected:window_rows.append(dict(run_label=label,first_update=lo,last_update=hi,metric=metric,**stats([r[metric] for r in selected])))
            training_summaries.append(dict(run_label=label,updates=len(rows),first=rows[0]['effective_update'],last=rows[-1]['effective_update'],
                complete10000=len(rows)==10000 and rows[-1]['effective_update']==30000,loss=stats([r['loss'] for r in rows])))
        else:missing.append(dict(run_label=label,role='online_updates',reason='not_in_frozen_input_inventory'))
        for step in range(20000,30001,1000):
            if not any(r['run_label']==label and r['step']==step for r in internal):missing.append(dict(run_label=label,role='internal_EMA',step=step,reason='not_in_frozen_input_inventory'))
        for step in (20000,25000,30000):
            if (label,step) not in probes:missing.append(dict(run_label=label,role='fixed_train_probe',step=step,reason='not_in_frozen_input_inventory'))
    for seed in (0,1):
        a=updates.get(f'floor-s{seed}',[]);b=updates.get(f'reheat-s{seed}',[])
        if a and b:
            amap={r['effective_update']:r for r in a};bmap={r['effective_update']:r for r in b};common=sorted(amap.keys()&bmap.keys())
            need(all(amap[s]['input_row_ids_sha256']==bmap[s]['input_row_ids_sha256'] for s in common),'matched_seed_input_order')
            paired.append(dict(seed=seed,matched_updates=len(common),input_row_order_equal=True))
    # Restoration monitor identity is checked against the seed's original20k if supplied.
    for seed in (0,1):
        parent=raw_internal.get((f'T20-s{seed}',20000,'original20k'))
        restores=[(label,raw_internal.get((label,20000,'restoration_check'))) for label in (f'floor-s{seed}',f'reheat-s{seed}')]
        for label,value in restores:
            if parent and value:
                equal=value['anchor_scores']==parent['anchor_scores'];need(equal,'restoration20k_original_anchor_scores_changed')
                integrity.append(dict(kind='internal20k_restore_vs_original',run_label=label,anchor_scores_exact=True,new_observations=0))
        if all(v is not None for _,v in restores):need(restores[0][1]['anchor_scores']==restores[1][1]['anchor_scores'],'same_seed20k_restore_monitors')
    probe_rows,probe_tasks,probe_changes=probe_summaries(probes)
    contrast_rows=branch_contrasts(probes)
    allupdates=[row for label in LABELS for row in updates.get(label,[])]
    if allupdates:
        with gzip.open(out/'TRAINING-COMPACT.csv.gz','wt',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(allupdates[0]));writer.writeheader();writer.writerows(allupdates)
    table(out/'TRAINING-WINDOWS.csv',window_rows)
    table(out/'INTERNAL-CURVES.csv',[{k:v for k,v in r.items() if k!='task_values'} for r in internal])
    table(out/'PROBE-CURVES.csv',probe_rows);table(out/'PROBE-TASKS.csv',probe_tasks);table(out/'PROBE-CHANGES.csv',probe_changes)
    table(out/'PROBE-BRANCH-CONTRASTS.csv',contrast_rows)
    render_training(updates,out);render_internal(internal,out)
    if probe_rows:render_probe(probe_rows,out)
    source=[{k:r[k] for k in ('uri','bytes','sha256','etag','source_binding','role','run_label') if k in r} for r in files]
    allsources={row['uri']:row for row in cached_sources+source if 'uri' in row}
    with gzip.open(out/'CACHE-v1.json.gz','wt') as stream:
        json.dump(dict(schema='T30-curves-reusable-numerics-v1',updates=updates,internal=internal,integrity=integrity,controls=controls,
            probes=[dict(run_label=label,step=step,value=value) for (label,step),value in probes.items()],
            raw_internal=[dict(key=list(key),value=value) for key,value in raw_internal.items()],sources=list(allsources.values())),
            stream,separators=(',',':'),allow_nan=False)
    summary=dict(schema='T30-existing-training-curves-summary-v1',status='complete_for_frozen_available_inputs',
        online_training=training_summaries,paired_input_order=paired,identity_checks=integrity,
        internal_curve_rows=len(internal),fixed_probe_checkpoints=len(probes),fixed_probe_curve_rows=len(probe_rows),
        missing=missing,allocation_results=controls,
        all_four_training_archives_complete=len(updates)==len(controls)==4 and all(len(v)==10000 for v in updates.values()) and all(v['status']=='complete' and v['effective_updates']==30000 for v in controls.values()),
        fixed201_rolling_window=True,source_inputs=len(source),prior_numeric_cache_reused=bool(caches),raw_tables_copied_to_Mac=False,
        model_calls=0,training_updates=0,success_rate_measured=False,
        interpretation='Online moving-batch optimization, fixed training support fit, and independent-demo internal EMA loss are different populations; no pooled train-validation gap.',
        limitations=['Two training seeds remain separate; no seed-level confidence interval.',
                      'Fixed probe has one fixed diagnostic draw per state, not an estimate over random flow draws.',
                      'H5/H10 are expert-flow residual positions; no generated or issued action error is measured.',
                      'Lower loss does not establish task success or robust behavior.'])
    dump(out/'SUMMARY-v1.json',summary);dump(out/'SOURCE-LEDGER-v1.json',list(allsources.values()))
    return {'summary':'SUMMARY-v1.json','figures':len(list(out.glob('*.png'))),'missing_records':len(missing)}
