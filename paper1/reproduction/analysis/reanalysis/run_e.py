"""Frozen E1/E2, existing local case-level derived data only.

Main requires a root technical-approval receipt bound to the frozen files.
"""
from __future__ import annotations
import argparse, csv, gzip, hashlib, importlib.util, json, math, platform, random, sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

HERE=Path(__file__).resolve().parent
STUDY=HERE.parents[1]
BASE=HERE.parent
OUT=STUDY/'results/E'

def input_path(relative_path):
    path=BASE/relative_path
    if not path.is_file():
        raise FileNotFoundError(
            'Missing packaged reanalysis dependency: '+str(relative_path)+
            '. Expected at '+str(path)+
            '; keep reanalysis/, figure-source/ and statistics/ in the same package.')
    return path

def need(condition, reason):
    if not condition: raise ValueError(reason)

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def now(): return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()
def read_csv(path):
    with gzip.open(path,'rt') as f: return list(csv.DictReader(f))
def write_json(path,value): path.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def write_csv(path,rows,fields=None):
    fields=fields or list(rows[0])
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)

def finite(value):
    try: z=float(value)
    except (TypeError,ValueError): return None
    return z if math.isfinite(z) else None

def weighted_midcdf(values,weights):
    need(len(values)==len(weights) and len(values)>0,'nonempty_equal_lengths')
    need(all(math.isfinite(v) for v in values),'rank_nonfinite')
    need(all(math.isfinite(w) and w>=0 for w in weights),'invalid_weights')
    total=math.fsum(weights);need(total>0,'zero_total_weight')
    order=sorted((i for i,w in enumerate(weights) if w>0),key=lambda i:values[i])
    ranks=[None]*len(values);before=0.0;lo=0
    while lo<len(order):
        hi=lo+1
        while hi<len(order) and values[order[hi]]==values[order[lo]]: hi+=1
        tied=math.fsum(weights[order[j]] for j in range(lo,hi))
        rank=(before+.5*tied)/total
        for j in range(lo,hi): ranks[order[j]]=rank
        before+=tied;lo=hi
    return ranks

def weighted_rank_rho(x,y,weights):
    ux=weighted_midcdf(x,weights);uy=weighted_midcdf(y,weights)
    indices=[i for i,w in enumerate(weights) if w>0]
    if len({x[i] for i in indices})<2 or len({y[i] for i in indices})<2:
        return None,'zero_global_rank_variance'
    total=math.fsum(weights)
    wx=[weights[i]/total for i in indices]
    rx=[ux[i] for i in indices];ry=[uy[i] for i in indices]
    mx=math.fsum(w*v for w,v in zip(wx,rx));my=math.fsum(w*v for w,v in zip(wx,ry))
    vx=math.fsum(w*(v-mx)**2 for w,v in zip(wx,rx));vy=math.fsum(w*(v-my)**2 for w,v in zip(wx,ry))
    if vx==0 or vy==0: return None,'zero_global_rank_variance'
    rho=math.fsum(w*(a-mx)*(b-my) for w,a,b in zip(wx,rx,ry))/math.sqrt(vx*vy)
    need(-1-1e-12<=rho<=1+1e-12,'rho_range')
    return max(-1.0,min(1.0,rho)),None

def prepare_motion(rows):
    indexed={}
    for row in rows:
        need(row['noise'] in {'0','1'},'unexpected_noise')
        need(row['step'] in {'5','60'},'unexpected_step')
        key=(row['model'],row['variant_id'],row['noise'],row['step'])
        need(key not in indexed,'duplicate_motion_key');indexed[key]=row
        for field in ('q0_norm','eef_mm'):
            value=finite(row[field]);need(value is None or value>=0,'negative_'+field)
    pairs=defaultdict(dict)
    for key,row in indexed.items(): pairs[key[:3]][key[3]]=row
    variant_map={};ledger=[];qualified=[]
    for key,pair in sorted(pairs.items()):
        reasons=[]
        present=list(pair.values())
        base=present[0]
        for row in present:
            need(row['task_id']==base['task_id'] and row['family']==base['family'] and row['Y']==base['Y'],'cross_clock_identity_conflict')
            need(row['q0_norm']==base['q0_norm'],'cross_clock_q0_conflict')
        vi=(key[0],key[1]);identity=(base['task_id'],base['family'])
        need(vi not in variant_map or variant_map[vi]==identity,'variant_task_family_conflict')
        variant_map[vi]=identity
        for step in ('5','60'):
            if step not in pair: reasons.append('step'+step+'_missing');continue
            row=pair[step]
            if row['included']!='True': reasons.append('step'+step+'_source_not_included')
            if row['q0_status']!='qualified': reasons.append('step'+step+'_q0_unqualified')
            if row['motion_status']!='measured': reasons.append('step'+step+'_eef_unmeasured')
            for field in ('q0_norm','eef_mm'):
                if finite(row[field]) is None: reasons.append('step'+step+'_'+field+'_nonfinite_or_missing')
        item={'model':key[0],'variant_id':key[1],'noise':key[2],'task_id':base['task_id'],
              'family':base['family'],'Y':base['Y'],'included_both_clocks':not reasons,
              'exclusion_reasons':'|'.join(reasons),'q0_norm':finite(base['q0_norm']),
              'eef_mm_step5':finite(pair.get('5',{}).get('eef_mm')),
              'eef_mm_step60':finite(pair.get('60',{}).get('eef_mm')),
              'q0_unit':'norm_command_coordinate','eef_unit':'mm'}
        ledger.append(item)
        if not reasons: qualified.append(item)
    return ledger,qualified

def nested_weights(rows, task_sample=None):
    groups=defaultdict(lambda:defaultdict(list))
    for i,row in enumerate(rows):groups[row['task_id']][row['variant_id']].append(i)
    tasks=sorted(groups)
    sample=tasks if task_sample is None else task_sample
    need(sample and set(sample)<=set(tasks),'invalid_task_sample')
    multiplicity=Counter(sample);weights=[0.0]*len(rows)
    for task,variants in groups.items():
        for indices in variants.values():
            for i in indices:weights[i]=multiplicity[task]/(len(sample)*len(variants)*len(indices))
    need(math.isclose(math.fsum(weights),1,abs_tol=1e-12),'weights_sum')
    return weights

def percentile(values,p):
    need(values and 0<=p<=1,'percentile_input')
    ordered=sorted(values);at=(len(ordered)-1)*p;lo=math.floor(at);hi=math.ceil(at)
    return ordered[lo]+(at-lo)*(ordered[hi]-ordered[lo])

def bootstrap_tasks(tasks,draws,seed):
    rng=random.Random(seed)
    return [[tasks[rng.randrange(len(tasks))] for _ in tasks] for _ in range(draws)]

def evaluate_motion(rows,weights):
    x=[r['q0_norm'] for r in rows]
    a,ar=weighted_rank_rho(x,[r['eef_mm_step5'] for r in rows],weights)
    b,br=weighted_rank_rho(x,[r['eef_mm_step60'] for r in rows],weights)
    return a,b,a-b if a is not None and b is not None else None,ar or br

def run_e1(cfg):
    path=input_path(cfg['inputs'][0]['path']);raw=read_csv(path)
    ledger,qualified=prepare_motion(raw)
    write_csv(OUT/'E1-SUPPORT.csv',ledger)
    results={};draw_rows=[];weight_rows=[];matrix={}
    bc=cfg['E1']['bootstrap']
    for model in cfg['E1']['models']:
        rows=[r for r in qualified if r['model']==model]
        candidates=[r for r in ledger if r['model']==model]
        need(len(candidates)==536,'planned536_per_model')
        tasks=sorted({r['task_id'] for r in rows})
        sample_matrix=bootstrap_tasks(tasks,bc['draws'],bc['seed'])
        matrix[model]=sample_matrix
        draw_sha=hashlib.sha256(json.dumps(sample_matrix,separators=(',',':')).encode()).hexdigest()
        weights=nested_weights(rows);point=evaluate_motion(rows,weights)
        for r,w in zip(rows,weights):weight_rows.append({**r,'weight':w})
        values=[];reasons=Counter()
        for number,sample in enumerate(sample_matrix):
            value=evaluate_motion(rows,nested_weights(rows,sample));values.append(value)
            if value[3]: reasons[value[3]]+=1
            draw_rows.append({'model':model,'draw':number,'rho_step5':value[0],
                              'rho_step60':value[1],'delta':value[2],'undefined_reason':value[3]})
        valid=[v for v in values if v[2] is not None]
        ci_reason=None
        if point[2] is None:ci_reason=point[3]
        elif len(tasks)<2:ci_reason='fewer_than_two_eligible_tasks'
        elif len(valid)<bc['min_jointly_defined_draws']:ci_reason='fewer_than_9500_jointly_defined_draws'
        estimates={}
        for i,name in enumerate(('rho_step5','rho_step60','delta')):
            estimates[name]={'point':point[i],'pointwise_95_low':None if ci_reason else percentile([v[i] for v in valid],.025),
                             'pointwise_95_high':None if ci_reason else percentile([v[i] for v in valid],.975)}
        delta=estimates['delta']
        branch=('undefined_report_reason' if ci_reason else
                'early_stronger' if delta['pointwise_95_low']>0 else
                'late_stronger' if delta['pointwise_95_high']<0 else 'early_late_difference_unresolved')
        results[model]={'population':'all268 display; same eligible records at both clocks',
                        'planned_records':len(candidates),'qualified_records':len(rows),
                        'qualified_variants':len({r['variant_id'] for r in rows}),'qualified_tasks':len(tasks),
                        'outcome_counts':dict(Counter(r['Y'] for r in rows)),
                        'excluded_records':len(candidates)-len(rows),
                        'exclusion_reason_counts_nonexclusive':dict(Counter(r for x in candidates for r in x['exclusion_reasons'].split('|') if r)),
                        'estimates':estimates,'bootstrap_defined':len(valid),'bootstrap_undefined':len(values)-len(valid),
                        'undefined_reason_counts':dict(reasons),'interval_null_reason':ci_reason,
                        'task_ids':tasks,'draw_matrix_sha256':draw_sha,'wording_branch':branch}
    write_csv(OUT/'E1-WEIGHTS.csv',weight_rows)
    with gzip.open(OUT/'E1-BOOTSTRAP.csv.gz','wt',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(draw_rows[0]));writer.writeheader();writer.writerows(draw_rows)
    with gzip.open(OUT/'E1-TASK-DRAWS.json.gz','wt') as f:json.dump(matrix,f,separators=(',',':'))
    write_json(OUT/'E1-RESULT.json',results)
    return results

def s4_module():
    path=input_path('statistics/internal_link_pure.py')
    spec=importlib.util.spec_from_file_location('original_s4_pure',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

def e2_selected(raw):
    selection=[];qualified=[];seen=set()
    fields=('p_full_key_tv','w_relative_l2','ae_relative_l2','issued_translation_first5_delta_rms')
    for row in raw:
        if row['layer_display']!='18' or row['generation_call']!='0':continue
        need(row['case_id'] not in seen,'duplicate_E2_case');seen.add(row['case_id'])
        reasons=[]
        for key,value in {'cohort':'old_plus536_pi05','joint404_derived':'True','family':'Camera Viewpoints'}.items():
            if row[key]!=value:reasons.append('filter_'+key)
        for key in ('p_full_key_tv_status','w_relative_l2_status','ae_relative_l2_status','issued_status'):
            if row[key]!='measured':reasons.append('unmeasured_'+key)
        for key in fields:
            if finite(row[key]) is None:reasons.append('nonfinite_'+key)
        need(row['issued_translation_first5_unit']=='adapter_command_coordinate','wrong_issued_unit')
        selection.append({'case_id':row['case_id'],'row_id':row['row_id'],'noise_label':row['noise_label'],
                          'task':row['task'],'family':row['family'],'Y_cell':row['Y_cell'],
                          'selected':not reasons,'reasons':'|'.join(reasons)})
        if not reasons:qualified.append({**row,**{k:float(row[k]) for k in fields}})
    return selection,qualified

def run_e2(cfg):
    raw=read_csv(input_path(cfg['inputs'][1]['path']));selection,qualified=e2_selected(raw)
    original=json.loads(input_path('statistics/S4-SCALAR-RESULT.json').read_text())
    saved=original['families']['Camera Viewpoints'];module=s4_module()
    need(len(qualified)==102,'E2_exact_qualified_cases')
    need(dict(Counter(r['task'] for r in qualified))==saved['task_case_counts_primary_common'],'E2_original_task_support')
    need(saved['field_availability_on_derived404']['PG_H_main_relative_l2_measured']==102,'original_PG_qualification')
    groups=defaultdict(list)
    for row in qualified:groups[row['task']].append(row)
    scores={task:module.weighted_auc(rows,'w_relative_l2') for task,rows in groups.items()}
    eligible={task:a for task,a in scores.items() if a is not None}
    need(eligible==saved['harm_AUC_task_scores']['W_pre_gate_relative_l2'],'E2_saved_task_scores')
    point=module.equal_task_auc(eligible,sorted(groups))
    need(point==cfg['E2']['published_AUC'],'E2_published_point')
    task_rows=[]
    for task,rows in sorted(groups.items()):
        counts=Counter(r['Y_cell'] for r in rows)
        task_rows.append({'task':task,'cases':len(rows),'variants':len({r['row_id'] for r in rows}),
                          **{k:counts[k] for k in ('SF','SS','FS','FF')},'eligible':task in eligible,
                          'task_AUC':scores[task],'ineligible_reason':'' if task in eligible else 'no_SF_or_no_SS',
                          'case_ids':'|'.join(sorted(r['case_id'] for r in rows))})
    deletion=[]
    for task in sorted(eligible):
        rest=[t for t in sorted(eligible) if t!=task]
        auc=module.equal_task_auc(eligible,rest)
        remaining=[r for r in qualified if r['task'] in rest]
        counts=Counter(r['Y_cell'] for r in remaining)
        deletion.append({'deleted_task':task,'deleted_task_AUC':eligible[task],
                         'remaining_eligible_tasks':len(rest),'remaining_SF':counts['SF'],'remaining_SS':counts['SS'],
                         'original_AUC':point,'deleted_AUC':auc,'change':None if auc is None else auc-point,
                         'undefined_reason':'' if auc is not None else 'no_eligible_task_after_deletion',
                         'deleted_case_ids':'|'.join(sorted(r['case_id'] for r in groups[task]))})
    finite_deletions=[r['deleted_AUC'] for r in deletion if r['deleted_AUC'] is not None]
    summary={'published_point_reproduced':point,'qualified_cases':len(qualified),'qualified_tasks':len(groups),
             'eligible_tasks':len(eligible),'SF':sum(r['SF'] for r in task_rows if r['eligible']),
             'SS':sum(r['SS'] for r in task_rows if r['eligible']),
             'deletion_AUC_min':min(finite_deletions) if finite_deletions else None,
             'deletion_AUC_max':max(finite_deletions) if finite_deletions else None,
             'max_absolute_change':max(abs(r['change']) for r in deletion if r['change'] is not None) if finite_deletions else None,
             'defined_deletions':len(finite_deletions),'undefined_deletions':len(deletion)-len(finite_deletions),
             'wording_branch':'all_deletions_above_half' if finite_deletions and min(finite_deletions)>.5 else 'one_or_more_deletions_at_or_below_half_or_undefined',
             'inference':'descriptive influence on observed selected tasks; no held-out predictive claim; no new interval',
             'original_S4_interval_not_reestimated':saved['harm_AUC_SF_vs_SS']['W_pre_gate_relative_l2']}
    write_csv(OUT/'E2-CANDIDATE-SUPPORT.csv',selection);write_csv(OUT/'E2-TASK-AUC.csv',task_rows)
    write_csv(OUT/'E2-DELETION-INFLUENCE.csv',deletion);write_json(OUT/'E2-RESULT.json',summary)
    return summary

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--approval',required=True,type=Path);args=parser.parse_args()
    start=now();freeze=json.loads((HERE/'FREEZE-v1.json').read_text());approval=json.loads(args.approval.read_text())
    need(approval.get('status')=='APPROVED_FOR_FROZEN_E_COMPUTATION','root_approval_required')
    for record in freeze['files']:need(sha(STUDY/record['path'])==record['sha256'],'frozen_file_changed:'+record['path'])
    need(approval.get('config_sha256')==sha(HERE/'ESTIMANDS_E_v1.json'),'approval_config_binding')
    need(approval.get('spec_sha256')==sha(HERE/'ESTIMANDS_E_v1.md'),'approval_spec_binding')
    cfg=json.loads((HERE/'ESTIMANDS_E_v1.json').read_text())
    for record in cfg['inputs']:need(sha(BASE/record['path'])==record['sha256'],'input_hash:'+record['path'])
    need(not (OUT/'RUN-RECEIPT.json').exists(),'do_not_overwrite_completed_run')
    a=run_e1(cfg);b=run_e2(cfg)
    outputs=[p for p in OUT.iterdir() if p.is_file() and p.name!='RUN-RECEIPT.json']
    receipt={'status':'COMPUTED_FROZEN_EXISTING_DATA_REANALYSIS','started_at':start,'ended_at':now(),
             'python':sys.version,'platform':platform.platform(),'code_sha256':sha(Path(__file__)),
             'config_sha256':sha(HERE/'ESTIMANDS_E_v1.json'),'spec_sha256':sha(HERE/'ESTIMANDS_E_v1.md'),
             'approval_sha256':sha(args.approval),'command':'python3 -B analysis/E/run_e.py --approval '+str(args.approval),
             'inputs':cfg['inputs'],'outputs':[{'path':str(p.relative_to(STUDY)),'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(outputs)],
             'new_model_calls':0,'new_episodes':0,'cloud_jobs':0,'new_scientific_analysis':['E1','E2']}
    write_json(OUT/'RUN-RECEIPT.json',receipt)
    print(json.dumps({'E1':a,'E2':b},indent=2))

if __name__=='__main__':main()
