"""Paired Plus behavior over the exact 536 adopted case IDs; no model or arrays."""
import argparse,csv,hashlib,json,random
from collections import Counter,defaultdict
from pathlib import Path

HERE=Path(__file__).resolve().parent
Q=HERE.parent
PLAN=Q/'plus-valid-mask-analysis-v1/full536-extension-v1/CASE-PLAN-v1.csv'
PLAN_RECEIPT=Q/'plus-valid-mask-analysis-v1/full536-extension-v1/PLAN-IDENTITY-v1.json'
COVERAGE=Q/'paired-case-coverage-v1/final-v1/COVERAGE-v1.json'
STEP_DIR=Q/'plus-valid-mask-analysis-v1/full536-extension-v1/shared-clean-audit-v1'
FAMILIES=('Background Textures','Camera Viewpoints','Light Conditions','Sensor Noise')
METRICS=('reference','treatment','risk_difference','harm','recovery','SS','SF','FS','FF')
SEED=20261010
REPETITIONS=10000

def raw(path):return Path(path).read_bytes()
def identity(path):
    data=raw(path);return {'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}
def read(path):return json.loads(raw(path))
def csvrows(path):
    with Path(path).open(newline='') as stream:return list(csv.DictReader(stream))
def table(path,rows):
    with path.open('x',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
def require(ok,reason):
    if not ok:raise ValueError(reason)
def mean(values):return sum(values)/len(values)
def vector_mean(vectors):
    require(bool(vectors),'nonempty_weighting_level')
    return {key:mean([v[key] for v in vectors])for key in METRICS}
def point(cell):
    require(cell in ('SS','SF','FS','FF'),'accepted_paired_Y_cell')
    ref=int(cell[0]=='S');treat=int(cell[1]=='S')
    return {'reference':ref,'treatment':treat,'risk_difference':treat-ref,
            'harm':int(cell=='SF'),'recovery':int(cell=='FS'),
            **{name:int(cell==name)for name in ('SS','SF','FS','FF')}}
def interval(draws,lower,upper):
    values=sorted(draws)
    def q(p):
        x=p*(len(values)-1);lo=int(x);hi=min(lo+1,len(values)-1)
        return values[lo]+(values[hi]-values[lo])*(x-lo)
    return [q(lower),q(upper)]
def bootstrap(unit_vectors,repetitions=REPETITIONS,rng=None,bonferroni=True):
    rng=rng or random.Random(SEED)
    units=list(unit_vectors);n=len(units);require(n>=2,'at_least_two_task_blocks')
    draws={key:[]for key in ('risk_difference','harm','recovery')}
    for _ in range(repetitions):
        sample=[unit_vectors[units[rng.randrange(n)]]for _ in range(n)]
        for key in draws:draws[key].append(mean([v[key]for v in sample]))
    result={key:{'pointwise_95':interval(values,.025,.975)}
            for key,values in draws.items()}
    if bonferroni:
        for key,values in draws.items():
            result[key]['four_family_bonferroni_95']=interval(values,.00625,.99375)
    return result

def collect():
    plan_receipt=read(PLAN_RECEIPT)
    require(identity(PLAN)==plan_receipt['case_plan'],'frozen_536_case_plan_SHA')
    coverage=read(COVERAGE)
    require(coverage['status']=='complete' and coverage['unique_complete']==536
            and not coverage['missing_case_ids'] and not coverage['duplicate_case_ids'],
            'complete_unique_adopted_536_source_gate')
    source_sets={'v3':set(coverage['sources'][0]['case_ids']),
                 'v4':set(coverage['sources'][1]['case_ids']),
                 'v5':set(coverage['sources'][2]['case_ids'])}
    require([len(source_sets[k])for k in ('v3','v4','v5')]==[270,252,14],
            'exact_adopted_source_counts')
    plan=csvrows(PLAN);require(len(plan)==536 and len({r['case_id']for r in plan})==536,
                              'exact_536_plan_rows')
    all_rows=[];source_checks=[];reference_exec_count=0
    for cohort in ('v3','v4','v5'):
        root=Q/f'parallel-collection-execution-{cohort}/run-v1'
        for path in root.glob('*/analysis/CASE-*.json'):
            value=read(path);case=value['identity']['case_id']
            if case not in source_sets[cohort]:continue
            seq=path.stem.split('-')[-1]
            commit_path=path.parents[1]/'progress'/f'CASE-COMMIT-{seq}.json'
            commit=read(commit_path)['commit']
            require(commit['source_package_sha256']==
                    plan_receipt['source_packages'][cohort[-1]]
                    and commit['run_id']==value['binding']['run_id']
                    and commit['analysis']['uri'].endswith('/ANALYSIS-CASE-v1.json'),
                    'actual_case_commit_source')
            require(value['analysis_status']=='complete' and
                    value['M']['status']=='complete','adopted_case_actual_M_complete')
            y=value['Y'];ref=y['reference'];treat=y['treatment']
            require(ref['status']in('accepted_success','accepted_failure')and
                    treat['status']in('accepted_success','accepted_failure')and
                    ref['success']is(ref['status']=='accepted_success')and
                    treat['success']is(treat['status']=='accepted_success'),
                    'actual_paired_Y_both_accepted')
            refid=value.get('reference_identity')or{}
            require(refid.get('execution_reused')is False and bool(refid.get('clean_id')),
                    'independent_reference_execution_per_case')
            reference_exec_count+=1
            cell=('S'if ref['success']else'F')+('S'if treat['success']else'F')
            all_rows.append({'case_id':case,'cohort':cohort,'run_id':value['binding']['run_id'],
                             'family':value['identity']['family'],'suite':value['identity']['suite'],
                             'task':value['identity']['task'],'Y_cell':cell,
                             'reference_success':int(ref['success']),
                             'treatment_success':int(treat['success']),
                             'clean_id':refid['clean_id'],
                             'source_package_sha256':commit['source_package_sha256'],
                             'runtime_source_sha256':value['binding']['source_sha256'],
                             'analysis_local_sha256':identity(path)['sha256'],
                             'reference_policy_steps':'','reference_queries':'',
                             'treatment_policy_steps':'','treatment_queries':''})
            source_checks.append((case,cohort,path))
    require(len(all_rows)==reference_exec_count==536 and
            len({r['case_id']for r in all_rows})==536,'536_independent_reference_Y_and_unique_cases')
    byid={r['case_id']:r for r in all_rows}
    for planned in plan:
        case=planned['case_id'];row=byid[case]
        require(row['cohort']==planned['cohort']and
                row['run_id']==planned['run_id']and
                row['family']==planned['family']and
                row['suite']==planned['suite']and
                row['task']==planned['parent_task_candidate']and
                row['source_package_sha256']==planned['source_package_sha256'],
                'actual_case_matches_frozen_plan')
        row['row_id']=planned['row_id'];row['noise_label']=int(planned['noise_label'])
    audit=read(STEP_DIR/'READBACK-v1.json')
    stepfile=STEP_DIR/'case-reference-small-v1.csv'
    require(identity(stepfile)==audit['files'][stepfile.name]and
            audit['case_meta_GETs']==508,'independent_508_step_receipt')
    step_rows=csvrows(stepfile);require(len(step_rows)==508 and
                                    len({x['case_id']for x in step_rows})==508,
                                    'unique_508_available_reference_steps')
    for step in step_rows:
        row=byid[step['case_id']]
        require(step['clean_id']==row['clean_id']and step['task']==row['task']
                and step['family']==row['family']and
                step['reference_status']==('accepted_success'if row['reference_success']else'accepted_failure'),
                'reference_step_source_matches_adopted_Y')
        row['reference_policy_steps']=int(step['reference_policy_steps'])
        row['reference_queries']=int(step['reference_queries'])
    all_rows.sort(key=lambda r:r['case_id'])
    return all_rows,{'case_plan':identity(PLAN),'case_plan_receipt':identity(PLAN_RECEIPT),
                     'coverage':identity(COVERAGE),'reference_step_audit':identity(STEP_DIR/'READBACK-v1.json'),
                     'reference_step_table':identity(stepfile)}

def analyze(rows):
    grouped=defaultdict(list)
    for row in rows:grouped[row['row_id']].append(row)
    task_family=defaultdict(list)
    for variant,cases in grouped.items():
        require(len(cases)==2 and {r['noise_label']for r in cases}=={0,1},
                'two_frozen_noise_repeats_per_variant')
        require(len({(r['task'],r['family'])for r in cases})==1,'variant_task_family')
        key=(cases[0]['task'],cases[0]['family'])
        task_family[key].append(vector_mean([point(r['Y_cell'])for r in cases]))
    tf={key:vector_mean(values)for key,values in task_family.items()}
    tasks=defaultdict(list);families=defaultdict(dict)
    for (task,family),value in tf.items():
        tasks[task].append(value);families[family][task]=value
    require(len(tasks)==40 and set(families)==set(FAMILIES),'forty_tasks_four_families')
    task_values={task:vector_mean(values)for task,values in tasks.items()}
    overall=vector_mean(list(task_values.values()))
    rng=random.Random(SEED)
    overall_ci=bootstrap(task_values,rng=rng,bonferroni=False)
    family_rows=[];family_cis={}
    for family in FAMILIES:
        unit=families[family];value=vector_mean(list(unit.values()))
        ci=bootstrap(unit,rng=rng);family_cis[family]=ci
        cases=[r for r in rows if r['family']==family];count=Counter(r['Y_cell']for r in cases)
        family_rows.append({'family':family,'tasks':len(unit),
                            'variants':len({r['row_id']for r in cases}),'cases':len(cases),
                            **{'raw_'+c:count[c]for c in ('SS','SF','FS','FF')},
                            **{k:value[k]for k in ('reference','treatment','risk_difference','harm','recovery')},
                            **{k+'_pointwise_95_lo':ci[k]['pointwise_95'][0]for k in ci},
                            **{k+'_pointwise_95_hi':ci[k]['pointwise_95'][1]for k in ci},
                            **{k+'_bonferroni_95_lo':ci[k]['four_family_bonferroni_95'][0]for k in ci},
                            **{k+'_bonferroni_95_hi':ci[k]['four_family_bonferroni_95'][1]for k in ci}})
    task_rows=[]
    for task,value in sorted(task_values.items()):
        cases=[r for r in rows if r['task']==task];count=Counter(r['Y_cell']for r in cases)
        task_rows.append({'task':task,'families':len(tasks[task]),
                          'variants':len({r['row_id']for r in cases}),'cases':len(cases),
                          **{'raw_'+c:count[c]for c in ('SS','SF','FS','FF')},
                          **{k:value[k]for k in ('reference','treatment','risk_difference','harm','recovery')}})
    task_family_rows=[]
    for (task,family),value in sorted(tf.items()):
        cases=[r for r in rows if r['task']==task and r['family']==family]
        task_family_rows.append({'task':task,'family':family,'variants':len(task_family[(task,family)]),
                                 'cases':len(cases),**{k:value[k]for k in
                                 ('reference','treatment','risk_difference','harm','recovery')}})
    count=Counter(r['Y_cell']for r in rows)
    require(sum(count.values())==536 and sum(count[c]for c in('SS','SF','FS','FF'))==536,
            'all_536_four_cells')
    require(abs(overall['risk_difference']-(overall['recovery']-overall['harm']))<1e-12,
            'paired_difference_equals_recovery_minus_harm')
    reference_steps=[r['reference_policy_steps']for r in rows if r['reference_policy_steps']!='']
    return {'overall':overall,'overall_ci':overall_ci,'raw_four_cells':dict(count),
            'raw_case_weighted_risk_difference':(count['FS']-count['SF'])/536,
            'family_rows':family_rows,'task_rows':task_rows,'task_family_rows':task_family_rows,
            'steps':{'reference_policy_steps_available':len(reference_steps),
                     'reference_policy_steps_mean_available_only':mean(reference_steps),
                     'reference_queries_available':sum(r['reference_queries']!=''for r in rows),
                     'treatment_policy_steps_available':0,
                     'treatment_queries_available':0,
                     'missing_is_not_zero':True}}

def main():
    p=argparse.ArgumentParser();p.add_argument('--out-dir',type=Path);a=p.parse_args()
    rows,sources=collect();result=analyze(rows)
    if a.out_dir:
        a.out_dir.mkdir(exist_ok=False)
        table(a.out_dir/'case-points-v1.csv',rows)
        table(a.out_dir/'family-v1.csv',result['family_rows'])
        table(a.out_dir/'task-v1.csv',result['task_rows'])
        table(a.out_dir/'task-family-v1.csv',result['task_family_rows'])
        receipt={'schema':'plus-paired-outcome-task-weighted-v1',
                 'status':'complete_selected_panel','source_inputs':sources,
                 'selected_case_ids':536,'reference_executions':536,
                 'reference_execution_reuse':False,'clean_id_collapsed':False,
                 'hierarchy':'equal noise within variant, equal variant within task-family, equal family within task, equal task overall',
                 'bootstrap':{'unit':'task','seed':SEED,'replicates':REPETITIONS,
                              'pointwise_quantiles':[.025,.975],
                              'four_family_bonferroni_quantiles':[.00625,.99375],
                              'episode_resampling':False},
                 'raw_four_cells':result['raw_four_cells'],
                 'raw_case_weighted_risk_difference':result['raw_case_weighted_risk_difference'],
                 'overall':result['overall'],'overall_ci':result['overall_ci'],
                 'steps':result['steps'],
                 'scope':'finite selected LIBERO-Plus panel, not the official full leaderboard or full-Plus generalization',
                 'files':{name:identity(a.out_dir/name)for name in
                          ('case-points-v1.csv','family-v1.csv','task-v1.csv','task-family-v1.csv')}}
        (a.out_dir/'RESULT-v1.json').write_text(json.dumps(receipt,sort_keys=True,indent=2)+'\n')
    print(json.dumps({'status':'complete_selected_panel','cases':536,'tasks':len(result['task_rows']),
                      'four_cells':result['raw_four_cells'],
                      'overall_risk_difference':result['overall']['risk_difference'],
                      'reference_steps_available':result['steps']['reference_policy_steps_available']}))
if __name__=='__main__':main()
