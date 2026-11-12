"""Band-separated and common-task comparisons of the same adopted 536 cases."""
import argparse,csv,json,random
from collections import Counter,defaultdict
from itertools import combinations
from pathlib import Path
import analyze as A

HERE=Path(__file__).resolve().parent
SELECTION=HERE.parent/'parallel-sampling-v1/SELECTION-v1.json'
PRIOR=HERE/'run-v2/RESULT-v1.json'
BANDS=('lower_label','higher_label','unlabeled')

def selected_bands():
    frozen=A.read(SELECTION)
    require=frozen['selected_variant_count']==268 and frozen['bands']=={
        'lower_label':[1,2,3],'higher_label':[4,5],'unlabeled':[None]}
    A.require(require,'frozen_label_band_contract')
    chosen={r['row_id']:r for r in frozen['selected']}
    A.require(len(chosen)==268 and set(chosen)==set(frozen['selected_row_ids']),
              'selected_variant_band_identity')
    return chosen
def group(cases):
    variants=defaultdict(list)
    for row in cases:variants[row['row_id']].append(row)
    result=defaultdict(list)
    for variant,rows in variants.items():
        A.require(len(rows)==2 and {r['noise_label']for r in rows}=={0,1},
                  'two_noise_variant')
        key=(rows[0]['task'],rows[0]['family'],rows[0]['band'])
        A.require(len({(r['task'],r['family'],r['band'])for r in rows})==1,
                  'fixed_variant_identity')
        result[key].append(A.vector_mean([A.point(r['Y_cell'])for r in rows]))
    return {key:A.vector_mean(vectors)for key,vectors in result.items()}
def add_raw(row,cases):
    cells=Counter(r['Y_cell']for r in cases)
    return {**row,'cases':len(cases),'variants':len({r['row_id']for r in cases}),
            **{'raw_'+c:cells[c]for c in ('SS','SF','FS','FF')}}
def summarize(cases,tfb):
    rng=random.Random(A.SEED)
    band_rows=[];band_units=defaultdict(dict)
    for (task,family,band),value in tfb.items():
        band_units[(family,band)][task]=value
    for (family,band),unit in sorted(band_units.items()):
        value=A.vector_mean(list(unit.values()))
        ci=A.bootstrap(unit,rng=rng,bonferroni=False)
        subset=[r for r in cases if r['family']==family and r['band']==band]
        band_rows.append(add_raw({'family':family,'band':band,'tasks':len(unit),
                                  **{k:value[k]for k in
                                  ('reference','treatment','risk_difference','harm','recovery')},
                                  **{k+'_pointwise_95_lo':ci[k]['pointwise_95'][0]for k in ci},
                                  **{k+'_pointwise_95_hi':ci[k]['pointwise_95'][1]for k in ci}},subset))
    # The draft's null Light stratum is excluded here, never pooled with
    # lower/higher labels. Within a task-family, available label bands are
    # equally weighted after each variant's two noise repeats.
    family_task_bands=defaultdict(list)
    for (task,family,band),value in tfb.items():
        if band!='unlabeled':family_task_bands[(task,family)].append(value)
    family_task={key:A.vector_mean(values)for key,values in family_task_bands.items()}
    family_units=defaultdict(dict);task_units=defaultdict(list)
    for (task,family),value in family_task.items():
        family_units[family][task]=value;task_units[task].append(value)
    A.require(set(family_units)==set(A.FAMILIES)and len(task_units)==40,
              'all_labelled_family_task_support')
    labelled_overall=A.vector_mean([A.vector_mean(v)for v in task_units.values()])
    labelled_ci=A.bootstrap({k:A.vector_mean(v)for k,v in task_units.items()},
                            rng=rng,bonferroni=False)
    labelled_rows=[]
    for family in A.FAMILIES:
        unit=family_units[family];value=A.vector_mean(list(unit.values()))
        ci=A.bootstrap(unit,rng=rng)
        subset=[r for r in cases if r['family']==family and r['band']!='unlabeled']
        labelled_rows.append(add_raw({'family':family,'tasks':len(unit),
                                      **{k:value[k]for k in
                                      ('reference','treatment','risk_difference','harm','recovery')},
                                      **{k+'_pointwise_95_lo':ci[k]['pointwise_95'][0]for k in ci},
                                      **{k+'_pointwise_95_hi':ci[k]['pointwise_95'][1]for k in ci},
                                      **{k+'_bonferroni_95_lo':ci[k]['four_family_bonferroni_95'][0]for k in ci},
                                      **{k+'_bonferroni_95_hi':ci[k]['four_family_bonferroni_95'][1]for k in ci}},
                                     subset))
    common=set.intersection(*(set(family_units[f])for f in A.FAMILIES))
    A.require(len(common)==28,'frozen_four_family_common_parent_tasks')
    common_tasks=sorted(common)
    common_task_rows=[]
    for task in common_tasks:
        for family in A.FAMILIES:
            value=family_units[family][task]
            subset=[r for r in cases if r['task']==task and r['family']==family
                    and r['band']!='unlabeled']
            common_task_rows.append(add_raw({'task':task,'family':family,
                                             **{k:value[k]for k in
                                             ('reference','treatment','risk_difference','harm','recovery')}},
                                            subset))
    common_draws={f:{k:[]for k in ('risk_difference','harm','recovery')}for f in A.FAMILIES}
    pairs=list(combinations(A.FAMILIES,2))
    pair_draws={pair:[]for pair in pairs}
    joint_rng=random.Random(A.SEED)
    for _ in range(A.REPETITIONS):
        sampled=[common_tasks[joint_rng.randrange(len(common_tasks))]
                 for _ in common_tasks]
        means={f:{k:A.mean([family_units[f][task][k]for task in sampled])
                  for k in ('risk_difference','harm','recovery')}for f in A.FAMILIES}
        for f in A.FAMILIES:
            for k in common_draws[f]:common_draws[f][k].append(means[f][k])
        for left,right in pairs:
            pair_draws[(left,right)].append(means[left]['risk_difference']-
                                            means[right]['risk_difference'])
    common_rows=[]
    for family in A.FAMILIES:
        unit={task:family_units[family][task]for task in common_tasks}
        value=A.vector_mean(list(unit.values()))
        subset=[r for r in cases if r['task']in common and r['family']==family
                and r['band']!='unlabeled']
        ci=common_draws[family]
        common_rows.append(add_raw({'family':family,'tasks':len(common),
                                    **{k:value[k]for k in
                                    ('reference','treatment','risk_difference','harm','recovery')},
                                    **{k+'_pointwise_95_lo':A.interval(ci[k],.025,.975)[0]for k in ci},
                                    **{k+'_pointwise_95_hi':A.interval(ci[k],.025,.975)[1]for k in ci},
                                    **{k+'_bonferroni_95_lo':A.interval(ci[k],.00625,.99375)[0]for k in ci},
                                    **{k+'_bonferroni_95_hi':A.interval(ci[k],.00625,.99375)[1]for k in ci}},
                                   subset))
    pair_rows=[]
    for left,right in pairs:
        vals=[family_units[left][task]['risk_difference']-
              family_units[right][task]['risk_difference']for task in common_tasks]
        ci=A.interval(pair_draws[(left,right)],.025,.975)
        pair_rows.append({'left_family':left,'right_family':right,'common_tasks':len(common),
                          'same_task_risk_difference_contrast':A.mean(vals),
                          'pointwise_95_lo':ci[0],'pointwise_95_hi':ci[1],
                          'interpretation':'descriptive_same_task_contrast_not_family_causal_order'})
    return {'band_rows':band_rows,'labelled_rows':labelled_rows,
            'labelled_overall':labelled_overall,'labelled_ci':labelled_ci,
            'common_rows':common_rows,'common_task_rows':common_task_rows,
            'common_pair_rows':pair_rows,'common_tasks':common_tasks}

def main():
    p=argparse.ArgumentParser();p.add_argument('--out-dir',type=Path);args=p.parse_args()
    cases,_=A.collect();chosen=selected_bands()
    for row in cases:
        selected=chosen[row['row_id']]
        A.require(selected['family']==row['family']and
                  selected['canonical_parent_task_candidate']==row['task'],
                  'actual_case_frozen_band_identity')
        row['band']=selected['band']
    tfb=group(cases);result=summarize(cases,tfb)
    prior=A.read(PRIOR)
    A.require(prior['selected_case_ids']==536 and
              prior['raw_four_cells']==Counter(r['Y_cell']for r in cases)and
              prior['overall']['risk_difference']==-.0625,
              'unchanged_v2_full_support_result')
    if args.out_dir:
        args.out_dir.mkdir(exist_ok=False)
        for name,rows in (('family-band-v1.csv',result['band_rows']),
                          ('labelled-family-v1.csv',result['labelled_rows']),
                          ('common-family-v1.csv',result['common_rows']),
                          ('common-task-family-v1.csv',result['common_task_rows']),
                          ('common-pairwise-v1.csv',result['common_pair_rows'])):
            A.table(args.out_dir/name,rows)
        receipt={'schema':'plus-paired-outcome-band-common-support-v1',
                 'status':'complete_descriptive_reanalysis_of_same_536',
                 'sources':{'selection':A.identity(SELECTION),'prior_run_v2':A.identity(PRIOR),
                            'case_plan':A.identity(A.PLAN),'coverage':A.identity(A.COVERAGE)},
                 'new_cases':0,'full_536_Y_cells_unchanged':True,
                 'prior_all_band_hierarchical_risk_difference':prior['overall']['risk_difference'],
                 'prior_all_band_role':'subsequently_specified_descriptive_summary_not_strictly_preregistered_primary',
                 'labelled_only_overall':result['labelled_overall'],
                 'labelled_only_overall_ci':result['labelled_ci'],
                 'light_null_separate':True,'light_null_tasks':next(
                     r['tasks']for r in result['band_rows']if r['family']=='Light Conditions'
                     and r['band']=='unlabeled'),
                 'common_parent_tasks':len(result['common_tasks']),
                 'common_task_ids':result['common_tasks'],
                 'all_support_family_task_counts':{r['family']:r['tasks']for r in
                                                     prior_family_rows()},
                 'band_comparisons_are_descriptive':True,
                 'family_pairwise_contrasts_not_causal_order':True,
                 'bootstrap':{'unit':'parent_task','seed':A.SEED,'replicates':A.REPETITIONS,
                              'pointwise_quantiles':[.025,.975],
                              'four_family_bonferroni_quantiles':[.00625,.99375],
                              'episode_resampling':False},
                 'files':{p.name:A.identity(p)for p in args.out_dir.glob('*.csv')}}
        (args.out_dir/'RESULT-v3.json').write_text(json.dumps(receipt,sort_keys=True,indent=2)+'\n')
    print(json.dumps({'status':'complete_descriptive_reanalysis_of_same_536',
                      'common_parent_tasks':len(result['common_tasks']),
                      'light_null_tasks':next(r['tasks']for r in result['band_rows']
                                              if r['family']=='Light Conditions'
                                              and r['band']=='unlabeled'),
                      'labelled_only_overall_risk_difference':
                      result['labelled_overall']['risk_difference']}))

def prior_family_rows():
    return A.csvrows(HERE/'run-v2/family-v1.csv')
if __name__=='__main__':main()
