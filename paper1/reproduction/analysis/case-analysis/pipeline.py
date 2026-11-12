#!/usr/bin/env python3
"""Offline, standard-library case-level derived-record analysis. No model or network."""
import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import itertools
import json
import math
from pathlib import Path
import platform
import random
import sys
import time

ARMS = ('R0', 'T0', 'R1', 'T1', 'R0prime')
PAIRS = {'TR0': ('R0','T0'), 'TR1': ('R1','T1'), 'R0_R1': ('R0','R1'), 'R0_R0prime': ('R0','R0prime')}
MODELS = ('pi05_libero','pi0_libero')
FAMILIES = ('Camera Viewpoints','Sensor Noise','Background Textures','Light Conditions')
HERE = Path(__file__).resolve().parent

def require(ok, reason):
    if not ok: raise ValueError(reason)

def digest(raw): return hashlib.sha256(raw).hexdigest()
def canonical(value): return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def write_json(path, value): path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def mean(values): return math.fsum(values)/len(values) if values else None

def validate(payload, plan):
    """Validate record-level contracts independently of file-hash validation."""
    require(payload.get('schema')=='joint-motion-c-plot-v1','input schema')
    require(payload.get('data_class')=='observed_existing_scalars','derived data class')
    require(payload.get('units')=={'orientation':'radian','position':'metre','q0':'adapter command coordinates; norm across returned first5 component entries'},'units contract')
    rows=payload['rows']; require(len(rows)==536,'536 planned model-variants')
    pindex={}
    for p in plan['rows']:
        key=(p['model'],p['variant_id']); require(key not in pindex,'duplicate planned key'); pindex[key]=p
    require(len(pindex)==536,'536 independent plan keys')
    seen=set(); task_map={}; unknown=0
    for row in rows:
        key=(row['model'],row['variant_id']); require(key not in seen,'duplicate observed key'); seen.add(key)
        require(key in pindex,'foreign observed key'); p=pindex[key]
        require(row['model'] in MODELS and row['family'] in FAMILIES,'model/family domain')
        require(type(row['common28']) is bool,'strict support boolean')
        require(all(row[k]==p[k] for k in ('suite','family','common28','condition','official_level')),'plan identity or pairing conflict')
        require(row['task_id']==p['suite']+':'+p['task'],'original task identity bridge')
        if p['task'] in task_map: require(task_map[p['task']]==p['suite'],'task hash collision across suites')
        task_map[p['task']]=p['suite']
        require(type(row['identity_qualified']) is bool and isinstance(row['validation_errors'],list),'qualification schema')
        for k in ('source_sha256','source_package_sha256','source_cache_sha256','source_commit_sha256'):
            require(isinstance(row[k],str) and len(row[k])==64 and all(c in '0123456789abcdef' for c in row[k]),'source SHA schema')
        require(isinstance(row['run_id'],str) and isinstance(row['job_id'],str) and type(row['sequence']) is int,'run envelope')
        require(set(row['outcomes'])==set(ARMS)==set(row['arm_qualification']),'five shared arm variables')
        for arm,value in row['outcomes'].items():
            require(value is None or type(value) is bool,'strict nullable outcome')
            unknown+=value is None
            q=row['arm_qualification'][arm]
            require(isinstance(q['issues'],list),'arm issues')
            require(q['available_pose_clocks'] is None or isinstance(q['available_pose_clocks'],list),'clock array')
            if value is not None:
                require(q['status']==('accepted_success' if value else 'accepted_failure'),'outcome qualification conflict')
        for n in ('0','1'):
            r,t=row['outcomes']['R'+n],row['outcomes']['T'+n]
            expected=None if r is None or t is None else ('S' if r else 'F')+('S' if t else 'F')
            require(row['noise_pair_cells'][n]==expected,'paired outcome cell conflict')
        require(set(row['motion'])==set(PAIRS),'motion pairing domain')
        for pair,clocks in row['motion'].items():
            require(set(clocks)=={'5','60'},'fixed sparse motion clocks')
            for clock,metrics in clocks.items():
                require(set(metrics)=={'position_distance','orientation_angle'},'motion component domain')
                for name,v in metrics.items():
                    unit='metre' if name=='position_distance' else 'radian'
                    require(v['unit']==unit,'illegal motion unit')
                    if v['value'] is not None:
                        require(type(v['value']) in (float,int) and math.isfinite(v['value']) and v['value']>=0,'negative or nonfinite distance')
                        require(v['status']=='measured','value without measured status')
                    else: require(v['status']!='measured','measured without value')
        require(set(row['q0'])=={'translation','rotation','gripper'},'q0 components')
        for name,q in row['q0'].items():
            require(q['status'] in ('qualified','unsupported'),'q0 qualification')
            if q['status']=='qualified':
                for field in ('D','M','V','norm0','norm1'):
                    require(type(q[field]) in (float,int) and math.isfinite(q[field]) and q[field]>=0,'q0 finite nonnegative')
                require(math.isclose(q['D'],(q['norm0']**2+q['norm1']**2)/2,rel_tol=1e-9,abs_tol=1e-13),'q0 norm scale')
                require(math.isclose(q['U'],q['M']-q['V'],rel_tol=1e-8,abs_tol=1e-13),'q0 shared identity')
                require(math.isclose(q['D'],q['M']+q['V'],rel_tol=1e-8,abs_tol=1e-13),'q0 total identity')
                if q['D']>0:
                    require(math.isclose(q['U_over_D'],q['U']/q['D'],rel_tol=1e-8,abs_tol=1e-12),'q0 U/D identity')
                else: require(q['U_over_D'] is None,'zero q0 denominator')
    require(seen==set(pindex),'plan support lost')
    require(unknown==payload['support']['unknown_outcomes']==39,'unknown39 retained')
    require(payload['support']['planned_arms']==2680 and payload['support']['planned_model_variants']==536,'planned global denominator')
    for model in MODELS:
        mr=[r for r in rows if r['model']==model];cr=[r for r in mr if r['common28']]
        require(len(mr)==268 and len(cr)==202,'model268 primary202')
        require(len({r['task_id'] for r in cr})==28 and len({r['task_id'] for r in mr})==40,'task28/40 support')
    return {'status':'VERIFIED','rows':536,'planned_arms':2680,'unknown_outcomes':unknown,'primary_variants_per_model':202,'all_variants_per_model':268,'identity_qualified':sum(r['identity_qualified'] for r in rows),'q0_note':'fields validated, not independently re-extracted from commands'}

def complete_metrics(v):
    r0,t0,r1,t1,rp=(int(v[a]) for a in ARMS)
    harm=(r0*(1-t0)+r1*(1-t1))/2
    recovery=(t0*(1-r0)+t1*(1-r1))/2
    flip=harm+recovery
    out=dict(clean_success=(r0+r1)/2,treated_success=(t0+t1)/2,risk_difference=recovery-harm,harm=harm,recovery=recovery,ss=(r0*t0+r1*t1)/2,ff=((1-r0)*(1-t0)+(1-r1)*(1-t1))/2,tr_flip=flip,rr_policy_flip=abs(r0-r1),rr_repeat_flip=abs(r0-rp),tt_policy_flip=abs(t0-t1),excess_flip_vs_policy=flip-abs(r0-r1),excess_flip_vs_repeat=flip-abs(r0-rp))
    for label,pairs in [('TR',((r0,t0),(r1,t1))),('R0_R1',((r0,r1),)),('R0_R0prime',((r0,rp),))]:
        for cell in ('SS','SF','FS','FF'):
            out[label+'_'+cell]=mean([float(('S' if a else 'F')+('S' if b else 'F')==cell) for a,b in pairs])
    return out

def variant_bounds(outcomes):
    require(set(outcomes)==set(ARMS),'five outcome variables required')
    require(all(v is None or type(v) is bool for v in outcomes.values()),'strict outcome values')
    missing=[a for a in ARMS if outcomes[a] is None]
    completions=[complete_metrics(dict(outcomes,**dict(zip(missing,assignment)))) for assignment in itertools.product((False,True),repeat=len(missing))]
    result={}
    for name in completions[0]:
        values=[c[name] for c in completions];lo,hi=min(values),max(values)
        result[name]={'lower':lo,'upper':hi,'point':lo if lo==hi else None}
    return result

def average_bounds(entries):
    lo=mean([v['lower'] for v in entries]);hi=mean([v['upper'] for v in entries])
    return {'lower':lo,'upper':hi,'point':lo if lo==hi else None}

def bootstrap_indices(tasks,reps,seed):
    """Identical recovered stdlib choice sequence; caller supplies original order."""
    rng=random.Random(seed)
    return [[rng.choice(list(tasks)) for _ in tasks] for _ in range(reps)]

def interval(values,alpha,reps):
    if len(values)!=reps or any(v is None or not math.isfinite(v) for v in values): return None
    ordered=sorted(values)
    return [ordered[int(alpha/2*reps)],ordered[int((1-alpha/2)*reps)-1]]

def task_key(row): return row['task_id'].split(':',1)[1]

def outcomes_panel(rows,scope_tasks,config):
    bytask=defaultdict(list)
    for row in rows: bytask[task_key(row)].append(variant_bounds(row['outcomes']))
    task_values={t:{k:average_bounds([v[k] for v in variants]) for k in variants[0]} for t,variants in bytask.items()}
    cfg=config['bootstrap'];draws=bootstrap_indices(scope_tasks,cfg['draws'],cfg['seed'])
    metric_values={}
    for name in next(iter(task_values.values())):
        values=average_bounds([v[name] for v in task_values.values()]);series=[]
        for draw in draws:
            selected=[task_values[t][name]['point'] for t in draw if t in task_values]
            series.append(mean(selected) if selected and all(v is not None for v in selected) else None)
        identified=values['point'] is not None
        values.update(status='VERIFIED' if identified else 'PARTIALLY_IDENTIFIED',identification_bounds=[values['lower'],values['upper']],ci95=interval(series,.05,cfg['draws']) if identified else None,ci98_75=interval(series,.0125,cfg['draws']) if identified and name=='risk_difference' else None,bootstrap_defined_draws=sum(x is not None for x in series),bootstrap_planned_draws=cfg['draws'],bounds_are_confidence_intervals=False)
        metric_values[name]=values
    raw_counts={label:Counter() for label in ('TR','R0_R1','R0_R0prime')}
    for row in rows:
        for label,pairs in [('TR',(('R0','T0'),('R1','T1'))),('R0_R1',(('R0','R1'),)),('R0_R0prime',(('R0','R0prime'),))]:
            for a,b in pairs:
                va,vb=row['outcomes'][a],row['outcomes'][b]
                raw_counts[label]['unknown' if va is None or vb is None else ('S' if va else 'F')+('S' if vb else 'F')]+=1
    return {'planned_variants':len(rows),'planned_tasks':len(bytask),'planned_TR_pairs':2*len(rows),'planned_arms':5*len(rows),'unknown_arms':sum(v is None for row in rows for v in row['outcomes'].values()),'raw_pair_counts':{k:{c:v[c] for c in ('SS','SF','FS','FF','unknown')} for k,v in raw_counts.items()},'metrics':metric_values,'task_metrics':task_values,'outcome_support':'fixed planned variants; two noise labels each weight1/2','bootstrap_scope_task_order':scope_tasks,'bootstrap_draw_indices_sha256':digest(canonical(draws)),'bounds_interpretation':'Each metric interval is its marginal sharp bound under shared per-case arm variables. Endpoints of different metrics need not be jointly attainable. No confidence coverage is claimed for identification bounds.'}

def read_motion(row,pair,clock,metric):
    if not row['identity_qualified'] or row['validation_errors']: return None,'identity_unqualified'
    if row['motion_status']!='processed': return None,'motion_status:'+str(row['motion_status'])
    value=row['motion'].get(pair,{}).get(str(clock),{}).get(metric)
    if value is None: return None,'fixed_clock_or_metric_absent'
    if value['status']!='measured' or value['value'] is None: return None,value.get('reason') or 'unavailable_unspecified'
    for arm in PAIRS[pair]:
        if clock not in (row['arm_qualification'][arm]['available_pose_clocks'] or []): return None,'fixed_clock_pose_not_verified:'+arm
    if pair=='R0_R0prime' and (row['outcomes']['R0prime'] is None or row['arm_qualification']['R0prime']['issues']):
        return None,'same_run_repeat_eligibility_not_proven'
    return float(value['value']),None

def motion_panel(rows,clock,metric,reference,config):
    support=[];joint=[]
    for row in rows:
        vals={};reasons={}
        for pair in ('TR0','TR1',reference):
            value,why=read_motion(row,pair,clock,metric);vals[pair]=value
            if why: reasons[pair]=why
        included=not reasons
        support.append({'model':row['model'],'variant_id':row['variant_id'],'task':task_key(row),'family':row['family'],'clock':clock,'metric':metric,'reference':reference,'included':included,'reason':json.dumps(reasons,sort_keys=True),'TR0':vals['TR0'],'TR1':vals['TR1'],'reference_value':vals[reference]})
        if included: joint.append({'task':task_key(row),'variant_id':row['variant_id'],'TR':mean([vals['TR0'],vals['TR1']]),'reference':vals[reference]})
    bytask=defaultdict(list)
    for r in joint: bytask[r['task']].append(r)
    task_values={t:{k:mean([v[k] for v in variants]) for k in ('TR','reference')} for t,variants in bytask.items()}
    tasks=sorted(task_values)
    def stats(draw):
        if not draw: return {k:None for k in ('TR_mean','reference_mean','mean_paired_difference','ratio_of_task_means')}
        tr=mean([task_values[t]['TR'] for t in draw]);ref=mean([task_values[t]['reference'] for t in draw])
        diff=mean([task_values[t]['TR']-task_values[t]['reference'] for t in draw])
        return dict(TR_mean=tr,reference_mean=ref,mean_paired_difference=diff,ratio_of_task_means=tr/ref if ref>0 else None)
    point=stats(tasks);cfg=config['bootstrap'];draws=bootstrap_indices(tasks,cfg['draws'],cfg['seed']) if len(tasks)>=2 else []
    summaries=[stats(draw) for draw in draws];ci={};defined={}
    for name in point:
        series=[r[name] for r in summaries];defined[name]=sum(v is not None and math.isfinite(v) for v in series)
        ci[name]=interval(series,.05,cfg['draws']) if point[name] is not None else None
    result={'clock':clock,'metric':metric,'unit':'metre' if metric=='position_distance' else 'radian','reference':reference,'planned_variants':len(rows),'planned_tasks':len({task_key(r) for r in rows}),'joint_variants':len(joint),'joint_tasks':len(tasks),'planned_tasks_without_joint_support':len({task_key(r) for r in rows})-len(tasks),'point':point,'ci95':ci,'bootstrap_defined_draws':defined,'bootstrap_planned_draws':cfg['draws'],'bootstrap_task_order':tasks,'bootstrap_draw_indices_sha256':digest(canonical(draws)),'joint_variant_ids':[r['variant_id'] for r in joint],'task_values':task_values,'status':'VERIFIED' if tasks else 'UNSUPPORTED_NO_JOINT_SUPPORT','ratio_status':'VERIFIED' if point['ratio_of_task_means'] is not None else 'UNDEFINED_ZERO_REFERENCE_OR_NO_SUPPORT','denominator':'metric/time/reference-specific joint support, not outcome plan','exclusion_counts':dict(Counter(r['reason'] for r in support if not r['included']))}
    return result,support

def run(payload,config):
    panels=[];support=[]
    for model in config['models']:
        for scope in config['scopes']:
            rows=[r for r in payload['rows'] if r['model']==model and (scope=='all268' or r['common28'])]
            tasks=sorted({task_key(r) for r in rows})
            for family in config['families']:
                fr=[r for r in rows if r['family']==family];out=outcomes_panel(fr,tasks,config);motion=[]
                for clock in config['clocks']:
                    for metric in config['motion_metrics']:
                        for ref in config['motion_references']:
                            mp,sp=motion_panel(fr,clock,metric,ref,config);motion.append(mp)
                            support.extend(dict(x,scope=scope) for x in sp)
                panels.append(dict(model=model,scope=scope,family=family,outcomes=out,motion=motion))
    return {'schema':'d1-case-derived-results-v1','scope':'existing case-level derived scalars; no raw trajectory reconstruction','panels':panels},support

def smoke(result):
    panel=next(p for p in result['panels'] if p['model']=='pi0_libero' and p['scope']=='common28' and p['family']=='Sensor Noise')
    motion=next(p for p in panel['motion'] if p['clock']==5 and p['metric']=='position_distance' and p['reference']=='R0_R1')
    vals={'variants':panel['outcomes']['planned_variants'],'tasks':panel['outcomes']['planned_tasks'],'step5_ratio':motion['point']['ratio_of_task_means'],'harm':panel['outcomes']['metrics']['harm']['point'],'recovery':panel['outcomes']['metrics']['recovery']['point'],'risk_difference':panel['outcomes']['metrics']['risk_difference']['point']}
    expected={'variants':55,'tasks':28,'step5_ratio':0.5803425149195939,'harm':0.16071428571428573,'recovery':0.017857142857142856,'risk_difference':-0.14285714285714288}
    require(all(math.isclose(vals[k],v,rel_tol=1e-12,abs_tol=1e-14) for k,v in expected.items()),'published Sensor regression')
    return {'status':'VERIFIED','actual':vals,'expected':expected,'tolerance':{'rel':1e-12,'abs':1e-14}}

def csv_outputs(out,result,support):
    with (out/'MOTION_SUPPORT.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(support[0]));w.writeheader();w.writerows(support)
    records=[]
    for p in result['panels']:
        for name,v in p['outcomes']['metrics'].items():
            records.append(dict(model=p['model'],scope=p['scope'],family=p['family'],metric=name,planned_variants=p['outcomes']['planned_variants'],tasks=p['outcomes']['planned_tasks'],point=v['point'],bound_lower=v['lower'],bound_upper=v['upper'],ci95_lower=v['ci95'][0] if v['ci95'] else None,ci95_upper=v['ci95'][1] if v['ci95'] else None,ci98_75_lower=v['ci98_75'][0] if v['ci98_75'] else None,ci98_75_upper=v['ci98_75'][1] if v['ci98_75'] else None,status=v['status']))
    with (out/'OUTCOMES.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    records=[]
    for p in result['panels']:
        for m in p['motion']:
            for name,value in m['point'].items():
                ci=m['ci95'][name]
                records.append(dict(model=p['model'],scope=p['scope'],family=p['family'],clock=m['clock'],metric=m['metric'],reference=m['reference'],estimand=name,unit='unitless' if name=='ratio_of_task_means' else m['unit'],point=value,ci95_lower=ci[0] if ci else None,ci95_upper=ci[1] if ci else None,planned_variants=m['planned_variants'],joint_variants=m['joint_variants'],joint_tasks=m['joint_tasks'],status=m['ratio_status'] if name=='ratio_of_task_means' else m['status']))
    with (out/'MOTION.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--input',type=Path,default=HERE/'data/MOTION-C-PLOT-v1.json.gz');parser.add_argument('--plan',type=Path,default=HERE/'data/PANEL-PLAN.json');parser.add_argument('--config',type=Path,default=HERE/'CONFIG.json');parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    started=time.perf_counter();args.output.mkdir(parents=True,exist_ok=True)
    raw=args.input.read_bytes();config_raw=args.config.read_bytes();config=json.loads(config_raw);plan_raw=args.plan.read_bytes()
    require(digest(config_raw)==(args.config.with_suffix('.sha256')).read_text().split()[0],'frozen config SHA')
    require(digest(raw)==config['input_sha256'] and len(raw)==config['input_bytes'],'input hash/size')
    require(digest(plan_raw)==config['plan_sha256'],'independent plan hash')
    payload=json.loads(gzip.decompress(raw));plan=json.loads(plan_raw);validation=validate(payload,plan)
    result,support=run(payload,config);regression=smoke(result)
    result['input_sha256']=digest(raw);result['config_sha256']=digest(config_raw);result['pipeline_sha256']=digest(Path(__file__).read_bytes())
    write_json(args.output/'RESULTS.json',result);write_json(args.output/'VALIDATION.json',validation);write_json(args.output/'KNOWN-POINT.json',regression);csv_outputs(args.output,result,support)
    runtime={'status':'COMPLETE','completed_at':datetime.now(timezone.utc).isoformat(),'elapsed_seconds':time.perf_counter()-started,'python':sys.version,'platform':platform.platform(),'executable':sys.executable,'command':sys.argv,'input_sha256':digest(raw),'config_sha256':digest(config_raw),'pipeline_sha256':digest(Path(__file__).read_bytes()),'raw_array_reads':0,'model_calls':0,'GPU':0,'new_episodes':0,'scientific_results_sha256':digest((args.output/'RESULTS.json').read_bytes())}
    write_json(args.output/'RUN.json',runtime)
    print(json.dumps({'status':'COMPLETE','panels':len(result['panels']),'motion_panels':sum(len(p['motion']) for p in result['panels']),'support_rows':len(support),'elapsed_seconds':runtime['elapsed_seconds'],'sensor':regression['actual']}))
if __name__=='__main__': main()
