"""Standard-library replay of current-basis task-score statistics.

Inputs are derived task-component MSE/energy scores, not activations or poses.
The original task draws are supplied explicitly, so no RNG library is required.
"""
from collections import defaultdict
from pathlib import Path
import argparse,gzip,hashlib,json,math

HERE=Path(__file__).resolve().parent
PAIRS=(('new_response','magnitude'),('new_response','full'),('new_response','new_clean'),
       ('new_response','new_random'),('new_response','old_response'),('flow_plus_new','flow'))

def require(ok,message):
    if not ok:raise ValueError(message)

def mean(values):
    require(bool(values),'empty mean')
    return sum(values)/len(values)

def quantile(values,q):
    require(bool(values) and 0<=q<=1,'invalid quantile input')
    a=sorted(values);position=q*(len(a)-1);lo=math.floor(position);hi=math.ceil(position)
    return a[lo]+(position-lo)*(a[hi]-a[lo])

def interval(values):return [quantile(values,.025),quantile(values,.975)]

def bootstrap_means(values,draws):
    require(len(draws)==2000 and all(len(row)==len(values) for row in draws),'bootstrap shape')
    require(all(type(i)is int and 0<=i<len(values) for row in draws for i in row),'bootstrap indices')
    return [sum(values[i] for i in row)/len(row) for row in draws]

def unique_groups(rows,fields):
    grouped=defaultdict(dict)
    for row in rows:
        key=tuple(row[k] for k in fields);task=row['task_component']
        require(task not in grouped[key],'duplicate task in comparison')
        grouped[key][task]=row
    return grouped

def load(root=HERE):
    root=Path(root);manifest=json.loads((root/'INPUT-MANIFEST.json').read_text())
    values={}
    for name,ref in manifest['files'].items():
        path=(root/name).resolve();require(path.is_relative_to(root.resolve()),'input outside package')
        raw=path.read_bytes()
        require(len(raw)==ref['bytes'] and hashlib.sha256(raw).hexdigest()==ref['sha256'],'input integrity')
        values[Path(name).name]=json.loads(gzip.decompress(raw) if name.endswith('.gz') else raw)
    return values['task_scores.json.gz'],values['draws.json.gz']['draws'],values['expected.json']

def run(scores,draws):
    require(len(scores['readout'])==13104 and len(scores['geometry'])==1008,'complete task-score population')
    grouped=unique_groups(scores['readout'],('family','layer','metric','model'))
    require(len(grouped)==468,'all readout cells')
    estimates=[];contrasts=[];resampled={}
    for (family,layer,metric,model),by_task in sorted(grouped.items()):
        rows=[by_task[t] for t in sorted(by_task)]
        require(len(rows)==28,'common task support')
        mse=[r['mse'] for r in rows];energy=[r['target_energy'] for r in rows]
        require(all(math.isfinite(x) and x>=0 for x in mse+energy),'finite nonnegative scores')
        boot=bootstrap_means(mse,draws[family+'|'+str(layer)]['indices'])
        resampled[(family,layer,metric,model)]=boot
        estimates.append({'family':family,'layer':layer,'metric':metric,'model':model,'task_count':len(rows),
          'case_count':sum(r['case_count'] for r in rows),'task_equal_mse':mean(mse),
          'task_equal_target_energy':mean(energy),'interval_95':interval(boot)})
    for family,layer,metric in sorted({k[:3] for k in grouped}):
        for candidate,baseline in PAIRS:
            ck=(family,layer,metric,candidate);bk=(family,layer,metric,baseline)
            ca,ba=grouped[ck],grouped[bk]
            require(set(ca)==set(ba),'paired task identities')
            ids=sorted(ca);c=[ca[t]['mse'] for t in ids];b=[ba[t]['mse'] for t in ids]
            bc,bb=resampled[ck],resampled[bk]
            diffs=[x-y for x,y in zip(bb,bc)]
            bad=sum(x<=0 for x in bb)
            ratio_ci=None if bad else interval([x/y for x,y in zip(diffs,bb)])
            contrasts.append({'family':family,'layer':layer,'metric':metric,'candidate':candidate,'baseline':baseline,
              'difference_baseline_minus_candidate':{'task_count':len(ids),'mean':mean([x-y for x,y in zip(b,c)]),'interval_95':interval(diffs)},
              'relative_error_reduction':(mean(b)-mean(c))/mean(b) if mean(b)>0 else None,
              'relative_error_reduction_interval_95':ratio_ci,'undefined_bootstrap_denominators':bad})
    geogroups=unique_groups(scores['geometry'],('family','layer','model'));geometry=[]
    require(len(geogroups)==36,'all geometry cells')
    for (family,layer,model),by_task in sorted(geogroups.items()):
        rows=[by_task[t] for t in sorted(by_task)];vals=[]
        for row in rows:
            value=1-row['sse']/row['residual_energy'] if row['residual_energy']>0 else None
            require(value==row['residual_retention'],'task retention matches archived energies')
            vals.append(value)
        stat=None
        if all(v is not None for v in vals):
            boot=bootstrap_means(vals,draws[family+'|'+str(layer)]['indices'])
            stat={'task_count':len(vals),'mean':mean(vals),'interval_95':interval(boot)}
        geometry.append({'family':family,'layer':layer,'model':model,'case_count':sum(r['case_count'] for r in rows),
          'defined_tasks':sum(v is not None for v in vals),'planned_tasks':len(vals),'retention':stat})
    return {'primary_cases':404,'primary_tasks':28,'estimates':estimates,'contrasts':contrasts,'geometry_estimates':geometry}

def compare(actual,expected,path='root',report=None):
    if report is None:report={'checks':0,'maximum_absolute_difference':0.0}
    if isinstance(expected,dict):
        require(isinstance(actual,dict) and set(actual)==set(expected),'keys: '+path)
        for k in expected:compare(actual[k],expected[k],path+'.'+k,report)
    elif isinstance(expected,list):
        require(isinstance(actual,list) and len(actual)==len(expected),'length: '+path)
        for i,(a,b) in enumerate(zip(actual,expected)):compare(a,b,path+f'[{i}]',report)
    elif isinstance(expected,float):
        require(isinstance(actual,(int,float)) and math.isfinite(actual),'finite: '+path)
        err=abs(actual-expected);report['maximum_absolute_difference']=max(report['maximum_absolute_difference'],err)
        require(err<=1e-12+1e-10*abs(expected),'numerical mismatch: '+path);report['checks']+=1
    else:
        require(actual==expected,'value: '+path);report['checks']+=1
    return report

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();require(not args.out.exists(),'output directory must be new')
    scores,draws,expected=load();actual=run(scores,draws);report=compare(actual,expected)
    report.update(status='PASS',readout_estimates=len(actual['estimates']),paired_contrasts=len(actual['contrasts']),
                  geometry_estimates=len(actual['geometry_estimates']),PCA_or_readout_refits=0,model_calls=0)
    args.out.mkdir(parents=True)
    (args.out/'RESULT.json').write_text(json.dumps(actual,sort_keys=True,indent=2)+'\n')
    (args.out/'CHECKS.json').write_text(json.dumps(report,sort_keys=True,indent=2)+'\n')
    print(json.dumps(report,sort_keys=True))

if __name__=='__main__':main()
