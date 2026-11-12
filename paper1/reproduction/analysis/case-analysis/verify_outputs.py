#!/usr/bin/env python3
"""Compare regenerated values with independently sourced historical result files."""
import argparse
import hashlib
import json
import math
from pathlib import Path
HERE=Path(__file__).resolve().parent

def same(a,b):
    if a is None or b is None:return a is b
    if isinstance(a,list):return isinstance(b,list) and len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
    return math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-14)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('output',type=Path);a=ap.parse_args();result=json.loads((a.output/'RESULTS.json').read_text());ref=json.loads((HERE/'expected/PUBLISHED-REFERENCE.json').read_text());errors=[];checks=0
    for old in ref['items']:
        panel=next(p for p in result['panels'] if all(p[k]==old[k] for k in ('model','scope','family')))
        if old['kind']=='motion':
            new=next(m for m in panel['motion'] if all(m[k]==old[k] for k in ('clock','metric','reference')))
            fields=['joint_variants','joint_tasks']
            for section in ('point','ci95'):
                for field,value in old[section].items():
                    checks+=1
                    if not same(new[section][field],value):errors.append({'key':old,'field':section+'/'+field,'observed':new[section][field],'expected':value})
        else:new=panel['outcomes']['metrics'][old['metric']];fields=['point','lower','upper','ci95','ci98_75','bootstrap_defined_draws']
        for field in fields:
            checks+=1
            if not same(new[field],old[field]):errors.append({'key':old,'field':field,'observed':new[field],'expected':old[field]})
    report={'status':'VERIFIED' if not errors else 'FAILED','historical_entries':len(ref['items']),'scalar_or_interval_checks':checks,'errors':errors,'reference_sha256':hashlib.sha256((HERE/'expected/PUBLISHED-REFERENCE.json').read_bytes()).hexdigest(),'result_sha256':hashlib.sha256((a.output/'RESULTS.json').read_bytes()).hexdigest(),'tolerance':ref['tolerance'],'level':'Recomputes from case-derived records, compares with historical estimates; neither raw acquisition replay nor new scientific estimate.'}
    (a.output/'REGRESSION.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='errors'}));raise SystemExit(0 if not errors else 1)
if __name__=='__main__':main()
