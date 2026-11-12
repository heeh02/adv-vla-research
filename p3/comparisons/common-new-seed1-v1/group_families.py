"""Reproduce the small paired-index join against the original init-file routing."""
from pathlib import Path
import importlib.util,json,hashlib,argparse
from collections import defaultdict,Counter
HERE=Path(__file__).resolve().parent;C=HERE.parents[1]
def main():
 p=argparse.ArgumentParser();p.add_argument('--check',action='store_true');a=p.parse_args()
 path=C/'reheat-floor-commoncase-v1/analyze.py';s=importlib.util.spec_from_file_location('_fixed_init_route',path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
 proof_path=C/'plus-init-family-evidence-v1/ROUTING-EVIDENCE-v1.json';proof=json.loads(proof_path.read_text());cases,routes=m.metadata(proof);rows=json.loads((HERE/'PAIRED-TRACE-v1.json').read_text());groups=defaultdict(Counter)
 for r in rows:
  o=r['ordinal'];assert r['category']==cases[o]['category'];family=routes[o]['init_family'];assert family is not None;g=groups[(r['category'],family)];g['planned']+=1
  if r['cell']:g[r['cell']]+=1
  else:g['rejected']+=1
 out=[]
 for (category,family),g in sorted(groups.items()):out.append(dict(category=category,initialization_family=family,planned=g['planned'],qualified=sum(g[k]for k in ['SS','SF','FS','FF']),gains=g['FS'],harms=g['SF'],net=g['FS']-g['SF'],SS=g['SS'],FF=g['FF'],rejected=g['rejected']))
 assert sum(g['qualified']for g in out)==491 and sum(g['net']for g in out)==9
 if a.check:assert json.loads((HERE/'FAMILY-GROUPS-v1.json').read_text())['groups']==out
 print(json.dumps({'status':'EXACT_SAVED_GROUPS_REPRODUCED'if a.check else'DERIVED_GROUPS','families_by_category':len(out),'qualified':491,'net':9,'cloud_calls':0,'raw_arrays':0}))
if __name__=='__main__':main()
