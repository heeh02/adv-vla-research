"""Portable entry for the exact frozen existing-data E calculations."""
import argparse,hashlib,json
from pathlib import Path
import run_e
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
out=a.out.resolve()
if out.exists():raise ValueError('Use a new output directory')
cfg=json.loads((Path(__file__).parent/'CONFIG.json').read_text())
for f in cfg['inputs']:
 run_e.need(hashlib.sha256(run_e.input_path(f['path']).read_bytes()).hexdigest()==f['sha256'],
            'packaged_input_hash:'+f['path'])
out.mkdir(parents=True);run_e.OUT=out
run_e.run_e1(cfg);run_e.run_e2(cfg)
print(json.dumps({'status':'completed_existing_data_reanalysis','output':str(out)}))
