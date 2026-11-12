"""Invoke the frozen incremental adapter inside the admitted original CPU image."""
import argparse,sys
from pathlib import Path
import contract as C

def execute(root,science,archive,hard):
    root=Path(root);package=root/'package';q=C.request_gate(C.read(package/'metadata/REQUEST-v1.json'))
    admission=C.read(root/'RUNTIME-ADMISSION-v1.json')
    C.need((science,archive,hard)==tuple(admission[k] for k in ('science_deadline_epoch','archive_target_epoch','deadline_epoch')),'original_admitted_deadlines')
    C.need(admission['resources']==C.RESOURCES and admission['input_spec']==q['increment_input_spec'],'exact_increment_CPU_runtime')
    sys.path.insert(0,str(package/'worker/science'));import cloud_adapter
    return cloud_adapter.execute(package/'worker/science/prepared-v2',package/'worker/deps',
                                 root/'staged-inputs',root/'results',root/'RUNTIME-ADMISSION-v1.json')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--execute-cloud',action='store_true');p.add_argument('--root',required=True)
    for n in ('science-deadline-epoch','archive-target-epoch','deadline-epoch'):p.add_argument('--'+n,type=float,required=True)
    a=p.parse_args();C.need(a.execute_cloud and sys.platform.startswith('linux'),'explicit_cloud_CPU')
    r=execute(a.root,a.science_deadline_epoch,a.archive_target_epoch,a.deadline_epoch)
    raise SystemExit(0 if r['snapshot_comparison_complete'] and r['snapshot_cloud_durable'] else 2)
