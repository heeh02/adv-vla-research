"""Root-only entry for the exact plus474 CPU0 controller; no JD call path."""
import argparse
import sys
from pathlib import Path
import cpu0_admission as A


def main():
 p=argparse.ArgumentParser();p.add_argument('--target',choices=('plus474',),required=True)
 p.add_argument('--registry',type=Path);p.add_argument('--root-exclusive-T30',action='store_true')
 p.add_argument('--no-other-owned-route',action='store_true');p.add_argument('upstream',nargs=argparse.REMAINDER);a=p.parse_args()
 args=a.upstream[1:]if a.upstream[:1]==['--']else a.upstream
 A.need(args and args[0]in('dry','execute','reconcile','monitor','readback'),'one_original_CPU_action')
 if args[0]!='dry':A.need(a.root_exclusive_T30 and a.no_other_owned_route,'root_exclusive_original_attestation')
 R,target,targets=A.selected(a.target)
 S=A.load('_CPU0_authority_paths_only',A.HERE.parents[2]/'resource-thread-cap-v1/thread_scope.py');S.authority()
 registry=a.registry or S.ACCOUNT;A.need(registry.resolve()==S.ACCOUNT.resolve(),'exact_shared_ACCOUNT_not_case_ledger')
 A.need('--registry' not in args,'one_registry_parameter')
 if '--external-inventory' not in args:args+=['--external-inventory',str(A.HERE/'EXTERNAL-NOT-APPLICABLE-v1.json')]
 A.install(R,target,targets);sys.argv=[str(R.HERE/'root.py'),*args,'--registry',str(registry)]
 return R.main()


if __name__=='__main__':raise SystemExit(main())
