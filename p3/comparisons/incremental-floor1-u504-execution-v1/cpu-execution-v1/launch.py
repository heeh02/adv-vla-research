"""Mandatory existing thread/JD attribution + deterministic route composition."""
import argparse,sys
from pathlib import Path
import root
HERE=Path(__file__).resolve().parent;CLOSEOUT=HERE.parents[2]

def main():
    p=argparse.ArgumentParser();p.add_argument('--registry',type=Path);p.add_argument('--outside-registration-root',type=Path,action='append',default=[])
    p.add_argument('--root-exclusive-T30',action='store_true');p.add_argument('--no-other-owned-route',action='store_true');p.add_argument('upstream',nargs=argparse.REMAINDER);a=p.parse_args()
    R=root.prepare();R.verify_sources()
    sys.path.insert(0,str(CLOSEOUT/'resource-cf-wire-compat-v1'));import wire_scope
    S=wire_scope.install();S.authority()
    wrapper=S.load('_increment_existing_scope_installer',root.ORIGINAL/'scoped_root.py')
    registry=a.registry or S.ACCOUNT
    wrapper.install(R,S,registry,a.outside_registration_root,S.attested(a.root_exclusive_T30,a.no_other_owned_route))
    args=a.upstream[1:] if a.upstream[:1]==['--'] else a.upstream
    S.need(args and '--registry' not in args,'one_explicit_action_and_registry')
    sys.argv=[str(HERE/'root.py'),*args,'--registry',str(registry)];return R.main()

if __name__=='__main__':raise SystemExit(main())
