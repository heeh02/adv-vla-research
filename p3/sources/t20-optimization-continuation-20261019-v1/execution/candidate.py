"""Create one independently named contract from already restored small metadata."""
import argparse
from pathlib import Path
import continuation_contract as C
import source_loader as S

def main():
    parser=argparse.ArgumentParser()
    for name in ('frozen-repo','parent-contract','parent-metadata','parent-commit-ref',
                 'decision','logical-run-id','scientific-attempt-id','output'):
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--branch',choices=('floor','reheat'),required=True)
    args=parser.parse_args()
    S.configure(args.frozen_repo)
    parent=C.read(args.parent_contract)
    bundle=C.load_bundle(args.parent_metadata,C.read(args.parent_commit_ref),parent)
    C.need(Path(args.decision).resolve()==C.HERE.parent/'SCIENTIFIC_DECISION-v1.json', 'one_root_scientific_decision')
    tail=C.build(parent,bundle,args.logical_run_id,args.branch,
        scientific_attempt_id=args.scientific_attempt_id,decision_identity=C.identity(args.decision),
        source_identity=C.identity(C.HERE/'SOURCE_MANIFEST-v1.json'))
    C.write(args.output,tail)
    print(C.canonical(dict(execute=False,contract=C.identity(args.output),canonical_sha256=C.digest(tail),
        parent_full_model_bytes_read=0,cloud_calls=0)).decode())

if __name__=='__main__':
    main()
