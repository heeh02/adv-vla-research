"""Root reviews the concrete fixed source/readset before the unique CPU Create."""
import argparse,json
from pathlib import Path
import contract as C
import build,root
HERE=Path(__file__).resolve().parent
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--prepared',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--root-reviewed',action='store_true');a=p.parse_args()
 d,q=build.verify(a.prepared)
 value=dict(root_reviewed=a.root_reviewed,execution_authorized=a.root_reviewed,run_id=C.RUN,
  resources=C.RESOURCES,phases=C.PHASES,limits=C.LIMITS,source_package_sha256=d['source_package']['sha256'],
  bridge_ready=q['bridge_ready'],repair_ready=q['repair_ready'],controller_manifest=C.ident(HERE/'CONTROLLER-MANIFEST-v1.json'),
  input_credential_resources=q['input_credential_resources'],credential_scope_wider_than_runtime_readset=True,
  unknown_Create_no_retry=True,increment_source_manifest=q['increment_source_manifest'],increment_input_spec=q['increment_input_spec'],
  snapshot_T30_count=3626,full_population_evaluation_complete=False)
 if a.root_reviewed:root.prepare().review_gate(value,d,q)
 with a.out.open('xb')as f:f.write(C.encoded(value))
 print(json.dumps(dict(review=str(a.out),root_reviewed=a.root_reviewed,cloud_calls=0)))
