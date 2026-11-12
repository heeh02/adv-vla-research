"""Enter the accepted stage_source's GET-only unknown-PUT branch."""
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
PACKAGE=HERE/"cpu0-package-v1"
sys.path.insert(0,str(HERE.parent))
import root_adapter  # noqa: E402


def main():
    root,descriptor,q=root_adapter.prepare(PACKAGE)
    state=PACKAGE/"run-v1"
    if not (state/"SOURCE-PUT-INTENT-001.json").exists() or (state/"SUBMIT-INTENT-v1.json").exists():
        raise ValueError("Only pre-Create unknown source can be reconciled")
    H,(L,M,Backend,proxy,life,stop,G,P)=root.helpers()
    auth={};backend=None
    before={p.name for p in state.glob("SOURCE-ABSENT-*.json")}
    try:
        with H.quiet(),proxy():
            auth=L.read_RAM_auth()
            backend=Backend(auth,None)
            bos=root_adapter.module("_paper3_accepted_exact_BOS_writer",PACKAGE/"worker/bos_stdlib.py")
            try:
                root.stage_source(backend,bos,PACKAGE,state,descriptor,retry_review=None)
            except ValueError as error:
                if str(error)!="source_404_requires_explicit_root_retry_authorization":
                    raise
    finally:
        auth.clear()
        if backend is not None:backend.close()
    new=[p for p in state.glob("SOURCE-ABSENT-*.json") if p.name not in before]
    if len(new)!=1:
        raise ValueError("One explicit new source-absence receipt required")
    receipt=new[0]
    value=json.loads(receipt.read_bytes())
    if value["found"] is not False or value["PUT_calls"]!=0 or value["source_package"]!=descriptor["source_package"]:
        raise ValueError("Accepted source-absence receipt contract differs")
    print(json.dumps({"status":"accepted_stage_source_explicit_404_absent",
        "receipt":{"path":str(receipt),**root.C.ident(receipt)},
        "PUT_calls":0,"Create_calls":0}))


if __name__=="__main__":main()
