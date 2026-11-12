"""Read one exact unknown source object without PUT/Create/CAS or secret output."""
import hashlib
import argparse
import json
from pathlib import Path
import re
import sys
import time

HERE=Path(__file__).resolve().parent
PACKAGE=HERE/"cpu0-package-v1"
sys.path.insert(0,str(HERE.parent))
import root_adapter  # noqa: E402


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--receipt",required=True)
    args=parser.parse_args()
    root,descriptor,q=root_adapter.prepare(PACKAGE)
    state=PACKAGE/"run-v1"
    intent=state/"SOURCE-PUT-INTENT-001.json"
    unknown=sorted(state.glob("SOURCE-PUT-UNKNOWN-*.json"))
    if not intent.is_file() or len(unknown)!=1 or (state/"SUBMIT-INTENT-v1.json").exists():
        raise ValueError("Only the exact pre-Create first unknown source is eligible")
    prior=json.loads(intent.read_bytes())
    if prior["run_id"]!=q["run_id"] or prior["source_package"]!=descriptor["source_package"]:
        raise ValueError("Unknown write intent differs from immutable package")
    source=q["output_prefix"]+"control/SOURCE_PACKAGE-v1.tar.gz"
    resource=source.removeprefix("bos://")
    H,(L,M,Backend,proxy,life,stop,G,P)=root.helpers()
    auth={};env={};backend=None
    observation={"schema":"paper3-exact-source-unknown-readonly-v1",
        "run_id":q["run_id"],"source_uri":source,"expected":descriptor["source_package"],
        "intent_sha256":hashlib.sha256(intent.read_bytes()).hexdigest(),
        "unknown_sha256":hashlib.sha256(unknown[0].read_bytes()).hexdigest(),
        "PUT_calls":0,"Create_calls":0,"registry_writes":0}
    try:
        with H.quiet(),proxy():
            auth=L.read_RAM_auth()
            backend=Backend(auth,None)
            env=backend.sts([{"service":"bce:bos","region":"bj","effect":"Allow",
                "resource":[resource],"permission":["READ"]}],900)
            bos=root_adapter.module("_paper3_exact_unknown_BOS_reader",PACKAGE/"worker/bos_stdlib.py")
            with backend.old._environment(env):
                writer=bos.BOSWriter(q["output_prefix"],time.time()+180)
                actual=writer._get_identity(writer.prefix+"control/SOURCE_PACKAGE-v1.tar.gz")
            if actual is None:
                observation.update(status="explicit_404_absent",found=False,whole_GET_SHA=False)
            else:
                matching=actual==(descriptor["source_package"]["bytes"],descriptor["source_package"]["sha256"])
                observation.update(status="present_exact_match" if matching else "present_conflict",
                                   found=True,actual={"bytes":actual[0],"sha256":actual[1]},
                                   whole_GET_SHA=True)
    except Exception as error:
        # BOS exception text can include a provider RequestId and is never
        # printed/saved. Only the HTTP status and validated service code pass.
        message=str(error)
        code=re.search(r"Code=([A-Za-z][A-Za-z0-9_]{0,63})",message)
        observation.update(status="read_unavailable",error_type=type(error).__name__,
                           missing_module=getattr(error,"name",None) if type(error).__name__=="ModuleNotFoundError" else None,
                           http_code=getattr(error,"http_code",None),
                           service_code=code.group(1) if code else None)
    finally:
        env.clear();auth.clear()
        if backend is not None:backend.close()
    output=Path(args.receipt).resolve()
    with output.open("xb") as stream:stream.write((json.dumps(observation,sort_keys=True)+"\n").encode())
    print(json.dumps({"status":observation["status"],"receipt":str(output),
                      "whole_GET_SHA":observation.get("whole_GET_SHA"),
                      "http_code":observation.get("http_code"),"service_code":observation.get("service_code"),
                      "PUT_calls":0,"Create_calls":0}))
    return 0 if observation["status"] in ("present_exact_match","explicit_404_absent") else 2


if __name__=="__main__":raise SystemExit(main())
