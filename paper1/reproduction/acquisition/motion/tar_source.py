"""Read only the three diagnostic's small JSON members; no job or raw arrays."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import tarfile
import time

HERE = Path(__file__).resolve().parent
Q = HERE.parents[1]
DIAG = Q / "reference-lifecycle-diagnostic-v1"
IMAGE_AUDIT = Q / "reference-image-audit-cpu-v2/analysis_worker.py"
ARMS = ("A-reference-first", "B-variant-first-defer-close", "C-variant-first-close-before-reference")
RUN = "jinwen-adv-vla-plus-reference-lifecycle-20261010-v1"
MAX_JSON = 256 * 1024
MAX_TOTAL = 1024 * 1024


def need(ok, reason):
    if not ok:
        raise RuntimeError(reason)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def reference_scope():
    batch = json.loads((DIAG / "BATCH-v1.json").read_bytes())
    need(batch["batch_id"] == "jinwen-adv-vla-plus-reference-lifecycle-batch-20261010-v1"
         and len(batch["runs"]) == 1 and batch["runs"][0]["run_id"] == RUN, "one_frozen_diagnostic")
    return batch["runs"][0]["data_output_prefix"]


def result(scope):
    path = DIAG / "run-v1/background-spatial/ACTUAL_RESULT-v1.json"
    raw = path.read_bytes()
    need(len(raw) < 2 * 1024 * 1024, "small_actual_result")
    value = json.loads(raw)
    need(value["run_id"] == RUN and value["worker_complete"] is True
         and value["final_archive_closed"] is True and value["failure"] is None
         and len(value["incremental"]["commits"]) == 3, "actual_completed_three_arm_result")
    ref = value["final_evidence_index"]
    need(ref["uri"] == scope + "RUNTIME_RESTORE_INDEX-v1.json" and ref["bytes"] < MAX_JSON,
         "exact_final_index")
    return value, ref, {"bytes":len(raw),"sha256":sha(raw)}


def load_range():
    spec = importlib.util.spec_from_file_location("existing_exact_case_range_reader", IMAGE_AUDIT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.RangeReader


def build_map(index, rows, scope):
    need(index["schema"] == "plus-runtime-restoration-capsule-v1"
         and index["archive_includes_models"] is False
         and len(index["file_manifests"]) == 1
         and len(rows) == index["members"], "final_archive_schema_and_count")
    need([r["path"] for r in rows] == sorted(r["path"] for r in rows)
         and len({r["path"] for r in rows}) == len(rows), "sorted_unique_manifest")
    parts = index["parts"]
    need(parts and all(p["uri"].startswith(scope + "capsule/parts/") for p in parts), "only_diagnostic_parts")
    cursor = 0
    for part in parts:
        need(part["offset"] == cursor and 0 < part["bytes"] <= 64 * 1024 * 1024, "contiguous_bounded_parts")
        cursor += part["bytes"]
    need(cursor == index["archive"]["bytes"], "all_parts_archive_size")
    cursor, mapped = 0, {}
    for row in rows:
        name = row["path"]
        need(isinstance(name, str) and name and ".." not in name.split("/") and not name.startswith("/")
             and isinstance(row["bytes"], int) and row["bytes"] >= 0, "safe_member")
        info = tarfile.TarInfo(name)
        info.size = row["bytes"]
        info.mode = row["mode"]
        info.mtime = info.uid = info.gid = 0
        header = info.tobuf(format=tarfile.PAX_FORMAT)
        mapped[name] = {"row": row, "header": header, "header_offset": cursor,
                        "payload_offset": cursor + len(header)}
        cursor += len(header) + math.ceil(row["bytes"] / 512) * 512
    need(math.ceil((cursor + 1024) / 10240) * 10240 == index["archive"]["bytes"],
         "deterministic_tar_layout_matches_verified_archive")
    return mapped


def selected(mapped):
    names = []
    for arm in ARMS:
        for leaf in ("Q0-BINDINGS-v1.json", "FIRST-STEP-v1.json"):
            name = f"collection/diagnostic/{arm}/{leaf}"
            need(name in mapped, "required_diagnostic_small_member_missing")
            names.append(name)
    prompts = [name for name in mapped if name.endswith("/MATCHED-PROMPT-v1.json")]
    names += prompts
    need(len(prompts) <= 3 and len(names) <= 9 and all(mapped[n]["row"]["bytes"] <= MAX_JSON for n in names)
         and sum(mapped[n]["row"]["bytes"] for n in names) <= MAX_TOTAL,
         "only_bounded_small_JSON_selection")
    return names, prompts


def read_one(reader, mapped, name):
    item = mapped[name]
    header = reader.read(item["header_offset"], len(item["header"]))
    need(header == item["header"], "actual_tar_header_bytes_differ")
    raw = reader.read(item["payload_offset"], item["row"]["bytes"])
    need(len(raw) == item["row"]["bytes"] and sha(raw) == item["row"]["sha256"],
         "actual_member_bytes_SHA_differ")
    return raw, {"source_member": name, "bytes": len(raw), "sha256": sha(raw),
                 "header_offset": item["header_offset"], "header_sha256": sha(header),
                 "actual_header_verified": True}


def write_exact(out, name, raw):
    path = out / name
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        need(path.read_bytes() == raw, "existing_readback_differs")
    else:
        with path.open("xb") as stream:
            stream.write(raw)


def retrieve(backend, scope, ref, out):
    # The root owns auth; this is a single read-only STS for the exact run prefix.
    env = backend.sts([{"service":"bce:bos","region":"bj","effect":"Allow",
                        "resource":[scope.removeprefix("bos://") + "*"],"permission":["READ"]}], 900)
    try:
        backend.grants[scope] = {key: env[key] for key in backend_module_auth_names()}
        got = backend.get(scope, "RUNTIME_RESTORE_INDEX-v1.json", expected=ref, limit=MAX_JSON)
        need(got is not None, "final_index_missing")
        index, index_ref = got
        mref = index["file_manifests"][0]
        need(mref["uri"] == scope + mref["relative"] and mref["bytes"] <= 2 * 1024 * 1024,
             "exact_small_manifest")
        got = backend.get(scope, mref["relative"], expected=mref, limit=2 * 1024 * 1024)
        need(got is not None, "final_manifest_missing")
        manifest_text, manifest_ref = got
        rows = [json.loads(line) for line in manifest_text.splitlines() if line.strip()]
        mapped = build_map(index, rows, scope)
        names, prompts = selected(mapped)
        with backend.old._environment(env):
            reader = load_range()(backend.Writer(scope, time.time() + 420), index["parts"], time.time() + 420)
            proofs, parsed = [], {}
            for name in names:
                raw, proof = read_one(reader, mapped, name)
                write_exact(out, name, raw)
                proofs.append(proof)
                parsed[name] = json.loads(raw)
    finally:
        env.clear()
    q0 = [parsed[f"collection/diagnostic/{arm}/Q0-BINDINGS-v1.json"] for arm in ARMS]
    steps = [parsed[f"collection/diagnostic/{arm}/FIRST-STEP-v1.json"] for arm in ARMS]
    equality = {"q0_all_three_exact": q0[0] == q0[1] == q0[2],
                "first_step_raw_and_adapter_all_three_exact_by_side": {
                    side: all(steps[0][side][kind] == steps[i][side][kind]
                              for kind in ("raw", "adapter") for i in (1, 2))
                    for side in ("reference", "treatment")},
                "matched_prompt_members_in_final_archive": len(prompts),
                "prompt_equality_established_by_this_read": bool(prompts) and len(prompts) == 3
                    and parsed[prompts[0]] == parsed[prompts[1]] == parsed[prompts[2]]}
    receipt = {"schema":"plus-three-arm-small-json-readback-v1", "run_id":RUN,
               "final_index":index_ref,"final_manifest":manifest_ref,"selected_members":proofs,
               "actual_range_requests":reader.range_requests,"comparison":equality,
               "raw_NPZ_or_model_to_Mac":False,"cloud_operations":"metadata_GET_and_exact_small_JSON_Range_only"}
    write_exact(out, "READBACK-v1.json", (json.dumps(receipt, sort_keys=True, indent=2) + "\n").encode())
    return receipt


def backend_module_auth_names():
    return ("ONE_VLA_BOS_STS_ACCESS_KEY_ID", "ONE_VLA_BOS_STS_SECRET_ACCESS_KEY",
            "ONE_VLA_BOS_STS_SESSION_TOKEN", "ONE_VLA_BOS_STS_EXPIRATION_EPOCH",
            "ONE_VLA_BOS_ENDPOINT")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    scope = reference_scope()
    _, ref, local_ref = result(scope)
    if not args.execute:
        print(json.dumps({"status":"dry_small_json_readback", "run_id":RUN,
                          "actual_result":local_ref,"index":ref,"API_calls":0}))
        return 0
    # Reuse the diagnostic root's proven RAM auth, proxy and metadata Backend.
    sys.path.insert(0, str(DIAG))
    import root_execute as root
    root.verify()
    root.verify_host_bindings()
    auth = {}
    backend = None
    from platform_reuse import load
    from proxy_transport import loopback_proxy
    try:
        with loopback_proxy():
            auth = load().read_RAM_auth()
            backend = root.backend_module.Backend(auth, None)
            receipt = retrieve(backend, scope, ref, HERE / "lifecycle-readback-v1")
            print(json.dumps({"status":"small_JSON_readback_complete", "run_id":RUN,
                              "members":len(receipt["selected_members"]),"comparison":receipt["comparison"]}))
            return 0
    except BaseException as error:
        print(json.dumps({"status":"small_JSON_readback_attention", "error_type":type(error).__name__}))
        return 2
    finally:
        if backend is not None:
            backend.close()
        auth.clear()


if __name__ == "__main__":
    raise SystemExit(main())
