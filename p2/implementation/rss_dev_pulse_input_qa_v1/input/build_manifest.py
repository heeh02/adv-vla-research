"""Build a bounded eight-snapshot readset from local frozen metadata only."""
import hashlib
import json
from pathlib import Path
import tarfile

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[1]
RESEARCH = Path("/Users/researcher/.codex/worktrees/adv-vla-research-20260924/Adv-VLA-π0.5/docs/experiments/state-dependent-response-use-20261013-v1")
KEYS = tuple((f"SDRU-CDEV-T0{t}-R0{r}", q)
             for t in range(2) for r in range(2) for q in (11, 22))


def ident(path):
    raw = Path(path).read_bytes()
    return {"path": str(path), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def build():
    package = RESEARCH/"continuous/l9-intervention-root-v3/prepared-gpu-v1/SOURCE_PACKAGE-v1.tar.gz"
    package_ref = ident(package)
    assert package_ref["sha256"] == "4bffe213a5cdeefcc2f804d5d98faccd6259ea9db26371029214f7ab40f5ab4c"
    with tarfile.open(package, "r:gz") as tar:
        parent_bytes = tar.extractfile("metadata/L9-INPUT-PLAN-v1.json").read()
    parent = json.loads(parent_bytes)
    native_path = BASE/"rss_cpu_v2/inventory/DEV-READ-PLAN-original.json"
    native = json.loads(native_path.read_text())
    candidate_path = BASE.parent/"rss-cpu-goal-20261020-v1/MINIMAL-NEXT-COLLECTION-CANDIDATE-v1.json"
    candidate = json.loads(candidate_path.read_text())
    index_path = RESEARCH/"continuous/l9-intervention-root-v3/run-v1/archive-index-v2/INDEX-v1.json"
    index = json.loads(index_path.read_text())
    selected = {(r["root_id"], r["query_index"], r["arm"]): r
                for r in candidate["exact_missing_labels"]}
    assert set((r, q) for r, q, _ in selected) == set(KEYS)
    refs = {}

    def add(ref, codec):
        ref = dict(ref)
        ref.setdefault("path", ref["uri"].rsplit("/", 1)[1])
        if ref["uri"] in refs:
            old = refs[ref["uri"]]
            assert all(old[k] == ref[k] for k in ("bytes", "sha256"))
            assert old["codec"] == codec
        else:
            refs[ref["uri"]] = {
                "uri": ref["uri"], "bytes": ref["bytes"], "sha256": ref["sha256"],
                "logical_path": ref["path"], "codec": codec,
            }
        return ref

    roots = []
    archives = []
    for root in parent["roots"]:
        reduced = {k: root[k] for k in ("root_id", "prompt", "binding", "actual_initial_selection")}
        reduced["proofs"] = {k: add(v, "plain_json") for k, v in root["proofs"].items()}
        pa = root["prompt_archive"]
        reduced["prompt_archive"] = {
            "ready": add(pa["ready"], "archive_control"),
            "ack": add(pa["ack"], "archive_control"),
            "members": [add(v, "parent_values_v1") for v in pa["members"]],
        }
        reduced["queries"] = []
        for query in root["queries"]:
            if (root["root_id"], query["query_index"]) not in KEYS:
                continue
            reduced["queries"].append({
                "query_index": query["query_index"],
                "ready": add(query["ready"], "archive_control"),
                "ack": add(query["ack"], "archive_control"),
                # Retain original tiny RNG audits; actual metadata selects the
                # two diagnostic_fixed0 C/T pairs, not guessed file numbers.
                "members": [add(v, "parent_values_v1") for v in query["members"]],
            })
        roots.append(reduced)
    native_map = {r["uri"]: r for r in native["members"]}
    snapshots = []
    for root, q in KEYS:
        row = selected[root, q, "shared"]
        ss = {
            "root_id": root, "query_index": q,
            "original_parent_source": row["parent_source"],
            "pulse_label_identities": {a: {
                "input_identity": selected[root, q, a]["input_identity"],
                "edit_identity": selected[root, q, a]["edit_identity"]}
                for a in ("shared", "random_matched")},
        }
        records = {
            "audit": row["input_audit_source"],
            "edits": row["frozen_edit_source"],
            "natural_T": row["native_T_baseline"],
        }
        for role in ("natural_C", "effect_shared", "effect_random_matched"):
            matches = [r for r in native["members"]
                       if r.get("root_id") == root and r.get("query_index") == q
                       and r.get("role") == role]
            assert len(matches) == 1
            records[role] = matches[0]
        ss["records"] = {}
        for role, ref in records.items():
            original = native_map[ref["uri"]]
            assert all(original[k] == ref[k] for k in ("bytes", "sha256", "path"))
            ss["records"][role] = add(original, "native_arrays_v1")
        snapshots.append(ss)
    native_uris = {r["uri"] for s in snapshots for r in s["records"].values()}
    for archive in index["archives"]:
        members = [r for r in archive["members"] if r["uri"] in native_uris]
        if not members:
            continue
        ready = dict(archive["ready"])
        ready["object_key"] = ready["uri"].split("/", 3)[3].split(
            "jinwen-state-response-cont-20261013-l9-r3/", 1)[1]
        # The index's IPC acknowledgement points to a separately verified
        # durable acknowledgement. Use the actual durable body for ArchiveQuery.
        local = index_path.parent/f"{index['archives'].index(archive):02d}-ACK.json"
        ack_document = json.loads(local.read_text())
        prefix = ready["uri"].split("queries/", 1)[0]
        ack = {"uri": prefix+ack_document["durable_ack_object_key"],
               "object_key": ack_document["durable_ack_object_key"],
               **{k: ack_document["durable_ack_verification"][k] for k in ("bytes", "sha256")}}
        archives.append({
            "binding": archive["binding"], "role": archive["role"],
            "ready": add(ready, "archive_control"), "ack": add(ack, "archive_control"),
            "members": [dict(r) for r in members],
        })
    assert {r["uri"] for a in archives for r in a["members"]} == native_uris
    objects = []
    for i, ref in enumerate(sorted(refs.values(), key=lambda r: r["uri"])):
        objects.append({**ref, "path": f"input-{i:04d}"+Path(ref["logical_path"]).suffix})
    total = sum(r["bytes"] for r in objects)
    assert len(objects) <= 800 and total <= 512*1024**2
    return {
        "schema": "rss-dev-pulse-input-readset-v1",
        "status": "CANDIDATE_NOT_CLOUD_QUALIFIED_OR_GPU_AUTHORITY",
        "snapshots": snapshots, "roots": roots, "native_archives": archives,
        "input_objects": objects, "input_object_count": len(objects),
        "input_bytes": total, "limits": {"objects": 800, "bytes": 512*1024**2},
        "source_binding": native["source_binding"],
        "source_evidence": {
            "original_native_plan": ident(native_path), "candidate": ident(candidate_path),
            "parent_package": package_ref, "native_archive_index": ident(index_path),
            "parent_plan_member": {
                "path": "metadata/L9-INPUT-PLAN-v1.json", "bytes": len(parent_bytes),
                "sha256": hashlib.sha256(parent_bytes).hexdigest()},
        },
        "cloud_reads": 0, "model_calls": 0, "environment_steps": 0,
        "model_weights_in_readset": False,
        "scope": "Eight saved DEV observations and frozen native payloads; not physics restore or new pulse labels",
    }


if __name__ == "__main__":
    result = build()
    (HERE/"INPUTS.json").write_text(json.dumps(result, indent=2, ensure_ascii=False)+"\n")
    print(json.dumps({"objects": result["input_object_count"], "bytes": result["input_bytes"]}))
