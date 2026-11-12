"""Build one immutable record per selected Plus variant for five-arm runs."""
import hashlib
import json
from pathlib import Path


def read_frame(args, shard, C, enrich, output, classification):
    raw = Path(args.frame).read_bytes()
    C.need(hashlib.sha256(raw).hexdigest() == shard["frame_sha256"], "frozen frame identity")
    frame = json.loads(raw)
    mraw = Path(args.runtime_file_manifest).read_bytes()
    expected = frame["runtime_file_manifest"]
    C.need(len(mraw) == expected["bytes"] and hashlib.sha256(mraw).hexdigest() == expected["sha256"],
           "original runtime member manifest identity")
    manifest = {}
    for line in mraw.splitlines():
        if line.strip():
            row = json.loads(line)
            C.need(row["path"] not in manifest, "unique runtime members")
            manifest[row["path"]] = row
    idx = {row["row_id"]: row for row in frame["rows"]}
    chosen = shard["case_rows"]
    C.need(len(chosen) == len(set(chosen)) and all(row_id in idx for row_id in chosen),
           "frozen unique selected frame IDs")
    # The frozen FRAME-v1 currently carries route names while the restored
    # runtime manifest is an external member list.  The isolated package
    # therefore carries a 268-row content-addressed manifest.  Use it only
    # to fill the route byte identities; never select or replace rows here.
    manifest_rows = {}
    manifest_name = shard.get("variant_manifest")
    if manifest_name:
        candidate = Path(getattr(args, "shard", "")).resolve().parent / manifest_name
        if candidate.is_file():
            variant_manifest = json.loads(candidate.read_text())
            C.need(variant_manifest.get("row_count") == len(variant_manifest.get("rows", [])),
                   "variant manifest row count")
            C.need(variant_manifest.get("source_frame", {}).get("sha256") == shard["frame_sha256"],
                   "variant manifest frame binding")
            manifest_rows = {row["row_id"]: row for row in variant_manifest["rows"]}
            C.need(set(chosen).issubset(manifest_rows), "selected row absent from variant manifest")
    result = []
    for row_id in chosen:
        row = idx[row_id]
        clean = row["resource_routes"]["clean_BDDL"]
        bound = manifest_rows.get(row_id)
        if bound is not None:
            C.need(bound["variant_BDDL"]["member"] == row["resource_routes"]["variant_BDDL"]["member"],
                   "variant BDDL route drift")
            C.need(bound["clean_BDDL"]["member"] == clean["member"], "clean BDDL route drift")
            C.need(bound["official_init_library"]["member"] == row["resource_routes"]["official_init_library"]["member"],
                   "official init route drift")
            clean_id = {key: bound["clean_BDDL"][key] for key in ("bytes", "sha256")}
        else:
            clean_id = clean["known_member_identity"]
        C.need(isinstance(clean_id, dict) and clean_id.get("bytes", 0) > 0 and len(clean_id.get("sha256", "")) == 64,
               "clean BDDL identity required")
        clean_member = clean["member"]
        C.need(clean_member.startswith("official/"), "clean BDDL member must be official")
        contract = {
            "id": row_id + ":actual-base-source",
            "goal_binding_id": "goal:" + clean_id["sha256"],
            "base_bddl": {
                "relative_to_plus_root": clean_member.removeprefix("official/"),
                "bytes": clean_id["bytes"],
                "sha256": clean_id["sha256"],
            },
        }
        case = C.make_case(row["suite"], row["official_task_id_zero_based"], classification)
        C.need(case["classification_id"] == row["official_classification_id_one_based"] and
               case["task_name"] == row["official_variant_name"], "frame official identity")
        identity = {
            "population_id": shard["population_id"],
            "suite": row["suite"],
            "family": row["official_category"],
            "task": row["canonical_parent_task_candidate"]["family_id"],
            "case_id": row_id,
            "pair_id": row_id,
        }
        result.append({
            "identity": identity,
            "official_case": case,
            "reference_contract": contract,
            "goal_binding_id": contract["goal_binding_id"],
            "matched_prompt": None,
            "frame_row": row,
            "exposure": row.get("exposure"),
            "difficulty": row.get("official_difficulty_level"),
            "noise": {"parent_family_id": identity["task"], "noise_label": 0,
                      "protocol": "plus-shared-policy-noise-v1"},
        })
    return result, manifest

