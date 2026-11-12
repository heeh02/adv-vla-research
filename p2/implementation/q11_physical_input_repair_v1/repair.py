"""Preserve qualified semantic references when adding the model-free runtime.

Metadata only. No model, simulator, native arrays, network or execution entry.
The already submitted physical package is immutable and is never edited here.
"""
import copy


def need(ok, code):
    if not ok:
        raise ValueError(code)


def transport(row):
    return {k: row[k] for k in ("path", "uri", "bytes", "sha256")}


def assemble(qualified, attempted):
    need(qualified["schema"] == "paper2-q11-replay-input-qa-inputs-v1" and
         attempted["schema"] == "paper2-q11-cached-physical-inputs-v1",
         "exact_input_schema_pair")
    good = {r["path"]: r for r in qualified["input_objects"]}
    all_rows = {r["path"]: r for r in attempted["input_objects"]}
    need(len(good) == len(qualified["input_objects"]) == 228 and
         len(all_rows) == len(attempted["input_objects"]) == 243, "exact_unique_input_counts")
    need(set(all_rows)-set(good) == set(qualified["pinned_runtime_paths"]) and
         len(qualified["pinned_runtime_paths"]) == 15, "only_fifteen_existing_runtime_additions")
    need(all(name in all_rows and transport(row) == transport(all_rows[name])
             for name, row in good.items()), "same_exact_qualified_data_objects")
    result = copy.deepcopy(attempted)
    result["input_objects"] = [copy.deepcopy(good.get(row["path"], row))
                               for row in attempted["input_objects"]]
    need([transport(r) for r in result["input_objects"]] ==
         [transport(r) for r in attempted["input_objects"]], "no_object_or_order_change")
    refs = {r["path"]: r for r in result["input_objects"]}
    for arm, row in qualified["cached_command_paths"].items():
        for key in ("compact", "checks"):
            ref = refs[row[key]]
            need(ref == good[row[key]] and ref["logical_path"].startswith("ledger/"),
                 "preserve_full_qualified_semantic_reference")
    result["status"] = "METADATA_REPAIR_NOT_EXECUTED"
    result["assembly_fix"] = {
        "rule": "Keep every qualified data reference verbatim; append only exact runtime additions.",
        "old_qualified_input_count": 228, "runtime_additions": 15,
        "same_transport_objects_and_order": True, "source_thresholds_changed": False,
        "new_environment_executed": False}
    return result
