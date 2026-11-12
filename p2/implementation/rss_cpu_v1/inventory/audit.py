"""Exact existing CDEV initialization audit. CPU only; no network or model imports.

Production entry: validate(stage, refs, check). ``refs`` is INPUTS-EXTRA-ROOTS
or its input_objects list. Real payloads may only be decoded on Linux. A caller
must preserve any exception as an unverified input, never infer four clusters.
"""
from pathlib import Path
import hashlib
import json
import re
import sys
import zipfile
import numpy as np

ROOTS = tuple(f"SDRU-CDEV-T0{t}-R0{r}" for t in range(2) for r in range(2))
KINDS = ("parent_summary", "initial_vector")
MAX_OBJECT = 256 * 1024
MAX_EXPANDED = 256 * 1024


def need(ok, code):
    if not ok:
        raise ValueError(code)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def decode_metadata(value):
    if isinstance(value, dict):
        need("__array__" not in value, "initial_metadata_must_not_embed_array")
        if "__sequence__" in value:
            need(value["__sequence__"] in ("list", "tuple"), "known_metadata_sequence")
            return [decode_metadata(v) for v in value["values"]]
        return {k: decode_metadata(v) for k, v in value.items()}
    if isinstance(value, list):
        return [decode_metadata(v) for v in value]
    return value


def typed_identity(value):
    header = {"shape": list(value.shape), "dtype": value.dtype.str, "order": "C"}
    payload = value.tobytes(order="C")
    return {**header, "bytes": len(payload), "sha256": sha(canonical(header) + b"\0" + payload),
            "hash_domain": "plus-policy-native-array-v1", "raw_sha256": sha(payload)}


def exact_file(stage, ref, check):
    check()
    name = ref.get("stage_name", ref["path"])
    need(isinstance(name, str) and name not in ("", ".", "..") and "/" not in name and
         not any(c in name for c in "\\\r\n\0"), "flat_exact_staged_name")
    path = Path(stage) / name
    need(path.is_file() and not path.is_symlink() and path.parent.resolve() == Path(stage).resolve(),
         "owned_regular_input_file")
    need(type(ref["bytes"]) is int and 0 < ref["bytes"] <= MAX_OBJECT and
         re.fullmatch("[0-9a-f]{64}", ref["sha256"]), "bounded_input_identity")
    need(path.stat().st_size == ref["bytes"], "root_input_size")
    raw = path.read_bytes()
    need(len(raw) == ref["bytes"] and sha(raw) == ref["sha256"], "root_input_SHA")
    check()
    return path, raw


def initial_vector(path, ref, check):
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        need(len(names) == len(set(names)) == 2 and
             set(names) == {"__schema__.npy", "root/arrays/initial_vector.npy"}, "exact_initial_NPZ_roster")
        need(sum(r.file_size for r in archive.infolist()) <= MAX_EXPANDED and
             archive.getinfo("__schema__.npy").file_size <= 128 * 1024 and
             not any(r.flag_bits & 1 or r.is_dir() for r in archive.infolist()), "bounded_initial_NPZ")
    with np.load(path, allow_pickle=False) as z:
        raw = z["__schema__"]
        need(raw.dtype == np.uint8 and raw.ndim == 1, "initial_schema_uint8")
        tree = json.loads(raw.tobytes())
        need(set(tree["arrays"]) == {"initial_vector"}, "single_initial_vector_leaf")
        desc = tree["arrays"]["initial_vector"]
        need(desc["__array__"] == "root/arrays/initial_vector", "fixed_initial_payload_key")
        value = np.asarray(z[desc["__array__"]])
        need(value.dtype == np.dtype("<f8") and value.ndim == 1 and value.size in (84, 123) and
             np.isfinite(value).all(), "original_finite_float64_initial_shape")
        actual_desc = {"shape": list(value.shape), "logical_dtype": "float64", "storage_dtype": "float64",
                       "bytes": value.nbytes, "payload_sha256": sha(value.tobytes())}
        need({k: desc[k] for k in actual_desc} == actual_desc, "initial_descriptor_payload_SHA")
        identity = typed_identity(value)
        need(identity == ref["expected"]["native_initial_vector"], "initial_typed_and_raw_SHA")
        meta = decode_metadata(tree["metadata"])
    check()
    return identity, meta


def check_selection(selection, ref):
    exp = ref["expected"]
    need(selection["root_trajectory_id"] == ref["root_id"], "initial_root_identity")
    for name in ("original_library_index", "unique_vector_rank", "source_library", "native_initial_vector"):
        need(selection[name] == exp[name], "initial_selection_" + name)
    need(type(selection["original_library_index"]) is int and selection["original_library_index"] in (0, 1),
         "original_index_not_noise_label")
    need(selection["rule"] == "first_two_typed_byte_distinct_vectors_in_original_library_order" and
         selection["old_optics_case_is_not_current_state"] is True, "actual_distinct_selection_rule")


def validate(stage, refs, check=lambda: None, *, synthetic=False):
    """Verify eight exact files and return small cluster proofs without payloads.

    No roots are silently dropped. Input/schema/source errors raise ValueError.
    Different initial bytes establish distinct clusters, not IID sampling.
    """
    need(synthetic is True or sys.platform.startswith("linux"), "real_initial_array_decode_cloud_Linux_only")
    if isinstance(refs, dict):
        refs = refs["input_objects"]
    refs = list(refs)
    keys = [(r["root_id"], r["kind"]) for r in refs]
    need(len(keys) == len(set(keys)) == 8 and set(keys) == {(r, k) for r in ROOTS for k in KINDS},
         "exact_four_root_two_object_support")
    need(len({r.get("stage_name", r["path"]) for r in refs}) == 8, "unique_staged_names")
    by = dict(zip(keys, refs))
    rows = []
    for root in ROOTS:
        sref, vref = by[root, "parent_summary"], by[root, "initial_vector"]
        need(sref["expected"] == vref["expected"], "same_expected_root_source")
        _, raw = exact_file(stage, sref, check)
        summary = json.loads(raw)
        need(summary["root_id"] == root and summary["run_id"] == sref["expected"]["source_run_id"] and
             summary["root_complete"] is True, "actual_complete_parent_root_source")
        selection = summary["worker_result"]["actual_initial_selection"]
        check_selection(selection, sref)
        path, _ = exact_file(stage, vref, check)
        identity, vector_meta = initial_vector(path, vref, check)
        check_selection(vector_meta, vref)
        need(vector_meta == selection, "initial_event_matches_actual_parent_summary")
        expected = sref["expected"]
        rows.append({"root_id": root, "task_component_id": expected["task_component_id"],
                     "parent_task_id": expected["parent_task_id"], "source_run_id": summary["run_id"],
                     "library": selection["source_library"], "original_library_index": selection["original_library_index"],
                     "native_initial_vector": identity,
                     "cluster_key": [expected["task_component_id"], selection["source_library"]["sha256"],
                                     selection["original_library_index"], identity["sha256"]],
                     "summary_ref": {k: sref[k] for k in ("uri", "bytes", "sha256")},
                     "vector_ref": {k: vref[k] for k in ("uri", "bytes", "sha256")},
                     "status": "ACTUAL_INITIAL_PAYLOAD_AND_METADATA_VERIFIED"})
    tasks = sorted({r["task_component_id"] for r in rows})
    need(len(tasks) == 2, "exact_two_DEV_tasks")
    for task in tasks:
        part = [r for r in rows if r["task_component_id"] == task]
        need(len(part) == 2 and len({r["library"]["sha256"] for r in part}) == 1 and
             {r["original_library_index"] for r in part} == {0, 1} and
             len({r["native_initial_vector"]["sha256"] for r in part}) == 2 and
             len({r["native_initial_vector"]["raw_sha256"] for r in part}) == 2,
             "two_actual_distinct_initial_vectors_per_task")
    return {"schema": "rss-CDEV-initial-cluster-audit-v1", "status": "COMPLETE", "input_qualified": True,
            "synthetic": synthetic, "input_objects_verified": 8, "input_body_bytes": sum(r["bytes"] for r in refs),
            "clusters": rows, "distinct_initialization_clusters": 4, "task_count": 2,
            "statistical_independence_claim": False, "power_sufficiency_claim": False,
            "sampling_scope": "Two deterministic first byte-distinct library vectors per task; not IID draws.",
            "model_calls": 0, "env_steps": 0, "GPU": 0, "array_payloads_returned": False}


def audit(stage, inputs_manifest, check=lambda: None, *, synthetic=False):
    """Driver seam over a combined input manifest; only inventory refs decoded.

    DEV695 selectors/labels remain the retained dev_loader's responsibility.
    Optional CONF PVP evidence is five small JSON objects, not neural arrays.
    """
    refs = inputs_manifest["input_objects"] if isinstance(inputs_manifest, dict) else list(inputs_manifest)
    root_refs = [r for r in refs if r.get("kind") in KINDS]
    result = validate(stage, root_refs, check, synthetic=synthetic)
    groups = []
    for row in result["clusters"]:
        groups.append({"root_id": row["root_id"], "cluster_id": sha(canonical(row["cluster_key"])),
                       "task_id": row["root_id"].split("-")[2], "task_component_id": row["task_component_id"],
                       "library_sha256": row["library"]["sha256"], "init_index": row["original_library_index"],
                       "init_state_sha256": row["native_initial_vector"]["sha256"],
                       "source_ref": {"parent_summary": row["summary_ref"], "initial_vector": row["vector_ref"]}})
    pvp_refs = [r for r in refs if r.get("kind") in ("conf_pvp_result", "conf_pvp_axes")]
    pulse = {"DEV_actual_selector_checked_here": False,
             "DEV_selector_authority": "root dev_loader must emit192 actual persistent selector proofs",
             "DEV_pulse_reuse_claimed": False, "CONF_metadata_checked": False,
             "other_unindexed_runs": "UNKNOWN_NOT_PROVEN_ABSENT"}
    if pvp_refs:
        need(len(pvp_refs) == 5 and sum(r["kind"] == "conf_pvp_result" for r in pvp_refs) == 1,
             "exact_five_CONF_metadata_objects")
        coverage_keys, axis_keys, shards = set(), set(), set()
        for ref in pvp_refs:
            _, raw = exact_file(stage, ref, check)
            value = json.loads(raw)
            if ref["kind"] == "conf_pvp_result":
                need(value["schema"] == "PVP-existing-Gstar-CPU-readout-v1" and value["complete"] is True and
                     value["synthetic"] is False and value["planned_queries"] == value["completed_queries"] == 24 and
                     not value["errors"] and value["new_fit"] is False and
                     value["new_model_calls"] == value["new_environment_steps"] == 0,
                     "actual_complete_prior_CONF_PVP_result")
                for item in value["coverage"]:
                    key = (item["root_id"], item["query_index"])
                    need(key not in coverage_keys and item["status"] == "COMPLETE" and
                         item["same_j5_and_x6"] is True and item["same_j6_L1_L9_QP"] is True,
                         "complete_unique_CONF_PVP_coverage")
                    coverage_keys.add(key)
            else:
                shard = ref["expected_shard"]
                need(shard in {"CCONF-T06", "CCONF-T07", "CCONF-T08", "CCONF-T09"} and shard not in shards,
                     "four_unique_CONF_axis_shards")
                shards.add(shard)
                need(value["roles"] == ["baseline_none_1", "shared_pulse", "shared_persistent"] and
                     value["adapter_axes"] == "role/action_slot/7command_dim" and
                     value["first5_proposed_not_issued"] is True and len(value["keys"]) == 6,
                     "CONF_only_three_role_endpoint_axes")
                for root, query in value["keys"]:
                    key = (root, query)
                    need(root.startswith("SDRU-" + shard + "-") and query in (11, 16, 22) and key not in axis_keys,
                         "CONF_axis_query_identity")
                    axis_keys.add(key)
        need(len(axis_keys) == len(coverage_keys) == 24 and axis_keys == coverage_keys,
             "CONF_axes_match_complete_actual_coverage")
        pulse.update(CONF_metadata_checked=True, CONF_shared_pulse_anchors=24,
                     CONF_random_matched_pulse_in_this_exact_campaign=0,
                     CONF_status="CONF_ONLY_SHARED_PULSE_VERIFIED_FROM_PRIOR_CPU_RESULT_AND_AXES",
                     CONF_raw_neural_payloads_redecoded=False, CONF_relabel_as_DEV_allowed=False,
                     no_global_random_pulse_absence_claim=True)
    selected = root_refs + pvp_refs
    receipts = [{k: r[k] for k in ("kind", "uri", "bytes", "sha256")} for r in selected]
    return {"schema": "rss-CPU-inventory-audit-v1", "status": "COMPLETE", "input_qualified": True,
            "synthetic": synthetic, "groups": groups,
            "independence": {"distinct_native_initialization_clusters": 4, "tasks": 2,
                             "IID_sampling_claim": False, "reliable_power_claim": False,
                             "selection": "first two typed-byte-distinct vectors per task, deterministic",
                             "all_query_arm_noise_rows_of_cluster_stay_in_same_fold": True},
            "pulse_inventory": pulse, "root_payload_proof": result, "receipts": receipts,
            "inventory_objects_verified": len(selected),
            "inventory_body_bytes": sum(r["bytes"] for r in selected),
            "DEV695_objects_redecoded_here": 0, "model_calls": 0, "env_steps": 0, "GPU": 0}
