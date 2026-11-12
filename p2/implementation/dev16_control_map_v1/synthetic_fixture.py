"""Fabricated small command-only typed NPZ; no real arrays loaded."""
import io
import json
from pathlib import Path
import numpy as np
import adapter as D
import binding as B
import codec as C


def typed_bytes(meta, array):
    leaf = C.from_array(np.asarray(array, dtype=np.float64))
    tree = {"metadata": meta, "arrays": {"adapter7": {"__array__": "leaf_0", **leaf.identity()}}}
    buf = io.BytesIO()
    np.savez_compressed(buf, __schema__=np.frombuffer(json.dumps(tree, sort_keys=True).encode(), np.uint8),
                        leaf_0=leaf.storage)
    return buf.getvalue()


def stage_fixture(stage):
    stage = Path(stage)
    stage.mkdir()
    plan = C.json_read((D.HERE / "INPUTS.json").read_bytes())["native72_plan_entries"]
    refs, science, evidence, snapshots, native_names, child_names = [], [], [], [], [], []
    root_proof = {"schema": "paper2-dev16-all-root-control-binding-v1",
                  "roots": [{"root_id": root, "controller_mapping_qualified": True,
                             "selected_controller": "OSC_POSE", "selected_gripper": "PandaGripper"}
                            for root in sorted({k[0] for k in B.KEYS})], "snapshots": []}
    def save(name, raw, parent=None, child=True):
        (stage / name).write_bytes(raw)
        ref = {"path": name, "logical_path": name, "uri": "synthetic://" + name, **D.fid(raw)}
        refs.append(ref)
        if parent is not None:
            parent.append({"path": name, **D.fid(raw), "state": "VERIFIED",
                           "receipt": {"uri": ref["uri"], **D.fid(raw)}})
            native_names.append(name)
            if parent is science and child:
                child_names.append(name)
        return ref
    def js(name, value, parent=None, child=True):
        return save(name, D.encoded(value), parent, child)
    for i, key in enumerate(B.KEYS):
        saved = f"synthetic_saved_{i}"
        inp_ref = {"path": f"reference_input_{i}.npz", "bytes": 100, "sha256": "a" * 64}
        spec = {"root_id": key[0], "query_index": key[1], "saved_input_source_identity": saved,
                "historical_query_source_identity": f"synthetic_history_{i}",
                "input_identity": f"synthetic_input_{i}", "input_reference_only": inp_ref, "new": {}}
        prows = {"root_id": key[0], "query_index": key[1], "saved_input_source_identity": saved,
                 "historical_query_source_identity": spec["historical_query_source_identity"],
                 "input_identity": spec["input_identity"],
                 "native_objects": {}}
        for arm, role in (("N", "baseline_none_1"), ("S", "shared_pulse"), ("R", "random_pulse")):
            entry = next(e for e in plan if (e["root_id"], e["query_index"], e["role"]) == (*key, role))
            spec[arm + "_entry"] = entry
            arr = np.zeros((1, 10, 7), dtype=np.float64)
            arr[0, :, :6] = np.linspace(-2, 2, 60).reshape(10, 6)
            arr[0, :, 6] = np.array([-2., -1., -0., 0., .001, .5, 1., 2., -.3, .1])
            if arm == "S":
                arr += .01
            if arm == "R":
                arr -= .02
            meta = {"schema": "rss-pulse-compact-call-v1", "entry": entry, "input_ref": inp_ref,
                    "native_validation_errors": [], "provenance": {
                        "source_identity": saved, "native_source_unchanged": True}}
            ref = save(f"call_{i}_{arm}.npz", typed_bytes(meta, arr), science)
            check = js(f"check_{i}_{arm}.json", {
                "schema": "rss-pulse-compact-event-v1", "event": "CALL_CHECKS",
                "payload": {"entry": entry, "passed": True, "output": ref}}, science)
            spec["new"][arm], spec["new"][arm + "_checks"] = ref["path"], check["path"]
            prows["native_objects"][arm] = {**ref, "checks": check}
        snapshots.append(spec)
        root_proof["snapshots"].append(prows)
    common = {}
    model = {"model_bound": True, "adapter_bound": True,
             "normalization_sha256": "synthetic_norm", "loaded_parameter_tree_sha256": "synthetic_model"}
    common["MODEL_BINDING"] = js("model.json", model, evidence)["path"]
    mr = refs[-1]
    source = {"event": "SOURCE_BOUND", "payload": {
        "source_bound": True, "model_bound": True, "adapter_bound": True, "inputs_bound": True,
        "synthetic": False, "source_manifest": {"sha256": "synthetic_source"}, "model_binding": mr}}
    common["SOURCE_BOUND_event"] = js("source.json", source, science)["path"]
    callids = [e["call_id"] for e in plan]
    lists = {k: callids for k in ("charged_calls", "completed_calls", "passed_calls", "returned_native_calls")}
    terminal = {"result": {"complete": True, "qualification_complete": True, "failures": [],
                           "observability": [{"root_id": k[0], "query_index": k[1]} for k in B.KEYS],
                           **lists}, "unexecuted_call_ids": [], "stopped_on_storage_failure": False}
    common["TERMINAL.json"] = js("terminal.json", terminal, science)["path"]
    result = {"run_id": B.RUN, "status": "DEV_PULSE_NATIVE_COMPLETE", "synthetic": False,
              "worker_complete": True, "model_binding": model, "source_manifest_sha256": "synthetic_source",
              **lists}
    common["RESULT.json"] = js("result.json", result, science)["path"]
    archive = {"files": [{"path": r["logical_path"], **{k: r[k] for k in ("bytes", "sha256")}}
                         for r in refs if r["path"] in child_names]}
    common["ARCHIVE-MANIFEST.json"] = js("archive.json", archive, science, child=False)["path"]
    runtime = {"science_archive": {"complete_for_observed_files": True, "errors": [], "objects": science},
               "runtime_evidence_archive": {"objects": evidence}}
    common["RUNTIME-RESULT.json"] = js("runtime.json", runtime)["path"]
    inputs = {"schema": "paper2-dev16-control-map-inputs-v1", "producer_run": B.RUN,
              "producer_job": "redacted-f594f0253751", "input_objects": refs,
              "input_object_count": len(refs), "input_body_bytes": sum(r["bytes"] for r in refs),
              "native72_plan_entries": plan, "common": common, "snapshots": snapshots,
              "native_archive_bound_paths": native_names, "native_child_bound_paths": child_names}
    assert len(refs) == 54
    return inputs, root_proof
