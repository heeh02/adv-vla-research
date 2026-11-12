"""Fixed existing-data command mapping; no policy, simulator or controller import."""
import argparse
import gzip
import json
import math
from pathlib import Path
import sys
import time
import traceback
import numpy as np
import codec as C
import binding as B
import analysis as A

HERE = Path(__file__).resolve().parent
RESULT = "dev16-control-map-result.json"
OUTPUTS = (RESULT, "POINTS.json.gz", "WINDOWS.json", "SUMMARY.json", "QA.json", "PROTOCOL.json")
SOURCE_FILES = (
    "adapter.py", "analysis.py", "binding.py", "codec.py", "INPUTS.json", "PROTOCOL.json",
    "CODEC-PINS.json", "CONTROL-SOURCE-PINS.json", "ALL-ROOT-CONTROL-BINDING.json",
    "pinned/retained_replication.py", "pinned/base_controller.py", "pinned/panda_gripper.py",
    "pinned/osc_pose.json", "test_control_map.py", "synthetic_fixture.py", "freeze.py", "README.md")
COUNTS = {"snapshots": 8, "native_calls": 24, "position_contrasts": 240,
          "point_comparisons": 1680, "window_contrasts": 48, "window_comparisons": 336}


def encoded(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def fid(raw):
    return {"bytes": len(raw), "sha256": C.sha(raw)}


def validate_archive(runtime, archive, refs, inputs):
    """Retained nativewrite audit's URI/size/SHA parent and child binding."""
    parent = runtime["science_archive"]
    C.need(parent["complete_for_observed_files"] is True and parent["errors"] == [],
           "complete_native_parent_archive")
    objects = parent["objects"] + runtime["runtime_evidence_archive"]["objects"]
    by_sha = {}
    for obj in objects:
        by_sha.setdefault(obj["sha256"], []).append(obj)
    for name in inputs["native_archive_bound_paths"]:
        ref = refs[name]
        hits = [o for o in by_sha.get(ref["sha256"], []) if o["bytes"] == ref["bytes"]
                and o["receipt"]["uri"] == ref["uri"]]
        C.need(len(hits) == 1 and hits[0]["state"] == "VERIFIED",
               "native_object_exact_URI_VERIFIED_parent_archive")
    child = {r["path"]: r for r in archive["files"]}
    C.need(len(child) == len(archive["files"]), "unique_native_child_archive")
    for name in inputs["native_child_bound_paths"]:
        ref = refs[name]
        C.need(ref["logical_path"] in child and all(child[ref["logical_path"]][k] == ref[k]
               for k in ("bytes", "sha256")), "native_object_exact_child_manifest")


def root_semantics(proof, spec, refs):
    C.need(proof["schema"] == "paper2-dev16-all-root-control-binding-v1", "static_binding_schema")
    roots = {r["root_id"]: r for r in proof["roots"]}
    C.need(len(roots) == len(proof["roots"]) == 4 and set(roots) == {k[0] for k in B.KEYS},
           "all4_static_root_bindings")
    snapshots = {(r["root_id"], r["query_index"]): r for r in proof["snapshots"]}
    C.need(len(snapshots) == len(proof["snapshots"]) == 8 and set(snapshots) == set(B.KEYS),
           "all8_static_snapshot_bindings")
    key = spec["root_id"], spec["query_index"]
    row = snapshots[key]
    C.need(row["saved_input_source_identity"] == spec["saved_input_source_identity"],
           "static_root_saved_source_identity")
    C.need(row["historical_query_source_identity"] == spec["historical_query_source_identity"]
           and row["input_identity"] == spec["input_identity"], "static_root_distinct_identity_domains")
    for arm in ("N", "S", "R"):
        ref = refs[spec["new"][arm]]
        actual = row["native_objects"][arm]
        C.need(all(actual[k] == ref[k] for k in ("uri", "bytes", "sha256")),
               "static_root_exact_native_command_object")
        checks = refs[spec["new"][arm + "_checks"]]
        C.need(all(actual["checks"][k] == checks[k] for k in ("uri", "bytes", "sha256")),
               "static_root_exact_checks")
    root = roots[key[0]]
    return bool(root["controller_mapping_qualified"] is True and
                root["selected_controller"] == "OSC_POSE" and
                root["selected_gripper"] == "PandaGripper")


def formula_source():
    pins = C.json_read((HERE / "CONTROL-SOURCE-PINS.json").read_bytes())
    for row in pins["files"]:
        C.need(fid((HERE / row["path"]).read_bytes()) ==
               {k: row[k] for k in ("bytes", "sha256")}, "exact_control_formula_source")
    config = C.json_read((HERE / "pinned/osc_pose.json").read_bytes())
    C.need(config["type"] == "OSC_POSE" and config["impedance_mode"] == "fixed" and
           config["control_delta"] is True and config["interpolation"] is None and
           config["position_limits"] is None and config["orientation_limits"] is None and
           config["input_min"] == -1 and config["input_max"] == 1 and
           config["output_max"] == A.SCALE.tolist() and
           config["output_min"] == (-A.SCALE).tolist(), "exact_OSC_clip_scale_config")
    return pins, config


def execute(stage, inputs, check, *, synthetic=False, root_proof=None):
    refs = {r["path"]: r for r in inputs["input_objects"]}
    C.need(len(refs) == len(inputs["input_objects"]) == inputs["input_object_count"] and
           len({r["uri"] for r in refs.values()}) == len(refs), "unique_input_roster")
    C.need(sum(r["bytes"] for r in refs.values()) == inputs["input_body_bytes"] <= 96 * 1024**2,
           "bounded_input_total")
    C.need({p.name for p in stage.iterdir()} == set(refs), "exact_stage_roster")
    for name, ref in refs.items():
        check()
        path = stage / name
        C.need(path.is_file() and not path.is_symlink() and path.parent == stage and
               path.stat().st_size == ref["bytes"] <= 64 * 1024**2, "bounded_regular_input")
        C.need(fid(path.read_bytes()) == {k: ref[k] for k in ("bytes", "sha256")}, "input_whole_SHA")
    def doc(name):
        check()
        return C.json_read((stage / name).read_bytes())
    common = inputs["common"]
    terminal, result = doc(common["TERMINAL.json"]), doc(common["RESULT.json"])
    B.validate_terminal(terminal, result, inputs["native72_plan_entries"])
    C.need(inputs["producer_run"] == B.RUN and inputs["producer_job"] == "redacted-f594f0253751",
           "exact_native_producer")
    model = doc(common["MODEL_BINDING"])
    C.need(model["model_bound"] is True and model["adapter_bound"] is True and
           all(model[k] == result["model_binding"][k] for k in
               ("normalization_sha256", "loaded_parameter_tree_sha256")), "loaded_model_binding")
    event = doc(common["SOURCE_BOUND_event"])
    C.need(event["event"] == "SOURCE_BOUND", "source_event_role")
    payload = event["payload"]
    C.need(all(payload[k] is True for k in ("source_bound", "model_bound", "adapter_bound", "inputs_bound"))
           and payload["synthetic"] is False and
           payload["source_manifest"]["sha256"] == result["source_manifest_sha256"] and
           all(payload["model_binding"][k] == refs[common["MODEL_BINDING"]][k]
               for k in ("bytes", "sha256")), "native_source_model_binding")
    validate_archive(doc(common["RUNTIME-RESULT.json"]), doc(common["ARCHIVE-MANIFEST.json"]),
                     refs, inputs)
    pins, config = formula_source()
    proof = root_proof if root_proof is not None else C.json_read(
        (HERE / "ALL-ROOT-CONTROL-BINDING.json").read_bytes())
    C.need([(s["root_id"], s["query_index"]) for s in inputs["snapshots"]] == list(B.KEYS),
           "fixed8_snapshot_order")
    points, windows, cases, identities = [], [], [], []
    for spec in inputs["snapshots"]:
        key = spec["root_id"], spec["query_index"]
        semantic_eligible = root_semantics(proof, spec, refs)
        commands, call_identities = {}, {}
        for arm, role in (("N", "baseline_none_1"), ("S", "shared_pulse"), ("R", "random_pulse")):
            check()
            entry = spec[arm + "_entry"]
            C.need((entry["root_id"], entry["query_index"], entry["role"], entry["arm"]) ==
                   (*key, role, arm), "exact_native_role_not_filename")
            C.need(entry in inputs["native72_plan_entries"], "entry_in_native72")
            C.need(entry["receiver"] == "T" and entry["num_steps"] == 10 and
                   entry["flow_mask"] == [i == 5 and arm != "N" for i in range(10)] and
                   entry["layer_mask"] == [i == 8 and arm != "N" for i in range(18)] and
                   entry["kv_mask"] == [False, arm != "N"] and entry["use_vmod"] == (arm != "N"),
                   "declared_none_or_L9_j5_V_pulse")
            ref = refs[spec["new"][arm]]
            meta, leaves = C.typed(stage / ref["path"], ref, synthetic=synthetic)
            B.validate_call(meta, doc(spec["new"][arm + "_checks"]), entry,
                            spec["input_reference_only"], ref)
            C.need(meta["provenance"]["source_identity"] == spec["saved_input_source_identity"] and
                   meta["provenance"]["native_source_unchanged"] is True, "native_call_source")
            value = B.field(leaves, ("adapter7",), (1, 10, 7), "float64")
            command = np.array(value.numeric()[0], copy=True)
            C.need(np.isfinite(command).all(), "finite_actual_commands")
            commands[arm] = command
            call_identities[arm] = {"call_id": entry["call_id"], "source_ref": ref,
                                    "adapter7": value.identity()}
        p, w, c = A.snapshot(key, commands, semantic_claim_eligible=semantic_eligible)
        points.extend(p)
        windows.extend(w)
        cases.extend(c)
        identities.append({"root_id": key[0], "query_index": key[1],
                           "static_controller_binding_eligible": semantic_eligible,
                           "saved_input_source_identity": spec["saved_input_source_identity"],
                           "historical_query_source_identity": spec["historical_query_source_identity"],
                           "input_identity": spec["input_identity"], "native_calls": call_identities,
                           "original_input_body_not_decoded": spec["input_reference_only"]})
    counts = {"snapshots": len(identities), "native_calls": sum(len(r["native_calls"]) for r in identities),
              "position_contrasts": len(points) // 7, "point_comparisons": len(points),
              "window_contrasts": len(cases), "window_comparisons": len(windows)}
    C.need(counts == COUNTS, "complete_fixed_support_counts")
    summary = A.summarize(points, windows, cases)
    summary["counts"] = counts
    summary["all8_static_bindings_eligible"] = all(r["static_controller_binding_eligible"] for r in identities)
    summary["population"] = {"snapshots": 8, "initialization_clusters": 4, "tasks": 2,
                             "new_task_holdout": False}
    qa = {"schema": "paper2-dev16-control-map-QA-v1", "pass": True, "synthetic": synthetic,
          "counts": counts, "input_objects_verified": len(refs), "input_body_bytes": inputs["input_body_bytes"],
          "all72_final_lists_bound": True, "native24_CALL_CHECKS_bound": True,
          "model_binding_metadata": result["model_binding"], "selected_identities": identities,
          "source_formula_pins": pins, "OSC_config": config,
          "root_binding": fid(encoded(proof)), "root_binding_identity_domain": "canonical_sorted_JSON",
          "typed_compact_full_leaves_integrity_checked": True,
          "only_adapter7_used_in_math": True, "reference_only_input_bodies_downloaded": 0,
          "unknown_static_binding_cases_retained": True,
          "specific_current_action_or_actuator_target_computed": False,
          "controller_runtime_imports": 0, "new_environment_steps": 0}
    return {"POINTS.json.gz": points, "WINDOWS.json": windows, "SUMMARY.json": summary, "QA.json": qa}


def run(stage, out, source_manifest_sha256, deadline_epoch):
    C.need(sys.platform.startswith("linux"), "real_audit_cloud_Linux_only")
    C.need(math.isfinite(deadline_epoch), "finite_deadline")
    stage, out = Path(stage), Path(out)
    C.need(not out.exists() and not out.is_symlink(), "child_owns_absent_output")
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    result = {"schema": "paper2-dev16-control-map-result-v1", "status": "INCOMPLETE",
              "input_qualified": False, "analysis_complete": False, "synthetic": False,
              "GPU_count": 0, "new_model_calls": 0, "new_env_steps": 0,
              "environment_constructions": 0, "runtime_imports": 0, "new_fits": 0,
              "training_updates": 0, "JPFS_new_bytes": 0,
              "physical_response_measured": False, "prediction_validated": False}
    def check():
        C.need(time.time() < deadline_epoch, "absolute_science_deadline")
    total_output = 0
    def write(name, raw):
        nonlocal total_output
        total_output += len(raw)
        C.need(name in OUTPUTS and len(raw) <= 2 * 1024**2 and total_output <= 8 * 1024**2,
               "bounded_declared_output")
        with (out / name).open("xb") as stream:
            stream.write(raw)
        return {"path": name, **fid(raw)}
    phase = "source"
    try:
        check()
        raw = (HERE / "SOURCE-MANIFEST.json").read_bytes()
        C.need(C.sha(raw) == source_manifest_sha256, "externally_bound_source_SHA")
        manifest = C.json_read(raw)
        C.need(manifest["schema"] == "paper2-dev16-control-map-source-v1" and
               {r["path"] for r in manifest["files"]} == set(SOURCE_FILES) and
               len(manifest["files"]) == len(SOURCE_FILES), "closed_source_roster")
        for row in manifest["files"]:
            path = HERE / row["path"]
            C.need(path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(HERE) and
                   fid(path.read_bytes()) == {k: row[k] for k in ("bytes", "sha256")}, "source_member_SHA")
        inputs_raw = (HERE / "INPUTS.json").read_bytes()
        inputs = C.json_read(inputs_raw)
        C.need(inputs["schema"] == "paper2-dev16-control-map-inputs-v1", "frozen_input_schema")
        result.update(source_manifest=fid(raw), input_manifest=fid(inputs_raw))
        phase = "qualified_existing_command_mapping"
        docs = execute(stage, inputs, check)
        phase = "outputs"
        outputs = []
        for name, doc in docs.items():
            check()
            raw = encoded(doc)
            if name.endswith(".gz"):
                raw = gzip.compress(raw, mtime=0)
            outputs.append(write(name, raw))
        outputs.append(write("PROTOCOL.json", (HERE / "PROTOCOL.json").read_bytes()))
        result.update(status="DEV16_CONTROL_MAP_COMPLETE", input_qualified=True,
                      analysis_complete=True, input_objects_verified=inputs["input_object_count"],
                      input_body_bytes=inputs["input_body_bytes"], counts=COUNTS,
                      outputs=outputs, output_names=list(OUTPUTS))
    except Exception as error:
        code = str(error)
        result.update(status="FAILED_PRESERVED", failure={
            "phase": phase, "type": type(error).__name__,
            "code": code if len(code) <= 180 and all(c.isalnum() or c == "_" for c in code) else None,
            "frames": [{"file": Path(f.filename).name, "line": f.lineno, "function": f.name}
                       for f in traceback.extract_tb(error.__traceback__)[-8:]]})
    result["elapsed_seconds"] = time.monotonic() - started
    write(RESULT, encoded(result))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--source-manifest-sha256", required=True)
    parser.add_argument("--deadline-epoch", type=float, required=True)
    args = parser.parse_args()
    result = run(args.stage, args.out, args.source_manifest_sha256, args.deadline_epoch)
    print(json.dumps({k: result[k] for k in ("schema", "status", "analysis_complete")}))
    raise SystemExit(0 if result["analysis_complete"] else 2)
