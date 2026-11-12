"""Source-bound existing-data CPU bridge; no model or statistical fits."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import traceback
import numpy as np
import codec as C
import analysis as A

HERE = Path(__file__).resolve().parent
RESULT = "dev16-feature-bridge-result.json"
OUTPUTS = (RESULT, "FIELD-COMPARISONS.json", "FEATURE-COMPARISONS.json",
           "ENDPOINT-COMPARISONS.json", "DECISION-TABLE.json", "QA.json", "PROTOCOL.json")
SOURCE_FILES = ("adapter.py", "analysis.py", "codec.py", "INPUTS.json", "PROTOCOL.json",
                "CODEC-PINS.json", "pinned/retained_replication.py", "ROOT-QUESTION-SPEC.json",
                "synthetic_fixture.py", "test_bridge.py", "freeze.py", "README.md")


def encoded(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def fid(raw):
    return {"bytes": len(raw), "sha256": C.sha(raw)}


def aid(a):
    return {"shape": list(a.shape), "dtype": a.dtype.str, "sha256": C.sha(a.tobytes(order="C"))}


def execute(stage, inputs, check, *, synthetic=False):
    refs = {r["path"]: r for r in inputs["input_objects"]}
    C.need(len(refs) == len(inputs["input_objects"]) == inputs["input_object_count"] and
           len({r["uri"] for r in refs.values()}) == len(refs), "unique_input_roster")
    C.need(sum(r["bytes"] for r in refs.values()) == inputs["input_body_bytes"] <= 192 * 1024**2,
           "bounded_total_input")
    C.need({p.name for p in stage.iterdir()} == set(refs), "exact_stage_roster")
    for name, r in refs.items():
        check()
        p = stage / name
        C.need(p.is_file() and not p.is_symlink() and p.parent == stage and
               p.stat().st_size == r["bytes"] <= 64 * 1024**2, "bounded_regular_input")
        C.need(fid(p.read_bytes()) == {k: r[k] for k in ("bytes", "sha256")}, "input_whole_SHA")
    def doc(name):
        check()
        return C.json_read((stage / name).read_bytes())
    def record(name):
        check()
        return C.typed(stage / name, refs[name], synthetic=synthetic)
    common = inputs["common"]
    terminal, native_result = doc(common["TERMINAL.json"]), doc(common["RESULT.json"])
    obs = A.validate_terminal(terminal, native_result, inputs["native72_plan_entries"])
    C.need(inputs["producer_run"] == A.RUN and inputs["producer_job"] == "redacted-f594f0253751",
           "exact_native_producer")
    model = doc(common["MODEL_BINDING"])
    binding = native_result["model_binding"]
    C.need(model["model_bound"] is True and model["adapter_bound"] is True and
           model["normalization_sha256"] == binding["normalization_sha256"] and
           model["loaded_parameter_tree_sha256"] == binding["loaded_parameter_tree_sha256"],
           "loaded_model_binding")
    source_event = doc(common["SOURCE_BOUND_event"])
    C.need(source_event["event"] == "SOURCE_BOUND", "source_bound_event")
    sp = source_event["payload"]
    C.need(all(sp[k] is True for k in ("source_bound", "model_bound", "adapter_bound", "inputs_bound"))
           and sp["synthetic"] is False, "source_bound_native_flags")
    C.need(sp["source_manifest"]["sha256"] == native_result["source_manifest_sha256"] and
           all(sp["model_binding"][k] == refs[common["MODEL_BINDING"]][k] for k in ("bytes", "sha256")),
           "source_event_manifest_model_binding")
    old_source = doc(common["old_source_proof"])
    old_rows = {(r["root_id"], r["query_index"]): r for r in old_source["all_rows"]}
    cache = C.flat_npz(stage / common["cache"], synthetic=synthetic)
    identities = doc(common["cache_ids"])
    identity_map = {tuple(r["key"]): r for r in identities}
    keys = [(str(k[0]), int(k[1]), str(k[2]), str(k[3])) for k in cache["row_keys"]]
    C.need(len(keys) == len(set(keys)) == (8 if synthetic else 168), "complete_unique_cache_roster")
    index = {k: i for i, k in enumerate(keys)}
    frozen = C.flat_npz(stage / common["frozen_arrays"], synthetic=synthetic)
    manifest = doc(common["frozen_manifest"])
    C.need(manifest["synthetic"] is False and manifest["model"]["schema"] ==
           "task-blind-frozen-model-v1", "actual_frozen_model")
    fm = manifest["model"]
    C.need(C.sha(encoded(fm)) == manifest["model_identity"], "old_model_manifest_identity")
    C.need(set(frozen) == set(fm["arrays"]), "complete_old_frozen_arrays")
    for k, v in frozen.items():
        C.need(aid(v) == fm["arrays"][k], "old_frozen_array_identity")
    _, woleaves = record(common["WO"])
    wo = A.field(woleaves, ("native_L9_WO",), (8, 256, 1024), "bfloat16")
    C.need(np.array_equal(frozen["WO_native_storage"], wo.storage), "frozen_model_same_WO")
    fields, features, endpoints, decisions = [], [], [], []
    C.need([(s["root_id"], s["query_index"]) for s in inputs["snapshots"]] == list(A.KEYS),
           "fixed8_snapshot_order")
    for s in inputs["snapshots"]:
        check()
        key = (s["root_id"], s["query_index"])
        source = old_rows[key]
        C.need(source["query_fingerprint"] == s["input_identity"] and
               source["input_source_identity"] == s["historical_query_source_identity"], "original_query_proof_join")
        A.validate_source_domains(s, source, refs)
        cachekey = (*key, "shared", "KT")
        identity = identity_map[cachekey]
        C.need(identity["provenance"]["input_identity"] == s["input_identity"], "cache_row_provenance")
        for k in ("current_headpos_read", "current_gated_write", "context9", "current4", "effect"):
            C.need(aid(cache[k][index[cachekey]]) == identity["arrays"][k], "cache_selected_array_identity")
        old = {k: record(name) for k, name in s["old"].items()}
        inp = record(s["new"]["input"])
        new = {k: record(s["new"][k]) for k in ("N", "P")}
        for arm in ("N", "P"):
            A.validate_call(new[arm][0], doc(s["new"][arm + "_checks"]), s[arm + "_entry"],
                            refs[s["new"]["input"]], refs[s["new"][arm]])
            C.need(new[arm][0]["provenance"]["source_identity"] == s["saved_input_source_identity"] and
                   new[arm][0]["provenance"]["native_source_unchanged"] is True,
                   "actual_new_call_source")
        spec = {**s, "_synthetic": synthetic}
        answer = A.one_snapshot(spec, old, new, inp, cache, index, frozen, wo.numeric(), obs[key], refs)
        for target, values in zip((fields, features, endpoints, decisions), answer):
            target.extend(values)
    C.need(len(fields) == 96 and len(features) == 56 and len(endpoints) == 848 and
           len(decisions) == 128, "complete_fixed_comparison_support")
    qa = {"schema": "paper2-dev16-feature-bridge-QA-v1", "pass": True, "synthetic": synthetic,
          "input_objects_verified": len(refs), "input_body_bytes": inputs["input_body_bytes"],
          "all8_retained": True, "native72_final_lists_checked": True,
          "new_N_P_checks_events_bound": 16, "whole_graph_equivalence": "UNMEASURED",
          "fresh_C_context": "UNMEASURED", "new_fits": 0, "new_model_calls": 0, "new_env_steps": 0,
          "numerical_mismatch_is_analysis_result": True, "physical_threshold": None,
          "FP64_cache_reconstruction": "No tolerance-based graph acceptance; all exact and numerical deltas exported.",
          "old_frozen_results_replaced": False, "model_binding": binding,
          "counts": {"snapshots": 8, "native_field_comparisons": 96, "feature_comparisons": 56,
                     "endpoint_comparisons": 848, "decisions": 128},
          "endpoint_count_explanation": "8*(3 comparisons*2 windows*7 axes +2 raw32 endpoints*32 coordinates)",
          "cutoff": "Full unedited j5 shadow information; no strict pre-j5 claim."}
    return {"FIELD-COMPARISONS.json": fields, "FEATURE-COMPARISONS.json": features,
            "ENDPOINT-COMPARISONS.json": endpoints, "DECISION-TABLE.json": decisions, "QA.json": qa}


def run(stage, out, source_manifest_sha256, deadline_epoch):
    C.need(sys.platform.startswith("linux"), "real_bridge_cloud_Linux_only")
    C.need(math.isfinite(deadline_epoch), "finite_deadline")
    out, stage = Path(out), Path(stage)
    C.need(not out.exists() and not out.is_symlink(), "child_owns_absent_output")
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    result = {"schema": "paper2-dev16-feature-bridge-result-v1", "status": "INCOMPLETE",
              "input_qualified": False, "analysis_complete": False, "synthetic": False,
              "new_model_calls": 0, "new_env_steps": 0, "new_fits": 0, "training_updates": 0,
              "JPFS_new_bytes": 0, "GPU_count": 0, "prediction_validated": False,
              "whole_graph_equivalence": "UNMEASURED"}
    def check():
        C.need(time.time() < deadline_epoch, "absolute_science_deadline")
    def write(name, raw):
        C.need(name in OUTPUTS and len(raw) <= 2 * 1024**2, "bounded_declared_output")
        with (out / name).open("xb") as f:
            f.write(raw)
        return {"path": name, **fid(raw)}
    phase = "source"
    try:
        check()
        raw = (HERE / "SOURCE-MANIFEST.json").read_bytes()
        C.need(C.sha(raw) == source_manifest_sha256, "externally_bound_source_SHA")
        manifest = C.json_read(raw)
        C.need(manifest["schema"] == "paper2-dev16-feature-bridge-source-v1" and
               {r["path"] for r in manifest["files"]} == set(SOURCE_FILES) and
               len(manifest["files"]) == len(SOURCE_FILES), "closed_source_roster")
        for r in manifest["files"]:
            p = HERE / r["path"]
            C.need(p.is_file() and not p.is_symlink() and p.resolve().is_relative_to(HERE),
                   "regular_owned_source")
            C.need(fid(p.read_bytes()) == {k: r[k] for k in ("bytes", "sha256")}, "source_member_SHA")
        inputs_raw = (HERE / "INPUTS.json").read_bytes()
        inputs = C.json_read(inputs_raw)
        C.need(inputs["schema"] == "paper2-dev16-feature-bridge-inputs-v1" and
               inputs["input_object_count"] == 90 and inputs["input_body_bytes"] == 115674821,
               "frozen90_input_roster")
        result.update(source_manifest=fid(raw), input_manifest=fid(inputs_raw))
        phase = "typed_bridge"
        docs = execute(stage, inputs, check)
        phase = "outputs"
        outputs = []
        for name, doc in docs.items():
            check()
            outputs.append(write(name, encoded(doc)))
        outputs.append(write("PROTOCOL.json", (HERE / "PROTOCOL.json").read_bytes()))
        result.update(status="DEV16_FEATURE_BRIDGE_COMPLETE", input_qualified=True,
                      analysis_complete=True, input_objects_verified=90, input_body_bytes=115674821,
                      snapshots=8, comparison_counts=docs["QA.json"]["counts"],
                      outputs=outputs, output_names=list(OUTPUTS))
    except Exception as error:
        code = str(error)
        result.update(status="FAILED_PRESERVED", failure={"phase": phase, "type": type(error).__name__,
                      "code": code if len(code) <= 180 and all(c.isalnum() or c == "_" for c in code) else None,
                      "frames": [{"file": Path(f.filename).name, "line": f.lineno, "function": f.name}
                                 for f in traceback.extract_tb(error.__traceback__)[-8:]]})
    result["elapsed_seconds"] = time.monotonic() - started
    write(RESULT, encoded(result))
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--stage", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--source-manifest-sha256", required=True)
    p.add_argument("--deadline-epoch", type=float, required=True)
    a = p.parse_args()
    r = run(a.stage, a.out, a.source_manifest_sha256, a.deadline_epoch)
    print(json.dumps({k: r[k] for k in ("schema", "status", "analysis_complete")}))
    raise SystemExit(0 if r["analysis_complete"] else 2)
