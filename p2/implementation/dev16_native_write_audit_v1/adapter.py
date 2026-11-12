"""Bounded source-bound CPU entrypoint; no model, network or fitting."""
import argparse
import csv
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
RESULT = "dev16-native-write-audit-result.json"
OUTPUTS = (RESULT, "STATE-CHECKS.json", "WRITE-FIDELITY.json.gz", "DECOMPOSITION.json.gz",
           "DOSE-AND-ENDPOINTS.json", "QA.json", "PROTOCOL.json", "SUMMARY.json")
SOURCE_FILES = ("adapter.py", "analysis.py", "binding.py", "codec.py", "INPUTS.json",
                "PROTOCOL.json", "CODEC-PINS.json", "pinned/retained_replication.py",
                "pinned/quantization_frozen.py", "test_audit.py", "synthetic_fixture.py",
                "freeze.py", "README.md")
COUNTS = {"snapshots": 8, "pre_state_comparisons": 192, "write_fidelity": 936,
          "decomposition": 312, "dose": 424, "endpoints": 336, "existing_scalar_checks": 48}


def encoded(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def fid(raw):
    return {"bytes": len(raw), "sha256": C.sha(raw)}


def validate_archive(runtime, archive, refs, inputs):
    """Bind each selected new object to the parent verified archive."""
    parent = runtime["science_archive"]
    C.need(parent["complete_for_observed_files"] is True and parent["errors"] == [],
           "complete_native_parent_archive")
    objects = parent["objects"] + runtime["runtime_evidence_archive"]["objects"]
    # Producer variations are normalized by the immutable input roster builder;
    # runtime parent still must contain each exact path/size/SHA and VERIFIED GET.
    by_sha = {}
    for obj in objects:
        by_sha.setdefault(obj["sha256"], []).append(obj)
    for name in inputs["native_archive_bound_paths"]:
        r = refs[name]
        hits = [o for o in by_sha.get(r["sha256"], []) if o["bytes"] == r["bytes"]
                and o["receipt"]["uri"] == r["uri"]]
        C.need(len(hits) == 1, "native_object_in_parent_archive")
        obj = hits[0]
        receipt = obj["receipt"]
        C.need(obj["state"] == "VERIFIED" and receipt["uri"] == r["uri"],
               "native_parent_VERIFIED_exact_URI")
    child = {r["path"]: r for r in archive["files"]}
    for name in inputs["native_child_bound_paths"]:
        r = refs[name]
        logical = r["logical_path"]
        C.need(logical in child and all(child[logical][k] == r[k] for k in ("bytes", "sha256")),
               "native_object_in_child_manifest")


def execute(stage, inputs, check, *, synthetic=False):
    refs = {r["path"]: r for r in inputs["input_objects"]}
    C.need(len(refs) == len(inputs["input_objects"]) == inputs["input_object_count"] and
           len({r["uri"] for r in refs.values()}) == len(refs), "unique_input_roster")
    C.need(sum(r["bytes"] for r in refs.values()) == inputs["input_body_bytes"] <= 128 * 1024**2,
           "bounded_input_total")
    C.need({p.name for p in stage.iterdir()} == set(refs), "exact_stage_roster")
    for name, ref in refs.items():
        check()
        p = stage / name
        C.need(p.is_file() and not p.is_symlink() and p.parent == stage and
               p.stat().st_size == ref["bytes"] <= 16 * 1024**2, "bounded_regular_input")
        C.need(fid(p.read_bytes()) == {k: ref[k] for k in ("bytes", "sha256")}, "input_whole_SHA")
    def doc(name):
        check()
        return C.json_read((stage / name).read_bytes())
    def record(name):
        check()
        return C.typed(stage / name, refs[name], synthetic=synthetic)
    common = inputs["common"]
    terminal, result = doc(common["TERMINAL.json"]), doc(common["RESULT.json"])
    observability = B.validate_terminal(terminal, result, inputs["native72_plan_entries"])
    C.need(inputs["producer_run"] == B.RUN and inputs["producer_job"] == "redacted-f594f0253751",
           "exact_native_producer")
    model = doc(common["MODEL_BINDING"])
    C.need(model["model_bound"] is True and model["adapter_bound"] is True and
           all(model[k] == result["model_binding"][k] for k in
               ("normalization_sha256", "loaded_parameter_tree_sha256")), "loaded_model_binding")
    source_event = doc(common["SOURCE_BOUND_event"])
    C.need(source_event["event"] == "SOURCE_BOUND", "source_event_role")
    sp = source_event["payload"]
    C.need(all(sp[k] is True for k in ("source_bound", "model_bound", "adapter_bound", "inputs_bound"))
           and sp["synthetic"] is False and sp["source_manifest"]["sha256"] ==
           result["source_manifest_sha256"] and all(sp["model_binding"][k] ==
           refs[common["MODEL_BINDING"]][k] for k in ("bytes", "sha256")), "native_source_model_binding")
    validate_archive(doc(common["RUNTIME-RESULT.json"]), doc(common["ARCHIVE-MANIFEST.json"]),
                     refs, inputs)
    old_proof = doc(common["old_source_proof"])
    old_rows = {(r["root_id"], r["query_index"]): r for r in old_proof["all_rows"]}
    _, wo_fields = record(common["WO"])
    wo = B.field(wo_fields, ("native_L9_WO",), (8, 256, 1024), "bfloat16").numeric()
    all_state, all_fidelity, all_decomp, all_dose, all_endpoint, all_scalar = [], [], [], [], [], []
    C.need([(s["root_id"], s["query_index"]) for s in inputs["snapshots"]] == list(A.KEYS),
           "fixed8_snapshot_order")
    selected_identities, prefix_bindings = [], []
    for spec in inputs["snapshots"]:
        check()
        key = spec["root_id"], spec["query_index"]
        old = old_rows[key]
        C.need(old["query_fingerprint"] == spec["input_identity"] and
               old["input_source_identity"] == spec["historical_query_source_identity"] and
               old["parent_source"]["source_identity"] == spec["saved_input_source_identity"],
               "distinct_source_domains_bound")
        # Natural C/T payloads are not needed. Their exact receipt metadata is
        # retained to reconstruct the separate historical query hash domain.
        C.need(B.query_source_identity(spec["saved_input_source_identity"],
                                      spec["historical_natural_receipts"]["natural_C"],
                                      spec["historical_natural_receipts"]["natural_T"]) ==
               spec["historical_query_source_identity"], "historical_query_source_reconstruction")
        inp_meta, inp_fields = record(spec["new"]["input"])
        C.need((inp_meta["root_id"], inp_meta["query_index"]) == key and
               inp_meta["source_identity"] == spec["saved_input_source_identity"], "input_key_source")
        em, ef = record(spec["old"]["edits"])
        C.need((em["root_id"], em["query_index"]) == key, "old_edit_key")
        C.need(B.refsame(inp_meta["sources"]["native"]["edits"], refs[spec["old"]["edits"]]),
               "old_edit_record_in_native_input")
        audit_meta, audit = record(spec["old"]["audit"])
        C.need((audit_meta["root_id"], audit_meta["query_index"]) == key and
               audit_meta["source"]["source_identity"] == spec["saved_input_source_identity"],
               "old_state_noise_key_source")
        C.need(B.refsame(inp_meta["sources"]["native"]["audit"], refs[spec["old"]["audit"]]),
               "old_audit_record_in_native_input")
        for name, leaf in (("noise", "explicit_noise"), ("decode_state", "consumed_decode_state")):
            C.need(inp_meta["original_arrays"][name] == audit[(leaf,)].input_identity(),
                   "same_original_noise_state_identity")
        n = inp_fields[("prefix_mask",)].storage.shape[-1]
        C.need(n == 968 or synthetic, "native_prefix968")
        input_mask = B.field(inp_fields, ("prefix_mask",), (1, n), "bool")
        input_positions = B.field(inp_fields, ("prefix_positions",), (1, n), "int32")
        tm, tf = record(spec["old"]["natural_T"])
        C.need((tm["root_id"], tm["query_index"], tm["role"]) == (*key, "natural_T") and
               tm["source_identity"] == spec["saved_input_source_identity"],
               "old_natural_T_key_source")
        C.need(B.refsame(inp_meta["sources"]["native"]["natural_T"],
                        refs[spec["old"]["natural_T"]]), "old_natural_T_in_native_input")
        vt = B.field(tf, ("receiver_L9_V",), (1, n, 1, 256), "bfloat16")
        C.need(vt.input_identity() == inp_meta["original_arrays"]["old_VT"], "old_VT_input_identity")
        for name, current in (("prefix_mask", input_mask), ("prefix_positions", input_positions)):
            C.need(current.identity() == tf[(name,)].identity(), "saved_prefix_address_identity")
        edits = {}
        for arm, original in (("S", "shared"), ("R", "random_matched")):
            e = B.field(ef, ("actual_E", original), (1, n, 1, 256), "float32")
            vm = B.field(ef, ("Vmod", original), (1, n, 1, 256), "bfloat16")
            C.need(e.input_identity() == inp_meta["original_arrays"]["actual_E"][arm] and
                   vm.input_identity() == inp_meta["original_arrays"]["modified_v"][arm],
                   "exact_actualE_native_input_identity")
            C.need(np.array_equal(e.numeric(), np.subtract(vm.numeric(), vt.numeric(), dtype=np.float32)),
                   "actualE_from_frozen_BF16_payload")
            label = inp_meta["sources"]["original_pulse_label_ids"][original]
            C.need(label == {"input_identity": spec["input_identity"],
                             "edit_identity": spec["edit_identities"][original]},
                   "exact_old_label_fingerprint")
            edits[arm] = e.numeric()[0, :, 0].astype(np.float64)
            C.need(np.all(edits[arm][~input_mask.storage[0]] == 0), "actual_E_padding_exact_zero")
        native = {}
        for arm, role in (("N", "baseline_none_1"), ("S", "shared_pulse"), ("R", "random_pulse")):
            entry = spec[arm + "_entry"]
            C.need((entry["root_id"], entry["query_index"], entry["role"], entry["arm"]) ==
                   (*key, role, arm), "exact_native_role_not_filename")
            C.need(entry in inputs["native72_plan_entries"], "entry_in_native72")
            C.need(entry["receiver"] == "T" and entry["num_steps"] == 10 and
                   entry["flow_mask"] == [i == 5 and arm != "N" for i in range(10)] and
                   entry["layer_mask"] == [i == 8 and arm != "N" for i in range(18)] and
                   entry["kv_mask"] == [False, arm != "N"] and entry["use_vmod"] == (arm != "N"),
                   "declared_none_or_L9_j5_V_pulse")
            meta, leaves = record(spec["new"][arm])
            B.validate_call(meta, doc(spec["new"][arm + "_checks"]), entry,
                            refs[spec["new"]["input"]], refs[spec["new"][arm]])
            C.need(meta["provenance"]["source_identity"] == spec["saved_input_source_identity"] and
                   meta["provenance"]["native_source_unchanged"] is True, "native_call_source")
            native[arm] = leaves
        nmask = native["N"][("j5_mask",)]
        prefix_exact = bool(np.array_equal(nmask.storage[0, :, :n],
                                          np.broadcast_to(input_mask.storage[0], (10, n))))
        prefix_bindings.append({"root_id": key[0], "query_index": key[1],
                                "native_N_prefix_mask_matches_bound_input": prefix_exact,
                                "total_prefix_tokens": n,
                                "valid_prefix_tokens": int(input_mask.storage.sum()),
                                "padding_actualE_exact_zero": True,
                                "saved_prefix_positions_exact": True})
        values = A.snapshot(key, native, edits, wo, observability[key], check=check,
                            prefix_binding_exact=prefix_exact,
                            valid_prefix_tokens=int(input_mask.storage.sum()))
        for target, value in zip((all_state, all_fidelity, all_decomp, all_dose, all_endpoint, all_scalar), values):
            target.extend(value)
        selected_identities.append({"root_id": key[0], "query_index": key[1],
                                    "saved_input_source_identity": spec["saved_input_source_identity"],
                                    "historical_query_source_identity": spec["historical_query_source_identity"],
                                    "input_identity": spec["input_identity"],
                                    "native_calls": {arm: spec[arm + "_entry"]["call_id"]
                                                     for arm in ("N", "S", "R")}})
    counts = {"snapshots": len(selected_identities), "pre_state_comparisons": len(all_state),
              "write_fidelity": len(all_fidelity), "decomposition": len(all_decomp),
              "dose": len(all_dose), "endpoints": len(all_endpoint), "existing_scalar_checks": len(all_scalar)}
    C.need(counts == COUNTS, "complete_fixed_support_counts")
    # The previous metadata-only readout is independently bound for exact
    # endpoint definition/support, without importing cross-axis norms.
    with (stage / common["CPU_READOUT_SUPPORT"]).open() as f:
        readout = list(csv.DictReader(f))
    verify_readout(all_endpoint, readout)
    mismatches = [dict(root_id=r["root_id"], query_index=r["query_index"], arm=r["arm"], field=r["field"])
                  for r in all_state if not r["typed_exact"]]
    summary = {"schema": "paper2-dev16-native-write-audit-summary-v1", "counts": counts,
               "population": {"snapshots": 8, "roots": 4, "tasks": 2},
               "state_typed_mismatches": mismatches, "state_mismatch_count": len(mismatches),
               "native_prefix_binding_mismatches": [r for r in prefix_bindings
                                                    if not r["native_N_prefix_mask_matches_bound_input"]],
               "all8_retained": True, "same_state_interpretation_is_gated_per_pair": True,
               "native_gated_product_captured": False, "new_fits": 0,
               "write_dose_difference_invalidates_equal_E_total_effect": False,
               "same_W_B_GPU_automatic_followup": False,
               "already_measured_stage_norms": True,
               "projection_is_target_visible_not_prediction": True,
               "independent_sample_unit": "8 snapshots nested in 4 roots and 2 tasks",
               "physical_or_success_threshold": None,
               "fidelity_orientation": "reference = native output difference; candidate = FP64 proposed proxy",
               "dose_orientation": "reference = R; candidate = S",
               "full10_fidelity_all_cases": [r for r in all_fidelity if r["window"] == "full10"],
               "full10_dose_all_cases": [r for r in all_dose if r["window"] in
                                        ("full10", "all_prefix_tokens")],
               "full10_decomposition_all_cases": [r for r in all_decomp if r["window"] == "full10"]}
    qa = {"schema": "paper2-dev16-native-write-audit-QA-v1", "pass": True, "synthetic": synthetic,
          "counts": counts, "input_objects_verified": len(refs), "input_body_bytes": inputs["input_body_bytes"],
          "all72_final_lists_bound": True, "native24_CALL_CHECKS_bound": True,
          "model_binding": result["model_binding"], "selected_identities": selected_identities,
          "existing_scalar_checks": all_scalar, "endpoint_previous_CPU_readout_verified": True,
          "prefix_binding_checks": prefix_bindings,
          "scalar_tolerance_never_used_for_typed_native_equality": True,
          "state_mismatches_are_preserved_scientific_results": True}
    return {"STATE-CHECKS.json": all_state, "WRITE-FIDELITY.json.gz": all_fidelity,
            "DECOMPOSITION.json.gz": all_decomp,
            "DOSE-AND-ENDPOINTS.json": {"dose_reference_R_candidate_S": all_dose, "endpoints": all_endpoint},
            "QA.json": qa, "SUMMARY.json": summary}


def verify_readout(endpoints, rows):
    """Prior frozen CSV scalar readout, never recalibrated."""
    index = {(r["root_id"], int(r["query_index"]), r["contrast"].replace("_minus_", "-"),
              r["window"], r["command_dimension"]): r for r in rows}
    C.need(len(index) == len(rows), "unique_previous_CPU_endpoint_rows")
    for row in endpoints:
        key = tuple(row[k] for k in ("root_id", "query_index", "contrast", "window", "dimension"))
        C.need(key in index, "endpoint_in_previous_CPU_readout")
        c = A.scalar_check(row["RMS"], index[key]["RMS"])
        C.need(c["scalar_close"], "endpoint_CPU_readout_RMS_close")


def run(stage, out, source_manifest_sha256, deadline_epoch):
    C.need(sys.platform.startswith("linux"), "real_audit_cloud_Linux_only")
    C.need(math.isfinite(deadline_epoch), "finite_deadline")
    out, stage = Path(out), Path(stage)
    C.need(not out.exists() and not out.is_symlink(), "child_owns_absent_output")
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    result = {"schema": "paper2-dev16-native-write-audit-result-v1", "status": "INCOMPLETE",
              "input_qualified": False, "analysis_complete": False, "synthetic": False,
              "GPU_count": 0, "new_model_calls": 0, "new_env_steps": 0, "new_fits": 0,
              "training_updates": 0, "JPFS_new_bytes": 0, "prediction_validated": False}
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
        C.need(manifest["schema"] == "paper2-dev16-native-write-audit-source-v1" and
               {r["path"] for r in manifest["files"]} == set(SOURCE_FILES) and
               len(manifest["files"]) == len(SOURCE_FILES), "closed_source_roster")
        for r in manifest["files"]:
            p = HERE / r["path"]
            C.need(p.is_file() and not p.is_symlink() and p.resolve().is_relative_to(HERE) and
                   fid(p.read_bytes()) == {k: r[k] for k in ("bytes", "sha256")}, "source_member_SHA")
        inputs_raw = (HERE / "INPUTS.json").read_bytes()
        inputs = C.json_read(inputs_raw)
        C.need(inputs["schema"] == "paper2-dev16-native-write-audit-inputs-v1", "frozen_input_schema")
        result.update(source_manifest=fid(raw), input_manifest=fid(inputs_raw))
        phase = "typed_native_write_audit"
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
        result.update(status="DEV16_NATIVE_WRITE_AUDIT_COMPLETE", input_qualified=True,
                      analysis_complete=True, input_objects_verified=inputs["input_object_count"],
                      input_body_bytes=inputs["input_body_bytes"], counts=COUNTS,
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
