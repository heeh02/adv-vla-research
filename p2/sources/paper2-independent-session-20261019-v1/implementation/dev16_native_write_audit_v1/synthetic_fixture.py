"""Fabricated native-shaped arrays only. No producer NPZ is read."""
import copy
import csv
import io
import json
from pathlib import Path
import numpy as np
import adapter as D
import analysis as A
import binding as B
import codec as C


def bf(a):
    return C.Leaf((np.asarray(a, np.float32).view(np.uint32) >> 16).astype(np.uint16), "bfloat16")


def plain(a, dtype):
    return C.from_array(np.asarray(a, dtype))


def typed_bytes(meta, leaves):
    values, tree = {}, {}
    for i, (keys, leaf) in enumerate(leaves.items()):
        name = "leaf_" + str(i)
        values[name] = leaf.storage
        node = tree
        for key in keys[:-1]:
            node = node.setdefault(key, {})
        node[keys[-1]] = {"__array__": name, **leaf.identity()}
    values["__schema__"] = np.frombuffer(json.dumps({"metadata": meta, "arrays": tree},
                                                 sort_keys=True).encode(), np.uint8)
    buf = io.BytesIO()
    np.savez_compressed(buf, **values)
    return buf.getvalue()


def example():
    n = 6
    x = plain(np.arange(11 * 320).reshape(11, 1, 10, 32) / 1024, "float32")
    vel = bf(np.zeros((10, 1, 10, 32)))
    p = bf(np.ones((1, 1, 8, 10, n + 10)) / (n + 10))
    arrays = {("flow__x",): x, ("flow__velocity_native",): vel,
              ("flow__time",): plain(np.arange(11) / 10, "float32"),
              ("flow__dt",): plain(np.ones(10) / 10, "float32"),
              ("flow__time_weak_type",): plain(np.zeros(11), "bool"),
              ("j5_L9_query_post_rope_scaled",): bf(np.ones((1, 10, 1, 8, 256))),
              ("j5_L9_probabilities_native",): p,
              ("j5_mask",): plain(np.ones((1, 10, n + 10)), "bool"),
              ("j5_suffix_positions",): plain(np.arange(n, n + 10)[None], "int32"),
              ("j5_query_defined",): plain(np.ones((1, 10)), "bool"),
              ("reads__attention_gate",): bf(np.full((5, 3, 1, 1, 1024), .5)),
              ("reads__receiver_residual_input",): bf(np.ones((5, 3, 1, 10, 1024))),
              ("reads__attention_output_post_WO_pre_gate",): bf(np.zeros((5, 3, 1, 10, 1024))),
              ("reads__hidden_after_attention_residual",): bf(np.ones((5, 3, 1, 10, 1024))),
              ("adapter7",): plain(np.ones((1, 10, 7)), "float64")}
    native = {"N": arrays, "S": dict(arrays), "R": dict(arrays)}
    vt = bf(np.ones((1, n, 1, 256)))
    vms = {"shared": bf(np.full((1, n, 1, 256), 1.5)),
           "random_matched": bf(np.full((1, n, 1, 256), .5))}
    edit = {}
    for role, vm in vms.items():
        edit[("Vmod", role)] = vm
        edit[("actual_E", role)] = plain(vm.numeric() - vt.numeric(), "float32")
    for arm, gain in (("S", 1), ("R", -.5)):
        pre = np.zeros((5, 3, 1, 10, 1024), np.float32)
        pre[0, 1, 0] = (np.arange(10240).reshape(10, 1024) % 13 - 6) * gain / 16
        after = np.ones_like(pre)
        after[0, 1, 0] += pre[0, 1, 0] * .5
        native[arm][("reads__attention_output_post_WO_pre_gate",)] = bf(pre)
        native[arm][("reads__hidden_after_attention_residual",)] = bf(after)
        native[arm][("adapter7",)] = plain(1 + gain * np.arange(70).reshape(1, 10, 7) / 1024,
                                         "float64")
    w = np.zeros((8, 256, 1024), np.float32)
    w[:, np.arange(256), np.arange(256)] = 1
    wo = bf(w)
    inp = {("prefix_mask",): plain(np.ones((1, n)), "bool"),
           ("prefix_positions",): plain(np.arange(n)[None], "int32")}
    audit = {("explicit_noise",): x.select(0),
             ("consumed_decode_state",): plain(np.zeros((1, 32)), "float32")}
    return native, edit, wo, inp, audit, vt


def observability(key, native, edit):
    out = {"root_id": key[0], "query_index": key[1], "arms": {},
           "S_minus_R_adapter7": (native["S"][("adapter7",)].numeric() -
                                 native["R"][("adapter7",)].numeric()).tolist()}
    for arm, role in (("S", "shared"), ("R", "random_matched")):
        stages = {}
        for stage, name in (("L9_j5_write_pre_gate", "attention_output_post_WO_pre_gate"),
                            ("L9_j5_post_attention_residual", "hidden_after_attention_residual")):
            b, a = (A.site(native[x], name) for x in ("N", arm))
            delta = a.numeric().astype(np.float64) - b.numeric().astype(np.float64)
            stages[stage] = {"squared_norm": float(np.sum(delta**2)),
                             "maximum_absolute_difference": float(np.max(np.abs(delta))),
                             "numeric_nonzero_coordinates": int(np.count_nonzero(delta)),
                             "bitwise_changed_coordinates": int(np.count_nonzero(a.storage != b.storage))}
        out["arms"][arm] = {"actual_E_squared_norm": float(np.sum(
            edit[("actual_E", role)].numeric().astype(np.float64)**2)),
            "stage_changes": stages, "adapter7_effect": (native[arm][("adapter7",)].numeric() -
                                                          native["N"][("adapter7",)].numeric()).tolist()}
    return out


def stage_fixture(stage):
    stage = Path(stage)
    stage.mkdir()
    plan = json.loads((Path(__file__).parent / "INPUTS.json").read_text())["native72_plan_entries"]
    refs, science, evidence, samples, old_rows, obs_rows, csv_rows = [], [], [], [], [], [], []
    native_names, child_names = [], []
    def save(name, raw, parent=None):
        (stage / name).write_bytes(raw)
        r = {"path": name, "logical_path": name, "uri": "synthetic://" + name,
             "bytes": len(raw), "sha256": C.sha(raw), "role": "synthetic"}
        refs.append(r)
        if parent is not None:
            parent.append({"path": name, **D.fid(raw), "state": "VERIFIED",
                           "receipt": {"uri": r["uri"], **D.fid(raw)}})
            native_names.append(name)
            if parent is science:
                child_names.append(name)
        return r
    def js(name, value, parent=None):
        return save(name, D.encoded(value), parent)
    for i, key in enumerate(A.KEYS):
        native, edit, wo, inp, audit, vt = example()
        saved = f"synthetic_saved_{i}"
        oldrefs = {}
        oldrefs["edits"] = save(f"old_edit_{i}.npz", typed_bytes(
            {"root_id": key[0], "query_index": key[1]}, edit))
        oldrefs["audit"] = save(f"old_audit_{i}.npz", typed_bytes(
            {"root_id": key[0], "query_index": key[1], "source": {"source_identity": saved}}, audit))
        oldrefs["natural_T"] = save(f"old_T_{i}.npz", typed_bytes(
            {"root_id": key[0], "query_index": key[1], "role": "natural_T", "source_identity": saved},
            {("receiver_L9_V",): vt, **inp}))
        cref = {"path": f"not_loaded_C_{i}.npz", "bytes": 9, "sha256": "synthetic_C"}
        historical = B.query_source_identity(saved, cref, oldrefs["natural_T"])
        spec = {"root_id": key[0], "query_index": key[1], "saved_input_source_identity": saved,
                "historical_query_source_identity": historical, "input_identity": f"synthetic_query_{i}",
                "edit_identities": {"shared": f"synthetic_S_{i}", "random_matched": f"synthetic_R_{i}"},
                "historical_natural_receipts": {"natural_C": cref, "natural_T": oldrefs["natural_T"]},
                "old": {k: r["path"] for k, r in oldrefs.items()}, "new": {}}
        original = {"old_VT": vt.input_identity(), "noise": audit[("explicit_noise",)].input_identity(),
                    "decode_state": audit[("consumed_decode_state",)].input_identity(),
                    "actual_E": {}, "modified_v": {}}
        for arm, role in (("S", "shared"), ("R", "random_matched")):
            original["actual_E"][arm] = edit[("actual_E", role)].input_identity()
            original["modified_v"][arm] = edit[("Vmod", role)].input_identity()
        im = {"root_id": key[0], "query_index": key[1], "source_identity": saved,
              "original_arrays": original, "sources": {"native": oldrefs,
              "original_pulse_label_ids": {role: {"input_identity": spec["input_identity"],
                  "edit_identity": spec["edit_identities"][role]} for role in ("shared", "random_matched")}}}
        ir = save(f"input_{i}.npz", typed_bytes(im, inp), science)
        spec["new"]["input"] = ir["path"]
        for arm, role in (("N", "baseline_none_1"), ("S", "shared_pulse"), ("R", "random_pulse")):
            entry = next(e for e in plan if (e["root_id"], e["query_index"], e["role"]) == (*key, role))
            spec[arm + "_entry"] = entry
            meta = {"schema": "rss-pulse-compact-call-v1", "entry": entry, "input_ref": ir,
                    "native_validation_errors": [], "provenance": {"source_identity": saved,
                                                                   "native_source_unchanged": True}}
            nr = save(f"call_{i}_{arm}.npz", typed_bytes(meta, native[arm]), science)
            er = js(f"checks_{i}_{arm}.json", {"schema": "rss-pulse-compact-event-v1", "event": "CALL_CHECKS",
                    "payload": {"entry": entry, "passed": True, "output": nr}}, science)
            spec["new"][arm], spec["new"][arm + "_checks"] = nr["path"], er["path"]
        old_rows.append({"root_id": key[0], "query_index": key[1],
                         "query_fingerprint": spec["input_identity"], "input_source_identity": historical,
                         "parent_source": {"source_identity": saved}})
        obs_rows.append(observability(key, native, edit))
        for contrast, after, before in (("S_minus_N", "S", "N"), ("R_minus_N", "R", "N"),
                                        ("S_minus_R", "S", "R")):
            effect = (native[after][("adapter7",)].numeric() - native[before][("adapter7",)].numeric())[0]
            for window, stop in (("first5", 5), ("full10", 10)):
                for axis, dimension in enumerate(A.DIMS):
                    csv_rows.append({"root_id": key[0], "query_index": key[1], "contrast": contrast,
                                     "command_dimension": dimension, "window": window,
                                     "RMS": float(np.sqrt(np.mean(effect[:stop, axis]**2)))})
        samples.append(spec)
    common = {}
    common["WO"] = save("wo.npz", typed_bytes({}, {("native_L9_WO",): wo}))["path"]
    common["old_source_proof"] = js("old_source.json", {"all_rows": old_rows})["path"]
    model = {"model_bound": True, "adapter_bound": True, "normalization_sha256": "synthetic_norm",
             "loaded_parameter_tree_sha256": "synthetic_model"}
    mr = js("model.json", model, evidence)
    common["MODEL_BINDING"] = mr["path"]
    common["SOURCE_BOUND_event"] = js("source.json", {"event": "SOURCE_BOUND", "payload": {
        "source_bound": True, "model_bound": True, "adapter_bound": True, "inputs_bound": True,
        "synthetic": False, "source_manifest": {"sha256": "synthetic_source"}, "model_binding": mr}},
        science)["path"]
    lists = {k: [e["call_id"] for e in plan] for k in (
        "charged_calls", "completed_calls", "passed_calls", "returned_native_calls")}
    common["RESULT.json"] = js("result.json", {"run_id": B.RUN, "status": "DEV_PULSE_NATIVE_COMPLETE",
        "synthetic": False, "worker_complete": True, "source_manifest_sha256": "synthetic_source",
        "model_binding": model, **lists}, science)["path"]
    common["TERMINAL.json"] = js("terminal.json", {"unexecuted_call_ids": [],
        "stopped_on_storage_failure": False, "result": {"complete": True, "qualification_complete": True,
        "failures": [], "observability": obs_rows, **lists}}, science)["path"]
    files = [{k: o[k] for k in ("path", "bytes", "sha256")} for o in science]
    ar = js("archive.json", {"files": files}, science)
    child_names.remove(ar["path"])
    common["ARCHIVE-MANIFEST.json"] = ar["path"]
    common["RUNTIME-RESULT.json"] = js("runtime.json", {
        "science_archive": {"complete_for_observed_files": True, "errors": [], "objects": science},
        "runtime_evidence_archive": {"complete_for_observed_files": True, "errors": [], "objects": evidence}})["path"]
    buf = io.StringIO()
    writer = csv.DictWriter(buf, list(csv_rows[0]))
    writer.writeheader()
    writer.writerows(csv_rows)
    common["CPU_READOUT_SUPPORT"] = save("readout.csv", buf.getvalue().encode())["path"]
    return {"producer_run": B.RUN, "producer_job": "redacted-f594f0253751",
            "input_objects": refs, "input_object_count": len(refs),
            "input_body_bytes": sum(r["bytes"] for r in refs), "common": common,
            "snapshots": samples, "native72_plan_entries": plan,
            "native_archive_bound_paths": native_names, "native_child_bound_paths": child_names}
