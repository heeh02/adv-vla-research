"""Explicit fabricated tensors only; never a producer-data reader."""
import io
import json
from pathlib import Path
import numpy as np
import analysis as A
import codec as C


def bf(a):
    return C.Leaf((np.asarray(a, np.float32).view(np.uint32) >> 16).astype(np.uint16), "bfloat16")


def plain(a, dtype):
    return C.from_array(np.asarray(a, dtype))


def put(tree, keys, value):
    for key in keys[:-1]:
        tree = tree.setdefault(key, {})
    tree[keys[-1]] = value


def typed_bytes(meta, leaves):
    arrays, tree = {}, {}
    for i, (keys, leaf) in enumerate(leaves.items()):
        name = "leaf_" + str(i)
        arrays[name] = leaf.storage
        put(tree, keys, {"__array__": name, **leaf.identity()})
    arrays["__schema__"] = np.frombuffer(json.dumps({"metadata": meta, "arrays": tree},
                                                  sort_keys=True).encode(), np.uint8)
    buf = io.BytesIO()
    np.savez_compressed(buf, **arrays)
    return buf.getvalue()


def fixture(root="SDRU-CDEV-T00-R00", query=11):
    """One small-prefix case, with realistic feature and action dimensions."""
    N = 6
    flow = plain(np.arange(11 * 320).reshape(11, 1, 10, 32) / 1024, "float32")
    velocity = bf(np.ones((10, 1, 10, 32)))
    time = plain(np.arange(11) / 10, "float32")
    weak = plain(np.zeros(11), "bool")
    prob = bf(np.ones((1, 1, 8, 10, N + 10)) / (N + 10))
    gate = bf(np.ones((1, 1, 1024)))
    vt, vm = bf(np.ones((1, N, 1, 256))), bf(np.full((1, N, 1, 256), 1.5))
    e = plain(vm.numeric() - vt.numeric(), "float32")
    mask = plain(np.ones((1, N)), "bool")
    pos = plain(np.arange(N)[None], "int32")
    endpoint = plain(np.ones((1, 10, 7)), "float64")
    raw = plain(np.ones((1, 10, 32)), "float32")
    c = {("flow", "x"): flow, ("flow", "velocity_native"): velocity,
         ("flow", "time"): time, ("flow", "time_weak_type"): weak,
         ("first_read", "probabilities_native"): prob, ("first_read", "attention_gate"): gate,
         ("first_read", "mask"): plain(np.ones((1, 10, N + 10)), "bool"),
         ("first_read", "suffix_positions"): plain(np.arange(N, N + 10)[None], "int32"),
         ("prefix_mask",): mask, ("prefix_positions",): pos,
         ("receiver_L9_V",): vt, ("receiver_L9_K",): vt,
         ("adapter7",): endpoint, ("raw32",): raw}
    t = dict(c)
    audit = {("explicit_noise",): flow.select(0),
             ("consumed_decode_state",): plain(np.zeros((1, 32)), "float32")}
    ep = dict(c)
    ep[("adapter7",)] = plain(endpoint.numeric() + np.arange(70).reshape(1, 10, 7) / 128, "float64")
    ep[("raw32",)] = plain(raw.numeric() + .25, "float32")
    ed = {("actual_E", "shared"): e, ("Vmod", "shared"): vm}
    new = {("flow__x",): flow, ("flow__velocity_native",): velocity,
           ("flow__time",): time, ("flow__time_weak_type",): weak,
           ("j5_L9_probabilities_native",): prob,
           ("reads__attention_gate",): bf(np.ones((5, 3, 1, 1, 1024))),
           ("j5_mask",): c[("first_read", "mask")],
           ("j5_suffix_positions",): c[("first_read", "suffix_positions")],
           ("adapter7",): endpoint, ("raw32",): raw}
    np_ = dict(new)
    np_[("adapter7",)] = ep[("adapter7",)]
    np_[("raw32",)] = ep[("raw32",)]
    w = np.zeros((8, 256, 1024), np.float32)
    w[:, np.arange(256), np.arange(256)] = 1
    wo = bf(w)
    frozen = {"WO_native_storage": wo.storage}
    for name, width in (("state", 2592), ("first_flow", 320), ("write", 10240)):
        basis = np.zeros((width, 4), np.float64)
        basis[np.arange(4), np.arange(4)] = 1
        frozen.update({name + "_mean": np.zeros(width), name + "_basis": basis,
                       name + "_scale": np.ones(4)})
    source = {"source_identity": "synthetic_saved_state", "root_id": root,
              "query_index": query, "env_step_index": 5 * query}
    om = {k: {"root_id": root, "query_index": query, "source_identity": source["source_identity"]}
          for k in ("natural_C", "natural_T", "effect_shared", "edits", "audit")}
    for k in ("natural_C", "natural_T", "effect_shared"):
        om[k]["role"] = k
    om["audit"]["source"] = source
    om["effect_shared"]["technical"] = {"selectors": {
        "kv_mask": [False, True], "layer_mask": [i == 8 for i in range(18)],
        "flow_mask": [i >= 5 for i in range(10)], "use_vmod": True}}
    old = {k: (om[k], v) for k, v in
           (("natural_C", c), ("natural_T", t), ("effect_shared", ep), ("edits", ed), ("audit", audit))}
    context = A.old_context(c, t, audit, frozen, query)
    read, write = A.proxy(prob.numeric()[0, 0], e.numeric()[0, :, 0], wo.numeric(), gate.numeric()[0, 0])
    cache = {"current_headpos_read": read, "current_gated_write": write, "context9": context,
             "current4": ((write.ravel() - frozen["write_mean"]) @ frozen["write_basis"]) / frozen["write_scale"],
             "effect": ep[("adapter7",)].numeric()[0] - endpoint.numeric()[0]}
    im = {"root_id": root, "query_index": query, "source_identity": source["source_identity"],
          "original_arrays": {"noise": audit[("explicit_noise",)].input_identity(),
                              "decode_state": audit[("consumed_decode_state",)].input_identity(),
                              "old_KT": vt.input_identity(), "old_VT": vt.input_identity(),
                              "actual_E": {arm: e.input_identity() for arm in ("S", "P")},
                              "modified_v": {arm: vm.input_identity() for arm in ("S", "P")}},
          "sources": {}}
    obs = {"root_id": root, "query_index": query,
           "arms": {arm: {"adapter7_effect": (cache["effect"][None] * ratio).tolist()}
                    for arm, ratio in (("P", 1.), ("S", .5))},
           "S_minus_R_adapter7": (cache["effect"][None] * .25).tolist()}
    return old, new, np_, im, {("prefix_mask",): mask, ("prefix_positions",): pos}, wo, frozen, cache, obs


def stage_fixture(path):
    """Full 8-case synthetic execution fixture; no actual NPZ is opened."""
    import adapter as D
    path = Path(path)
    path.mkdir()
    actual_spec = C.json_read((Path(__file__).parent / "INPUTS.json").read_bytes())
    entries = actual_spec["native72_plan_entries"]
    refs, samples, obs_all, cache_all, ids, old_source = [], [], [], [], [], []
    def save(name, raw, role="synthetic"):
        (path / name).write_bytes(raw)
        r = {"path": name, "logical_path": name, "uri": "synthetic://" + name,
             "bytes": len(raw), "sha256": C.sha(raw), "role": role}
        refs.append(r)
        return r
    def js(name, value):
        return save(name, D.encoded(value))
    def npz(name, value):
        buf = io.BytesIO()
        np.savez_compressed(buf, **value)
        return save(name, buf.getvalue())
    for index, (root, query) in enumerate(A.KEYS):
        old, n, p, im, ia, wo, frozen, cache, obs = fixture(root, query)
        spec = {"root_id": root, "query_index": query, "source_identity": im["source_identity"],
                "input_identity": f"synthetic_input_{index}", "edit_identities": {"shared": "synthetic_E"},
                "old": {}, "new": {}}
        local = {}
        for role in ("audit", "natural_C", "natural_T", "edits"):
            local[role] = save(f"old_{index}_{role}.npz", typed_bytes(*old[role]))
        old["effect_shared"][0].update(
            baseline_refs={"T": local["natural_T"]}, edit_ref=local["edits"])
        local["effect_shared"] = save(f"old_{index}_effect_shared.npz", typed_bytes(*old["effect_shared"]))
        spec["old"] = {k: r["path"] for k, r in local.items()}
        im["sources"] = {"native": local,
                         "original_pulse_label_ids": {"shared": {"input_identity": spec["input_identity"],
                                                               "edit_identity": "synthetic_E"}}}
        inp = save(f"input_{index}.npz", typed_bytes(im, ia))
        spec["new"]["input"] = inp["path"]
        for arm, role, arrays in (("N", "baseline_none_1", n), ("P", "shared_persistent", p)):
            entry = next(e for e in entries if (e["root_id"], e["query_index"], e["role"]) == (root, query, role))
            spec[arm + "_entry"] = entry
            meta = {"schema": "rss-pulse-compact-call-v1", "entry": entry,
                    "input_ref": inp, "native_validation_errors": [],
                    "provenance": {"source_identity": im["source_identity"], "native_source_unchanged": True}}
            call = save(f"call_{index}_{arm}.npz", typed_bytes(meta, arrays))
            event = js(f"checks_{index}_{arm}.json",
                       {"schema": "rss-pulse-compact-event-v1", "event": "CALL_CHECKS",
                        "payload": {"entry": entry, "passed": True, "output": call}})
            spec["new"][arm], spec["new"][arm + "_checks"] = call["path"], event["path"]
        samples.append(spec)
        obs_all.append(obs)
        cache_all.append(cache)
        ids.append({"key": [root, query, "shared", "KT"],
                    "provenance": {"input_identity": spec["input_identity"]},
                    "arrays": {k: D.aid(v) for k, v in cache.items()}})
        old_source.append({"root_id": root, "query_index": query,
                           "query_fingerprint": spec["input_identity"],
                           "input_source_identity": spec["source_identity"]})
    common = {}
    common["WO"] = save("wo.npz", typed_bytes({}, {("native_L9_WO",): wo}))["path"]
    cache = {k: np.stack([c[k] for c in cache_all]) for k in cache_all[0]}
    cache["row_keys"] = np.asarray([[r, str(q), "shared", "KT"] for r, q in A.KEYS])
    cache["split"] = np.asarray(["DEV"] * 8)
    common["cache"] = npz("cache.npz", cache)["path"]
    common["cache_ids"] = js("cache_ids.json", ids)["path"]
    common["frozen_arrays"] = npz("frozen.npz", frozen)["path"]
    fm = {"schema": "task-blind-frozen-model-v1", "arrays": {k: D.aid(v) for k, v in frozen.items()}}
    common["frozen_manifest"] = js("frozen.json", {"synthetic": False, "model": fm,
                                                        "model_identity": C.sha(D.encoded(fm))})["path"]
    binding = {"model_bound": True, "adapter_bound": True, "normalization_sha256": "synthetic_norm",
               "loaded_parameter_tree_sha256": "synthetic_model"}
    model_ref = js("model.json", binding)
    common["MODEL_BINDING"] = model_ref["path"]
    common["SOURCE_BOUND_event"] = js("source.json", {"event": "SOURCE_BOUND", "payload": {
        "source_bound": True, "model_bound": True, "adapter_bound": True, "inputs_bound": True,
        "synthetic": False, "source_manifest": {"sha256": "synthetic_source"}, "model_binding": model_ref}})["path"]
    common["old_source_proof"] = js("old_source.json", {"all_rows": old_source})["path"]
    lists = {k: [e["call_id"] for e in entries] for k in
             ("charged_calls", "completed_calls", "passed_calls", "returned_native_calls")}
    result = {"run_id": A.RUN, "status": "DEV_PULSE_NATIVE_COMPLETE", "synthetic": False,
              "worker_complete": True, "model_binding": binding,
              "source_manifest_sha256": "synthetic_source", **lists}
    term = {"unexecuted_call_ids": [], "stopped_on_storage_failure": False,
            "result": {"complete": True, "qualification_complete": True, "failures": [],
                       "observability": obs_all, **lists}}
    common["RESULT.json"] = js("result.json", result)["path"]
    common["TERMINAL.json"] = js("terminal.json", term)["path"]
    return {"producer_run": A.RUN, "producer_job": "redacted-f594f0253751", "input_objects": refs,
            "input_object_count": len(refs), "input_body_bytes": sum(r["bytes"] for r in refs),
            "snapshots": samples, "common": common, "native72_plan_entries": entries}
