"""Actual measured-field bridge, never a whole-graph/prediction acceptance."""
import json
import math
import numpy as np
from codec import Leaf, from_array, need, sha

RUN = "jinwen-paper2-dev-pulse-native-20261020-v2"
KEYS = tuple((f"SDRU-CDEV-T0{t}-R0{r}", q) for t in range(2) for r in range(2)
             for q in (11, 22))
DIMS = tuple("translation_command_" + s for s in "xyz") + tuple(
    "rotation_command_" + s for s in "xyz") + ("gripper_command",)


def compare(left, right):
    need(left.storage.shape == right.storage.shape, "paired_field_shape")
    a, b = left.numeric().astype(np.float64), right.numeric().astype(np.float64)
    need(a.size and np.isfinite(a).all() and np.isfinite(b).all(), "finite_paired_field")
    delta = b - a
    rr, cr, dr = (float(np.sqrt(np.mean(x * x))) for x in (a, b, delta))
    return {"reference": left.identity(), "candidate": right.identity(),
            "typed_exact": left.identity() == right.identity(),
            "raw_payload_exact": left.identity()["payload_sha256"] == right.identity()["payload_sha256"],
            "numeric_equal": bool(np.array_equal(a, b)),
            "numeric_nonzero_differences": int(np.count_nonzero(delta)),
            "max_abs": float(np.max(np.abs(delta))), "RMS_difference": dr,
            "RMS_reference": rr, "RMS_candidate": cr,
            "relative_RMS_to_reference": dr / rr if rr else None,
            "reference_zero": rr == 0,
            "relative_RMS_to_candidate": dr / cr if cr else None,
            "candidate_zero": cr == 0}


def refsame(a, b):
    return all(a.get(k) == b.get(k) for k in ("bytes", "sha256")) and (
        a.get("path", "").rsplit("/", 1)[-1] == b.get("logical_path", b.get("path", "")).rsplit("/", 1)[-1])


def query_source_identity(saved_source, c_ref, t_ref):
    """Original dev_reader Query source domain; distinct from saved input."""
    def receipt(r):
        return {"path": r.get("logical_path", r["path"]),
                "bytes": r["bytes"], "sha256": r["sha256"]}
    body = {"input_source_identity": saved_source, "natural_C_receipt": receipt(c_ref),
            "natural_T_receipt": receipt(t_ref), "layer_index": 8,
            "first_read_call": 5, "flow_sites": [0, 5]}
    return sha((json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode())


def validate_source_domains(spec, source, refs):
    saved = spec["saved_input_source_identity"]
    historical = spec["historical_query_source_identity"]
    need(source["parent_source"]["source_identity"] == saved, "saved_input_source_domain")
    need(source["input_source_identity"] == historical, "historical_query_source_domain")
    need(query_source_identity(saved, refs[spec["old"]["natural_C"]],
                               refs[spec["old"]["natural_T"]]) == historical,
         "query_source_domain_reconstruction")


def field(leaves, path, shape, logical):
    a = leaves[tuple(path)]
    need(a.storage.shape == shape and a.logical == logical, "native_field_ABI")
    return a


def proxy(p, e, wo, gate):
    # Same contraction order as the independently archived signed cache.
    p, e, wo, gate = (np.asarray(x, np.float64) for x in (p, e, wo, gate))
    read = np.einsum("hpn,nd->hpd", p[:, :, :len(e)], e, optimize=True)
    intermediate = np.einsum("hpn,nd->phd", p[:, :, :len(e)], e)
    write = np.einsum("phd,hdf->pf", intermediate, wo) * gate[None, :]
    return read, write


def old_context(c, t, audit, frozen, query):
    vector = np.concatenate([leaves[("flow", k)].numeric()[j].ravel().astype(np.float64)
                             for leaves in (c, t)
                             for k, j in (("x", 0), ("velocity_native", 0),
                                          ("x", 5), ("velocity_native", 5))] +
                            [audit[("consumed_decode_state",)].numeric().ravel().astype(np.float64)])
    delta = np.subtract(t[("flow", "velocity_native")].numeric()[0].astype(np.float32),
                        c[("flow", "velocity_native")].numeric()[0].astype(np.float32),
                        dtype=np.float32).ravel().astype(np.float64)
    def apply(x, prefix):
        return ((x - frozen[prefix + "_mean"]) @ frozen[prefix + "_basis"]) / frozen[prefix + "_scale"]
    return np.concatenate([apply(vector, "state"), apply(delta, "first_flow"), [5 * query / 520.]])


def validate_terminal(terminal, result, entries):
    want = [r["call_id"] for r in entries]
    need(len(want) == len(set(want)) == 72, "exact_native72_roster")
    r = terminal["result"]
    need(result["run_id"] == RUN and result["status"] == "DEV_PULSE_NATIVE_COMPLETE" and
         result["synthetic"] is False and result["worker_complete"] is True and
         r["complete"] is True and r["qualification_complete"] is True and
         r["failures"] == [], "complete_real_native72")
    for name in ("charged_calls", "completed_calls", "passed_calls", "returned_native_calls"):
        need(result[name] == want and r[name] == want, "exact_complete_native_call_lists")
    need(terminal["unexecuted_call_ids"] == [] and not terminal["stopped_on_storage_failure"],
         "complete_native_storage")
    need(len(r["observability"]) == 8, "exact8_observability_rows")
    obs = {(o["root_id"], o["query_index"]): o for o in r["observability"]}
    need(set(obs) == set(KEYS) and len(obs) == 8, "all8_observability_rows")
    return obs


def validate_call(meta, event, entry, inp_ref, call_ref):
    need(meta["schema"] == "rss-pulse-compact-call-v1" and meta["entry"] == entry and
         meta["native_validation_errors"] == [], "exact_compact_call_metadata")
    need(refsame(meta["input_ref"], inp_ref), "exact_call_input_reference")
    need(event["schema"] == "rss-pulse-compact-event-v1" and event["event"] == "CALL_CHECKS",
         "exact_CALL_CHECKS_event")
    p = event["payload"]
    need(p["entry"] == entry and p["passed"] is True and refsame(p["output"], call_ref),
         "later_passed_checks_exact_output")
    # Embedded pending_native_checks is intentionally not reinterpreted.


def one_snapshot(spec, old, new, inp, cache, cache_index, frozen, wo, obs, refs):
    root, query = spec["root_id"], spec["query_index"]
    om = {k: v[0] for k, v in old.items()}
    oa = {k: v[1] for k, v in old.items()}
    tm, t = old["natural_T"]
    cm, c = old["natural_C"]
    im, ia = inp
    nm, n = new["N"]
    pm, pcall = new["P"]
    for name, m in om.items():
        need((m["root_id"], m["query_index"]) == (root, query), "old_record_key")
    need(tm["role"] == "natural_T" and cm["role"] == "natural_C" and
         om["effect_shared"]["role"] == "effect_shared", "old_record_roles")
    need(im["root_id"] == root and im["query_index"] == query and
         im["source_identity"] == spec["saved_input_source_identity"], "new_input_key_source")
    need(om["audit"]["source"]["source_identity"] == spec["saved_input_source_identity"] and
         all(om[k]["source_identity"] == spec["saved_input_source_identity"] for k in
             ("natural_C", "natural_T", "effect_shared")), "old_call_source")
    need(om["audit"]["source"]["root_id"] == root and
         om["audit"]["source"]["query_index"] == query and
         om["audit"]["source"]["env_step_index"] == 5 * query, "original_policy_clock")
    for k in ("natural_C", "natural_T", "effect_shared", "edits", "audit"):
        need(refsame(im["sources"]["native"][k], refs[spec["old"][k]]), "old_ref_in_new_input")
    need(im["sources"]["original_pulse_label_ids"]["shared"] ==
         {"input_identity": spec["input_identity"], "edit_identity": spec["edit_identities"]["shared"]},
         "exact_old_query_edit_fingerprint")
    audit = oa["audit"]
    for key, path in (("noise", ("explicit_noise",)), ("decode_state", ("consumed_decode_state",))):
        need(im["original_arrays"][key] == audit[path].input_identity(), "same_original_input_array_identity")
    for key, path in (("old_KT", ("receiver_L9_K",)), ("old_VT", ("receiver_L9_V",))):
        need(im["original_arrays"][key] == t[path].input_identity(), "same_original_prefix_identity")
    N = t[("prefix_mask",)].storage.shape[-1]
    need(N == 968 or spec.get("_synthetic") is True, "actual_prefix968")
    # Both old and new payloads use the original BF16 Vmod and FP32 subtraction.
    e = field(oa["edits"], ("actual_E", "shared"), (1, N, 1, 256), "float32")
    vm = field(oa["edits"], ("Vmod", "shared"), (1, N, 1, 256), "bfloat16")
    vt = field(t, ("receiver_L9_V",), (1, N, 1, 256), "bfloat16")
    need(np.array_equal(e.numeric(), np.subtract(vm.numeric(), vt.numeric(), dtype=np.float32)),
         "actual_E_from_BF16_payload")
    for arm in ("S", "P"):
        need(im["original_arrays"]["actual_E"][arm] == e.input_identity() and
             im["original_arrays"]["modified_v"][arm] == vm.input_identity(), "same_new_actual_E_and_payload")
    sel = om["effect_shared"]["technical"]["selectors"]
    need(sel == {"kv_mask": [False, True], "layer_mask": [i == 8 for i in range(18)],
                 "flow_mask": [i >= 5 for i in range(10)], "use_vmod": True}, "old_P_is_persistent")
    need(refsame(om["effect_shared"]["baseline_refs"]["T"], refs[spec["old"]["natural_T"]]) and
         refsame(om["effect_shared"]["edit_ref"], refs[spec["old"]["edits"]]), "old_P_own_baseline_edit")
    fields = []
    def add(name, left, right):
        fields.append({"root_id": root, "query_index": query, "field": name, **compare(left, right)})
    for label, key, j, logical in (("x0", "x", 0, "float32"), ("v0", "velocity_native", 0, "bfloat16"),
                                    ("x5", "x", 5, "float32"), ("v5", "velocity_native", 5, "bfloat16")):
        left, right = t[("flow", key)].select(j), n[("flow__" + key,)].select(j)
        need(left.logical == right.logical == logical and left.storage.shape == right.storage.shape ==
             (1, 10, 32), "flow_ABI")
        add("T_" + label, left, right)
    mappings = (
        ("T_P_j5_L9", ("first_read", "probabilities_native"), ("j5_L9_probabilities_native",), None, (1, 1, 8, 10, N + 10), "bfloat16"),
        ("T_gate_j5_L9", ("first_read", "attention_gate"), ("reads__attention_gate",), (0, 1), (1, 1, 1024), "bfloat16"),
        ("T_mask_j5", ("first_read", "mask"), ("j5_mask",), None, (1, 10, N + 10), "bool"),
        ("T_suffix_positions_j5", ("first_read", "suffix_positions"), ("j5_suffix_positions",), None, (1, 10), "int32"))
    for label, op, np_, ix, shape, dtype in mappings:
        left, right = field(t, op, shape, dtype), n[np_]
        if ix is not None:
            right = right.select(ix)
        need(right.storage.shape == shape and right.logical == dtype, "new_read_ABI")
        add(label, left, right)
    for key, logical in (("time", "float32"), ("time_weak_type", "bool")):
        left, right = t[("flow", key)].select([0, 5]), n[("flow__" + key,)].select([0, 5])
        need(left.storage.shape == right.storage.shape == (2,) and left.logical == right.logical == logical,
             "native_clock_ABI")
        add(key + "_j0_j5", left, right)
    for key, shape, dtype in (("prefix_mask", (1, N), "bool"), ("prefix_positions", (1, N), "int32")):
        add(key, field(t, (key,), shape, dtype), field(ia, (key,), shape, dtype))
    po = t[("first_read", "probabilities_native")].numeric()[0, 0]
    pn = n[("j5_L9_probabilities_native",)].numeric()[0, 0]
    go = t[("first_read", "attention_gate")].numeric()[0, 0]
    gn = n[("reads__attention_gate",)].numeric()[0, 1, 0, 0]
    eo = e.numeric()[0, :, 0]
    ro, wo_proxy = proxy(po, eo, wo, go)
    rn, wn_proxy = proxy(pn, eo, wo, gn)
    i = cache_index[(root, query, "shared", "KT")]
    need(str(cache["split"][i]) == "DEV", "old_cache_DEV_role")
    feature_rows = []
    def feature(name, left, right, scope):
        feature_rows.append({"root_id": root, "query_index": query, "feature": name,
                             "scope": scope, "mixed_units_physical_magnitude": False,
                             "magnitude_scope": "numeric_algebra_or_cache_QA_only",
                             **compare(from_array(left), from_array(right))})
    feature("cache_headpos", cache["current_headpos_read"][i], ro, "old_cache_vs_old_reconstruction")
    feature("cache_full_write", cache["current_gated_write"][i], wo_proxy, "old_cache_vs_old_reconstruction")
    context = old_context(c, t, audit, frozen, query)
    write4 = ((wo_proxy.ravel() - frozen["write_mean"]) @ frozen["write_basis"]) / frozen["write_scale"]
    feature("cache_c9", cache["context9"][i], context, "old_cache_vs_old_reconstruction")
    feature("cache_current4", cache["current4"][i], write4, "old_cache_vs_old_reconstruction")
    feature("current_headpos", ro, rn, "old_T_vs_new_N_same_E_WO_FP64_proxy")
    feature("current_full_write", wo_proxy, wn_proxy, "old_T_vs_new_N_same_E_WO_FP64_proxy")
    old_n = field(t, ("adapter7",), (1, 10, 7), "float64").numeric()[0]
    old_p = field(oa["effect_shared"], ("adapter7",), (1, 10, 7), "float64").numeric()[0]
    new_n = field(n, ("adapter7",), (1, 10, 7), "float64").numeric()[0]
    new_p = field(pcall, ("adapter7",), (1, 10, 7), "float64").numeric()[0]
    old_y, new_y = old_p - old_n, new_p - new_n
    need(np.array_equal(new_y[None], np.asarray(obs["arms"]["P"]["adapter7_effect"], np.float64)),
         "terminal_P_effect_matches_actual_P_minus_own_N")
    feature("cache_old_P_effect", cache["effect"][i], old_y, "old_cache_target_vs_own_baseline_reconstruction")
    s = np.asarray(obs["arms"]["S"]["adapter7_effect"], np.float64)[0]
    sr = np.asarray(obs["S_minus_R_adapter7"], np.float64)[0]
    need(s.shape == sr.shape == (10, 7) and np.isfinite(s).all() and np.isfinite(sr).all(),
         "complete_finite_pulse_signal_arrays")
    endpoint_rows = []
    for window, end in (("first5", 5), ("full10", 10)):
        for d, dim in enumerate(DIMS):
            for name, left, right in (("none_endpoint", old_n, new_n),
                                      ("persistent_endpoint", old_p, new_p),
                                      ("persistent_effect_own_baselines", old_y, new_y)):
                row = {"root_id": root, "query_index": query, "window": window, "dimension": dim,
                       "comparison": name, **compare(from_array(left[:end, d]), from_array(right[:end, d]))}
                if name == "persistent_effect_own_baselines":
                    srms, c_rms = (float(np.sqrt(np.mean(x[:end, d] ** 2))) for x in (s, sr))
                    row.update(new_S_RMS=srms, new_S_minus_R_RMS=c_rms,
                               residual_RMS_over_new_S_RMS=row["RMS_difference"] / srms if srms else None,
                               residual_RMS_over_new_S_minus_R_RMS=row["RMS_difference"] / c_rms if c_rms else None,
                               new_S_zero=srms == 0, new_S_minus_R_zero=c_rms == 0)
                endpoint_rows.append(row)
    # Raw32 is a different command space; retain all dimensions individually.
    for dim in range(32):
        for name, left, right in (("none_raw32", t, n), ("persistent_raw32", oa["effect_shared"], pcall)):
            l = field(left, ("raw32",), (1, 10, 32), "float32").select((0, slice(None), dim))
            r = field(right, ("raw32",), (1, 10, 32), "float32").select((0, slice(None), dim))
            endpoint_rows.append({"root_id": root, "query_index": query, "window": "full10",
                                  "dimension": f"normalized_raw_coordinate_{dim}", "comparison": name, **compare(l, r)})
    decisions = []
    for row in fields:
        decisions.append({"root_id": root, "query_index": query, "object": row["field"],
                          "status": "VALID_MEASURED_EXACT" if row["typed_exact"] else "MEASURED_DIFFERENCE",
                          "meaning": "measured_field_typed_exact" if row["typed_exact"] else "measured_field_not_typed_exact"})
    decisions += [
        {"root_id": root, "query_index": query, "object": "old_c9_current4",
         "status": "HISTORICAL_ONLY", "meaning": "full_DEV_frozen_PCA_not_new_root_fold_features"},
        {"root_id": root, "query_index": query, "object": "fresh_C_AE_context",
         "status": "UNMEASURED", "meaning": "not_collected_no_T_to_C_extrapolation"},
        {"root_id": root, "query_index": query, "object": "whole_graph_equivalence",
         "status": "UNMEASURED", "meaning": "subset_parity_cannot_prove_whole_graph"},
        {"root_id": root, "query_index": query, "object": "pulse_prediction",
         "status": "UNMEASURED", "meaning": "no_fit_or_prediction_test_in_this_bridge"}]
    return fields, feature_rows, endpoint_rows, decisions
