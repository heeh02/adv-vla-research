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


