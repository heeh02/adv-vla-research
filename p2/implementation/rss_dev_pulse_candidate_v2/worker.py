"""DEV orchestration around the unchanged native graph; no loader or launcher.

Production requires an independently bound loaded campaign, source guard and
explicit future GPU authority. CPU qualification never calls this path.
"""
import hashlib
import importlib.util
import pickle
from pathlib import Path
import random
import sys
import time
import traceback
import numpy as np

HERE = Path(__file__).resolve().parent


def sibling(name):
    """Do not borrow a cloud parent's unrelated top-level `contract` module."""
    key = "_rss_DEV_pulse_worker_" + name
    spec = importlib.util.spec_from_file_location(key, HERE / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[key] = module
    spec.loader.exec_module(module)
    return module


C = sibling("contract")
native_bridge = sibling("native_bridge")
LoadedContextGuard = sibling("runtime_guard").LoadedContextGuard
NATIVE = native_bridge.load()
Q = NATIVE["candidate_query"]
OLD = NATIVE["native_worker"]


def input_module():
    key = "_rss_dev_pulse_saved_inputs"
    if key not in sys.modules:
        spec = importlib.util.spec_from_file_location(key, HERE / "input/adapter.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[key] = module
        spec.loader.exec_module(module)
    return sys.modules[key]


def rng():
    return {k: hashlib.sha256(pickle.dumps(v, protocol=4)).hexdigest()
            for k, v in (("python", random.getstate()), ("numpy", np.random.get_state()))}


def before_edit_exact(none, edited):
    """No extra calls: effects must begin at the declared j5/L9 setter."""
    a, b = none["metadata"], edited["metadata"]
    for field, count in (("x", 6), ("velocity_native", 5)):
        C.need(OLD.I.exact(a["flow"][field][:count], b["flow"][field][:count]),
               "pre_j5_unchanged_" + field)
    for field in Q.FIELDS:
        C.need(OLD.I.exact(a["reads"][field][0, 0], b["reads"][field][0, 0]),
               "j5_L1_unchanged_" + field)
    for field in ("query_post_rope_scaled", "probabilities_native",
                  "receiver_residual_input", "prefix_key_used",
                  "suffix_key_post_rope", "suffix_value", "attention_gate"):
        C.need(OLD.I.exact(a["reads"][field][0, 1], b["reads"][field][0, 1]),
               "j5_L9_pre_attention_unchanged_" + field)
    return True


def difference(left, right):
    """Descriptive numeric changes, separate from typed byte identity."""
    a, b = np.asarray(left), np.asarray(right)
    C.need(a.shape == b.shape and a.dtype == b.dtype, "effect_same_typed_support")
    C.need(np.isfinite(a).all() and np.isfinite(b).all(), "effect_finite")
    delta = a.astype(np.float64) - b.astype(np.float64)
    ab = np.ascontiguousarray(a).view(np.uint8).reshape(-1, a.dtype.itemsize)
    bb = np.ascontiguousarray(b).view(np.uint8).reshape(-1, b.dtype.itemsize)
    return dict(shape=list(a.shape), logical_dtype=a.dtype.name,
                numeric_nonzero_coordinates=int(np.count_nonzero(delta)),
                bitwise_changed_coordinates=int(np.count_nonzero(np.any(ab != bb, axis=1))),
                coordinates=a.size, squared_norm=float(np.sum(delta * delta)),
                maximum_absolute_difference=float(np.max(np.abs(delta))) if delta.size else 0.,
                magnitude_scope="stored-coordinate numerical diagnostic; mixed command units are not a physical norm",
                physical_meaningful_threshold=None,
                numeric_nonzero_is_not_physical_motion=True)


def dose_summary(saved):
    """Keep token arrays in their immutable original record and input proof."""
    dose = saved.dose
    residual = np.asarray(dose["random_match"]["token_residual"], dtype=np.float64)
    valid = np.asarray(saved.prefix_mask)[0]
    C.need(residual.shape == valid.shape and np.isfinite(residual).all(),
           "dose_summary_exact_token_support")
    return {
        "shared_global_norm": dose["shared"]["global_norm"],
        "random_global_norm": dose["random_matched"]["global_norm"],
        "global_random_minus_shared_norm": dose["global_random_minus_shared_norm"],
        "exact_actual_dose_match": dose["exact_actual_dose_match"],
        "valid_tokens": int(valid.sum()),
        "nonzero_valid_token_residuals": int(np.count_nonzero(residual[valid])),
        "maximum_absolute_valid_token_residual": (
            float(np.max(np.abs(residual[valid]))) if valid.any() else None),
        "full_report_sha256": C.identity(dose),
        "original_edit_record": saved.sources["native"]["edits"],
        "full_token_arrays": "preserved in original edit record and CPU input qualification proof",
        "redraw_or_rescale": False,
    }


def observability(saved, results):
    n = results["baseline_none_1"]
    arms = {}
    for arm, role in (("S", "shared_pulse"), ("R", "random_pulse"),
                      ("P", "shared_persistent")):
        actual, m, base = results[role], results[role]["metadata"], n["metadata"]
        stages = {
            "L9_j5_write_pre_gate": difference(
                m["reads"]["attention_output_post_WO_pre_gate"][0, 1],
                base["reads"]["attention_output_post_WO_pre_gate"][0, 1]),
            "L9_j5_post_attention_residual": difference(
                m["reads"]["hidden_after_attention_residual"][0, 1],
                base["reads"]["hidden_after_attention_residual"][0, 1]),
            "L9_j5_post_block": difference(m["reads"]["hidden_post_block"][0, 1],
                                         base["reads"]["hidden_post_block"][0, 1]),
            "velocity_j5": difference(m["flow"]["velocity_native"][5],
                                      base["flow"]["velocity_native"][5]),
            "x6": difference(m["flow"]["x"][6], base["flow"]["x"][6]),
            "raw32_final": difference(actual["raw32"], n["raw32"]),
            "adapter7_first5": difference(actual["adapter7"][:, :5], n["adapter7"][:, :5]),
            "adapter7_full10": difference(actual["adapter7"], n["adapter7"]),
        }
        effect = np.asarray(saved.actual_E[arm])
        arms[arm] = {
            "actual_E_numeric_zero": bool(np.all(effect == 0)),
            "Vmod_bitwise_same_as_VT": OLD.I.exact(saved.modified_v[arm], saved.old_VT),
            "actual_E_squared_norm": float(np.sum(effect.astype(np.float64)**2)),
            "stage_changes": stages,
            "adapter7_effect": (actual["adapter7"].astype(np.float64) -
                                n["adapter7"].astype(np.float64)).tolist(),
            "native_compute_precision_changed": False,
        }
    s, r, p = (results[k]["adapter7"] for k in
               ("shared_pulse", "random_pulse", "shared_persistent"))
    return {
        "root_id": saved.root_id, "query_index": saved.query_index, "arms": arms,
        "S_minus_R": difference(s, r), "P_minus_S": difference(p, s),
        "S_minus_R_adapter7": (s.astype(np.float64)-r.astype(np.float64)).tolist(),
        "P_minus_S_adapter7": (p.astype(np.float64)-s.astype(np.float64)).tolist(),
        "actual_dose": dose_summary(saved),
        "statistics_scope": "one archived snapshot; not independent root/task confirmation",
        "qualification_status": "PROVISIONAL_UNTIL_TERMINAL_ALL72_CHECKS_PASS",
        "eligible_for_scientific_summary": False,
        "physical_motion_and_task_success": None,
        "P_minus_S_identifies_duration_or_feedback_alone": False,
        "R_delayed_repeat_performed": False,
    }


class Runtime:
    synthetic = False

    def __init__(self, campaign):
        self.campaign = campaign
        self.rt = Q.production_runtime(campaign)
        self.common_compiled = self.rt["compiled"]

    def encode(self, saved):
        return input_module().encode_loaded(self.campaign, saved)

    def compute(self, branches, saved, vmod, schedule):
        return self.rt["common"](branches["T"], branches["C"], saved.noise, vmod, schedule)

    def wait(self, value):
        return self.rt["wait"](value)

    def decode(self, raw, branches):
        return self.rt["decode"](raw, branches["T"])


def run(campaign, inputs, plan, *, ledger, source_guard, deadline_epoch,
        runtime=None, authority=None):
    keys = C.validate(plan)
    by = {x.key: x for x in inputs}
    C.need(len(by) == len(inputs) == 8 and set(by) == set(keys), "all_eight_inputs_no_survivor_selection")
    synthetic = runtime is not None and getattr(runtime, "synthetic", None) is True
    if not synthetic:
        C.require_native_authority(plan, authority)
        C.need(runtime is None, "production_runtime_cannot_be_replaced")
        C.need(type(source_guard) is LoadedContextGuard and not source_guard.synthetic,
               "concrete_loaded_native_context_guard_required")
    C.need(all(x.synthetic is synthetic for x in inputs), "real_synthetic_input_scope")
    rt = runtime
    result = dict(schema="DEV-pulse-worker-result-v1", status="FAILED_PRESERVED",
                  synthetic=synthetic, complete=False, qualification_complete=False,
                  initial_qualification_complete=False, planned_queries=plan["planned_queries"],
                  charged_calls=[], returned_native_calls=[], completed_calls=[],
                  passed_calls=[], failures=[],
                  observability=[], environment_steps=0, training_updates=0,
                  new_statistical_fits=0, R_delayed_repeat=False,
                  physical_meaningful_threshold=None, source_roster_native_qualified=False)
    result["graph_scope"] = "new execution of the pinned PVP G-star graph with fresh N/S/R/P"
    result["old_G_endpoint_bitwise_parity_claim"] = False
    result["old_persistent_results_relabelled_as_pulse"] = False
    contexts, actual, active = {}, None, None
    phase = "initial_source_guard"

    def check():
        C.need(time.time() < deadline_epoch, "absolute_science_deadline")

    def guard():
        proof = source_guard(campaign, inputs, plan, rt)
        C.need(all(proof.get(k) is True for k in
                   ("source_bound", "model_bound", "adapter_bound", "inputs_bound")),
               "bound_loaded_model_adapter_source_and_inputs")
        C.need(proof.get("synthetic", False) is synthetic, "source_guard_synthetic_scope")
        return proof

    try:
        check()
        rt = runtime if synthetic else Runtime(campaign)
        if not synthetic:
            source_guard.bind_runtime(rt)
            C.need(source_guard.production_bound, "actual_native_dispatch_bound_before_calls")
        ledger.event("SOURCE_BOUND", guard())
        for saved in inputs:
            check()
            branches = rt.encode(saved)
            contexts[saved.key] = {"saved": saved, "branches": branches, "results": {},
                                   "input_identity": input_module().identity(
                                       (branches, saved.noise, saved.modified_v))}
            ledger.persist_input(saved)
        for entry in plan["entries"]:
            check()
            active, actual, phase = entry, None, entry["phase"]
            ctx = contexts[(entry["root_id"], entry["query_index"])]
            saved, branches, results = ctx["saved"], ctx["branches"], ctx["results"]
            arm, role = entry["arm"], entry["role"]
            if phase != "qualification":
                C.need(result["initial_qualification_complete"], "all32_checks_before_effects")
            vmod = np.zeros_like(saved.old_VT) if arm == "N" else saved.modified_v[arm]
            before = input_module().identity((branches, saved.noise, vmod))
            checkpoints = {"before_compute": rng()}
            guard()
            ledger.charge(entry)
            result["charged_calls"].append(entry["call_id"])
            raw, metadata = rt.compute(branches, saved, vmod, entry["schedule"])
            rt.wait((raw, metadata))
            actual = {"raw32": np.asarray(raw), "metadata": OLD.host(metadata)}
            result["returned_native_calls"].append(entry["call_id"])
            checkpoints["after_compute"] = rng()
            adapter = rt.decode(raw, branches)
            rt.wait(adapter)
            checkpoints["after_decode"] = rng()
            actual["adapter7"] = np.asarray(adapter)
            m = actual["metadata"]
            # Preserve output and full leaf identities before any qualification
            # rejection, including an elapsed science deadline.
            ref = ledger.persist_call(entry, actual, {
                "source_identity": saved.source_identity,
                "original_sources": saved.sources, "native_source_unchanged": True,
                "finished_epoch": time.time()})
            result["completed_calls"].append(entry["call_id"])
            check()
            Q.check_result(m)
            input_module().current_prefix_check(saved, m, arm)
            C.need(actual["raw32"].shape == (1, 10, 32) and
                   actual["raw32"].dtype == np.float32 and
                   actual["adapter7"].shape == (1, 10, 7) and
                   actual["adapter7"].dtype == np.float64 and
                   np.isfinite(actual["raw32"]).all() and np.isfinite(actual["adapter7"]).all(),
                   "original_endpoint_ABI")
            C.need(OLD.I.exact(m["Vmod_argument"], vmod), "actual_frozen_payload_argument")
            C.need(OLD.I.exact(m["flow"]["x"][0], saved.noise), "actual_original_noise")
            C.need(before == input_module().identity((branches, saved.noise, vmod)),
                   "input_mutation")
            C.need(ctx["input_identity"] == input_module().identity(
                (branches, saved.noise, saved.modified_v)), "frozen_context_mutation")
            C.need(all(OLD.I.exact(m["selectors"][k], v) for k, v in zip(
                ("kv_mask", "layer_mask", "flow_mask", "use_vmod"), Q.selectors(entry["schedule"]))),
                "actual_dynamic_selectors")
            if role == "baseline_none_1":
                results[role] = actual
            else:
                n = results["baseline_none_1"]
                C.need(OLD.tree_exact(n["metadata"]["pg"], m["pg"]), "all_calls_same_actual_PG")
                if phase == "qualification":
                    C.need(OLD.result_exact(n, actual), "actual_repeat_and_both_nulls_exact")
                elif phase == "effect":
                    before_edit_exact(n, actual)
                    if OLD.I.exact(vmod, saved.old_VT):
                        C.need(OLD.result_exact(n, actual), "bitwise_own_null_effect_exact")
                    results[role] = actual
                    if role == "shared_persistent":
                        OLD.pulse_pair(results["shared_pulse"], actual)
                    if role == "random_pulse":
                        record = observability(saved, results)
                        ledger.event("OBSERVABILITY", record)
                        result["observability"].append(record)
                else:
                    original = role.removesuffix("_repeat")
                    C.need(OLD.result_exact(results[original], actual),
                           "effect_delayed_repeat_exact")
            checkpoints["after_persist_and_checks"] = rng()
            C.need(all(v["numpy"] == checkpoints["before_compute"]["numpy"]
                       for v in checkpoints.values()), "NumPy_RNG_mutation")
            guard()
            ledger.event("CALL_CHECKS", {"entry": entry, "passed": True, "output": ref,
                                       "RNG": checkpoints, "Python_RNG_policy": "diagnostic",
                                       "NumPy_RNG_policy": "strict_unchanged"})
            C.need(rng()["numpy"] == checkpoints["before_compute"]["numpy"],
                   "NumPy_RNG_mutation_after_checks")
            result["passed_calls"].append(entry["call_id"])
            if len(result["passed_calls"]) == 32:
                result["initial_qualification_complete"] = True
        result.update(complete=True, qualification_complete=True,
                      source_roster_native_qualified=not synthetic,
                      status="SYNTHETIC_WORKER_COMPLETE" if synthetic else "DEV_PULSE_NATIVE_COMPLETE")
    except BaseException as error:
        code = str(error)
        if len(code) > 160 or not all(c.isalnum() or c in "_:.-" for c in code):
            code = type(error).__name__
        failure = {"phase": phase, "call_id": active["call_id"] if active else None,
                   "type": type(error).__name__, "code": code,
                   "frames": [{"file": Path(f.filename).name, "function": f.name, "line": f.lineno}
                              for f in traceback.extract_tb(error.__traceback__)[-12:]]}
        if active is not None and actual is not None:
            if active["call_id"] not in getattr(ledger, "call_refs", {}):
                try:
                    failure["partial_output"] = ledger.persist_call(
                        active, actual, {"source_identity": saved.source_identity,
                                         "partial_after_failure": True,
                                         "adapter_missing": "adapter7" not in actual})
                except BaseException as partial_error:
                    failure["partial_preservation_status"] = type(partial_error).__name__
                    if hasattr(partial_error, "receipt"):
                        failure["partial_output_failure_receipt"] = partial_error.receipt
            try:
                failure["preserved"] = ledger.preserve_failure(active, actual, error)
            except BaseException as archive_error:
                failure["preservation_error"] = type(archive_error).__name__
        result["failures"].append(failure)
        result.update(status="FAILED_PRESERVED", complete=False, qualification_complete=False)
    finally:
        for record in result["observability"]:
            record["qualification_status"] = (
                "ALL72_CHECKS_PASSED" if result["complete"] else "INCOMPLETE_QUALIFICATION")
            record["eligible_for_scientific_summary"] = result["complete"] and not synthetic
        passed, charged = set(result["passed_calls"]), set(result["charged_calls"])
        result["query_status"] = []
        for root, q in keys:
            entries = [e for e in plan["entries"] if (e["root_id"], e["query_index"]) == (root, q)]
            result["query_status"].append(dict(
                root_id=root, query_index=q, planned_calls=9,
                charged_calls=sum(e["call_id"] in charged for e in entries),
                passed_calls=sum(e["call_id"] in passed for e in entries),
                unexecuted_call_ids=[e["call_id"] for e in entries if e["call_id"] not in charged],
                status="COMPLETE" if all(e["call_id"] in passed for e in entries)
                else "FAILED_OR_UNEXECUTED"))
        result["full_generations_charged"] = len(charged)
        result["prefix_prefills_charged"] = 2 * len(charged)
        result["suffix_evaluations_charged"] = 10 * len(charged)
        result["terminal_ref"] = ledger.finish(result)
    return result
