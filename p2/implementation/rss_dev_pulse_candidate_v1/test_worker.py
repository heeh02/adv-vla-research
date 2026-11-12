"""CPU-only orchestration tests; NumPy fake is not native numeric qualification.

The fake creates explicit shape-correct BF16/FP32 records and exercises the
unchanged native result/first-write comparators. It is not a JAX/model emulator.
"""
import copy
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import ml_dtypes
import contract as C
import worker as W

BF16 = ml_dtypes.bfloat16


def inputs(*, signed_zero_R=False):
    out = []
    for i, (root, query) in enumerate(C.KEYS):
        vt = np.full((1, 3, 1, 256), .25, BF16)
        vt[0, 0, 0, 0] = 0
        s = vt.copy()
        s[0, 0, 0, 1] += BF16(.125)
        r = vt.copy()
        if signed_zero_R:
            r.view(np.uint16)[0, 0, 0, 0] = 0x8000
        else:
            r[0, 0, 0, 2] += BF16(.25)
        vm = {"S": s, "R": r, "P": s, "own_null": vt}
        effects = {role: a.astype(np.float32)-vt.astype(np.float32) for role, a in vm.items()}
        ns = np.linalg.norm(effects["S"][0, :, 0].astype(np.float64), axis=-1)
        nr = np.linalg.norm(effects["R"][0, :, 0].astype(np.float64), axis=-1)
        dose = {"synthetic": True, "exact_norm_match_not_asserted": True,
                "shared": {"global_norm": float(np.linalg.norm(ns))},
                "random_matched": {"global_norm": float(np.linalg.norm(nr))},
                "random_match": {"token_residual": (nr-ns).tolist()},
                "global_random_minus_shared_norm": float(np.linalg.norm(nr)-np.linalg.norm(ns)),
                "exact_actual_dose_match": bool(np.array_equal(ns, nr))}
        out.append(SimpleNamespace(
            key=(root, query), root_id=root, query_index=query, synthetic=True,
            old_VT=vt, old_KT=np.full_like(vt, .5),
            noise=np.zeros((1, 10, 32), np.float32),
            prefix_mask=np.array([[True, True, False]], bool),
            prefix_positions=np.array([[0, 1, 1]], np.int32),
            modified_v=vm, actual_E=effects,
            dose=dose,
            source_identity=f"{i:064x}", sources={"synthetic_fixture": True,
                "native": {"edits": {"path": "synthetic-edit.npz", "bytes": 1, "sha256": "a"*64}}}))
    return out


class MemoryLedger:
    """Ordered, fail-preserving test sink; no real storage qualification claim."""
    def __init__(self, plan):
        self.plan, self.log, self.charged, self.persisted = plan, [], [], []
        self.inputs, self.finished, self.failures = [], None, []
        self.call_refs = {}

    def event(self, name, payload):
        self.log.append(("event", name, payload))

    def persist_input(self, saved):
        self.inputs.append(saved.key)
        self.log.append(("input", saved.key))

    def charge(self, entry):
        if entry != self.plan["entries"][len(self.charged)]:
            raise ValueError("test_exact_precharge_order")
        self.charged.append(copy.deepcopy(entry))
        self.log.append(("charge", entry["call_id"]))

    def persist_call(self, entry, actual, metadata):
        if entry["call_id"] in self.call_refs:
            raise ValueError("test_call_must_not_be_persisted_twice")
        self.persisted.append(entry["call_id"])
        self.log.append(("persist", entry["call_id"], set(actual)))
        ref = {"call_id": entry["call_id"], "synthetic_memory_only": True}
        self.call_refs[entry["call_id"]] = ref
        return ref

    def preserve_failure(self, entry, actual, error):
        record = {"call_id": entry["call_id"], "fields": sorted(actual),
                  "type": type(error).__name__, "synthetic_memory_only": True}
        self.failures.append(record)
        self.log.append(("failure", entry["call_id"], record))
        return record

    def finish(self, result):
        self.finished = copy.deepcopy(result)
        self.log.append(("finish",))
        return {"synthetic_memory_only": True}


class FakeRuntime:
    synthetic = True

    def __init__(self, ledger, fault=None):
        self.ledger, self.fault, self.calls = ledger, fault, []
        self.common_compiled = None

    def encode(self, saved):
        return {"C": {"marker": np.array([1.], np.float32)},
                "T": {"marker": np.array([2.], np.float32)}}

    def wait(self, value):
        return value

    def decode(self, raw, branches):
        return np.array(raw[:, :, :7], dtype=np.float64)

    def compute(self, branches, saved, vmod, schedule):
        entry = self.ledger.charged[-1]
        self.calls.append({
            "entry": copy.deepcopy(entry), "payload": vmod,
            "payload_was_original_object": (
                entry["arm"] == "N" or vmod is saved.modified_v[entry["arm"]])})
        n = saved.old_VT.shape[1]
        z = lambda shape: np.zeros(shape, BF16)
        shapes = {
            "query_post_rope_scaled": (5, 3, 1, 10, 1, 8, 256),
            "probabilities_native": (5, 3, 1, 1, 8, 10, n+10),
            "attention_output_post_WO_pre_gate": (5, 3, 1, 10, 1024),
            "attention_gate": (5, 3, 1, 1, 1024),
            "receiver_residual_input": (5, 3, 1, 10, 1024),
            "hidden_after_attention_residual": (5, 3, 1, 10, 1024),
            "hidden_post_block": (5, 3, 1, 10, 1024),
            "prefix_key_used": (5, 3, 1, n, 1, 256),
            "prefix_value_used": (5, 3, 1, n, 1, 256),
            "suffix_key_post_rope": (5, 3, 1, 10, 1, 256),
            "suffix_value": (5, 3, 1, 10, 1, 256),
        }
        reads = {name: z(shape) for name, shape in shapes.items()}
        reads["prefix_key_used"][:] = saved.old_KT
        reads["prefix_value_used"][:] = saved.old_VT
        reads["attention_gate"][:] = 1
        mask = np.broadcast_to(
            np.r_[saved.prefix_mask[0], np.ones(10, bool)], (5, 1, 10, n+10)).copy()
        reads.update(
            mask=mask,
            suffix_positions=np.broadcast_to(
                (int(saved.prefix_mask.sum())+np.arange(10, dtype=np.int32))[None],
                (5, 1, 10)).copy(),
            query_defined=np.ones((5, 1, 10), bool),
            adarms_condition=np.zeros((5, 1, 1024), np.float32),
            final_readout=z((5, 1, 10, 1024)))
        selectors = dict(zip(("kv_mask", "layer_mask", "flow_mask", "use_vmod"),
                             W.Q.selectors(schedule)))
        magnitude = float(np.sum(vmod.astype(np.float32)-saved.old_VT.astype(np.float32)))
        flow_x = np.zeros((11, 1, 10, 32), np.float32)
        flow_x[0] = saved.noise
        velocity = z((10, 1, 10, 32))
        for j in range(10):
            active = bool(selectors["flow_mask"][j])
            if active:
                reads["prefix_value_used"][j-5, 1] = vmod
                for field in ("attention_output_post_WO_pre_gate",
                              "hidden_after_attention_residual", "hidden_post_block"):
                    # L1 is upstream of the L9 edit and is deliberately unchanged.
                    reads[field][j-5, 1:, :, :, 0] = BF16(magnitude)
                velocity[j, :, :, 0] = BF16(magnitude)
            flow_x[j+1] = flow_x[j] + np.float32(-.1)*velocity[j].astype(np.float32)
        pg = {side: {
            "hidden_input": z((3, 1, n, 2048)),
            "hidden_pre_attention_norm": z((3, 1, n, 2048)),
            "cache_k": np.stack([saved.old_KT]*3),
            "cache_v": np.stack([saved.old_VT]*3),
            "selected_layers": np.array(W.Q.LAYERS, np.int32),
            "positions": saved.prefix_positions.copy(),
        } for side in ("C", "T")}
        meta = {
            "pg": pg, "reads": reads,
            "flow": {"x": flow_x, "velocity_native": velocity,
                     "time": (1-np.arange(11, dtype=np.float32)*np.float32(.1)).astype(np.float32),
                     "time_weak_type": np.array([True]+[False]*10, bool),
                     "dt": np.full(10, -.1, np.float32)},
            "gates": {name: np.array(True) for name in W.Q.GATES},
            "capture_layers": np.array(W.Q.LAYERS, np.int32),
            "capture_calls": np.array(W.Q.CALLS, np.int32),
            "capture_filled": np.ones(5, bool),
            "actual_prefix_selector_matches": np.ones((5, 3, 2), bool),
            "prefix_mask": saved.prefix_mask.copy(),
            "prefix_positions": saved.prefix_positions.copy(),
            "Vmod_argument": vmod.copy(), "selectors": selectors,
            "completed_updates": np.array(10, np.int32),
            "final_time": np.array(-1e-7, np.float32),
            "sampling_abi": {
                "num_steps": np.array(10, np.int32),
                "num_steps_is_trace_input": np.array(True),
                "num_steps_weak_type": np.array(False),
                "actual_dt": np.array(-.1, np.float32),
                "dt_weak_type": np.array(False)},
        }
        raw = flow_x[-1].copy()
        if self.fault is not None:
            self.fault(entry, saved, branches, raw, meta)
        return raw, meta


def guard(campaign, saved, plan, runtime):
    return dict(source_bound=True, model_bound=True, adapter_bound=True,
                inputs_bound=True, synthetic=True)


def plan():
    return json.loads((Path(__file__).parent/"PLAN-CANDIDATE.json").read_text())


def run(saved=None, fault=None, source_guard=guard, deadline=1e30, rt_class=FakeRuntime):
    config = plan()
    saved = inputs() if saved is None else saved
    ledger = MemoryLedger(config)
    runtime = rt_class(ledger, fault)
    result = W.run(None, saved, config, ledger=ledger, source_guard=source_guard,
                   deadline_epoch=deadline, runtime=runtime)
    return result, ledger, runtime


class Tests(unittest.TestCase):
    def test_complete72_order_all32_before_effects_and_exact_frozen_R(self):
        result, ledger, runtime = run()
        self.assertTrue(result["complete"], result["failures"])
        self.assertEqual(result["status"], "SYNTHETIC_WORKER_COMPLETE")
        self.assertFalse(result["source_roster_native_qualified"])
        self.assertEqual(result["full_generations_charged"], 72)
        self.assertEqual(result["prefix_prefills_charged"], 144)
        self.assertEqual(result["suffix_evaluations_charged"], 720)
        self.assertEqual([x["entry"] for x in runtime.calls], C.expected_entries())
        first_effect = next(i for i, x in enumerate(ledger.log) if x[:1] == ("charge",) and
                            "shared_pulse" in x[1])
        checks_before = [x for x in ledger.log[:first_effect]
                         if x[:2] == ("event", "CALL_CHECKS")]
        self.assertEqual(len(checks_before), 32)
        self.assertEqual(len(result["observability"]), 8)
        self.assertEqual(len(result["query_status"]), 8)
        self.assertTrue(all(q["planned_calls"] == q["charged_calls"] == q["passed_calls"] == 9
                            for q in result["query_status"]))
        self.assertTrue(all(x["payload_was_original_object"] for x in runtime.calls))
        self.assertEqual(sum(x["entry"]["arm"] == "R" for x in runtime.calls), 8)
        self.assertFalse(result["R_delayed_repeat"])
        self.assertEqual(len(ledger.persisted), 72)

    def test_each_active_own_null_schedule_failure_stops_before_any_effect(self):
        for role in ("V_own_null_pulse", "V_own_null_persistent"):
            with self.subTest(role=role):
                def fault(entry, saved, branches, raw, meta):
                    if entry["role"] == role:
                        raw[0, 0, 0] = 1
                result, ledger, runtime = run(fault=fault)
                self.assertFalse(result["complete"])
                self.assertFalse(result["initial_qualification_complete"])
                self.assertEqual(result["failures"][0]["code"], "actual_repeat_and_both_nulls_exact")
                self.assertTrue(all(c["entry"]["phase"] == "qualification" for c in runtime.calls))
                self.assertEqual(ledger.persisted[-1], result["failures"][0]["call_id"])
                self.assertEqual(ledger.failures[-1]["call_id"], ledger.persisted[-1])
                self.assertEqual(len(result["query_status"]), 8)

    def test_R_preedit_violation_is_preserved_and_denominator_stays_eight(self):
        def fault(entry, saved, branches, raw, meta):
            if entry["role"] == "random_pulse":
                meta["flow"]["x"][5, 0, 0, 0] = .25
        result, ledger, runtime = run(fault=fault)
        self.assertTrue(result["initial_qualification_complete"])
        self.assertFalse(result["complete"])
        self.assertEqual(result["failures"][0]["code"], "pre_j5_unchanged_x")
        self.assertEqual(len(result["charged_calls"]), 35)
        self.assertEqual(len(result["passed_calls"]), 34)
        self.assertEqual(ledger.persisted[-1], result["failures"][0]["call_id"])
        self.assertEqual(len(result["query_status"]), 8)
        self.assertEqual(len(result["observability"]), 0)

    def test_input_mutation_is_preserved_before_rejection(self):
        def fault(entry, saved, branches, raw, meta):
            if entry["role"] == "random_pulse":
                branches["T"]["marker"][0] = 123
        result, ledger, runtime = run(fault=fault)
        self.assertFalse(result["complete"])
        self.assertEqual(result["failures"][0]["code"], "input_mutation")
        self.assertEqual(ledger.persisted[-1], result["failures"][0]["call_id"])
        self.assertTrue(ledger.failures)

    def test_failed_loaded_source_guard_consumes_no_call_and_retains_all_snapshots(self):
        def bad(*args):
            return dict(source_bound=False, model_bound=True, adapter_bound=True,
                        inputs_bound=True, synthetic=True)
        result, ledger, runtime = run(source_guard=bad)
        self.assertFalse(result["complete"])
        self.assertEqual(result["failures"][0]["phase"], "initial_source_guard")
        self.assertEqual(result["full_generations_charged"], 0)
        self.assertEqual(len(result["query_status"]), 8)
        self.assertEqual(len(runtime.calls), 0)
        self.assertTrue(all(len(q["unexecuted_call_ids"]) == 9 for q in result["query_status"]))
        self.assertIsNotNone(ledger.finished)

    def test_numeric_zero_E_is_distinct_from_bitwise_identical_V(self):
        result, ledger, runtime = run(inputs(signed_zero_R=True))
        self.assertTrue(result["complete"], result["failures"])
        for row in result["observability"]:
            random = row["arms"]["R"]
            self.assertTrue(random["actual_E_numeric_zero"])
            self.assertFalse(random["Vmod_bitwise_same_as_VT"])
            self.assertEqual(random["stage_changes"]["adapter7_full10"]["numeric_nonzero_coordinates"], 0)
        a = np.array([-0.], np.float32)
        b = np.array([0.], np.float32)
        measured = W.difference(a, b)
        self.assertEqual(measured["numeric_nonzero_coordinates"], 0)
        self.assertEqual(measured["bitwise_changed_coordinates"], 1)

    def test_GPU0_native_entry_rejects_before_runtime_or_encoder_creation(self):
        config = plan()
        config["source_roster_pending_binding"] = False
        saved = inputs()
        for item in saved:
            item.synthetic = False
        with patch.object(W, "Runtime", side_effect=AssertionError("must_not_construct_model_runtime")) as constructor:
            with self.assertRaisesRegex(ValueError, "GPU0_requires"):
                W.run(None, saved, config, ledger=MemoryLedger(config), source_guard=guard,
                      deadline_epoch=1e30, runtime=None, authority=None)
            constructor.assert_not_called()

    def test_boolean_or_synthetic_guard_cannot_enter_production_runtime(self):
        config, saved = plan(), inputs()
        for item in saved:
            item.synthetic = False
        synthetic_guard = W.LoadedContextGuard.__new__(W.LoadedContextGuard)
        synthetic_guard._synthetic = True
        # Exercise the second admission boundary without manufacturing a real
        # authorization document or constructing a policy.
        for candidate in (guard, synthetic_guard):
            with self.subTest(candidate=type(candidate).__name__), \
                 patch.object(W.C, "require_native_authority", return_value=None), \
                 patch.object(W, "Runtime", side_effect=AssertionError("no_runtime")) as constructor:
                with self.assertRaisesRegex(ValueError, "concrete_loaded_native_context_guard"):
                    W.run(None, saved, config, ledger=MemoryLedger(config),
                          source_guard=candidate, deadline_epoch=1e30)
                constructor.assert_not_called()

    def test_foreign_parent_contract_cannot_replace_candidate_contract(self):
        foreign = SimpleNamespace(validate=lambda _: (_ for _ in ()).throw(AssertionError("foreign")))
        path = Path(W.__file__)
        with patch.dict(sys.modules, {"contract": foreign}):
            spec = importlib.util.spec_from_file_location("_worker_foreign_parent_test", path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            self.assertEqual(module.C.validate(plan()), C.KEYS)
            self.assertIsNot(module.C, foreign)
            self.assertEqual(Path(module.C.__file__).resolve(), path.with_name("contract.py").resolve())

    def test_deadline_after_compute_preserves_result_before_failure(self):
        clock = {"now": 1.}
        def fault(entry, saved, branches, raw, meta):
            clock["now"] = 11.
        with patch.object(W.time, "time", side_effect=lambda: clock["now"]):
            result, ledger, runtime = run(fault=fault, deadline=10.)
        self.assertFalse(result["complete"])
        self.assertEqual(result["failures"][0]["code"], "absolute_science_deadline")
        self.assertEqual(result["full_generations_charged"], 1)
        self.assertEqual(len(result["completed_calls"]), 1)
        self.assertEqual(ledger.persisted, result["completed_calls"])
        self.assertEqual(len(ledger.failures), 1)

    def test_compute_failure_keeps_charged_unknown_out_of_unexecuted(self):
        def fault(entry, saved, branches, raw, meta):
            raise RuntimeError("synthetic_compute_failure")
        result, ledger, runtime = run(fault=fault)
        self.assertEqual(result["full_generations_charged"], 1)
        self.assertEqual(result["completed_calls"], [])
        self.assertEqual(result["passed_calls"], [])
        self.assertEqual(len(result["query_status"]), 8)
        first = result["query_status"][0]
        self.assertEqual(first["charged_calls"], 1)
        self.assertEqual(len(first["unexecuted_call_ids"]), 8)
        self.assertNotIn(result["charged_calls"][0], first["unexecuted_call_ids"])
        self.assertEqual(first["status"], "FAILED_OR_UNEXECUTED")

    def test_initial_deadline_failure_returns_all_eight_with_zero_charge(self):
        result, ledger, runtime = run(deadline=-1.)
        self.assertFalse(result["complete"])
        self.assertEqual(result["failures"][0]["code"], "absolute_science_deadline")
        self.assertEqual(result["full_generations_charged"], 0)
        self.assertEqual(len(result["query_status"]), 8)
        self.assertEqual(runtime.calls, [])

    def test_decode_failure_preserves_returned_native_without_fabricated_adapter(self):
        class FailingDecoder(FakeRuntime):
            def decode(self, raw, branches):
                raise RuntimeError("synthetic_decode_failure")
        result, ledger, runtime = run(rt_class=FailingDecoder)
        self.assertFalse(result["complete"])
        self.assertEqual(result["failures"][0]["code"], "synthetic_decode_failure")
        self.assertEqual(len(result["returned_native_calls"]), 1)
        self.assertEqual(result["completed_calls"], [])
        self.assertEqual(result["passed_calls"], [])
        self.assertEqual(len(ledger.persisted), 1)
        self.assertEqual(ledger.failures[0]["fields"], ["metadata", "raw32"])
        self.assertEqual(len(result["query_status"]), 8)

    def test_late_repeat_failure_keeps_all_observability_provisional(self):
        def fault(entry, saved, branches, raw, meta):
            if entry["role"] == "shared_persistent_repeat":
                raw[0, 0, 1] = 1.
        result, ledger, runtime = run(fault=fault)
        self.assertFalse(result["complete"])
        self.assertEqual(result["failures"][0]["code"], "effect_delayed_repeat_exact")
        self.assertEqual(result["full_generations_charged"], 58)
        self.assertEqual(len(result["observability"]), 8)
        self.assertTrue(all(row["qualification_status"] == "INCOMPLETE_QUALIFICATION" and
                            not row["eligible_for_scientific_summary"]
                            for row in result["observability"]))

    def test_pre_L9_write_change_fails_even_when_flow_is_unchanged(self):
        def fault(entry, saved, branches, raw, meta):
            if entry["role"] == "random_pulse":
                meta["reads"]["attention_output_post_WO_pre_gate"][0, 0, 0, 0, 0] = BF16(1)
        result, ledger, runtime = run(fault=fault)
        self.assertFalse(result["complete"])
        self.assertEqual(result["failures"][0]["code"],
                         "j5_L1_unchanged_attention_output_post_WO_pre_gate")
        self.assertEqual(ledger.persisted[-1], result["failures"][0]["call_id"])

    def test_max968_token_dose_event_is_compact_and_keeps_full_report_identity(self):
        saved = inputs()[0]
        saved.prefix_mask = np.ones((1, 968), bool)
        report = saved.dose["random_match"]
        report["token_residual"] = (np.arange(968, dtype=float)/1000).tolist()
        for field in ("token_target_norm", "token_actual_norm", "token_scale"):
            report[field] = np.arange(968, dtype=float).tolist()
        for field in ("zero_target", "zero_direction_nonzero_target", "unbracketed"):
            report[field] = np.zeros(968, bool).tolist()
        summary = W.dose_summary(saved)
        self.assertEqual(summary["valid_tokens"], 968)
        self.assertEqual(summary["full_report_sha256"], C.identity(saved.dose))
        self.assertEqual(summary["nonzero_valid_token_residuals"], 967)
        self.assertNotIn("token_residual", summary)
        event = C.encoded({"schema": "rss-pulse-compact-event-v1",
                           "event": "OBSERVABILITY", "payload": {"actual_dose": summary}})
        self.assertLess(len(event), 4096)
        self.assertLess(len(event), 128*1024)


if __name__ == "__main__":
    unittest.main()
