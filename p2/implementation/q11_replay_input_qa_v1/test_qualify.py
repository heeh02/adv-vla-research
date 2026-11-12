"""Synthetic ledger and continuity failures; no actual archived array reads."""
import copy
import hashlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import numpy as np
import qualify as Q


class FakeSource:
    def __init__(self):
        vector = np.zeros(123, np.float64)
        aid = Q.identity(vector)
        self.selection = {"native_initial_vector": {
            "shape": aid["shape"], "dtype": aid["dtype"], "bytes": aid["bytes"],
            "raw_sha256": aid["sha256"]}}
        self.initial = [SimpleNamespace(event="ACTUAL_ROOT_INITIAL_VECTOR",
                                       metadata=self.selection, values={"initial_vector": vector})]
        self.records = {q: [] for q in range(13)}
        self.bad = None

    @staticmethod
    def state(step):
        vector = np.zeros(123, np.float64)
        vector[0] = step
        return {"flat_physics": vector, "sim_time": np.array(step / 20.0),
                "returned_eef_pos": np.array([step, 0, 0], np.float64),
                "returned_eef_quat": np.array([0, 0, 0, 1], np.float64)}

    def parent_actions(self, q):
        return np.full((1, 10, 7), q / 100, np.float64)

    def parent_raw32(self, q):
        return np.zeros((1, 10, 32), np.float32)

    def issued(self, q, slot):
        step = q * 5 + slot + 1
        values = {"action": self.parent_actions(q)[0, slot].copy(),
                  "before": self.state(step - 1), "after": self.state(step)}
        meta = {"query_index": q, "action_position": slot,
                "policy_step": step, "absolute_env_step_index": step + 10}
        if step == 23:
            if self.bad == "action":
                values["action"][0] += 1
            if self.bad == "chain":
                values["before"]["flat_physics"][2] = 1
            if self.bad == "clock":
                meta["absolute_env_step_index"] += 1
            if self.bad == "dtype":
                values["action"] = values["action"].astype(np.float32)
        return {"metadata": meta, "values": values, "source_receipt": "synthetic"}

    def outcome(self, q, slot):
        return {"query_index": q, "action_position": slot, "policy_step": q * 5 + slot + 1,
                "done": self.bad == "terminal" and q == 4, "official_success": False}

    def parent_audit(self, q):
        return {"input_unchanged": True, "environment_rng_restored": self.bad != "rng",
                "rng_changed_during_call": {"numpy": False}}

    def diagnostic(self, q):
        return {"before": {"physics.current": "equal"}, "after_restore": {
            "physics.current": "wrong" if self.bad == "protected" else "equal"},
            "forbidden_changes": [], "restore_errors": [], "observation_errors": []}

    def anchor(self, q):
        state = self.state(q * 5)
        if self.bad == "anchor" and q == 12:
            state["flat_physics"][3] = 1
        return {"physical_sim_state": state["flat_physics"], "observation": {},
                "source_receipts": {"pair": "synthetic", "clock": "synthetic"}}

    def initial_rng(self):
        return {"synthetic": True}

    def rng_id(self, value):
        return value


def ledger_fixture(stage):
    binding = {"root_id": Q.ROOT, "run_id": "synthetic-source", "source": "synthetic"}
    base = ("redacted-24fc043a6cae"
            "state-dependent-response-use-20261013-v1/synthetic-source/")
    refs, groups = [], []
    def put(uri, raw):
        path = hashlib.sha256(uri.encode()).hexdigest() + Path(uri).suffix
        (stage / path).write_bytes(raw)
        ref = {"path": path, "uri": uri, "bytes": len(raw), "sha256": Q.C.sha(raw)}
        refs.append(ref)
        return ref
    for sequence, q in enumerate([*range(13), None]):
        tail = "initialization" if q is None else f"query-{q:04d}"
        key = f"queries/{Q.ROOT}/{tail}/nonce{sequence}/"
        prefix = base + key
        array = np.array([1.0, 2.0], np.float64)
        tree = {"metadata": {"original_name": f"parent/query-{q}-input.npz",
                             "role": "synthetic_input"},
                "values": {"example": {"array": "a", "shape": [2], "logical_dtype": "float64",
                                       "sha256": Q.C.sha(array.tobytes())}}}
        stream = io.BytesIO()
        np.savez_compressed(stream, __schema__=np.frombuffer(Q.encoded(tree), np.uint8), a=array)
        filename = "000001-PRODUCER_ARRAYS.npz"
        member = put(prefix + filename, stream.getvalue())
        brief = {"path": filename, "bytes": member["bytes"], "sha256": member["sha256"]}
        common = {"binding": binding, "sequence": sequence, "nonce": f"nonce{sequence}",
                  "query_index": q, "role": "partial" if q is None else "complete"}
        ready = {"schema": "continuous-query-ready-v1", **common, "members": [brief],
                 "total_bytes": member["bytes"]}
        ready_raw = Q.encoded(ready)
        ready_ref = put(prefix + "READY-v1.json", ready_raw)
        ack = {"schema": "continuous-query-ack-v1", **common, "status": "VERIFIED",
               "ready_sha256": ready_ref["sha256"], "manifest_key": key + "READY-v1.json",
               "manifest_verification": {"method": "fresh_GET", "bytes": len(ready_raw),
                                          "sha256": ready_ref["sha256"]},
               "members": [{**brief, "object_key": key + filename,
                            "verification": {"method": "fresh_GET", "bytes": member["bytes"],
                                             "sha256": member["sha256"]}}]}
        ack_ref = put(prefix + "ACK-v1.json", Q.encoded(ack))
        groups.append({"query_index": q, "prefix_uri": prefix, "ready": ready_ref, "ack": ack_ref,
                       "members": [{**brief, "uri": member["uri"], "event": "PRODUCER_ARRAYS",
                                    "required_original_name": tree["metadata"]["original_name"],
                                    "required_role": "synthetic_input"}]})
    return {"source_parent_binding": binding, "source_parent_task": {"env_seed": 0},
            "input_objects": refs, "source_parent_archives": groups}


def native_fixture(stage, source):
    refs, objects, child_files = [], [], []
    def put(name, raw, *, child=True):
        (stage / name).write_bytes(raw)
        ref = {"path": name, "logical_path": "ledger/" + name,
               "uri": "synthetic://producer/science/ledger/" + name,
               "bytes": len(raw), "sha256": Q.C.sha(raw)}
        refs.append(ref)
        objects.append({**ref, "state": "VERIFIED", "receipt": {"uri": ref["uri"]}})
        if child:
            child_files.append({"path": ref["logical_path"], "bytes": ref["bytes"],
                                "sha256": ref["sha256"]})
        return ref
    def doc(name, value, **kw):
        return put(name, Q.encoded(value), **kw)
    def typed(name, meta, value):
        leaf = Q.C.from_array(value)
        schema = {"metadata": meta, "arrays": {"adapter7": {"__array__": "x", **leaf.identity()}}}
        stream = io.BytesIO()
        np.savez_compressed(stream, __schema__=np.frombuffer(Q.encoded(schema), np.uint8), x=value)
        return put(name, stream.getvalue())
    plan = json.loads((Q.HERE / "INPUTS.json").read_bytes())["native72_plan_entries"]
    domains = {"saved_input_source_identity": "saved", "historical_query_source_identity": "historical",
               "input_identity": "query"}
    inp = typed("input.npz", {"root_id": Q.ROOT, "query_index": 11, "source_identity": "saved"},
                np.zeros((1, 10, 7), np.float64))
    cached, spec = {}, {"input_reference": inp}
    for i, (arm, role) in enumerate((("N", "baseline_none_1"), ("S", "shared_pulse"), ("R", "random_pulse"))):
        entry = next(r for r in plan if (r["root_id"], r["query_index"], r["role"]) ==
                     (Q.ROOT, 11, role))
        spec[arm + "_entry"] = entry
        meta = {"schema": "rss-pulse-compact-call-v1", "entry": entry, "input_ref": inp,
                "native_validation_errors": [], "provenance": {
                    "source_identity": "saved", "native_source_unchanged": True}}
        compact = typed("call_" + arm + ".npz", meta, np.full((1, 10, 7), i, np.float64))
        checks = doc("check_" + arm + ".json", {
            "schema": "rss-pulse-compact-event-v1", "event": "CALL_CHECKS",
            "payload": {"entry": entry, "passed": True, "output": compact}})
        cached[arm] = {"compact": compact["path"], "checks": checks["path"], "entry": entry}
    model = {"model_bound": True, "adapter_bound": True,
             "normalization_sha256": "norm", "loaded_parameter_tree_sha256": "model"}
    model_ref = doc("model.json", model)
    event = doc("source.json", {"event": "SOURCE_BOUND", "payload": {
        "source_manifest": {"sha256": "source"}, "synthetic": False, "model_binding": model_ref,
        "source_bound": True, "model_bound": True, "adapter_bound": True, "inputs_bound": True}})
    ids = [r["call_id"] for r in plan]
    lists = {k: ids for k in ("charged_calls", "completed_calls", "passed_calls", "returned_native_calls")}
    terminal = doc("terminal.json", {"result": {
        "complete": True, "qualification_complete": True, "failures": [],
        "observability": [{"root_id": k[0], "query_index": k[1]} for k in Q.B.KEYS], **lists},
        "unexecuted_call_ids": [], "stopped_on_storage_failure": False})
    result = doc("result.json", {"run_id": Q.B.RUN, "status": "DEV_PULSE_NATIVE_COMPLETE",
        "synthetic": False, "worker_complete": True, "model_binding": model,
        "source_manifest_sha256": "source", **lists})
    archive = doc("archive.json", {"files": child_files}, child=False)
    runtime = doc("runtime.json", {"science_archive": {
        "complete_for_observed_files": True, "errors": [], "objects": objects[:]},
        "runtime_evidence_archive": {"objects": []}}, child=False)
    old_refs = {k: doc(k + ".json", {"synthetic": True}, child=False)
                for k in ("dependencies", "model_old", "runtime_old")}
    role_refs = dict(zip(("dependencies", "model", "runtime"), old_refs.values()))
    old = doc("old.json", {"all_rows": [{"root_id": Q.ROOT, "query_index": 11,
        "query_fingerprint": "query", "input_source_identity": "historical",
        "parent_source": {"source_identity": "saved", "pair_receipt": "synthetic",
            "query_clock_receipt": "synthetic", "actual_initial_selection": source.selection,
            "original_refs": role_refs}}]}, child=False)
    paths = dict(zip(("TERMINAL.json", "RESULT.json", "RUNTIME-RESULT.json",
                     "ARCHIVE-MANIFEST.json", "MODEL_BINDING", "SOURCE_BOUND_event",
                     "saved_input", "old_source_proof"),
                    (r["path"] for r in (terminal, result, runtime, archive, model_ref, event, inp, old))))
    return {"input_objects": refs, "native_provenance_paths": paths,
            "native72_plan_entries": plan, "cached_command_paths": cached,
            "native_snapshot_spec": spec, "source_identity_domains": domains,
            "source_parent_initial_selection": source.selection,
            "original_runtime_proof_paths": {k: v["path"] for k, v in role_refs.items()}}


class Qualification(unittest.TestCase):
    def test_complete60step_source_chain_and_three_anchors(self):
        source = FakeSource()
        out = Q.parent_checks(source, {"source_parent_initial_selection": source.selection}, lambda: None)
        self.assertEqual(len(out["steps"]), 60)
        self.assertEqual([r["query_index"] for r in out["anchors"]], [0, 11, 12])
        self.assertEqual(out["q12_commands_used"], 0)
        self.assertFalse(out["reconstructed_environment_qualified"])

    def test_source_corruption_and_early_terminal_fail_closed(self):
        for failure in ("action", "chain", "clock", "dtype", "terminal", "rng", "protected", "anchor"):
            with self.subTest(failure=failure):
                source = FakeSource()
                source.bad = failure
                with self.assertRaises(ValueError):
                    Q.parent_checks(source, {"source_parent_initial_selection": source.selection}, lambda: None)

    def test_q12_commands_rejected(self):
        source = FakeSource()
        source.records[12] = [SimpleNamespace(event="PARENT_ACTIONS")]
        with self.assertRaisesRegex(ValueError, "q12_witness_only"):
            Q.parent_checks(source, {"source_parent_initial_selection": source.selection}, lambda: None)

    def test_actual_initial_payload_mismatch_rejected(self):
        source = FakeSource()
        source.initial[0].values["initial_vector"][3] = 2
        with self.assertRaisesRegex(ValueError, "initial_vector_payload"):
            Q.parent_checks(source, {"source_parent_initial_selection": source.selection}, lambda: None)

    def test_deadline_called_during_each_source_check(self):
        calls = []
        source = FakeSource()
        def check():
            calls.append(1)
            if len(calls) > 8:
                raise TimeoutError()
        with self.assertRaises(TimeoutError):
            Q.parent_checks(source, {"source_parent_initial_selection": source.selection}, check)
        self.assertEqual(len(calls), 9)

    def test_original_archive_reader_on14_synthetic_groups(self):
        with tempfile.TemporaryDirectory() as temp:
            stage = Path(temp)
            inputs = ledger_fixture(stage)
            source, checks = Q.decode_parent(stage, inputs, lambda: None, synthetic=True)
            self.assertEqual(set(source.records), set(range(13)))
            self.assertEqual(len(source.initial), 1)
            self.assertEqual(len(checks), 14)

    def test_original_role_and_duplicate_query_rejected(self):
        for failure in ("role", "duplicate"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as temp:
                stage = Path(temp)
                inputs = ledger_fixture(stage)
                if failure == "role":
                    inputs["source_parent_archives"][12]["members"][0]["required_role"] = "wrong"
                else:
                    inputs["source_parent_archives"].append(copy.deepcopy(inputs["source_parent_archives"][0]))
                with self.assertRaises(ValueError):
                    Q.decode_parent(stage, inputs, lambda: None, synthetic=True)

    def test_archive_ACK_corruption_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            stage = Path(temp)
            inputs = ledger_fixture(stage)
            ref = inputs["source_parent_archives"][0]["ack"]
            (stage / ref["path"]).write_bytes(b"{}")
            with self.assertRaises(ValueError):
                Q.decode_parent(stage, inputs, lambda: None, synthetic=True)

    def test_real_arrays_require_Linux(self):
        from unittest.mock import patch
        with patch.object(Q.sys, "platform", "darwin"), tempfile.TemporaryDirectory() as temp:
            stage = Path(temp)
            inputs = ledger_fixture(stage)
            with self.assertRaisesRegex(ValueError, "cloud_Linux"):
                Q.decode_parent(stage, inputs, lambda: None, synthetic=False)

    def test_cached_commands_bind_without_assuming_N_is_parent(self):
        with tempfile.TemporaryDirectory() as temp:
            stage = Path(temp)
            source = FakeSource()
            inputs = native_fixture(stage, source)
            result = Q.native_join(stage, inputs, source, lambda: None, synthetic=True)
            self.assertEqual(set(result["cached_commands"]), {"N", "S", "R"})
            self.assertFalse(result["cached_N_equals_original_parent_typed"])
            self.assertFalse(result["new_environment_qualified"])

    def test_cached_identity_domains_and_final_checks_fail_closed(self):
        for failure in ("domain", "role", "failed_checks", "child_manifest"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as temp:
                stage = Path(temp)
                source = FakeSource()
                inputs = native_fixture(stage, source)
                if failure == "domain":
                    inputs["source_identity_domains"]["historical_query_source_identity"] = "saved"
                elif failure == "role":
                    inputs["cached_command_paths"]["R"]["entry"] = inputs["cached_command_paths"]["S"]["entry"]
                elif failure == "failed_checks":
                    path = stage / inputs["cached_command_paths"]["R"]["checks"]
                    doc = json.loads(path.read_bytes())
                    doc["payload"]["passed"] = False
                    path.write_bytes(Q.encoded(doc))
                else:
                    path = stage / inputs["native_provenance_paths"]["ARCHIVE-MANIFEST.json"]
                    doc = json.loads(path.read_bytes())
                    doc["files"] = [r for r in doc["files"] if r["path"] != "ledger/call_S.npz"]
                    path.write_bytes(Q.encoded(doc))
                with self.assertRaises(ValueError):
                    Q.native_join(stage, inputs, source, lambda: None, synthetic=True)


if __name__ == "__main__":
    unittest.main()
