"""CPU-only glue/identity/failure tests; no model, GPU, cloud, or environment.

The connected-pipeline tests use the existing candidate's synthetic worker
fixtures. Only unittest mocks connect fake input/model restoration; production
execute() has no test-component or runtime injection parameter.
"""
import contextlib
import copy
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import science_entry as E

HERE = Path(__file__).resolve().parent
CANDIDATE = HERE.parent / "rss_dev_pulse_candidate_v2"
with patch.object(sys, "path", [str(CANDIDATE), *sys.path]):
    F = E.module("_rss_execution_synthetic_worker_fixtures", CANDIDATE / "test_worker.py")
W = F.W


class DiskTestLedger(F.MemoryLedger):
    """Tiny synthetic ledger; actual max-shape serialization is tested upstream."""
    def __init__(self, out, plan, deadline):
        super().__init__(plan)
        self.root = Path(out)
        self.root.mkdir(parents=True, exist_ok=False)
        self.deadline = deadline

    def finish(self, result):
        super().finish(result)
        return E.write_json(self.root / "TERMINAL.json",
                            {"result": result, "synthetic_test_sink": True})


class GlueTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=HERE)
        self.root = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)
        self.plan = json.loads((CANDIDATE / "PLAN-CANDIDATE.json").read_bytes())
        self.events = []
        self.context = {"run_id": "synthetic-glue", "output_dir": str(self.root / "out"),
            "input_stage_root": str(self.root / "inputs"),
            "science_deadline_epoch": E.time.time() + 60,
            "model_restore_receipt_path": str(self.root / "evidence/RESTORE-v1.json"),
            "checkpoint_root": str(self.root / "inputs/assets/checkpoint"),
            "request": {"synthetic": True}, "authority": None}
        Path(self.context["output_dir"]).mkdir()
        (self.root / "evidence").mkdir()
        self.binding = {"model_bound": True, "adapter_bound": True,
                        "loaded_parameter_tree_sha256": "a" * 64, "normalization_sha256": "b" * 64}
        E.write_json(self.root / "evidence/MODEL-BINDING-v1.json", self.binding)
        self.bound = {"package": HERE.parent, "candidate": CANDIDATE,
            "source_manifest_ref": {"path": "synthetic_manifest"},
            "closure": {"input_manifest": E.identity(CANDIDATE / "input/INPUTS.json"),
                        "original_model_guard": {"path": "unused-synthetic-guard.py"}}}
        self.inputs = F.inputs()
        self.proof = {"snapshots": [{"synthetic": True} for _ in self.inputs],
                      "input_bytes": 0, "input_objects": []}
        self.result = {"synthetic": True, "model_loads_started": 0, "model_loads_completed": 0}
        self.model = object()
        self.original_closure = lambda campaign: self.binding

    def connect(self, *, fault=None, preload_error=None, guard_error=None):
        owner = self

        class Guard:
            synthetic = False  # Pipeline only. It is NEVER a worker native guard.

            def __init__(self, campaign, inputs, plan, **kwargs):
                owner.events.append("loaded_guard")
                owner.assertIs(campaign, owner.model)
                owner.assertIs(inputs, owner.inputs)
                owner.assertIs(plan, owner.plan)
                owner.assertIs(kwargs["model_guard"], owner.original_closure)
                owner.assertEqual(kwargs["candidate_root"], CANDIDATE)
                if guard_error:
                    raise E.EntryFault(guard_error)

        def preload(*args):
            self.events.append("preload")
            if preload_error:
                raise E.EntryFault(preload_error)
            return {"cpu_proof": self.proof,
                    **{k: {} for k in ("input_qualification_ref", "restore_receipt_ref", "runtime_restore_ref")}}

        def load_inputs(*args, **kwargs):
            self.events.append("input_load")
            self.assertNotIn("synthetic", kwargs)
            return {"snapshots": self.inputs, "proof": copy.deepcopy(self.proof)}

        def load_model(context):
            self.events.append("model_load")
            return self.model

        def bind(campaign, **kwargs):
            self.events.append("original_bind")
            self.assertIs(campaign, self.model)
            self.assertEqual(kwargs["request"], self.context["request"])
            return self.original_closure

        def run(campaign, inputs, plan, **kwargs):
            self.events.append("worker_run")
            self.assertNotIn("runtime", kwargs)  # Native caller cannot select a fake.
            self.assertIs(type(kwargs["source_guard"]), Guard)
            self.assertIsNone(kwargs["authority"])
            ledger = kwargs["ledger"]
            runtime = F.FakeRuntime(ledger, fault=fault)
            # Only this test wrapper supplies the known synthetic runtime. The
            # real worker then executes its actual 72-order/qualification loop.
            return W.run(campaign, inputs, plan, ledger=ledger, runtime=runtime,
                         deadline_epoch=kwargs["deadline_epoch"],
                         source_guard=lambda *a: {"source_bound": True, "model_bound": True,
                             "adapter_bound": True, "inputs_bound": True, "synthetic": True})

        worker = SimpleNamespace(LoadedContextGuard=Guard, run=run,
            input_module=lambda: SimpleNamespace(load=load_inputs))
        with patch.object(E, "validate_preload_evidence", preload), \
             patch.object(E, "load_campaign", load_model), \
             patch.object(E, "module", lambda *a: SimpleNamespace(bind=bind)):
            return E.connected_pipeline(self.context, self.bound, self.plan, worker,
                SimpleNamespace(CompactLedger=DiskTestLedger), self.result, lambda: None)

    def test_all72_calls_use_actual_worker_order_and_single_loader(self):
        result = self.connect()
        self.assertEqual(result["status"], "SYNTHETIC_WORKER_COMPLETE")
        self.assertTrue(result["synthetic"])
        self.assertEqual(self.events, ["preload", "input_load", "model_load",
                                      "original_bind", "loaded_guard", "worker_run"])
        self.assertEqual(self.result["model_loads_started"], 1)
        self.assertEqual(self.result["model_loads_completed"], 1)
        self.assertEqual(result["charged_calls"], [r["call_id"] for r in self.plan["entries"]])
        self.assertEqual(result["full_generations_charged"], 72)
        self.assertEqual(result["prefix_prefills_charged"], 144)
        self.assertEqual(result["suffix_evaluations_charged"], 720)
        self.assertEqual(self.result["returned_native_calls"], result["returned_native_calls"])
        payload, _ = E.read_json(Path(self.context["output_dir"]) / "ledger/TERMINAL.json")
        self.assertEqual(len(payload["result"]["passed_calls"]), 72)

    def test_failed_preload_never_calls_input_or_model(self):
        with self.assertRaisesRegex(E.EntryFault, "bad_restore"):
            self.connect(preload_error="bad_restore")
        self.assertEqual(self.events, ["preload"])
        self.assertEqual(self.result["model_loads_started"], 0)

    def test_concrete_binding_failure_never_calls_worker(self):
        with self.assertRaisesRegex(E.EntryFault, "bad_binding"):
            self.connect(guard_error="bad_binding")
        self.assertEqual(self.events[-1], "loaded_guard")
        self.assertNotIn("worker_run", self.events)

    def test_native_failure_is_preserved_not_requeued(self):
        # Existing fixture deliberately mutates a captured PG tensor on call 1.
        result = self.connect(fault="pg")
        self.assertFalse(result["complete"])
        self.assertEqual(result["status"], "FAILED_PRESERVED")
        self.assertGreater(len(result["charged_calls"]), 0)
        self.assertLess(len(result["charged_calls"]), 72)
        self.assertTrue(result["failures"])
        payload, _ = E.read_json(Path(self.context["output_dir"]) / "ledger/TERMINAL.json")
        self.assertEqual(payload["result"]["charged_calls"], result["charged_calls"])


class AdmissionAndPreservationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=HERE)
        self.root = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)

    def context(self):
        return {"run_id": "test-rejected", "output_dir": str(self.root / "out"),
                "science_deadline_epoch": E.time.time() + 60,
                "package_root": str(self.root / "package"),
                "candidate_root": str(self.root / "package/candidate")}

    def test_default_is_inert_even_with_nonexistent_context(self):
        with patch.object(E, "execute", side_effect=AssertionError("must_not_execute")):
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                self.assertEqual(E.main(["--context", "/does/not/exist"]), 0)
            value = json.loads(output.getvalue())
            self.assertEqual(value["status"], "DRY_ONLY_GPU0")
            self.assertEqual(value["model_loads"], 0)

    def test_rejected_context_has_small_terminal_and_archive(self):
        context = self.context()
        result = E.execute(context, "a" * 64)
        self.assertEqual(result["status"], "FAILED_PRESERVED")
        self.assertEqual(result["model_loads_started"], 0)
        self.assertEqual(result["charged_status"], "worker_not_started")
        manifest, _ = E.read_json(Path(context["output_dir"]) / "ARCHIVE-MANIFEST.json")
        self.assertEqual([x["path"] for x in manifest["files"]], ["RESULT.json"])

    def test_source_failure_cannot_import_policy(self):
        context = self.context()
        with patch.object(E, "validate_context"), \
             patch.object(E.sys, "platform", "linux"), \
             patch.object(E.sys, "version_info", (3, 12)), \
             patch.object(E, "verify_source", side_effect=E.EntryFault("bad_source_SHA")), \
             patch.object(E, "module", side_effect=AssertionError("import_after_bad_SHA")):
            result = E.execute(context, "a" * 64)
        self.assertEqual(result["error_code"], "bad_source_SHA")
        self.assertEqual(result["model_loads_started"], 0)

    def test_missing_real_authority_stops_before_worker_import(self):
        context = self.context()
        with patch.object(E, "validate_context"), \
             patch.object(E.sys, "platform", "linux"), \
             patch.object(E.sys, "version_info", (3, 12)), \
             patch.object(E, "verify_source", return_value={"candidate": CANDIDATE}):
            calls = []
            original = E.module

            def record(name, path):
                calls.append(Path(path).name)
                self.assertEqual(Path(path).name, "contract.py")
                return original(name, path)

            with patch.object(E, "module", record):
                result = E.execute(context, "a" * 64)
        self.assertEqual(result["error_code"], "GPU0_requires_new_explicit_user_authorization")
        self.assertEqual(result["model_loads_started"], 0)
        self.assertEqual(calls, ["contract.py"])

    def test_deadlines_and_credential_environment_fail_closed(self):
        now = E.time.time()
        context = {"schema": E.SCHEMA, "run_id": "fixture",
            "science_deadline_epoch": now + 10, "archive_deadline_epoch": now + 20,
            "release_deadline_epoch": now + 30}
        for key in ("package_root", "candidate_root", "repo_root", "checkpoint_root",
                    "input_stage_root", "input_qualification_path", "private_root",
                    "model_restore_receipt_path", "runtime_restore_receipt_path", "output_dir"):
            context[key] = str(self.root / key)
        context["repo_root"] = str(Path(context["package_root"]) / "repo")
        with patch.dict(E.os.environ, {}, clear=True):
            E.validate_context(context)
        with patch.dict(E.os.environ, {"TEST_API_KEY": "synthetic-never-secret"}, clear=True):
            with self.assertRaisesRegex(E.EntryFault, "credential_free"):
                E.validate_context(context)
        context["science_deadline_epoch"] = now - 1
        with patch.dict(E.os.environ, {}, clear=True):
            with self.assertRaisesRegex(E.EntryFault, "ordered_future"):
                E.validate_context(context)

    def partial_run(self, malformed_terminal=False):
        context = self.context()
        plan = json.loads((CANDIDATE / "PLAN-CANDIDATE.json").read_bytes())
        with patch.object(E, "validate_context"), \
             patch.object(E.sys, "platform", "linux"), \
             patch.object(E.sys, "version_info", (3, 12)), \
             patch.object(E, "verify_source", return_value={"candidate": CANDIDATE}), \
             patch.object(E, "module", return_value=SimpleNamespace(require_native_authority=lambda *a: None)):
            def interrupted(context, bound, plan, worker, storage, result, check):
                result["worker_started"] = True
                ledger = Path(context["output_dir"]) / "ledger"
                ledger.mkdir()
                E.write_json(ledger / "event-0001.json", {"event": "CHARGED", "entry": plan["entries"][0]})
                if malformed_terminal:
                    (ledger / "TERMINAL.json").write_bytes(b"{")
                raise RuntimeError("synthetic_finalization_interruption")
            with patch.object(E, "connected_pipeline", interrupted):
                result = E.execute(context, "a" * 64)
        return context, result

    def test_interrupted_finalization_keeps_unknown_and_partial_archive(self):
        context, result = self.partial_run()
        self.assertEqual(result["charged_status"], "unknown_consult_durable_ledger")
        self.assertFalse(result["worker_complete"])
        manifest, _ = E.read_json(Path(context["output_dir"]) / "ARCHIVE-MANIFEST.json")
        self.assertIn("ledger/event-0001.json", {r["path"] for r in manifest["files"]})

    def test_malformed_terminal_kept_without_hiding_launcher_result(self):
        context, result = self.partial_run(malformed_terminal=True)
        self.assertEqual(result["charged_status"], "unknown_consult_durable_ledger")
        self.assertEqual(result["terminal_read_error"], "JSONDecodeError")
        manifest, _ = E.read_json(Path(context["output_dir"]) / "ARCHIVE-MANIFEST.json")
        self.assertIn("ledger/TERMINAL.json", {r["path"] for r in manifest["files"]})

    def test_archive_rejects_symlink(self):
        out = self.root / "out"
        out.mkdir()
        (self.root / "outside").write_text("fixture")
        (out / "link").symlink_to(self.root / "outside")
        with self.assertRaisesRegex(E.EntryFault, "no_output_symlinks"):
            E.archive_descriptors(out)

    def test_source_manifest_exact_members_zero_byte_and_mutation(self):
        package = self.root / "package"
        (package / "candidate").mkdir(parents=True)
        (package / "execution").mkdir()
        (package / "metadata").mkdir()
        (package / "repo").mkdir()
        (package / "repo/__init__.py").write_bytes(b"")
        (package / "execution/science_entry.py").write_bytes(HERE.joinpath("science_entry.py").read_bytes())
        old = {"files": [{"path": "repo/__init__.py", **E.identity(package / "repo/__init__.py")}]}
        E.write_json(package / "metadata/SOURCE_MANIFEST-original-v1.json", old)
        E.write_json(package / "candidate/NATIVE-CLOSURE.json", {
            "source_provenance": {"PVP_source_manifest": E.identity(package / "metadata/SOURCE_MANIFEST-original-v1.json")},
            "required_original_package_files": old["files"]})
        E.write_json(package / "candidate/SOURCE-MANIFEST.json", {"files": [{
            "path": "NATIVE-CLOSURE.json", **E.identity(package / "candidate/NATIVE-CLOSURE.json")}]})
        rows = [{"path": str(p.relative_to(package)), **E.identity(p)}
                for p in sorted(package.rglob("*")) if p.is_file()]
        E.write_json(package / "metadata/SOURCE_MANIFEST-v1.json", {"files": rows})
        expected = E.identity(package / "metadata/SOURCE_MANIFEST-v1.json")["sha256"]
        with patch.object(E, "HERE", package / "execution"):
            bound = E.verify_source(package, package / "candidate", expected)
            self.assertEqual(bound["sources"]["repo/__init__.py"]["bytes"], 0)
            (package / "repo/__init__.py").write_bytes(b"changed")
            with self.assertRaisesRegex(E.EntryFault, "exact_full_source_member"):
                E.verify_source(package, package / "candidate", expected)


if __name__ == "__main__":
    unittest.main()
