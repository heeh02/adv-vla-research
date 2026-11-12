"""Explicit synthetic receipt/object tests; never load model/runtime packages."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import runtime_guard as G


def array_id(value):
    a = np.asarray(value)
    return {"shape": list(a.shape), "dtype": a.dtype.str,
            "sha256": hashlib.sha256(a.tobytes()).hexdigest()}


def tree_id(value):
    def convert(v):
        if isinstance(v, dict):
            return {k: convert(x) for k, x in v.items()}
        return array_id(v)
    return G.identity(convert(value))


def edit_id(value):
    a = np.asarray(value)
    header = {"shape": list(a.shape), "logical_dtype": a.dtype.name,
              "stored_dtype": a.dtype.str, "bytes": a.nbytes}
    raw = json.dumps(header, sort_keys=True, separators=(",", ":")).encode()+b"\0"+a.tobytes()
    return {**header, "sha256": hashlib.sha256(raw).hexdigest()}


class FakeNativeRecord:
    def __init__(self, mask, positions, ref):
        self.ref = ref
        self.leaves = {}
        for key, value in (("prefix_mask", mask), ("prefix_positions", positions)):
            array = value.copy()
            self.leaves[key] = SimpleNamespace(
                storage=array, descriptor={"payload_sha256": hashlib.sha256(array.tobytes()).hexdigest()})

    def field(self, name):
        return self.leaves[name]


FAKE_ORIGINAL = '''def bind(campaign):
    model = campaign.policy._model
    objects = {"p": model.value}
    summary = campaign.summary
    def guard(actual_campaign):
        if actual_campaign is not campaign or actual_campaign.policy._model is not model:
            raise ValueError("synthetic_campaign_changed")
        if model.value is not objects["p"]:
            raise ValueError("synthetic_parameter_object_changed")
        campaign.guard_checks += 1
        return summary
    return guard
'''


def fixture(folder):
    folder = Path(folder)
    package, candidate, evidence = folder/"package", folder/"package/candidate", folder/"evidence"
    candidate.mkdir(parents=True)
    evidence.mkdir()
    (package/"metadata").mkdir()
    source_rows = []

    def write(path, data, source=False, raw=False):
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = data.encode() if raw else G.encoded(data)
        path.write_bytes(payload)
        ref = {"path": str(path.relative_to(package)) if source else str(path),
               "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
        if source:
            source_rows.append(ref)
        return ref

    model = SimpleNamespace(value=object())
    stats = {"mean": np.zeros(4), "std": np.ones(4)}
    transforms = lambda: SimpleNamespace(transforms=[
        SimpleNamespace(norm_stats=copy.deepcopy(stats), use_quantiles=True)])
    campaign = SimpleNamespace(
        policy=SimpleNamespace(_model=model, _input_transform=transforms(),
                               _output_transform=transforms()), guard_checks=0)
    model_summary = {
        "schema": "state-response-actual-model-binding-v1",
        "model_bound": True, "adapter_bound": True, "new_model_forward_calls": 0,
        "checkpoint_artifact_id": "synthetic-model", "normalization_sha256": "n"*64,
        "loaded_parameter_tree_sha256": "p"*64, "parameter_dtypes": ["bfloat16"],
        "effective_model_config": {"synthetic": True}}
    campaign.summary = model_summary
    guard_ref = write(package/"original_guard.py", FAKE_ORIGINAL, source=True, raw=True)
    spec = importlib.util.spec_from_file_location("_synthetic_original_loaded_guard", package/"original_guard.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model_guard = module.bind(campaign)
    model_rows = [{
        "id": "m0", "uri": "redacted-1828c13b4c3f", "local_relative": "synthetic-model",
        "original_artifact_id": "synthetic-model"}]
    model_plan_ref = write(package/"metadata/MODEL_READ_PLAN-v1.json",
                           {"read_members": model_rows}, source=True)
    saved = []
    for i in range(8):
        vt = np.ones(2, np.float32)
        s, r = vt.copy(), vt.copy()
        s[0] += .25
        r[1] += .5
        vm = {"S": s, "P": s, "R": r, "own_null": vt}
        effects = {key: a-vt for key, a in vm.items()}
        dose = {arm: {"Vmod": edit_id(vm[role]), "actual_E": edit_id(effects[role])}
                for arm, role in (("shared", "S"), ("random_matched", "R"))}
        mask, positions = np.ones(2, bool), np.arange(2, dtype=np.int32)
        old_ref = {"path": "synthetic-natural-T.npz", "bytes": 3, "sha256": "f"*64}
        saved.append(SimpleNamespace(
            key=("synthetic-root-"+str(i//2), (11, 22)[i % 2]),
            root_id="synthetic-root-"+str(i//2), query_index=(11, 22)[i % 2],
            source_identity=f"{i:064x}", synthetic=True, noise=np.zeros(3, np.float32),
            decode_state=np.zeros(2, np.float32), old_KT=np.zeros(2, np.float32),
            old_VT=vt, prefix_mask=mask, prefix_positions=positions,
            modified_v=vm, actual_E=effects,
            encoded_identities={"C": "c"*64, "T": "t"*64},
            raw_observations={"C": np.zeros(3), "T": np.ones(3)},
            sources={"synthetic": True, "native": {"natural_T": old_ref}}, dose=dose,
            old_calls={"natural_T": FakeNativeRecord(mask, positions, old_ref)},
            model_binding=model_summary))
    input_objects = [{"uri": "redacted-4367738f38e4", "bytes": 3,
                      "sha256": "a"*64, "codec": "synthetic", "path": "input-0000.npz"}]
    manifest = {"input_objects": input_objects, "input_object_count": 1, "input_bytes": 3,
                "snapshots": [{"root_id": x.root_id, "query_index": x.query_index} for x in saved]}
    input_manifest_ref = write(candidate/"INPUTS.json", manifest, source=True)
    proof = {
        "schema": "rss-dev-pulse-input-qualification-v1", "analysis_complete": True,
        "input_source_qualified": True, "synthetic": True, "native_runtime_qualified": False,
        "GPU": 0, "model_calls": 0, "environment_steps": 0, "new_pulse_labels": 0,
        "input_objects": [{k: x[k] for k in ("uri", "bytes", "sha256", "codec")} for x in input_objects],
        "input_bytes": 3, "snapshots": [{
            "root_id": x.root_id, "query_index": x.query_index, "source_identity": x.source_identity,
            "encoded_identities": x.encoded_identities, "source_refs": x.sources,
            "actual_dose": x.dose, "noise": array_id(x.noise),
            "decode_state": array_id(x.decode_state), "prefix_KT": array_id(x.old_KT),
            "prefix_VT": array_id(x.old_VT), "raw_C_T_identity": tree_id(x.raw_observations)}
            for x in saved]}
    proof_ref = write(evidence/"CPU-INPUT-PROOF.json", proof)
    plan = {"planned_queries": [list(x.key) for x in saved], "synthetic": True}
    plan_ref = write(candidate/"PLAN.json", plan, source=True)
    restore = {"frozen_plan": {k: model_plan_ref[k] for k in ("bytes", "sha256")},
               "whole_SHA_verified_objects": model_rows}
    restore_ref = write(evidence/"RESTORE.json", restore)
    base = {"archive": "synthetic-base", "members": 1, "parts": 1,
            "logical_file_bytes": 3, "all_actual_TAR_headers_and_member_SHA_verified": True,
            "restored_assets_zip": False, "model_payloads": False}
    overlay = {**base, "archive": "synthetic-overlay"}
    environment_plan = {"bytes": 3, "sha256": "e"*64}
    base_index = {"path": "synthetic-index.json", "bytes": 3, "sha256": "b"*64}
    runtime_restore = {
        "schema": "recorded-replay-runtime-restore-v1", "model_payloads": 0,
        "dependency_installations": 0, "restore_helper": "unchanged_composite_restore.restore_pair",
        "base": base, "overlay": overlay, "expanded_runtime_bytes": 6,
        "dependencies": {
            "plan": environment_plan, "qualified_actual_import": True,
            "restored_private_environment_rebased": True, "dependency_installations": 0,
            "helper": {"bytes": 3, "sha256": "h"*64}},
        "runtime_manifest_reference": {
            "schema": "cloud-local-original-runtime-manifest-reference-v1", "base_index": base_index,
            "members": 1, "files": [], "new_remote_GETs": 0, "new_manifest_copies": 0}}
    runtime_ref = write(evidence/"RUNTIME-RESTORE.json", runtime_restore)
    model_ref = write(evidence/"MODEL-BINDING.json",
                      {**model_summary, "leaves": [], "original_files": [], "sources": {}})
    source_manifest = {"files": source_rows}
    source_ref = write(package/"metadata/SOURCE_MANIFEST-v1.json", source_manifest)
    source_ref["path"] = "metadata/SOURCE_MANIFEST-v1.json"
    closure = {
        "schema": "DEV-pulse-loaded-context-closure-v1",
        "required_original_package_files": [guard_ref, model_plan_ref],
        "input_manifest": input_manifest_ref,
        "original_model_guard": guard_ref, "model_plan": model_plan_ref,
        "model_objects": model_rows, "model_object_count": 1,
        "runtime_restore_schema": "recorded-replay-runtime-restore-v1",
        "expected_runtime_restore": {
            "base": base, "overlay": overlay, "environment_plan": environment_plan,
            "expanded_runtime_bytes": 6, "base_index": base_index, "base_file_manifests": []},
        "expected_model": {k: model_summary[k] for k in
                           ("checkpoint_artifact_id", "normalization_sha256",
                            "loaded_parameter_tree_sha256", "parameter_dtypes")}}
    args = {
        "closure": closure, "input_adapter": SimpleNamespace(
            array_id=array_id, identity=tree_id, value=lambda leaf: leaf.storage,
            pinned=lambda name: SimpleNamespace(aid=edit_id)),
        "package_root": package, "candidate_root": candidate,
        "source_manifest_ref": source_ref, "input_manifest_ref": input_manifest_ref,
        "input_qualification_ref": proof_ref, "restore_receipt_ref": restore_ref,
        "runtime_restore_ref": runtime_ref, "model_binding_ref": model_ref,
        "model_guard": model_guard,
        "request": {"model_plan": model_plan_ref, "runtime_config": plan_ref}}
    class SyntheticRuntime:
        synthetic = True
        def __init__(self):
            self.campaign = campaign
            compiled = lambda: None
            self.common_compiled = compiled
            self.rt = {"compiled": compiled, "common": lambda: compiled(),
                       "decode": lambda: campaign, "wait": lambda x: x}
        def encode(self, value): return value
        def compute(self, value): return value
        def decode(self, value): return value
        def wait(self, value): return value
    runtime = SyntheticRuntime()
    return campaign, saved, plan, runtime, args


def bound(folder):
    campaign, saved, plan, runtime, args = fixture(folder)
    guard = G.LoadedContextGuard.for_synthetic_tests(campaign, saved, plan, **args)
    guard.bind_runtime(runtime)
    return guard, campaign, saved, plan, runtime, args


class Tests(unittest.TestCase):
    def test_synthetic_binding_never_becomes_production_qualification(self):
        before = {k for k in sys.modules if k.startswith(("jax", "flax", "openpi"))}
        with tempfile.TemporaryDirectory() as d:
            guard, campaign, saved, plan, runtime, args = bound(d)
            proof = guard(campaign, saved, plan, runtime)
            self.assertTrue(proof["synthetic"])
            self.assertFalse(guard.production_bound)
            self.assertFalse(proof["GPU_authorization_granted_by_guard"])
            self.assertEqual(proof["per_call_weight_copy_bytes"], 0)
            self.assertEqual(proof["new_model_forward_calls"], 0)
            self.assertEqual(proof["normalizer_bytes_per_check"], 128)
            self.assertEqual(campaign.guard_checks, 2)
        self.assertEqual(before, {k for k in sys.modules if k.startswith(("jax", "flax", "openpi"))})

    def test_percall_guard_uses_stat_and_objects_without_file_rereads(self):
        with tempfile.TemporaryDirectory() as d:
            guard, campaign, saved, plan, runtime, args = bound(d)
            with patch.object(Path, "read_bytes", side_effect=AssertionError("no_file_rehash_per_call")):
                first = guard(campaign, saved, plan, runtime)
                second = guard(campaign, saved, plan, runtime)
            self.assertEqual(first["source_hash_bytes_at_bind"], second["source_hash_bytes_at_bind"])
            self.assertEqual(second["guard_calls"], 2)
            self.assertEqual(second["per_call_weight_copy_bytes"], 0)

    def test_arbitrary_alltrue_callback_cannot_replace_original_guard_closure(self):
        with tempfile.TemporaryDirectory() as d:
            campaign, saved, plan, runtime, args = fixture(d)
            args["model_guard"] = lambda _: campaign.summary
            with self.assertRaisesRegex(ValueError, "actual_callable_source_origin"):
                G.LoadedContextGuard.for_synthetic_tests(campaign, saved, plan, **args)

    def test_source_or_receipt_mutation_invalidates_bound_context(self):
        for target in ("original_guard.py", "evidence"):
            with self.subTest(target=target), tempfile.TemporaryDirectory() as d:
                guard, campaign, saved, plan, runtime, args = bound(d)
                path = (Path(args["input_qualification_ref"]["path"]) if target == "evidence"
                        else Path(args["package_root"])/target)
                path.write_bytes(path.read_bytes()+b" ")
                with self.assertRaisesRegex(ValueError, "immutable_bound_source_and_evidence"):
                    guard(campaign, saved, plan, runtime)

    def test_compiled_callable_and_snapshot_replacement_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            guard, campaign, saved, plan, runtime, args = bound(d)
            runtime.common_compiled = lambda: None
            with self.assertRaisesRegex(ValueError, "one_bound_campaign"):
                guard(campaign, saved, plan, runtime)
        with tempfile.TemporaryDirectory() as d:
            guard, campaign, saved, plan, runtime, args = bound(d)
            saved[0] = copy.copy(saved[0])
            with self.assertRaisesRegex(ValueError, "bound_saved_input_objects"):
                guard(campaign, saved, plan, runtime)

    def test_loaded_parameter_replacement_hits_original_lightweight_guard(self):
        with tempfile.TemporaryDirectory() as d:
            guard, campaign, saved, plan, runtime, args = bound(d)
            campaign.policy._model.value = object()
            with self.assertRaisesRegex(ValueError, "synthetic_parameter_object_changed"):
                guard(campaign, saved, plan, runtime)

    def test_normalizer_inplace_mutation_and_transform_replacement_rejected(self):
        for replace in (False, True):
            with self.subTest(replace=replace), tempfile.TemporaryDirectory() as d:
                guard, campaign, saved, plan, runtime, args = bound(d)
                if replace:
                    campaign.policy._input_transform = copy.deepcopy(campaign.policy._input_transform)
                else:
                    campaign.policy._input_transform.transforms[0].norm_stats["mean"][0] = 7
                with self.assertRaisesRegex(ValueError, "actual_adapter_normalizer_mutation"):
                    guard(campaign, saved, plan, runtime)

    def test_plan_mutation_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            guard, campaign, saved, plan, runtime, args = bound(d)
            plan["unplanned_extension"] = True
            with self.assertRaisesRegex(ValueError, "bound_plan_or_request_mutation"):
                guard(campaign, saved, plan, runtime)

    def test_missing_restore_objects_rejected_even_with_true_model_summary(self):
        with tempfile.TemporaryDirectory() as d:
            campaign, saved, plan, runtime, args = fixture(d)
            ref = args["restore_receipt_ref"]
            doc = json.loads(Path(ref["path"]).read_text())
            doc["whole_SHA_verified_objects"] = []
            raw = G.encoded(doc)
            Path(ref["path"]).write_bytes(raw)
            args["restore_receipt_ref"] = {
                "path": ref["path"], "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
            with self.assertRaisesRegex(ValueError, "all_original_model_objects_restored"):
                G.LoadedContextGuard.for_synthetic_tests(campaign, saved, plan, **args)

    def test_runtime_restore_boolean_shell_is_not_sufficient(self):
        with tempfile.TemporaryDirectory() as d:
            campaign, saved, plan, runtime, args = fixture(d)
            ref = args["runtime_restore_ref"]
            doc = json.loads(Path(ref["path"]).read_text())
            doc["base"] = {}
            raw = G.encoded(doc)
            Path(ref["path"]).write_bytes(raw)
            args["runtime_restore_ref"] = {
                "path": ref["path"], "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
            with self.assertRaisesRegex(ValueError, "original_complete_capsule_restore_proof"):
                G.LoadedContextGuard.for_synthetic_tests(campaign, saved, plan, **args)

    def test_runtime_must_be_bound_once_before_any_use(self):
        with tempfile.TemporaryDirectory() as d:
            campaign, saved, plan, runtime, args = fixture(d)
            guard = G.LoadedContextGuard.for_synthetic_tests(campaign, saved, plan, **args)
            with self.assertRaisesRegex(ValueError, "one_bound_campaign"):
                guard(campaign, saved, plan, runtime)
            guard.bind_runtime(runtime)
            with self.assertRaisesRegex(ValueError, "runtime_bound_once"):
                guard.bind_runtime(runtime)

    def test_production_constructor_refuses_nonlinux_without_importing_models(self):
        with patch.object(G.sys, "platform", "darwin"):
            with self.assertRaisesRegex(ValueError, "native_context_guard_Linux_only"):
                G.LoadedContextGuard(
                    None, None, None, package_root=None, source_manifest_ref=None,
                    input_manifest_ref=None, input_qualification_ref=None, restore_receipt_ref=None,
                    runtime_restore_ref=None, model_binding_ref=None, model_guard=None, request=None)

    def test_coherent_payload_E_tamper_cannot_borrow_original_dose_proof(self):
        with tempfile.TemporaryDirectory() as d:
            campaign, saved, plan, runtime, args = fixture(d)
            saved[0].modified_v["S"][0] = 123
            saved[0].actual_E["S"][0] = 122
            with self.assertRaisesRegex(ValueError, "actual_frozen_Vmod_E_match_CPU_proof"):
                G.LoadedContextGuard.for_synthetic_tests(campaign, saved, plan, **args)

    def test_prefix_addresses_match_original_typed_natural_record(self):
        with tempfile.TemporaryDirectory() as d:
            campaign, saved, plan, runtime, args = fixture(d)
            saved[0].prefix_mask[0] = False
            with self.assertRaisesRegex(ValueError, "actual_prefix_layout_matches"):
                G.LoadedContextGuard.for_synthetic_tests(campaign, saved, plan, **args)

    def test_dispatch_common_replacement_fails_even_if_compiled_attribute_stays_same(self):
        with tempfile.TemporaryDirectory() as d:
            guard, campaign, saved, plan, runtime, args = bound(d)
            original_compiled = runtime.common_compiled
            runtime.rt["common"] = lambda: "different inference path"
            self.assertIs(runtime.common_compiled, original_compiled)
            with self.assertRaisesRegex(ValueError, "native_runtime_dispatch_replaced"):
                guard(campaign, saved, plan, runtime)

    def test_zero_valued_V_tensor_is_not_the_active_zero_E_own_null(self):
        with tempfile.TemporaryDirectory() as d:
            campaign, saved, plan, runtime, args = fixture(d)
            saved[0].modified_v["own_null"] = np.zeros_like(saved[0].old_VT)
            with self.assertRaisesRegex(ValueError, "active_own_null_is_original_VT"):
                G.LoadedContextGuard.for_synthetic_tests(campaign, saved, plan, **args)


if __name__ == "__main__":
    unittest.main()
