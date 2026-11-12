"""Loaded native context binding; no restoration, model construction or forwards.

Production consumes the closure returned by the *original executed*
model_guard.bind(), after its one-time file/parameter hashing. Importing this
module imports no JAX, Flax, policy, simulator, or model package.
"""
import dataclasses
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path, PurePosixPath
import sys
import types

HERE = Path(__file__).resolve().parent
_TEST_FACTORY = object()


def need(value, code):
    if not value:
        raise ValueError(code)


def encoded(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)+"\n").encode()


def identity(value):
    return hashlib.sha256(encoded(value)).hexdigest()


def token(path):
    st = path.stat()
    return st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns


def inside(root, relative):
    p = PurePosixPath(relative)
    need(isinstance(relative, str) and relative and not p.is_absolute() and
         "\\" not in relative and str(p) == relative and
         all(part not in ("", ".", "..") for part in relative.split("/")), "canonical_package_path")
    path = root/relative
    need(path.resolve().is_relative_to(root) and not path.is_symlink(),
         "source_path_inside_owned_package")
    return path


def nested_code(raw, filename, names):
    code = compile(raw, str(filename), "exec", dont_inherit=True, optimize=sys.flags.optimize)
    for name in names:
        matches = [c for c in code.co_consts if isinstance(c, types.CodeType) and c.co_name == name]
        need(len(matches) == 1, "unique_executed_function_code")
        code = matches[0]
    return code


def same_code(function, raw, filename, names):
    need(isinstance(function, types.FunctionType), "actual_python_function_not_arbitrary_callable")
    need(Path(function.__code__.co_filename).resolve() == filename.resolve(),
         "actual_callable_source_origin")
    need(function.__code__ == nested_code(raw, filename, names), "actual_executed_callable_code")


def small_tree(value, budget):
    """Hash small adapter normalization arrays, never model parameters."""
    if dataclasses.is_dataclass(value):
        return {f.name: small_tree(getattr(value, f.name), budget) for f in dataclasses.fields(value)}
    if isinstance(value, dict):
        return {str(k): small_tree(v, budget) for k, v in sorted(value.items())}
    if isinstance(value, (tuple, list)):
        return [small_tree(v, budget) for v in value]
    if value is None or type(value) in (bool, int, float, str):
        return value
    need(hasattr(value, "dtype") and hasattr(value, "shape") and hasattr(value, "tobytes"),
         "bounded_small_normalizer_array")
    raw = value.tobytes()
    budget[0] += len(raw)
    need(budget[0] <= 1024**2, "normalizer_hash_not_model_weight_copy")
    return {"shape": list(value.shape), "dtype": str(value.dtype),
            "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def normalizers(campaign):
    out, ids, budget = {}, [], [0]
    for side in ("_input_transform", "_output_transform"):
        chain = getattr(campaign.policy, side)
        transforms = chain.transforms
        ids.append((id(chain), id(transforms), tuple(id(v) for v in transforms)))
        stats = []
        for transform in transforms:
            if hasattr(transform, "norm_stats"):
                stats.append({
                    "class": type(transform).__module__+"."+type(transform).__qualname__,
                    "use_quantiles": bool(transform.use_quantiles),
                    "norm_stats": small_tree(transform.norm_stats, budget)})
        need(len(stats) == 1, "one_input_and_one_output_native_normalizer")
        out[side] = stats
    return tuple(ids), identity(out), budget[0]


def snapshot_tag(saved):
    return (
        id(saved), saved.key, saved.source_identity, id(saved.noise), id(saved.decode_state),
        id(saved.old_KT), id(saved.old_VT), id(saved.prefix_mask), id(saved.prefix_positions),
        tuple((key, id(value)) for key, value in sorted(saved.modified_v.items())),
        tuple((key, id(value)) for key, value in sorted(saved.actual_E.items())),
    )


def dispatch_tag(runtime):
    dispatch = getattr(runtime, "rt", None)
    if dispatch is None:
        return None
    need(isinstance(dispatch, dict), "native_runtime_dispatch_dictionary")
    return (
        id(dispatch), id(getattr(runtime, "campaign", None)),
        tuple((name, id(value), id(getattr(value, "__code__", None)))
              for name, value in sorted(dispatch.items())),
        tuple((name, id(getattr(getattr(runtime, name), "__func__", None)),
               id(getattr(getattr(getattr(runtime, name), "__func__", None), "__code__", None)))
              for name in ("encode", "compute", "decode", "wait")),
    )


class LoadedContextGuard:
    """Only source-qualified original bind closures can create a native guard."""
    def __init__(self, campaign, inputs, plan, *, package_root, source_manifest_ref,
                 input_manifest_ref, input_qualification_ref, restore_receipt_ref,
                 runtime_restore_ref, model_binding_ref, model_guard, request,
                 candidate_root=None):
        need(sys.platform.startswith("linux"), "native_context_guard_Linux_only")
        closure = json.loads((HERE/"NATIVE-CLOSURE.json").read_bytes())
        self._initialize(
            campaign, inputs, plan, package_root=package_root,
            source_manifest_ref=source_manifest_ref, input_manifest_ref=input_manifest_ref,
            input_qualification_ref=input_qualification_ref, restore_receipt_ref=restore_receipt_ref,
            runtime_restore_ref=runtime_restore_ref, model_binding_ref=model_binding_ref,
            model_guard=model_guard, request=request, candidate_root=candidate_root or HERE,
            closure=closure, test_token=None)

    @classmethod
    def for_synthetic_tests(cls, campaign, inputs, plan, *, closure, input_adapter, **kwargs):
        """Explicit test path always remains synthetic and worker-ineligible."""
        instance = cls.__new__(cls)
        instance._initialize(campaign, inputs, plan, closure=closure,
                             test_token=_TEST_FACTORY, input_adapter=input_adapter, **kwargs)
        return instance

    @property
    def synthetic(self):
        return self._synthetic

    @property
    def production_bound(self):
        return not self.synthetic and self.runtime is not None

    def _file(self, ref, *, source=False):
        path = (inside(self.package, ref["path"]) if source
                else Path(ref["path"]).resolve(strict=True))
        need(path.is_file() and not Path(ref["path"]).is_symlink(), "regular_bound_evidence_file")
        before = token(path)
        need(type(ref["bytes"]) is int and (0 <= ref["bytes"] if source else 0 < ref["bytes"]) and
             ref["bytes"] <= 32*1024**2 and
             before[2] == ref["bytes"], "bounded_exact_evidence_size")
        raw = path.read_bytes()
        need(token(path) == before and hashlib.sha256(raw).hexdigest() == ref["sha256"],
             "bound_evidence_whole_SHA")
        self.tokens[path] = before
        self.hash_bytes_once += len(raw)
        return path, raw

    def _document(self, ref, *, source=False):
        return json.loads(self._file(ref, source=source)[1])

    def _initialize(self, campaign, inputs, plan, *, package_root, source_manifest_ref,
                    input_manifest_ref, input_qualification_ref, restore_receipt_ref,
                    runtime_restore_ref, model_binding_ref, model_guard, request,
                    candidate_root, closure, test_token, input_adapter=None):
        self._synthetic = test_token is _TEST_FACTORY
        self.package = Path(package_root).resolve(strict=True)
        self.candidate = Path(candidate_root).resolve(strict=True)
        need(self.candidate.is_relative_to(self.package), "candidate_inside_bound_package")
        need(closure["schema"] == "DEV-pulse-loaded-context-closure-v1", "exact_native_closure_schema")
        self.tokens, self.hash_bytes_once = {}, 0
        self.closure = closure
        manifest_path, raw_manifest = self._file(source_manifest_ref, source=True)
        need(manifest_path == self.package/"metadata/SOURCE_MANIFEST-v1.json",
             "original_model_guard_manifest_location")
        manifest = json.loads(raw_manifest)
        rows = manifest["files"]
        need(0 < len(rows) <= 4000 and sum(r["bytes"] for r in rows) <= 128*1024**2,
             "bounded_complete_source_manifest_no_weights")
        self.sources = {r["path"]: r for r in rows}
        need(len(self.sources) == len(rows), "unique_complete_source_manifest")
        for row in rows:
            self._file(row, source=True)
        for row in closure["required_original_package_files"]:
            actual = self.sources.get(row["path"])
            need(actual and all(actual[k] == row[k] for k in ("bytes", "sha256")),
                 "executed_original_runtime_source_closure")
        self.source_manifest_ref = dict(source_manifest_ref)
        if not self.synthetic:
            for name in closure["required_candidate_source_files"]:
                relative = str((self.candidate/name).relative_to(self.package))
                need(relative in self.sources, "complete_candidate_source_manifest")
            need(self.candidate == HERE, "actual_guard_module_inside_candidate")
            pins = json.loads((self.candidate/"NATIVE-SOURCE-PINS.json").read_bytes())
            for row in pins["files"]:
                rel = str((self.candidate/row["path"]).relative_to(self.package))
                need(rel in self.sources and all(self.sources[rel][k] == row[k]
                     for k in ("bytes", "sha256")), "unchanged_executed_native_overlay")
            for row in json.loads((self.candidate/"input/FROZEN-SOURCE-PINS.json").read_bytes())["files"]:
                rel = str((self.candidate/"input/frozen"/row["path"]).relative_to(self.package))
                need(rel in self.sources and all(self.sources[rel][k] == row[k]
                     for k in ("bytes", "sha256")), "closed_input_helper_sources")
        input_path, input_raw = self._file(input_manifest_ref, source=True)
        expected_input = closure["input_manifest"]
        need(all(input_manifest_ref[k] == expected_input[k] for k in ("bytes", "sha256")),
             "exact_qualified_INPUTS_manifest")
        input_manifest = json.loads(input_raw)
        proof = self._document(input_qualification_ref)
        need(proof["schema"] == "rss-dev-pulse-input-qualification-v1" and
             proof["analysis_complete"] is True and proof["input_source_qualified"] is True and
             proof["synthetic"] is self.synthetic and proof["native_runtime_qualified"] is False,
             "actual_CPU_input_qualification_required")
        need(all(proof[k] == 0 for k in ("GPU", "model_calls", "environment_steps", "new_pulse_labels")),
             "input_proof_not_native_or_pulse_evidence")
        expected = {r["uri"]: {k: r[k] for k in ("uri", "bytes", "sha256", "codec")}
                    for r in input_manifest["input_objects"]}
        actual = {r["uri"]: r for r in proof["input_objects"]}
        need(len(actual) == len(proof["input_objects"]) == len(expected) ==
             input_manifest["input_object_count"] and actual == expected and
             proof["input_bytes"] == input_manifest["input_bytes"], "all_exact_CPU_input_objects")
        keys = [(s["root_id"], s["query_index"]) for s in input_manifest["snapshots"]]
        need(len(inputs) == len(proof["snapshots"]) == len(keys) == 8 and
             [s.key for s in inputs] == keys and plan["planned_queries"] == [list(k) for k in keys],
             "exact_eight_qualified_input_objects")
        need(all(s.synthetic is self.synthetic for s in inputs), "input_real_synthetic_scope")
        if input_adapter is None:
            path = self.candidate/"input/adapter.py"
            spec = importlib.util.spec_from_file_location("_DEV_loaded_guard_input_adapter", path)
            input_adapter = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = input_adapter
            spec.loader.exec_module(input_adapter)
        for saved, row in zip(inputs, proof["snapshots"]):
            need(saved.key == (row["root_id"], row["query_index"]) and
                 saved.source_identity == row["source_identity"] and
                 saved.encoded_identities == row["encoded_identities"] and
                 saved.sources == row["source_refs"] and saved.dose == row["actual_dose"],
                 "actual_input_object_source_and_dose")
            for field, proof_key in (("noise", "noise"), ("decode_state", "decode_state"),
                                     ("old_KT", "prefix_KT"), ("old_VT", "prefix_VT")):
                need(input_adapter.array_id(getattr(saved, field)) == row[proof_key],
                     "actual_loaded_input_typed_identity")
            need(input_adapter.identity(saved.raw_observations) == row["raw_C_T_identity"],
                 "actual_loaded_raw_C_T_identity")
            edit_id = input_adapter.pinned("edits_reference").aid
            need(set(saved.modified_v) == set(saved.actual_E) == {"S", "R", "P", "own_null"},
                 "exact_native_payload_role_roster")
            for role, old_arm in (("S", "shared"), ("P", "shared"), ("R", "random_matched")):
                old_dose = row["actual_dose"][old_arm]
                need(edit_id(saved.modified_v[role]) == old_dose["Vmod"] and
                     edit_id(saved.actual_E[role]) == old_dose["actual_E"],
                     "actual_frozen_Vmod_E_match_CPU_proof")
            need(edit_id(saved.modified_v["own_null"]) == edit_id(saved.old_VT) and
                 saved.actual_E["own_null"].shape == saved.old_VT.shape and
                 saved.actual_E["own_null"].dtype.name == "float32" and
                 bool((saved.actual_E["own_null"] == 0).all()),
                 "active_own_null_is_original_VT_and_zero_E")
            original_t = saved.old_calls["natural_T"]
            old_ref = row["source_refs"]["native"]["natural_T"]
            need(all(original_t.ref[k] == old_ref[k] for k in ("path", "bytes", "sha256")),
                 "original_natural_T_prefix_reference")
            for name in ("prefix_mask", "prefix_positions"):
                leaf = original_t.field(name)
                need(hashlib.sha256(leaf.storage.tobytes()).hexdigest() ==
                     leaf.descriptor["payload_sha256"], "original_prefix_leaf_payload_SHA")
                need(input_adapter.array_id(getattr(saved, name)) ==
                     input_adapter.array_id(input_adapter.value(leaf)),
                     "actual_prefix_layout_matches_original_native_T")
        runtime_proof = self._document(runtime_restore_ref)
        need(runtime_proof["schema"] == closure["runtime_restore_schema"] and
             runtime_proof["model_payloads"] == 0 and runtime_proof["dependency_installations"] == 0 and
             runtime_proof["restore_helper"] == "unchanged_composite_restore.restore_pair" and
             all(k in runtime_proof for k in ("base", "overlay", "dependencies", "runtime_manifest_reference")),
             "concrete_original_runtime_restore_receipt")
        runtime_expected = closure["expected_runtime_restore"]
        need(all(runtime_proof[role] == runtime_expected[role] for role in ("base", "overlay")) and
             runtime_proof["expanded_runtime_bytes"] == runtime_expected["expanded_runtime_bytes"],
             "original_complete_capsule_restore_proof")
        dependency = runtime_proof["dependencies"]
        need(dependency["plan"] == runtime_expected["environment_plan"] and
             dependency["qualified_actual_import"] is True and
             dependency["restored_private_environment_rebased"] is True and
             dependency["dependency_installations"] == 0 and
             type(dependency["helper"]["bytes"]) is int and dependency["helper"]["bytes"] > 0 and
             len(dependency["helper"]["sha256"]) == 64, "qualified_original_dependency_binding")
        reference = runtime_proof["runtime_manifest_reference"]
        need(reference["schema"] == "cloud-local-original-runtime-manifest-reference-v1" and
             reference["base_index"] == runtime_expected["base_index"] and
             reference["members"] == runtime_expected["base"]["members"] and
             reference["files"] == runtime_expected["base_file_manifests"] and
             reference["new_remote_GETs"] == reference["new_manifest_copies"] == 0,
             "original_runtime_manifest_reference")
        restore = self._document(restore_receipt_ref)
        model_plan = closure["model_plan"]
        need(restore["frozen_plan"] == {k: model_plan[k] for k in ("bytes", "sha256")},
             "original_model_restore_plan")
        model_rows = closure["model_objects"]
        restored = {r["id"]: r for r in restore["whole_SHA_verified_objects"]}
        need(len(restored) == len(restore["whole_SHA_verified_objects"]) ==
             len(model_rows) == closure["model_object_count"], "all_original_model_objects_restored")
        for row in model_rows:
            observed = restored.get(row["id"])
            need(observed and all(observed[k] == row[k] for k in
                 ("uri", "local_relative", "original_artifact_id")), "model_restore_exact_sources")
        need(request["model_plan"] == model_plan, "request_exact_original_model_plan")
        plan_ref = request["runtime_config"]
        bound_plan = self._document(plan_ref, source=True)
        need(bound_plan == plan, "actual_plan_equals_packaged_plan")
        self.plan_digest, self.request_digest = identity(plan), identity(request)
        model_binding = self._document(model_binding_ref)
        expected_model = closure["expected_model"]
        need(model_binding["model_bound"] is True and model_binding["adapter_bound"] is True and
             model_binding["new_model_forward_calls"] == 0 and
             all(model_binding[k] == value for k, value in expected_model.items()),
             "actual_original_loaded_model_binding")
        for saved in inputs:
            need(all(saved.model_binding[k] == model_binding[k] for k in
                 ("checkpoint_artifact_id", "normalization_sha256", "effective_model_config")),
                 "same_model_normalizer_as_saved_DEV")
        model_path = inside(self.package, closure["original_model_guard"]["path"])
        guard_source = model_path.read_bytes()
        same_code(model_guard, guard_source, model_path, ("bind", "guard"))
        nonlocals = inspect.getclosurevars(model_guard).nonlocals
        need(nonlocals.get("campaign") is campaign and
             nonlocals.get("model") is campaign.policy._model and
             isinstance(nonlocals.get("objects"), dict) and nonlocals["objects"],
             "actual_original_guard_bound_campaign_parameters")
        summary = nonlocals.get("summary")
        need(isinstance(summary, dict) and summary ==
             {k: v for k, v in model_binding.items() if k not in ("leaves", "original_files", "sources")},
             "original_guard_summary_matches_fresh_binding_receipt")
        returned = model_guard(campaign)
        need(returned == summary, "original_model_guard_success")
        self.campaign, self.inputs, self.plan, self.request = campaign, inputs, plan, request
        self.model_guard, self.model_summary = model_guard, dict(summary)
        self.model_objects = nonlocals["objects"]
        self.input_tags = tuple(snapshot_tag(s) for s in inputs)
        self.normalizer_snapshot = normalizers(campaign)
        self.runtime = self.compiled = self.compiled_state = self.compiled_jitted = None
        self.compiled_values = None
        self.dispatch_identity = self.compiled_callable_identity = None
        self.input_qualification_ref = dict(input_qualification_ref)
        self.model_binding_ref = dict(model_binding_ref)
        self.calls = 0

    def bind_runtime(self, runtime):
        need(self.runtime is None and getattr(runtime, "synthetic", None) is self.synthetic,
             "runtime_bound_once_same_real_synthetic_scope")
        compiled = runtime.common_compiled
        need(callable(compiled), "concrete_compiled_native_callable")
        if not self.synthetic:
            need(runtime.campaign is self.campaign and isinstance(runtime.rt, dict) and
                 set(runtime.rt) == {"common", "decode", "wait", "compiled"} and
                 runtime.rt["compiled"] is compiled, "actual_runtime_dispatch_uses_bound_compiled")
            wrapped = getattr(compiled, "__wrapped__", None)
            need(inspect.ismethod(wrapped) and wrapped.__self__ is self.campaign.policy._model,
                 "compiled_query_bound_to_loaded_model")
            query_path = self.candidate/"retained/native/candidate_query.py"
            same_code(wrapped.__func__, query_path.read_bytes(), query_path, ("query",))
            nnx_path = self.package/"repo/src/openpi/shared/nnx_utils.py"
            same_code(compiled, nnx_path.read_bytes(), nnx_path, ("module_jit", "wrapper"))
            captured = inspect.getclosurevars(compiled).nonlocals
            state, jitted = captured.get("state"), captured.get("jitted_fn")
            need(state is not None and callable(jitted) and hasattr(state, "flat_state"),
                 "original_module_jit_frozen_state")
            values = state.flat_state()
            need(set(self.model_objects) <= set(values) and all(
                values[k].value is self.model_objects[k] for k in self.model_objects),
                "compiled_state_same_loaded_parameters")
            self.compiled_state, self.compiled_jitted = state, jitted
            # module_jit freezes the complete NNX state; the old guard binds
            # Param leaves. Keep any additional state objects fixed as well.
            self.compiled_values = {k: variable.value for k, variable in values.items()}
            same_code(runtime.rt["common"], query_path.read_bytes(), query_path,
                      ("production_runtime", "common"))
            need(inspect.getclosurevars(runtime.rt["common"]).nonlocals.get("compiled") is compiled,
                 "native_common_dispatch_captures_actual_compiled")
            same_code(runtime.rt["decode"], query_path.read_bytes(), query_path,
                      ("production_runtime", "<lambda>"))
            need(inspect.getclosurevars(runtime.rt["decode"]).nonlocals.get("campaign") is self.campaign,
                 "native_decode_dispatch_captures_actual_campaign")
            need("jax" in sys.modules and runtime.rt["wait"] is sys.modules["jax"].block_until_ready,
                 "native_runtime_original_wait")
            worker_path = self.candidate/"worker.py"
            worker_source = worker_path.read_bytes()
            for name in ("encode", "compute", "decode", "wait"):
                method = getattr(runtime, name)
                need(inspect.ismethod(method) and method.__self__ is runtime,
                     "native_runtime_actual_bound_method")
                same_code(method.__func__, worker_source, worker_path, ("Runtime", name))
            self.compiled_callable_identity = (
                compiled.__code__, wrapped, wrapped.__func__, wrapped.__func__.__code__)
        self.runtime, self.compiled = runtime, compiled
        self.dispatch_identity = dispatch_tag(runtime)
        return self

    def __call__(self, campaign, inputs, plan, runtime):
        need(self.runtime is not None and campaign is self.campaign and inputs is self.inputs and
             plan is self.plan and runtime is self.runtime and
             runtime.common_compiled is self.compiled, "one_bound_campaign_inputs_plan_runtime_callable")
        need(identity(plan) == self.plan_digest and identity(self.request) == self.request_digest,
             "bound_plan_or_request_mutation")
        need(tuple(snapshot_tag(s) for s in inputs) == self.input_tags,
             "bound_saved_input_objects_replaced")
        need(dispatch_tag(runtime) == self.dispatch_identity, "native_runtime_dispatch_replaced")
        for path, old in self.tokens.items():
            need(path.is_file() and not path.is_symlink() and token(path) == old,
                 "immutable_bound_source_and_evidence_objects")
        need(self.model_guard(campaign) == self.model_summary, "original_loaded_parameter_guard")
        need(normalizers(campaign) == self.normalizer_snapshot, "actual_adapter_normalizer_mutation")
        if not self.synthetic:
            wrapped = getattr(self.compiled, "__wrapped__", None)
            need((self.compiled.__code__, wrapped, wrapped.__func__, wrapped.__func__.__code__) ==
                 self.compiled_callable_identity, "compiled_callable_code_or_binding_changed")
            need(inspect.getclosurevars(runtime.rt["common"]).nonlocals.get("compiled") is self.compiled and
                 inspect.getclosurevars(runtime.rt["decode"]).nonlocals.get("campaign") is self.campaign,
                 "native_runtime_dispatch_closure_changed")
            captured = inspect.getclosurevars(self.compiled).nonlocals
            need(captured.get("state") is self.compiled_state and
                 captured.get("jitted_fn") is self.compiled_jitted, "compiled_closure_identity_changed")
            values = self.compiled_state.flat_state()
            need(set(values) == set(self.compiled_values) and all(
                values[k].value is self.compiled_values[k] for k in values),
                "compiled_parameter_objects_changed")
        self.calls += 1
        return {
            "source_bound": True, "model_bound": True, "adapter_bound": True,
            "inputs_bound": True, "synthetic": self.synthetic,
            "source_manifest": self.source_manifest_ref,
            "input_qualification": self.input_qualification_ref,
            "model_binding": self.model_binding_ref, "guard_calls": self.calls,
            "parameter_check": "original_once_SHA_bind_then_same_loaded_and_compiled_objects",
            "normalizer_check": "same_transform_objects_and_bounded_small_array_hash",
            "normalizer_bytes_per_check": self.normalizer_snapshot[2],
            "source_hash_bytes_at_bind": self.hash_bytes_once,
            "per_call_weight_copy_bytes": 0, "new_model_forward_calls": 0,
            "GPU_authorization_granted_by_guard": False,
        }
