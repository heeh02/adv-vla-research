"""Credential-free DEV72 child entry; default is an inert description.

Only the separately authorized cloud parent can supply the sealed combined
package, restored native environment, exact inputs, and fresh execution context.
No cloud client, network download, simulator, or synthetic runtime is provided.
"""
import argparse
from contextlib import contextmanager
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import sys
import time

HERE = Path(__file__).resolve().parent
MAX_SMALL = 2 * 1024**2
MAX_ARCHIVE = 256 * 1024**2
MAX_SOURCE_BYTES = 128 * 1024**2
SCHEMA = "DEV-pulse-native-execution-context-v1"
BAD_ENV = ("ONE_VLA_", "AWS_", "BCE_", "BAIDU_", "ADV_VLA_SOURCE",
           "NIGHT_GUARD_", "GOOGLE_APPLICATION_CREDENTIALS", "AZURE_")


class EntryFault(ValueError):
    pass


def need(value, code):
    if not value:
        raise EntryFault(code)


def encoded(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                       allow_nan=False) + "\n").encode()


def relative(name):
    need(isinstance(name, str) and name and "\\" not in name,
         "canonical_package_relative_path")
    p = PurePosixPath(name)
    need(not p.is_absolute() and str(p) == name and
         all(x not in ("", ".", "..") for x in name.split("/")),
         "canonical_package_relative_path")
    return name


def regular(path, root=None):
    path = Path(path)
    need(path.is_file() and not path.is_symlink(), "owned_regular_file")
    if root is not None:
        root = Path(root).resolve(strict=True)
        need(path.resolve(strict=True).is_relative_to(root), "owned_file_inside_root")
        need(not any(p.is_symlink() for p in path.parents if p != root and p.is_relative_to(root)),
             "no_symlink_parent")
    return path


def identity(path, check=lambda: None):
    path = regular(path)
    before = path.stat()
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for raw in iter(lambda: stream.read(1024**2), b""):
            check()
            h.update(raw)
    after = path.stat()
    need((before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) ==
         (after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns),
         "immutable_file_during_hash")
    return {"bytes": after.st_size, "sha256": h.hexdigest()}


def read_json(path, *, limit=MAX_SMALL, expected=None):
    path = regular(path)
    need(path.stat().st_size <= limit, "bounded_JSON")
    raw = path.read_bytes()
    ref = {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
    if expected is not None:
        need(ref == {k: expected[k] for k in ("bytes", "sha256")}, "exact_JSON_identity")
    return json.loads(raw), {"path": str(path.resolve()), **ref}


def write_json(path, value):
    raw = encoded(value)
    need(len(raw) <= MAX_SMALL, "small_terminal_record")
    with Path(path).open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    return {"path": Path(path).name, "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest()}


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    loaded = importlib.util.module_from_spec(spec)
    sys.modules[name] = loaded
    spec.loader.exec_module(loaded)
    return loaded


def verify_source(package, candidate, manifest_sha, check=lambda: None):
    """Check every original/added source before importing candidate code."""
    package, candidate = Path(package).resolve(strict=True), Path(candidate).resolve(strict=True)
    need(candidate == package / "candidate" and HERE == package / "execution",
         "actual_entry_and_candidate_inside_combined_package")
    mp = package / "metadata/SOURCE_MANIFEST-v1.json"
    manifest, source_ref = read_json(mp, limit=4 * MAX_SMALL)
    need(source_ref["sha256"] == manifest_sha, "parent_bound_combined_manifest_SHA")
    rows = manifest["files"]
    need(0 < len(rows) <= 4000 and
         all(type(r["bytes"]) is int and 0 <= r["bytes"] <= 32 * 1024**2 for r in rows) and
         sum(r["bytes"] for r in rows) <= MAX_SOURCE_BYTES, "bounded_full_source_manifest")
    by = {}
    for row in rows:
        check()
        name = relative(row["path"])
        need(name not in by, "unique_full_source_roster")
        path = regular(package / name, package)
        need(identity(path, check) == {k: row[k] for k in ("bytes", "sha256")},
             "exact_full_source_member")
        by[name] = row
    need("execution/science_entry.py" in by, "executed_launcher_bound_in_full_manifest")
    closure, _ = read_json(candidate / "NATIVE-CLOSURE.json")
    oldref = closure["source_provenance"]["PVP_source_manifest"]
    old, _ = read_json(package / "metadata/SOURCE_MANIFEST-original-v1.json",
                       expected=oldref)
    for row in old["files"]:
        observed = by.get(row["path"])
        need(observed is not None and all(observed[k] == row[k] for k in ("bytes", "sha256")),
             "all_original_PVP_source_bytes_unchanged")
    for row in closure["required_original_package_files"]:
        need(row["path"] in by and all(by[row["path"]][k] == row[k]
             for k in ("bytes", "sha256")), "complete_executed_native_source_closure")
    sealed, _ = read_json(candidate / "SOURCE-MANIFEST.json")
    for row in sealed["files"]:
        name = "candidate/" + relative(row["path"])
        need(name in by and all(by[name][k] == row[k] for k in ("bytes", "sha256")),
             "sealed_candidate_source_unchanged")
    actual = set()
    for path in package.rglob("*"):
        need(not path.is_symlink(), "no_symlink_in_combined_source")
        if path.is_file():
            actual.add(str(path.relative_to(package)))
    need(actual == set(by) | {"metadata/SOURCE_MANIFEST-v1.json"},
         "no_unmanifested_executable_or_source_file")
    return {"package": package, "candidate": candidate, "closure": closure,
            "source_manifest_ref": {"path": "metadata/SOURCE_MANIFEST-v1.json",
                                    **{k: source_ref[k] for k in ("bytes", "sha256")}},
            "sources": by}


def validate_context(context):
    need(context.get("schema") == SCHEMA, "exact_native_context_schema")
    need(isinstance(context.get("run_id"), str) and
         re.fullmatch(r"[A-Za-z0-9_.-]{1,160}", context["run_id"]), "bounded_run_identity")
    for key in ("science_deadline_epoch", "archive_deadline_epoch", "release_deadline_epoch"):
        need(type(context.get(key)) in (int, float) and math.isfinite(context[key]),
             "finite_absolute_deadlines")
    need(time.time() < context["science_deadline_epoch"] <
         context["archive_deadline_epoch"] < context["release_deadline_epoch"],
         "ordered_future_absolute_deadlines")
    for key in ("package_root", "candidate_root", "repo_root", "checkpoint_root",
                "input_stage_root", "input_qualification_path", "private_root",
                "model_restore_receipt_path", "runtime_restore_receipt_path", "output_dir"):
        need(isinstance(context.get(key), str) and Path(context[key]).is_absolute(),
             "absolute_context_paths")
    need(Path(context["repo_root"]).resolve() == Path(context["package_root"]).resolve() / "repo",
         "original_repo_path")
    need(not any(k.startswith(BAD_ENV) or k.upper().endswith(
         ("_ACCESS_KEY", "_SECRET_KEY", "_SESSION_TOKEN", "_API_KEY", "_PASSWORD",
          "_SECRET", "_TOKEN")) for k in os.environ),
         "credential_free_science_process")


def validate_preload_evidence(context, bound, plan, check):
    """Real restore records precede model construction; no weight rehash here.

    Transport already whole-SHA verifies all bodies. Original model_guard.bind
    performs the original 17-file and actual loaded parameter verification once.
    """
    closure, package = bound["closure"], bound["package"]
    request = context["request"]
    need(request["model_plan"] == closure["model_plan"], "request_original_model_plan")
    expected_plan = {"path": "candidate/PLAN-CANDIDATE.json",
                     **identity(bound["candidate"] / "PLAN-CANDIDATE.json", check)}
    need(request["runtime_config"] == expected_plan, "request_exact_candidate_plan")
    saved_plan, _ = read_json(package / request["runtime_config"]["path"],
                              expected=request["runtime_config"])
    need(saved_plan == plan, "same_packaged_plan_object")
    cpu, cpu_ref = read_json(context["input_qualification_path"],
        expected=plan["CPU_input_qualification"]["proof"])
    need(cpu["schema"] == "rss-dev-pulse-input-qualification-v1" and
         cpu["analysis_complete"] is True and cpu["input_source_qualified"] is True and
         cpu["synthetic"] is False and cpu["native_runtime_qualified"] is False and
         all(cpu[k] == 0 for k in ("GPU", "model_calls", "environment_steps", "new_pulse_labels")),
         "actual_CPU_qualification_not_native_evidence")
    model, model_ref = read_json(context["model_restore_receipt_path"])
    need(model["frozen_plan"] == {k: closure["model_plan"][k] for k in ("bytes", "sha256")} and
         len(model["whole_SHA_verified_objects"]) == closure["model_object_count"] == 17,
         "all17_original_model_restore_objects")
    actual = {row["id"]: row for row in model["whole_SHA_verified_objects"]}
    need(len(actual) == 17, "unique_model_restore_objects")
    root = Path(context["checkpoint_root"]).resolve(strict=True).parents[1]
    for row in closure["model_objects"]:
        check()
        need(row["id"] in actual and all(actual[row["id"]].get(k) == row[k]
             for k in ("uri", "local_relative", "original_artifact_id", "bytes", "sha256")),
             "exact_model_restore_identity")
        path = regular(root / relative(row["local_relative"]), root)
        need(path.stat().st_size == row["bytes"], "restored_model_body_present")
    runtime, runtime_ref = read_json(context["runtime_restore_receipt_path"])
    expected = closure["expected_runtime_restore"]
    need(runtime["schema"] == closure["runtime_restore_schema"] and
         runtime["model_payloads"] == runtime["dependency_installations"] == 0 and
         runtime["restore_helper"] == "unchanged_composite_restore.restore_pair" and
         all(runtime[role] == expected[role] for role in ("base", "overlay")) and
         runtime["expanded_runtime_bytes"] == expected["expanded_runtime_bytes"],
         "actual_original_runtime_restore")
    dep = runtime["dependencies"]
    need(dep["plan"] == expected["environment_plan"] and
         dep["qualified_actual_import"] is True and
         dep["restored_private_environment_rebased"] is True and
         dep["dependency_installations"] == 0, "original_private_environment_qualified")
    ref = runtime["runtime_manifest_reference"]
    need(ref == context["runtime_context"]["runtime_manifest_ref"] and
         ref["base_index"] == expected["base_index"] and
         ref["files"] == expected["base_file_manifests"] and
         ref["members"] == expected["base"]["members"] and
         ref["new_remote_GETs"] == ref["new_manifest_copies"] == 0,
         "exact_runtime_manifest_reference")
    with original_worker_imports(package):
        rc = module("_rss_pulse_original_runtime_context", package / "worker/runtime_context.py")
        parsed = rc.parse(ref, context["private_root"], package)
    need(len(parsed) == ref["members"], "actual_runtime_manifest_hydrated")
    check()
    return {"input_qualification_ref": cpu_ref, "restore_receipt_ref": model_ref,
            "runtime_restore_ref": runtime_ref, "cpu_proof": cpu}


@contextmanager
def original_worker_imports(package):
    old = sys.modules.pop("contract", None)
    paths = sys.path.copy()
    try:
        sys.path.insert(0, str(Path(package) / "worker"))
        yield
    finally:
        sys.path[:] = paths
        sys.modules.pop("contract", None)
        if old is not None:
            sys.modules["contract"] = old


def load_campaign(context):
    """The only CampaignPolicy constructor; reachable after all preload gates."""
    package = Path(context["package_root"]).resolve(strict=True)
    need(not any(name == "openpi" or name.startswith("openpi.") for name in sys.modules),
         "fresh_child_no_preloaded_model_modules")
    sys.path[:0] = [str(package / "repo/src"),
                   str(package / "repo/packages/openpi-client/src")]
    from openpi.eval.campaign_policy import CampaignPolicy
    origin = Path(sys.modules[CampaignPolicy.__module__].__file__).resolve(strict=True)
    need(origin == package / "repo/src/openpi/eval/campaign_policy.py",
         "CampaignPolicy_from_original_source")
    return CampaignPolicy(Path(context["checkpoint_root"]))


def connected_pipeline(context, bound, plan, worker, storage, result, check):
    """Sequential connection only; production obtains modules from sealed paths."""
    result["step"] = "validate_actual_preload_evidence"
    evidence = validate_preload_evidence(context, bound, plan, check)
    result["step"] = "load_exact_saved_inputs"
    check()
    manifest_path = bound["candidate"] / "input/INPUTS.json"
    manifest, manifest_ref = read_json(manifest_path, expected=bound["closure"]["input_manifest"])
    loaded = worker.input_module().load(context["input_stage_root"], manifest=manifest, check=check)
    # The full concrete guard rechecks all typed arrays and source/dose objects.
    need(loaded["proof"]["snapshots"] == evidence["cpu_proof"]["snapshots"] and
         loaded["proof"]["input_bytes"] == evidence["cpu_proof"]["input_bytes"] and
         {r["uri"]: r for r in loaded["proof"]["input_objects"]} ==
         {r["uri"]: r for r in evidence["cpu_proof"]["input_objects"]},
         "fresh_input_loading_matches_actual_CPU_qualification")
    inputs = loaded["snapshots"]
    result["step"] = "load_original_CampaignPolicy_once"
    check()
    result["model_loads_started"] += 1
    campaign = load_campaign(context)
    result["model_loads_completed"] += 1
    result["step"] = "bind_original_loaded_model"
    check()
    gm = module("_rss_pulse_original_model_guard",
                bound["package"] / bound["closure"]["original_model_guard"]["path"])
    model_guard = gm.bind(campaign, package_root=bound["package"],
        checkpoint_root=context["checkpoint_root"],
        restore_receipt_path=context["model_restore_receipt_path"], request=context["request"])
    binding_path = Path(context["model_restore_receipt_path"]).parent / "MODEL-BINDING-v1.json"
    binding, binding_ref = read_json(binding_path)
    result["model_binding"] = {k: binding[k] for k in
        ("model_bound", "adapter_bound", "loaded_parameter_tree_sha256", "normalization_sha256")}
    result["step"] = "bind_concrete_loaded_context"
    input_ref = {"path": str(manifest_path.relative_to(bound["package"])),
                 **{k: manifest_ref[k] for k in ("bytes", "sha256")}}
    # Use the worker's own sibling-loaded class; separate imports fail its exact
    # type gate. A callback returning booleans never enters the native branch.
    guard = worker.LoadedContextGuard(campaign, inputs, plan,
        package_root=bound["package"], source_manifest_ref=bound["source_manifest_ref"],
        input_manifest_ref=input_ref, input_qualification_ref=evidence["input_qualification_ref"],
        restore_receipt_ref=evidence["restore_receipt_ref"],
        runtime_restore_ref=evidence["runtime_restore_ref"], model_binding_ref=binding_ref,
        model_guard=model_guard, request=context["request"], candidate_root=bound["candidate"])
    need(type(guard) is worker.LoadedContextGuard and not guard.synthetic,
         "canonical_nonsynthetic_loaded_context_guard")
    result["step"] = "native72_worker"
    check()
    ledger = storage.CompactLedger(Path(context["output_dir"]) / "ledger", plan,
                                   context["science_deadline_epoch"])
    result["worker_started"] = True
    outcome = worker.run(campaign, inputs, plan, ledger=ledger, source_guard=guard,
                        deadline_epoch=context["science_deadline_epoch"], authority=context["authority"])
    result["worker_returned"] = True
    result["worker_status"] = outcome["status"]
    result["worker_complete"] = outcome["complete"]
    result["qualification_complete"] = outcome["qualification_complete"]
    result["charged_calls"] = outcome["charged_calls"]
    result["returned_native_calls"] = outcome["returned_native_calls"]
    result["completed_calls"] = outcome["completed_calls"]
    result["passed_calls"] = outcome["passed_calls"]
    result["worker_terminal"] = {"path": "ledger/" + outcome["terminal_ref"]["path"],
                                **{k: outcome["terminal_ref"][k] for k in ("bytes", "sha256")}}
    result["status"] = outcome["status"]
    return outcome


def archive_descriptors(out):
    """Immutable file identities, including partial failures, with a hard cap."""
    out = Path(out).resolve(strict=True)
    rows, total = [], 0
    for path in sorted(out.rglob("*")):
        need(not path.is_symlink(), "no_output_symlinks")
        if path.is_file() and path.name != "ARCHIVE-MANIFEST.json":
            regular(path, out)
            ref = identity(path)
            total += ref["bytes"]
            need(total <= MAX_ARCHIVE - MAX_SMALL, "bounded_scientific_archive_with_manifest_reserve")
            rows.append({"path": str(path.relative_to(out)), **ref})
    return {"schema": "DEV-pulse-native-output-manifest-v1", "files": rows,
            "body_bytes": total, "archive_cap_bytes": MAX_ARCHIVE,
            "includes_partial_files": True, "cloud_archive_confirmed": False,
            "GPU_release_confirmed": False}


def execute(context, source_manifest_sha256):
    """Cloud-native path; no injectable runtime and no synthetic authority."""
    need(isinstance(context.get("output_dir"), str) and Path(context["output_dir"]).is_absolute(),
         "absolute_output_directory")
    out = Path(context["output_dir"])
    need(not out.exists() and not out.is_symlink(), "fresh_output_directory_no_resume")
    out.mkdir(parents=True, exist_ok=False)
    result = {"schema": "DEV-pulse-native-launcher-result-v1", "run_id": context["run_id"],
        "synthetic": False, "status": "FAILED_PRESERVED", "worker_complete": False,
        "qualification_complete": False, "worker_started": False, "worker_returned": False,
        "model_loads_started": 0, "model_loads_completed": 0, "charged_calls": [],
        "returned_native_calls": [],
        "completed_calls": [], "passed_calls": [], "environment_steps": 0, "training_updates": 0,
        "automatic_retry": False, "new_scientific_grid": False, "GPU_release_confirmed": False,
        "source_manifest_sha256": source_manifest_sha256, "step": "verify_source"}

    def check():
        need(time.time() < context["science_deadline_epoch"], "absolute_science_deadline")

    try:
        validate_context(context)
        need(sys.platform.startswith("linux") and sys.version_info[:2] == (3, 12),
             "original_cloud_Linux_Python312_only")
        check()
        bound = verify_source(context["package_root"], context["candidate_root"],
                              source_manifest_sha256, check)
        # Contract is stdlib-only; worker import also has no policy/JAX load.
        contract = module("_rss_pulse_candidate_contract", bound["candidate"] / "contract.py")
        plan, _ = read_json(bound["candidate"] / "PLAN-CANDIDATE.json")
        contract.require_native_authority(plan, context.get("authority"))
        result["step"] = "load_sealed_candidate_components"
        worker = module("_rss_pulse_execution_worker", bound["candidate"] / "worker.py")
        storage = module("_rss_pulse_execution_storage", bound["candidate"] / "storage/compact.py")
        connected_pipeline(context, bound, plan, worker, storage, result, check)
    except BaseException as error:
        code = str(error)
        if not re.fullmatch(r"[A-Za-z0-9_:.-]{1,160}", code):
            code = "external_exception_redacted"
        result.update(status="FAILED_PRESERVED", worker_complete=False,
                      qualification_complete=False, error_type=type(error).__name__, error_code=code)
    finally:
        # If the worker was interrupted during finalization, do not infer that
        # its charged calls were unstarted. Its durable ledger is authoritative.
        terminal = out / "ledger/TERMINAL.json"
        terminal_read = False
        if terminal.is_file():
            try:
                payload, terminal_ref = read_json(terminal)
                recorded = payload["result"]
                for key in ("charged_calls", "returned_native_calls", "completed_calls", "passed_calls"):
                    result[key] = recorded.get(key, [])
                result["worker_terminal"] = {"path": "ledger/TERMINAL.json",
                    **{k: terminal_ref[k] for k in ("bytes", "sha256")}}
                terminal_read = True
            except BaseException as error:
                result.update(status="FAILED_PRESERVED", worker_complete=False,
                    qualification_complete=False, terminal_read_error=type(error).__name__)
        result["charged_status"] = ("read_worker_terminal" if terminal_read
            else "unknown_consult_durable_ledger" if result["worker_started"] else "worker_not_started")
        result["finished_epoch"] = time.time()
        write_json(out / "RESULT.json", result)
        write_json(out / "ARCHIVE-MANIFEST.json", archive_descriptors(out))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context", type=Path)
    parser.add_argument("--source-manifest-sha256")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    if not args.execute:
        print(json.dumps({"status": "DRY_ONLY_GPU0", "model_loads": 0, "model_calls": 0,
            "cloud_calls": 0, "environment_steps": 0, "planned_full_generations": 72,
            "execution_requires": "explicit authority and exact restored cloud context"}, sort_keys=True))
        return 0
    need(args.context is not None and args.source_manifest_sha256 is not None,
         "explicit_context_and_manifest_pin")
    context, _ = read_json(args.context)
    result = execute(context, args.source_manifest_sha256)
    print(json.dumps({k: result[k] for k in ("status", "run_id", "worker_complete")}, sort_keys=True))
    return 0 if result["worker_complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
