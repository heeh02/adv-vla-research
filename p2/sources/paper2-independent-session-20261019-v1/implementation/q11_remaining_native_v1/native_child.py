"""Credential-free, separately authorized CPU native child; default is dry."""
import importlib.abc
import importlib.util
import json
import os
from pathlib import Path
import signal
import sys
import time
import traceback
import numpy as np
import pilot_contract as C
from campaign_core import need
from resume_core import run_remaining, cumulative_budget, load_witness, PREVIOUS_CALLS
from native_budget import Budget
from physical_writer import Events, write_new


class NoModelImports(importlib.abc.MetaPathFinder):
    def __init__(self):
        self.attempts = 0

    @staticmethod
    def forbidden(fullname):
        return (fullname.split(".")[0] in ("jax", "jaxlib", "flax", "tensorflow") or
                fullname.startswith(("openpi.models", "openpi.policies", "openpi.training")) or
                fullname == "openpi.eval.campaign_policy")

    def find_spec(self, fullname, path=None, target=None):
        if self.forbidden(fullname):
            self.attempts += 1
            raise RuntimeError("model_import_forbidden:"+fullname)
        return None


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


def validate_image_sources(package):
    pins = json.loads((package/"IMAGE-SOURCE-PINS.json").read_bytes())
    spec = importlib.util.find_spec("robosuite")
    need(spec is not None and spec.origin, "installed_robosuite_spec")
    parent = Path(spec.origin).parent.parent
    for row in pins["files"]:
        path = parent/row["path"]
        need(path.is_file() and not path.is_symlink() and
             C.ident(path) == {k: row[k] for k in ("bytes", "sha256")},
             "exact_installed_robot_source")
    return {"source_files_verified": len(pins["files"]), "target_imported_by_check": False}


def load_data(package, stage, inputs, check):
    # All retained reader bodies are in the enclosing immutable source manifest.
    sys.path.insert(0, str(package/"retained_reader"))
    import qualify as Q
    import codec
    import binding
    source, archives = Q.decode_parent(stage, inputs, check)
    parent = Q.parent_checks(source, inputs, check)
    joined = Q.native_join(stage, inputs, source, check)
    refs = {r["path"]: r for r in inputs["input_objects"]}
    cached = {}
    for arm, row in inputs["cached_command_paths"].items():
        check()
        ref = refs[row["compact"]]
        _, leaves = codec.typed(stage/ref["path"], ref, synthetic=False)
        cached[arm] = {"actions": np.array(binding.field(
            leaves, ("adapter7",), (1, 10, 7), "float64").numeric(), copy=True),
                      "raw32": np.array(binding.field(
            leaves, ("raw32",), (1, 10, 32), "float32").numeric(), copy=True)}
    return source, cached, {"parent_archives": archives, "parent": parent, "join": joined}


def run(context_path):
    ctx = json.loads(Path(context_path).read_bytes())
    package, out = Path(ctx["package"]), Path(ctx["science"])
    # Approval and immutable binding are checked before EGL or simulator imports.
    C.authority_gate(ctx["authority"], ctx["source_manifest"], C.ident(package/"INPUTS.json"),
                     t0=ctx["T0_epoch"], deadline=ctx["deadline_epoch"])
    C.verify_sources(package, ctx["source_manifest"])
    need(sys.platform.startswith("linux") and not any(k.startswith(
        ("ONE_VLA_", "AWS_", "BCE_", "BAIDU_", "ADV_VLA_SOURCE")) for k in os.environ),
        "credential_free_Linux_native_child")
    guard = NoModelImports()
    need(not any(guard.forbidden(name) for name in sys.modules), "no_preloaded_model_modules")
    sys.meta_path.insert(0, guard)
    native_deadline = ctx["deadline_epoch"]-20
    budget = None
    retained = witness_raw = None
    def check():
        need(time.time() < native_deadline, "native_absolute_science_deadline")
    # Evidence may still be saved during the separate archive reserve.
    def evidence_check():
        need(time.time() < ctx["archive_deadline_epoch"], "physical_evidence_archive_deadline")
    log = Events(out/"NATIVE-EVENTS.jsonl.gz", check=evidence_check, max_compressed=15*1024**2)
    runtimes = []
    campaign = None
    failure = None
    phase = "image_source_qualification"
    def stop(*_):
        raise TimeoutError("native_absolute_science_deadline")
    signal.signal(signal.SIGALRM, stop)
    signal.signal(signal.SIGTERM, stop)
    signal.setitimer(signal.ITIMER_REAL, max(.001, native_deadline-time.time()))
    try:
        image = validate_image_sources(package)
        log.emit("IMAGE_SOURCE_QUALIFIED", image)
        phase = "parent_and_cached_input_qualification"
        inputs = json.loads((package/"INPUTS.json").read_bytes())
        source, cached, input_proof = load_data(package, Path(ctx["stage"]), inputs, check)
        write_new(out/"INPUT-QUALIFICATION.json", input_proof, limit=1024**2)
        phase = "retained_observation_requalification"
        from reference_loader import load_reference
        retained,witness_raw,reference_proof=load_reference(package,Path(ctx["stage"]),inputs,check)
        write_new(out/"RESUME-REFERENCE.json",reference_proof,limit=1024**2)
        budget=cumulative_budget(load_witness(witness_raw),native_deadline)
        phase = "remaining_physical_campaign"
        sys.path.insert(0, str(package/"retained_cpu3/worker"))
        # Candidate facade import remains distinct from original replay code.
        from native_facade import NativeReplay
        old_package = package/"retained_cpu3"
        old_manifest = json.loads((old_package/"metadata/SOURCE_MANIFEST-v1.json").read_bytes())
        original = json.loads((old_package/"metadata/roots"/(source.root_id+".json")).read_bytes())
        def factory(role):
            runtime = NativeReplay({"role": role, "native_budget": budget, "authority": ctx["authority"],
                "resources": C.RESOURCES, "package_root": str(old_package),
                "source_manifest": old_manifest, "original_runtime_config": original,
                "restored": ctx["restored"], "EGL_probe": ctx["EGL_probe"]}, emit=log.emit)
            runtimes.append(runtime)
            return runtime
        campaign = run_remaining(source, cached, factory, witness=witness_raw, authority=ctx["authority"],
                                emit=log.emit, deadline_epoch=native_deadline, synthetic=False)
        from physical_contrasts import summarize
        write_new(out/"PHYSICAL-CONTRASTS.json", summarize(campaign, retained), limit=1024**2)
    except BaseException as error:
        failure = {"phase": phase, "type": type(error).__name__, "code": str(error)[:180],
                   "frames": [{"file": Path(f.filename).name, "line": f.lineno, "function": f.name}
                              for f in traceback.extract_tb(error.__traceback__)[-8:]]}
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        log.close()
        sys.meta_path.remove(guard)
    summary = {"schema": "paper2-q11-remaining-native-summary-v1", "synthetic": False,
        "completed_remaining_roles": campaign is not None and campaign["completed_remaining_roles"] and failure is None,
        "failure": failure, "GPU": 0, "new_model_calls": sum(r.model_calls for r in runtimes),
        "model_attempts": sum(r.model_attempts for r in runtimes),
        "model_import_attempts": guard.attempts, "native_budget_cumulative": None if budget is None else budget.snapshot(),
        "prior_native_calls":PREVIOUS_CALLS,
        "new_native_calls":None if budget is None else {k:budget.counts[k]-v for k,v in PREVIOUS_CALLS.items()},
        "new_environment_steps":sum(r.env_step_calls for r in runtimes),
        "previous_environment_steps":140,
        "cumulative_environment_steps":140+sum(r.env_step_calls for r in runtimes),
        "retained_N_original_status":"FAILED_NO_RETRY",
        "retained_N_close_failure":{"code":"one_window_finish","type":"PairingError"},
        "source_and_N_reexecuted":False,
        "reference_witness_verified":witness_raw is not None,
        "old_full_replay_qualification_changed": False, "task_transfer_or_success_rate_established": False,
        "roles": [] if campaign is None else
                 [{k: v for k, v in row.items() if k not in
                   ("trajectory", "queries", "start_critical_witness")}
                  for row in campaign["roles"]]}
    if campaign is not None:
        write_new(out/"PHYSICAL-CAMPAIGN.json.gz", campaign, limit=8*1024**2, compressed=True)
    write_new(out/"NATIVE-SUMMARY.json", summary, limit=1024**2)
    return summary


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(json.dumps({"status": "DRY_NO_NATIVE_IMPORT_OR_ENVIRONMENT", "GPU": 0}))
    else:
        result = run(sys.argv[1])
        print(json.dumps({"completed_remaining_roles": result["completed_remaining_roles"],
                          "new_env_steps": result["new_environment_steps"]}))
        raise SystemExit(0 if result["completed_remaining_roles"] else 2)
