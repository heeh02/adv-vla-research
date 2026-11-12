"""Local CPU cloud supervisor; no credentials, BOS access or automatic retries."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
import pilot_contract as C
from campaign_core import need
from physical_writer import write_new
from native_child import module, NoModelImports

HERE = Path(__file__).resolve().parent


def stage_gate(stage, inputs, check):
    refs = inputs["input_objects"]
    need(inputs["schema"] == "paper2-q11-remaining-native-inputs-v1" and
         len(refs) == C.INPUT_COUNT and len({r["path"] for r in refs}) == C.INPUT_COUNT and
         sum(r["bytes"] for r in refs) == C.INPUT_BYTES, "fixed_physical_input_roster")
    need(stage.is_dir() and not stage.is_symlink() and
         {p.name for p in stage.iterdir()} == {r["path"] for r in refs}, "exact_physical_stage")
    import hashlib
    for row in refs:
        check()
        path = stage/row["path"]
        need(path.is_file() and not path.is_symlink() and path.stat().st_size == row["bytes"],
             "exact_physical_input_bytes")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for raw in iter(lambda: stream.read(1024**2), b""):
                check()
                digest.update(raw)
        need(digest.hexdigest() == row["sha256"], "exact_physical_input_SHA")


def bounded_wait(child, observed_log_bytes, check):
    while child.poll() is None:
        check()
        need(observed_log_bytes[0] <= 2*1024**2, "native_child_log_2MiB")
        try:
            child.wait(timeout=.5)
        except subprocess.TimeoutExpired:
            pass


def run(context_path):
    ctx = json.loads(Path(context_path).read_bytes())
    stage, out, work = (Path(ctx[k]) for k in ("stage", "science", "runtime_work"))
    C.authority_gate(ctx["authority"], ctx["source_manifest"], C.ident(HERE/"INPUTS.json"),
                     t0=ctx["T0_epoch"], deadline=ctx["deadline_epoch"])
    C.verify_sources(HERE, ctx["source_manifest"])
    need(sys.platform.startswith("linux") and not any(k.startswith(
        ("ONE_VLA_", "AWS_", "BCE_", "BAIDU_", "ADV_VLA_SOURCE")) for k in os.environ),
        "credential_free_cloud_supervisor")
    need(not out.exists() and not work.exists(), "new_physical_output_and_runtime_directory")
    out.mkdir()
    work.mkdir()
    budget = {"last_scan": 0., "peak": 0}
    def check():
        need(time.time() < ctx["deadline_epoch"]-10, "absolute_physical_science_deadline")
        # Restore checks are frequent; don't walk 30k files for every tar block.
        if time.monotonic()-budget["last_scan"] >= 2:
            total = sum(p.stat().st_size for root in (work, stage, out)
                        for p in root.rglob("*") if p.is_file())
            need(total <= C.SCRATCH_BYTES-64*1024**2, "physical_private_scratch_4GiB")
            budget["peak"] = max(total, budget["peak"])
            budget["last_scan"] = time.monotonic()
    child = None
    failure = None
    native = None
    probe = None
    log_thread = None
    log_bytes = [0]
    phase = "input_body_gate"
    def stop(*_):
        raise TimeoutError("supervisor_science_deadline")
    signal.signal(signal.SIGALRM, stop)
    signal.signal(signal.SIGTERM, stop)
    signal.setitimer(signal.ITIMER_REAL, max(.001, ctx["deadline_epoch"]-10-time.time()))
    try:
        inputs = json.loads((HERE/"INPUTS.json").read_bytes())
        stage_gate(stage, inputs, check)
        phase = "software_EGL_probe"
        old = HERE/"retained_cpu3"
        backend = module("backend_probe", old/"worker/backend_probe.py")
        probe = backend.probe_isolated(deadline_epoch=min(ctx["deadline_epoch"], time.time()+65))
        write_new(out/"EGL-PROBE.json", probe, limit=1024**2)
        need(probe["status"] == "SOFTWARE_EGL_AVAILABLE", "same_job_software_EGL_required")
        phase = "exact_model_free_runtime_restore"
        # Original restoration helper requires only these constants/functions.
        previous = sys.modules.get("contract")
        sys.modules["contract"] = C
        try:
            adapter = module("_q11_exact_runtime_adapter", old/"worker/runtime_adapter.py")
        finally:
            if previous is None:
                sys.modules.pop("contract", None)
            else:
                sys.modules["contract"] = previous
        paths = {r["uri"]: str(stage/r["path"]) for r in inputs["input_objects"]}
        restored, env, proof = adapter.restore(old, work, paths, check=check)
        write_new(out/"RUNTIME-RESTORE.json", proof, limit=1024**2)
        phase = "native_child"
        child_context = {**ctx, "package": str(HERE), "restored": restored, "EGL_probe": probe}
        context = work/"NATIVE-CONTEXT.json"
        context.write_bytes(C.encoded(child_context))
        context.chmod(0o440)
        log = out/"NATIVE-CHILD-LOG.txt"
        env.update(OMP_NUM_THREADS="8", OPENBLAS_NUM_THREADS="8", MKL_NUM_THREADS="8")
        child = subprocess.Popen([sys.executable, "-B", str(HERE/"native_child.py"), str(context)],
            cwd=work/"science-tmp", env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            start_new_session=True)
        def drain_log():
            with log.open("xb") as stream:
                for chunk in iter(lambda: child.stdout.read(4096), b""):
                    remaining = max(0, 2*1024**2-log_bytes[0])
                    stream.write(chunk[:remaining])
                    log_bytes[0] += len(chunk)
            child.stdout.close()
        log_thread = threading.Thread(target=drain_log, daemon=True)
        log_thread.start()
        bounded_wait(child, log_bytes, check)
        log_thread.join(timeout=2)
        need(not log_thread.is_alive() and log_bytes[0] <= 2*1024**2,
             "complete_bounded_native_child_log")
        path = out/"NATIVE-SUMMARY.json"
        if path.exists():
            native = json.loads(path.read_bytes())
        need(child.returncode == 0 and native is not None and native["completed_remaining_roles"],
             "native_campaign_failed_no_retry")
    except BaseException as error:
        failure = {"phase": phase, "type": type(error).__name__, "code": str(error)[:180]}
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=3)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait(timeout=3)
        path = out/"NATIVE-SUMMARY.json"
        if native is None and path.exists():
            try:
                native = json.loads(path.read_bytes())
            except Exception as error:
                failure = failure or {"phase": "partial_native_summary",
                                      "type": type(error).__name__, "code": "raw_file_preserved"}
        if log_thread is not None:
            log_thread.join(timeout=2)
    result = {"schema": "paper2-q11-remaining-native-result-v1", "synthetic": False,
        "status": "COMPLETE" if failure is None else "FAILED_OR_PARTIAL_PRESERVED",
        "completed_remaining_roles": failure is None and native is not None and native["completed_remaining_roles"],
        "source_manifest": ctx["source_manifest"], "inputs_identity": C.ident(HERE/"INPUTS.json"),
        "input_objects": C.INPUT_COUNT, "input_bytes": C.INPUT_BYTES,
        "GPU": 0, "resources": C.RESOURCES, "failure": failure,
        "new_model_calls": native["new_model_calls"] if native else (None if child else 0),
        "new_env_steps": native["new_environment_steps"] if native else (None if child else 0),
        "runtime_imports_and_environment_allowed_only_by_new_direct_authority": True,
        "previous_environment_steps":140,
        "cumulative_environment_steps":native["cumulative_environment_steps"] if native else None,
        "retained_N_original_status":"FAILED_NO_RETRY",
        "source_and_N_reexecuted":False,
        "native_summary_observed": native is not None,
        "native_log_bytes_observed": log_bytes[0], "native_log_retention_bytes": 2*1024**2,
        "scratch_sampled_peak_bytes": budget["peak"], "automatic_retries": 0,
        "scientific_claim": "One source-qualified five-command case; no success rate or task transfer."}
    write_new(out/"PHYSICAL-RESULT.json", result, limit=1024**2)
    return result


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(json.dumps({"status": "DRY_NO_RESTORE_OR_ENVIRONMENT", "GPU": 0}))
    else:
        result = run(sys.argv[1])
        print(json.dumps({"status": result["status"], "completed_remaining_roles": result["completed_remaining_roles"]}))
        raise SystemExit(0 if result["completed_remaining_roles"] else 2)
