"""Actual saved-input qualification and explicitly synthetic storage checks."""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
import time
import traceback
import zipfile
import numpy as np

HERE = Path(__file__).resolve().parent


def need(ok, code):
    if not ok:
        raise ValueError(code)


def fid(path):
    raw = Path(path).read_bytes()
    return {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def write(path, value):
    raw = (json.dumps(value, sort_keys=True, separators=(",", ":"),
                      allow_nan=False) + "\n").encode()
    with Path(path).open("xb") as f:
        f.write(raw)
    return fid(path)


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


def input_metadata(saved, array_identity):
    def ids(value):
        return {k: ids(v) for k, v in value.items()} if isinstance(value, dict) else array_identity(value)
    return {
        "root_id": saved.root_id, "query_index": saved.query_index,
        "source_identity": saved.source_identity, "sources": saved.sources,
        "encoded_identities": saved.encoded_identities,
        "original_arrays": {k: ids(getattr(saved, k)) for k in
                            ("noise", "decode_state", "old_KT", "old_VT", "modified_v", "actual_E")},
        "large_input_bodies": "original_immutable_references_only",
    }


def check_input_record(path, saved, array_identity, metadata_encoder):
    with np.load(path, allow_pickle=False) as archive:
        schema = json.loads(archive["__schema__"].tobytes())
        metadata_arrays = {}
        expected_metadata = metadata_encoder(input_metadata(saved, array_identity),
                                             path="root/metadata", arrays=metadata_arrays)
        need(not metadata_arrays and schema["metadata"] == expected_metadata,
             "same_actual_input_metadata_and_typed_refs")
        need(set(schema["arrays"]) == {"prefix_mask", "prefix_positions"},
             "exact_small_input_array_roster")
        for name, descriptor in schema["arrays"].items():
            body = archive[descriptor["__array__"]]
            need(list(body.shape) == descriptor["shape"] and
                 body.nbytes == descriptor["bytes"] and
                 hashlib.sha256(body.tobytes()).hexdigest() == descriptor["payload_sha256"],
                 "actual_input_record_roundtrip")
            expected = np.asarray(getattr(saved, name))
            need(body.shape == expected.shape and body.dtype == expected.dtype and
                 body.tobytes() == expected.tobytes(), "same_actual_input_array")


def storage_software_qa(out, deadline):
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1",
           "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "JAX_PLATFORMS": "cpu"}
    timed_out = False
    try:
        run = subprocess.run([sys.executable, "-B", "-W", "error", "-m", "unittest",
                              "-v", "test_compact.py"], cwd=HERE / "storage", env=env,
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                             timeout=max(1., min(180., deadline - time.time())))
        raw_log, returncode = run.stdout + run.stderr, run.returncode
    except subprocess.TimeoutExpired as error:
        timed_out, returncode = True, None
        raw_log = (error.stdout or b"") + (error.stderr or b"")
    log = raw_log.decode("utf-8", errors="replace")
    count = re.search(r"Ran (\d+) tests?", log)
    passed = bool(not timed_out and len(raw_log) <= 128 * 1024 and returncode == 0 and
                  count is not None and int(count.group(1)) == 11 and re.search(r"\nOK\s*$", log))
    write(Path(out) / "storage-software-QA.json", {
        "schema": "rss-compact-storage-software-QA-v1", "synthetic": True,
        "status": "PASS" if passed else "TIMEOUT" if timed_out else "FAIL",
        "tests_passed": 11 if passed else None, "returncode": returncode,
        "reported_tests_run": int(count.group(1)) if count is not None else None,
        "new_model_calls": 0, "new_env_steps": 0,
        "source": fid(HERE / "storage/compact.py"),
        "log": raw_log[:128 * 1024].decode("utf-8", errors="replace"),
        "log_bytes": len(raw_log), "log_sha256": hashlib.sha256(raw_log).hexdigest(),
        "log_truncated": len(raw_log) > 128 * 1024,
        "native_qualification_claim": False})
    return passed


def main():
    parser = argparse.ArgumentParser()
    for name in ("stage", "out", "source-manifest-sha256", "deadline-epoch"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    start = time.monotonic()
    deadline = float(args.deadline_epoch)
    out = Path(args.out)
    out.mkdir(exist_ok=True)
    need(not any(out.iterdir()), "fresh_output_directory")
    work = out.parent / "input-qa-private-work"
    phase, timing = "source_verification", {}
    protocol = json.loads((HERE / "PROTOCOL.json").read_text())
    result = dict(schema="paper2-dev-pulse-input-qa-result-v1",
                  status="INCOMPLETE", analysis_complete=False, input_qualified=False,
                  synthetic=False, scope=protocol["scope"], GPU=0, new_fits=0,
                  new_model_calls=0, new_env_steps=0, training_updates=0, JPFS_new_bytes=0,
                  new_pulse_labels=0, native_runtime_qualified=False,
                  scientific_effect_or_physical_motion_claim=False)

    def check():
        need(time.time() < deadline, "absolute_science_deadline")

    try:
        need(sys.platform.startswith("linux"), "actual_arrays_only_cloud_Linux")
        check()
        manifest_path = HERE / "SOURCE-MANIFEST.json"
        source_id = fid(manifest_path)
        need(source_id["sha256"] == args.source_manifest_sha256, "bound_source_manifest")
        manifest = json.loads(manifest_path.read_text())
        need(manifest["schema"] == "paper2-dev-pulse-input-qa-source-v1", "source_schema")
        for row in manifest["files"]:
            check()
            path = HERE / row["path"]
            need(path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(HERE),
                 "regular_bound_source_member")
            need(fid(path) == {k: row[k] for k in ("bytes", "sha256")}, "source_member_SHA")
        need(fid(HERE / "INPUTS.json") == fid(HERE / "input/INPUTS.json"), "same_input_manifest")
        inputs = json.loads((HERE / "INPUTS.json").read_text())
        result.update(source_manifest=source_id, inputs_manifest=fid(HERE / "INPUTS.json"))
        work.mkdir(exist_ok=False)

        phase = "actual_existing_input_qualification"
        stamp = time.monotonic()
        A = module("_actual_DEV_input_adapter", HERE / "input/adapter.py")
        loaded = A.load(Path(args.stage), inputs, check=check, synthetic=False)
        proof, snapshots = loaded["proof"], loaded["snapshots"]
        need(proof["input_source_qualified"] is True and proof["synthetic"] is False and
             len(snapshots) == 8 and len(proof["input_objects"]) == 216,
             "actual_complete_216_object_eight_snapshot_qualification")
        need(proof["native_runtime_qualified"] is False and proof["new_pulse_labels"] == 0,
             "input_audit_does_not_claim_native_execution")
        write(out / "input-qualification.json", proof)
        result.update(input_qualified=True, input_objects_verified=216,
                      input_body_bytes=inputs["input_bytes"])
        timing[phase] = time.monotonic() - stamp

        phase = "actual_compact_input_bindings"
        stamp = time.monotonic()
        S = module("_actual_DEV_compact_storage", HERE / "storage/compact.py")
        plan = json.loads((HERE / "PLAN-CANDIDATE.json").read_text())
        ledger = S.CompactLedger(work / "bindings", plan, deadline)
        refs = []
        for saved in snapshots:
            check()
            source_before = A.identity((saved.raw_observations, saved.noise, saved.modified_v))
            ref = ledger.persist_input(saved)
            check_input_record(ledger.root / ref["path"], saved, S.IDENTITY.array_id, S.CODEC.flatten)
            need(source_before == A.identity((saved.raw_observations, saved.noise, saved.modified_v)),
                 "actual_input_storage_no_mutation")
            refs.append({"root_id": saved.root_id, "query_index": saved.query_index, **ref})
        ledger.finish({"status": "ACTUAL_INPUT_BINDINGS_ONLY_NO_CALLS",
                       "model_calls": 0, "native_runtime_qualified": False})
        need(ledger.next == 0 and not ledger.charged and not ledger.calls,
             "zero_native_calls_during_input_storage_audit")
        with zipfile.ZipFile(out / "verified-input-bindings.zip", "x", zipfile.ZIP_STORED) as archive:
            for path in sorted(ledger.root.iterdir()):
                archive.write(path, path.name)
        write(out / "input-storage-roundtrip.json", {
            "schema": "rss-actual-eight-input-storage-qualification-v1",
            "synthetic": False, "status": "PASS", "records": refs,
            "maximum_record_bytes": max(r["bytes"] for r in refs),
            "native_calls": 0, "native_output_storage_qualified": False,
            "binding_archive": fid(out / "verified-input-bindings.zip")})
        timing[phase] = time.monotonic() - stamp

        phase = "synthetic_maximum_storage_tests"
        stamp = time.monotonic()
        check()
        passed = storage_software_qa(out, deadline)
        need(passed, "all11_synthetic_maximum_storage_tests")
        timing[phase] = time.monotonic() - stamp
        need(not any(k == "jax" or k.startswith(("openpi", "libero", "robosuite"))
                     for k in sys.modules), "no_model_or_environment_modules_in_actual_input_audit")
        check()
        result["outputs"] = [{"path": path.name, **fid(path)} for path in sorted(out.iterdir())]
        need(all(r["bytes"] <= 8 * 1024**2 for r in result["outputs"]), "bounded_small_QA_outputs")
        result.update(status="INPUT_AND_STORAGE_QA_COMPLETE_NATIVE_NOT_RUN",
                      analysis_complete=True, snapshots=8,
                      prefix_token_counts=[int(s.prefix_mask.shape[1]) for s in snapshots],
                      synthetic_storage_tests=11, all_native_calls=0)
    except BaseException as error:
        result.update(status="INCOMPLETE", analysis_complete=False)
        code = str(error)
        result["failure"] = {
            "phase": phase, "type": type(error).__name__,
            "code": code if len(code) <= 160 and all(c.isalnum() or c in "_:.-" for c in code) else None,
            "frames": [{"file": Path(f.filename).name, "function": f.name, "line": f.lineno}
                       for f in traceback.extract_tb(error.__traceback__)[-12:]]}
    result.update(elapsed_seconds=time.monotonic() - start, phase_seconds=timing)
    write(out / "rss-cpu-result.json", result)
    print(json.dumps({"status": result["status"], "analysis_complete": result["analysis_complete"],
                      "failure": result.get("failure")}), flush=True)
    return 0 if result["analysis_complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
