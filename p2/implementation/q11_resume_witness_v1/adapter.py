"""Cloud-only existing-record witness validation; no environment or model import."""
import argparse
import json
import math
from pathlib import Path
import sys
import time
import traceback
import physical_codec as C
import validate as V

HERE = Path(__file__).resolve().parent
RESULT = "q11-resume-witness-result.json"
OUTPUTS = (RESULT, "RESUME-WITNESS.json", "EVIDENCE-CHECKS.json", "PROTOCOL.json")


def encoded(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)+"\n").encode()


def run(stage, out, manifest_sha, deadline):
    C.need(sys.platform.startswith("linux"), "real_arrays_cloud_Linux_only")
    C.need(math.isfinite(deadline), "finite_deadline")
    stage, out = Path(stage), Path(out)
    C.need(not out.exists(), "new_witness_output")
    out.mkdir(parents=True)
    result = {"schema": "paper2-q11-resume-witness-result-v1", "status": "FAILED_PRESERVED",
        "analysis_complete": False, "input_qualified": False, "synthetic": False,
        "GPU_count": 0, "new_model_calls": 0, "new_env_steps": 0, "training_updates": 0,
        "JPFS_new_bytes": 0, "new_fits": 0, "runtime_imports": 0, "environment_constructions": 0,
        "cross_job_parity_established": False, "pulse_contrast_established": False,
        "new_environment_execution_authorized": False}
    started, output_bytes = time.monotonic(), 0
    def check():
        C.need(time.time() < deadline, "witness_absolute_deadline")
    def write(name, value):
        nonlocal output_bytes
        raw = encoded(value)
        output_bytes += len(raw)
        C.need(name in OUTPUTS and len(raw) <= 2*1024**2 and output_bytes <= 8*1024**2,
               "bounded_witness_outputs")
        with (out/name).open("xb") as f:
            f.write(raw)
    phase = "source"
    try:
        check()
        raw = (HERE/"SOURCE-MANIFEST.json").read_bytes()
        C.need(C.identity(raw)["sha256"] == manifest_sha, "exact_witness_source_manifest")
        manifest = C.json_bytes(raw)
        C.need(manifest["schema"] == "paper2-q11-resume-witness-source-v1" and
               len(manifest["files"]) == len({r["path"] for r in manifest["files"]}),
               "unique_witness_source_roster")
        for row in manifest["files"]:
            path = HERE/row["path"]
            C.need(path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(HERE) and
                   C.identity(path.read_bytes()) == {k: row[k] for k in ("bytes", "sha256")},
                   "witness_source_member_SHA")
        inputs_raw = (HERE/"INPUTS.json").read_bytes()
        inputs = C.json_bytes(inputs_raw)
        result.update(source_manifest=C.identity(raw), input_manifest=C.identity(inputs_raw))
        refs = {r["path"]: r for r in inputs["input_objects"]}
        C.need(len(refs) == inputs["input_object_count"] == 6 and
               inputs["input_body_bytes"] == sum(r["bytes"] for r in refs.values()) == 389052,
               "six_exact_archived_records")
        C.need({p.name for p in stage.iterdir()} == set(refs), "exact_witness_stage_roster")
        docs = {}
        phase = "whole_object_and_typed_payloads"
        decoded = None
        for name, ref in refs.items():
            check()
            path = stage/name
            C.need(path.is_file() and not path.is_symlink() and path.stat().st_size == ref["bytes"] and
                   C.identity(path.read_bytes()) == {k: ref[k] for k in ("bytes", "sha256")},
                   "exact_witness_input_SHA")
            if ref["logical_path"].endswith(".gz"):
                docs[ref["logical_path"]], decoded = C.load(path)
            else:
                docs[ref["logical_path"]] = C.json_bytes(path.read_bytes())
        phase = "provenance_and_observed_record_eligibility"
        V.provenance(docs, inputs, HERE)
        witness, checks = V.inspect_campaign(docs["science/PHYSICAL-CAMPAIGN.json.gz"],
            docs["science/NATIVE-SUMMARY.json"], docs["science/INPUT-QUALIFICATION.json"])
        witness.update(original_campaign_reference=next(r for r in refs.values()
            if r["logical_path"] == "science/PHYSICAL-CAMPAIGN.json.gz"),
            original_source_run_id=inputs["source_run_id"], original_source_manifest=inputs["source_manifest"],
            original_source_input_manifest=inputs["source_input_manifest"])
        checks["typed_payload_decode"] = decoded
        unexpected = [n for n in sys.modules if n.split(".")[0] in
                      ("mujoco", "robosuite", "libero", "openpi", "jax", "jaxlib", "torch")]
        result["runtime_imports"] = len(unexpected)
        C.need(not unexpected, "existing_records_only_no_target_runtime")
        write("RESUME-WITNESS.json", witness)
        write("EVIDENCE-CHECKS.json", checks)
        write("PROTOCOL.json", C.json_bytes((HERE/"PROTOCOL.json").read_bytes()))
        result.update(status="RETAINED_OBSERVATION_INPUTS_QUALIFIED", analysis_complete=True,
            input_qualified=True, input_objects_verified=6, input_body_bytes=389052, counts=V.COUNTS,
            typed_arrays_verified=decoded["arrays"], original_N_close_failure_preserved=True,
            source_and_N_rerun_allowed=False, remaining_roles=["N_repeat", "S", "R"],
            maximum_additional_environment_steps=210)
    except Exception as error:
        result["failure"] = {"phase": phase, "type": type(error).__name__, "code": str(error)[:180],
            "frames": [{"file": Path(f.filename).name, "line": f.lineno, "function": f.name}
                       for f in traceback.extract_tb(error.__traceback__)[-8:]]}
    result["elapsed_seconds"] = time.monotonic()-started
    write(RESULT, result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--source-manifest-sha256", required=True)
    parser.add_argument("--deadline-epoch", type=float, required=True)
    a = parser.parse_args()
    result = run(a.stage, a.out, a.source_manifest_sha256, a.deadline_epoch)
    print(json.dumps({"status": result["status"], "analysis_complete": result["analysis_complete"]}))
    raise SystemExit(0 if result["analysis_complete"] else 2)
