"""Cloud-only existing-data input qualification; never runs a simulator."""
import argparse
import json
import math
from pathlib import Path
import sys
import time
import traceback
import codec as C
import qualify as Q

HERE = Path(__file__).resolve().parent
RESULT = "q11-replay-input-qa-result.json"
OUTPUTS = (RESULT, "PARENT-PROOF.json", "NATIVE-JOIN.json", "QA.json", "PROTOCOL.json")
SOURCE_FILES = (
    "adapter.py", "qualify.py", "codec.py", "binding.py", "INPUTS.json", "PROTOCOL.json",
    "CODEC-PINS.json", "RETAINED-PINS.json", "ORIGINAL-ROOT.json",
    "pinned/retained_readout.py", "pinned/original_source_reader.py",
    "pinned/canonical_step_capture.py", "pinned/retained_replication.py",
    "test_qualify.py", "freeze.py", "README.md",
)


def fid(raw):
    return {"bytes": len(raw), "sha256": C.sha(raw)}


def execute(stage, inputs, check, *, synthetic=False):
    C.need(inputs["root_id"] == Q.ROOT and inputs["query_index"] == 11 and
           inputs["schema"] == "paper2-q11-replay-input-qa-inputs-v1",
           "fixed_q11_input_scope")
    refs = {r["path"]: r for r in inputs["input_objects"]}
    C.need(len(refs) == len(inputs["input_objects"]) == inputs["input_object_count"] == 228 and
           len({r["uri"] for r in refs.values()}) == 228, "unique_228_inputs")
    C.need(sum(r["bytes"] for r in refs.values()) == inputs["input_body_bytes"] <= 16 * 1024**2,
           "exact_bounded_input_total")
    C.need(not set(inputs["pinned_runtime_paths"]) & set(refs), "runtime_archives_not_downloaded")
    C.need(stage.is_dir() and not stage.is_symlink() and
           {p.name for p in stage.iterdir()} == set(refs), "exact_stage_roster")
    for name, ref in refs.items():
        check()
        path = stage / name
        C.need(path.is_file() and not path.is_symlink() and path.parent == stage and
               0 < path.stat().st_size == ref["bytes"] <= 4 * 1024**2,
               "bounded_input_file")
        C.need(fid(path.read_bytes()) == {k: ref[k] for k in ("bytes", "sha256")},
               "input_whole_SHA")
    original = C.json_read((HERE / "ORIGINAL-ROOT.json").read_bytes())
    C.need(original["root"]["root_trajectory_slot_id"] == Q.ROOT and
           original["frame_record"]["official_case"]["env_seed"] ==
           inputs["source_parent_task"]["env_seed"], "exact_original_root_configuration")
    source, archive_checks = Q.decode_parent(stage, inputs, check, synthetic=synthetic)
    parent = Q.parent_checks(source, inputs, check)
    joined = Q.native_join(stage, inputs, source, check, synthetic=synthetic)
    counts = {"archives": len(archive_checks), "parent_chunks": 12,
              "issued_steps": len(parent["steps"]), "outcomes": len(parent["steps"]),
              "protected_queries": len(parent["protected_queries"]),
              "anchors": len(parent["anchors"]),
              "cached_native_calls": len(joined["cached_commands"])}
    C.need(counts == Q.COUNTS, "complete_input_qualification_support")
    qa = {"schema": "paper2-q11-replay-input-QA-v1", "pass": True,
          "counts": counts, "input_objects_verified": len(refs),
          "input_body_bytes": inputs["input_body_bytes"],
          "parent_archives": archive_checks, "source_prefix_continuity": True,
          "q12_decoded_role_verified": True, "q12_commands": 0,
          "cached_N_is_not_used_for_source_reconstruction": True,
          "initialization_metadata_and_payload_verified": True,
          "new_environment_qualification": False, "new_model_calls": 0,
          "new_env_steps": 0, "new_fits": 0, "GPU": 0,
          "original_runtime_proof_body_keys": {
              k: sorted(C.json_read((stage / n).read_bytes()).keys())
              for k, n in inputs["original_runtime_proof_paths"].items()},
          "runtime_archives_not_restored": True, "root_configuration": fid(
              (HERE / "ORIGINAL-ROOT.json").read_bytes()),
          "scope": "Recorded-data eligibility only; no physics/render/controller execution."}
    return {"PARENT-PROOF.json": parent, "NATIVE-JOIN.json": joined, "QA.json": qa}


def run(stage, out, manifest_sha, deadline):
    C.need(sys.platform.startswith("linux"), "real_arrays_cloud_Linux_only")
    C.need(math.isfinite(deadline), "finite_deadline")
    stage, out = Path(stage), Path(out)
    C.need(not out.exists() and not out.is_symlink(), "new_output_directory")
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    result = {"schema": "paper2-q11-replay-input-qa-result-v1", "status": "INCOMPLETE",
              "analysis_complete": False, "input_qualified": False, "synthetic": False,
              "GPU_count": 0, "new_model_calls": 0, "new_env_steps": 0, "new_fits": 0,
              "environment_constructions": 0, "runtime_imports": 0,
              "training_updates": 0, "JPFS_new_bytes": 0, "physical_response_measured": False,
              "reconstructed_environment_qualified": False}
    def check():
        C.need(time.time() < deadline, "science_absolute_deadline")
    used = 0
    def write(name, raw):
        nonlocal used
        used += len(raw)
        C.need(name in OUTPUTS and len(raw) <= 2 * 1024**2 and used <= 8 * 1024**2,
               "small_qualification_outputs")
        with (out / name).open("xb") as stream:
            stream.write(raw)
        return {"path": name, **fid(raw)}
    phase = "source"
    try:
        check()
        raw = (HERE / "SOURCE-MANIFEST.json").read_bytes()
        C.need(C.sha(raw) == manifest_sha, "exact_source_manifest")
        manifest = C.json_read(raw)
        C.need(manifest["schema"] == "paper2-q11-replay-input-qa-source-v1" and
               len(manifest["files"]) == len(SOURCE_FILES) and
               {r["path"] for r in manifest["files"]} == set(SOURCE_FILES), "closed_source_roster")
        for row in manifest["files"]:
            path = HERE / row["path"]
            C.need(path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(HERE) and
                   fid(path.read_bytes()) == {k: row[k] for k in ("bytes", "sha256")},
                   "source_member_SHA")
        input_raw = (HERE / "INPUTS.json").read_bytes()
        inputs = C.json_read(input_raw)
        result.update(source_manifest=fid(raw), input_manifest=fid(input_raw))
        phase = "recorded_input_qualification"
        docs = execute(stage, inputs, check)
        outputs = [write(name, Q.encoded(doc)) for name, doc in docs.items()]
        outputs.append(write("PROTOCOL.json", (HERE / "PROTOCOL.json").read_bytes()))
        result.update(status="Q11_REPLAY_INPUT_QA_COMPLETE", analysis_complete=True,
                      input_qualified=True, input_objects_verified=inputs["input_object_count"],
                      input_body_bytes=inputs["input_body_bytes"], counts=Q.COUNTS,
                      outputs=outputs, output_names=list(OUTPUTS))
    except Exception as error:
        code = str(error)
        result.update(status="FAILED_PRESERVED", failure={
            "phase": phase, "type": type(error).__name__,
            "code": code if len(code) <= 180 and all(c.isalnum() or c == "_" for c in code) else None,
            "frames": [{"file": Path(f.filename).name, "line": f.lineno, "function": f.name}
                       for f in traceback.extract_tb(error.__traceback__)[-8:]]})
    result["elapsed_seconds"] = time.monotonic() - started
    write(RESULT, Q.encoded(result))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--source-manifest-sha256", required=True)
    parser.add_argument("--deadline-epoch", type=float, required=True)
    args = parser.parse_args()
    result = run(args.stage, args.out, args.source_manifest_sha256, args.deadline_epoch)
    print(json.dumps({k: result[k] for k in ("status", "analysis_complete")}))
    raise SystemExit(0 if result["analysis_complete"] else 2)
