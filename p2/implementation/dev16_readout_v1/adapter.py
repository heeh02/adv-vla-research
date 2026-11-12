"""Bounded Linux-only three-JSON DEV16 consumer; root owns cloud transport."""
import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import sys
import time
import traceback

import analysis as A

HERE = Path(__file__).resolve().parent
RESULT = "dev16-readout-result.json"
OUTPUTS = (RESULT, "AXIS-WINDOW-SUPPORT.csv", "SUMMARY.json",
           "QA.json", "PROTOCOL.json", "SNAPSHOT-SUPPORT.json")
SOURCE_FILES = ("adapter.py", "analysis.py", "INPUTS.json", "PROTOCOL.json",
                "synthetic_fixture.py", "test_readout.py", "README.md")


def ident(raw):
    return {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def encoded(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def strict_json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            A.need(key not in result, "duplicate_JSON_key")
            result[key] = value
        return result
    def constant(_):
        raise ValueError("nonfinite_JSON")
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=constant)


def write(out, name, raw):
    A.need(name in OUTPUTS and len(raw) <= 2 * 1024**2, "declared_output_under2MiB")
    with (out / name).open("xb") as f:
        f.write(raw)
    return {"path": name, **ident(raw)}


def csv_bytes(rows):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode()


def run(stage, out, source_manifest_sha256, deadline_epoch):
    # Before any producer body, source metadata or output creation.
    A.need(sys.platform.startswith("linux"), "real_DEV16_analysis_cloud_Linux_only")
    A.need(type(deadline_epoch) in (int, float) and math.isfinite(deadline_epoch),
           "finite_absolute_deadline")
    started = time.monotonic()
    out = Path(out)
    A.need(not out.exists() and not out.is_symlink(), "child_owns_absent_output_directory")
    out.mkdir(parents=True, exist_ok=False)
    result = {"schema": "paper2-dev16-readout-result-v1", "status": "INCOMPLETE",
              "input_qualified": False, "analysis_complete": False, "synthetic": False,
              "new_model_calls": 0, "new_env_steps": 0, "training_updates": 0,
              "new_fits": 0, "JPFS_new_bytes": 0, "GPU_count": 0,
              "physical_meaningful_threshold": None, "new_blind_holdout": False}
    phase = "source"
    def check():
        A.need(time.time() < deadline_epoch, "absolute_science_deadline")
    try:
        check()
        raw = (HERE / "SOURCE-MANIFEST.json").read_bytes()
        source_ref = ident(raw)
        A.need(source_ref["sha256"] == source_manifest_sha256, "externally_bound_source_SHA")
        manifest = strict_json(raw)
        A.need(manifest["schema"] == "paper2-dev16-readout-source-v1", "source_schema")
        files = manifest["files"]
        A.need(len(files) == len(SOURCE_FILES) and
               {r["path"] for r in files} == set(SOURCE_FILES), "closed_source_file_roster")
        for row in files:
            path = HERE / row["path"]
            A.need(path.is_file() and not path.is_symlink() and
                   path.resolve().is_relative_to(HERE), "regular_owned_source")
            A.need(ident(path.read_bytes()) == {k: row[k] for k in ("bytes", "sha256")},
                   "source_member_SHA")
        input_raw = (HERE / "INPUTS.json").read_bytes()
        input_manifest = strict_json(input_raw)
        A.need(input_manifest["schema"] == "paper2-dev16-readout-inputs-v1" and
               input_manifest["producer_run"] == A.RUN and
               input_manifest["producer_job"] == "redacted-f594f0253751", "exact_native_producer")
        refs = input_manifest["input_objects"]
        names = {"TERMINAL.json", "RESULT.json", "RUNTIME-RESULT.json"}
        A.need(len(refs) == 3 and {r["path"] for r in refs} == names and
               sum(r["bytes"] for r in refs) == input_manifest["input_body_bytes"] == 329170,
               "exact_three_input_roster_bytes")
        refs = {r["path"]: r for r in refs}
        result.update(source_manifest=source_ref, input_manifest=ident(input_raw))
        phase = "producer_inputs"
        stage = Path(stage)
        A.need(stage.is_dir() and not stage.is_symlink() and
               {p.name for p in stage.iterdir()} == names, "exact_staged_input_roster")
        bodies = {}
        for name in sorted(names):
            check()
            path = stage / name
            A.need(path.is_file() and not path.is_symlink() and
                   path.stat().st_size == refs[name]["bytes"], "bounded_regular_staged_input")
            raw = path.read_bytes()
            A.need(ident(raw) == {k: refs[name][k] for k in ("bytes", "sha256")},
                   "input_body_SHA")
            bodies[name] = strict_json(raw)
        phase = "native_qualification_and_readout"
        rows, summary, qa, snapshots = A.analyze(
            bodies["TERMINAL.json"], bodies["RESULT.json"], bodies["RUNTIME-RESULT.json"],
            refs, synthetic=False, check=check)
        result.update(input_qualified=True, input_objects_verified=3, input_body_bytes=329170)
        qa.update(source_manifest=source_ref, input_manifest=ident(input_raw),
                  input_objects=list(refs.values()))
        phase = "output"
        outputs = []
        for name, value in (("SUMMARY.json", summary), ("QA.json", qa),
                            ("SNAPSHOT-SUPPORT.json", snapshots)):
            check()
            outputs.append(write(out, name, encoded(value)))
        outputs.append(write(out, "AXIS-WINDOW-SUPPORT.csv", csv_bytes(rows)))
        protocol_raw = (HERE / "PROTOCOL.json").read_bytes()
        strict_json(protocol_raw)
        outputs.append(write(out, "PROTOCOL.json", protocol_raw))
        A.need({p.name for p in out.iterdir()} == set(OUTPUTS) - {RESULT},
               "exact_five_outputs_before_main")
        check()
        result.update(status="DEV16_READOUT_COMPLETE", analysis_complete=True,
                      scalar_cells=560, qualified_snapshots=8, producer_qualified_pulse_labels=16,
                      outputs=outputs, output_names=list(OUTPUTS))
    except Exception as error:
        code = str(error)
        result.update(status="FAILED_PRESERVED", analysis_complete=False,
                      failure={"phase": phase, "type": type(error).__name__,
                               "code": code if len(code) <= 180 and
                               all(c.isalnum() or c in "_-" for c in code) else None,
                               "frames": [{"file": Path(f.filename).name, "function": f.name,
                                           "line": f.lineno}
                                          for f in traceback.extract_tb(error.__traceback__)[-8:]]})
    result["elapsed_seconds"] = time.monotonic() - started
    write(out, RESULT, encoded(result))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--source-manifest-sha256", required=True)
    parser.add_argument("--deadline-epoch", type=float, required=True)
    args = parser.parse_args()
    answer = run(args.stage, args.out, args.source_manifest_sha256, args.deadline_epoch)
    print(json.dumps({k: answer[k] for k in ("schema", "status", "analysis_complete")}))
    raise SystemExit(0 if answer["analysis_complete"] else 2)
