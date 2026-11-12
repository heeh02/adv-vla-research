"""Run the exact prepared scientific increment in the admitted CPU image."""

import argparse
import json
from pathlib import Path
import sys


def execute(root, science, archive, hard):
    root = Path(root)
    package = root / "package"
    request = json.loads((package / "metadata/REQUEST-v1.json").read_bytes())
    admission = json.loads((root / "RUNTIME-ADMISSION-v1.json").read_bytes())
    if request["run_id"] != admission["run_id"] or request["input_spec"] != admission["input_spec"]:
        raise ValueError("admission_request_source_binding")
    if (science, archive, hard) != tuple(
        admission[k] for k in ("science_deadline_epoch", "archive_target_epoch", "deadline_epoch")
    ):
        raise ValueError("original_admitted_deadlines")
    if admission["resources"] != {"CPU": 2, "memory_GiB": 8, "GPU": 0}:
        raise ValueError("admitted_CPU2_memory8_GPU0")
    sys.path.insert(0, str(package / "worker/science"))
    import cloud_adapter  # noqa: PLC0415

    return cloud_adapter.execute(
        package / "worker/science/prepared-v1",
        package / "worker/deps",
        package / "worker/borrowed/worker.py",
        root / "staged-inputs",
        root / "results",
        root / "RUNTIME-ADMISSION-v1.json",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute-cloud", action="store_true")
    parser.add_argument("--root", required=True)
    for name in ("science-deadline-epoch", "archive-target-epoch", "deadline-epoch"):
        parser.add_argument("--" + name, type=float, required=True)
    args = parser.parse_args()
    if not args.execute_cloud or not sys.platform.startswith("linux"):
        raise ValueError("explicit_cloud_CPU_only")
    result = execute(args.root, args.science_deadline_epoch, args.archive_target_epoch, args.deadline_epoch)
    return 0 if result["status"] == "science_and_delta_whole_GET_SHA_published" else 2


if __name__ == "__main__":
    raise SystemExit(main())
