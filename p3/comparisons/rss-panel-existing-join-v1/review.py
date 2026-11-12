"""Write one concrete offline root review of an exact CPU0 package."""

import argparse
import json
from pathlib import Path

import contract
import root_adapter


def review(package, output, *, root_reviewed=False):
    package = Path(package).resolve()
    root, descriptor, q = root_adapter.prepare(package)
    value = {
        "schema": "paper3-data-driven-CPU0-root-review-v1",
        "root_reviewed": root_reviewed,
        "execution_authorized": root_reviewed,
        "run_id": q["run_id"],
        "scientific_run_id": q["scientific_run_id"],
        "model_id": q["model_id"],
        "new_count": q["new_count"],
        "resources": q["resources"],
        "phases": q["phases"],
        "limits": q["limits"],
        "source_package_sha256": descriptor["source_package"]["sha256"],
        "input_spec": q["input_spec"],
        "input_credential_resources": q["input_credential_resources"],
        "credential_scope_wider_than_runtime_readset": True,
        "runtime_exact_URI_count": q["runtime_exact_URI_count"],
        "runtime_exact_input_bytes": q["runtime_exact_input_bytes"],
        "output_prefix": q["output_prefix"],
        "output_credential_resource": q["output_credential_resource"],
        "constant_changes": root.data_increment_constant_changes,
        "old3626_and_baseline_republished": False,
        "unknown_Create_no_retry": True,
        "no_cloud_calls_in_review": True,
        "full_population_evaluation_complete": False,
    }
    if root_reviewed:
        root.review_gate(value, descriptor, q)
    path = Path(output)
    with path.open("xb") as stream:
        stream.write(contract.encoded(value))
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--root-reviewed", action="store_true")
    args = parser.parse_args()
    value = review(args.package, args.out, root_reviewed=args.root_reviewed)
    print(
        json.dumps(
            {
                "status": "root_reviewed" if value["root_reviewed"] else "candidate_only",
                "run_id": value["run_id"],
                "review": args.out,
                "cloud_calls": 0,
            }
        )
    )


if __name__ == "__main__":
    main()
