"""Root-only admitted CPU0 launch/reconcile using the accepted host."""

import argparse
from pathlib import Path
import sys

import contract
import root_adapter


def launch(package, action, *, review=None, registry=None, root_exclusive=False, no_other_route=False):
    package = Path(package).resolve()
    root, descriptor, q = root_adapter.prepare(package)
    if action not in ("dry", "execute", "reconcile", "monitor", "readback"):
        raise ValueError("one_CPU0_root_action")
    scope = root_adapter.module(
        "_paper3_thread_scope", root_adapter.CLOSEOUT / "resource-thread-cap-v1/thread_scope.py"
    )
    scope.authority()
    canonical = scope.ACCOUNT.resolve()
    selected = Path(registry).resolve() if registry else canonical
    contract.need(selected == canonical, "exact_shared_ACCOUNT_not_case_ledger")
    if action != "dry":
        contract.need(root_exclusive and no_other_route, "root_exclusive_science_CPU_route")
        if action == "execute":
            contract.need(review is not None, "root_review_required_before_Create")
            root.review_gate(contract.read(review), descriptor, q)
    args = [
        str(package / "root.py"),
        action,
        "--prepared",
        str(package / "prepared-v1"),
        "--state",
        str(package / "run-v1"),
        "--registry",
        str(selected),
        "--external-inventory",
        str(
            root_adapter.CLOSEOUT
            / "comparison-v1/incremental-floor0-u1136-n474-v1/cpu-execution-v1/EXTERNAL-NOT-APPLICABLE-v1.json"
        ),
    ]
    if review:
        args.extend(("--review", str(Path(review).resolve())))
    original = sys.argv
    try:
        sys.argv = args
        return root.main()
    finally:
        sys.argv = original


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", required=True)
    parser.add_argument("--action", choices=("dry", "execute", "reconcile", "monitor", "readback"), required=True)
    parser.add_argument("--review")
    parser.add_argument("--registry")
    parser.add_argument("--root-exclusive-T30", action="store_true")
    parser.add_argument("--no-other-owned-route", action="store_true")
    args = parser.parse_args()
    return launch(
        args.package,
        args.action,
        review=args.review,
        registry=args.registry,
        root_exclusive=args.root_exclusive_T30,
        no_other_route=args.no_other_owned_route,
    )


if __name__ == "__main__":
    raise SystemExit(main())
