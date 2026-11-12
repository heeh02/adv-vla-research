"""Fail-closed source/static checks for the isolated five-arm candidate."""
from __future__ import annotations

import ast
import hashlib
import json
import py_compile
from pathlib import Path


HERE = Path(__file__).resolve().parent
ARMS = ["R0", "T0", "R1", "T1", "R0prime"]


def identity(path: Path) -> dict:
    raw = path.read_bytes()
    return {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def main() -> None:
    checks = {}
    worker = HERE / "five_arm_collect.py"
    tree = ast.parse(worker.read_text())
    arms = next(
        node.value
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "ARMS" for target in node.targets)
    )
    checks["fixed_arm_order"] = ast.literal_eval(arms) == tuple(ARMS)
    manifest = json.loads((HERE / "VARIANT-MANIFEST-v1.json").read_text())
    checks["variant_manifest_count_268"] = manifest["row_count"] == len(manifest["rows"]) == 268
    checks["variant_manifest_hash_bindings"] = all(
        all(row[key].get("bytes", 0) > 0 and len(row[key].get("sha256", "")) == 64
            for key in ("variant_BDDL", "clean_BDDL", "official_init_library"))
        for row in manifest["rows"]
    )
    binding = json.loads((HERE / "FRAME-BINDING-v1.json").read_text())
    checks["frame_binding"] = binding["frame_sha256"] == manifest["source_frame"]["sha256"]
    shard = json.loads((HERE / "SHARD-v1.json").read_text())
    checks["static_shard_268"] = shard["case_count"] == len(shard["case_rows"]) == 268
    checks["shard_arm_order"] = shard["arm_order"] == ARMS
    cloud_calls = []
    for path in HERE.rglob("*.py"):
        if path.name == "static_gate.py":
            continue
        try:
            syntax = ast.parse(path.read_text())
        except SyntaxError:
            continue
        for node in ast.walk(syntax):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and "createjob" in node.func.id.lower():
                cloud_calls.append(str(path.relative_to(HERE)))
    checks["no_cloud_submission"] = not cloud_calls
    py_files = sorted(path for path in HERE.rglob("*.py") if "__pycache__" not in path.parts)
    failures = []
    for path in py_files:
        try:
            py_compile.compile(str(path), doraise=True)
        except Exception as error:  # pragma: no cover - report path is the test
            failures.append({"path": str(path.relative_to(HERE)), "type": type(error).__name__})
    checks["py_compile"] = not failures
    report = {"schema": "plus-five-arm-static-gate-v1", "status": "PASS" if all(checks.values()) else "FAIL",
              "checks": checks, "compile_failures": failures, "cloud_call_sites": cloud_calls,
              "runtime_sources": {"v4": "runtime-v4", "v6": "runtime-v6"},
              "cloud_CreateJob_called": False}
    (HERE / "STATIC-GATE-v1.json").write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
    if report["status"] != "PASS":
        raise SystemExit(json.dumps(report, sort_keys=True))
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
