"""One-package adaptation of the previously accepted CPU0 root host.

The old host's Create CAS/unknown-reconcile/monitor/account implementation is
borrowed. Only exact per-package identity, CPU8GiB, deadlines and readback
schema change. Importing this module cannot create jobs.
"""

import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from urllib.parse import urlsplit

import build
import contract as C  # noqa: N812 - accepted host names its bound contract C

HERE = Path(__file__).resolve().parent
CLOSEOUT = HERE.parents[1]
BASE = CLOSEOUT / "comparison-cpu-execution-v1/repair-v1/cpu-execution-v1/root.py"
CPU0 = CLOSEOUT / "comparison-v1/incremental-floor0-u1136-n474-v1/cpu-execution-v1/cpu0_admission.py"
BASE_SHA = "7b36fd77f82ea0a9e11566bef26406476313188b9e51cb19370ebe21d1b10b3b"


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


def transform(root, name, replacements):
    source = BASE.read_bytes()
    C.need(hashlib.sha256(source).hexdigest() == BASE_SHA, "accepted_root_host_source_SHA")
    node = next(item for item in ast.parse(source).body if isinstance(item, ast.FunctionDef) and item.name == name)
    changed = {repr(key): 0 for key in replacements}

    class Exact(ast.NodeTransformer):
        def visit_Constant(self, item):
            for old, new in replacements.items():
                if type(item.value) is type(old) and item.value == old:
                    changed[repr(old)] += 1
                    return ast.copy_location(ast.Constant(value=new), item)
            return item

    node = Exact().visit(node)
    C.need(all(count > 0 for count in changed.values()), "expected_exact_host_constant_sites:" + name)
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), str(__file__) + ":" + name, "exec"
        ),
        root.__dict__,
    )
    return changed


def small_readback(root, backend, state, record, q):
    prior = root.logged(state, "READBACK-v1.json")
    if prior:
        return prior
    get = root.bounded_get_function()
    cap = 8 * 1024**2
    meter = {"consumed": max([C.read(p)["consumed"] for p in Path(state).glob("READBACK-BYTES-*.json")], default=0)}
    files = {}
    env = {}
    try:
        env = backend.sts(
            [
                {
                    "service": "bce:bos",
                    "region": "bj",
                    "effect": "Allow",
                    "resource": [C.OUTPUT.removeprefix("bos://") + name for name in C.SMALL],
                    "permission": ["READ"],
                }
            ],
            900,
        )
        client = backend.old._bos_client(env)  # noqa: SLF001 - accepted host readback API

        def fetch(name, expected=None):
            C.need(name in C.SMALL, "small_metadata_only")
            uri = C.OUTPUT + name
            split = urlsplit(uri)
            raw = get(client, split.netloc, split.path.lstrip("/"), meter=meter, cap=cap, expect=expected)
            C.need(len(raw) <= 4 * 1024**2, "individual_small_readback4MiB")
            destination = Path(state) / "readback" / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                C.need(destination.read_bytes() == raw, "immutable_readback")
            else:
                destination.write_bytes(raw)
            files[name] = {"uri": uri, **C.identity(raw)}
            return json.loads(raw)

        delivery = fetch("DELIVERY-v1.json")
        C.need(
            delivery["schema"] == "paper3-added-endpoint-delivery-v1"
            and delivery["status"] == "science_and_delta_whole_GET_SHA_published"
            and delivery["run_id"] == q["scientific_run_id"]
            and delivery["model_id"] == q["model_id"]
            and delivery["new_count"] == q["new_count"]
            and delivery["old_baseline_or3626_republished"] is False
            and delivery["new_OUTCOME_GETs"] == q["new_count"],
            "actual_new_endpoint_delivery_scope",
        )
        refs = {item["relative"]: item for item in delivery["snapshot_publication"]}
        expected = {
            "NEW-PAIR-TRACE-v1.jsonl.gz",
            "NORMALIZED-NEW-v1.jsonl.gz",
            "INPUT-RECEIPT-v1.json",
            "RESULT-v1.json",
        }
        C.need(
            set(refs) == expected
            and len(refs) == 4
            and all(
                item["uri"] == C.OUTPUT + name and item["whole_GET_SHA_verified"] is True for name, item in refs.items()
            ),
            "four_only_new_whole_SHA_publications",
        )
        result = fetch("RESULT-v1.json", refs["RESULT-v1.json"])
        receipt = fetch("INPUT-RECEIPT-v1.json", refs["INPUT-RECEIPT-v1.json"])
        C.need(
            result["schema"] == "paper3-added-endpoint-pairs-v1"
            and result["new_count"] == q["new_count"]
            and result["new_trace_rows"] == 2 * q["new_count"]
            and result["new_OUTCOME_GETs"] == q["new_count"]
            and result["old_OUTCOME_GETs"] == 0
            and result["full_population_evaluation_complete"] is False
            and result["CI"] is None
            and result["GPU"] == result["model_calls"] == 0,
            "new_cohort_only_science_result",
        )
        C.need(
            result["input_spec"] == q["input_spec"]
            and receipt["input_spec"] == q["input_spec"]
            and receipt["verified_objects"] == q["runtime_exact_URI_count"]
            and receipt["verified_bytes"] == q["runtime_exact_input_bytes"]
            and receipt["new_OUTCOME_GETs"] == q["new_count"]
            and receipt["old_OUTCOME_GETs"] == 0,
            "actual_exact_readset_proof",
        )
        return root.journal(
            state,
            "READBACK-v1.json",
            {
                "schema": "paper3-data-driven-CPU0-readback-v1",
                "run_id": C.RUN,
                "snapshot_cloud_durable": True,
                "full_population_evaluation_complete": False,
                "new_count": q["new_count"],
                "files": files,
                "large_new_trace_and_delta_remain_BOS": True,
                "provider_release_separately_required": True,
            },
        )
    finally:
        root.event(state, "READBACK-BYTES", consumed=meter["consumed"], cap=cap, failed_reads_not_refunded=True)
        env.clear()


def prepare(package):
    package = Path(package).resolve()
    descriptor, q = build.verify(package / "prepared-v1", package)
    C.configure(q)
    # The accepted host imports its local dependencies by module name. Install
    # only this private module instance; restore the caller's module registry.
    previous = {name: sys.modules.get(name) for name in ("contract", "build")}
    try:
        sys.modules["contract"] = C
        sys.modules["build"] = build
        root = module("_paper3_data_increment_accepted_host", BASE)
    finally:
        for name, value in previous.items():
            if value is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value
    root.C = C

    class BuildShim:
        @staticmethod
        def verify(path):
            return build.verify(path, package)

    root.build = BuildShim
    root.HERE = package

    def verify_sources():
        C.need(hashlib.sha256(BASE.read_bytes()).hexdigest() == BASE_SHA, "accepted_root_source_unchanged")
        build.verify(package / "prepared-v1", package)
        C.exact(
            package / "bootstrap_runtime.py",
            {"bytes": len(build.bootstrap_source(q)), "sha256": hashlib.sha256(build.bootstrap_source(q)).hexdigest()},
        )

    root.verify_sources = verify_sources
    changes = {}
    for name, replacements in (
        ("explicit_create", {4: 8}),
        ("cpu_capacity", {4: 8}),
        ("exact_job", {4: 8, "actual_GPU0_CPU2_memory4": "actual_GPU0_CPU2_memory8"}),
        ("create_once", {4: 8, 1200: 3600, 600: 3000, 900: 3480, 3600: 7200}),
        ("observe", {"hard20min_deadline": "hard1h_deadline"}),
        ("main", {"statistics_recovery_complete": "snapshot_cloud_durable"}),
    ):
        changes[name] = transform(root, name, replacements)

    def review_gate(review, d, request):
        C.need(
            review.get("root_reviewed") is True
            and review.get("execution_authorized") is True
            and review.get("run_id") == C.RUN,
            "root_reviewed_exact_CPU_run",
        )
        C.need(
            review.get("source_package_sha256") == d["source_package"]["sha256"]
            and review.get("input_spec") == request["input_spec"]
            and review.get("input_credential_resources") == request["input_credential_resources"]
            and review.get("output_prefix") == C.OUTPUT
            and review.get("resources") == C.RESOURCES
            and review.get("phases") == C.PHASES
            and review.get("limits") == C.LIMITS
            and review.get("unknown_Create_no_retry") is True,
            "reviewed_source_readset_write_scope_deadlines",
        )

    root.review_gate = review_gate
    root.readback = lambda backend, state, record: small_readback(root, backend, state, record, q)
    cpu0 = module("_paper3_accepted_CPU0_admission", CPU0)
    targets = json.loads((CPU0.parent / "TARGETS-v1.json").read_bytes())
    target = {"run_id": C.RUN, "resources": C.RESOURCES, "source_package": descriptor["source_package"]}
    cpu0.install(root, target, targets)
    root.data_increment_constant_changes = changes
    return root, descriptor, q
