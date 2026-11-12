"""Prepare an exact, new-adoption-only Plus T30 comparison increment.

Local metadata only. No cloud reads, submission, adoption, registry or BOS write.
"""

import argparse
import base64
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
CLOSEOUT = HERE.parents[1]
BRIDGE = CLOSEOUT / "comparison-cloud-bridge-v1"
OLD = CLOSEOUT / "comparison-v1/incremental-floor0-u1136-n474-v1"
PARENT = OLD / "cpu-execution-v1/run-v1/readback"
BASELINE_NAMES = ("old_cache", "merged_delta")
INPUT_LIMIT = 400 * 1024**2
BOS_PREFIX = "redacted-24fc043a6cae"


def need(ok, reason):
    if not ok:
        raise ValueError(reason)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def identity(raw):
    return {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def read_ref(ref):
    p = Path(ref["path"])
    need(p.is_file() and p.stat().st_size <= 16 * 1024**2, "bounded_local_metadata")
    raw = p.read_bytes()
    need(identity(raw) == {k: ref[k] for k in ("bytes", "sha256")}, "local_metadata_exact_SHA")
    return json.loads(raw)


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def fixed_parent(parent_root):
    delivery_path = parent_root / "DELIVERY-v1.json"
    result_path = parent_root / "RESULT-v1.json"
    delivery = json.loads(delivery_path.read_bytes())
    result = json.loads(result_path.read_bytes())
    need(
        delivery["snapshot_cloud_durable"] is True
        and delivery["snapshot_comparison_complete"] is True
        and delivery["T30_total"] == result["T30_total"] == 3626
        and result["full_population_evaluation_complete"] is False,
        "actual3626_snapshot_only",
    )
    published = {r["relative"]: r for r in delivery["snapshot_publication"]}
    for name in ("RESULT-v1.json", "SINGLE-GROUPS-v1.csv.gz", "PAIRED-GROUPS-v1.csv.gz"):
        r = published[name]
        need(r["whole_GET_SHA_verified"] is True and r["uri"].startswith(BOS_PREFIX), "parent_whole_GET_SHA")
        if name == "RESULT-v1.json":
            need(
                identity(result_path.read_bytes()) == {k: r[k] for k in ("bytes", "sha256")}, "parent_RESULT_bytes_SHA"
            )
    cache = {r["relative"]: r for r in delivery["cache_publication"] or []}
    need(
        delivery["cache_status"] == "COMPLETE_VERIFIED_NEW474_DELTA"
        and cache["NORMALIZED-NEW474-v1.jsonl.gz"]["whole_GET_SHA_verified"] is True,
        "parent_new474_delta_is_actual_durable",
    )
    return {
        "delivery": {"path": str(delivery_path), **identity(delivery_path.read_bytes())},
        "result": {k: published["RESULT-v1.json"][k] for k in ("uri", "bytes", "sha256")},
        "tables": [
            {k: published[name][k] for k in ("uri", "bytes", "sha256")}
            for name in ("SINGLE-GROUPS-v1.csv.gz", "PAIRED-GROUPS-v1.csv.gz")
        ],
        "new474_delta": {k: cache["NORMALIZED-NEW474-v1.jsonl.gz"][k] for k in ("uri", "bytes", "sha256")},
        "T30_total": 3626,
        "not_a_full_population_result": True,
    }


def processed_refs(paths, model, run, current):
    """Only root-reviewed, durable prior increments may subtract endpoints."""
    seen = {}
    for path in paths:
        receipt = json.loads(Path(path).read_bytes())
        need(
            receipt.get("schema") == "paper3-added-endpoints-durable-receipt-v1"
            and receipt.get("root_reviewed") is True
            and receipt.get("whole_GET_SHA_verified") is True
            and receipt.get("source_result")
            and receipt.get("parent_T30_total") == 3626,
            "prior_delta_must_be_root_reviewed_whole_SHA_durable",
        )
        published = receipt.get("result_publication", {})
        need(
            published.get("uri", "").startswith(BOS_PREFIX)
            and published.get("whole_GET_SHA_verified") is True
            and {k: published.get(k) for k in ("bytes", "sha256")}
            == {k: receipt["source_result"][k] for k in ("bytes", "sha256")},
            "prior_result_publication_exact_whole_SHA",
        )
        result = read_ref(receipt["source_result"])
        need(
            result.get("schema") == "paper3-added-endpoint-pairs-v1"
            and result.get("model_id") == model
            and result.get("run_id") == run
            and result.get("parent_T30_total") == 3626,
            "prior_delta_source_identity",
        )
        for result_key, receipt_key in (
            ("new_trace", "trace_publication"),
            ("new_normalized_delta", "normalized_delta_publication"),
        ):
            publication = receipt.get(receipt_key, {})
            need(
                publication.get("uri", "").startswith(BOS_PREFIX)
                and publication.get("whole_GET_SHA_verified") is True
                and {k: publication.get(k) for k in ("bytes", "sha256")}
                == {k: result[result_key][k] for k in ("bytes", "sha256")},
                "prior_trace_and_delta_must_both_be_durable",
            )
        for row in result["new_endpoint_identity"]:
            key = row["ordinal"]
            need(
                key in current and current[key]["outcome_canonical_sha256"] == row["outcome_canonical_sha256"],
                "prior_delta_not_in_current_adoption_or_changed",
            )
            need(key not in seen, "duplicate_prior_durable_delta")
            seen[key] = row["outcome_canonical_sha256"]
    return seen


def prepare(closeout, adoptions, output, *, processed=()):
    c = Path(closeout).resolve()
    bridge_root = c / "comparison-cloud-bridge-v1"
    old, parent_root = (
        c / "comparison-v1/incremental-floor0-u1136-n474-v1",
        c / "comparison-v1/incremental-floor0-u1136-n474-v1/cpu-execution-v1/run-v1/readback",
    )
    out = Path(output).resolve()
    need(not out.exists(), "new_preparation_directory_only")
    need(bool(adoptions), "at_least_one_actual_root_adoption")
    reader_path = c / "comparison-v1/compare_endpoints.py"
    reader = module("_paper3_existing_adoption_reader", reader_path)
    bridge_path = bridge_root / "bridge.py"
    bridge = module("_paper3_frozen_bridge", bridge_path)
    spec, docs = bridge.load_controls(bridge_root / "inputs-2648-v1")
    parent = fixed_parent(parent_root)
    rows, sources, summary_paths = {}, [], {}
    for adoption_path in adoptions:
        path = Path(adoption_path).resolve()
        adoption = json.loads(path.read_bytes())
        need(
            adoption.get("applied") is True
            and adoption.get("old_results_overwritten") is False
            and adoption.get("scientific_attempt_changed") is False,
            "actual_root_CAS_adoption_only",
        )
        proof = reader.read_ref(adoption["proof"])
        request = reader.read_ref(proof["request"])
        registration = reader.read_ref(proof["registration"])
        launch = reader.read_ref(registration["launch_ref"])
        plan = reader.read_ref(launch["plan"])
        need(
            adoption["run_id"]
            == proof["run_id"]
            == request["run_id"]
            == registration["run_id"]
            == launch["run_id"]
            == plan["run_id"]
            and adoption["job_id"] == proof["job_id"] == registration["job_id"]
            and registration["request_sha256"]
            == launch["request_canonical_sha256"]
            == plan["request_canonical_sha256"]
            == reader.digest(request),
            "REGISTERED_launch_PLAN_request_adoption_exact_chain",
        )
        model = request["model"]
        need(
            model["id"] in spec["models"] and model == spec["models"][model["id"]],
            "model_identity_from_registered_request",
        )
        need(request["panel"] == "Plus" and request["attempt"] == 0, "fixed_Plus_attempt0")
        need(
            plan["model_id"] == model["id"]
            and plan["checkpoint"] == model["checkpoint"]
            and plan["ordinals"] == request["ordinals"],
            "PLAN_checkpoint_and_original_ordinal_binding",
        )
        proof_by_ordinal = {item["population_ordinal"]: item for item in proof["cases"]}
        for row in reader.collect_adoption(path):
            bridge.validate_endpoint(row, spec["models"], docs["historical_plan"]["population_cases"])
            need(
                row["run_id"] == request["run_id"]
                and row["attempt"] == 0
                and row["official_case"] == docs["historical_plan"]["population_cases"][row["ordinal"]],
                "actual_original_population_case",
            )
            previous = rows.get(row["ordinal"])
            need(
                previous is None
                or all(
                    previous[k] == row[k]
                    for k in (
                        "model_id",
                        "run_id",
                        "attempt",
                        "outcome_canonical_sha256",
                        "success",
                        "status",
                        "summary",
                        "outcome",
                    )
                ),
                "conflicting_duplicate_adoption",
            )
            rows[row["ordinal"]] = row
            summary_path = Path(proof_by_ordinal[row["ordinal"]]["case_summary"]["path"])
            old_path = summary_paths.get(row["ordinal"])
            need(
                old_path is None or identity(old_path.read_bytes()) == identity(summary_path.read_bytes()),
                "duplicate_adoption_summary_bytes_differ",
            )
            summary_paths[row["ordinal"]] = summary_path
        sources.append(
            {
                "adoption": {"path": str(path), **identity(path.read_bytes())},
                "proof": adoption["proof"],
                "request": proof["request"],
                "registration": proof["registration"],
                "launch": registration["launch_ref"],
                "plan": launch["plan"],
                "generation": adoption["generation"],
                "job_id": adoption["job_id"],
            }
        )
    models = {r["model_id"] for r in rows.values()}
    runs = {r["run_id"] for r in rows.values()}
    need(len(models) == len(runs) == 1, "one_model_run_per_increment")
    model, run = next(iter(models)), next(iter(runs))
    member = [r for r in rows.values() if r["model_id"] == model]
    need(all(r["ordinal"] in range(10030) for r in member), "original_population_ordinal")
    old_meta = json.loads(gzip.decompress((old / "prepared-v2/NEW-ENDPOINTS-v1.json.gz").read_bytes()))
    prior_floor1 = json.loads(gzip.decompress((old / "prepared-v2/OLD-FLOOR1-504-ENDPOINTS-v1.json.gz").read_bytes()))
    inherited = {(r["model_id"], r["ordinal"]) for r in docs["t30_endpoints"] + old_meta + prior_floor1}
    need(
        len(inherited) == 3626 and not any((model, o) in inherited for o in rows),
        "all_adopted_new_relative_to_actual3626",
    )
    removed = processed_refs(processed, model, run, rows)
    new = [rows[o] for o in sorted(rows) if o not in removed]
    need(new, "no_new_adopted_endpoint_after_durable_subtraction")
    contrast = [
        r
        for r in spec["comparison"]["contrasts"]
        if r["treatment"] == model and r["control"] in ("original", "T-s0-u20000-EMA", "T-s1-u20000-EMA")
    ]
    need(
        len(contrast) == 2
        and "original" in {r["control"] for r in contrast}
        and len({r["control"] for r in contrast if r["control"].startswith("T-s")}) == 1,
        "two_frozen_same_seed_T20_Original_contrasts",
    )
    baseline = [{k: spec["caches"][name][k] for k in ("uri", "bytes", "sha256")} for name in BASELINE_NAMES]
    refs = baseline + [{k: r["outcome"][k] for k in ("uri", "bytes", "sha256")} for r in new]
    need(
        len(refs) == len({r["uri"] for r in refs}) and all(r["uri"].startswith(BOS_PREFIX) for r in refs),
        "unique_Jinwen_input_objects",
    )
    total = sum(r["bytes"] for r in refs)
    need(
        total <= INPUT_LIMIT and all(type(r["bytes"]) is int and r["bytes"] > 0 for r in refs),
        "exact_input_budget_400MiB",
    )
    # Existing verified case summaries are small metadata, bundled rather than
    # issuing summary cloud GETs. Their URI/bytes/SHA remain on every endpoint.
    summary_bytes = {}
    for r in new:
        s = r["summary"]
        payload = summary_paths[r["ordinal"]].read_bytes()
        need(
            identity(payload) == {k: s[k] for k in ("bytes", "sha256")} and len(payload) <= 64 * 1024,
            "packaged_summary_exact_small",
        )
        summary_bytes[s["uri"]] = base64.b64encode(payload).decode()
    deps = (old / "prepared-v2/DEPENDENCIES-v1.json").read_bytes()
    old_worker = old / "worker.py"
    need(len(deps) < 32 * 1024, "frozen_dependencies_small")
    cohort_digest = hashlib.sha256(canonical([(r["ordinal"], r["outcome_canonical_sha256"]) for r in new])).hexdigest()[
        :16
    ]
    output_prefix = (
        BOS_PREFIX + "paper3-comparison-increment/" + run + "-new" + str(len(new)) + "-" + cohort_digest + "/"
    )
    prepared = {
        "schema": "paper3-data-driven-endpoint-increment-v1",
        "worker_source": {
            "bytes": (HERE / "worker.py").stat().st_size,
            "sha256": hashlib.sha256((HERE / "worker.py").read_bytes()).hexdigest(),
        },
        "cloud_adapter_source": {
            "bytes": (HERE / "cloud_adapter.py").stat().st_size,
            "sha256": hashlib.sha256((HERE / "cloud_adapter.py").read_bytes()).hexdigest(),
        },
        "model_id": model,
        "run_id": run,
        "output_prefix": output_prefix,
        "model_source": spec["models"][model],
        "case_list_identity": spec["case_list_identity"],
        "parent_snapshot": parent,
        "parent_T30_total": 3626,
        "adoption_sources": sources,
        "processed_sources": [{"path": str(Path(p).resolve()), **identity(Path(p).read_bytes())} for p in processed],
        "new_endpoint_identity": [
            {"ordinal": r["ordinal"], "outcome_canonical_sha256": r["outcome_canonical_sha256"]} for r in new
        ],
        "contrasts": contrast,
        "readset": {
            "objects": sorted(refs, key=lambda r: r["uri"]),
            "object_count": len(refs),
            "exact_total_bytes": total,
            "baseline_cache_bytes": sum(r["bytes"] for r in baseline),
            "new_outcome_GETs": len(new),
            "old_OUTCOME_GETs": 0,
            "summary_cloud_GETs": 0,
        },
        "frozen_dependencies": identity(deps),
        "borrowed_worker": {"path": str(old_worker), **identity(old_worker.read_bytes())},
        "adoption_total": len(rows),
        "already_processed": len(removed),
        "new_count": len(new),
        "scope": "new adopted Plus endpoints; same-seed T20/Original direct original gate; no population effect claim",
        "old_baselines_or3626_tables_reemitted": False,
        "GPU": 0,
        "model_calls": 0,
        "cloud_calls_in_prepare": 0,
    }
    out.mkdir(parents=True, exist_ok=False)
    (out / "DEPENDENCIES-v1.json").write_bytes(deps)
    (out / "NEW-ENDPOINTS-v1.json.gz").write_bytes(gzip.compress(canonical(new), mtime=0))
    (out / "NEW-SUMMARIES-v1.json.gz").write_bytes(gzip.compress(canonical(summary_bytes), mtime=0))
    for name in ("DEPENDENCIES-v1.json", "NEW-ENDPOINTS-v1.json.gz", "NEW-SUMMARIES-v1.json.gz"):
        prepared[name] = identity((out / name).read_bytes())
    (out / "INPUT-SPEC-v1.json").write_bytes(canonical(prepared) + b"\n")
    return {
        "status": "prepared_not_submitted",
        "new_count": len(new),
        "already_processed": len(removed),
        "input_objects": len(refs),
        "input_bytes": total,
        "control_bytes": sum(p.stat().st_size for p in out.iterdir()),
        "output": str(out),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--closeout", required=True)
    parser.add_argument("--adoption", required=True, action="append")
    parser.add_argument("--processed", action="append", default=[])
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.closeout, args.adoption, args.out, processed=args.processed)))


if __name__ == "__main__":
    main()
