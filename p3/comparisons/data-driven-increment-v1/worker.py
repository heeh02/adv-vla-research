"""Read only exact new OUTCOMEs and frozen baseline caches on cloud CPU.

No provider API, model/array IO, adoption, registry write or automatic publish.
Original frozen normalizer and pair gate are called for every newly affected
pair. Prior 3626 tables and old outcomes remain immutable references.
"""

import argparse
import base64
from collections import Counter
import gzip
import importlib.util
import io
import json
from pathlib import Path
import platform
import time


def need(ok, reason):
    if not ok:
        raise ValueError(reason)


PARENT_RESULT_SHA = "5938d81bbf9fa746c2944e84aca6d150dcb7dade9c5e1b4d3576f770f18ef6a7"
PARENT_TABLE_SHAS = {
    "870059244895a2ed6f716cd48b11340de8945caf2ad1cb2d052b719e039909bd",
    "aa37c54aeec3dfc3264922afeeabd772c96e8d972d07a7a4f678c21eb97cf8a7",
}


def bounded_gzip(raw, limit):
    with gzip.GzipFile(fileobj=io.BytesIO(raw)) as stream:
        value = stream.read(limit + 1)
    need(len(value) <= limit, "bounded_control_gzip")
    return value


def load_borrowed(path, ref):
    p = Path(path)
    spec = importlib.util.spec_from_file_location("_paper3_borrowed_increment", p)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    need(module.file_identity(p) == {k: ref[k] for k in ("bytes", "sha256")}, "borrowed_worker_SHA")
    return module


class SummaryOverlay:
    """Locally bundled exact summaries; only original OUTCOME uses GET."""

    def __init__(self, body, rows, transport, borrowed):
        raw = json.loads(bounded_gzip(body, 8 * 1024**2))
        self.refs = {r["summary"]["uri"]: r["summary"] for r in rows}
        need(
            len(self.refs) == len(rows) == len(raw) and set(raw) == set(self.refs), "complete_unique_packaged_summaries"
        )
        self.summaries = {}
        for uri, encoded in raw.items():
            data = base64.b64decode(encoded, validate=True)
            self.summaries[uri] = borrowed.exact(data, self.refs[uri])
        self.transport, self.seen = transport, set()

    def get(self, uri, limit):
        if uri not in self.summaries:
            return self.transport.get(uri, limit)
        need(uri not in self.seen and limit == self.refs[uri]["bytes"], "one_exact_local_summary_read")
        self.seen.add(uri)
        return self.summaries[uri]


def run(inputs, dependency_root, borrowed_worker_path, store, work, output, deadline_epoch):
    need(platform.system() == "Linux", "cloud_CPU_only_no_Mac_full_cache")
    need(0 < deadline_epoch - time.time() <= 3600, "bounded_absolute_deadline")
    inputs, work = Path(inputs), Path(work)
    task = json.loads((inputs / "INPUT-SPEC-v1.json").read_bytes())
    need(
        task["schema"] == "paper3-data-driven-endpoint-increment-v1"
        and task["parent_T30_total"] == 3626
        and task["GPU"] == task["model_calls"] == 0,
        "prepared_scope_and_parent",
    )
    parent = task["parent_snapshot"]
    need(
        parent["result"]["sha256"] == PARENT_RESULT_SHA
        and parent["T30_total"] == 3626
        and {x["sha256"] for x in parent["tables"]} == PARENT_TABLE_SHAS,
        "immutable_actual3626_parent_refs",
    )
    borrowed = load_borrowed(borrowed_worker_path, task["borrowed_worker"])
    need(borrowed.file_identity(__file__) == task["worker_source"], "current_worker_source_SHA")
    deps = json.loads(borrowed.exact((inputs / "DEPENDENCIES-v1.json").read_bytes(), task["DEPENDENCIES-v1.json"]))
    b, _repair, f, paths = borrowed.dependencies(deps, dependency_root)
    spec, docs = b.load_controls(paths["frozen-bridge/inputs-2648-v1/INPUT-SPEC-v1.json"].parent)
    need(
        spec["case_list_identity"] == task["case_list_identity"]
        and spec["models"][task["model_id"]] == task["model_source"],
        "unchanged_population_model_source",
    )
    endpoints = json.loads(
        bounded_gzip(
            borrowed.exact((inputs / "NEW-ENDPOINTS-v1.json.gz").read_bytes(), task["NEW-ENDPOINTS-v1.json.gz"]),
            8 * 1024**2,
        )
    )
    need(
        len(endpoints) == task["new_count"] > 0
        and sorted((x["ordinal"], x["outcome_canonical_sha256"]) for x in endpoints)
        == sorted((x["ordinal"], x["outcome_canonical_sha256"]) for x in task["new_endpoint_identity"]),
        "exact_new_adopted_endpoint_set",
    )
    need(
        all(
            x["model_id"] == task["model_id"]
            and x["run_id"] == task["run_id"]
            and x["attempt"] == 0
            and x["adopted"] is True
            for x in endpoints
        ),
        "one_actual_model_run_attempt0",
    )
    for row in endpoints:
        b.validate_endpoint(row, spec["models"], docs["historical_plan"]["population_cases"])
        need(
            row["official_case"] == docs["historical_plan"]["population_cases"][row["ordinal"]],
            "same_original_population_case",
        )
    readset = task["readset"]["objects"]
    need(
        len(readset) == task["readset"]["object_count"] == 2 + len(endpoints)
        and sum(x["bytes"] for x in readset) == task["readset"]["exact_total_bytes"]
        and {x["uri"] for x in readset}
        == {spec["caches"][name]["uri"] for name in ("old_cache", "merged_delta")}
        | {x["outcome"]["uri"] for x in endpoints},
        "exact_baselines_and_only_new_OUTCOME_readset",
    )
    transport = borrowed.ExactReads(store, readset, deadline_epoch)
    work.mkdir(parents=True, exist_ok=False)
    output = borrowed.OutputBudget(output)
    normalize, restore, gate = b.source_functions()
    ordinals = {x["ordinal"] for x in endpoints}
    controls = {c["control"] for c in task["contrasts"]}
    buckets, counts = {}, Counter()
    for name, count in (("old_cache", 106958), ("merged_delta", 7234)):
        ref = spec["caches"][name]
        path = work / (name + ".jsonl.gz")
        transport.download(ref["uri"], path, ref)
        source = b.stream_cache(path, count)
        selected = (x for x in source if x["ordinal"] in ordinals and x["model_id"] in controls)
        tagged = f.tagged_cache_rows(selected, ref)
        for row in b.cache_rows(docs["historical_plan"], name, tagged, restore, spec["case_list_identity"]):
            need(time.time() < deadline_epoch, "absolute_science_deadline")
            b.validate_endpoint(row, spec["models"], docs["historical_plan"]["population_cases"])
            b.add_endpoint(buckets, row)
            counts["baseline_rows"] += 1
    summary = borrowed.exact((inputs / "NEW-SUMMARIES-v1.json.gz").read_bytes(), task["NEW-SUMMARIES-v1.json.gz"])
    overlay = SummaryOverlay(summary, endpoints, transport, borrowed)
    materialized = []
    for raw in sorted(endpoints, key=lambda x: x["ordinal"]):
        need(time.time() < deadline_epoch, "absolute_science_deadline")
        row = b.materialize_t30(raw, spec["models"][raw["model_id"]], overlay, normalize)
        row["population_identity"] = spec["case_list_identity"]
        row = borrowed.compact_endpoint(restore(row))
        b.add_endpoint(buckets, row)
        materialized.append(row)
    need(
        len(overlay.seen) == len(endpoints)
        and len(transport.seen) == len(readset)
        and transport.bytes == task["readset"]["exact_total_bytes"],
        "each_baseline_and_new_OUTCOME_read_exactly_once",
    )
    traces, groups = [], []
    for contrast in task["contrasts"]:
        cell = Counter()
        reasons = Counter()
        for raw in sorted(endpoints, key=lambda x: x["ordinal"]):
            o = raw["ordinal"]
            control = buckets.get(("Plus", contrast["control"], o))
            treatment = buckets.get(("Plus", contrast["treatment"], o))
            need(treatment is not None, "new_treatment_must_materialize")
            if control is not None:
                need(
                    control["case_definition_sha256"] == treatment["case_definition_sha256"]
                    and control["population_identity"] == treatment["population_identity"],
                    "same_case_population_before_pair_gate",
                )
            diagnostic = b.pair_diagnostics(control, treatment, gate)
            reasons[diagnostic["reason"]] += 1
            if diagnostic["reason"] == "qualified_recorded_pair":
                cell[("S" if control["success"] else "F") + ("S" if treatment["success"] else "F")] += 1
            traces.append(
                {
                    "contrast": contrast["id"],
                    "ordinal": o,
                    "diagnostic": diagnostic,
                    "control_source": f.source_trace(control),
                    "treatment_source": f.source_trace(treatment),
                    "scope": "newly adopted cohort only; strict original q0/runtime/shared-noise gate",
                }
            )
        q = sum(cell.values())
        n = len(endpoints)
        net = cell["FS"] - cell["SF"]
        unresolved = n - q
        groups.append(
            {
                "contrast": contrast["id"],
                "control": contrast["control"],
                "treatment": contrast["treatment"],
                "new_cohort_n": n,
                "qualified_pairs": q,
                "SS": cell["SS"],
                "SF": cell["SF"],
                "FS": cell["FS"],
                "FF": cell["FF"],
                "gains": cell["FS"],
                "harms": cell["SF"],
                "net_qualified_count": net,
                "delta_SR": net / n if q == n else None,
                "qualified_subset_delta_SR": net / q if q else None,
                "qualified_subset_is_selected_not_population": True,
                "strict_completion_bounds": [(net - unresolved) / n, (net + unresolved) / n],
                "pairing_reasons": dict(reasons),
            }
        )
    with output.gzip("NEW-PAIR-TRACE-v1.jsonl.gz") as stream:
        for row in traces:
            stream.write(borrowed.canonical(row) + b"\n")
    with output.gzip("NORMALIZED-NEW-v1.jsonl.gz") as stream:
        for row in sorted(materialized, key=lambda x: x["ordinal"]):
            stream.write(borrowed.canonical(row) + b"\n")
    result = {
        "schema": "paper3-added-endpoint-pairs-v1",
        "status": "computed_unpublished",
        "model_id": task["model_id"],
        "run_id": task["run_id"],
        "parent_T30_total": 3626,
        "parent_snapshot": task["parent_snapshot"],
        "new_endpoint_identity": task["new_endpoint_identity"],
        "new_count": len(endpoints),
        "already_processed": task["already_processed"],
        "new_cohort_pairs": groups,
        "new_trace_rows": len(traces),
        "new_OUTCOME_GETs": len(endpoints),
        "old_OUTCOME_GETs": 0,
        "baseline_GETs": 2,
        "baseline_rows_scanned": 106958 + 7234,
        "baseline_rows_selected": counts["baseline_rows"],
        "input_objects": len(transport.seen),
        "input_bytes": transport.bytes,
        "input_spec": borrowed.file_identity(inputs / "INPUT-SPEC-v1.json"),
        "worker_source": borrowed.file_identity(__file__),
        "new_normalized_delta": borrowed.file_identity(output.directory / "NORMALIZED-NEW-v1.jsonl.gz"),
        "new_trace": borrowed.file_identity(output.directory / "NEW-PAIR-TRACE-v1.jsonl.gz"),
        "old3626_tables_reemitted": False,
        "old_baseline_cache_reemitted": False,
        "q0_runtime_noise_gate_changed": False,
        "GPU": 0,
        "model_calls": 0,
        "full_population_evaluation_complete": False,
        "CI": None,
        "training_seeds_pooled": False,
        "publication_status": "not_invoked_by_worker",
        "interpretation": "Only newly adopted cohort strict pairs; no full-population or missing-case point estimate.",
    }
    output.json(
        "INPUT-RECEIPT-v1.json",
        {
            "schema": "paper3-added-endpoint-input-receipt-v1",
            "readset": readset,
            "verified_objects": len(transport.seen),
            "verified_bytes": transport.bytes,
            "new_OUTCOME_GETs": len(endpoints),
            "old_OUTCOME_GETs": 0,
            "baseline_cache_GETs": 2,
            "summary_cloud_GETs": 0,
            "parent_snapshot": task["parent_snapshot"],
            "input_spec": result["input_spec"],
            "worker_source": result["worker_source"],
        },
    )
    output.json("RESULT-v1.json", result, reserve=False)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", required=True)
    parser.add_argument("--dependencies", required=True)
    parser.add_argument("--borrowed-worker", required=True)
    parser.add_argument("--work", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--deadline-epoch", type=float, required=True)
    parser.parse_args()
    # A network/storage adapter must provide an exact read-only Store. The
    # standalone CLI cannot silently fabricate credentials or use network.
    raise SystemExit("Use run(...) with the root-admitted ExactStore; no standalone cloud execution")


if __name__ == "__main__":
    main()
