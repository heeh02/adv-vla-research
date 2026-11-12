"""Small, read-only T30 comparison calculator; never dispatches or adopts results.

Scientific input qualification is reused verbatim from the frozen existing-data
reader. Its absence remains a pairing gap even when a legal outcome is known.
Full historical endpoint caches must be processed on cloud CPU, not on the Mac.
"""
import argparse
import ast
import copy
import gzip
import hashlib
import json
import platform
from collections import Counter
from pathlib import Path

LEGACY_ADOPTION_SHA = {
    "e255118949e9d7cf9990c2f51e1c0cf15df3a80a24d97864c71b8085ab5837e6",
    "fe5e2278407c6fcef7bfebed13212debb8a45c88d1d7ed2208f54bf932463886",
}

def need(value, message):
    if not value:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def read_ref(ref):
    p = Path(ref["path"])
    body = p.read_bytes()
    need(len(body) == ref["bytes"] and hashlib.sha256(body).hexdigest() == ref["sha256"], "source bytes/SHA: " + str(p))
    return json.loads(body)


def load_gate(pin):
    body = Path(pin["path"]).read_bytes()
    need(len(body) == pin["bytes"] and hashlib.sha256(body).hexdigest() == pin["sha256"], "frozen pair gate source changed")
    names = {"restore_noise_keys", "pair_gate"}
    nodes = [n for n in ast.parse(body).body if isinstance(n, ast.FunctionDef) and n.name in names]
    need({n.name for n in nodes} == names, "original gate functions exist")
    ns = {"need": need, "canonical": canonical}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), pin["path"], "exec"), ns)
    return ns["restore_noise_keys"], ns["pair_gate"]


def collect_adoption(path, panel="Plus"):
    """Read only already-verified local metadata. No OUTCOME download or array IO.

    The two original 64-case receipts predate the applied boolean; later receipts
    must explicitly say applied. Scientific outcome and allocation identity are
    checked against each readback and the exact request.
    """
    body = Path(path).read_bytes()
    ad = json.loads(body)
    if "applied" in ad:
        need(ad["applied"] is True, "candidate adoption is not root adoption")
    else:
        need(hashlib.sha256(body).hexdigest() in LEGACY_ADOPTION_SHA, "only two exact legacy root adoptions lack applied")
    panel = ad.get("panel", panel)
    need(ad.get("scientific_attempt_changed") is False and ad.get("old_results_overwritten") is False, "adoption invariants")
    proof = read_ref(ad["proof"])
    request = read_ref(proof["request"])
    need(ad["run_id"] == proof["run_id"] == request["run_id"], "run identity")
    need(ad["job_id"] == proof["job_id"], "job identity")
    model = request["model"]
    adopted = ad["adopted_ordinals"]
    need(len(adopted) == len(set(adopted)), "unique adoption ordinals")
    byord = {r["population_ordinal"]: r for r in proof["cases"]}
    rows = []
    for ordinal in adopted:
        r = byord[ordinal]
        need(r.get("independent_case_closed") is True and r.get("independent_whole_and_all_member_BOS_readback_verified") is True, "independent case evidence")
        s = read_ref(r["case_summary"])
        need(s["id"] == r["id"] and s["population_ordinal"] == ordinal and s["run_id"] == ad["run_id"], "case identity")
        need(s["attempt"] == r["attempt"] and type(s["attempt"]) is int, "scientific attempt")
        need(s["success"] is r["success"] and s["status"] == r["status"], "adopted outcome invariant")
        need(s["model_identity_sha256"] == digest(model), "model identity hash")
        rows.append({
            "panel": panel, "model_id": model["id"], "checkpoint_sha256": model["checkpoint"]["sha256"],
            "ordinal": ordinal, "official_case": s["case"], "case_definition_sha256": digest(s["case"]),
            "population_sha256": s["population_sha256"], "id": s["id"], "run_id": s["run_id"],
            "attempt": s["attempt"], "adopted": True, "success": s["success"], "status": s["status"],
            "summary": {k: r["case_summary"][k] for k in ("uri", "bytes", "sha256")},
            "outcome": s["outcome"], "outcome_canonical_sha256": s["outcome_canonical_sha256"],
            "input": {"status": "not_materialized_for_pairing", "signature": None},
            "source_package_sha256": s["source_package_sha256"],
            "local_adoption": str(Path(path).resolve()),
        })
    return rows


def compare(plan, records):
    restore, gate = load_gate(plan["frozen_pair_gate"])
    buckets = {}
    duplicates = 0
    for raw in records:
        r = copy.deepcopy(raw)
        p, m, o = r["panel"], r["model_id"], r["ordinal"]
        need(p in plan["panels"] and m in plan["models"], "declared panel/model")
        need(type(o) is int and 0 <= o < plan["panels"][p]["planned"], "population ordinal")
        need(r["checkpoint_sha256"] == plan["models"][m]["checkpoint_sha256"], "checkpoint binding")
        need(r.get("adopted") is True and type(r.get("success")) is bool, "root-adopted boolean endpoint")
        need(type(r.get("attempt")) is int and r["attempt"] >= 0, "scientific attempt")
        need(r.get("status") in ("complete", "model_failure"), "legal terminal; technical failures remain absent")
        need(r["status"] != "model_failure" or r["success"] is False, "model failure cannot be successful")
        need(r["case_definition_sha256"] == digest(r["official_case"]), "exact case definition")
        need(len(r["summary"]["sha256"]) == 64 and len(r["outcome_canonical_sha256"]) == 64, "source references")
        r = restore(r)
        key = (p, m, o)
        if key in buckets:
            a = buckets[key]
            for field in ("run_id", "attempt", "case_definition_sha256", "outcome_canonical_sha256", "summary", "success", "status"):
                need(a[field] == r[field], "conflicting adopted duplicate: " + field)
            duplicates += 1
            ai, ri = a.get("input", {}), r.get("input", {})
            if ai.get("signature") is None and ri.get("signature") is not None:
                buckets[key] = r
            elif ai.get("signature") is not None and ri.get("signature") is not None:
                need(canonical(ai) == canonical(ri), "conflicting duplicate input qualification")
        else:
            buckets[key] = r

    singles, pairs = [], []
    for panel, p in plan["panels"].items():
        n = p["planned"]
        for m, info in plan["models"].items():
            rows = [r for (pp, mm, _), r in buckets.items() if pp == panel and mm == m]
            yes = sum(r["success"] for r in rows)
            missing = n - len(rows)
            singles.append(dict(panel=panel, model_id=m, seed=info["seed"], planned=n, closed=len(rows),
                success=yes, normal_failure=sum(not r["success"] and r["status"] == "complete" for r in rows),
                model_failure=sum(r["status"] == "model_failure" for r in rows), unresolved=missing,
                success_rate=yes/n if missing == 0 else None, completion_bounds=[yes/n, (yes+missing)/n],
                missing_categories_not_inferred=True))
        for c in plan["contrasts"]:
            reasons, cells = Counter(), Counter({k: 0 for k in ("SS", "SF", "FS", "FF")})
            same_case_both_present = 0
            for o in range(n):
                a = buckets.get((panel, c["control"], o))
                b = buckets.get((panel, c["treatment"], o))
                if a and b:
                    if a["case_definition_sha256"] != b["case_definition_sha256"]:
                        reason = "exact_case_definition_difference"
                    elif not a.get("population_sha256") or a.get("population_sha256") != b.get("population_sha256"):
                        reason = "population_identity_difference_or_missing"
                    else:
                        same_case_both_present += 1
                        reason = gate(a, b)
                else:
                    reason = "missing_endpoint"
                reasons[reason] += 1
                if reason == "qualified_recorded_pair":
                    cells[("S" if a["success"] else "F")+("S" if b["success"] else "F")] += 1
            q = sum(cells.values())
            unqualified = n-q
            net = cells["FS"]-cells["SF"]
            pairs.append(dict(panel=panel, **c, planned=n, same_case_both_present=same_case_both_present,
                qualified_pairs=q, cells=dict(cells), gains=cells["FS"], harms=cells["SF"],
                unresolved_or_unqualified=unqualified, pairing_reasons=dict(reasons),
                delta_success_rate=net/n if q == n else None,
                strict_pair_completion_bounds=[(net-unqualified)/n, (net+unqualified)/n],
                bounds_are_not_confidence_intervals=True,
                complete_case_subset_not_used_for_population_estimate=True))
    return dict(schema="T30-comparison-readonly-v1", single_endpoints=singles, paired_contrasts=pairs,
        unique_adopted_records=len(buckets), exact_duplicate_reads_not_samples=duplicates,
        original_gate=plan["frozen_pair_gate"], training_seed_pooling=False, CI=None,
        no_cloud_calls=True, no_registry_writes=True,
        limitations=["Partial sequential samples do not estimate full-population SR.",
            "Missing q0/runtime/noise blocks paired effects; legal single outcomes remain retained.",
            "The fixed benchmark is not a random sample of all robot tasks.",
            "Only two training seeds; episode counts do not estimate seed variability."])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", type=Path, required=True)
    ap.add_argument("--adoption", type=Path, action="append", default=[])
    ap.add_argument("--root-results", type=Path, help="Only direct children */ADOPTION-v1.json with applied=true; no readback-only records")
    ap.add_argument("--normalized-jsonl", type=Path, action="append", default=[])
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    need(not a.out.exists(), "output must be new")
    # Historical caches >200 MB must not be restored or scanned on this Mac.
    if platform.system() == "Darwin":
        need(sum(p.stat().st_size for p in a.normalized_jsonl) <= 32*1024**2, "full historical caches require cloud CPU")
    plan = json.loads(a.plan.read_bytes())
    records = []
    adoptions = list(a.adoption)
    if a.root_results:
        expected = Path(__file__).resolve().parent.parent/"root/results"
        need(a.root_results.resolve() == expected, "discovery is restricted to this paper3 root/results")
        for p in sorted(a.root_results.glob("*/ADOPTION-v1.json")):
            need(not p.is_symlink() and p.resolve().is_relative_to(expected.resolve()), "do not follow external adoption symlinks")
            ad = json.loads(p.read_bytes())
            if ad.get("applied") is True:
                adoptions.append(p)
    adoptions = sorted(set(p.resolve() for p in adoptions))
    source_refs = []
    for p in adoptions:
        records.extend(collect_adoption(p))
        body = p.read_bytes()
        source_refs.append(dict(path=str(p), bytes=len(body), sha256=hashlib.sha256(body).hexdigest()))
    for p in a.normalized_jsonl:
        op = gzip.open if p.suffix == ".gz" else open
        with op(p, "rt") as f:
            records.extend(json.loads(line) for line in f if line.strip())
    report = compare(plan, records)
    report["adoption_sources"] = source_refs
    a.out.mkdir(parents=True)
    (a.out/"COMPARISON-v1.json").write_bytes(canonical(report)+b"\n")
    coverage = dict(schema="T30-root-adoption-coverage-v1", adoption_sources=source_refs,
        exact_duplicate_reads_not_samples=report["exact_duplicate_reads_not_samples"],
        models=[dict(r, checkpoint_sha256=plan["models"][r["model_id"]]["checkpoint_sha256"])
            for r in report["single_endpoints"] if r["model_id"].startswith(("T-floor-", "T-reheat-"))],
        missing_is_not_false_or_proven_unstarted=True, unknown_or_unadopted_not_counted=True,
        root_adoptions_only=True, no_cloud_calls=True)
    (a.out/"COVERAGE-v1.json").write_bytes(canonical(coverage)+b"\n")
    print(json.dumps({"unique_adopted_records": report["unique_adopted_records"], "report": str(a.out/"COMPARISON-v1.json"), "new_experiments": 0}))


if __name__ == "__main__":
    main()
