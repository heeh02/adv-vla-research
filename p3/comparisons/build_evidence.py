"""Recompute compact numeric interpretation from existing verified small tables."""
import csv
import hashlib
import json
from pathlib import Path
from compare_endpoints import canonical, read_ref, need

HERE = Path(__file__).resolve().parent
MAIN = next(p for p in HERE.parents if (p/"AGENTS.md").is_file())
T = MAIN/"docs/experiments/t20-optimization-continuation-20261019-v1"
B = T/"curve-analysis-v1/technical-v2/prepared-v2/run-v1/readback/science"
C = HERE.parent


def ref(path):
    data = path.read_bytes()
    return dict(path=str(path), bytes=len(data), sha256=hashlib.sha256(data).hexdigest())


def rows(name):
    return list(csv.DictReader((B/name).open()))


def write(name, obj):
    (HERE/name).write_bytes(canonical(obj)+b"\n")


def csvout(name, values):
    with (HERE/name).open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(values[0]))
        w.writeheader(); w.writerows(values)


def main():
    actual = json.loads((T/"curve-analysis-v1/technical-v2/ACTUAL-RESULT-v2.json").read_bytes())
    summary = read_ref(actual["summary"])
    need(summary["status"] == "complete_for_frozen_available_inputs", "actual CPU summary")
    internal = rows("INTERNAL-CURVES.csv")
    probe = rows("PROBE-CURVES.csv")
    windows = rows("TRAINING-WINDOWS.csv")
    readiness = json.loads((C/"science/READINESS-v1.json").read_bytes())
    models = {}
    for m in readiness["models"]:
        model = m["model"]
        read_ref(m["checkpoint_metadata_verified"])
        seed = int(model["id"].split("-s")[1][0])
        models[model["id"]] = dict(checkpoint_sha256=model["checkpoint"]["sha256"], seed=seed,
            source_kind=m["source_kind"], parent_T20=m["parent_T20"])
    baseline_path = MAIN/"docs/experiments/existing-data-synthesis-20261018-v1/behavior/actual-complete-v1/SINGLE-ENDPOINT-BASELINE-v2.json"
    baseline = json.loads(baseline_path.read_bytes())
    for m in baseline["models"]:
        if m["model"] in ("original", "T-s0-u20000-EMA", "T-s1-u20000-EMA"):
            models[m["model"]] = dict(checkpoint_sha256=m["checkpoint_sha256"], seed=m["seed"], source_kind="historical_adopted_endpoint")
    gate = readiness["comparisons"]["gate_source"]
    need(ref(Path(gate["path"])) == gate, "pair gate source remains frozen")
    write("INPUT-PLAN-v1.json", dict(schema="T30-comparison-inputs-v1", models=models,
        panels={"Plus": {"planned": 10030}, "L10": {"planned": 300}},
        contrasts=readiness["comparisons"]["contrasts"], frozen_pair_gate=gate))

    changes = []
    for label in ("floor-s0", "floor-s1", "reheat-s0", "reheat-s1"):
        for condition in ("C0_Hbase", "C15_Hbase", "C0_Hoff", "C15_Hoff"):
            vals = {int(r["step"]): r for r in internal if r["run_label"] == label and r["condition"] == condition}
            base = float(vals[20000]["mean"])
            need(vals[20000]["measurement_role"] == "restoration_check", "restored baseline separate")
            for step in (25000, 29000, 30000):
                end = float(vals[step]["mean"]) if step in vals else None
                changes.append(dict(run_label=label, condition=condition, step=step, restored20k=base,
                    value=end, absolute_change=end-base if end is not None else None,
                    percent_change=100*(end/base-1) if end is not None else None,
                    states=800, demos=400, tasks=40, metric="H10_D32", view="EMA",
                    missing_reason=None if end is not None else "30k validation not preserved in archive"))
    csvout("INTERNAL-VALIDATION-CHANGES-v1.csv", changes)
    fit = []
    for label in ("floor-s0", "floor-s1", "reheat-s0", "reheat-s1"):
        for role in ("C_old", "T_added"):
            for metric in ("H10_D7", "H10_D32"):
                vals = {int(r["step"]): r for r in probe if r["run_label"] == label and r["role"] == role and r["metric"] == metric and r["view"] == "EMA"}
                base = float(vals[20000]["task_macro_mean"])
                for step in (25000, 30000):
                    end = float(vals[step]["task_macro_mean"]) if step in vals else None
                    fit.append(dict(run_label=label, role=role, metric=metric, view="EMA", step=step,
                        baseline20k=base, value=end, absolute_change=end-base if end is not None else None,
                        percent_change=100*(end/base-1) if end is not None else None,
                        states=400, demos=200, tasks=40))
    csvout("FIXED-TRAIN-PROBE-CHANGES-v1.csv", fit)
    last = []
    obs_path = T/"infra/allocation-v3-live/USER-STATUS-20261019-v5.json"
    obs = json.loads(obs_path.read_bytes())
    for j in obs["jobs"]:
        tele = read_ref(j["telemetry_BOS_ref"])
        need(j["telemetry"]["valid_actual_update"] is True, "reader classified a valid update")
        need(all(tele[k] == j["telemetry"][k] for k in ("logical_run_id", "allocation_id", "sequence", "observed_epoch")), "raw/normalized telemetry identity")
        need(all(tele["training"][k] == v for k, v in j["telemetry"]["training"].items()), "raw telemetry fields agree")
        need(tele["training"]["actual_stock_step"] == 30000 and tele["training"]["status"] == "observed_complete_update", "actual completed update30000")
        x = tele["training"]
        label = j["logical_run_id"].replace("jinwen-adv-vla-t30-", "").split("-20261019")[0]
        last.append(dict(run_label=label, effective_step=x["effective_update"], last_batch_loss=x["loss"],
            grad_norm=x["grad_norm"], LR_before_update=x["LR_before_update"],
            last_batch_not_convergence_statistic=True,
            complete_online_curve=label in ("floor-s0", "reheat-s1"),
            observed_UTC=tele["observed_UTC"], source=j["telemetry_BOS_ref"]))
    write("FINAL-UPDATE-EVIDENCE-v1.json", last)
    w = []
    for label in ("floor-s0", "reheat-s1"):
        rr = [r for r in windows if r["run_label"] == label and r["metric"] == "loss"]
        a = next(r for r in rr if r["first_update"] == "20001")
        z = next(r for r in rr if r["first_update"] == "29001")
        w.append(dict(run_label=label, first1000_mean=float(a["mean"]), last1000_mean=float(z["mean"]),
            percent_change_mean=100*(float(z["mean"])/float(a["mean"])-1),
            first1000_median=float(a["median"]), last1000_median=float(z["median"]),
            percent_change_median=100*(float(z["median"])/float(a["median"])-1), n_per_window=1000,
            moving_batch_not_fixed_population=True))
    write("ONLINE-WINDOW-CHANGES-v1.json", w)
    # These are endpoint-availability bounds, not measured input-qualified pair counts.
    old = {r["model"]: r for r in baseline["models"]}
    coverage = []
    for seed in (0, 1):
        a, b = old[f"T-s{seed}-u20000-EMA"], old["original"]
        coverage.append(dict(panel="Plus", control="original", treatment=a["model"], planned=10030,
            control_closed=b["closed"], treatment_closed=a["closed"],
            ordinal_intersection_possible_lower=max(0, a["closed"]+b["closed"]-10030),
            ordinal_intersection_possible_upper=min(a["closed"], b["closed"]),
            actual_same_case_intersection=None, input_qualified_pairs=None,
            reason="small aggregate counts do not identify actual common ordinals or q0/runtime/noise equality"))
    write("BASELINE-COVERAGE-v1.json", dict(snapshot_scientific_generation=106,
        status_generation=107, baseline_ref=ref(baseline_path),
        selected_endpoints=[old[x] for x in ("original", "T-s0-u20000-EMA", "T-s1-u20000-EMA")],
        pair_readiness=coverage, L10=readiness["old_L10"], current_live_claim=False))
    write("SOURCE-MAP-v1.json", dict(curve_actual=ref(T/"curve-analysis-v1/technical-v2/ACTUAL-RESULT-v2.json"),
        curve_summary=actual["summary"], tables=[ref(B/x) for x in ("INTERNAL-CURVES.csv", "PROBE-CURVES.csv", "PROBE-CHANGES.csv", "PROBE-BRANCH-CONTRASTS.csv", "TRAINING-WINDOWS.csv")],
        final_update_observation=ref(obs_path), readiness=ref(C/"science/READINESS-v1.json"),
        old_Plus_catalog=readiness["old_Plus_catalog"], old_L10=readiness["old_L10"],
        normalized_endpoint_adapter=readiness["comparisons"]["normalizer_source"],
        exact_pair_gate=gate, cloud_CPU_required_for_full_cache=True,
        no_large_source_caches_restored_here=True))
    print(json.dumps({"validation_rows": len(changes), "fixed_fit_rows": len(fit), "verified_final_updates": len(last), "no_cloud_calls": True}))


if __name__ == "__main__":
    main()
