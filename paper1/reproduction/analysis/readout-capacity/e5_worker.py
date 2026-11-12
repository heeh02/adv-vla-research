"""E5 readout-capacity worker: affine (reference) + rfridge + MLP on identical inputs.

Scope: fit the three readout families on the SAME 72 frozen splits / feature blocks /
390-d targets as the affine current-basis readout, aggregate to task-component MSE,
bootstrap with the SAME frozen draws, form the SAME 6 contrast pairs, and report
relative error reductions, intervals, and the number of contrasts whose sign flips
affine->rfridge and affine->mlp.

Two entry modes:
  --dry-synthetic : self-contained numpy dry run on synthetic arrays (no BOS, no creds),
                    proving the pipeline runs, is deterministic, and that the affine path
                    equals a direct frozen legacy fit. Run with python3 -I.
  --execute-cloud : cloud capsule entry (reads COMPACT-J0-FEATURES + bases + draws from the
                    private root the launcher stages); writes small RESULT tables only. The
                    launcher/credentialed submission is the total-controller's step, not this
                    module's; a Mac subagent never submits (总控唯一云端提交).
"""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
import numpy as np

# This module's own directory (safe, authored here) — needed under python3 -I,
# which does not add the script directory to sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import e5_readout_math as E

HERE = Path(__file__).resolve().parent
LAYER_LABEL = {0: "L1", 8: "L9", 17: "L18"}
METRICS = tuple("adapter7_" + c + "_first5" for c in ("translation", "rotation", "gripper"))
SIMPLE_BLOCKS = ("magnitude", "full", "flow", "new_response", "new_clean", "new_random", "old_response")
PAIRS = (("new_response", "magnitude"), ("new_response", "full"), ("new_response", "new_clean"),
         ("new_response", "new_random"), ("new_response", "old_response"), ("flow_plus_new", "flow"))
READOUTS = ("affine", "rfridge", "mlp")
BASIS_KEY = {"new_response": "position_delta__basis32", "new_clean": "position_clean__basis32",
             "new_random": "curve_random32", "old_response": "delta_uncentered_PCA"}


def identity_transform(M, d):
    return M.Transform(mean=np.zeros(d), scale=1.0)


def build_predictors(M, feats, bases, layer, n):
    h = feats["hidden_delta"][:, M.LAYERS.index(layer)].astype(np.float64)   # [n,10,1024]
    pred = {"magnitude": np.linalg.norm(h, axis=-1),                          # [n,10]
            "full": h.reshape(n, 10240),
            "flow": feats["flow0_delta"].astype(np.float64)}                  # [n,320]
    for block, key in BASIS_KEY.items():
        if key in bases:
            b = np.asarray(bases[key], np.float64)[:, :16]
            pred[block] = (h @ b).reshape(n, 160)
    return pred


def fit_one(M, readout, block, Xtr, Xte, ytr, w, seed_text, groups):
    if readout == "affine":
        model = M.fit_weighted_affine_ridge(Xtr, ytr, w, 0.01)
        return model.predict(Xte), {"family": "affine", "lambda": 0.01, "feature_dim_in": int(Xtr.shape[1])}
    if readout == "rfridge":
        D = 2048 if Xtr.shape[1] >= 64 else max(16, 4 * Xtr.shape[1])
        predict, proof = E.fit_rfridge(M, Xtr, ytr, w, seed_text, groups, D=D)
        return predict(Xte), proof
    predict, proof = E.fit_mlp(M, Xtr, ytr, w, seed_text, groups)
    return predict(Xte), proof


def fit_split(M, feats, rows, components, item, bases, hyper, allowed=None):
    s = item["support"]
    train, test, _ = M.split_rows(rows, s["family"], s["mode"], s["fold"], components)
    tr_rows = [rows[i] for i in train]
    M.validate_split(tr_rows, [rows[i] for i in test], components, s)  # full-test support identity gate
    # Evaluation restricted to the common-task eval set (allowed); training uses all
    # non-heldout cases (unchanged), exactly as the current-basis readout of record.
    if allowed is not None:
        test = [i for i in test if components[rows[i]["task_id"]] in allowed]
    te_rows = [rows[i] for i in test]
    w = M.case_weights(tr_rows, components)
    groups = [components[r["task_id"]] for r in tr_rows]
    y = feats["target"].astype(np.float64)
    ytr, yte = y[train], y[test]
    pred = build_predictors(M, feats, bases, s["layer"], len(rows))
    context = {k: s[k] for k in ("family", "layer", "mode", "fold")}
    task_rows = []
    for readout in hyper["readouts"]:
        preds = {}
        for block in SIMPLE_BLOCKS:
            if block not in pred:
                continue
            seed = f"{readout}|{s['family']}|{s['layer']}|{s['fold']}|{block}"
            preds[block], _ = fit_one(M, readout, block, pred[block][train], pred[block][test], ytr, w, seed, groups)
        # flow_plus_new: per-subblock standardized concat (affine uses block ridge).
        if "new_response" in pred:
            if readout == "affine":
                blk = M.fit_block_ridge({"flow0": pred["flow"][train], "delta_coeff": pred["new_response"][train]},
                                        ytr, w, 0.01)
                preds["flow_plus_new"] = blk.predict({"flow0": pred["flow"][test], "delta_coeff": pred["new_response"][test]})
            else:
                tf = M.fit_transform(pred["flow"][train], w)
                tn = M.fit_transform(pred["new_response"][train], w)
                ztr = np.concatenate([tf.apply(pred["flow"][train]), tn.apply(pred["new_response"][train])], 1)
                zte = np.concatenate([tf.apply(pred["flow"][test]), tn.apply(pred["new_response"][test])], 1)
                ident = identity_transform(M, ztr.shape[1])
                seed = f"{readout}|{s['family']}|{s['layer']}|{s['fold']}|flow_plus_new"
                if readout == "rfridge":
                    fn, _ = E.fit_rfridge(M, ztr, ytr, w, seed, groups, D=2048, transform=ident)
                else:
                    fn, _ = E.fit_mlp(M, ztr, ytr, w, seed, groups, transform=ident)
                preds["flow_plus_new"] = fn(zte)
        for block, pr in preds.items():
            scores = M.score_action_components(yte, pr)
            for r in M.aggregate_component_scores({k: scores[k] for k in METRICS}, te_rows, components):
                task_rows.append({**context, "readout": readout, "model": block, **r})
    return task_rows


def summarize(M, task_rows, draws):
    """Join folds (each component once per cell) then bootstrap with the frozen draws."""
    keys = ("readout", "family", "layer", "metric", "model")
    grouped = defaultdict(dict)
    for r in task_rows:
        key = tuple(r[k] for k in keys)
        if r["task_component"] in grouped[key]:
            raise ValueError("one_prediction_per_task_after_fold_join")
        grouped[key][r["task_component"]] = r
    estimates, contrasts = [], []

    draw_cache = {}

    def boot(vals, fam, layer):
        dk = f"{fam}|{layer}"
        if dk not in draw_cache:
            draw_cache[dk] = np.asarray(draws[dk]["indices"], dtype=np.intp)
        idx = draw_cache[dk]
        a = np.asarray(vals, float)
        means = a[idx].mean(axis=1)                 # vectorized 2000-draw bootstrap
        return float(a.mean()), [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))], means

    resampled = {}
    for key, by_task in grouped.items():
        readout, fam, layer, metric, model = key
        ids = sorted(by_task)
        mse = [by_task[t]["mse"] for t in ids]
        mean, ci, means = boot(mse, fam, layer)
        resampled[key] = means
        estimates.append({"readout": readout, "family": fam, "layer": layer, "metric": metric, "model": model,
                          "task_count": len(ids), "case_count": sum(by_task[t]["case_count"] for t in ids),
                          "task_equal_mse": mean, "interval_95": ci})
    strata = sorted({k[:4] for k in grouped})
    for (readout, fam, layer, metric) in strata:
        for cand, base in PAIRS:
            ck, bk = (readout, fam, layer, metric, cand), (readout, fam, layer, metric, base)
            if ck not in grouped or bk not in grouped:
                contrasts.append({"readout": readout, "family": fam, "layer": layer, "metric": metric,
                                  "candidate": cand, "baseline": base, "status": "unavailable"})
                continue
            ca, ba = grouped[ck], grouped[bk]
            ids = sorted(ca)
            c = np.array([ca[t]["mse"] for t in ids]); b = np.array([ba[t]["mse"] for t in ids])
            bc, bb = resampled[ck], resampled[bk]
            diffs = bb - bc
            bad = int((bb <= 0).sum())
            rr = float((b.mean() - c.mean()) / b.mean()) if b.mean() > 0 else None
            rr_ci = None if bad else [float(np.quantile(diffs / bb, 0.025)), float(np.quantile(diffs / bb, 0.975))]
            contrasts.append({"readout": readout, "family": fam, "layer": layer, "metric": metric,
                              "candidate": cand, "baseline": base, "status": "measured",
                              "task_count": len(ids),
                              "relative_error_reduction": rr, "relative_error_reduction_interval_95": rr_ci,
                              "rel_ci_sign": E.ci_sign(rr_ci), "rel_point_sign": E.point_sign(rr),
                              "diff_mean_baseline_minus_candidate": float((b - c).mean()),
                              "undefined_bootstrap_denominators": bad})
    return estimates, contrasts


def sign_change_counts(contrasts):
    by = {(c["readout"], c["family"], c["layer"], c["metric"], c["candidate"], c["baseline"]): c
          for c in contrasts if c.get("status") == "measured"}
    out = {}
    for target in ("rfridge", "mlp"):
        pt = ci = total = 0
        pt_cells, ci_cells = [], []
        for k, cref in by.items():
            if k[0] != "affine":
                continue
            kk = (target,) + k[1:]
            if kk not in by:
                continue
            cnl = by[kk]
            total += 1
            if cnl["rel_point_sign"] != cref["rel_point_sign"]:
                pt += 1
                pt_cells.append({"family": k[1], "layer": LAYER_LABEL.get(k[2], k[2]), "metric": k[3],
                                 "candidate": k[4], "baseline": k[5],
                                 "affine": cref["rel_point_sign"], target: cnl["rel_point_sign"]})
            if cnl["rel_ci_sign"] != cref["rel_ci_sign"]:
                ci += 1
                ci_cells.append({"family": k[1], "layer": LAYER_LABEL.get(k[2], k[2]), "metric": k[3],
                                 "candidate": k[4], "baseline": k[5],
                                 "affine": cref["rel_ci_sign"], target: cnl["rel_ci_sign"]})
        out[target] = {"contrasts_compared": total, "point_sign_changes": pt, "ci_category_changes": ci,
                       "point_sign_change_cells": pt_cells, "ci_category_change_cells": ci_cells}
    return out


def common_components(M, rows, components):
    from collections import defaultdict as _dd
    fam = _dd(set)
    for r in rows:
        fam[r["family"]].add(components[r["task_id"]])
    fams = sorted(fam)
    return set.intersection(*[fam[f] for f in fams])


def run(M, feats, rows, bases_by_split, splits, draws, hyper):
    components = M.task_components(rows)
    allowed = common_components(M, rows, components)
    import sys as _sys, time as _time
    task_rows = []
    expl = [it for it in splits if it["support"]["mode"] == "exploratory_unseen_task_cross_family"]
    for n, item in enumerate(expl, 1):
        t0 = _time.time()
        bkey = f"{item['support']['family']}|{item['support']['layer']}|{item['support']['fold']}"
        task_rows.extend(fit_split(M, feats, rows, components, item, bases_by_split[bkey], hyper, allowed=allowed))
        s = item["support"]
        print(f"[split {n}/{len(expl)}] {s['family']} L{s['layer']} fold{s['fold']} {round(_time.time()-t0,1)}s", flush=True)
    estimates, contrasts = summarize(M, task_rows, draws)
    return {"estimates": estimates, "contrasts": contrasts, "sign_change_counts": sign_change_counts(contrasts),
            "n_task_rows": len(task_rows), "readouts": list(hyper["readouts"])}


# --------------------------------------------------------------------------- synthetic dry run
def synth(seed=0):
    rng = np.random.default_rng(seed)
    families = list(M_FAMILIES)
    layers = [8]
    n_comp_per_fam = 3
    variants, noises = 2, 2
    rows, comp_clean = [], {}
    cid = 0
    # Shared-clean components: give each (family-agnostic) task its own clean id so
    # task_components groups them; keep 1 task per component for a clean structural run.
    for fam in families:
        for cpart in range(n_comp_per_fam):
            task_id = f"{fam[:3]}-task-{cpart}"
            clean_id = f"clean-{fam[:3]}-{cpart}"
            for vpart in range(variants):
                for noise in range(noises):
                    rows.append({"case_id": f"case-{cid}", "task_id": task_id, "clean_id": clean_id,
                                 "family": fam, "row_id": f"{task_id}-v{vpart}", "noise_label": noise,
                                 "suite": "libero_10"})
                    cid += 1
    rows.sort(key=lambda r: r["case_id"])
    n = len(rows)
    hidden = (rng.standard_normal((n, 3, 10, 1024)) * 0.1).astype(np.float32)

    def ortho():
        q, _ = np.linalg.qr(rng.standard_normal((1024, 32)))
        return q[:, :32]
    bases = {k: ortho() for k in ("position_delta__basis32", "position_clean__basis32",
                                  "curve_random32", "delta_uncentered_PCA")}
    flow = (rng.standard_normal((n, 320)) * 0.1).astype(np.float32)
    # Give the target real signal from the L9 response coordinates (so new_response
    # carries information) plus independent noise; shapes mirror the real 390-d target.
    h9 = hidden[:, 1].astype(np.float64)                               # [n,10,1024]
    coords = (h9 @ bases["position_delta__basis32"][:, :16]).reshape(n, 160)
    W = rng.standard_normal((160, 390)) * 0.2
    target = coords @ W + rng.standard_normal((n, 390)) * 0.05
    feats = {"hidden_delta": hidden, "flow0_delta": flow, "target": target.astype(np.float64)}
    return rows, feats, bases, families, layers


M_FAMILIES = ("Background Textures", "Camera Viewpoints", "Light Conditions", "Sensor Noise")


def build_synth_splits(M, rows, bases, families, layers):
    components = M.task_components(rows)
    splits, bases_by_split, comps_sorted = [], {}, sorted(set(components.values()))
    for fam in families:
        for layer in layers:
            for fold in range(5):
                train, test, _ = M.split_rows(rows, fam, "exploratory_unseen_task_cross_family", fold, components)
                if not train or not test:
                    continue
                tr_rows, te_rows = [rows[i] for i in train], [rows[i] for i in test]
                support = {"family": fam, "layer": layer, "mode": "exploratory_unseen_task_cross_family", "fold": fold,
                           "training_case_count": len(tr_rows), "test_case_count": len(te_rows),
                           "training_case_ids_sha256": M.case_set_sha(tr_rows),
                           "test_case_ids_sha256": M.case_set_sha(te_rows)}
                splits.append({"support": support})
                bases_by_split[f"{fam}|{layer}|{fold}"] = bases
    return splits, components, comps_sorted


def synth_draws(comps_present, families, layers, reps=200, seed=7):
    rng = np.random.default_rng(seed)
    draws = {}
    for fam in families:
        for layer in layers:
            k = len(comps_present[(fam, layer)])
            if k:
                draws[f"{fam}|{layer}"] = {"indices": rng.integers(0, k, (reps, k)).tolist()}
    return draws


def dry_synthetic():
    legacy_path = HERE.parent / "frozen-reference" / "legacy_math.py"
    M = E.load_legacy(legacy_path)
    rows, feats, bases, families, layers = synth()
    splits, components, _ = build_synth_splits(M, rows, bases, families, layers)
    # which components appear in each (family,layer) test after fold join
    present = defaultdict(set)
    for item in splits:
        s = item["support"]
        _, test, _ = M.split_rows(rows, s["family"], s["mode"], s["fold"], components)
        for i in test:
            present[(s["family"], s["layer"])].add(components[rows[i]["task_id"]])
    present = {k: sorted(v) for k, v in present.items()}
    draws = synth_draws(present, families, layers)
    hyper = {"readouts": READOUTS}
    # shrink nonlinear sizes for a fast structural run
    import e5_readout_math as _E
    orig_rf, orig_mlp = _E.fit_rfridge, _E.fit_mlp

    def small_rf(M, Xtr, Ytr, w, st, g, **kw):
        kw["D"] = 64
        return orig_rf(M, Xtr, Ytr, w, st, g, **kw)

    def small_mlp(M, Xtr, Ytr, w, st, g, **kw):
        kw.update(hidden=16, max_epochs=40, patience=10)
        return orig_mlp(M, Xtr, Ytr, w, st, g, **kw)

    _E.fit_rfridge, _E.fit_mlp = small_rf, small_mlp
    try:
        result = run(M, feats, rows, {k: bases for k in [f"{f}|{l}|{fo}" for f in families for l in layers for fo in range(5)]},
                     splits, draws, hyper)
    finally:
        _E.fit_rfridge, _E.fit_mlp = orig_rf, orig_mlp
    # affine-equals-direct-legacy sanity on one block/split
    sane = affine_matches_legacy(M, feats, rows, components, splits)
    receipt = {"status": "dry_synthetic_ok", "cases": len(rows), "splits": len(splits),
               "components_total": len(set(components.values())),
               "readouts": list(READOUTS),
               "n_estimates": len(result["estimates"]), "n_contrasts": len(result["contrasts"]),
               "all_scores_finite": all(np.isfinite(e["task_equal_mse"]) for e in result["estimates"]),
               "sign_change_counts": result["sign_change_counts"],
               "affine_equals_frozen_legacy_max_abs_diff": sane}
    print(json.dumps(receipt, indent=2, sort_keys=True))
    (HERE.parent / "DRY-RUN-v1.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return 0


def affine_matches_legacy(M, feats, rows, components, splits):
    item = splits[0]; s = item["support"]
    train, test, _ = M.split_rows(rows, s["family"], s["mode"], s["fold"], components)
    w = M.case_weights([rows[i] for i in train], components)
    y = feats["target"].astype(np.float64)
    h = feats["hidden_delta"][:, M.LAYERS.index(s["layer"])].astype(np.float64)
    X = h.reshape(len(rows), 10240)
    mine, _ = fit_one(M, "affine", "full", X[train], X[test], y[train], w, "x", [components[rows[i]["task_id"]] for i in train])
    direct = M.fit_weighted_affine_ridge(X[train], y[train], w, 0.01).predict(X[test])
    return float(np.max(np.abs(mine - direct)))


# --------------------------------------------------------------------------- real local run (owner-granted bounded exception)
import gzip


def load_real(features_npz, rows_json, split_index_json, fits_dir, draws_path):
    legacy_path = HERE.parent / "frozen-reference" / "legacy_math.py"
    M = E.load_legacy(legacy_path)
    feat = np.load(features_npz, allow_pickle=False)
    case_ids = [str(c) for c in feat["case_ids"]]
    feats = {"hidden_delta": feat["hidden_delta"], "flow0_delta": feat["flow0_delta"], "target": feat["target"]}
    rows_in = json.loads(Path(rows_json).read_text())
    by_id = {r["case_id"]: r for r in rows_in}
    assert set(by_id) == set(case_ids) and len(by_id) == len(case_ids) == 536, "row/case_id mismatch"
    rows = [by_id[c] for c in case_ids]                       # align rows to feature order
    index = json.loads(Path(split_index_json).read_text())
    fits = Path(fits_dir)
    bases_by_split, splits = {}, []
    for it in index:
        key = f"{it['family']}|{it['layer']}|{it['fold']}"
        d = np.load(fits / it["local"], allow_pickle=False)
        bases_by_split[key] = {"position_delta__basis32": np.asarray(d["position_delta__basis32"], np.float64),
                               "position_clean__basis32": np.asarray(d["position_clean__basis32"], np.float64),
                               "curve_random32": np.asarray(d["curve_random32"], np.float64),
                               "delta_uncentered_PCA": np.asarray(d["uncentered_delta__basis32"], np.float64)}
        splits.append({"support": {"family": it["family"], "layer": it["layer"], "mode": it["mode"], "fold": it["fold"],
                                   "training_case_count": it["training_case_count"], "test_case_count": it["test_case_count"],
                                   "training_case_ids_sha256": it["training_case_ids_sha256"],
                                   "test_case_ids_sha256": it["test_case_ids_sha256"]}})
    draws = json.loads(gzip.decompress(Path(draws_path).read_bytes()))["draws"]
    return M, feats, rows, bases_by_split, splits, draws


def affine_gate(M, result, expected_path):
    """Affine task_equal_mse and new_response contrasts must match expected.json."""
    exp = json.loads(Path(expected_path).read_text())
    exp_est = {(r["family"], r["layer"], r["metric"], r["model"]): r for r in exp["estimates"]}
    exp_con = {(r["family"], r["layer"], r["metric"], r["candidate"], r["baseline"]): r for r in exp["contrasts"]}
    max_est = max_con = 0.0
    n_est = n_con = 0
    for e in result["estimates"]:
        if e["readout"] != "affine":
            continue
        k = (e["family"], e["layer"], e["metric"], e["model"])
        if k in exp_est:
            max_est = max(max_est, abs(e["task_equal_mse"] - exp_est[k]["task_equal_mse"]))
            n_est += 1
    for c in result["contrasts"]:
        if c["readout"] != "affine" or c.get("status") != "measured":
            continue
        k = (c["family"], c["layer"], c["metric"], c["candidate"], c["baseline"])
        if k in exp_con and exp_con[k].get("relative_error_reduction") is not None and c["relative_error_reduction"] is not None:
            max_con = max(max_con, abs(c["relative_error_reduction"] - exp_con[k]["relative_error_reduction"]))
            n_con += 1
    per_fam = defaultdict(lambda: {"dMSE": 0.0, "dRel": 0.0, "est": 0, "con": 0})
    for e in result["estimates"]:
        if e["readout"] != "affine":
            continue
        k = (e["family"], e["layer"], e["metric"], e["model"])
        if k in exp_est:
            per_fam[e["family"]]["dMSE"] = max(per_fam[e["family"]]["dMSE"], abs(e["task_equal_mse"] - exp_est[k]["task_equal_mse"]))
            per_fam[e["family"]]["est"] += 1
    for c in result["contrasts"]:
        if c["readout"] != "affine" or c.get("status") != "measured":
            continue
        k = (c["family"], c["layer"], c["metric"], c["candidate"], c["baseline"])
        if k in exp_con and exp_con[k].get("relative_error_reduction") is not None and c["relative_error_reduction"] is not None:
            per_fam[c["family"]]["dRel"] = max(per_fam[c["family"]]["dRel"], abs(c["relative_error_reduction"] - exp_con[k]["relative_error_reduction"]))
            per_fam[c["family"]]["con"] += 1
    per_fam = {f: {**v, "within_1e-10": v["dMSE"] <= 1e-10 and v["dRel"] <= 1e-10} for f, v in per_fam.items()}
    return {"estimates_compared": n_est, "max_abs_diff_task_equal_mse": max_est,
            "contrasts_compared": n_con, "max_abs_diff_relative_error_reduction": max_con,
            "within_1e-10": max_est <= 1e-10 and max_con <= 1e-10, "per_family": per_fam}


def execute_local(args):
    M, feats, rows, bases_by_split, splits, draws = load_real(
        args.features, args.rows, args.split_index, args.fits_dir, args.draws)
    hyper = {"readouts": READOUTS}
    result = run(M, feats, rows, bases_by_split, splits, draws, hyper)
    gate = affine_gate(M, result, args.expected)
    outdir = Path(args.out)
    results = {"schema": "e5-readout-capacity-results-v1",
               "run": "local-mac-bounded-exception-20261017",
               "primary_cases": 404, "primary_tasks": 28, "splits_used": len(splits),
               "readouts": list(READOUTS),
               "affine_reproduction_gate": gate,
               "estimates": result["estimates"], "contrasts": result["contrasts"],
               "sign_change_counts": result["sign_change_counts"],
               "n_task_rows": result["n_task_rows"]}
    (outdir / "RESULTS-v1.json").write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "execute_local_ok", "affine_gate": gate,
                      "sign_change_counts": {k: {kk: v[kk] for kk in ("contrasts_compared", "point_sign_changes", "ci_category_changes")}
                                             for k, v in result["sign_change_counts"].items()}}, indent=2))
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-synthetic", action="store_true")
    ap.add_argument("--execute-local", action="store_true")
    ap.add_argument("--features"); ap.add_argument("--rows"); ap.add_argument("--split-index")
    ap.add_argument("--fits-dir"); ap.add_argument("--draws"); ap.add_argument("--expected"); ap.add_argument("--out")
    args = ap.parse_args()
    if args.dry_synthetic:
        return dry_synthetic()
    if args.execute_local:
        return execute_local(args)
    print(json.dumps({"status": "no_mode", "hint": "use --dry-synthetic or --execute-local"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
