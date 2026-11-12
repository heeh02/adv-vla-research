"""E1 Phase D: pi0 horizon-50 readout over the C3 within-run recordings (cloud CPU).

Assembles within-run clean/perturbed pairs from the C3 per-arm recordings (archived
actions NEVER enter here — PI condition 2), fits the per-split position-centered rank-16
dictionaries (basis_hn) and runs the readout (readout_hn) across the family x layer x fold
cells, then task means / bootstrap contrasts / retention. Mirrors the pi0.5 Section-4
quantities (same estimators/denominators) at horizon 50. Pure CPU numpy.

Feature definitions (from the pi0.5 producer action-readout-repair-v5/worker.load_pair):
  hidden_delta = H(T)[call0,:,0] - H(R)[call0,:,0], action positions only (pi0 drops the
                 state token: H is [10,3,1,51,1024] -> take [...,-50:,:] -> [3,50,1024]);
  clean_hidden = H(R)[call0,:,0][...,-50:,:];
  flow0_delta  = velocity(T)[0,0] - velocity(R)[0,0]  (reshape 50*32=1600);
  target       = concat((T-R) raw 50*32=1600, (T-R) adapter 50*7=350) = 1950.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import legacy_math_hn as L
import readout_hn as RO
import basis_hn as BZ

POS = 50
L.set_positions(POS)
RO.MODE = 'exploratory_unseen_task_cross_family'
LAYERS = (0, 8, 17)
FAMILIES = ('Background Textures', 'Camera Viewpoints', 'Light Conditions', 'Sensor Noise')
CLEAN_FOR_NOISE = {0: 'R0', 1: 'R1'}
PERT_FOR_NOISE = {0: 'T0', 1: 'T1'}


def _bf16_void_to_f32(a):
    """Recorder H/velocity are bf16; without ml_dtypes np.load yields a 2-byte void.
    bf16 is the high 16 bits of float32, so upcast exactly with a bit-shift (pure numpy)."""
    a = np.asarray(a)
    if a.dtype == np.dtype("V2"):
        return (a.view(np.uint16).astype(np.uint32) << 16).view(np.float32)
    if a.dtype.kind == "f":
        return a.astype(np.float32)
    return a  # strings / ints unchanged


def load_recordings(science_dir):
    """science_dir/<row_id>/<arm>/recording-v1.npz -> {(row_id,noise,role): arrays}."""
    recs = {}
    for npz in sorted(Path(science_dir).rglob("recording-v1.npz")):
        with np.load(npz, allow_pickle=False) as z:
            d = {k: _bf16_void_to_f32(z[k]) for k in z.files}
        rid = str(d["row_id"]); noise = int(d["noise_label"]); role = str(d["arm_role"])
        recs[(rid, noise, role)] = d
    return recs


def assemble(recs, coverage):
    """coverage: {"<row_id>|<noise>": {clean_id, task_id, family, suite}} (per variant,noise)."""
    pairs = {}
    for (rid, noise, role), d in recs.items():
        pairs.setdefault((rid, noise), {})[role] = d
    rows, HD, CH, FD, TG = [], [], [], [], []
    dropped = []
    for (rid, noise), arms in sorted(pairs.items()):
        if "clean" not in arms or "perturbed" not in arms:
            dropped.append({"row_id": rid, "noise_label": noise, "have": sorted(arms)}); continue
        covkey = f"{rid}|{noise}"
        if covkey not in coverage:
            dropped.append({"row_id": rid, "noise_label": noise, "reason": "no_coverage"}); continue
        R, T = arms["clean"], arms["perturbed"]
        HR = np.asarray(R["hidden"], np.float64)[0, :, 0][:, -POS:, :]      # [3,POS,1024]
        HT = np.asarray(T["hidden"], np.float64)[0, :, 0][:, -POS:, :]
        HD.append(HT - HR); CH.append(HR)
        FD.append((np.asarray(T["velocity"], np.float64)[0, 0] - np.asarray(R["velocity"], np.float64)[0, 0]).reshape(POS * 32))
        raw = (np.asarray(T["raw_actions"], np.float64) - np.asarray(R["raw_actions"], np.float64)).reshape(POS * 32)
        adp = (np.asarray(T["actions_adapter"], np.float64) - np.asarray(R["actions_adapter"], np.float64)).reshape(POS * 7)
        TG.append(np.concatenate((raw, adp)))
        cov = coverage[covkey]
        rows.append({"case_id": f"{rid}|n{noise}", "row_id": rid, "noise_label": noise,
                     "task_id": cov["task_id"], "family": cov["family"], "suite": cov["suite"],
                     "clean_id": cov["clean_id"]})
    features = {"hidden_delta": np.stack(HD), "clean_hidden": np.stack(CH),
                "flow0_delta": np.stack(FD), "target": np.stack(TG)}
    return features, rows, dropped


def run(features, rows):
    components = L.task_components(rows)
    meta = {r["case_id"]: r for r in rows}
    all_cases, all_geom, cells, skipped = [], [], [], []
    for family in FAMILIES:
        for layer in LAYERS:
            for fold in range(5):
                try:
                    tr, te, _ = L.split_rows(rows, family, RO.MODE, fold, components)
                except Exception as e:
                    skipped.append({"family": family, "layer": layer, "fold": fold, "reason": str(e)[:60]}); continue
                if not tr or not te:
                    skipped.append({"family": family, "layer": layer, "fold": fold, "reason": "empty_split"}); continue
                tr_rows = [rows[i] for i in tr]; te_rows = [rows[i] for i in te]
                slot = LAYERS.index(layer)
                new = BZ.fit_new_bases(features["hidden_delta"][tr][:, slot],
                                       features["clean_hidden"][tr][:, slot], tr_rows, layer, POS)
                support = {"family": family, "mode": RO.MODE, "fold": fold, "layer": layer,
                           "training_case_count": len(tr), "test_case_count": len(te),
                           "training_case_ids_sha256": L.case_set_sha(tr_rows),
                           "test_case_ids_sha256": L.case_set_sha(te_rows)}
                cases, geom, _ = RO.predict_split(features, rows, components, {"support": support}, None, new)
                all_cases += cases; all_geom += geom
                cells.append({"family": family, "layer": layer, "fold": fold,
                              "train": len(tr), "test": len(te)})
    allowed = set(meta)
    task_rows = RO.task_rows(all_cases, meta, components, allowed)
    estimates, contrasts = RO.summary(task_rows)
    gtasks, gsumm = RO.geometry_summary(all_geom, meta, components, allowed)
    return {"n_pairs": len(rows), "n_cells_realized": len(cells), "cells": cells, "skipped_cells": skipped,
            "n_task_components": len(set(components.values())),
            "estimates": estimates, "contrasts": contrasts, "retention": gsumm}


# --------- synthetic self-test (no cloud/model) ---------
def _synthetic(seed=3):
    import hashlib
    rng = np.random.default_rng(seed)
    cov = {}; recs = {}
    # enough variants per family/fold for non-empty held-out splits
    for fam in FAMILIES:
        for k in range(14):
            rid = f"var-{fam[:2]}-{k}"
            task = hashlib.sha256((rid + "task").encode()).hexdigest()
            for noise in (0, 1):
                cov[f"{rid}|{noise}"] = {"clean_id": f"clean-{fam[:2]}-{k // 2}-{noise}", "task_id": task,
                                         "family": fam, "suite": "libero_10"}
                HR = 0.05 * rng.standard_normal((10, 3, 1, 51, 1024)).astype(np.float32)
                resp = 0.05 * rng.standard_normal((10, 3, 1, 51, 1024)).astype(np.float32)
                for role, hid in (("clean", HR), ("perturbed", HR + resp)):
                    recs[(rid, noise, role)] = {
                        "hidden": hid, "velocity": 0.05 * rng.standard_normal((10, 1, 50, 32)).astype(np.float32),
                        "raw_actions": 0.05 * rng.standard_normal((50, 32)).astype(np.float32),
                        "actions_adapter": 0.05 * rng.standard_normal((50, 7)),
                        "row_id": np.asarray(rid), "noise_label": np.asarray(noise),
                        "arm_role": np.asarray(role)}
    return recs, cov


def _self_test():
    recs, cov = _synthetic()
    features, rows, dropped = assemble(recs, cov)
    assert features["hidden_delta"].shape[1:] == (3, 50, 1024), features["hidden_delta"].shape
    assert features["target"].shape[1] == 1950 and features["flow0_delta"].shape[1] == 1600
    out = run(features, rows)
    assert out["n_cells_realized"] >= 20, out["n_cells_realized"]
    assert out["estimates"] and all(np.isfinite(e["task_equal_mse"]) for e in out["estimates"])
    models = {e["model"] for e in out["estimates"]}
    assert {"new_response", "new_clean", "new_random", "magnitude", "full", "flow", "flow_plus_new", "zero", "train_mean"} <= models, models
    cpairs = {(c["candidate"], c["baseline"]) for c in out["contrasts"]}
    assert ("new_response", "magnitude") in cpairs and ("flow_plus_new", "flow") in cpairs and \
           not any(b == "old_response" for _, b in cpairs), cpairs
    assert out["retention"] and {g["model"] for g in out["retention"]} == {"new_response", "new_clean", "new_random"}
    print(f"D self-test PASS: pairs={out['n_pairs']} cells={out['n_cells_realized']} "
          f"estimates={len(out['estimates'])} contrasts={len(out['contrasts'])} retention={len(out['retention'])} "
          f"dropped={len(dropped)}")


if __name__ == "__main__":
    _self_test()
