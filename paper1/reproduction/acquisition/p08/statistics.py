"""Bounded P08 geometry: pure CPU functions, no IO, model or cloud calls.

Fits use FP32 views of archived BF16 values. Statistical scalar reductions
may accumulate in float64; this is not new model precision or old native P08.
"""
from collections import Counter, defaultdict
import hashlib
import json
import numpy as np

LAYERS = (1, 9, 18)
RANK = 16
SEED = 20261012
TRAIN_FOLDS = (0, 1, 2)
TEST_FOLDS = (3, 4)
BOOTSTRAPS = 2000


def task_key(row):
    return str(row["suite"]) + ":" + str(row["task"])


def persistent(row):
    return row["history_condition"] == row["declared_current_condition"] != "clean"


def freeze_metadata(planned):
    """Use old ledger folds, all planned rows and outcome-blind example rules."""
    seen, tasks, episodes = set(), {}, defaultdict(list)
    for row in planned:
        if row["anchor_id"] in seen:
            raise ValueError("duplicate planned anchor")
        seen.add(row["anchor_id"])
        key, fold = task_key(row), int(row["task_fold"])
        if fold not in TRAIN_FOLDS + TEST_FOLDS:
            raise ValueError("unknown original task fold")
        info = {"task_key": key, "suite": row["suite"], "task": str(row["task"]),
                "task_fold": fold, "role": "train" if fold in TRAIN_FOLDS else "heldout"}
        if key in tasks and tasks[key] != info:
            raise ValueError("task fold changes across owners")
        tasks[key] = info
        episodes[row["episode_id"]].append(row)
    for group in episodes.values():
        if {int(r["slot"]) for r in group} != {1, 2} or len(group) != 2:
            raise ValueError("planned episode must retain both slots")
        for field in ("suite", "task", "task_fold", "history_condition", "root", "noise_repeat"):
            if len({str(r[field]) for r in group}) != 1:
                raise ValueError("episode pairing metadata differs")
    balance = {}
    for suite in sorted({r["suite"] for r in tasks.values()}):
        group = [r for r in tasks.values() if r["suite"] == suite]
        balance[suite] = {"train": sum(r["role"] == "train" for r in group),
                          "heldout": sum(r["role"] == "heldout" for r in group)}
    conditions = sorted({r["history_condition"] for r in planned if persistent(r)})
    examples = []
    for condition in conditions:
        candidates = [e for e, rs in episodes.items() if all(persistent(r) for r in rs)
                      and rs[0]["history_condition"] == condition and int(rs[0]["task_fold"]) in TEST_FOLDS]
        if not candidates:
            raise ValueError("no planned heldout episode for condition")
        chosen = min(candidates, key=lambda e: (hashlib.sha256(e.encode("utf-8")).hexdigest(), e))
        rows = sorted(episodes[chosen], key=lambda r: int(r["slot"]))
        examples.append({"condition": condition, "episode_id": chosen,
                         "selection_hash": hashlib.sha256(chosen.encode("utf-8")).hexdigest(),
                         "task_key": task_key(rows[0]),
                         "anchors": [{k: r.get(k) for k in ("anchor_id", "slot", "control_step", "query_index",
                                     "original_present", "extraction_status", "file_id", "selected_file_path",
                                     "selected_file_sha256")} for r in rows]})
    return {"schema": "p08-old-task-fold-split-v1", "tasks": [tasks[k] for k in sorted(tasks)],
            "suite_balance": balance, "fit_conditions": conditions,
            "planned_anchors": len(planned), "planned_episodes": len(episodes),
            "history_conditions": sorted({r["history_condition"] for r in planned}),
            "train_folds": list(TRAIN_FOLDS), "heldout_folds": list(TEST_FOLDS),
            "heatmap_examples": examples, "outcomes_used_for_selection": False}


def owner_weights(rows):
    """Equal task -> historical condition -> root/init -> original noise."""
    if not rows:
        raise ValueError("empty statistical support")
    leaves = {(): list(range(len(rows)))}
    w = np.ones(len(rows), np.float64)
    for field in ("task", "history_condition", "root", "noise_repeat"):
        new = {}
        for path, indices in leaves.items():
            children = defaultdict(list)
            for i in indices:
                key = task_key(rows[i]) if field == "task" else str(rows[i][field])
                children[key].append(i)
            for key, ids in children.items():
                w[ids] /= len(children)
                new[path + (key,)] = ids
        leaves = new
    for ids in leaves.values():
        if len(ids) != 1:
            raise ValueError("duplicate owner or mixed slots in weighting support")
    if not np.isclose(w.sum(), 1):
        raise ValueError("unnormalized owner weights")
    return w


def bf16_bits_to_float32(bits):
    a = np.asarray(bits)
    if a.dtype.kind != "u" or a.dtype.itemsize != 2:
        raise ValueError("BF16 requires original uint16 bit payload")
    return (a.astype(np.uint32) << 16).view(np.float32)


def _array(x):
    a = np.asarray(x)
    if a.dtype.kind not in "f" and a.dtype.name != "bfloat16":
        raise ValueError("decode logical BF16 before statistical calls")
    a = a.astype(np.float32, copy=False)
    if not np.isfinite(a).all():
        raise ValueError("nonfinite stored tensor")
    return a


def _sha(a):
    return hashlib.sha256(np.ascontiguousarray(a, dtype="<f4").tobytes()).hexdigest()


def _canon_sign(b):
    b = b.copy()
    for j in range(b.shape[1]):
        pivot = int(np.argmax(np.abs(b[:, j])))
        if b[pivot, j] < 0:
            b[:, j] *= np.float32(-1)
    return b


def _pca(x, weights, rank):
    """Position-specific center, pooled channel scatter, FP32 linear algebra."""
    w = np.asarray(weights, np.float32)
    mu = np.einsum("n,npd->pd", w, x, dtype=np.float32)
    residual = x - mu
    matrix = (residual * np.sqrt(w[:, None, None] / np.float32(x.shape[1]))).reshape(-1, x.shape[-1])
    scatter = matrix.T @ matrix
    values, vectors = np.linalg.eigh(scatter)
    order = np.argsort(values)[::-1]
    values, vectors = values[order], vectors[:, order]
    tolerance = float(np.finfo(np.float32).eps * x.shape[-1] * max(float(values[0]), 0))
    numerical_rank = int(np.count_nonzero(values > tolerance))
    b = _canon_sign(vectors[:, :rank]) if numerical_rank >= rank else None
    return mu, b, {"numerical_rank": numerical_rank, "rank_requested": rank,
                   "status": "fitted" if b is not None else "insufficient_numerical_rank",
                   "rank_tolerance": tolerance, "raw_min_eigenvalue": float(values[-1]),
                   "top_eigenvalues": [float(v) for v in values[:rank+1]],
                   "scatter_trace": float(np.trace(scatter, dtype=np.float64))}


def fit_bases(owners, hidden_delta, hidden_clean, split, rank=RANK, layers=LAYERS):
    """Fit only persistent train-task slot1; ignore all heldout payloads.

    Arrays correspond to available owners. Planned missing owners remain in
    split/support metadata outside this function; no imputation or reallocation.
    """
    delta, clean = _array(hidden_delta), _array(hidden_clean)
    if delta.shape != clean.shape or delta.ndim != 4 or delta.shape[:2] != (len(owners), len(layers)):
        raise ValueError("expected owners x layers x positions x channels")
    if rank < 3 or rank > delta.shape[-1]:
        raise ValueError("rank must support display3 and channel width")
    declared = {r["task_key"]: r for r in split["tasks"]}
    for row in owners:
        key = task_key(row)
        if key not in declared or int(row["task_fold"]) != declared[key]["task_fold"]:
            raise ValueError("owner not in frozen original fold split")
    if len({r["anchor_id"] for r in owners}) != len(owners):
        raise ValueError("duplicate available owner")
    selected = [i for i, r in enumerate(owners) if int(r["slot"]) == 1 and persistent(r)
                and declared[task_key(r)]["role"] == "train"]
    selected.sort(key=lambda i: owners[i]["anchor_id"])
    selected_rows = [owners[i] for i in selected]
    w = owner_weights(selected_rows)
    if set(r["history_condition"] for r in selected_rows) != set(split["fit_conditions"]):
        raise ValueError("training support lost a complete declared condition")
    fit_ids = sorted(r["anchor_id"] for r in selected_rows)
    fit_id_sha = hashlib.sha256(json.dumps(fit_ids, separators=(",", ":")).encode()).hexdigest()
    result = {"rank": rank, "layers": {}, "fit_owner_ids": fit_ids,
              "fit_owner_ids_sha256": fit_id_sha, "fit_owner_count": len(selected),
              "fit_task_keys": sorted({task_key(r) for r in selected_rows}),
              "declared_train_task_keys": sorted(k for k, r in declared.items() if r["role"] == "train"),
              "declared_heldout_task_keys": sorted(k for k, r in declared.items() if r["role"] == "heldout"),
              "numeric_role": "FP32 geometry of archived BF16 values; float64 scalar accumulation",
              "fit_weight_hierarchy": "task-condition-root-noise-position", "seed": SEED}
    for li, layer in enumerate(layers):
        mu, response, response_meta = _pca(delta[selected, li], w, rank)
        clean_mu, clean_basis, clean_meta = _pca(clean[selected, li], w, rank)
        # Exactly the fixed seed for each separate layer: matched random axes,
        # without choosing among seeds or changing them after seeing a spectrum.
        random = np.random.default_rng(SEED).standard_normal((delta.shape[-1], rank), dtype=np.float32)
        random = _canon_sign(np.linalg.qr(random, mode="reduced")[0])
        bases = {"response": response, "clean": clean_basis, "random": random}
        basis_meta = {}
        for name, b in bases.items():
            basis_meta[name] = {"status": "fitted" if b is not None else "insufficient_numerical_rank",
                                "sha256": _sha(b) if b is not None else None,
                                "shape": [delta.shape[-1], rank] if b is not None else None,
                                "orthogonality_max_error": float(np.max(np.abs(b.T @ b - np.eye(rank)))) if b is not None else None}
        result["layers"][str(layer)] = {"center": mu, "clean_fit_center": clean_mu, "bases": bases,
            "metadata": {"layer": layer, "object": "native_AE_hidden_delta_at_j0",
                         "position_center_sha256": _sha(mu), "position_center_shape": list(mu.shape),
                         "scale": "none", "sign_rule": "largest absolute loading positive; lowest channel wins tie",
                         "fit_slot": 1, "fit_owner_ids_sha256": fit_id_sha, "rank": rank, "display_rank": 3,
                         "basis": basis_meta, "response_fit": response_meta, "clean_fit": clean_meta,
                         "reconstruction_center": "same response position mean for all controls",
                         "clean_center_role": "fit clean directions only; never reconstruct response with clean mean"}}
    return result


def _energy(x):
    a = np.asarray(x, np.float32)
    return float(np.sum(a.astype(np.float64) ** 2))


def coordinate_pr(x):
    """Coordinate energy PR=(sum x_j^2)^2/sum x_j^4, not intrinsic dimension."""
    a = np.asarray(x, np.float64).ravel()
    square = a * a
    energy = float(square.sum())
    return energy * energy / float(square @ square) if energy else None


def signed_cosine(a, b):
    a, b = _array(a), _array(b)
    if a.shape != b.shape:
        raise ValueError("cosine shapes differ")
    na, nb = _energy(a), _energy(b)
    if not na or not nb:
        return {"value": None, "reason": "one_or_both_zero", "norm0": np.sqrt(na).item(), "norm1": np.sqrt(nb).item()}
    value = float(np.sum(a.astype(np.float64) * b.astype(np.float64)) / np.sqrt(na * nb))
    return {"value": value, "reason": "", "norm0": np.sqrt(na).item(), "norm1": np.sqrt(nb).item()}


def _projection_metrics(target, center, b):
    centered = target - center
    raw_energy, residual_energy = _energy(target), _energy(centered)
    if b is None:
        return {"status": "basis_unavailable", "sse": None, "retention_centered": None,
                "one_minus_raw_normalized_error": None, "squared_error_over_clean_energy": None}
    coordinates = centered @ b
    reconstruction = center + coordinates @ b.T
    error = _energy(target - reconstruction)
    return {"status": "computed", "sse": error,
            "retention_centered": 1 - error / residual_energy if residual_energy else None,
            "one_minus_raw_normalized_error": 1 - error / raw_energy if raw_energy else None}


def describe_owner(row, delta, clean, fit):
    """Small coordinates and scalars for every available owner; no tensors out."""
    delta, clean = _array(delta), _array(clean)
    layers = list(fit["layers"])
    if delta.shape != clean.shape or delta.shape[0] != len(layers):
        raise ValueError("owner shape differs")
    if task_key(row) not in fit["declared_train_task_keys"] + fit["declared_heldout_task_keys"]:
        raise ValueError("readout task not in frozen split")
    role = "train" if task_key(row) in fit["declared_train_task_keys"] else "heldout"
    output = {"anchor_id": row["anchor_id"], "episode_id": row["episode_id"],
              "task_key": task_key(row), "history_condition": row["history_condition"],
              "declared_current_condition": row["declared_current_condition"],
              "slot": int(row["slot"]), "task_role": role,
              "evaluation_role": ("fit_descriptive" if role == "train" and int(row["slot"]) == 1 and persistent(row)
                                  else "primary_heldout_slot2" if role == "heldout" and int(row["slot"]) == 2 and persistent(row)
                                  else "matched_heldout_slot1" if role == "heldout" and int(row["slot"]) == 1 and persistent(row)
                                  else "train_slot2_descriptive" if role == "train" and persistent(row)
                                  else "clean_or_q0only_control"), "layers": {}}
    for li, layer in enumerate(layers):
        x, h = delta[li], clean[li]
        model = fit["layers"][layer]
        mu = model["center"]
        if x.shape != mu.shape:
            raise ValueError("position/channel schema differs from basis")
        raw, clean_energy = _energy(x), _energy(h)
        centered_energy = _energy(x - mu)
        channel_energy = np.sum(x.astype(np.float64) ** 2, axis=0)
        channel_pr = float(channel_energy.sum() ** 2 / (channel_energy @ channel_energy)) if raw else None
        metrics = {"raw_response_energy": raw, "clean_energy": clean_energy,
                   "response_rms": float(np.sqrt(raw / x.size)),
                   "relative_energy": raw / clean_energy if clean_energy else None,
                   "relative_rms": float(np.sqrt(raw / clean_energy)) if clean_energy else None,
                   "coordinate_PR_joint_position_channel": coordinate_pr(x),
                   "coordinate_PR_channel_energy_pooled_positions": channel_pr,
                   "centered_response_energy": centered_energy,
                   "mean_only_sse": centered_energy,
                   "mean_only_one_minus_raw_normalized_error": 1 - centered_energy / raw if raw else None,
                   "precision": "FP32 decoded values and projection; robust scalar reductions"}
        controls = {}
        for name, b in model["bases"].items():
            controls[name] = _projection_metrics(x, mu, b)
            e = controls[name]["sse"]
            controls[name]["squared_error_over_clean_energy"] = e / clean_energy if e is not None and clean_energy else None
        positions = []
        b = model["bases"]["response"]
        for p in range(x.shape[0]):
            e, ce = _energy(x[p]), _energy(h[p])
            full = _projection_metrics(x[p], mu[p], b)
            three = _projection_metrics(x[p], mu[p], b[:, :3] if b is not None else None)
            positions.append({"position": p, "xyz": ((x[p] - mu[p]) @ b[:, :3]).tolist() if b is not None else None,
                              "raw_energy": e, "relative_energy": e / ce if ce else None,
                              "coordinate_PR": coordinate_pr(x[p]),
                              "retention3": three["retention_centered"], "retention16": full["retention_centered"]})
        output["layers"][layer] = {"metrics": metrics, "controls": controls, "projection": {"points": positions}}
    return output


def describe_pair(first, second):
    """Paired sparse anchors; full and position-wise signed response cosine."""
    a, b = _array(first), _array(second)
    if a.shape != b.shape or a.ndim != 3 or a.shape[0] != len(LAYERS):
        raise ValueError("pair expects layers x positions x channels")
    return {str(layer): {"full_position_channel_cosine": signed_cosine(a[i], b[i]),
                        "position_cosines": [signed_cosine(a[i, p], b[i, p]) for p in range(a.shape[1])]}
            for i, layer in enumerate(LAYERS)}


def attention_summary(clean, actual, valid_clean, valid_actual, valid_delta):
    """Unconditional shadow camera mass and half-L1; not old conditioned TV."""
    c, a = _array(clean), _array(actual)
    if c.shape != a.shape or c.shape != (3, 8, 2, 16, 16) or np.any(c < 0) or np.any(a < 0):
        raise ValueError("invalid archived camera probability map")
    vc, va, vd = map(np.asarray, (valid_clean, valid_actual, valid_delta))
    if any(v.shape != (3, 8, 4) or v.dtype != bool for v in (vc, va, vd)) or not np.array_equal(vd, vc & va):
        raise ValueError("source validity intersection differs")
    cm, am = np.sum(c, (-2, -1), dtype=np.float64), np.sum(a, (-2, -1), dtype=np.float64)
    half_l1 = .5 * np.sum(np.abs(a.astype(np.float64) - c.astype(np.float64)), (-2, -1))
    heads, fixed = [], []
    for li, layer in enumerate(LAYERS):
        for camera, name in enumerate(("main", "wrist")):
            group = []
            for head in range(8):
                row = {"layer": layer, "camera": name, "head": head,
                       "camera_mass_clean": float(cm[li, head, camera]) if vc[li, head, camera] else None,
                       "camera_mass_actual": float(am[li, head, camera]) if va[li, head, camera] else None,
                       "delta_camera_mass": float(am[li, head, camera] - cm[li, head, camera]) if vd[li, head, camera] else None,
                       "unconditional_camera_half_L1": float(half_l1[li, head, camera]) if vd[li, head, camera] else None,
                       "clean_valid": bool(vc[li, head, camera]), "actual_valid": bool(va[li, head, camera])}
                heads.append(row); group.append(row)
            mean = {"layer": layer, "camera": name, "heads_required": 8,
                    "role": "shadow_unconditional_camera_map_first5_query_mean"}
            for field in ("camera_mass_clean", "camera_mass_actual", "delta_camera_mass", "unconditional_camera_half_L1"):
                good = [r[field] for r in group if r[field] is not None]
                mean[field] = float(np.mean(good)) if len(good) == 8 else None
                mean[field + "_valid_heads"] = len(good)
            fixed.append(mean)
    return {"heads": heads, "fixed8": fixed, "precision": "reference reductions of archived BF16 probability values",
            "within_camera_conditional_TV": "reuse published P08 table, separate statistic"}


def frozen_axis_limits(owner_readouts):
    """One display bound per layer/axis across all available owners/positions."""
    by = defaultdict(list)
    for row in owner_readouts:
        for layer, info in row["layers"].items():
            by[layer].extend(p["xyz"] for p in info["projection"]["points"] if p["xyz"] is not None)
    return {layer: [[-float(v), float(v)] for v in np.maximum(np.max(np.abs(np.asarray(x)), axis=0) * 1.02, 1e-12)]
            for layer, x in by.items() if x}


def summarize_fixed_metric(rows, field, planned_tasks, seed=SEED, repeats=BOOTSTRAPS):
    """One condition/layer/slot/control series; paired task bootstrap indices."""
    if len(set(planned_tasks)) != len(planned_tasks) or not planned_tasks:
        raise ValueError("frozen planned task list required")
    by = defaultdict(list)
    for row in rows:
        if task_key(row) not in planned_tasks:
            raise ValueError("row outside frozen reporting tasks")
        if row.get(field) is not None and np.isfinite(row[field]):
            by[task_key(row)].append(row)
    values = {t: float(owner_weights(g) @ np.asarray([r[field] for r in g])) for t, g in by.items()}
    tasks = sorted(planned_tasks)
    v = np.array([values.get(t, np.nan) for t in tasks])
    ix = np.random.default_rng(seed).integers(0, len(tasks), (repeats, len(tasks)))
    sampled = v[ix]; n = np.isfinite(sampled).sum(axis=1)
    boot = np.nansum(sampled, axis=1)[n > 0] / n[n > 0]
    defined = v[np.isfinite(v)]
    return {"mean": float(defined.mean()) if len(defined) else None,
            "interval95": np.quantile(boot, [.025, .975]).tolist() if len(defined) >= 2 and len(boot) else None,
            "task_values": values, "planned_tasks": len(tasks), "defined_tasks": len(values),
            "planned_rows": len(rows), "defined_rows": sum(len(g) for g in by.values()),
            "valid_bootstraps": len(boot), "bootstrap_repeats": repeats, "seed": seed,
            "interval_role": "descriptive held-task resampling conditional on fixed fitted basis"}
