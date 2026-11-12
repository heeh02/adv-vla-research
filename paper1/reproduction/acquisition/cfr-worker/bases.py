"""I-1 bases bundle, I-2 unit map and the per-unit identity chain (plan 3.1 item 6).

New code. Interfaces: stage1a/INTERFACES-v1.md (I-1, I-2). The identity chain is checked
independently of the projection identities:
  task/family -> fold (SPLIT-FROZEN) -> expected fit SHA (SPLIT-FROZEN) -> manifest source
  fit SHA -> manifest array SHA -> loaded array SHA -> layer/token support (shapes).
Array identity domain: SHA-256 over C-contiguous little-endian float64 bytes.
Bases are consumed as FP32 copies; the FP32 copy SHA and the measured Gram errors are
recorded (TECH-CONTRACT-v1 4.2 "basis payload checks").
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .common import TechnicalFailure, array_id, file_identity, float64_le_sha, need
from .repair_math import BASIS_ORDER, gram_error, gram_max_abs

SPLIT_SHA256 = "0ec0748d86ddc59039585ad3a1442f1ca3bbb0bfcc06ede1670806d01f4b0e5c"
FAMILY_KEY = {"Camera Viewpoints": "camera", "Sensor Noise": "sensor"}
LAYER_STORED_INDEX = 8
G_SEED_TEXTS = {"G2": "counterfactual-repair-v1-random-2", "G3": "counterfactual-repair-v1-random-3"}


def load_split(path):
    ident = file_identity(path)
    need(ident["sha256"] == SPLIT_SHA256, "SPLIT_FROZEN_identity_differs")
    return json.loads(Path(path).read_bytes()), ident


def _entry(container, name):
    if isinstance(container, dict):
        return container.get(name)
    if isinstance(container, list):
        rows = [row for row in container if isinstance(row, dict) and (row.get("name") == name or row.get("key") == name)]
        return rows[0] if len(rows) == 1 else None
    return None


def _first(row, *keys):
    for key in keys:
        if isinstance(row, dict) and row.get(key) is not None:
            return row[key]
    return None


def manifest_array(manifest, name):
    """Return (expected_array_sha256, source_fit_ref) for one array of the I-1 manifest."""
    row = _entry(manifest.get("arrays"), name)
    need(row is not None, "I1_manifest_lacks_array:" + name)
    expected = _first(row, "array_sha256", "sha256_float64_le", "sha256")
    source = _first(row, "source", "source_fit", "fit")
    need(isinstance(expected, str) and len(expected) == 64, "I1_manifest_array_sha_missing:" + name)
    return expected, source if isinstance(source, dict) else None


def regenerate_fixed_random(seed_text, channels=1024, layer=LAYER_STORED_INDEX, rank=16):
    """Diagnostic copy of mean_math.fixed_random (mean_math.py:31-34, SHA 53a397c5...)."""
    seed = int.from_bytes(hashlib.sha256((seed_text + "|" + str(layer)).encode()).digest()[:8], "little")
    z = np.random.default_rng(seed).standard_normal((channels, rank))
    q, _ = np.linalg.qr(z, mode="reduced")
    b = np.array(q, copy=True)
    for col in range(b.shape[1]):
        if b[np.argmax(abs(b[:, col])), col] < 0:
            b[:, col] *= -1
    return b


class BasesBundle:
    """Loaded I-1 arrays with identity proofs; never fetches anything."""

    def __init__(self, npz_path, manifest_path):
        self.npz_identity = file_identity(npz_path)
        self.manifest_identity = file_identity(manifest_path)
        self.manifest = json.loads(Path(manifest_path).read_bytes())
        npz_ref = self.manifest.get("npz") or self.manifest.get("bundle") or {}
        if isinstance(npz_ref, dict) and npz_ref.get("sha256"):
            need(npz_ref["sha256"] == self.npz_identity["sha256"]
                 and npz_ref.get("bytes", self.npz_identity["bytes"]) == self.npz_identity["bytes"],
                 "I1_npz_identity_differs_from_manifest")
        with np.load(npz_path, allow_pickle=False) as data:
            self.members = sorted(data.files)
            self.arrays = {name: np.array(data[name], copy=True) for name in data.files}
        self.loaded = {}

    def array(self, name, shape):
        need(name in self.arrays, "I1_npz_lacks_member:" + name)
        a = self.arrays[name]
        need(a.dtype == np.float64 and a.shape == shape and np.isfinite(a).all(), "I1_array_ABI:" + name)
        expected, source = manifest_array(self.manifest, name)
        loaded = float64_le_sha(a)
        need(loaded == expected, "I1_loaded_array_sha_differs:" + name)
        f32 = a.astype(np.float32)
        proof = {"name": name, "expected_array_sha256": expected, "loaded_array_sha256": loaded,
                 "source_fit": source, "shape": list(a.shape), "dtype": "float64",
                 "fp32_consumed": array_id(f32)}
        if shape == (1024, 16):
            proof.update(gram_max_abs_f64=gram_max_abs(a), gram_spectral_f64=gram_error(a),
                         gram_max_abs_fp32=gram_max_abs(f32), gram_spectral_fp32=gram_error(f32))
        self.loaded[name] = proof
        return f32, proof

    def random_bases(self):
        out, proofs = {}, {}
        for name in ("G1", "G2", "G3"):
            out[name], proofs[name] = self.array(name, (1024, 16))
            if name in G_SEED_TEXTS:
                regen = regenerate_fixed_random(G_SEED_TEXTS[name])
                proofs[name]["regenerated_in_job_max_abs_diff"] = float(np.max(np.abs(regen - self.arrays[name])))
                proofs[name]["regenerated_in_job_byte_equal"] = bool(regen.tobytes() == self.arrays[name].astype("<f8").tobytes())
        return out, proofs


def family_arrays(family, fold):
    key = FAMILY_KEY[family]
    return {"B": f"B__{key}__{fold}", "C": f"C__{key}__{fold}", "M": f"M__{key}__{fold}"}


def normalize_unit(row):
    """Normalise one I-2 UNIT-MAP row; records which alias supplied each field."""
    variant = _first(row, "official_variant_name", "variant_name", "bddl_name", "variant")
    arrays = _first(row, "arrays", "array_keys")
    unit = {"row_id": row.get("row_id"), "task_id": _first(row, "task_id", "task"),
            "family": row.get("family"), "suite": row.get("suite"), "variant_name": variant,
            "fold": _first(row, "fold", "basis_fold"), "arrays": arrays,
            "classification_id": _first(row, "classification_id"), "transform": row.get("transform"),
            "kind": row.get("kind"), "problem_class": row.get("problem_class")}
    need(all(unit[k] is not None for k in ("row_id", "task_id", "family", "suite", "variant_name", "fold")),
         "I2_unit_row_incomplete:" + str(row.get("row_id")))
    need(unit["family"] in FAMILY_KEY, "I2_unknown_family")
    expected_arrays = family_arrays(unit["family"], unit["fold"])
    if arrays is None:
        unit["arrays"] = expected_arrays
    else:
        need({k: arrays.get(k) for k in ("B", "C", "M")} == expected_arrays, "I2_array_keys_differ_from_family_fold")
    return unit


def load_unit_map(path):
    raw = json.loads(Path(path).read_bytes())
    rows = raw.get("units") if isinstance(raw, dict) else raw
    if isinstance(rows, dict):
        rows = [{"row_id": key, **value} for key, value in rows.items()]
    need(isinstance(rows, list) and rows, "I2_units_required")
    units = [normalize_unit(row) for row in rows]
    need(len({u["row_id"] for u in units}) == len(units), "I2_duplicate_row_id")
    return units, file_identity(path)


def identity_chain(unit, split, bundle, *, split_role="dev"):
    """Run the per-unit identity chain; returns (bases dict FP32, template FP32, record)."""
    family = unit["family"]
    part = split["families"][family][split_role]
    need(unit["row_id"] in part["row_ids"], "identity_chain_row_not_in_split_role:" + split_role)
    task = unit["task_id"]
    need(task in part["basis_by_task"], "identity_chain_task_not_in_split_role")
    mapping = part["basis_by_task"][task]
    need(int(mapping["fold"]) == int(unit["fold"]), "identity_chain_fold_differs_from_split")
    expected_fit = mapping["fit"]
    names = family_arrays(family, unit["fold"])
    b, b_proof = bundle.array(names["B"], (1024, 16))
    c, c_proof = bundle.array(names["C"], (1024, 16))
    m, m_proof = bundle.array(names["M"], (10, 1024))
    for proof in (b_proof, c_proof, m_proof):
        source = proof["source_fit"]
        need(source is not None and source.get("sha256") == expected_fit["sha256"],
             "identity_chain_source_fit_differs:" + proof["name"])
    record = {"row_id": unit["row_id"], "task_id": task, "family": family, "split_role": split_role,
              "fold": int(unit["fold"]), "expected_fit": expected_fit,
              "arrays": {"B": b_proof, "C": c_proof, "M": m_proof},
              "layer_and_token_support": {"stored_layer_index": LAYER_STORED_INDEX, "paper_layer": "L9",
                                          "site": "action_expert_post_block_H", "tokens": 10, "channels": 1024,
                                          "B_shape": b_proof["shape"], "M_shape": m_proof["shape"]},
              "pass": True}
    return {"B": b, "C": c}, m, record


def stack_bases(unit_bases, random_bases):
    """FP32 [5,1024,16] in BASIS_ORDER (B, C, G1, G2, G3)."""
    stack = np.stack([unit_bases["B"], unit_bases["C"], random_bases["G1"], random_bases["G2"], random_bases["G3"]])
    if stack.shape != (5, 1024, 16) or stack.dtype != np.float32:
        raise TechnicalFailure("five_FP32_bases_stack_ABI")
    return stack, dict(zip(BASIS_ORDER, stack))
