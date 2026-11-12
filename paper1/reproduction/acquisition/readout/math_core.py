"""Pure fixed-rank target-visible subspace diagnostics; no model or cloud I/O."""
from __future__ import annotations
from collections import defaultdict
import hashlib
import math
import numpy as np

RANK = 16
LAYERS = (0, 8, 17)
LEVELS = ('task_id', 'family', 'variant_id', 'noise_id', 'token_position')

def require(ok, code):
    if not ok: raise ValueError(code)

def hierarchical_weights(rows):
    """Equal task, then family, variant, noise and token weight on available rows."""
    require(bool(rows), 'nonempty_training_rows')
    keys = [tuple(row[key] for key in LEVELS) for row in rows]
    require(len(keys) == len(set(keys)), 'unique_task_family_variant_noise_token')
    counts = [defaultdict(set) for _ in LEVELS]
    for key in keys:
        for depth in range(len(LEVELS)):
            counts[depth][key[:depth]].add(key[depth])
    weights = []
    for key in keys:
        w = 1.0
        for depth in range(len(LEVELS)):
            w /= len(counts[depth][key[:depth]])
        weights.append(w)
    result = np.asarray(weights, np.float64)
    require(np.isfinite(result).all() and np.all(result > 0)
            and math.isclose(float(result.sum()), 1.0, rel_tol=0, abs_tol=1e-12),
            'normalized_nested_weights')
    return result

def matrices(delta, clean, weights):
    x, r, w = np.asarray(delta, np.float64), np.asarray(clean, np.float64), np.asarray(weights, np.float64)
    require(x.ndim == r.ndim == 2 and x.shape == r.shape and x.shape[0] > 0 and x.shape[1] > 0,
            'same_native_layer_channel_ABI')
    require(w.shape == (x.shape[0],) and np.isfinite(x).all() and np.isfinite(r).all()
            and np.isfinite(w).all() and np.all(w > 0) and float(w.sum()) > 0,
            'finite_same_information_and_positive_weights')
    return x, r, w / w.sum()

def _fix_sign(basis):
    b = np.array(basis, dtype=np.float64, copy=True)
    for col in range(b.shape[1]):
        pivot = np.argmax(np.abs(b[:, col]))
        if b[pivot, col] < 0: b[:, col] *= -1
    return b

def uncentered_pca_fit(values, weights, rank=RANK):
    x = np.asarray(values, np.float64); w = np.asarray(weights, np.float64)
    require(x.ndim == 2 and x.shape[0] == len(w) and x.shape[0] > 0 and 0 < rank <= x.shape[1],
            'fixed_rank_training_ABI')
    require(np.isfinite(x).all() and np.isfinite(w).all() and np.all(w > 0), 'finite_PCA_training')
    if x.shape[0] < rank:
        return {'status': 'unavailable_effective_rank_below_fixed16',
                'effective_rank': int(np.linalg.matrix_rank(np.sqrt(w[:, None]) * x)),
                'requested_rank': rank, 'basis': None}
    covariance = x.T @ (w[:, None] * x)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1][:rank]
    # Rank is scale relative: a fixed absolute floor would reject valid small
    # BF16-decoded response vectors solely because their physical scale is low.
    tolerance = max(0.0, float(eigenvalues[order[0]])) * 1e-12
    effective_rank = int(np.count_nonzero(eigenvalues > tolerance))
    if effective_rank < rank:
        return {'status': 'unavailable_effective_rank_below_fixed16', 'effective_rank': effective_rank,
                'requested_rank': rank, 'basis': None}
    return {'status': 'measured', 'effective_rank': effective_rank, 'requested_rank': rank,
            'basis': _fix_sign(eigenvectors[:, order])}

def uncentered_pca_basis(values, weights, rank=RANK):
    fit = uncentered_pca_fit(values, weights, rank)
    require(fit['status'] == 'measured', 'training_effective_rank_below_fixed_budget')
    return fit['basis']

def top_coordinate_basis(delta, weights, rank=RANK):
    x = np.asarray(delta, np.float64); w = np.asarray(weights, np.float64)
    require(x.ndim == 2 and w.shape == (x.shape[0],) and 0 < rank <= x.shape[1]
            and np.isfinite(x).all() and np.isfinite(w).all() and np.all(w > 0), 'top_coordinate_training_ABI')
    energy = (w[:, None] * np.square(x)).sum(axis=0)
    order = np.lexsort((np.arange(x.shape[1]), -energy))[:rank]
    return np.eye(x.shape[1], dtype=np.float64)[:, order]

def fixed_random_basis(channels, layer, rank=RANK, seed_text='plus-shared-channel-rank16-v1'):
    require(layer in LAYERS and 0 < rank <= channels, 'fixed_random_layer_rank')
    seed = int.from_bytes(hashlib.sha256((seed_text + '|' + str(layer)).encode()).digest()[:8], 'little')
    gaussian = np.random.default_rng(seed).standard_normal((channels, rank))
    q, _ = np.linalg.qr(gaussian, mode='reduced')
    return _fix_sign(q)

def four_bases(train_delta, train_clean, weights, layer, rank=RANK):
    x, r, w = matrices(train_delta, train_clean, weights)
    require(rank == RANK and layer in LAYERS and x.shape[1] == 1024,
            'fixed_native_rank16_layer1024')
    return {'delta_uncentered_PCA': uncentered_pca_basis(x, w, rank),
            'delta_top16_native_coordinates': top_coordinate_basis(x, w, rank),
            'fixed_seed_random_orthogonal': fixed_random_basis(x.shape[1], layer, rank),
            'same_rows_clean_uncentered_PCA': uncentered_pca_basis(r, w, rank)}

def four_bases_status(train_delta, train_clean, weights, layer):
    """Keep each fixed budget; a low-rank PCA is unavailable, never reduced."""
    x, r, w = matrices(train_delta, train_clean, weights)
    require(layer in LAYERS and x.shape[1] == 1024,
            'fixed_native_rank16_layer1024')
    delta_fit = uncentered_pca_fit(x, w, RANK)
    clean_fit = uncentered_pca_fit(r, w, RANK)
    result = {'delta_uncentered_PCA': delta_fit,
              'same_rows_clean_uncentered_PCA': clean_fit,
              'delta_top16_native_coordinates': {'status': 'measured', 'effective_rank': RANK,
                                                  'requested_rank': RANK,
                                                  'basis': top_coordinate_basis(x, w, RANK)},
              'fixed_seed_random_orthogonal': {'status': 'measured', 'effective_rank': RANK,
                                               'requested_rank': RANK,
                                               'basis': fixed_random_basis(x.shape[1], layer, RANK)}}
    return result

def dictionary_cost(name, channels=1024, rank=RANK):
    require(name in ('delta_uncentered_PCA','same_rows_clean_uncentered_PCA',
                     'delta_top16_native_coordinates','fixed_seed_random_orthogonal')
            and channels == 1024 and rank == RANK, 'fixed_dictionary_budget')
    learned = channels * rank if name.endswith('PCA') else 0
    return {'coefficient_count_per_token': rank, 'dense_matrix_entries_if_materialized': channels * rank,
            'dense_float64_bytes_if_materialized': channels * rank * 8,
            'learned_float64_dictionary_entries': learned,
            'native_channel_ids_stored': rank if name == 'delta_top16_native_coordinates' else 0,
            'fixed_seed_only': name == 'fixed_seed_random_orthogonal',
            'same_coefficient_budget_not_same_dictionary_parameter_cost': True}

def reconstruct_visible_target(target, basis):
    x, b = np.asarray(target, np.float64), np.asarray(basis, np.float64)
    require(x.ndim == 2 and b.ndim == 2 and x.shape[1] == b.shape[0]
            and np.isfinite(x).all() and np.isfinite(b).all()
            and np.allclose(b.T @ b, np.eye(b.shape[1]), rtol=1e-9, atol=1e-9),
            'orthonormal_same_layer_projection')
    return (x @ b) @ b.T

def visible_reconstruction_score(target, basis, weights):
    x = np.asarray(target, np.float64); w = np.asarray(weights, np.float64)
    require(x.ndim == 2 and w.shape == (x.shape[0],) and np.isfinite(w).all()
            and np.all(w > 0), 'test_weight_ABI')
    reconstruction = reconstruct_visible_target(x, basis)
    residual = x - reconstruction
    target_energy = float(np.sum(w[:, None] * np.square(x)))
    residual_energy = float(np.sum(w[:, None] * np.square(residual)))
    normalizer = float(w.sum()) * x.shape[1]
    return {'rank': int(basis.shape[1]), 'channels': int(x.shape[1]),
            'weighted_target_energy': target_energy, 'weighted_residual_energy': residual_energy,
            'target_rms': math.sqrt(target_energy / normalizer),
            'residual_rms': math.sqrt(residual_energy / normalizer),
            'explained_energy_fraction': 1 - residual_energy / target_energy if target_energy > 0 else None,
            'target_visible_at_coefficient_fit': True, 'predictive_or_harmful_subspace_claim': False}
