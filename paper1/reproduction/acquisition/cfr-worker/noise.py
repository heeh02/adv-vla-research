"""Per-query explicit policy noise shared by all arms of a unit and label (TECH-CONTRACT-v1 3.4).

noise = default_rng(SeedSequence([TAG, family_idx, suite_idx, classification_id, init_idx,
                                  noise_label, q])).standard_normal((1,10,32)).astype(float32)
This extends the tuple pattern of A/adapter.py:48-54 with family, init and label; it never
uses campaign_store.episode_noise (10*suite+task packing collides for Plus ids). The
generator is stateless with respect to the process-global NumPy/Python RNG used by the
LIBERO-Plus sensor corruption. TAG for 1A is a technical tag; the formal TAG is frozen in 1B.
"""

from __future__ import annotations

import hashlib

import numpy as np

FAMILY_INDEX = {"Camera Viewpoints": 0, "Sensor Noise": 1}
SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
TAG_STAGE_1A = 2026101901  # technical tag for Stage 1A only (1B freezes the formal TAG)


def query_noise(*, tag, family, suite, classification_id, init_index, noise_label, query):
    for value in (tag, classification_id, init_index, noise_label, query):
        if type(value) is not int or value < 0:
            raise ValueError("noise_tuple_requires_nonnegative_ints")
    entropy = [tag, FAMILY_INDEX[family], SUITES.index(suite), classification_id, init_index, noise_label, query]
    value = np.random.default_rng(np.random.SeedSequence(entropy)).standard_normal((1, 10, 32)).astype(np.float32)
    return value


def noise_sha(value):
    a = np.ascontiguousarray(value)
    if a.shape != (1, 10, 32) or a.dtype != np.float32:
        raise ValueError("noise_ABI")
    return hashlib.sha256(a.tobytes()).hexdigest()


def bind_schedule(*, tag, family, suite, classification_id, init_index, noise_label, horizon_queries):
    """Hash schedule for the whole horizon, bound before any policy call (runner.py:313-331 pattern)."""
    hashes = [noise_sha(query_noise(tag=tag, family=family, suite=suite, classification_id=classification_id,
                                    init_index=init_index, noise_label=noise_label, query=q))
              for q in range(horizon_queries)]
    return {"schema": "cf-repair-noise-schedule-v1", "tag": tag, "family": family, "family_index": FAMILY_INDEX[family],
            "suite": suite, "suite_index": SUITES.index(suite), "classification_id": classification_id,
            "init_index": init_index, "noise_label": noise_label, "horizon_queries": horizon_queries,
            "noise_sha256_by_query": hashes, "generator": "numpy.default_rng(SeedSequence(tuple)).standard_normal",
            "process_global_rng_used": False}


def checked_noise(schedule, query):
    value = query_noise(tag=schedule["tag"], family=schedule["family"], suite=schedule["suite"],
                        classification_id=schedule["classification_id"], init_index=schedule["init_index"],
                        noise_label=schedule["noise_label"], query=query)
    if noise_sha(value) != schedule["noise_sha256_by_query"][query]:
        raise ValueError("noise_differs_from_bound_schedule")
    return value
