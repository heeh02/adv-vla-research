"""Pure frozen S4 estimators extracted from the original analysis source.
No cloud, model, acquisition or private paths are included.
"""
from __future__ import annotations
from collections import Counter
import math
REPLICATES=2000
MIN_DEFINED_DRAWS_FOR_CI=1900

def need(condition: bool, code: str) -> None:
    if not condition:
        raise ValueError(code)

def midrank(values: list[float]) -> list[float]:
    """Average ties, then map to (rank-.5)/n; no jitter."""
    n = len(values)
    need(n > 0 and all((math.isfinite(x) for x in values)), 'finite_nonempty_rank_input')
    order = sorted(range(n), key=lambda i: values[i])
    result = [0.0] * n
    lo = 0
    while lo < n:
        hi = lo + 1
        while hi < n and values[order[hi]] == values[order[lo]]:
            hi += 1
        average_rank = (lo + 1 + hi) / 2
        score = (average_rank - 0.5) / n
        for j in range(lo, hi):
            result[order[j]] = score
        lo = hi
    return result

def task_weights(cases: list[dict]) -> list[float]:
    variants = Counter((c['row_id'] for c in cases))
    need(len(variants) > 0 and all((1 <= count <= 2 for count in variants.values())), 'one_or_two_noise_per_variant')
    weights = [1 / (len(variants) * variants[c['row_id']]) for c in cases]
    need(math.isclose(math.fsum(weights), 1, abs_tol=1e-12), 'noise_variant_weights_sum_one')
    return weights

def task_moments(cases: list[dict], x: str, endpoint: str) -> tuple[float, float, float]:
    need(len(cases) >= 2, 'within_task_at_least_two_cases')
    weights = task_weights(cases)
    ux = midrank([c[x] for c in cases])
    uy = midrank([c[endpoint] for c in cases])
    mx = math.fsum((w * v for w, v in zip(weights, ux)))
    my = math.fsum((w * v for w, v in zip(weights, uy)))
    covariance = math.fsum((w * (a - mx) * (b - my) for w, a, b in zip(weights, ux, uy)))
    xvariance = math.fsum((w * (a - mx) ** 2 for w, a in zip(weights, ux)))
    yvariance = math.fsum((w * (b - my) ** 2 for w, b in zip(weights, uy)))
    return (covariance, xvariance, yvariance)

def pooled_rho(moments: dict[str, tuple[float, float, float]], task_ids: list[str] | tuple[str, ...]) -> float | None:
    parts = [moments[t] for t in task_ids if t in moments]
    if not parts:
        return None
    c = math.fsum((v[0] for v in parts))
    vx = math.fsum((v[1] for v in parts))
    vy = math.fsum((v[2] for v in parts))
    if vx <= 0 or vy <= 0:
        return None
    result = c / math.sqrt(vx * vy)
    need(-1 - 1e-12 <= result <= 1 + 1e-12, 'rank_correlation_range')
    return max(-1.0, min(1.0, result))

def weighted_auc(cases: list[dict], x: str) -> float | None:
    positives = [(c, w) for c, w in zip(cases, task_weights(cases)) if c['Y_cell'] == 'SF']
    negatives = [(c, w) for c, w in zip(cases, task_weights(cases)) if c['Y_cell'] == 'SS']
    if not positives or not negatives:
        return None
    den_pos = math.fsum((w for _, w in positives))
    den_neg = math.fsum((w for _, w in negatives))
    numerator = math.fsum((wp * wn * (1.0 if p[x] > n[x] else 0.5 if p[x] == n[x] else 0.0) for p, wp in positives for n, wn in negatives))
    return numerator / (den_pos * den_neg)

def equal_task_auc(auc_by_task: dict[str, float], task_ids: list[str] | tuple[str, ...]) -> float | None:
    values = [auc_by_task[t] for t in task_ids if t in auc_by_task]
    return math.fsum(values) / len(values) if values else None

def percentile(values: list[float], probability: float) -> float:
    need(values and 0 <= probability <= 1, 'nonempty_percentile')
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    return ordered[lower] + (position - lower) * (ordered[upper] - ordered[lower])

def bootstrap_result(point: float | None, draw_values: list[float | None], eligible_task_count: int) -> dict:
    valid = [value for value in draw_values if value is not None and math.isfinite(value)]
    need(len(draw_values) == REPLICATES, 'exact_bootstrap_draw_count')
    if point is None:
        reason = 'point_undefined_zero_pooled_variance_or_no_eligible_task'
    elif eligible_task_count < 2:
        reason = 'fewer_than_two_eligible_task_ids'
    elif len(valid) < MIN_DEFINED_DRAWS_FOR_CI:
        reason = 'fewer_than_1900_defined_bootstrap_draws'
    else:
        reason = None
    return {'point': point, 'pointwise_95_low': percentile(valid, 0.025) if reason is None else None, 'pointwise_95_high': percentile(valid, 0.975) if reason is None else None, 'defined_draws': len(valid), 'undefined_draws': REPLICATES - len(valid), 'interval_null_reason': reason, 'eligible_task_ids': eligible_task_count}
