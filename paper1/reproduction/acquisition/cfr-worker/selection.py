"""Frozen Stage 1A sample-selection rules (dev units only; TECH-CONTRACT-v1 7, TECH-1A-SIZING-v1).

New code. The rules use only SPLIT-FROZEN membership, the I-2 unit order/metadata, the
official variant name and (optionally) the T0 recoverability codes; never any outcome,
command or hidden-state value. Technical units keep their own identity and never enter
the confirmation population.
"""

from __future__ import annotations

import re

RULES_VERSION = "cf-repair-1a-selection-v1"
VIEW = re.compile(r"_view_(-?\d+)_(-?\d+)_(\d+)_(-?\d+)_(-?\d+)_initstate_(\d+)(?:_noise_(\d+))?$")
TRANSFORMS = ((1, 10, "motion"), (11, 20, "gauss"), (21, 30, "zoom"), (31, 40, "fog"), (41, 50, "glass"))
EXCLUDED_T0_PREFIXES = ("PARTIAL", "NOT_RECOVERABLE")


def parse_variant(name):
    match = VIEW.search(name)
    if not match:
        raise ValueError("plus_variant_suffix_unparsed")
    h, v, s, r, e, init, noise = match.groups()
    params = {"horizon_view": int(h), "vertical_view": int(v), "scale_percent": int(s), "end_point_rot": int(r),
              "end_point_vertical": int(e), "init_state": int(init), "noise_id": int(noise or 0)}
    if params["end_point_rot"] or params["end_point_vertical"]:
        kind = "RE"
    elif params["scale_percent"] != 100:
        kind = "S"
    elif params["vertical_view"]:
        kind = "HV"
    elif params["horizon_view"]:
        kind = "H"
    else:
        kind = "default_view"
    transform = next((t for lo, hi, t in TRANSFORMS if lo <= params["noise_id"] <= hi), None)
    return {**params, "camera_kind": kind, "sensor_transform": transform}


def dev_rows(units, split, family, t0_codes=None):
    allowed = set(split["families"][family]["dev"]["row_ids"])
    rows = sorted((u for u in units if u["family"] == family and u["row_id"] in allowed), key=lambda u: u["row_id"])
    excluded = []
    if t0_codes:
        keep = []
        for u in rows:
            code = str(t0_codes.get(u["row_id"], "UNDETERMINED"))
            if code.startswith(EXCLUDED_T0_PREFIXES):
                excluded.append({"row_id": u["row_id"], "t0_code": code})
            else:
                keep.append(u)
        rows = keep
    return rows, excluded


def core_units(units, split, family, n, *, t0_codes=None):
    """First rows (by row_id) with distinct task_id; filled in order if fewer tasks than n."""
    rows, excluded = dev_rows(units, split, family, t0_codes)
    chosen, tasks = [], set()
    for u in rows:
        if u["task_id"] not in tasks and len(chosen) < n:
            chosen.append(u)
            tasks.add(u["task_id"])
    for u in rows:
        if len(chosen) >= n:
            break
        if u not in chosen:
            chosen.append(u)
    if len(chosen) < n:
        raise ValueError("insufficient_dev_units_for_rule")
    return chosen, {"rule": "first_by_row_id_distinct_task_then_fill", "family": family, "n": n,
                    "excluded_by_T0": excluded, "row_ids": [u["row_id"] for u in chosen]}


def camera_render_sample(units, split, *, limit=8, t0_codes=None):
    """T3 Camera: every dev Camera row if <= limit, else greedy cover of (suite, kind) then fill."""
    rows, excluded = dev_rows(units, split, "Camera Viewpoints", t0_codes)
    if len(rows) <= limit:
        chosen = list(rows)
    else:
        chosen, seen = [], set()
        for u in rows:
            key = (u["suite"], parse_variant(u["variant_name"])["camera_kind"])
            if key not in seen:
                chosen.append(u)
                seen.add(key)
        for u in rows:
            if len(chosen) >= limit:
                break
            if u not in chosen:
                chosen.append(u)
        chosen = chosen[:limit]
    kinds = sorted({parse_variant(u["variant_name"])["camera_kind"] for u in chosen})
    return chosen, {"rule": "all_dev_camera_rows_if_le_limit_else_cover_suite_kind", "limit": limit,
                    "kinds_covered": kinds, "missing_kinds": sorted({"H", "HV", "S", "RE"} - set(kinds)),
                    "problem_classes": "recorded at runtime from env.problem_name; unsampled classes reported",
                    "excluded_by_T0": excluded, "row_ids": [u["row_id"] for u in chosen]}


def sensor_render_sample(units, split, *, t0_codes=None):
    """T3 Sensor: the first dev row (by row_id) of each transform."""
    rows, excluded = dev_rows(units, split, "Sensor Noise", t0_codes)
    chosen, missing = [], []
    for _lo, _hi, transform in TRANSFORMS:
        match = [u for u in rows if parse_variant(u["variant_name"])["sensor_transform"] == transform]
        if match:
            chosen.append(match[0])
        else:
            missing.append(transform)
    return chosen, {"rule": "first_dev_row_per_transform", "missing_transforms": missing,
                    "excluded_by_T0": excluded, "row_ids": [u["row_id"] for u in chosen]}
