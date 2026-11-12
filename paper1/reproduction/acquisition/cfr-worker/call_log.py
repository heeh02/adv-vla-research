"""Per-query, per-call, per-token log records (TECH-CONTRACT-v1 4.3; I-3 CALL-LOG-v1.jsonl.gz).

New code. One JSON line per (sample, arm, query, generation call); every per-token quantity
is a length-10 list. Identity-chain fields (family, fold, expected and loaded array SHA)
are attached to every line so the Baidu readout never has to join another file.
"""

from __future__ import annotations

import hashlib

import numpy as np

from .common import jsonable
from .repair_math import BASIS_ORDER, call_measurements

SCHEMA = "cf-repair-call-log-v1"
REQUIRED_TOKEN_FIELDS = ("H_T_norm", "H_C_norm", "D_norm", "w_intended_norm", "W_actual_norm", "W_actual_maxabs",
                         "cast_error_norm", "declared_noop", "actual_noop", "hidden_distance_before",
                         "hidden_distance_after", "tau")
REQUIRED_BASIS_FIELDS = ("p_norm", "pM_norm", "pR_norm", "energy_share")


def identity_fields(identity_record, random_proofs, g_assigned):
    arrays = identity_record["arrays"]
    out = {"family": identity_record["family"], "fold": identity_record["fold"], "row_id": identity_record["row_id"],
           "task_id": identity_record["task_id"], "expected_fit_sha256": identity_record["expected_fit"]["sha256"]}
    for role in ("B", "C", "M"):
        out[role] = {"name": arrays[role]["name"], "expected_array_sha256": arrays[role]["expected_array_sha256"],
                     "loaded_array_sha256": arrays[role]["loaded_array_sha256"],
                     "fp32_consumed_sha256": arrays[role]["fp32_consumed"]["sha256"]}
    for name, proof in random_proofs.items():
        out[name] = {"expected_array_sha256": proof["expected_array_sha256"],
                     "loaded_array_sha256": proof["loaded_array_sha256"],
                     "fp32_consumed_sha256": proof["fp32_consumed"]["sha256"]}
    out["G_assigned"] = g_assigned
    return out


def call_records(trace, *, base, cfg, bases_by_name, template, gram, identity):
    """Yield one record per generation call of a repair-route query."""
    for call in range(10):
        measure = call_measurements(
            h_t=trace["H_T"][call, 0], h_c=trace["H_C"][call, 0], h_p=trace["H_P"][call, 0],
            w=trace["w"][call, 0], declared_noop=trace["declared_noop"][call, 0],
            route=cfg["route"], arm_name=cfg["basis"], bases=bases_by_name, template=template, gram=gram,
            s=trace["s"][call, 0], argmin=trace["argmin"][call, 0], eq_names=cfg["eq_names"],
            tau=trace["tau"][call, 0], active=bool(trace["applied"][call]))
        x = np.ascontiguousarray(trace["x_in"][call])
        record = {"schema": SCHEMA, **base, "arm": cfg["arm"], "route": cfg["route"], "alpha": cfg["alpha"],
                  "arm_basis": cfg["basis"], "call_index": call,
                  "t_j": float(trace["t_in"][call]), "t_j_hex": float(trace["t_in"][call]).hex(),
                  "x_j_sha256": hashlib.sha256(x.tobytes()).hexdigest(),
                  "schedule_active": bool(trace["applied"][call]),
                  "in_graph_fp32_p_norm": {name: trace["p_norm"][call, i, 0] for i, name in enumerate(BASIS_ORDER)},
                  "tokens": measure, "identity_chain": identity}
        yield jsonable(record)


def validate_record(record):
    """Schema check used by local tests and by the worker before writing."""
    problems = []
    if record.get("schema") != SCHEMA:
        problems.append("schema")
    tokens = record.get("tokens", {})
    for name in REQUIRED_TOKEN_FIELDS:
        value = tokens.get(name)
        if not isinstance(value, list) or len(value) != 10:
            problems.append("token_field:" + name)
    for basis in BASIS_ORDER:
        row = tokens.get("bases", {}).get(basis, {})
        for name in REQUIRED_BASIS_FIELDS:
            if not isinstance(row.get(name), list) or len(row[name]) != 10:
                problems.append(f"basis_field:{basis}.{name}")
    chain = record.get("identity_chain", {})
    for key in ("family", "fold", "B", "C", "M"):
        if key not in chain:
            problems.append("identity_chain:" + key)
    for role in ("B", "C", "M"):
        row = chain.get(role, {})
        if not (row.get("expected_array_sha256") and row.get("loaded_array_sha256")):
            problems.append("identity_chain_sha:" + role)
        elif row["expected_array_sha256"] != row["loaded_array_sha256"]:
            problems.append("identity_chain_mismatch:" + role)
    for key in ("query_index", "call_index", "t_j", "x_j_sha256", "arm", "route"):
        if key not in record:
            problems.append("field:" + key)
    return problems
