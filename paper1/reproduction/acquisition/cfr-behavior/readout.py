"""v4.1 CONF secondary implementation. Cloud-local inputs only; no network/model.

The unchanged primary extractor, analysis, and q0 analysis remain authoritative.
This source reads only natural CONF episode JSON, the explicitly named small NPY
members in their existing NPZs, and their scientific CALL-LOG. Never scans 1A,
SEALED, H/flow/raw32/noise or chooses a scientific attempt.
"""
from __future__ import annotations
import argparse
import ast
import gzip
import hashlib
import importlib.util
import itertools
import json
import math
from pathlib import Path
import struct
import sys
import zipfile

ARMS = ("N_C", "N_T", "R_resp", "R_clean", "R_rand", "R_full")
REPAIR = ARMS[2:]
FAMILIES = ("Camera Viewpoints", "Sensor Noise")
PROTOCOL_SHA = "e098fcab1e9005861028938d46d4ae372ab7c5c261c02bec5caa90c3be4eb7b5"
ADOPTION_REF = {"bytes":1433,"sha256":"43a87a37b10458e7c2077759f137c5f43d3b9bb6052a8aee913b0e90f1639a3e"}
SMALL = 16 * 1024 ** 2


def need(value, code):
    if not value:
        raise ValueError(code)


def ident(path):
    path = Path(path)
    need(path.is_file() and not path.is_symlink(), "regular_input")
    h, n = hashlib.sha256(), 0
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 ** 2), b""):
            h.update(b); n += len(b)
    return {"bytes": n, "sha256": h.hexdigest()}


def exact(path, ref):
    need(ident(path) == {k: ref[k] for k in ("bytes", "sha256")}, "input_full_SHA")


def load(path, cap=SMALL):
    need(Path(path).stat().st_size <= cap, "bounded_JSON")
    return json.loads(Path(path).read_text())


def load_module(name, path, ref):
    exact(path, ref)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def validate_release(release,pid,gid,admitted):
    need(release.get("schema")=="cf-repair-CONF-outcome-release-v1"
         and release.get("protocol_identity")==pid and release.get("G1_attachment_identity")==gid,
         "original_global_release_PROTOCOL_G1")
    need(release.get("all_admitted_population_closed") is True and release.get("no_partial_shard_unblinding") is True,
         "global_CONF_release_before_secondary_effect_reads")
    rows=release.get("admitted_rows",[]);closure=release.get("row_closure",{})
    need(len(rows)==len(set(rows)) and set(rows)==set(admitted)==set(closure),"exact_all_admitted_release_support")
    for value in closure.values():
        ref=value.get("evidence_identity",{})
        need(value.get("status") in ("COLLECTION_TERMINAL","PREDEFINED_STOP_RECORDED")
             and type(ref.get("bytes")) is int and ref["bytes"]>0
             and isinstance(ref.get("sha256"),str) and len(ref["sha256"])==64
             and all(x in "0123456789abcdef" for x in ref["sha256"])
             and set(value).isdisjoint({"success","outcome","success_rate","harm","recovery"}),
             "original_technical_closure_not_outcome_driven")


def finite(values):
    return all(type(v) in (float, int) and math.isfinite(v) for v in values)


def norm(values):
    return math.sqrt(math.fsum(v * v for v in values))


def distance(a, b):
    need(len(a) == len(b) and finite(a) and finite(b), "finite_same_shape_vectors")
    return norm([x - y for x, y in zip(a, b)])


def read_array(root, manifest, raw_ref, member, shape):
    """A hash check streams the original NPZ; only this small member is decoded."""
    allowed = {"plan_adapter7", "clean_adapter7", "arm_adapter7",
               "native_observation__robot0_eef_pos"}
    need(member in allowed, "no_neural_or_noise_members")
    name = raw_ref.get("relative")
    need(isinstance(name, str) and name.startswith("SHARDS/") and name.endswith(".npz")
         and not any(p in ("", ".", "..") for p in name.split("/")) and "\\" not in name,
         "episode_bound_SHARDS_path")
    ref = manifest.get(name)
    if ref is None:
        return None, "referenced_array_missing"
    need(ref.get("sealed") is False and not ref.get("sealed_fields")
         and ref["bytes"] == raw_ref["bytes"], "unsealed_same_episode_array")
    path = Path(root) / name
    need(path.resolve().is_relative_to(Path(root).resolve()), "array_inside_owned_input_root")
    if not path.is_file():
        return None, "array_not_restored"
    exact(path, ref)
    with zipfile.ZipFile(path) as z:
        names = z.namelist(); need(len(names) == len(set(names)), "unique_NPZ_members")
        if member + ".npy" not in names:
            return None, "native_field_NA"
        info = z.getinfo(member + ".npy")
        need(info.file_size <= 4096 and not info.flag_bits & 1, "small_unencrypted_member")
        raw = z.read(info)
    need(raw[:6] == b"\x93NUMPY" and raw[6:8] in (b"\x01\x00", b"\x02\x00"), "NPY_v1_v2")
    width = 2 if raw[6] == 1 else 4; start = 8 + width
    size = int.from_bytes(raw[8:start], "little")
    need(size <= 2048 and start + size <= len(raw), "bounded_NPY_header")
    header = ast.literal_eval(raw[start:start + size].decode("latin1"))
    need(header.get("shape") == tuple(shape) and header.get("fortran_order") is False
         and header.get("descr") in ("<f4", "<f8"), "original_finite_float_array_ABI")
    count = math.prod(shape); fmt, item = ("f", 4) if header["descr"] == "<f4" else ("d", 8)
    payload = raw[start + size:]; need(len(payload) == count * item, "exact_NPY_length")
    values = list(struct.unpack("<" + str(count) + fmt, payload))
    need(finite(values), "finite_array_values")
    return values, None


def direction_metrics(plan, clean, treated, group):
    cols = {"translation": range(3), "rotation": range(3, 6), "gripper": (6,)}[group]
    ix = [i * 7 + j for i in range(5) for j in cols]
    a, c, t = ([v[i] for i in ix] for v in (plan, clean, treated))
    d, e = [x - y for x, y in zip(c, t)], [x - y for x, y in zip(a, t)]
    dnorm = norm(d)
    parallel = math.fsum(x * y for x, y in zip(e, d)) / dnorm if dnorm > 0 else None
    off = norm([x - parallel * y / dnorm for x, y in zip(e, d)]) if dnorm > 0 else None
    return {"command_group": group, "native_error": dnorm, "edited_error": distance(a, c),
            "error_gain": dnorm - distance(a, c), "signed_along_clean": parallel,
            "off_target": off, "positive_overshoot": max(0., parallel - dnorm) if parallel is not None else None,
            "direction_NA": dnorm == 0, "units": "adapter_command_units"}


def issued_prefix(doc, plan):
    issued = doc.get("behavior", {}).get("issued_adapter7", [])
    need(isinstance(issued, list) and all(isinstance(v, list) and len(v) == 7 and finite(v) for v in issued),
         "recorded_finite_issued_adapter7")
    queries = doc.get("queries", [])
    start = 0
    for q in queries:
        need(q["issued_controls_start"] == start and type(q["issued_controls_count"]) is int
             and 0 <= q["issued_controls_count"] <= 5, "contiguous_original_issued_windows")
        start += q["issued_controls_count"]
    need(start == len(issued), "issued_windows_cover_actual_commands")
    q0 = next((q for q in queries if q["query"] == 0), None)
    prefix = issued[:q0["issued_controls_count"]] if q0 else []
    diffs = [a - plan[7 * i + j] for i, row in enumerate(prefix) for j, a in enumerate(row)] if plan is not None else None
    return {"actual_total_controls": len(issued), "first_query_actual_controls": len(prefix),
            "first_query_issued_prefix7": prefix, "generated_plan_length": 10 if plan is not None else None,
            "plan_prefix_maxabs": max(map(abs, diffs), default=0.) if diffs is not None else None,
            "plan_prefix_L2": norm(diffs) if diffs is not None else None,
            "L2_role": "numerical_identity_diagnostic_not_mixed_unit_action_strength",
            "no_unissued_commands_fabricated": True}


def physical_rows(doc, root, manifest):
    result = []
    complete = doc.get("execution_status") == "complete"
    observations = doc.get("physical_observations", [])
    for endpoint in (5, 60, "terminal"):
        selected = [p for p in observations if (p.get("phase") == "terminal" if endpoint == "terminal" else
                    p.get("phase") == "fixed_returned_control" and p.get("control_step") == endpoint)]
        need(len(selected) <= 1, "one_original_motion_observation_per_endpoint")
        entry = {"endpoint": endpoint, "position": None, "source_ref": None,
                 "terminal_is_own_episode_endpoint_not_same_time": endpoint == "terminal"}
        if not selected:
            structural = complete and endpoint != "terminal" and endpoint > doc.get("actual_returned_control_steps", 0)
            entry.update(status="STRUCTURAL_UNREACHED" if structural else "TECHNICAL_MISSING")
        elif endpoint == "terminal" and not complete:
            entry.update(status="TECHNICAL_PARTIAL_TERMINAL_NOT_ENDPOINT")
        else:
            obs = selected[0]
            values, reason = read_array(root, manifest, obs["arrays"], "native_observation__robot0_eef_pos", (3,))
            entry.update(position=values, status="AVAILABLE" if values is not None else "FIELD_NA",
                         reason=reason, control_step=obs["control_step"], sim_time=obs.get("sim_time"),
                         source_ref=manifest.get(obs["arrays"]["relative"]), geometry=obs.get("geometry"),
                         native_returned_observation_only=True)
        result.append(entry)
    return result


def clean_anchors(doc, root, manifest):
    indexed = {r["query"]: r for r in doc.get("qualified_anchor_records", [])}
    need(len(indexed) == len(doc.get("qualified_anchor_records", [])), "unique_qualified_anchors")
    rows = []
    for query in doc.get("planned_anchor_queries", []):
        need(query in (0, 12) and doc["arm"] in ("N_T", "R_resp"), "only_frozen_clean_anchor_roles")
        r = indexed.get(query)
        if r is None:
            structural = doc.get("execution_status") == "complete" and query >= doc.get("query_count", 0)
            rows.append({"query": query, "status": "STRUCTURAL_UNREACHED" if structural else "TECHNICAL_MISSING"})
            continue
        clean, _ = read_array(root, manifest, r["arrays"], "clean_adapter7", (1, 10, 7))
        arm, _ = read_array(root, manifest, r["arrays"], "arm_adapter7", (1, 10, 7))
        row = {"query": query, "status": "AVAILABLE" if clean is not None and arm is not None else "FIELD_NA",
               "physical_identity": r["physical_identity"], "noise_identity": r["noise_identity"],
               "reference_plan_executed": False, "q0_common_S0": query == 0,
               "q12_reference_at_this_arms_own_state": query == 12, "group_errors": {}}
        if row["status"] == "AVAILABLE":
            for group, cols in (("translation", range(3)), ("rotation", range(3, 6)), ("gripper", (6,))):
                ix = [i * 7 + j for i in range(5) for j in cols]
                row["group_errors"][group] = distance([arm[i] for i in ix], [clean[i] for i in ix])
        rows.append(row)
    return rows


def dose_calls(path, ref, sample_keys):
    exact(path, ref)
    seen = set()
    with gzip.open(path, "rt") as f:
        for line in f:
            need(len(line) <= 2 * 1024 ** 2, "bounded_call_log_line")
            row = json.loads(line)
            sample = row.get("sample_id")
            if sample not in sample_keys:
                continue
            key = (sample, row["query_index"], row["call_index"])
            need(key not in seen, "no_duplicate_call_log_rows"); seen.add(key)
            need(row.get("schema") == "cf-repair-call-log-v1" and row["arm"] in REPAIR
                 and row["arm"] == sample_keys[sample]["arm"] and row.get("episode") is True,
                 "formal_episode_repair_log_only")
            t = row["tokens"]; w, noop = t["W_actual_norm"], t["actual_noop"]
            need(len(w) == len(noop) == 10 and finite(w) and all(v >= 0 for v in w)
                 and all(type(v) is bool for v in noop)
                 and all(len(t[k])==10 and all(type(v) is bool for v in t[k])
                         for k in ("declared_noop","H_prime_equals_H_T_bits")), "actual_BF16_token_dose_ABI")
            yield {**sample_keys[sample], "query": row["query_index"], "call": row["call_index"],
                   "actual_token_L2_sum": math.fsum(w), "actual_token_squared_L2_sum": math.fsum(v*v for v in w),
                   "undeclared_cast_noop_tokens": sum(noop), "declared_noop_tokens": sum(t["declared_noop"]),
                   "actual_zero_write_tokens": sum(t["H_prime_equals_H_T_bits"]), "token_count": 10,
                   "quantity_role": "actual_BF16_edit_write_not_attention_output_W",
                   "schedule_active": row["schedule_active"], "x_identity": row["x_j_sha256"]}


def secondary_requests(manifest, documents, protocol, assignment, shard_rows, g1, frozen):
    """Exact second-pass CPU GET list, computed from existing episode metadata.

    No array is decoded and no effect is computed. The frozen helper checks the
    existing metadata status/success types; therefore even this planning pass
    remains under the existing global CONF-unsealing authority.
    """
    members=frozen.members(manifest)
    first=frozen.first_plan_requests(manifest,documents,protocol,assignment,shard_rows,g1)
    requests={r["path"]:dict(r) for r in first["requests"]}
    roles={name:["existing_key_secondary_q0_plan"] for name in requests}
    def add(raw,role):
        name=raw.get("relative")
        need(isinstance(name,str) and name.startswith("SHARDS/") and name.endswith(".npz")
             and not any(x in ("", ".", "..") for x in name.split("/")),"specific_episode_SHARDS_request")
        ref=members.get(name)
        if ref is None:return
        need(ref.get("sealed") is False and not ref.get("sealed_fields") and ref["bytes"]==raw["bytes"],
             "unsealed_episode_bound_second_pass_request")
        requests[name]=dict(ref);roles.setdefault(name,[]).append(role)
    expected={r["path"] for r in frozen.episode_requests(manifest,protocol,assignment,shard_rows,g1)["requests"]}
    need(set(documents)<=expected,"only_formal_natural_episode_metadata")
    for doc in documents.values():
        if doc.get("schema")!="cf-repair-science-episode-v1":continue
        for row in doc.get("physical_observations",[]):
            if row.get("phase")=="terminal" or (row.get("phase")=="fixed_returned_control" and row.get("control_step") in (5,60)):
                add(row["arrays"],"original_fixed_or_terminal_EEF")
        for row in doc.get("qualified_anchor_records",[]):
            need(doc["arm"] in ("N_T","R_resp") and row["query"] in (0,12),"original_qualified_clean_anchor_only")
            add(row["arrays"],"q0_q12_current_state_clean_plan")
    ref=members.get("CALL-LOG-v1.jsonl.gz")
    if ref:
        need(ref.get("sealed") is False and not ref.get("sealed_fields"),"unsealed_science_call_log_request")
        requests[ref["path"]]=dict(ref);roles[ref["path"]]=["actual_edit_dose_no_neural_array_decoding"]
    ordered=[requests[n] for n in sorted(requests)]
    return {"schema":"cf-repair-CONF-behavior-second-pass-plan-v1","requests":ordered,
            "roles":roles,"exact_GET_objects":len(ordered),"exact_payload_bytes":sum(r["bytes"] for r in ordered),
            "NPZ_members_decoded":["plan_adapter7.npy","clean_adapter7.npy","arm_adapter7.npy",
                "native_observation__robot0_eef_pos.npy"],
            "H_flow_raw32_noise_decoded":False,"no_new_scientific_or_model_calls":True}


def extract(index, protocol, assignment, g1, base, frozen):
    """Only callers after root source/adoption/closure binding may read real inputs."""
    rows = frozen.admitted_rows(protocol, assignment, g1)
    seen, documents, cases, logs = set(), {}, {}, []
    for job in index["jobs"]:
        shard = job["shard_rows"]
        need(len(shard) == len(set(shard)) and set(shard) <= set(rows) and not seen.intersection(shard),
             "one_authoritative_CONF_attempt_per_row")
        seen.update(shard)
        need(job["protocol_sha256"] == PROTOCOL_SHA and job["G1_attachment_identity"] == index["G1_attachment_identity"],
             "same_original_1B_G1_for_each_source_job")
        if job.get("manifest_identity") is None:
            continue
        root = Path(job["root"]); manifest_path = root / "MANIFEST-v1.json"
        exact(manifest_path, job["manifest_identity"]); manifest = load(manifest_path)
        members = frozen.members(manifest)
        requested = frozen.episode_requests(manifest, protocol, assignment, shard, g1)
        docs = frozen.load_episode_documents(root, requested["requests"])
        # Use the frozen actual-role / q0 qualification and reference checks.
        entries = frozen.first_plan_requests(manifest, docs, protocol, assignment, shard, g1)["entries"]
        by_sample = {}
        for entry in entries:
            key = (entry["row_id"], entry["noise_label"], entry["arm"])
            doc_path = "SCIENCE-EPISODES/" + frozen.sample(*key) + ".json"
            doc = docs.get(doc_path)
            if doc is None:
                continue
            metadata = {"row_id": key[0], "noise_label": key[1], "arm": key[2],
                        "family": rows[key[0]]["family"], "task_id": rows[key[0]]["task_id"], "init_index": 0}
            if doc.get("schema") != "cf-repair-science-episode-v1":
                need(doc.get("scientific_attempted") is False and str(doc.get("execution_status", "")).startswith("not_started"),
                     "only_original_driver_not_started_placeholder")
                cases[key] = {**metadata, "status": "TECHNICAL_MISSING", "success": None,
                    "original_execution_status": doc.get("execution_status"), "q0_plan": None,
                    "motion": [], "clean_anchors": [], "queries": [], "issued": None,
                    "generation_history_known": True, "source_episode_identity": entry.get("episode_ref")}
                continue
            need(doc.get("included_in_primary_six_arm_contrasts") is True,"formal_six_arm_episode")
            plan = None
            if entry.get("plan_ref") is not None and (root / entry["plan_ref"]["path"]).is_file():
                plan = list(frozen.plan_adapter7(root / entry["plan_ref"]["path"], entry["plan_ref"]))
            cases[key] = {**metadata, "status": entry["sr_status"], "success": entry["success"],
                          "original_execution_status": doc.get("execution_status"),
                          "generation_history_known": True,
                          "source_q0_plan_identity":entry.get("plan_ref"),
                          "q0_plan": plan, "issued": issued_prefix(doc, plan),
                          "motion": physical_rows(doc, root, members),
                          "clean_anchors": clean_anchors(doc, root, members),
                          "queries": [{k: q[k] for k in ("query", "issued_controls_start", "issued_controls_count", "execution_status")}
                                      for q in doc.get("queries", [])],
                          "source_episode_identity": entry.get("episode_ref")}
            sample = frozen.sample(*key)
            by_sample[sample] = metadata
        ref = members.get("CALL-LOG-v1.jsonl.gz")
        if ref is not None and (root / ref["path"]).is_file():
            need(ref.get("sealed") is False and not ref.get("sealed_fields"), "formal_unsealed_call_log")
            logs.extend(dose_calls(root / ref["path"], ref, by_sample))
    for rid, row in rows.items():
        for label, arm in itertools.product((0, 1), ARMS):
            cases.setdefault((rid, label, arm), {"row_id": rid, "noise_label": label, "arm": arm,
                "family": row["family"], "task_id": row["task_id"], "init_index": 0,
                "status": "TECHNICAL_MISSING", "success": None, "q0_plan": None, "motion": [],
                "clean_anchors": [], "queries": [], "issued": None, "generation_history_known": False})
    return cases, logs


def summary_value(rows, values, base, family_index, boot, *, available=False):
    task_values = base.hierarchy(rows, values)
    full = base.mean(list(task_values.values()))
    if not available:
        return {"point": full, "task_values": task_values,
                "descriptive_interval": base.task_interval(list(task_values.values()), seed=boot["seed"]+family_index,
                    replicates=boot["replicates"], tail=.025)}
    # Same frozen hierarchy, but explicitly conditional on available units;
    # a variant averages its available labels and tasks average available variants.
    groups = {}
    for r in rows:
        v = [values[(r["row_id"], 0, n)] for n in (0, 1)]
        v = [x for x in v if x is not None]
        if v: groups.setdefault(r["task_id"], []).append(base.mean(v))
    tasks = {t: base.mean(v) for t, v in sorted(groups.items())}
    return {"point": base.mean(list(tasks.values())), "task_values": tasks,
            "descriptive_interval": base.task_interval(list(tasks.values()), seed=boot["seed"]+family_index,
                replicates=boot["replicates"], tail=.025),
            "support_role": "explicit_common_available_units_not_full_population",
            "full_population_point": full, "covered_tasks": len(tasks)}


def outcome_components(lhs, rhs):
    options = [(l, r) for l in ([lhs] if lhs is not None else [0, 1])
               for r in ([rhs] if rhs is not None else [0, 1])]
    tests = {"harm": lambda l,r: r == 1 and l == 0,
             "recovery": lambda l,r: r == 0 and l == 1,
             "joint_success": lambda l,r: l == r == 1,
             "joint_failure": lambda l,r: l == r == 0}
    result = {}
    for name, test in tests.items():
        v = [int(test(l, r)) for l, r in options]
        result[name] = {"value": v[0] if len(set(v)) == 1 else None, "lower": min(v), "upper": max(v)}
    return result


def analyze(cases, logs, protocol, assignment, g1, base):
    all_rows = {r["row_id"]: r for r in assignment["rows"] if r["split"] == "confirm"}
    motion, anchors, behavior = [], [], []
    dose_by_key = {}
    for r in logs:
        k = (r["row_id"], r["noise_label"], r["arm"]); dose_by_key.setdefault(k, []).append(r)
    for key, case in sorted(cases.items()):
        rid, label, arm = key
        target, other = cases[(rid, label, "N_C")], cases[(rid, 1-label, "N_C")]
        treated = cases[(rid, label, "N_T")]
        plans = (case.get("q0_plan"), target.get("q0_plan"), treated.get("q0_plan"))
        entry = {k: v for k,v in case.items() if k not in ("q0_plan", "motion", "clean_anchors", "queries")}
        entry["q0_direction"] = [direction_metrics(*plans, group) for group in ("translation", "rotation", "gripper")] if all(p is not None for p in plans) else None
        dlog = dose_by_key.get(key, [])
        issued_q = {q["query"] for q in case["queries"] if q["issued_controls_count"] > 0}
        generated_q = {q["query"] for q in case["queries"]}
        expected_calls = len(generated_q)*10 if arm in REPAIR else 0
        actual_keys = {(r["query"],r["call"]) for r in dlog}
        whole_generation_history = (case.get("original_execution_status")=="complete"
            or str(case.get("original_execution_status","")).startswith("not_started"))
        complete_dose = arm not in REPAIR or (case.get("generation_history_known") is True and whole_generation_history
            and actual_keys == {(q,j) for q in generated_q for j in range(10)})
        def exposure(items):
            return {"token_L2_sum": math.fsum(r["actual_token_L2_sum"] for r in items),
                    "pooled_token_L2": math.sqrt(math.fsum(r["actual_token_squared_L2_sum"] for r in items)),
                    "undeclared_cast_noop_tokens": sum(r["undeclared_cast_noop_tokens"] for r in items),
                    "declared_noop_tokens": sum(r["declared_noop_tokens"] for r in items),
                    "actual_zero_write_tokens": sum(r["actual_zero_write_tokens"] for r in items), "calls": len(items),
                    "tokens": sum(r["token_count"] for r in items)}
        entry["dose"] = {"status": "AVAILABLE" if complete_dose else "TECHNICAL_LOG_MISSING",
                         "expected_calls": expected_calls, "observed_calls": len(dlog),
                         "all_generated_queries": exposure(dlog) if complete_dose else None,
                         "observed_logged_exposure": exposure(dlog),
                         "queries_with_any_issued_command": exposure([r for r in dlog if r["query"] in issued_q]) if complete_dose else None,
                         "native_no_edit_by_design": arm not in REPAIR}
        behavior.append(entry)
        for record in case["clean_anchors"]:
            anchors.append({**{k: case[k] for k in ("row_id","noise_label","arm","family","task_id")}, **record})
        by_endpoint = [{r["endpoint"]: r for r in c["motion"]} for c in (case,target,other)]
        for endpoint in (5, 60, "terminal"):
            obs = [x.get(endpoint) for x in by_endpoint]
            good = all(r and r.get("status") == "AVAILABLE" for r in obs)
            m = {k: case[k] for k in ("row_id","noise_label","arm","family","task_id")}
            m.update(endpoint=endpoint, separation=None, clean_resampling=None,
                     source_statuses=[r.get("status") if r else "TECHNICAL_MISSING" for r in obs],
                     status="AVAILABLE" if good else "UNAVAILABLE", source_observations=obs,
                     terminal_is_each_arms_own_endpoint=endpoint == "terminal")
            if good:
                m.update(separation=distance(obs[0]["position"],obs[1]["position"]),
                         clean_resampling=distance(obs[1]["position"],obs[2]["position"]))
            motion.append(m)
    boot = protocol["statistics"]["bootstrap"]
    outcome_summary, motion_summary, dose_summary, anchor_summary = [], [], [], []
    comparisons = [(a,"N_C") for a in ARMS[1:]] + [("R_resp",a) for a in ("N_T","R_clean","R_rand")]
    motion_index={(r["row_id"],r["noise_label"],r["arm"],r["endpoint"]):r for r in motion}
    behavior_index={(r["row_id"],r["noise_label"],r["arm"]):r for r in behavior}
    for fi,family in enumerate(FAMILIES):
        if g1[family].get("advance") is not True: continue
        rows=[all_rows[r] for r in protocol["stage3"]["rows"][family]]
        for lhs,rhs in comparisons:
            parts={name:{kind:{} for kind in ("value","lower","upper")} for name in ("harm","recovery","joint_success","joint_failure")}
            known=0
            for row,label in itertools.product(rows,(0,1)):
                rid=row["row_id"]; l,r=cases[(rid,label,lhs)]["success"],cases[(rid,label,rhs)]["success"]
                known+=int(l is not None and r is not None)
                for name,values in outcome_components(l,r).items():
                    for kind,value in values.items():parts[name][kind][(rid,0,label)]=value
            out={"family":family,"lhs":lhs,"rhs":rhs,"planned_noise_pairs":len(rows)*2,"known_pairs":known,"components":{},
                 "descriptive_only_primary_G2_unchanged":True}
            for name,values in parts.items():
                metric=summary_value(rows,values["value"],base,fi,boot)
                metric["identification_bounds"]=[base.mean(list(base.hierarchy(rows,values[k]).values())) for k in ("lower","upper")]
                out["components"][name]=metric
            h=out["components"]["harm"]["point"];r=out["components"]["recovery"]["point"]
            out["net_success_difference"]=r-h if h is not None and r is not None else None
            outcome_summary.append(out)
        for endpoint in (5,60,"terminal"):
            common={(r["row_id"],n) for r,n in itertools.product(rows,(0,1))
                    if all(motion_index[(r["row_id"],n,a,endpoint)]["status"]=="AVAILABLE" for a in ARMS)}
            for arm in ARMS:
                ds,bs={},{}
                pairwise_available=0
                for row,n in itertools.product(rows,(0,1)):
                    rid=row["row_id"]; m=motion_index[(rid,n,arm,endpoint)];k=(rid,0,n)
                    pairwise_available+=int(m["status"]=="AVAILABLE")
                    ds[k]=m["separation"] if (rid,n) in common else None
                    bs[k]=m["clean_resampling"] if (rid,n) in common else None
                d,b=(summary_value(rows,v,base,fi,boot,available=True) for v in (ds,bs))
                motion_summary.append({"family":family,"endpoint":endpoint,"arm":arm,
                    "planned_noise_cases":len(rows)*2,"pairwise_available_noise_cases":pairwise_available,
                    "six_arm_common_available_noise_cases":len(common),"separation":d,"clean_resampling":b,
                    "full_plan_cases_outside_common_support":len(rows)*2-len(common),
                    "ratio_of_task_weighted_means":d["point"]/b["point"] if d["point"] is not None and b["point"] not in (None,0) else None,
                    "zero_clean_reference_cases_retained":True,"unbounded_missing_values_not_zero_filled":True})
        for arm in ARMS:
            values={}
            for row,n in itertools.product(rows,(0,1)):
                rid=row["row_id"];v=behavior_index[(rid,n,arm)]["dose"]["all_generated_queries"]
                values[(rid,0,n)]=v["token_L2_sum"] if v is not None else None
            dose_summary.append({"family":family,"arm":arm,"generated_token_L2_sum":summary_value(rows,values,base,fi,boot),
                                 "no_equal_dose_assumption_across_closed_trajectories":True})
        anchor_index={(r["row_id"],r["noise_label"],r["arm"],r["query"]):r for r in anchors}
        for arm,query,group in itertools.product(("N_T","R_resp"),(0,12),("translation","rotation","gripper")):
            vals={};statuses={}
            for row,n in itertools.product(rows,(0,1)):
                rid=row["row_id"];r=anchor_index.get((rid,n,arm,query))
                status=r["status"] if r else "TECHNICAL_MISSING"
                statuses[status]=statuses.get(status,0)+1
                vals[(rid,0,n)]=r["group_errors"][group] if r and status=="AVAILABLE" else None
            anchor_summary.append({"family":family,"arm":arm,"query":query,"command_group":group,
                "planned_noise_cases":len(rows)*2,"support_statuses":statuses,
                "own_state_clean_error":summary_value(rows,vals,base,fi,boot,available=True),
                "q12_between_arms_not_same_state":query==12})
    return {"behavior":behavior,"motion":motion,"anchors":anchors,"dose_calls":logs,
            "summary":{"schema":"cf-repair-CONF-secondary-behavior-v1","stage":3,
                "outcomes":outcome_summary,"motion":motion_summary,"dose":dose_summary,"clean_anchors":anchor_summary,
                "geometry":{"status":"NA","planned_variants":len(cases)//12,
                    "reason":"all_current_frozen_named_geometry_entries_are_NA","not_a_success_proxy":True},
                "primary_G2_and_key_q0_comparisons_unchanged":True,"native_PW":"NA","clean_L18":"NA"}}


def main(argv=None):
    parser=argparse.ArgumentParser(__doc__)
    for name in ("index","protocol","assignments","base-analysis","frozen-extractor","g1-input","definitions","adoption","release","out"):
        parser.add_argument("--"+name,required=True,type=Path)
    parser.add_argument("--g1-sha",required=True);parser.add_argument("--definitions-sha",required=True)
    parser.add_argument("--cloud-array-read",action="store_true")
    a=parser.parse_args(argv)
    need(a.cloud_array_read and sys.platform.startswith("linux"),"actual_arrays_Baidu_cloud_CPU_only")
    protocol=load(a.protocol);pid=ident(a.protocol)
    need(pid["sha256"]==PROTOCOL_SHA and protocol.get("stage1B_frozen") is True
         and protocol.get("G1_actual_decisions") is None,"unchanged_original_1B")
    need(ident(a.definitions)["sha256"]==a.definitions_sha,"root_selected_secondary_definitions_identity")
    exact(a.adoption,ADOPTION_REF);adoption=load(a.adoption)
    need({k:adoption["definition"][k] for k in ("bytes","sha256")}==ident(a.definitions)
         and adoption["original_PROTOCOL_SHA"]==PROTOCOL_SHA
         and adoption["original_G2_and_q0_main_comparisons_unchanged"] is True,"root_actual_secondary_definition_adoption")
    need(protocol["stage3"]["motion_steps"]==[5,60]
         and all(v.get("status")=="NA" for v in protocol["stage3"]["named_geometry"].values()),
         "only_actual_frozen_motion_and_geometry_contract")
    base=load_module("_primary_behavior_primitives",a.base_analysis,protocol["bindings"]["analysis_code"])
    frozen=load_module("_frozen_CONF_episode_extractor",a.frozen_extractor,protocol["bindings"]["confirm_scalar_extractor_code"])
    aid=ident(a.assignments);need(aid=={k:protocol["bindings"]["assignment_table"][k] for k in ("bytes","sha256")},"same_frozen_assignments")
    g1,gid=base.load_g1_attachment(a.g1_input,a.g1_sha,protocol=protocol,protocol_identity=pid,analysis_identity=ident(a.base_analysis))
    index=load(a.index)
    need(index.get("schema")=="cf-repair-confirm-scalar-index-v1" and index.get("protocol_identity")==pid
         and index.get("assignment_input_identity")==aid,"root_authoritative_CONF_index_original_binding")
    assignment=load(a.assignments)
    validate_release(load(a.release),pid,gid,frozen.admitted_rows(protocol,assignment,g1))
    # Existing primary index binds G1 on each job; expose that same identity to
    # the independent secondary reader without editing its input document.
    index={**index,"G1_attachment_identity":gid}
    cases,logs=extract(index,protocol,assignment,g1,base,frozen)
    values=analyze(cases,logs,protocol,assignment,g1,base)
    a.out.mkdir(parents=True,exist_ok=False)
    sources={k:ident(v) for k,v in {"protocol":a.protocol,"assignments":a.assignments,"index":a.index,"G1":a.g1_input,
        "frozen_primary_analysis":a.base_analysis,"frozen_extractor":a.frozen_extractor,"definitions":a.definitions,
        "root_adoption":a.adoption,"global_release":a.release,"implementation":__file__}.items()}
    values["summary"]["source_bindings"]=sources
    for name,key in (("CASE-BEHAVIOR","behavior"),("MOTION","motion"),("CLEAN-ANCHOR","anchors"),("DOSE-CALL","dose_calls")):
        with (a.out/(name+".jsonl")).open("x") as f:
            for row in values[key]:f.write(json.dumps(row,sort_keys=True,allow_nan=False)+"\n")
    with (a.out/"SUMMARY-v1.json").open("x") as f:json.dump(values["summary"],f,sort_keys=True,indent=2,allow_nan=False);f.write("\n")


if __name__=="__main__":main()
