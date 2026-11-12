"""Fixed PVP precision preflight. Real native arrays are Linux-cloud only.

No transport or model code. The caller stages exactly six immutable objects.
The legacy native reader and PVP validator are reused byte-for-byte, pinned
before import. A single bounded JSON is the only output.
"""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import math
import re
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = "SDRU-CCONF-T06-V00700-I00-PC"
QUERY = 11
ROLES = ("baseline_none_1", "shared_pulse", "shared_persistent")
CALL_IDS = ("PVP-001", "PVP-097", "PVP-098")
POP_SHA = "7013c6a2d5229d499e755bfe59789ba71f9bdc0b58da029ff14fa4b481e3474f"
INPUTS_SHA = "b75b0d7dc52c31c263140bd5db996989eea7a2ee166d493ad7475f5ba352c999"
LEGACY = {
    "native_reader.py": "231e4814aa5466c1557753ab3d8f31a8478d2d8f0138111e96123e404cd64843",
    "pvp_numeric.py": "5886f70497c23a1089392301a5c1f0bfcdc511b2628d6696339babf361ff92fc",
}
RUNTIME_FILES = {"adapter.py", "core.py", "native_reader.py", "pvp_numeric.py", "INPUTS.json"}
RESULT_LIMIT = 256 * 1024


class InputError(ValueError):
    pass


def need(ok, code):
    if not ok:
        raise InputError(code)


def identity(raw):
    return {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def encode(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def same_ref(a, b):
    return all(a.get(k) == b.get(k) for k in ("path", "bytes", "sha256", "uri"))


def read_small(path, ref, cap):
    path = Path(path)
    need(path.is_file() and not path.is_symlink() and path.stat().st_size <= cap, "regular_bounded_small_file")
    raw = path.read_bytes()
    need(identity(raw) == {k: ref[k] for k in ("bytes", "sha256")}, "small_file_bytes_SHA")
    return raw


def load_inputs():
    raw = (HERE / "INPUTS.json").read_bytes()
    need(hashlib.sha256(raw).hexdigest() == INPUTS_SHA, "fixed_INPUTS_source_SHA")
    plan = json.loads(raw)
    need(plan["root_id"] == ROOT and plan["query_index"] == QUERY and plan["flow_index"] == 6 and
         plan["layer_index"] == 8 and plan["roles"] == list(ROLES), "fixed_one_anchor_j6_L9")
    refs = plan["input_objects"]
    need(len(refs) == 6 and len({r["path"] for r in refs}) == 6 and
         sum(r["bytes"] for r in refs) == 9151111, "exact_six_input_roster_and_bytes")
    need(all("/" not in r["path"] and r["path"] not in ("", ".", "..") for r in refs), "flat_fixed_staging_names")
    return plan


def verify_sources(expected_sha):
    need(isinstance(expected_sha, str) and re.fullmatch("[0-9a-f]{64}", expected_sha), "external_source_manifest_SHA")
    path = HERE / "SOURCE-MANIFEST.json"
    need(path.is_file() and not path.is_symlink() and path.stat().st_size <= 32768, "small_source_manifest")
    raw = path.read_bytes()
    need(hashlib.sha256(raw).hexdigest() == expected_sha, "source_manifest_SHA")
    document = json.loads(raw)
    need(document["schema"] == "SAME-W-precision-source-manifest-v1", "source_manifest_schema")
    refs = document["files"]
    need(len(refs) == len(RUNTIME_FILES) and {r["path"] for r in refs} == RUNTIME_FILES, "exact_runtime_source_roster")
    for ref in refs:
        read_small(HERE / ref["path"], ref, 1024**2)
    return identity(raw)


def load_helpers():
    """Import only frozen legacy bytes; temporary alias does not leak globally."""
    def module(name, filename, expected):
        path = HERE / filename
        raw = path.read_bytes()
        need(hashlib.sha256(raw).hexdigest() == expected, "frozen_legacy_source_SHA")
        spec = importlib.util.spec_from_file_location(name, path)
        value = importlib.util.module_from_spec(spec)
        # dataclasses resolves the module in sys.modules during class creation.
        sys.modules[name] = value
        exec(compile(raw, str(path), "exec"), value.__dict__)
        return value
    reader = module("_same_w_frozen_native_reader", "native_reader.py", LEGACY["native_reader.py"])
    previous = sys.modules.get("native_reader")
    sys.modules["native_reader"] = reader
    try:
        numeric = module("_same_w_frozen_pvp_numeric", "pvp_numeric.py", LEGACY["pvp_numeric.py"])
    finally:
        if previous is None:
            sys.modules.pop("native_reader", None)
        else:
            sys.modules["native_reader"] = previous
    return reader, numeric


def select_source_query(source, plan):
    need(source.get("qualification_from_complete_original_worker") is True and
         source.get("zero_new_fit") is True and source.get("source_result_ref") == plan["original_native_result_ref"],
         "original_qualified_source_evidence")
    selected = [q for q in source["queries"] if (q["root_id"], q["query_index"]) == (ROOT, QUERY)]
    need(len(selected) == 1, "one_actual_source_query")
    q = selected[0]
    need(q["task_id"] == "CCONF-T06" and q["parent_condition"] == "C" and q["graph"] == "Gstar" and
         q["policy_env_step"] == 55 and q["absolute_env_step"] == 65 and
         q["all_readout_calls_source_qualified"] is True and q["same_j5_and_x6_checked"] is True and
         q["same_j6_L1_L9_receiver_checked"] is True and q["same_E_pulse_persistent"] is True and
         q["same_E_all_three_calls_claim"] is False, "fixed_Gstar_source_proof_scope")
    refs = {r["role"]: r for r in plan["input_objects"]}
    need(same_ref(q["input_ref"], refs["input"]), "INPUT_bound_by_actual_source_ref_not_filename")
    calls = {r["role"]: r for r in q["selected_calls"]}
    need(set(calls) == set(ROLES), "three_actual_source_roles")
    for role, call_id in zip(ROLES, CALL_IDS):
        need(calls[role]["call_id"] == call_id and same_ref(calls[role]["ref"], refs[role]),
             "actual_role_call_id_and_wholefile_source_binding")
    return q


def planned_entries(planned, source_query, *, synthetic=False):
    need(planned.event == "PVP_PLANNED" and planned.metadata["synthetic"] is synthetic,
         "original_PLANNED_event_and_scope")
    proof = planned.metadata["proof"]
    need(all(proof.get(k) is True for k in ("source_bound", "model_bound", "inputs_bound")),
         "actual_model_source_input_guard")
    plan = planned.metadata["plan"]
    canonical = json.dumps(plan["global_population"], sort_keys=True, separators=(",", ":")).encode()
    need(hashlib.sha256(canonical).hexdigest() == POP_SHA and plan["shard_id"] == "CCONF-T06" and
         len(plan["entries"]) == 48 and [ROOT, QUERY] in plan["planned_queries"], "original_frozen_population")
    index = {entry["call_id"]: entry for entry in plan["entries"]}
    need(len(index) == 48, "unique_original_call_ids")
    entries = [index[call_id] for call_id in CALL_IDS]
    for role, entry in zip(ROLES, entries):
        need(entry["root_id"] == ROOT and entry["query_index"] == QUERY and entry["role"] == role and
             entry["receiver"] == "T" and entry["donor"] == "C" and entry["num_steps"] == 10,
             "fixed_actual_call_role_and_solver")
    return entries


def analyze_records(inp, calls, entries, source_query, reader, numeric, *, check=lambda: None):
    """No file/model IO; tests can pass codec-equivalent synthetic records."""
    import numpy as np
    spec = importlib.util.spec_from_file_location("_same_w_exact_dyadic_core", HERE / "core.py")
    core = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(core)
    check()
    n = numeric.validate(inp, calls, entries, ROOT, QUERY)
    need(inp.metadata["sources"] == source_query["old_saved_sources"], "exact_original_input_source_roster")
    need(n == source_query["prefix_capacity"] and
         int(calls[0].field("prefix_mask").numeric().sum()) == source_query["valid_prefix_tokens"],
         "same_declared_complete_prefix_support")
    need(bool(not np.any(inp.field("actual_E").numeric())) == source_query["exactzero_actual_E"], "actual_E_noop_identity")
    for name, shape, dtype in (("mask", (5, 1, 10, n + 10), "bool"),
        ("suffix_positions", (5, 1, 10), "int32"), ("query_defined", (5, 1, 10), "bool"),
        ("adarms_condition", (5, 1, 1024), "float32")):
        original = reader.field(calls[0], ("reads", name), shape, dtype).storage[1]
        for call in calls[1:]:
            actual = reader.field(call, ("reads", name), shape, dtype).storage[1]
            need(numeric.exact(original, actual), "all_three_actual_j6_" + name)
    for name in ("time", "time_weak_type", "dt"):
        original = calls[0].field("flow", name).storage[6]
        for call in calls[1:]:
            need(numeric.exact(np.asarray(original), np.asarray(call.field("flow", name).storage[6])),
                 "all_three_exact_carry_" + name)
    for call in calls[1:]:
        need(numeric.exact(calls[0].field("reads", "attention_gate").storage[1, 1],
                           call.field("reads", "attention_gate").storage[1, 1]), "same_tau_actual_gate")
    a_state, b_state = [call.field("flow", "x").storage[6] for call in calls[:2]]
    distinct = not numeric.exact(a_state, b_state)
    leaves = [reader.field(call, ("reads", "attention_output_post_WO_pre_gate"),
                           (5, 3, 1, 10, 1024), "bfloat16") for call in calls]
    need(all(leaf.storage.dtype == np.uint16 for leaf in leaves), "native_uint16_W_storage")
    slices = [leaf.storage[1, 1, 0] for leaf in leaves]
    result = core.analyze_triplet(*slices, expected_shape=(10, 1024), check=check)
    if not distinct:
        result["representability_status_before_receiver_gate"] = result["status"]
        result["status"] = "NO_GO_NO_DISTINCT_RECEIVER"
    result.update(schema="SAME-W-fixed-PVP-precision-result-v1", input_qualified=True, synthetic=inp.synthetic,
        root_id=ROOT, query_index=QUERY, flow_index=6, layer_index=8,
        reference_receiver="B_shared_pulse", reference_delta="shared_persistent_minus_shared_pulse",
        target_receiver="A_baseline_none_1", graph="Gstar", CONF_is_posthoc=True,
        distinct_A_B_state=distinct, actual_time_FP32=float(calls[0].field("flow", "time").numeric()[6]),
        actual_time_bits_sha256=hashlib.sha256(calls[0].field("flow", "time").storage[6:7].tobytes()).hexdigest(),
        original_input_identity=calls[0].metadata["input_identity"], original_PG_refs=calls[0].metadata["PG_refs"],
        W_source_leaves=[{"role": role, "wholefile": call.ref, "full_leaf": leaf.identity(),
            "slice_axes": [1, 1, 0, "all10", "all1024"], "slice_storage_dtype": "uint16",
            "slice_bytes": view.nbytes, "slice_payload_sha256": hashlib.sha256(view.tobytes(order="C")).hexdigest()}
            for role, call, leaf, view in zip(ROLES, calls, leaves, slices)],
        new_model_calls=0, new_env_steps=0, new_fit=False,
        limitation="Exact representability in archived Gstar only; no native setter, new continuation, action or training qualification.")
    return result


def run_stage(stage, output, *, source_manifest_sha256, deadline_epoch):
    need(sys.platform.startswith("linux"), "real_arrays_cloud_Linux_only")
    need(math.isfinite(deadline_epoch) and deadline_epoch > time.time(), "future_absolute_deadline")
    stage, output = Path(stage), Path(output)
    need(stage.is_dir() and not stage.is_symlink() and not output.exists() and
         stage.resolve() != output.resolve(), "owned_stage_and_new_result_path")
    source_identity = verify_sources(source_manifest_sha256)
    plan = load_inputs()
    reader, numeric = load_helpers()
    def check():
        need(time.time() < deadline_epoch, "precision_absolute_deadline")
    result = None
    try:
        refs = {r["role"]: r for r in plan["input_objects"]}
        check()
        source = json.loads(read_small(stage / refs["source_evidence"]["path"], refs["source_evidence"], 65536))
        query = select_source_query(source, plan)
        planned = reader.load(stage / refs["planned"]["path"], refs["planned"])
        entries = planned_entries(planned, query)
        check()
        inp = reader.load(stage / query["input_ref"]["path"], query["input_ref"])
        by_role = {call["role"]: call for call in query["selected_calls"]}
        calls = []
        for role in ROLES:
            check()
            ref = by_role[role]["ref"]
            calls.append(reader.load(stage / ref["path"], ref))
        need(sum(x.native_bytes for x in [inp, *calls]) <= 128 * 1024**2, "one_original_query_native_128MiB")
        result = analyze_records(inp, calls, entries, query, reader, numeric, check=check)
        result.update(synthetic=False, source_manifest=source_identity,
            source_evidence=refs["source_evidence"], inputs=plan["input_objects"], input_body_bytes=9151111)
    except Exception as error:
        # Safe failure preservation; no payload or arbitrary exception text.
        result = {"schema": "SAME-W-fixed-PVP-precision-result-v1", "analysis_complete": False,
            "status": "INPUT_OR_NUMERIC_FAILURE_NOT_SCIENTIFIC_NEGATIVE", "error_type": type(error).__name__,
            "input_qualified": False, "root_id": ROOT, "query_index": QUERY,
            "source_manifest": source_identity, "new_model_calls": 0, "new_env_steps": 0,
            "same_W_setter_qualified": False, "new_probe_generated": False}
    raw = encode(result)
    need(len(raw) <= RESULT_LIMIT, "small_precision_JSON_256KiB")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        stream.write(raw)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--source-manifest-sha256", required=True)
    parser.add_argument("--deadline-epoch", type=float, required=True)
    args = parser.parse_args()
    result = run_stage(args.stage, args.out, source_manifest_sha256=args.source_manifest_sha256,
                       deadline_epoch=args.deadline_epoch)
    print(json.dumps({k: result[k] for k in ("status", "analysis_complete", "same_W_setter_qualified")}))
    return 0 if result["analysis_complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
