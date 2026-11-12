"""Single fixed-probe geometry audit using frozen six-object PVP inputs.

No transport, credentials, model, simulation, native target output or GPU GO.
Real arrays remain Linux-only. The original precision qualification is rerun
on the same decoded records before the new pure-core geometry computation.
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
OLD = HERE / "frozen_precision"
OLD_MANIFEST = {"bytes": 990, "sha256": "192b4fa8f1cb8987eadab3a74e24ca5c3e1588546d5df93b10f2e23b996f112e"}
PRIOR_REF = {"bytes": 16547, "sha256": "25b1c8b47745460277c329aa909e0f3fe3c177cf9fa5e751dd84b8e1313d9b30"}
OLD_ADAPTER_SHA = "1c4a0586a75fa1cb9ee065b27eaaf5b029d73fd6d3a489cb9e55c43e80c23644"
DYADIC_SHA = "4b7197ed764671979ea5175d930c326427d25baa3f36f4341ef93f9f93690a45"
FROZEN_NAMES = ("adapter.py", "core.py", "native_reader.py", "pvp_numeric.py", "INPUTS.json", "SOURCE-MANIFEST.json")
SOURCE_FILES = {"adapter.py", "core.py", "exact_bf16.py", "ORIGINAL-PRECISION-RESULT.json"} | {
    "frozen_precision/" + n for n in FROZEN_NAMES}
OUTPUT_LIMIT = 256 * 1024
CORE_SCHEMA = "BF16-common-probe-exact-geometry-v1"
SCHEMA = "SAME-W-common-probe-audit-result-v1"
SELECTION = "coordinatewise_closest_same_sign_nonamplified_common_finite_BF16_increment"


class AuditFault(ValueError):
    pass


def need(ok, code):
    if not ok:
        raise AuditFault(code)


def ident(raw):
    return {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def encoded(value):
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n").encode()


def read_small(path, ref, cap=1024**2):
    path = Path(path)
    need(path.is_file() and not path.is_symlink() and path.stat().st_size <= cap, "regular_bounded_small_source")
    raw = path.read_bytes()
    need(ident(raw) == {k: ref[k] for k in ("bytes", "sha256")}, "source_bytes_SHA")
    return raw


def verify_sources(expected_sha):
    need(isinstance(expected_sha, str) and re.fullmatch("[0-9a-f]{64}", expected_sha), "external_manifest_SHA_required")
    path = HERE / "SOURCE-MANIFEST.json"
    need(path.is_file() and not path.is_symlink() and path.stat().st_size <= 32768, "small_source_manifest")
    raw = path.read_bytes()
    need(hashlib.sha256(raw).hexdigest() == expected_sha, "new_source_manifest_SHA")
    manifest = json.loads(raw)
    need(manifest["schema"] == "COMMON-PROBE-fixed-geometry-source-v1", "new_manifest_schema")
    rows = manifest["files"]
    need(len(rows) == len(SOURCE_FILES) and {r["path"] for r in rows} == SOURCE_FILES, "complete_runtime_source_roster")
    need(OLD.is_dir() and not OLD.is_symlink(), "frozen_source_directory_not_symlink")
    for ref in rows:
        read_small(HERE / ref["path"], ref)
    need(ident((OLD / "SOURCE-MANIFEST.json").read_bytes()) == OLD_MANIFEST, "old_manifest_not_rewritten")
    return ident(raw)


def _module(name, path, digest=None):
    path = Path(path)
    raw = path.read_bytes()
    if digest is not None:
        need(hashlib.sha256(raw).hexdigest() == digest, "explicit_import_source_SHA")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    exec(compile(raw, str(path), "exec"), module.__dict__)
    return module


def load_components():
    previous = _module("_common_probe_frozen_precision_adapter", OLD / "adapter.py", OLD_ADAPTER_SHA)
    need(previous.verify_sources(OLD_MANIFEST["sha256"]) == OLD_MANIFEST, "old_precision_source_contract")
    need((OLD / "core.py").read_bytes() == (HERE / "exact_bf16.py").read_bytes(), "same_exact_integer_decoder_bytes")
    exact = _module("_common_probe_frozen_exact_bf16", HERE / "exact_bf16.py", DYADIC_SHA)
    prior_alias = sys.modules.get("exact_bf16")
    sys.modules["exact_bf16"] = exact
    try:
        core = _module("_common_probe_geometry_core", HERE / "core.py")
    finally:
        if prior_alias is None:
            sys.modules.pop("exact_bf16", None)
        else:
            sys.modules["exact_bf16"] = prior_alias
    return previous, core


def prior_result():
    prior = json.loads(read_small(HERE / "ORIGINAL-PRECISION-RESULT.json", PRIOR_REF, 65536))
    need(prior["analysis_complete"] is True and prior["input_qualified"] is True and prior["synthetic"] is False and
         prior["status"] == "NO_GO_NATURAL_SAME_DELTA_UNREPRESENTABLE" and prior["counts"]["A_target_nonrepresentable"] == 623 and
         prior["counts"]["A_target_representable"] == 9617 and prior["counts"]["reference_delta_nonzero"] == 10091,
         "original_actual_precision_result")
    return prior


def compare_prior(actual, expected):
    """Exact stored metadata/count identity; no float tolerance or reinterpretation."""
    keys = ("status", "analysis_complete", "input_qualified", "synthetic", "root_id", "query_index", "flow_index", "layer_index",
        "reference_receiver", "reference_delta", "target_receiver", "graph", "CONF_is_posthoc", "distinct_A_B_state",
        "actual_time_bits_sha256", "original_input_identity", "original_PG_refs", "W_source_leaves", "counts", "per_position")
    for key in keys:
        need(actual[key] == expected[key], "reproduce_original_precision_" + key)
    return {"status": "EXACT_REPRODUCTION_OF_ORIGINAL_PRECISION_RESULT", "compared_fields": list(keys),
            "counts": actual["counts"], "original_result": PRIOR_REF,
            "original_technical_no_go_preserved": True}


def analyze_bound_records(inp, calls, entries, query, reader, numeric, old, core, expected, *, check=lambda: None):
    """Pure record entry for synthetic tests; production binds original files first."""
    check()
    reproduced = old.analyze_records(inp, calls, entries, query, reader, numeric, check=check)
    reproduction = compare_prior(reproduced, expected)
    need(reproduced["distinct_A_B_state"] is True, "distinct_legal_receiver_states")
    leaves = [reader.field(call, ("reads", "attention_output_post_WO_pre_gate"), (5, 3, 1, 10, 1024), "bfloat16") for call in calls]
    bits = [leaf.storage[1, 1, 0] for leaf in leaves]
    geometry = core.analyze_triplet(*bits, expected_shape=(10, 1024), check=check)
    need(geometry["schema"] == CORE_SCHEMA and geometry["analysis_complete"] is True and geometry["shape"] == [10, 1024] and
         geometry["selection_rule"] == SELECTION, "one_fixed_geometry_rule_and_support")
    counts = geometry["full"]["counts"]
    source_counts = reproduced["counts"]
    for current, previous in (("coordinates", "coordinates"), ("reference_nonzero", "reference_delta_nonzero"),
                             ("reference_zero", "reference_delta_zero"), ("natural_common_representable", "A_target_representable"),
                             ("natural_not_common_representable", "A_target_nonrepresentable")):
        need(counts[current] == source_counts[previous], "new_core_matches_original_" + current)
    need(counts["sign_reversals"] == counts["magnitude_increases"] == 0 and
         geometry["same_W_setter_qualified"] is False and geometry["model_or_behavior_effect_evaluated"] is False and
         geometry["native_target_arrays_returned"] is False and geometry["scientific_acceptance_threshold"] is None,
         "no_sign_flip_amplification_targets_or_scientific_GO")
    return {"schema": SCHEMA, "status": "GEOMETRY_AUDIT_COMPLETE", "analysis_complete": True,
        "synthetic": inp.synthetic, "input_qualified": True, "precision_reproduction": reproduction,
        "scope": {"root_id": old.ROOT, "query_index": old.QUERY, "flow_index": 6, "layer_index": 8,
            "A": "baseline_none_1", "B_before": "shared_pulse", "B_after": "shared_persistent", "CONF_is_posthoc": True},
        "source_identities": {k: reproduced[k] for k in ("W_source_leaves", "actual_time_bits_sha256", "original_input_identity", "original_PG_refs")},
        "geometry": geometry, "same_W_setter_qualified": False, "GPU_auto_GO": False,
        "scientific_acceptance_threshold": None, "new_model_calls": 0, "new_env_steps": 0,
        "model_or_behavior_effect_evaluated": False, "native_targets_written": False,
        "interpretation": "One artificial common increment geometry only; neither functional equivalence to natural E nor receiver causal efficacy is measured."}


def run_stage(stage, output, *, source_manifest_sha256, deadline_epoch):
    need(sys.platform.startswith("linux"), "real_native_arrays_cloud_Linux_only")
    need(type(deadline_epoch) in (float, int) and math.isfinite(deadline_epoch) and deadline_epoch > time.time(), "future_absolute_deadline")
    stage, output = Path(stage), Path(output)
    need(stage.is_dir() and not stage.is_symlink() and not output.exists(), "staged_inputs_and_exclusive_output")
    source_manifest = verify_sources(source_manifest_sha256)
    old, core = load_components()
    plan = old.load_inputs()
    prior = prior_result()
    reader, numeric = old.load_helpers()
    consumed = []
    def check():
        need(time.time() < deadline_epoch, "geometry_absolute_deadline")
    def load(ref):
        check()
        record = reader.load(stage / ref["path"], ref)
        consumed.append({k: ref[k] for k in ("path", "bytes", "sha256", "uri")})
        return record
    try:
        refs = {ref["role"]: ref for ref in plan["input_objects"]}
        check()
        source_raw = old.read_small(stage / refs["source_evidence"]["path"], refs["source_evidence"], 65536)
        consumed.append({k: refs["source_evidence"][k] for k in ("path", "bytes", "sha256", "uri")})
        query = old.select_source_query(json.loads(source_raw), plan)
        planned = load(refs["planned"])
        entries = old.planned_entries(planned, query)
        inp = load(query["input_ref"])
        selected = {call["role"]: call for call in query["selected_calls"]}
        calls = [load(selected[role]["ref"]) for role in old.ROLES]
        need(sum(x.native_bytes for x in [inp, *calls]) <= 128 * 1024**2, "single_original_query_native_128MiB")
        need(len(consumed) == 6 and sum(ref["bytes"] for ref in consumed) == 9151111, "exact_six_wholefile_reads_once")
        result = analyze_bound_records(inp, calls, entries, query, reader, numeric, old, core, prior, check=check)
        result.update(source_manifest=source_manifest, frozen_precision_manifest=OLD_MANIFEST,
                      original_precision_result=PRIOR_REF, input_objects=consumed, input_body_bytes=9151111,
                      decoded_native_records=5, decode_strategy="once_then_reuse_for_old_qualification_and_new_geometry")
    except Exception as error:
        result = {"schema": SCHEMA, "status": "INPUT_OR_GEOMETRY_FAILURE_NOT_SCIENTIFIC_NEGATIVE", "analysis_complete": False,
            "synthetic": False, "input_qualified": False, "error_type": type(error).__name__,
            "source_manifest": source_manifest, "original_precision_result": PRIOR_REF,
            "consumed_verified_refs": consumed, "same_W_setter_qualified": False, "GPU_auto_GO": False,
            "native_targets_written": False, "new_model_calls": 0, "new_env_steps": 0}
    raw = encoded(result)
    need(len(raw) <= OUTPUT_LIMIT, "single_small_audit_JSON_256KiB")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as handle:
        handle.write(raw)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--source-manifest-sha256", required=True)
    parser.add_argument("--deadline-epoch", type=float, required=True)
    args = parser.parse_args()
    result = run_stage(args.stage, args.out, source_manifest_sha256=args.source_manifest_sha256, deadline_epoch=args.deadline_epoch)
    print(json.dumps({key: result[key] for key in ("status", "analysis_complete", "same_W_setter_qualified", "GPU_auto_GO")}))
    return 0 if result["analysis_complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
