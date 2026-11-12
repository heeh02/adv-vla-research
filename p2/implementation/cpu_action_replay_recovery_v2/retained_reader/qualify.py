"""Existing recorded-data qualification, without constructing any environment."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np
import codec as C
import binding as B

HERE = Path(__file__).resolve().parent
ROOT = "SDRU-CDEV-T00-R00"
COUNTS = {"archives": 14, "parent_chunks": 12, "issued_steps": 60,
          "outcomes": 60, "protected_queries": 13, "anchors": 3,
          "cached_native_calls": 3}


def encoded(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                       allow_nan=False) + "\n").encode()


def identity(value):
    value = np.asarray(value)
    C.need(not value.dtype.hasobject, "no_object_array")
    return {"shape": list(value.shape), "dtype": value.dtype.str,
            "bytes": value.nbytes, "sha256": C.sha(value.tobytes(order="C"))}


def same(a, b):
    return identity(a) == identity(b)


def load_pinned(name):
    pins = C.json_read((HERE / "RETAINED-PINS.json").read_bytes())
    row = next(r for r in pins["files"] if r["path"] == "pinned/" + name + ".py")
    path = HERE / row["path"]
    raw = path.read_bytes()
    C.need(len(raw) == row["bytes"] and C.sha(raw) == row["sha256"],
           "exact_retained_reader_source")
    alias = "_q11_retained_" + name
    spec = importlib.util.spec_from_file_location(alias, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    spec.loader.exec_module(module)
    return module


def checked_tree(value):
    """Verify every retained lazy payload, including unused captured fields."""
    if hasattr(value, "storage") and hasattr(value, "descriptor"):
        array = value.storage()
        if array.dtype.kind in "biufc":
            C.need(np.isfinite(array).all(), "finite_saved_numeric_payload")
        return array
    if isinstance(value, dict):
        return {k: checked_tree(v) for k, v in value.items()}
    if isinstance(value, list):
        return [checked_tree(v) for v in value]
    if isinstance(value, float):
        C.need(np.isfinite(value), "finite_saved_scalar")
    return value


def decode_parent(stage, inputs, check, *, synthetic=False):
    """Use the original READY/ACK and typed readers on the declared subset."""
    readout = load_pinned("retained_readout")
    original = load_pinned("original_source_reader")
    canonical = load_pinned("canonical_step_capture")
    source = object.__new__(original.RootRecords)
    source.row = {"root_id": ROOT, "binding": inputs["source_parent_binding"],
                  "env_seed": inputs["source_parent_task"]["env_seed"],
                  "initial_selection": inputs.get("source_parent_initial_selection")}
    source.root_id, source.records, source.initial = ROOT, {}, []
    source.rng_module = canonical
    refs = {r["uri"]: r for r in inputs["input_objects"]}
    base = ("redacted-24fc043a6cae"
            "state-dependent-response-use-20261013-v1/" +
            inputs["source_parent_binding"]["run_id"] + "/")
    checks = []
    seen = set()
    for group in inputs["source_parent_archives"]:
        check()
        q = group["query_index"]
        C.need(q not in seen, "duplicate_source_query_archive")
        seen.add(q)
        prefix = group.get("prefix_uri", group["ready"]["uri"].rsplit("/", 1)[0] + "/")
        C.need(prefix.startswith(base), "original_parent_archive_scope")
        selected = {}
        for row in group["members"]:
            uri = prefix + row["path"]
            C.need(uri == row["uri"] and uri in refs, "exact_selected_parent_URI")
            ref = {**row, "object_key": uri[len(base):]}
            C.need(all(ref[k] == refs[uri][k] for k in ("bytes", "sha256")),
                   "selected_parent_roster_identity")
            selected[row["path"]] = ref
        C.need(len(selected) == len(group["members"]), "unique_selected_parent_members")
        controls = {}
        for name in ("ready", "ack"):
            row = group[name]
            C.need(row["uri"] in refs and all(row[k] == refs[row["uri"]][k]
                   for k in ("bytes", "sha256")), "original_parent_control_reference")
            controls[name] = {**row, "object_key": row["uri"][len(base):]}
        def local(ref):
            return stage / refs[ref["uri"]]["path"]
        def restore(ref):
            selected_ref = selected[ref["path"]]
            C.need(all(selected_ref[k] == ref[k] for k in
                       ("bytes", "sha256", "object_key")), "exact_reader_restore")
            return local(selected_ref)
        query = readout.ArchiveQuery(
            local(controls["ready"]).read_bytes(), local(controls["ack"]).read_bytes(),
            ready_ref=controls["ready"], ack_ref=controls["ack"],
            expected_binding=inputs["source_parent_binding"],
            restore_exact=restore, synthetic=synthetic,
        )
        C.need(query.ready["query_index"] == q, "archive_query_identity")
        actual = {r["path"]: r for r in query.members}
        decoded = []
        for name, ref in selected.items():
            check()
            C.need(name in actual and all(actual[name][k] == ref[k]
                   for k in ("bytes", "sha256", "object_key")), "selected_READY_member")
            record = readout.read_member(local(ref), actual[name], synthetic=synthetic)
            C.need(record.event == ref["event"], "decoded_selected_event")
            for field, required in (("original_name", "required_original_name"),
                                    ("role", "required_role")):
                if required in ref:
                    C.need(record.metadata.get(field) == ref[required],
                           "decoded_original_parent_input_role")
            checked_tree(record.values)
            decoded.append(record)
        if q is None:
            source.initial = decoded
        else:
            source.records[q] = decoded
        checks.append({"query_index": q, "selected_members": len(decoded),
                       "ready": controls["ready"], "ack": controls["ack"],
                       "actual_role": query.ready["role"],
                       "whole_selected_payloads_checked": True})
    C.need(seen == {None, *range(13)} and set(source.records) == set(range(13)),
           "exact_initial_and_q0_to_q12")
    return source, checks


def parent_checks(source, inputs, check):
    initial = [r for r in source.initial if r.event == "ACTUAL_ROOT_INITIAL_VECTOR"]
    C.need(len(initial) == 1, "unique_actual_initial_vector")
    selection = inputs["source_parent_initial_selection"]
    C.need(initial[0].metadata == selection, "exact_actual_initial_selection")
    array = checked_tree(initial[0].values)["initial_vector"]
    expected = selection["native_initial_vector"]
    aid = identity(array)
    C.need(aid["shape"] == expected["shape"] and aid["dtype"] == expected["dtype"] and
           aid["bytes"] == expected["bytes"] and aid["sha256"] == expected["raw_sha256"],
           "exact_actual_initial_vector_payload")
    rows, protected, anchors = [], [], []
    previous = None
    for q in range(13):
        check()
        audit = source.parent_audit(q)
        C.need(audit.get("input_unchanged") is True and
               audit.get("environment_rng_restored") is True and
               audit.get("rng_changed_during_call", {}).get("numpy") is False,
               "original_parent_RNG_preservation")
        proof = source.diagnostic(q)
        C.need(proof.get("before") == proof.get("after_restore") and
               not proof.get("forbidden_changes") and not proof.get("restore_errors") and
               not proof.get("observation_errors"), "original_diagnostic_preservation")
        protected.append({"query_index": q, "source_groups_before": proof["before"],
                          "source_restoration_verified": True})
        if q in (0, 11, 12):
            anchor = source.anchor(q)
            physics = anchor["physical_sim_state"]
            C.need(physics.dtype == np.float64 and physics.shape == (123,) and
                   np.isfinite(physics).all(), "current_query_physics_ABI")
            if previous is not None:
                C.need(same(previous["flat_physics"], physics), "step_to_anchor_continuity")
            anchors.append({"query_index": q, "physics": identity(physics),
                            "observations": {k: identity(v) for k, v in anchor["observation"].items()},
                            "source_receipts": anchor["source_receipts"],
                            "sensor_source_not_used_as_current_state": True})
        if q == 12:
            C.need(not any(r.event in ("PARENT_ACTIONS", "ACTUAL_ISSUED_STEP",
                                      "ACTUAL_STEP_OUTCOME") for r in source.records[q]),
                   "q12_witness_only")
            continue
        chunk = source.parent_actions(q)
        raw = source.parent_raw32(q)
        C.need(chunk.shape == (1, 10, 7) and chunk.dtype == np.float64 and
               raw.shape == (1, 10, 32) and raw.dtype == np.float32 and
               np.isfinite(chunk).all() and np.isfinite(raw).all(), "source_parent_action_ABI")
        for slot in range(5):
            check()
            issued, outcome = source.issued(q, slot), source.outcome(q, slot)
            meta, values = issued["metadata"], issued["values"]
            step = 5 * q + slot + 1
            C.need(meta["query_index"] == q and meta["action_position"] == slot and
                   meta["policy_step"] == step and meta["absolute_env_step_index"] == 10 + step,
                   "original_issued_clock")
            C.need(same(chunk[0, slot], values["action"]), "original_issued_chunk_exact")
            C.need(outcome["query_index"] == q and outcome["action_position"] == slot and
                   outcome["policy_step"] == step, "original_outcome_clock")
            C.need(outcome["done"] is False and outcome["official_success"] is False,
                   "recorded_prefix_nonterminal")
            for phase in ("before", "after"):
                v = values[phase]
                C.need(v["flat_physics"].dtype == np.float64 and
                       v["flat_physics"].shape == (123,), "step_physics_ABI")
                for field, shape in (("sim_time", ()), ("returned_eef_pos", (3,)),
                                     ("returned_eef_quat", (4,))):
                    value = np.asarray(v[field])
                    C.need(value.shape == shape and value.dtype.kind == "f" and
                           np.isfinite(value).all(), "step_witness_ABI")
            if previous is not None:
                C.need(all(same(previous[k], values["before"][k]) for k in
                           ("flat_physics", "sim_time", "returned_eef_pos", "returned_eef_quat")),
                       "consecutive_source_step_continuity")
            else:
                C.need(same(source.anchor(0)["physical_sim_state"], values["before"]["flat_physics"]),
                       "q0_anchor_first_before")
            previous = values["after"]
            rows.append({"query_index": q, "slot": slot, "policy_step": step,
                         "absolute_env_step": step + 10, "issued_matches_parent": True,
                         "source_receipt": issued["source_receipt"], "source_nonterminal": True,
                         "before": {k: identity(v) for k, v in values["before"].items()
                                    if isinstance(v, np.ndarray)},
                         "after": {k: identity(v) for k, v in values["after"].items()
                                   if isinstance(v, np.ndarray)}})
    C.need(len(rows) == 60 and len(protected) == 13 and len(anchors) == 3,
           "complete_selected_prefix_support")
    return {"schema": "paper2-q11-parent-input-proof-v1", "root_id": ROOT,
            "initial_selection": selection, "actual_initial_vector": aid,
            "steps": rows, "protected_queries": protected, "anchors": anchors,
            "q0_RNG_identity": source.rng_id(source.initial_rng()),
            "q12_commands_used": 0, "all_source_continuity_exact": True,
            "reconstructed_environment_qualified": False}


def native_join(stage, inputs, source, check, *, synthetic=False):
    refs = {r["path"]: r for r in inputs["input_objects"]}
    names = inputs["native_provenance_paths"]
    doc = lambda key: C.json_read((stage / names[key]).read_bytes())
    terminal, result = doc("TERMINAL.json"), doc("RESULT.json")
    B.validate_terminal(terminal, result, inputs["native72_plan_entries"])
    model, event = doc("MODEL_BINDING"), doc("SOURCE_BOUND_event")
    C.need(model["model_bound"] is True and model["adapter_bound"] is True and
           all(model[k] == result["model_binding"][k] for k in
               ("normalization_sha256", "loaded_parameter_tree_sha256")),
           "cached_native_model_binding")
    C.need(event["event"] == "SOURCE_BOUND" and
           event["payload"]["source_manifest"]["sha256"] == result["source_manifest_sha256"] and
           event["payload"]["synthetic"] is False and
           all(event["payload"]["model_binding"][k] == refs[names["MODEL_BINDING"]][k]
               for k in ("bytes", "sha256")) and
           all(event["payload"][k] is True for k in
               ("source_bound", "model_bound", "adapter_bound", "inputs_bound")),
           "cached_native_source_binding")
    runtime, child = doc("RUNTIME-RESULT.json"), doc("ARCHIVE-MANIFEST.json")
    C.need(runtime["science_archive"]["complete_for_observed_files"] is True and
           runtime["science_archive"]["errors"] == [], "complete_native_archive")
    archived = runtime["science_archive"]["objects"] + runtime["runtime_evidence_archive"]["objects"]
    child_files = {r["path"]: r for r in child["files"]}
    C.need(len(child_files) == len(child["files"]), "unique_native_manifest")
    spec = inputs["native_snapshot_spec"]
    cached_paths = [v[k] for v in inputs["cached_command_paths"].values()
                    for k in ("compact", "checks")]
    cached_paths += [names["saved_input"], names["TERMINAL.json"], names["RESULT.json"],
                     names["SOURCE_BOUND_event"], names["MODEL_BINDING"]]
    for path in cached_paths:
        ref = refs[path]
        matches = [r for r in archived if r["sha256"] == ref["sha256"] and
                   r["bytes"] == ref["bytes"] and r["receipt"]["uri"] == ref["uri"]]
        C.need(len(matches) == 1 and matches[0]["state"] == "VERIFIED",
               "cached_native_exact_archived_object")
    # Native input is actually decoded this time, unlike the preceding mapping audit.
    input_ref = refs[names["saved_input"]]
    meta, _ = C.typed(stage / input_ref["path"], input_ref, synthetic=synthetic)
    C.need((meta["root_id"], meta["query_index"]) == (ROOT, 11) and
           meta["source_identity"] == inputs["source_identity_domains"]["saved_input_source_identity"],
           "saved_native_input_source")
    old = [r for r in doc("old_source_proof")["all_rows"]
           if r["root_id"] == ROOT and r["query_index"] == 11]
    C.need(len(old) == 1, "unique_historical_query_proof")
    old = old[0]
    domains = inputs["source_identity_domains"]
    C.need(old["query_fingerprint"] == domains["input_identity"] and
           old["input_source_identity"] == domains["historical_query_source_identity"] and
           old["parent_source"]["source_identity"] == domains["saved_input_source_identity"],
           "three_distinct_source_identity_domains")
    anchor = source.anchor(11)
    C.need(old["parent_source"]["pair_receipt"] == anchor["source_receipts"]["pair"] and
           old["parent_source"]["query_clock_receipt"] == anchor["source_receipts"]["clock"],
           "cached_commands_join_exact_parent_q11")
    C.need(old["parent_source"]["actual_initial_selection"] ==
           inputs["source_parent_initial_selection"], "native_parent_initial_selection")
    for role, name in inputs["original_runtime_proof_paths"].items():
        ref = refs[name]
        expected = old["parent_source"]["original_refs"][role]
        C.need(all(ref[k] == expected[k] for k in ("uri", "bytes", "sha256")),
               "original_parent_runtime_proof_reference")
    commands, proofs = {}, {}
    for arm, expected_role in (("N", "baseline_none_1"), ("S", "shared_pulse"), ("R", "random_pulse")):
        check()
        row = inputs["cached_command_paths"][arm]
        entry = row["entry"]
        C.need(entry == spec[arm + "_entry"] and entry in inputs["native72_plan_entries"] and
               (entry["root_id"], entry["query_index"], entry["arm"], entry["role"]) ==
               (ROOT, 11, arm, expected_role), "cached_command_role")
        ref, checks_ref = refs[row["compact"]], refs[row["checks"]]
        cm, leaves = C.typed(stage / ref["path"], ref, synthetic=synthetic)
        B.validate_call(cm, C.json_read((stage / checks_ref["path"]).read_bytes()), entry,
                        spec["input_reference"], ref)
        C.need(cm["provenance"]["source_identity"] == domains["saved_input_source_identity"] and
               cm["provenance"]["native_source_unchanged"] is True, "native_call_parent_source")
        for r in (ref, checks_ref, input_ref):
            # All three are native scientific ledger members, not runtime evidence.
            logical = "ledger/" + r["uri"].rsplit("/", 1)[-1]
            C.need(logical in child_files and all(child_files[logical][k] == r[k]
                   for k in ("bytes", "sha256")), "cached_native_child_manifest_member")
        leaf = B.field(leaves, ("adapter7",), (1, 10, 7), "float64")
        commands[arm] = np.array(leaf.numeric(), copy=True)
        C.need(np.isfinite(commands[arm]).all(), "finite_cached_command")
        proofs[arm] = {"entry": entry, "compact": ref, "checks": checks_ref,
                       "adapter7": identity(commands[arm]),
                       "first5": identity(commands[arm][0, :5])}
    parent = source.parent_actions(11)
    return {"schema": "paper2-q11-native-parent-join-v1", "root_id": ROOT, "query_index": 11,
            "identity_domains": domains, "native_final72_qualification_bound": True,
            "actual_saved_input_body_decoded": True, "original_parent_anchor": anchor["source_receipts"],
            "cached_commands": proofs, "cached_N_repeat_uses_same_body": True,
            "parent_q11_adapter7": identity(parent),
            "cached_N_equals_original_parent_typed": same(commands["N"], parent),
            "cached_N_equals_original_parent_numeric": bool(np.array_equal(commands["N"], parent)),
            "cached_N_equals_parent_is_not_required": True,
            "original_parent_used_for_reconstruction_not_cached_N": True,
            "new_environment_qualified": False}
