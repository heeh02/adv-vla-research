"""Source-qualified saved DEV inputs. Real arrays stay on Linux cloud.

load() has no model, simulator, network, or encoding calls. encode_loaded()
is an explicit later native-runner helper; importing this module never calls it.
"""
from dataclasses import dataclass, fields, is_dataclass
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np

HERE = Path(__file__).resolve().parent
KEYS = tuple((f"SDRU-CDEV-T0{t}-R0{r}", q)
             for t in range(2) for r in range(2) for q in (11, 22))
ARM_MAP = {"S": "shared", "R": "random_matched", "P": "shared"}


def need(ok, code):
    if not ok:
        raise ValueError(code)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def pinned(name):
    ref = next(r for r in json.loads((HERE/"FROZEN-SOURCE-PINS.json").read_text())["files"]
               if r["path"] == name+".py")
    path = HERE/"frozen"/ref["path"]
    raw = path.read_bytes()
    need(len(raw) == ref["bytes"] and sha(raw) == ref["sha256"], "frozen_helper_byte_identity")
    module_name = "_rss_dev_pulse_input_"+name
    if module_name not in sys.modules:
        spec = importlib.util.spec_from_file_location(module_name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    return sys.modules[module_name]


def exact(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()


def explicit(value):
    if is_dataclass(value):
        return {f.name: explicit(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, dict):
        return {k: explicit(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [explicit(v) for v in value]
    return np.asarray(value) if hasattr(value, "dtype") else value


def identity(value):
    return pinned("retained_input_hash").tree_id(explicit(value))


def array_id(value):
    return pinned("retained_input_hash").array_id(value)


def value(leaf):
    """Materialize only the explicitly selected source codec's typed leaves."""
    N, O = pinned("native_reader"), pinned("parent_reader")
    if isinstance(leaf, N.Native):
        a = leaf.storage
        logical = leaf.dtype
    elif isinstance(leaf, O.NativeArray):
        a = leaf.storage()
        logical = leaf.dtype
    elif isinstance(leaf, dict):
        return {k: value(v) for k, v in leaf.items()}
    elif isinstance(leaf, list):
        return [value(v) for v in leaf]
    else:
        return leaf
    if logical == "bfloat16":
        import ml_dtypes
        a = a.view(ml_dtypes.bfloat16)
    a = np.array(a, copy=True)
    need(a.dtype.kind in "biufcUS" or a.dtype.name == "bfloat16", "supported_typed_leaf")
    if a.dtype.kind not in "US":
        need(np.isfinite(a).all(), "finite_typed_leaf")
    a.setflags(write=False)
    return a


@dataclass(frozen=True)
class Branch:
    observation: object
    decode_states: object


@dataclass(frozen=True)
class SavedInput:
    root_id: str
    query_index: int
    raw_observations: dict
    prompt: str
    noise: np.ndarray
    decode_state: np.ndarray
    encoded_identities: dict
    encoded_audits: dict
    old_KT: np.ndarray
    old_VT: np.ndarray
    prefix_mask: np.ndarray
    prefix_positions: np.ndarray
    modified_v: dict
    actual_E: dict
    dose: dict
    old_endpoints: dict
    old_calls: dict
    sources: dict
    source_identity: str
    model_binding: dict
    synthetic: bool = False

    @property
    def key(self):
        return self.root_id, self.query_index


class Staged:
    def __init__(self, stage, manifest, check, synthetic):
        need(synthetic or sys.platform.startswith("linux"), "real_arrays_cloud_Linux_only")
        self.stage = Path(stage).resolve()
        self.manifest, self.check, self.synthetic = manifest, check, synthetic
        objects = manifest["input_objects"]
        need(0 < len(objects) <= 800 and sum(r["bytes"] for r in objects) <= 512*1024**2,
             "bounded_exact_input_roster")
        self.refs = {r["uri"]: r for r in objects}
        need(len(self.refs) == len(objects), "unique_input_URIs")
        need(len({r["path"] for r in objects}) == len(objects), "unique_stage_paths")
        need(len(objects) == manifest["input_object_count"] and
             sum(r["bytes"] for r in objects) == manifest["input_bytes"], "exact_manifest_totals")
        self.receipts, self.cache = {}, {}

    def path(self, ref, codec):
        self.check()
        bound = self.refs.get(ref["uri"])
        need(bound is not None and bound["codec"] == codec and
             all(bound[k] == ref[k] for k in ("bytes", "sha256")), "exact_input_ref_and_codec")
        name = bound["path"]
        need(Path(name).name == name and name not in ("", ".", ".."), "flat_stage_path")
        path = self.stage/name
        need(path.is_file() and not path.is_symlink(), "regular_staged_file")
        raw = path.read_bytes()
        need(len(raw) == ref["bytes"] and sha(raw) == ref["sha256"], "whole_staged_input_SHA")
        self.receipts[ref["uri"]] = {k: bound[k] for k in ("uri", "bytes", "sha256", "codec")}
        return path

    def document(self, ref, codec="plain_json"):
        return json.loads(self.path(ref, codec).read_bytes())

    def record(self, ref, codec):
        key = ref["uri"], codec
        if key not in self.cache:
            path = self.path(ref, codec)
            if codec == "parent_values_v1":
                record = pinned("parent_reader").read_member(path, ref, synthetic=self.synthetic)
            elif codec == "native_arrays_v1":
                record = pinned("native_reader").load(path, ref, synthetic=self.synthetic)
            else:
                raise ValueError("explicit_original_codec_required")
            self.cache[key] = record
        return self.cache[key]

    def archive(self, archive, binding):
        ready, ack = archive["ready"], archive["ack"]
        # The original archive verifier checks every READY/ACK member identity,
        # even though only this readset's selected bodies are restored.
        original = pinned("parent_reader").ArchiveQuery(
            self.path(ready, "archive_control").read_bytes(),
            self.path(ack, "archive_control").read_bytes(),
            ready_ref=ready, ack_ref=ack, expected_binding=binding,
            restore_exact=lambda _: (_ for _ in ()).throw(ValueError("archive_body_read_not_requested")),
            synthetic=self.synthetic)
        available = {r["path"]: r for r in original.members}
        for ref in archive["members"]:
            old = available.get(ref["path"])
            need(old and all(old[k] == ref[k] for k in ("bytes", "sha256", "object_key")),
                 "selected_member_bound_to_original_READY_ACK")
        return True


def _native_array(record, keys, shape, dtype):
    leaf = record.field(*keys)
    need(leaf.shape == shape and leaf.dtype == dtype, "original_native_ABI_"+".".join(keys))
    return value(leaf)


def _ref_same(a, b):
    return all(a.get(k) == b.get(k) for k in ("path", "bytes", "sha256"))


def _dose(edits, vt, mask):
    """Audit frozen native payloads; never call the edit constructor."""
    vmods, effects, report = {}, {}, {}
    valid = mask[0]
    for arm in ("shared", "random_matched"):
        vm = _native_array(edits, ("Vmod", arm), vt.shape, "bfloat16")
        effect = _native_array(edits, ("actual_E", arm), vt.shape, "float32")
        need(exact(vm.astype(np.float32)-vt.astype(np.float32), effect),
             "actual_E_exact_native_payload_difference")
        need(exact(vm[:, ~valid], vt[:, ~valid]), "native_edit_padding_unchanged")
        native_report = edits.metadata["arms"][arm]
        helper = pinned("edits_reference")
        need(native_report["Vmod"] == helper.aid(vm) and
             native_report["actual_E"] == helper.aid(effect), "original_frozen_edit_typed_identity")
        token_norm = np.linalg.norm(effect[0, :, 0].astype(np.float64), axis=-1)
        global_norm = float(np.linalg.norm(effect[0, valid, 0].astype(np.float64)))
        need(np.isclose(global_norm, native_report["actual_norm"], rtol=1e-12, atol=0) and
             bool(not np.any(effect)) == native_report["zero_edit"], "original_native_dose_report")
        vmods[arm], effects[arm] = vm, effect
        report[arm] = {
            "Vmod": helper.aid(vm), "actual_E": helper.aid(effect),
            "token_norm": token_norm.tolist(), "global_norm": global_norm,
            "zero_edit": bool(not np.any(effect)), "padding_unchanged": True}
    rm = value(edits.arrays["random_match"])
    need(np.allclose(rm["token_target_norm"], np.array(report["shared"]["token_norm"]), rtol=1e-12, atol=0) and
         np.allclose(rm["token_actual_norm"], np.array(report["random_matched"]["token_norm"]), rtol=1e-12, atol=0),
         "frozen_actual_token_dose_fields")
    need(exact(rm["token_residual"], rm["token_actual_norm"]-rm["token_target_norm"]),
         "frozen_dose_residual_definition")
    required = {"token_target_norm", "token_actual_norm", "token_residual", "token_scale",
                "zero_target", "zero_direction_nonzero_target", "unbracketed"}
    need(set(rm) == required, "complete_original_random_match_arrays")
    need(edits.metadata["random_match"]["exact_norm_match_not_asserted"] is True,
         "original_dose_equality_not_invented")
    report["random_match"] = {k: np.asarray(v).tolist() for k, v in rm.items()}
    report["random_match_metadata"] = edits.metadata["random_match"]
    report["global_random_minus_shared_norm"] = (
        report["random_matched"]["global_norm"]-report["shared"]["global_norm"])
    report["exact_actual_dose_match"] = bool(np.all(rm["token_residual"][valid] == 0))
    report["numeric_replay_tolerance"] = {
        "norm_reduction_rtol": 1e-12, "atol": 0,
        "scope": "FP64 arithmetic replay only; never a shared/random dose-equality gate"}
    report["redraw_or_rescale"] = False
    return vmods, effects, report


def original_query_identity(native):
    """Replay the exact previous typed query/edit hash domains without fitting."""
    P = pinned("taskblind_prediction")
    audit, c, t, edits = (native[k] for k in ("audit", "natural_C", "natural_T", "edits"))
    source = audit.metadata["source"]
    def leaf(record, *keys):
        v = record.field(*keys)
        return P.Native(v.storage, v.dtype, sha(v.storage.tobytes()))
    def sliced(record, keys, index):
        v = record.field(*keys)
        a = v.storage[index]
        return P.Native(a, v.dtype, sha(a.tobytes()))
    flow = {side+"_"+site: sliced(rec, ("flow", kind), j)
            for side, rec in (("C", c), ("T", t))
            for site, kind, j in (("x0", "x", 0), ("v0", "velocity_native", 0),
                                 ("x5", "x", 5), ("v5", "velocity_native", 5))}
    ref = lambda r: {k: r.ref[k] for k in ("path", "bytes", "sha256")}
    sid = P.sha(P.encoded({
        "input_source_identity": source["source_identity"],
        "natural_C_receipt": ref(c), "natural_T_receipt": ref(t),
        "layer_index": 8, "first_read_call": 5, "flow_sites": [0, 5]}))
    times = np.stack([t.field("flow", "time").storage[j] for j in (0, 5)])
    weak = np.stack([t.field("flow", "time_weak_type").storage[j] for j in (0, 5)])
    query = P.Query(
        audit.metadata["root_id"], audit.metadata["query_index"], source["env_step_index"], sid,
        leaf(t, "first_read", "probabilities_native"), leaf(t, "first_read", "mask"),
        leaf(t, "prefix_mask"), leaf(t, "prefix_positions"), leaf(t, "first_read", "suffix_positions"),
        leaf(c, "receiver_L9_V"), leaf(t, "receiver_L9_V"), leaf(t, "first_read", "attention_gate"),
        flow, leaf(audit, "consumed_decode_state"),
        P.Native(times, "float32", sha(times.tobytes())), P.Native(weak, "bool", sha(weak.tobytes())),
        {arm: leaf(edits, "Vmod", arm) for arm in P.ARMS})
    query.validate(SimpleNamespace(roots={query.root_id: None}))
    return {
        arm: {"input_identity": query.fingerprint(), "edit_identity": query.edit_identity(arm)}
        for arm in ("shared", "random_matched")}


def bind_snapshot(spec, root, parents, native, model, *, synthetic=False):
    """Pure CPU semantic binding; accepts already typed and SHA-qualified records."""
    rid, q = spec["root_id"], spec["query_index"]
    need((rid, q) in KEYS and root["root_id"] == rid, "fixed_eight_DEV_snapshots")
    audit, c, t, edits = (native[k] for k in ("audit", "natural_C", "natural_T", "edits"))
    for key, event in (("audit", "GPU_INPUT_AUDIT"), ("edits", "FROZEN_QUERY_EDITS"),
                       ("natural_C", "CALL_OUTPUT"), ("natural_T", "CALL_OUTPUT"),
                       ("effect_shared", "CALL_OUTPUT"), ("effect_random_matched", "CALL_OUTPUT")):
        rec = native[key]
        need(rec.event == event and rec.metadata["root_id"] == rid and
             rec.metadata["query_index"] == q, "native_event_root_query_join")
    source = audit.metadata["source"]
    need(source == spec["original_parent_source"] and
         source["root_id"] == rid and source["query_index"] == q and source["env_step_index"] == 5*q,
         "exact_original_parent_source_proof")
    need(source["actual_prompt_sha256"] == sha(root["prompt"].encode()),
         "source_original_prompt_digest")
    for kind, ref in root["proofs"].items():
        need(all(source["original_refs"][kind][k] == ref[k] for k in ("uri", "bytes", "sha256")),
             "source_original_model_runtime_ref")
    for ref in root["prompt_archive"]["members"]:
        event = ref["path"].split("-", 1)[1].rsplit(".", 1)[0]
        need(source["actual_prompt_records"][event] == "sha256:"+ref["sha256"],
             "source_original_prompt_record_ref")
    encoded, audits = {}, {}
    for side in ("C", "T"):
        checks = audit.metadata["checks"][side]
        need(checks["exact"] is True and len(checks["checks"]) == 2 and all(
            x["exact"] is True and x["noise_exact"] is True and
            x["input_binding_before"] == x["input_binding_after"] == x["GPU_encoded_input_binding"]
            for x in checks["checks"]), "actual_GPU_encoding_audit")
        encoded[side], audits[side] = checks["branch_identity"], checks
    need(audit.metadata["checks"]["decode_states_equal"] is True, "C_T_consumed_decode_equal")
    noise = _native_array(audit, ("explicit_noise",), (1, 10, 32), "float32")
    state = _native_array(audit, ("consumed_decode_state",), (1, 32), "float32")
    pair, clock, sensor = (parents[k][0] for k in ("PAIRED_INPUT", "QUERY_REACHED", "SENSOR_PROVENANCE"))
    for rec in (pair, clock):
        need(rec.metadata["query_index"] == q and rec.metadata["env_step_index"] == 5*q and
             rec.metadata["absolute_env_step_index"] == 10+5*q, "original_physical_clock")
    need(pair.receipt == source["pair_receipt"] and clock.receipt == source["query_clock_receipt"],
         "raw_pair_and_clock_receipt_join")
    need(exact(value(pair.values["query_time_actual"]), value(clock.values["query_time_actual"])),
         "actual_query_clock_payload")
    need(pair.metadata.get("full_runtime_snapshot_id") is None and
         pair.metadata.get("full_replay_qualified") is False, "saved_input_not_physics_restore")
    need(sensor.metadata["query_index"] == q and sensor.metadata["accepted"] is True,
         "original_sensor_provenance")
    views = pair.values["views"]
    need(set(views) == {"clean", "nativeperturb"}, "original_C_T_view_roles")
    raw = {s: value(views[k]["observation"]) for s, k in (("C", "clean"), ("T", "nativeperturb"))}
    required_obs = {"agentview_image", "robot0_eye_in_hand_image", "robot0_eef_pos",
                    "robot0_eef_quat", "robot0_gripper_qpos"}
    for side, condition in (("C", "clean"), ("T", "nativeperturb")):
        need(required_obs <= set(raw[side]), "complete_original_raw_observation")
        rows = [r for r in parents["NEURAL_RNG_AUDIT"] if
                r.metadata.get("condition") == condition and r.metadata.get("kind") == "pg_ae" and
                r.metadata.get("noise_role") == "diagnostic_fixed" and r.metadata.get("noise_label") == 0]
        need(len(rows) == 2, "two_original_neural_audits_per_side")
        for rec in rows:
            m = rec.metadata
            need(m["root_trajectory_id"] == rid and m["query_index"] == q,
                 "original_neural_audit_root_query")
            aa = m["audit"]
            need(aa["input_binding_before"] == aa["input_binding_after"] and aa["input_unchanged"] is True and
                 aa["explicit_model_noise_identity"] == array_id(noise), "original_neural_noise_identity")
            need(any(x["GPU_encoded_input_binding"] == aa["input_binding_before"]
                     for x in audits[side]["checks"]), "old_and_later_encoding_audit_join")
    for role in ("natural_C", "natural_T", "effect_shared", "effect_random_matched"):
        rec = native[role]
        need(rec.metadata["role"] == role and rec.metadata["source_identity"] == source["source_identity"],
             "native_original_role_source")
        need(rec.metadata["synthetic"] is False or synthetic, "real_native_provenance")
        need(exact(value(rec.field("flow", "x"))[0], noise), "all_old_calls_same_noise")
        technical = rec.metadata["technical"]
        need(technical["gates"] and all(v is True for v in technical["gates"].values()) and
             technical["sampling_abi"]["num_steps"] == 10, "original_full10_native_gates")
        _native_array(rec, ("raw32",), (1, 10, 32), "float32")
        _native_array(rec, ("adapter7",), (1, 10, 7), "float64")
    mask = value(t.field("prefix_mask"))
    need(mask.dtype == bool and mask.shape == (1, 968) if not synthetic
         else mask.dtype == bool and mask.ndim == 2 and mask.shape[0] == 1, "original_prefix_ABI")
    n = mask.shape[1]
    positions = _native_array(t, ("prefix_positions",), (1, n), "int32")
    need(exact(positions, np.cumsum(mask, axis=-1, dtype=np.int32)-1), "native_prefix_positions")
    kt = _native_array(t, ("receiver_L9_K",), (1, n, 1, 256), "bfloat16")
    vt = _native_array(t, ("receiver_L9_V",), (1, n, 1, 256), "bfloat16")
    need(exact(mask, value(c.field("prefix_mask"))) and
         exact(positions, value(c.field("prefix_positions"))), "paired_prefix_layout")
    vmods, effects, dose = _dose(edits, vt, mask)
    for arm in ("shared", "random_matched"):
        rec = native["effect_"+arm]
        m = rec.metadata
        need(_ref_same(m["baseline_refs"]["C"], c.ref) and
             _ref_same(m["baseline_refs"]["T"], t.ref) and
             _ref_same(m["edit_ref"], edits.ref), "old_effect_baseline_payload_refs")
        sel = m["technical"]["selectors"]
        need(sel == {"kv_mask": [False, True], "layer_mask": [i == 8 for i in range(18)],
                     "flow_mask": [i >= 5 for i in range(10)], "use_vmod": True},
             "old_persistent_not_pulse")
        for name, actual in (("receiver_V", vt), ("Vmod_argument", vmods[arm]),
                             ("actually_used_V_at_j5", vmods[arm]),
                             ("actual_E_used_minus_receiver_FP32", effects[arm])):
            need(m["actual_edit"][name] == pinned("edits_reference").aid(actual),
                 "old_actually_consumed_edit_identity")
    actual_query_ids = original_query_identity(native)
    need(actual_query_ids == spec["pulse_label_identities"],
         "exact_original_query_and_edit_fingerprints")
    zeros = np.zeros_like(effects["shared"])
    zeros.setflags(write=False)
    sources = {
        "native": {k: {f: rec.ref[f] for f in ("path", "bytes", "sha256")} for k, rec in native.items()},
        "parent_source": source, "model": root["proofs"]["model"],
        "root_binding": root["binding"], "original_pulse_label_ids": actual_query_ids}
    return SavedInput(
        rid, q, raw, root["prompt"], noise, state, encoded, audits, kt, vt, mask, positions,
        {**{k: vmods[v] for k, v in ARM_MAP.items()}, "own_null": vt},
        {**{k: effects[v] for k, v in ARM_MAP.items()}, "own_null": zeros}, dose,
        {k: value(rec.field("adapter7")) for k, rec in native.items()
         if k in ("natural_C", "natural_T", "effect_shared", "effect_random_matched")},
        {k: rec for k, rec in native.items() if rec.event == "CALL_OUTPUT"},
        sources, source["source_identity"], model, synthetic)


def load(stage, manifest=None, check=lambda: None, synthetic=False):
    manifest = json.loads((HERE/"INPUTS.json").read_text()) if manifest is None else manifest
    need(manifest["schema"] == "rss-dev-pulse-input-readset-v1", "exact_input_manifest_schema")
    need([(s["root_id"], s["query_index"]) for s in manifest["snapshots"]] == list(KEYS),
         "all_eight_fixed_snapshots_no_survivor_selection")
    store = Staged(stage, manifest, check, synthetic)
    for archive in manifest["native_archives"]:
        need(archive["binding"] == manifest["source_binding"], "original_native_source_binding")
        store.archive(archive, manifest["source_binding"])
    roots, models = {}, {}
    for root in manifest["roots"]:
        rid = root["root_id"]
        need(rid not in roots, "unique_parent_root")
        roots[rid] = root
        docs = {k: store.document(v) for k, v in root["proofs"].items()}
        model = docs["model"]
        config = model["effective_model_config"]
        need(model["model_bound"] is True and model["adapter_bound"] is True and
             config["pi05"] is True and config["dtype"] == "bfloat16" and
             config["action_dim"] == 32 and config["action_horizon"] == 10 and
             config["max_token_len"] == 200 and config["discrete_state_input"] is False,
             "original_native_model_and_adapter_config")
        models[rid] = model
        pa = root["prompt_archive"]
        store.archive(pa, root["binding"])
        records = {r.event: r for r in
                   (store.record(ref, "parent_values_v1") for ref in pa["members"])}
        need(set(records) == {"ACTUAL_PROMPT_BINDING", "ROOT_RUNTIME_BINDING"}, "two_prompt_bindings")
        pb, rb = (records[k].metadata for k in ("ACTUAL_PROMPT_BINDING", "ROOT_RUNTIME_BINDING"))
        need(pb["actual_prompt"] == rb["actual_prompt"] == root["prompt"] and
             pb["actual_prompt_utf8_sha256"] == sha(root["prompt"].encode()), "original_prompt_exact")
        need(rb["environment_source"]["root_trajectory_id"] == rid and
             rb["environment_source"]["current_root_initial_selection"] == root["actual_initial_selection"],
             "actual_initial_selection_binding")
    identity_keys = ("checkpoint_artifact_id", "normalization_sha256", "effective_model_config")
    need(all(all(model[k] == next(iter(models.values()))[k] for k in identity_keys)
             for model in models.values()), "same_original_model_normalizer_across_roots")
    snapshots = []
    for spec in manifest["snapshots"]:
        check()
        root = roots[spec["root_id"]]
        query = next(q for q in root["queries"] if q["query_index"] == spec["query_index"])
        store.archive(query, root["binding"])
        parents = {}
        for ref in query["members"]:
            rec = store.record(ref, "parent_values_v1")
            parents.setdefault(rec.event, []).append(rec)
        need(all(len(parents.get(k, [])) == 1 for k in
                 ("PAIRED_INPUT", "QUERY_REACHED", "SENSOR_PROVENANCE")), "unique_parent_snapshot_records")
        native = {k: store.record(ref, "native_arrays_v1") for k, ref in spec["records"].items()}
        snapshots.append(bind_snapshot(spec, root, parents, native, models[root["root_id"]],
                                       synthetic=synthetic))
    need(len(store.receipts) == manifest["input_object_count"], "all_roster_objects_consumed")
    proof = {
        "schema": "rss-dev-pulse-input-qualification-v1", "analysis_complete": True,
        "synthetic": synthetic, "input_source_qualified": True,
        "native_runtime_qualified": False, "new_pulse_labels": 0,
        "snapshots": [{
            "root_id": s.root_id, "query_index": s.query_index, "source_identity": s.source_identity,
            "noise": array_id(s.noise), "decode_state": array_id(s.decode_state),
            "encoded_identities": s.encoded_identities, "raw_C_T_identity": identity(s.raw_observations),
            "prefix_KT": array_id(s.old_KT), "prefix_VT": array_id(s.old_VT),
            "source_refs": s.sources, "actual_dose": s.dose,
            "old_endpoint_identity": {k: array_id(v) for k, v in s.old_endpoints.items()},
        } for s in snapshots],
        "input_objects": list(store.receipts.values()), "input_bytes": manifest["input_bytes"],
        "model_calls": 0, "environment_steps": 0, "GPU": 0,
        "pulse_effect_or_firstwrite_equality_measured": False,
        "full_physics_restore_qualified": False,
        "future_native_requirements": [
            "loaded model/source/normalizer identity", "encode_loaded identities",
            "current_prefix_check and actually-used V identity", "72-call exact qualification roster"],
    }
    return {"snapshots": snapshots, "proof": proof}


def encode_loaded(campaign, saved):
    """Later native-only helper. No call from load(), import or CPU tests."""
    need(sys.platform.startswith("linux"), "native_encode_only_cloud_Linux")
    before, rng = identity(saved.raw_observations), identity(np.random.get_state())
    branches = {}
    for side in ("C", "T"):
        observation, batch = campaign.encode(
            [copy.deepcopy(saved.raw_observations[side])], [saved.prompt])
        branch = Branch(observation, batch["state"])
        need(identity(branch) == saved.encoded_identities[side], "actual_encoded_branch_identity")
        need(exact(batch["state"], saved.decode_state), "actual_consumed_decode_state")
        binding = identity({"branch": explicit(branch), "noise": saved.noise})
        need(all(binding == row["GPU_encoded_input_binding"]
                 for row in saved.encoded_audits[side]["checks"]), "actual_encoded_noise_binding")
        branches[side] = branch
    need(identity(saved.raw_observations) == before and
         identity(np.random.get_state()) == rng, "encode_input_RNG_unchanged")
    return branches


def current_prefix_check(saved, metadata, role):
    """Check a returned native call, not an inference from the CPU source audit."""
    need(role in ("N", "S", "R", "P", "own_null"), "declared_payload_role")
    pg = metadata["pg"]["T"]
    need(exact(np.asarray(pg["cache_k"])[1], saved.old_KT) and
         exact(np.asarray(pg["cache_v"])[1], saved.old_VT), "same_original_KT_VT_current_runtime")
    need(exact(metadata["prefix_mask"], saved.prefix_mask) and
         exact(metadata["prefix_positions"], saved.prefix_positions), "same_original_prefix_addresses")
    if role != "N":
        vm, effect = saved.modified_v[role], saved.actual_E[role]
        need(exact(metadata["Vmod_argument"], vm), "actual_argument_frozen_Vmod")
        need(exact(vm.astype(np.float32)-np.asarray(pg["cache_v"])[1].astype(np.float32), effect),
             "same_frozen_actual_E_current_runtime")
        need(exact(np.asarray(metadata["reads"]["prefix_value_used"])[0, 1], vm),
             "actually_consumed_j5_Vmod")
    return {"current_prefix_and_payload_exact": True, "role": role}
