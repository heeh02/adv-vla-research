import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import ml_dtypes
import adapter as A
import build_manifest as B


def fixture(folder):
    """Original writer flatten functions generate both actual archive codecs."""
    folder = Path(folder)
    objects = {}
    sequence = 0

    def put(path, raw, codec, key=None):
        nonlocal sequence
        uri = "redacted-3d37898e2c0e"+(key or path)
        ref = {"path": Path(path).name, "bytes": len(raw), "sha256": A.sha(raw),
               "object_key": key or path, "uri": uri}
        local = f"input-{sequence:04d}"+Path(path).suffix
        sequence += 1
        (folder/local).write_bytes(raw)
        objects[uri] = {**{k: ref[k] for k in ("uri", "bytes", "sha256")},
                        "logical_path": ref["path"], "path": local, "codec": codec}
        return ref

    def document(path, data, codec="plain_json", key=None):
        return put(path, A.pinned("parent_reader").encoded(data), codec, key)

    def record(name, metadata, values, codec, key):
        arrays = {}
        writer = A.pinned("parent_writer" if codec == "parent_values_v1" else "native_writer")
        field = "values" if codec == "parent_values_v1" else "arrays"
        tree = writer.flatten({"metadata": metadata, field: values}, arrays=arrays)
        raw = json.dumps(tree, sort_keys=True, allow_nan=False).encode()
        if arrays:
            import io
            buf = io.BytesIO()
            np.savez_compressed(buf, **arrays, __schema__=np.frombuffer(raw, np.uint8))
            raw = buf.getvalue()
        return put(name, raw, codec, key)

    def archive(binding, query, members, nonce):
        base = "queries/"+binding["root_id"]+"/"+(
            "initialization" if query is None else f"query-{query:04d}")+"/"+nonce
        rows = [{k: r[k] for k in ("path", "bytes", "sha256")} for r in members]
        ready = {"schema": "continuous-query-ready-v1", "binding": binding,
                 "sequence": 0, "nonce": nonce, "query_index": query,
                 "role": "partial" if query is None else "complete",
                 "members": rows, "total_bytes": sum(r["bytes"] for r in rows)}
        rr = document(base+"/READY-v1.json", ready, "archive_control", base+"/READY-v1.json")
        ack = {"schema": "continuous-query-ack-v1", "binding": binding, "status": "VERIFIED",
               "sequence": 0, "nonce": nonce, "query_index": query, "role": ready["role"],
               "ready_sha256": rr["sha256"], "manifest_key": rr["object_key"],
               "manifest_verification": {"method": "fresh_GET", **{k: rr[k] for k in ("bytes", "sha256")}},
               "members": [{**r, "object_key": base+"/"+r["path"],
                            "verification": {"method": "fresh_GET", **{k: r[k] for k in ("bytes", "sha256")}}}
                           for r in rows]}
        ar = document(base+"/ACK-v1.json", ack, "archive_control", base+"/ACK-v1.json")
        return {"ready": rr, "ack": ar, "members": members}

    roots, snapshots, native_records = [], [], []
    native_binding = {"root_id": "L9-FOUR-ROOTS", "run_id": "synthetic-native",
                      "source_manifest_sha256": "a"*64}
    native_base = "queries/L9-FOUR-ROOTS/initialization/native/"
    edit_helper = A.pinned("edits_reference")
    rng = np.random.default_rng(8)
    for ri, rid in enumerate(dict.fromkeys(r for r, _ in A.KEYS)):
        binding = {"root_id": rid, "run_id": "synthetic-parent-"+rid,
                   "source_manifest_sha256": str(ri)*64}
        prompt = "synthetic task "+str(ri//2)
        selection = {"index": ri % 2, "identity": str(ri+1)*64}
        model = {
            "model_bound": True, "adapter_bound": True,
            "effective_model_config": {"pi05": True, "dtype": "bfloat16", "action_dim": 32,
                                       "action_horizon": 10, "max_token_len": 200,
                                       "discrete_state_input": False},
            "checkpoint_artifact_id": "synthetic-model", "normalization_sha256": "b"*64}
        proofs = {k: document(rid+"/"+k+".json", v) for k, v in
                  (("model", model), ("runtime", {"synthetic": True}), ("dependencies", {"synthetic": True}))}
        root_prefix = "queries/"+rid+"/initialization/prompt/"
        pr = record("000001-ACTUAL_PROMPT_BINDING.json",
                    {"actual_prompt": prompt, "actual_prompt_utf8_sha256": A.sha(prompt.encode())},
                    {}, "parent_values_v1", root_prefix+"000001-ACTUAL_PROMPT_BINDING.json")
        rr = record("000002-ROOT_RUNTIME_BINDING.json",
                    {"actual_prompt": prompt,
                     "environment_source": {"root_trajectory_id": rid, "current_root_initial_selection": selection}},
                    {}, "parent_values_v1", root_prefix+"000002-ROOT_RUNTIME_BINDING.json")
        root = {"root_id": rid, "binding": binding, "prompt": prompt,
                "actual_initial_selection": selection, "proofs": proofs,
                "prompt_archive": archive(binding, None, [pr, rr], "prompt"), "queries": []}
        for q in (11, 22):
            base = f"queries/{rid}/query-{q:04d}/q{q}/"
            noise = rng.normal(size=(1, 10, 32)).astype(np.float32)
            state = rng.normal(size=(1, 32)).astype(np.float32)
            observations = {s: {"agentview_image": np.zeros((2, 2, 3), np.uint8),
                                "robot0_eye_in_hand_image": np.zeros((2, 2, 3), np.uint8),
                                "robot0_eef_pos": np.ones(3, np.float64),
                                "robot0_eef_quat": np.ones(4, np.float64),
                                "robot0_gripper_qpos": np.ones(2, np.float64)}
                            for s in ("C", "T")}
            branches = {s: A.Branch({"synthetic_encoded": np.array([ri, q], np.int32)}, state)
                        for s in ("C", "T")}
            neural = {}
            for side in ("C", "T"):
                bind = A.identity({"branch": A.explicit(branches[side]), "noise": noise})
                neural[side] = {"input_binding_before": bind, "input_binding_after": bind,
                                "input_unchanged": True, "explicit_model_noise_identity": A.array_id(noise)}
            pm = {"root_id": rid, "query_index": q, "env_step_index": 5*q,
                  "absolute_env_step_index": 10+5*q}
            pair = record("000100-PAIRED_INPUT.npz", {**pm, "full_runtime_snapshot_id": None,
                          "full_replay_qualified": False}, {"query_time_actual": np.array(q, np.float64),
                          "views": {v: {"observation": observations[s]} for s, v in
                                    (("C", "clean"), ("T", "nativeperturb"))}},
                          "parent_values_v1", base+"000100-PAIRED_INPUT.npz")
            clock = record("000101-QUERY_REACHED.npz", pm, {"query_time_actual": np.array(q, np.float64)},
                           "parent_values_v1", base+"000101-QUERY_REACHED.npz")
            sensor = record("000102-SENSOR_PROVENANCE.json", {**pm, "accepted": True}, {},
                            "parent_values_v1", base+"000102-SENSOR_PROVENANCE.json")
            parent_members = [pair, clock, sensor]
            for si, (side, condition) in enumerate((("C", "clean"), ("T", "nativeperturb"))):
                for repeat in range(2):
                    name = f"{103+2*si+repeat:06d}-NEURAL_RNG_AUDIT.json"
                    parent_members.append(record(name, {"root_trajectory_id": rid, "query_index": q,
                        "condition": condition, "kind": "pg_ae", "noise_role": "diagnostic_fixed",
                        "noise_label": 0, "audit": neural[side]}, {}, "parent_values_v1", base+name))
            qa = archive(binding, q, parent_members, "q"+str(q))
            root["queries"].append({"query_index": q, **qa})
            source = {"root_id": rid, "query_index": q, "env_step_index": 5*q,
                      "source_identity": A.sha((rid+str(q)).encode()),
                      "pair_receipt": "sha256:"+pair["sha256"], "query_clock_receipt": "sha256:"+clock["sha256"],
                      "actual_prompt_sha256": A.sha(prompt.encode()), "original_refs": proofs,
                      "actual_prompt_records": {"ACTUAL_PROMPT_BINDING": "sha256:"+pr["sha256"],
                                               "ROOT_RUNTIME_BINDING": "sha256:"+rr["sha256"]}}
            checks = {s: {"exact": True, "branch_identity": A.identity(branches[s]),
                          "checks": [{"exact": True, "noise_exact": True,
                                      "input_binding_before": neural[s]["input_binding_before"],
                                      "input_binding_after": neural[s]["input_binding_before"],
                                      "GPU_encoded_input_binding": neural[s]["input_binding_before"]} for _ in range(2)]}
                      for s in ("C", "T")}
            checks["decode_states_equal"] = True
            n = 3
            mask = np.array([[True, True, False]], bool)
            pos = np.cumsum(mask, axis=-1, dtype=np.int32)-1
            vt = rng.normal(size=(1, n, 1, 256)).astype(ml_dtypes.bfloat16)
            kt = rng.normal(size=vt.shape).astype(ml_dtypes.bfloat16)
            clean = (vt.astype(np.float32)-.3).astype(ml_dtypes.bfloat16)
            response = np.eye(256, 4)
            random = np.roll(response, 7, axis=0)
            bundle = edit_helper.construct(clean, vt, mask,
                edit_helper.Bases(response, random, "c"*64, {}), edit_helper.POLICY)
            native_meta = {"root_id": rid, "query_index": q}
            refs = {}

            def nr(role, event, metadata, arrays):
                name = f"{1000+len(native_records):06d}-{event}"+(".npz" if arrays else ".json")
                ref = record(name, metadata, arrays, "native_arrays_v1", native_base+name)
                native_records.append(ref)
                refs[role] = ref
                return ref
            nr("audit", "GPU_INPUT_AUDIT", {**native_meta, "source": source, "checks": checks},
               {"explicit_noise": noise, "consumed_decode_state": state})
            rm = bundle.report["random_match"]
            nr("edits", "FROZEN_QUERY_EDITS",
               {**native_meta, **{k: v for k, v in bundle.report.items() if k != "random_match"},
                "identity": bundle.identity,
                "random_match": {k: v for k, v in rm.items() if not isinstance(v, np.ndarray)}},
               {"Vmod": bundle.values, "actual_E": bundle.actual_E,
                "random_match": {k: v for k, v in rm.items() if isinstance(v, np.ndarray)}})
            selector = {"kv_mask": [False, True], "layer_mask": [i == 8 for i in range(18)],
                        "flow_mask": [i >= 5 for i in range(10)], "use_vmod": True}
            for role in ("natural_C", "natural_T", "effect_shared", "effect_random_matched"):
                meta = {**native_meta, "role": role, "source_identity": source["source_identity"],
                        "synthetic": True, "technical": {"gates": {"synthetic": True},
                                                         "sampling_abi": {"num_steps": 10}, "selectors": selector}}
                if role.startswith("effect"):
                    arm = role[7:]
                    meta.update(baseline_refs={"C": refs["natural_C"], "T": refs["natural_T"]},
                                edit_ref=refs["edits"],
                                actual_edit={k: edit_helper.aid(v) for k, v in
                                    (("receiver_V", vt), ("Vmod_argument", bundle.values[arm]),
                                     ("actually_used_V_at_j5", bundle.values[arm]),
                                     ("actual_E_used_minus_receiver_FP32", bundle.actual_E[arm]))})
                nr(role, "CALL_OUTPUT", meta, {
                    "raw32": noise.copy(), "adapter7": np.zeros((1, 10, 7), np.float64),
                    "flow": {"x": np.broadcast_to(noise, (11, 1, 10, 32)),
                             "velocity_native": np.zeros((10, 1, 10, 32), ml_dtypes.bfloat16),
                             "time": np.arange(11, dtype=np.float32)/10,
                             "time_weak_type": np.zeros(11, bool)},
                    "first_read": {
                        "probabilities_native": np.broadcast_to(
                            np.r_[mask[0], np.ones(10, bool)].astype(ml_dtypes.bfloat16)/(int(mask.sum())+10),
                            (1, 1, 8, 10, n+10)).astype(ml_dtypes.bfloat16),
                        "mask": np.broadcast_to(np.r_[mask[0], np.ones(10, bool)], (1, 10, n+10)),
                        "suffix_positions": (int(mask.sum())+np.arange(10, dtype=np.int32))[None],
                        "attention_gate": np.ones((1, 1, 1024), ml_dtypes.bfloat16)},
                    "prefix_mask": mask, "prefix_positions": pos,
                    "receiver_L9_K": kt, "receiver_L9_V": clean if role == "natural_C" else vt})
            records = {k: A.pinned("native_reader").load(
                folder/objects[ref["uri"]]["path"], ref, synthetic=True) for k, ref in refs.items()}
            snapshots.append({"root_id": rid, "query_index": q, "records": refs,
                              "original_parent_source": source,
                              "pulse_label_identities": A.original_query_identity(records)})
        roots.append(root)
    native_archive = archive(native_binding, None, native_records, "native")
    return {"schema": "rss-dev-pulse-input-readset-v1", "roots": roots, "snapshots": snapshots,
            "native_archives": [{**native_archive, "binding": native_binding}],
            "source_binding": native_binding, "input_objects": list(objects.values()),
            "input_object_count": len(objects), "input_bytes": sum(r["bytes"] for r in objects.values())}


class Tests(unittest.TestCase):
    def test_exact_real_manifest_is_small_and_reproducible_without_cloud(self):
        built = B.build()
        saved = json.loads((A.HERE/"INPUTS.json").read_text())
        self.assertEqual(built, saved)
        self.assertEqual(saved["input_object_count"], 216)
        self.assertEqual(saved["input_bytes"], 39527429)
        self.assertEqual(len(saved["snapshots"]), 8)
        self.assertFalse(saved["model_weights_in_readset"])

    def test_source_faithful_two_codec_end_to_end_and_no_calls(self):
        with tempfile.TemporaryDirectory() as d:
            manifest = fixture(d)
            result = A.load(d, manifest, synthetic=True)
            self.assertEqual(len(result["snapshots"]), 8)
            self.assertTrue(result["proof"]["input_source_qualified"])
            self.assertFalse(result["proof"]["native_runtime_qualified"])
            self.assertEqual(result["proof"]["new_pulse_labels"], 0)
            for saved in result["snapshots"]:
                self.assertIs(saved.modified_v["S"], saved.modified_v["P"])
                self.assertIs(saved.modified_v["own_null"], saved.old_VT)
                self.assertFalse(np.any(saved.actual_E["own_null"]))
                self.assertFalse(saved.noise.flags.writeable)
                self.assertFalse(saved.dose["redraw_or_rescale"])
                self.assertTrue(saved.dose["random_match_metadata"]["exact_norm_match_not_asserted"])
                self.assertIn("unbracketed", saved.dose["random_match"])
            json.dumps(result["proof"], allow_nan=False)

    def test_sha_corruption_stops_before_semantic_adoption(self):
        with tempfile.TemporaryDirectory() as d:
            manifest = fixture(d)
            ref = manifest["snapshots"][0]["records"]["edits"]
            obj = next(r for r in manifest["input_objects"] if r["uri"] == ref["uri"])
            path = Path(d)/obj["path"]
            path.write_bytes(path.read_bytes()+b"corrupt")
            with self.assertRaisesRegex(ValueError, "whole_staged_input_SHA"):
                A.load(d, manifest, synthetic=True)

    def test_wrong_codec_not_autodetected_or_silently_repaired(self):
        with tempfile.TemporaryDirectory() as d:
            manifest = fixture(d)
            ref = manifest["snapshots"][0]["records"]["audit"]
            store = A.Staged(d, manifest, lambda: None, True)
            with self.assertRaisesRegex(ValueError, "codec"):
                store.record(ref, "parent_values_v1")

    def test_fixed_population_and_stage_path_safety(self):
        with tempfile.TemporaryDirectory() as d:
            manifest = fixture(d)
            changed = copy.deepcopy(manifest)
            changed["snapshots"].pop()
            with self.assertRaisesRegex(ValueError, "all_eight"):
                A.load(d, changed, synthetic=True)
            changed = copy.deepcopy(manifest)
            changed["input_objects"][0]["path"] = "../escape"
            store = A.Staged(d, changed, lambda: None, True)
            ref = changed["input_objects"][0]
            with self.assertRaisesRegex(ValueError, "flat_stage"):
                store.path(ref, ref["codec"])

    def test_native_current_prefix_gate_is_separate_and_checks_used_payload(self):
        with tempfile.TemporaryDirectory() as d:
            manifest = fixture(d)
            saved = A.load(d, manifest, synthetic=True)["snapshots"][0]
            vm = saved.modified_v["S"]
            metadata = {
                "pg": {"T": {"cache_k": np.stack([saved.old_KT]*3),
                             "cache_v": np.stack([saved.old_VT]*3)}},
                "prefix_mask": saved.prefix_mask, "prefix_positions": saved.prefix_positions,
                "Vmod_argument": vm,
                "reads": {"prefix_value_used": np.broadcast_to(vm, (5, 3, *vm.shape))}}
            self.assertTrue(A.current_prefix_check(saved, metadata, "S")["current_prefix_and_payload_exact"])
            metadata["reads"]["prefix_value_used"] = np.broadcast_to(
                saved.old_VT, (5, 3, *vm.shape))
            with self.assertRaisesRegex(ValueError, "actually_consumed"):
                A.current_prefix_check(saved, metadata, "S")

    def test_real_arrays_refused_on_nonlinux_even_with_existing_fixture(self):
        with tempfile.TemporaryDirectory() as d:
            manifest = fixture(d)
            with patch.object(A.sys, "platform", "darwin"):
                with self.assertRaisesRegex(ValueError, "real_arrays_cloud"):
                    A.load(d, manifest, synthetic=False)

    def test_candidate_query_edit_fingerprint_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            manifest = fixture(d)
            manifest["snapshots"][0]["pulse_label_identities"]["shared"]["edit_identity"] = "0"*64
            with self.assertRaisesRegex(ValueError, "exact_original_query_and_edit_fingerprints"):
                A.load(d, manifest, synthetic=True)

    def test_wrong_actual_E_is_not_an_approximate_dose_match(self):
        with tempfile.TemporaryDirectory() as d:
            manifest = fixture(d)
            store = A.Staged(d, manifest, lambda: None, True)
            spec = manifest["snapshots"][0]
            native = {k: store.record(ref, "native_arrays_v1") for k, ref in spec["records"].items()}
            edited = copy.deepcopy(native["edits"])
            leaf = edited.arrays["actual_E"]["random_matched"]
            storage = leaf.storage.copy()
            storage[0, 0, 0, 0] += 1
            edited.arrays["actual_E"]["random_matched"] = A.pinned("native_reader").Native(
                storage, leaf.descriptor)
            with self.assertRaisesRegex(ValueError, "actual_E_exact_native_payload_difference"):
                A._dose(edited, A.value(native["natural_T"].field("receiver_L9_V")),
                        A.value(native["natural_T"].field("prefix_mask")))


if __name__ == "__main__":
    unittest.main()
