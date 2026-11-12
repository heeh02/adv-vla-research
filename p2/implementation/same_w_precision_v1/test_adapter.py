"""Synthetic records and local small source metadata only; no real NPZ reads."""
import copy
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import adapter as A


def fixture():
    N, M = A.load_helpers()
    def native(value, logical=None):
        a = np.asarray(value)
        desc = {"shape": list(a.shape), "logical_dtype": logical or a.dtype.name,
                "storage_dtype": str(a.dtype), "bytes": a.nbytes,
                "payload_sha256": hashlib.sha256(a.tobytes()).hexdigest()}
        return N.Native(a, desc)
    def wrap(value):
        if isinstance(value, dict):
            return {k: wrap(v) for k, v in value.items()}
        return native(value.storage, value.dtype) if isinstance(value, N.Native) else native(value)
    def bf(shape, bits=0):
        return native(np.full(shape, bits, np.uint16), "bfloat16")
    n = 16
    ref = {"path": "000002-PVP_INPUT_VERIFIED.npz", "bytes": 1, "sha256": "a" * 64, "uri": "synthetic:input"}
    inp = N.Record("PVP_INPUT_VERIFIED", {"root_id": A.ROOT, "query_index": 11, "sources": {"synthetic": True}},
        wrap({"noise": np.zeros((1, 10, 32), np.float32), "consumed_decode_state": np.zeros((1, 32), np.float32),
              "Vmod": bf((1, n, 1, 256), 0x3f00), "actual_E": np.full((1, n, 1, 256), .25, np.float32)}), ref, 1, True)
    calls, entries = [], []
    for ri, (role, call_id) in enumerate(zip(A.ROLES, A.CALL_IDS)):
        entry = {"call_id": call_id, "root_id": A.ROOT, "query_index": 11, "role": role, "receiver": "T", "donor": "C",
                 "num_steps": 10, "kv_mask": [False, ri > 0], "layer_mask": [ri > 0 and j == 8 for j in range(18)],
                 "flow_mask": [ri == 1 and j == 5 or ri == 2 and j >= 5 for j in range(10)], "use_vmod": ri > 0}
        entries.append(entry)
        reads = {k: bf((5, 3, 1, 10, 1024), 0x3f80 if ri else 0x4000) for k in M.FIELDS}
        if ri == 2:
            reads["attention_output_post_WO_pre_gate"].storage[1, 1] = 0x3f81
        reads.update(query_post_rope_scaled=bf((5, 3, 1, 10, 1, 8, 256)),
            probabilities_native=bf((5, 3, 1, 1, 8, 10, n + 10)), attention_gate=bf((5, 3, 1, 1, 1024), 0x3f00),
            suffix_key_post_rope=bf((5, 3, 1, 10, 1, 256)), suffix_value=bf((5, 3, 1, 10, 1, 256)),
            mask=np.ones((5, 1, 10, n + 10), bool), suffix_positions=np.broadcast_to(np.arange(n, n + 10, dtype=np.int32), (5, 1, 10)).copy(),
            query_defined=np.ones((5, 1, 10), bool), adarms_condition=np.zeros((5, 1, 1024), np.float32))
        x = np.zeros((11, 1, 10, 32), np.float32)
        velocity = bf((10, 1, 10, 32))
        if ri:
            x[6:] = .25
            velocity.storage[5:] = 0x3f80
        if ri == 2:
            x[7:] = .5
            velocity.storage[6:] = 0x4000
        arrays = wrap({"reads": reads, "flow": {"x": x, "velocity_native": velocity,
            "time": np.linspace(1, 0, 11, dtype=np.float32), "time_weak_type": np.ones(11, bool), "dt": np.full(10, -.1, np.float32)},
            "prefix_mask": np.ones((1, n), bool), "prefix_positions": np.arange(n, dtype=np.int32)[None],
            "selectors": {k: np.asarray(entry[k], bool) for k in ("kv_mask", "layer_mask", "flow_mask", "use_vmod")},
            "capture_layers": np.array(M.LAYERS, np.int32), "capture_calls": np.array(M.FLOWS, np.int32),
            "gates": {k: np.asarray(True) for k in M.GATES}, "raw32": x[-1], "adapter7": np.zeros((1, 10, 7), np.float64)})
        meta = {**entry, "graph": "Gstar", "input_ref": ref, "input_identity": "b" * 64,
                "PG_refs": {"T": {"path": "syntheticPG", "bytes": 1, "sha256": "c" * 64}},
                "Vmod_actual_identity": {"shape": [1, n, 1, 256], "logical_dtype": "bfloat16",
                    "bytes": inp.field("Vmod").storage.nbytes, "sha256": N.sha(inp.field("Vmod").storage.tobytes())}}
        calls.append(N.Record("PVP_CALL_OUTPUT", meta, arrays,
                              {"path": f"{ri+10:06d}-PVP_CALL_OUTPUT.npz", "bytes": 1, "sha256": str(ri) * 64}, 1, True))
    source = {"old_saved_sources": {"synthetic": True}, "prefix_capacity": n, "valid_prefix_tokens": n, "exactzero_actual_E": False}
    return inp, calls, entries, source, N, M


class AdapterTests(unittest.TestCase):
    def test_actual_ref_selection_uses_small_source_identity(self):
        plan = A.load_inputs()
        path = A.HERE.parents[1] / "analysis/pvp-precision-source/CCONF-T06-PVP-SOURCE-v1.json"
        # This is an already-local 29KB JSON, never a scientific tensor.
        source = json.loads(path.read_text())
        q = A.select_source_query(source, plan)
        self.assertEqual(q["input_ref"]["sha256"], next(r["sha256"] for r in plan["input_objects"] if r["role"] == "input"))
        bad = copy.deepcopy(source)
        bad["queries"][0]["selected_calls"][0]["role"] = "wrong"
        with self.assertRaises(A.InputError):
            A.select_source_query(bad, plan)

    def test_frozen_validator_and_new_masks_then_exact_core(self):
        args = fixture()
        result = A.analyze_records(*args)
        self.assertEqual(result["status"], "NO_GO_NATURAL_SAME_DELTA_UNREPRESENTABLE")
        self.assertEqual(result["counts"]["A_target_nonrepresentable"], 10240)
        self.assertEqual(result["counts"]["reference_delta_nonzero"], 10240)
        self.assertTrue(result["input_qualified"])
        self.assertFalse(result["same_W_setter_qualified"])
        self.assertLess(len(A.encode(result)), A.RESULT_LIMIT)

    def test_wrong_receiver_role_or_first_write_state_rejected(self):
        args = fixture()
        args[1][2].metadata["graph"] = "R"
        with self.assertRaises(ValueError):
            A.analyze_records(*args)
        args = fixture()
        args[1][2].field("flow", "x").storage[6, 0, 0, 0] += 1
        with self.assertRaises(ValueError):
            A.analyze_records(*args)

    def test_none_mask_time_gate_mismatch_not_accepted(self):
        for path, coords in [(('reads', 'mask'), (1, 0, 0, 0)),
                             (('flow', 'time_weak_type'), (6,)),
                             (('reads', 'attention_gate'), (1, 1, 0, 0, 0))]:
            args = fixture()
            value = args[1][0].field(*path).storage
            value[coords] = 0 if value[coords] else 1
            with self.assertRaises(ValueError):
                A.analyze_records(*args)

    def test_production_entry_blocks_Mac_before_native_array_reads(self):
        with patch.object(A.sys, "platform", "darwin"), patch.object(A, "load_helpers", side_effect=AssertionError("must not import reader")):
            with self.assertRaisesRegex(A.InputError, "cloud_Linux"):
                A.run_stage("/tmp/not-read", "/tmp/not-written", source_manifest_sha256="a" * 64, deadline_epoch=1)

    def test_frozen_reader_preserves_uint16_zero_sign_and_rejects_leaf_SHA(self):
        N, _ = A.load_helpers()
        bits = np.array([0, 0x8000, 1, 0x3f80, 0x7f7f], np.uint16)
        leaf = {"__array__": "a0", "shape": [5], "logical_dtype": "bfloat16", "storage_dtype": "uint16",
                "bytes": bits.nbytes, "payload_sha256": hashlib.sha256(bits.tobytes()).hexdigest()}
        schema = {"metadata": {"synthetic": True}, "arrays": {"W": leaf}}
        def packet():
            output = io.BytesIO()
            np.savez_compressed(output, a0=bits, __schema__=np.frombuffer(A.encode(schema), np.uint8))
            return output.getvalue()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture"
            raw = packet();path.write_bytes(raw)
            ref = {"path": "000001-PVP_CALL_OUTPUT.npz", **A.identity(raw)}
            record = N.load(path, ref, synthetic=True)
            np.testing.assert_array_equal(record.field("W").storage, bits)
            self.assertEqual(record.field("W").dtype, "bfloat16")
            leaf["payload_sha256"] = "0" * 64
            raw = packet();path.write_bytes(raw);ref.update(A.identity(raw))
            with self.assertRaisesRegex(ValueError, "payload_SHA"):
                N.load(path, ref, synthetic=True)

    def test_wrong_actual_input_ref_rejected_even_with_same_filename(self):
        args = fixture()
        args[1][1].metadata["input_ref"] = {**args[0].ref, "sha256": "f" * 64}
        with self.assertRaisesRegex(ValueError, "input_and_PG"):
            A.analyze_records(*args)


if __name__ == "__main__":
    unittest.main()
