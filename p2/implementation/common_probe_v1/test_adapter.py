"""Synthetic records and pinned small JSON only. No scientific tensor reads."""
from pathlib import Path
import copy
import hashlib
import importlib.util
import io
import json
import sys
import tempfile
import time
import unittest
from unittest.mock import Mock, patch
import numpy as np
import adapter as A


def fixture():
    old, core = A.load_components()
    # Reuse already tested synthetic codec-shaped records. The source file is
    # test code only; fixture() does not load any old research arrays.
    path = A.HERE.parent / "same_w_precision_v1/test_adapter.py"
    previous = sys.modules.get("adapter")
    sys.modules["adapter"] = old
    try:
        spec = importlib.util.spec_from_file_location("_common_probe_old_synthetic_fixture", path)
        tests = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(tests)
    finally:
        sys.modules["adapter"] = previous
    inp, calls, entries, query, reader, numeric = tests.fixture()
    expected = old.analyze_records(inp, calls, entries, query, reader, numeric)
    return inp, calls, entries, query, reader, numeric, old, core, expected


class Tests(unittest.TestCase):
    def test_frozen_prior_modules_and_actual_small_result(self):
        old, core = A.load_components()
        self.assertEqual(old.verify_sources(A.OLD_MANIFEST["sha256"]), A.OLD_MANIFEST)
        self.assertEqual((A.OLD / "core.py").read_bytes(), (A.HERE / "exact_bf16.py").read_bytes())
        prior = A.prior_result()
        self.assertEqual(prior["counts"]["A_target_nonrepresentable"], 623)
        self.assertEqual(prior["counts"]["reference_delta_nonzero"], 10091)
        self.assertEqual(old.load_inputs()["input_body_bytes"], 9151111)

    def test_synthetic_complete_geometry_keeps_no_GO_and_single_small_JSON(self):
        result = A.analyze_bound_records(*fixture())
        self.assertTrue(result["analysis_complete"])
        self.assertTrue(result["synthetic"])
        self.assertEqual(result["status"], "GEOMETRY_AUDIT_COMPLETE")
        self.assertFalse(result["same_W_setter_qualified"])
        self.assertFalse(result["GPU_auto_GO"])
        self.assertFalse(result["native_targets_written"])
        self.assertIsNone(result["scientific_acceptance_threshold"])
        self.assertEqual(result["geometry"]["full"]["counts"]["coordinates"], 10240)
        self.assertEqual(result["geometry"]["full"]["counts"]["natural_not_common_representable"], 10240)
        self.assertEqual(result["geometry"]["full"]["counts"]["sign_reversals"], 0)
        self.assertEqual(result["geometry"]["full"]["counts"]["magnitude_increases"], 0)
        self.assertLess(len(A.encoded(result)), A.OUTPUT_LIMIT)

    def test_original_precision_mismatch_stops_before_geometry(self):
        args = list(fixture())
        args[-1] = copy.deepcopy(args[-1])
        args[-1]["counts"]["A_target_nonrepresentable"] -= 1
        fake_core = Mock()
        args[-2] = fake_core
        with self.assertRaisesRegex(A.AuditFault, "reproduce_original_precision_counts"):
            A.analyze_bound_records(*args)
        fake_core.analyze_triplet.assert_not_called()

    def test_original_leaf_identity_and_support_mismatch_rejected(self):
        for field in ("W_source_leaves", "per_position", "actual_time_bits_sha256"):
            args = list(fixture())
            args[-1] = copy.deepcopy(args[-1])
            args[-1][field] = "wrong"
            with self.assertRaisesRegex(A.AuditFault, "reproduce_original_precision_" + field):
                A.analyze_bound_records(*args)

    def test_legacy_qualification_cannot_be_skipped(self):
        args = fixture()
        args[1][2].metadata["graph"] = "R"
        with self.assertRaises(ValueError):
            A.analyze_bound_records(*args)

    def test_real_entry_rejects_Mac_before_reading_inputs(self):
        with patch.object(A.sys, "platform", "darwin"), patch.object(A, "verify_sources", side_effect=AssertionError("must_not_read")):
            with self.assertRaisesRegex(A.AuditFault, "cloud_Linux"):
                A.run_stage("/tmp/unused", "/tmp/unused-output.json", source_manifest_sha256="a" * 64,
                            deadline_epoch=time.time() + 60)

    def test_synthetic_codec_records_end_to_end_without_redecode(self):
        inp, calls, entries, query, reader, numeric, old, core, _ = fixture()
        def serialize(record, name):
            arrays = {}
            def tree(value):
                if isinstance(value, reader.Native):
                    key = "a" + str(len(arrays))
                    arrays[key] = value.storage
                    return {"__array__": key, "shape": list(value.storage.shape), "logical_dtype": value.dtype,
                        "storage_dtype": str(value.storage.dtype), "bytes": value.storage.nbytes,
                        "payload_sha256": hashlib.sha256(value.storage.tobytes(order="C")).hexdigest()}
                if isinstance(value, dict):
                    return {k: tree(v) for k, v in value.items()}
                if isinstance(value, (list, tuple)):
                    return {"__sequence__": "tuple" if isinstance(value, tuple) else "list", "values": [tree(v) for v in value]}
                return value
            schema = tree({"metadata": record.metadata, "arrays": record.arrays})
            buffer = io.BytesIO()
            np.savez_compressed(buffer, **arrays, __schema__=np.frombuffer(A.encoded(schema), np.uint8))
            raw = buffer.getvalue()
            return raw, {"path": name, "uri": "synthetic:" + name, **A.ident(raw)}
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            raw, iref = serialize(inp, "000001-PVP_INPUT_VERIFIED.npz")
            (folder / iref["path"]).write_bytes(raw)
            decoded_input = reader.load(folder / iref["path"], iref, synthetic=True)
            decoded_calls = []
            for i, call in enumerate(calls):
                call.metadata["input_ref"] = iref
                raw, ref = serialize(call, f"{i+2:06d}-PVP_CALL_OUTPUT.npz")
                (folder / ref["path"]).write_bytes(raw)
                decoded_calls.append(reader.load(folder / ref["path"], ref, synthetic=True))
            expected = old.analyze_records(decoded_input, decoded_calls, entries, query, reader, numeric)
            # All four native records are already decoded and verified. Neither
            # old qualification nor new geometry may reopen or decode them.
            with patch.object(reader, "load", side_effect=AssertionError("second_decode_forbidden")):
                result = A.analyze_bound_records(decoded_input, decoded_calls, entries, query,
                    reader, numeric, old, core, expected)
            self.assertTrue(result["analysis_complete"])
            self.assertTrue(result["synthetic"])
            self.assertEqual(result["precision_reproduction"]["counts"]["A_target_nonrepresentable"], 10240)


if __name__ == "__main__":
    unittest.main()
