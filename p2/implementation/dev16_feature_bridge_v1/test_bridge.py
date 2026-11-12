import copy
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import analysis as A
import adapter as D
import codec as C
import synthetic_fixture as F


class BridgeTests(unittest.TestCase):
    def test_zero_reference_nullable(self):
        r = A.compare(C.from_array(np.zeros(3)), C.from_array(np.ones(3)))
        self.assertIsNone(r["relative_RMS_to_reference"])
        self.assertEqual(r["RMS_difference"], 1.)
        self.assertFalse(r["typed_exact"])

    def test_signed_zero_and_dtype(self):
        r = A.compare(C.from_array(np.array([0.])), C.from_array(np.array([-0.])))
        self.assertTrue(r["numeric_equal"])
        self.assertFalse(r["typed_exact"])
        r = A.compare(C.from_array(np.array([1.], np.float32)),
                      C.from_array(np.array([1.], np.float64)))
        self.assertTrue(r["numeric_equal"])
        self.assertFalse(r["typed_exact"])

    def test_BF16_bits_and_numeric(self):
        a = F.bf([1., 2.])
        self.assertTrue(np.array_equal(a.storage, [16256, 16384]))
        self.assertTrue(np.array_equal(a.numeric(), [1., 2.]))
        self.assertNotEqual(a.identity()["payload_sha256"], a.input_identity()["sha256"])

    def test_codec_payload_corruption_rejects(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "x.npz"
            raw = F.typed_bytes({}, {("x",): F.bf([1., 2.])})
            p.write_bytes(raw)
            ref = {"path": "x.npz", "bytes": len(raw), "sha256": C.sha(raw)}
            _, data = C.typed(p, ref, synthetic=True)
            self.assertTrue(np.array_equal(data[("x",)].numeric(), [1., 2.]))
            ref["sha256"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "original_file_SHA"):
                C.typed(p, ref, synthetic=True)

    def test_wrong_checks_entry_rejects(self):
        entry = {"call_id": "N"}
        inp = {"path": "in.npz", "bytes": 2, "sha256": "i"}
        out = {"path": "out.npz", "bytes": 3, "sha256": "o"}
        meta = {"schema": "rss-pulse-compact-call-v1", "entry": entry,
                "native_validation_errors": [], "input_ref": inp}
        event = {"schema": "rss-pulse-compact-event-v1", "event": "CALL_CHECKS",
                 "payload": {"entry": {"call_id": "P"}, "passed": True, "output": out}}
        with self.assertRaisesRegex(ValueError, "later_passed_checks"):
            A.validate_call(meta, event, entry, inp, out)
        event["event"] = "OUTPUT_PRESERVED"
        with self.assertRaisesRegex(ValueError, "exact_CALL_CHECKS"):
            A.validate_call(meta, event, entry, inp, out)

    def test_interspersed_observability_event_mapping(self):
        inputs = C.json_read((Path(__file__).parent / "INPUTS.json").read_bytes())
        sequence = ["START", "SOURCE_BOUND"]
        checks = {}
        for entry in inputs["native72_plan_entries"]:
            sequence.extend(["CHARGED", "OUTPUT_PRESERVED"])
            # Worker emits OBSERVABILITY before CALL_CHECKS for random_pulse.
            if entry["role"] == "random_pulse":
                sequence.append("OBSERVABILITY")
            checks[entry["call_id"]] = len(sequence)
            sequence.append("CALL_CHECKS")
        refs = {r["path"]: r for r in inputs["input_objects"]}
        for spec in inputs["snapshots"]:
            for arm in ("N", "P"):
                expected = f'ledger/event-{checks[spec[arm + "_entry"]["call_id"]]:04d}.json'
                actual = refs[spec["new"][arm + "_checks"]]["logical_path"]
                self.assertTrue(actual.endswith(expected))

    def test_complete_synthetic_and_baseline_shift(self):
        with tempfile.TemporaryDirectory() as td:
            stage = Path(td) / "stage"
            inputs = F.stage_fixture(stage)
            out = D.execute(stage, inputs, lambda: None, synthetic=True)
            self.assertEqual(out["QA.json"]["counts"],
                             {"snapshots": 8, "native_field_comparisons": 96,
                              "feature_comparisons": 56, "endpoint_comparisons": 848, "decisions": 128})
            self.assertTrue(all(r["typed_exact"] for r in out["FIELD-COMPARISONS.json"]))
            unknown = [r for r in out["DECISION-TABLE.json"] if r["object"] == "fresh_C_AE_context"]
            self.assertEqual(len(unknown), 8)
            self.assertTrue(all(r["status"] == "UNMEASURED" for r in unknown))
            # Same effect after adding a common endpoint offset must not erase
            # actual baseline/endpoint differences.
            a = A.compare(C.from_array(np.array([1.])), C.from_array(np.array([2.])))
            b = A.compare(C.from_array(np.array([1.25 - 1.])),
                          C.from_array(np.array([2.25 - 2.])))
            self.assertFalse(a["typed_exact"])
            self.assertTrue(b["typed_exact"])

    def test_native_difference_does_not_fail_or_hide_behind_zero_proxy(self):
        # Distinct probabilities under E=0 give the same proxy; primary field
        # comparison retains the mismatch independently.
        p = np.full((8, 10, 16), 1/16.)
        q = p.copy()
        q[:, :, 0] += .01
        e = np.zeros((6, 256))
        w = np.zeros((8, 256, 1024))
        r0, w0 = A.proxy(p, e, w, np.ones(1024))
        r1, w1 = A.proxy(q, e, w, np.ones(1024))
        self.assertFalse(A.compare(C.from_array(p), C.from_array(q))["typed_exact"])
        self.assertTrue(np.array_equal(r0, r1) and np.array_equal(w0, w1))


if __name__ == "__main__":
    unittest.main()
