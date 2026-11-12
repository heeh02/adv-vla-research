"""Small storage acceptance tests; no actual source arrays."""
import json
from pathlib import Path
import tempfile
import time
import subprocess
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import adapter as A

S = A.module("_test_driver_storage", A.HERE / "storage/compact.py")


class Tests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "input.npz"
        self.saved = SimpleNamespace(
            prefix_mask=np.array([[True, False, True]], bool),
            prefix_positions=np.array([[0, 0, 1]], np.int32),
            root_id="synthetic", query_index=11, source_identity="a"*64,
            sources={"synthetic": True}, encoded_identities={"C": "b"*64, "T": "c"*64},
            noise=np.zeros((1, 10, 32), np.float32), decode_state=np.zeros((1, 32), np.float32),
            old_KT=np.zeros((1, 3, 1, 256), np.float32),
            old_VT=np.zeros((1, 3, 1, 256), np.float32),
            modified_v={"S": np.zeros((1, 3, 1, 256), np.float32)},
            actual_E={"S": np.zeros((1, 3, 1, 256), np.float32)})
        self.arrays = {k: getattr(self.saved, k) for k in ("prefix_mask", "prefix_positions")}
        self.metadata = A.input_metadata(self.saved, S.IDENTITY.array_id)

    def test_original_codec_roundtrip(self):
        self.path.write_bytes(S.pack(self.metadata, self.arrays, 8192))
        A.check_input_record(self.path, self.saved, S.IDENTITY.array_id, S.CODEC.flatten)

    def test_self_consistent_wrong_array_is_rejected(self):
        changed = {**self.arrays, "prefix_positions": np.array([[0, 1, 2]], np.int32)}
        self.path.write_bytes(S.pack(self.metadata, changed, 8192))
        with self.assertRaisesRegex(ValueError, "same_actual_input_array"):
            A.check_input_record(self.path, self.saved, S.IDENTITY.array_id, S.CODEC.flatten)

    def test_unrecognized_record_field_is_rejected(self):
        self.path.write_bytes(S.pack(self.metadata,
                                    {**self.arrays, "extra": np.zeros(1)}, 8192))
        with self.assertRaisesRegex(ValueError, "exact_small_input_array_roster"):
            A.check_input_record(self.path, self.saved, S.IDENTITY.array_id, S.CODEC.flatten)

    def test_wrong_source_metadata_with_correct_arrays_is_rejected(self):
        self.path.write_bytes(S.pack({**self.metadata, "source_identity": "wrong"},
                                    self.arrays, 8192))
        with self.assertRaisesRegex(ValueError, "same_actual_input_metadata"):
            A.check_input_record(self.path, self.saved, S.IDENTITY.array_id, S.CODEC.flatten)

    def test_failed_storage_test_log_is_preserved(self):
        fake = SimpleNamespace(stdout=b"", stderr=b"Ran 11 tests\nFAILED (failures=1)\n", returncode=1)
        with patch.object(A.subprocess, "run", return_value=fake):
            self.assertFalse(A.storage_software_qa(self.path.parent, time.time()+10))
        report = json.loads((self.path.parent / "storage-software-QA.json").read_text())
        self.assertEqual(report["status"], "FAIL")
        self.assertIn("failures=1", report["log"])
        self.assertTrue(report["synthetic"])

    def test_timeout_storage_test_log_is_preserved(self):
        error = subprocess.TimeoutExpired("synthetic", 1, output=b"started\n", stderr=b"pending\n")
        with patch.object(A.subprocess, "run", side_effect=error):
            self.assertFalse(A.storage_software_qa(self.path.parent, time.time()+10))
        report = json.loads((self.path.parent / "storage-software-QA.json").read_text())
        self.assertEqual(report["status"], "TIMEOUT")
        self.assertEqual(report["log"], "started\npending\n")


if __name__ == "__main__":
    unittest.main()
