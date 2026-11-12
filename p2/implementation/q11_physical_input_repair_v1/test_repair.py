"""Regression checks on existing metadata only; no native arrays or environment."""
import copy
import json
from pathlib import Path
import unittest
from repair import assemble, transport

BASE = Path(__file__).resolve().parents[2]
GOOD = BASE/"implementation/q11_replay_input_qa_v1/INPUTS.json"
ATTEMPTED = BASE/"implementation/cpu_action_replay_candidate_v1/INPUTS.json"


class Repair(unittest.TestCase):
    def setUp(self):
        self.good = json.loads(GOOD.read_bytes())
        self.attempted = json.loads(ATTEMPTED.read_bytes())

    def test_all_qualified_semantic_rows_preserved_not_only_object_hashes(self):
        result = assemble(self.good, self.attempted)
        by = {r["path"]: r for r in result["input_objects"]}
        self.assertTrue(all(by[r["path"]] == r for r in self.good["input_objects"]))
        self.assertEqual([transport(r) for r in result["input_objects"]],
                         [transport(r) for r in self.attempted["input_objects"]])

    def test_changed_payload_identity_rejected(self):
        name = self.good["cached_command_paths"]["N"]["compact"]
        next(r for r in self.attempted["input_objects"] if r["path"] == name)["sha256"] = "0"*64
        with self.assertRaisesRegex(ValueError, "same_exact_qualified_data_objects"):
            assemble(self.good, self.attempted)

    def test_extra_or_missing_runtime_objects_rejected(self):
        self.attempted["input_objects"].pop()
        with self.assertRaisesRegex(ValueError, "exact_unique_input_counts"):
            assemble(self.good, self.attempted)

    def test_full_ref_loss_from_qualified_source_rejected(self):
        name = self.good["cached_command_paths"]["N"]["compact"]
        next(r for r in self.good["input_objects"] if r["path"] == name)["logical_path"] = "wrong/path"
        with self.assertRaisesRegex(ValueError, "semantic_reference"):
            assemble(self.good, self.attempted)


if __name__ == "__main__":
    unittest.main()
