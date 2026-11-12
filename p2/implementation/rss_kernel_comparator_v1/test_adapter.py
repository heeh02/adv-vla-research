"""Acceptance checks on small published score summaries, never raw arrays."""
import copy
import json
import unittest
import adapter


class ReproductionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reference = json.loads((adapter.HERE / "REFERENCE-v2.json").read_text())

    def test_complete_old_score_support(self):
        result = adapter.reproduction_qa(self.reference, self.reference)
        self.assertEqual(result["maximum_absolute_difference"], 0)
        self.assertEqual(len(result["tracks"]), 15)
        self.assertGreater(result["numeric_cells"], 8000)

    def test_numeric_drift_blocks_acceptance(self):
        changed = copy.deepcopy(self.reference)
        changed["OOF_root_arm_scores"]["nested__rbf"][0]["MSE"][0][0][0] += 1e-6
        with self.assertRaisesRegex(RuntimeError, "old_v2_reproduction_tolerance"):
            adapter.reproduction_qa(changed, self.reference)

    def test_missing_or_reidentified_support_blocks_acceptance(self):
        changed = copy.deepcopy(self.reference)
        changed["paired_persistent_contrast"]["root_scores"]["nested__rbf"][0]["cluster"] = "wrong"
        with self.assertRaisesRegex(RuntimeError, "reproduction_exact_identity"):
            adapter.reproduction_qa(changed, self.reference)
        changed = copy.deepcopy(self.reference)
        changed["OOF_root_arm_scores"]["nested__rbf"].pop()
        with self.assertRaisesRegex(RuntimeError, "reproduction_exact_score_support"):
            adapter.reproduction_qa(changed, self.reference)


if __name__ == "__main__":
    unittest.main()
