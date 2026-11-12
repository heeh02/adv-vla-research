"""Synthetic-only checks of support, normalization, failure gates and exports."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import renderer as R
from synthetic_fixture import make_review


class Tests(unittest.TestCase):
    def test_complete_fixed_support_and_no_counterexample_removed(self):
        review = make_review()
        data, indexes, tasks = R.validate_review(review)
        self.assertEqual(len(indexes["root"]), 120)
        self.assertEqual(len(indexes["task"]), 60)
        values = R.display_values(data, indexes)
        self.assertEqual(len(values["panel_A"]), 96)
        self.assertEqual(len(values["panel_B"]), 24)
        self.assertEqual(len(values["queries"]), 24)
        self.assertEqual(len(tasks), 4)
        self.assertEqual(R.task_label("CCONF-T06"), "T06")
        self.assertEqual(R.task_label("T06"), "T06")
        self.assertTrue(any(q["normalized_difference"] is not None and q["normalized_difference"] < 0
                            for q in values["queries"]))
        self.assertTrue(any(q["target"] == 0 and q["normalized_difference"] is None for q in values["queries"]))
        self.assertTrue(any(r["reason"] == "ZERO_TARGET" for r in values["unavailable_root_comparisons"]))
        self.assertTrue(any(r["reason"] == "MISSING_SSE" for r in values["unavailable_root_comparisons"]))

    def test_common_full_denominator_and_basis_additivity(self):
        data, indexes, _ = R.validate_review(make_review())
        values = R.display_values(data, indexes)
        for r in values["panel_C"]:
            self.assertAlmostEqual(r["B3_ratio"]+r["residual_ratio"], r["full_SSE"]/r["target"])
        review = make_review()
        r = next(r for r in review["figure_data"]["root_rows"] if r["view"] == "response_B3")
        r["full_target_delta_energy"] = r["target_delta_energy"]
        with self.assertRaisesRegex(ValueError, "same_full_denominator"):
            R.validate_review(review)

    def test_actual_on_mac_rejected_before_acceptance_or_render(self):
        review = {"schema": "PAPER2-result-review-v1", "synthetic": False}
        with patch.object(R.sys, "platform", "darwin"):
            with self.assertRaisesRegex(ValueError, "never_Mac"):
                R.validate_review(review, allow_real_cloud=True, cloud_job_id="synthetic-test-only")

    def test_rejected_or_omitted_membership_cannot_render(self):
        review = make_review()
        review["accepted"] = False
        with self.assertRaisesRegex(ValueError, "accepted_complete"):
            R.validate_review(review)
        review = make_review()
        review["figure_data"]["query_rows"].pop()
        with self.assertRaisesRegex(ValueError, "all_24"):
            R.validate_review(review)
        review = make_review()
        review["figure_data"]["root_rows"].append(copy.deepcopy(review["figure_data"]["root_rows"][0]))
        with self.assertRaisesRegex(ValueError, "unique_planned"):
            R.validate_review(review)

    def test_export_three_formats_synthetic_labels_and_sha_readback(self):
        review = make_review()
        before = R.encode(review)
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)/"synthetic"
            manifest = R.render_review(review, out, dpi=100)
            self.assertTrue(manifest["synthetic"])
            self.assertFalse(manifest["visual_review_complete"])
            self.assertEqual(len(manifest["outputs"]), 7)
            for ref in manifest["outputs"]:
                self.assertEqual(R.identity((out/ref["path"]).read_bytes()), {k: ref[k] for k in ("bytes", "sha256")})
            self.assertIn("SYNTHETIC SOFTWARE PREVIEW", (out/"SYNTHETIC-MASS-BASELINE.svg").read_text())
            self.assertEqual((out/"SYNTHETIC-MASS-BASELINE.pdf").read_bytes()[:5], b"%PDF-")
            values = json.loads((out/"FIGURE-VALUES.json").read_bytes())
            self.assertEqual(len(values["queries"]), 24)
            with self.assertRaisesRegex(ValueError, "no_overwrite"):
                R.render_review(review, out)
        self.assertEqual(before, R.encode(review))


if __name__ == "__main__":
    unittest.main()
