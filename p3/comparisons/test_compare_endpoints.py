import copy
import json
import unittest
from pathlib import Path
from compare_endpoints import compare, digest

HERE = Path(__file__).resolve().parent


class PairingTests(unittest.TestCase):
    def setUp(self):
        base = json.loads((HERE/"INPUT-PLAN-v1.json").read_bytes())
        self.plan = dict(base, models={"c": {"checkpoint_sha256": "a"*64, "seed": 0},
            "t": {"checkpoint_sha256": "b"*64, "seed": 0}}, panels={"Plus": {"planned": 2}},
            contrasts=[dict(id="t-c", control="c", treatment="t")])

    def row(self, m, o, success):
        case = {"task_id": o, "env_seed": 0}
        return dict(panel="Plus", model_id=m, checkpoint_sha256=("a" if m == "c" else "b")*64,
            ordinal=o, official_case=case, case_definition_sha256=digest(case),
            population_sha256="e"*64, run_id=m+"-run", attempt=0, adopted=True,
            success=success, status="complete", summary={"sha256": str(o+1)*64},
            outcome_canonical_sha256=("c" if m == "c" else "d")*64,
            input=dict(signature="q"+str(o), runtime_signature="r"*64,
                recorded_noise_by_query={"0": {"dtype": "float32", "shape": [1], "sha256": "f"*64}}))

    def result(self, rows):
        return compare(self.plan, rows)

    def test_partial_has_no_population_rate_or_delta(self):
        d = self.result([self.row("c", 0, True), self.row("t", 0, False)])
        self.assertIsNone(d["single_endpoints"][0]["success_rate"])
        p = d["paired_contrasts"][0]
        self.assertIsNone(p["delta_success_rate"])
        self.assertEqual(p["strict_pair_completion_bounds"], [-1.0, 0.0])

    def test_complete_balanced_gain_harm(self):
        d = self.result([self.row("c", 0, True), self.row("t", 0, False), self.row("c", 1, False), self.row("t", 1, True)])
        p = d["paired_contrasts"][0]
        self.assertEqual((p["gains"], p["harms"], p["delta_success_rate"]), (1, 1, 0.0))

    def test_input_missing_retains_single_outcome(self):
        a, b = self.row("c", 0, True), self.row("t", 0, False)
        b["input"] = {"signature": None}
        d = self.result([a, b])
        self.assertEqual(d["single_endpoints"][1]["normal_failure"], 1)
        self.assertEqual(d["paired_contrasts"][0]["qualified_pairs"], 0)

    def test_runtime_and_noise_mismatch_not_qualified(self):
        a, b = self.row("c", 0, True), self.row("t", 0, True)
        b["input"]["runtime_signature"] = "different"
        self.assertIn("runtime_content_difference_or_missing", self.result([a,b])["paired_contrasts"][0]["pairing_reasons"])
        b["input"]["runtime_signature"] = a["input"]["runtime_signature"]
        b["input"]["recorded_noise_by_query"]["0"]["dtype"] = "bfloat16"
        self.assertIn("observed_shared_query_noise_difference", self.result([a,b])["paired_contrasts"][0]["pairing_reasons"])

    def test_noncontiguous_noise_rejected(self):
        r = self.row("c", 0, True)
        r["input"]["recorded_noise_by_query"] = {"1": {"sha256": "f"*64}}
        with self.assertRaises(ValueError): self.result([r])

    def test_identical_duplicate_not_new_sample_but_conflict_rejected(self):
        r = self.row("c", 0, True)
        d = self.result([r, copy.deepcopy(r)])
        self.assertEqual((d["unique_adopted_records"], d["exact_duplicate_reads_not_samples"]), (1, 1))
        z = copy.deepcopy(r); z["success"] = False
        with self.assertRaises(ValueError): self.result([r,z])

    def test_wrong_checkpoint_bool_attempt_and_technical_status_rejected(self):
        for key, val in (("checkpoint_sha256", "z"*64), ("attempt", True), ("status", "technical_failure")):
            r = self.row("c", 0, True); r[key] = val
            with self.assertRaises(ValueError): self.result([r])

    def test_case_or_population_mismatch_not_a_pair(self):
        a, b = self.row("c", 0, True), self.row("t", 0, True)
        b["official_case"]["env_seed"] = 1; b["case_definition_sha256"] = digest(b["official_case"])
        self.assertIn("exact_case_definition_difference", self.result([a,b])["paired_contrasts"][0]["pairing_reasons"])
        b = self.row("t", 0, True); b.pop("population_sha256")
        self.assertIn("population_identity_difference_or_missing", self.result([a,b])["paired_contrasts"][0]["pairing_reasons"])


if __name__ == "__main__":
    unittest.main()
