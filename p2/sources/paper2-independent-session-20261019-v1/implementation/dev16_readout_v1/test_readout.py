"""Synthetic-only tests. No actual input file is opened or reduced."""
import copy
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import adapter
import analysis as A
import synthetic_fixture as F


class ReadoutTests(unittest.TestCase):
    def run_fixture(self, fixture=None):
        return A.analyze(*(fixture or F.make()), synthetic=True)

    def test_all560_and_independent_known_values(self):
        rows, summary, qa, snapshots = self.run_fixture()
        self.assertEqual(len(rows), 560)
        self.assertEqual(len(summary["summaries"]), 70)
        self.assertEqual(len(summary["S_R_RMS_comparisons"]), 14)
        self.assertEqual(len(snapshots["snapshots"]), 8)
        self.assertTrue(summary["synthetic"])
        self.assertTrue(qa["pass"])
        wanted = next(r for r in rows if r["root_id"] == A.ROOTS[0] and
                      r["query_index"] == 11 and r["command_dimension"] == "gripper_command"
                      and r["window"] == "first5" and r["contrast"] == "S_minus_N")
        self.assertEqual(wanted["RMS"], 1.)
        self.assertEqual(wanted["mean_signed"], .2)
        self.assertEqual(wanted["maxabs"], 1.)
        self.assertEqual(wanted["zero_positions"], 0)
        whole = next(r for r in rows if all(r[k] == wanted[k] for k in
                     ("root_id", "query_index", "command_dimension", "contrast")) and
                     r["window"] == "full10")
        self.assertEqual(whole["mean_signed"], 0.)
        self.assertEqual(whole["RMS"], 1.)
        self.assertTrue(all(r["unit"] == "command_adapter_units" for r in rows))
        self.assertTrue(all(r["snapshots"] == 8 for r in summary["summaries"]))

    def test_zero_tie_and_order_categories_preserved(self):
        _, summary, _, snapshots = self.run_fixture()
        classes = [r["class"] for r in snapshots["snapshots"][0]["S_R_axis_window_comparisons"][:6]]
        self.assertEqual(classes, ["both_zero", "S_zero_only", "R_zero_only",
                                  "equal_nonzero_RMS", "S_larger_RMS", "R_larger_RMS"])
        for row in summary["S_R_RMS_comparisons"]:
            self.assertEqual(sum(row["counts"].values()), 8)
        self.assertFalse(summary["dose"]["original_exact_actual_dose_match_true_count"])
        self.assertAlmostEqual(summary["dose"]["signed_relative_error"]["median"], 1e-6)

    def test_explicit_contrast_inconsistency_rejected(self):
        fixture = F.make()
        fixture[0]["result"]["observability"][0]["S_minus_R_adapter7"][0][0][0] += .001
        with self.assertRaisesRegex(ValueError, "explicit_contrast_numerical_consistency"):
            self.run_fixture(fixture)

    def test_small_arithmetic_roundoff_is_not_typed_identity(self):
        fixture = F.make()
        fixture[0]["result"]["observability"][0]["P_minus_S_adapter7"][0][0][0] += 1e-14
        rows, _, qa, _ = self.run_fixture(fixture)
        self.assertEqual(len(rows), 560)
        self.assertTrue(any(r["maximum_absolute_residual"] > 0
                            for r in qa["explicit_contrast_arithmetic_checks"]))
        self.assertTrue(all(r["typed_byte_identity_claim"] is False
                            for r in qa["explicit_contrast_arithmetic_checks"]))

    def test_original_counters_and_false_flags_unchanged(self):
        fixture = F.make()
        original = copy.deepcopy(fixture)
        _, _, qa, snapshots = self.run_fixture(fixture)
        self.assertEqual(fixture, original)
        self.assertEqual(snapshots["snapshots"][0]["old_stage_counters"]["S"]["stage_changes"],
                         fixture[0]["result"]["observability"][0]["arms"]["S"]["stage_changes"])
        self.assertTrue(all(value is False for value in
                            qa["historical_flags_preserved_not_authority_or_release"].values()))

    def test_producer_and_receipt_mismatches_rejected(self):
        for mutate, code in (
            (lambda f: f[1].update(source_manifest_sha256="0" * 64), "pinned_executed_native_source"),
            (lambda f: f[1]["worker_terminal"].update(sha256="0" * 64), "launcher_terminal_binding"),
            (lambda f: f[2]["science_archive"]["objects"][0].update(state="UNKNOWN"),
             "producer_record_archive_SHA_URI_binding"),
            (lambda f: f[2]["science_archive"]["objects"][0]["receipt"].update(uri="wrong"),
             "producer_record_archive_SHA_URI_binding"),
            (lambda f: f[2]["child"].update(returncode=1), "runtime_completed_child"),
        ):
            with self.subTest(code=code):
                fixture = F.make()
                mutate(fixture)
                with self.assertRaisesRegex(ValueError, code):
                    self.run_fixture(fixture)

    def test_qualification_population_shape_and_finiteness_rejected(self):
        for mutate, code in (
            (lambda f: f[0]["result"]["passed_calls"].pop(), "exact72_worker_passed_calls"),
            (lambda f: f[0]["result"]["query_status"][0].update(passed_calls=8), "all8queries9calls_pass"),
            (lambda f: f[0]["result"]["observability"][0].update(query_index=22), "observability_fixed8"),
            (lambda f: f[0]["result"]["observability"][0].update(eligible_for_scientific_summary=False),
             "terminal_all72_qualified_observability"),
            (lambda f: f[0]["result"]["observability"][0]["arms"]["S"]["adapter7_effect"][0].pop(),
             "adapter7_shape_1_10_7"),
            (lambda f: f[0]["result"]["observability"][0]["arms"]["S"]["adapter7_effect"][0][0].__setitem__(0, float("nan")),
             "adapter7_finite_numeric"),
            (lambda f: f[0]["result"]["observability"][0]["arms"]["S"]["adapter7_effect"][0][0].__setitem__(0, True),
             "adapter7_finite_numeric"),
        ):
            with self.subTest(code=code):
                fixture = F.make()
                mutate(fixture)
                with self.assertRaisesRegex(ValueError, code):
                    self.run_fixture(fixture)

    def test_zero_dose_is_undefined_not_fabricated_zero_error(self):
        fixture = F.make()
        d = fixture[0]["result"]["observability"][0]["actual_dose"]
        d.update(shared_global_norm=0., random_global_norm=2., global_random_minus_shared_norm=2.)
        _, summary, _, snapshots = self.run_fixture(fixture)
        self.assertEqual(summary["dose"]["defined_relative_errors"], 7)
        self.assertEqual(summary["dose"]["undefined_shared_zero"], 1)
        self.assertIsNone(snapshots["snapshots"][0]["global_dose_relative_error"])

    def test_real_entry_gate_precedes_every_file_read_and_output(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "must-not-exist"
            with mock.patch.object(adapter.sys, "platform", "darwin"), \
                    mock.patch.object(Path, "read_bytes", side_effect=AssertionError("read-before-gate")):
                with self.assertRaisesRegex(ValueError, "real_DEV16_analysis_cloud_Linux_only"):
                    adapter.run(Path(d) / "missing", out, "0" * 64, 0)
            self.assertFalse(out.exists())

    def test_json_duplicate_and_nonfinite_rejected(self):
        with self.assertRaisesRegex(ValueError, "duplicate_JSON_key"):
            adapter.strict_json(b'{"a":1,"a":2}')
        with self.assertRaisesRegex(ValueError, "nonfinite_JSON"):
            adapter.strict_json(b'{"a":NaN}')

    def test_exact_outputs_under_bound_synthetic_serialization(self):
        rows, summary, qa, snapshots = self.run_fixture()
        for raw in (adapter.csv_bytes(rows), adapter.encoded(summary),
                    adapter.encoded(qa), adapter.encoded(snapshots)):
            self.assertLess(len(raw), 2 * 1024**2)
        self.assertEqual(len(adapter.OUTPUTS), 6)
        self.assertEqual(len(qa["explicit_contrast_arithmetic_checks"]), 16)
        self.assertEqual(sum(r["coordinates_checked"] for r in
                             qa["explicit_contrast_arithmetic_checks"]), 1120)

    def test_deadline_callback_reached(self):
        with self.assertRaisesRegex(RuntimeError, "synthetic_deadline"):
            A.analyze(*F.make(), synthetic=True,
                      check=lambda: (_ for _ in ()).throw(RuntimeError("synthetic_deadline")))


if __name__ == "__main__":
    unittest.main()
