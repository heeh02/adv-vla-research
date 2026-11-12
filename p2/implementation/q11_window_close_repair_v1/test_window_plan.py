"""Exercise the exact original recorder class without importing any simulator."""
import ast
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from window_plan import make_plan, finish_reason

BASE = Path(__file__).resolve().parents[2]
ORIGINAL = BASE/"implementation/cpu_action_replay_recovery_v3/retained_cpu3"
RELATIVE = "study/continuous/collector-v2/collector.py"


def original_recorder():
    raw = (ORIGINAL/RELATIVE).read_bytes()
    manifest = json.loads((ORIGINAL/"metadata/SOURCE_MANIFEST-v1.json").read_bytes())
    pin = next(r for r in manifest["files"] if r["path"] == RELATIVE)
    assert len(raw) == pin["bytes"] and hashlib.sha256(raw).hexdigest() == pin["sha256"]
    tree = ast.parse(raw)
    exact_class = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                       and node.name == "ScheduledRecorder")
    def need(ok, code):
        if not ok:
            raise ValueError(code)
    namespace = {"copy": copy, "S": SimpleNamespace(need=need)}
    # No edits to its methods; only this dependency-free class definition runs.
    exec(compile(ast.Module(body=[exact_class], type_ignores=[]), str(ORIGINAL/RELATIVE), "exec"), namespace)
    return namespace["ScheduledRecorder"]


class Window(unittest.TestCase):
    def recorder(self, role):
        root = {"official_max_policy_steps": 520,
                "tiers": {str(q): {"synthetic_tier": True} for q in range(13)}}
        return original_recorder()(make_plan(root, role, "SDRU-CDEV-T00-R00"),
                                   lambda *_: "synthetic-receipt")

    def test_old_error_reproduced_by_actual_original_finish(self):
        recorder = self.recorder("source_qualification")
        recorder.next = 12
        with self.assertRaisesRegex(ValueError, "one_window_finish"):
            recorder.finish("candidate_window_closed")

    def test_source_still_requires_q12(self):
        recorder = self.recorder("source_qualification")
        self.assertEqual(len(recorder.rows), 13)
        recorder.next = 12
        with self.assertRaisesRegex(ValueError, "cannot_complete_missing_slots"):
            recorder.finish("window_complete")
        recorder.next = 13
        recorder.finish(finish_reason(13, 13, False))
        self.assertTrue(recorder.closed)

    def test_each_cached_branch_closes_after_exact_q11_window(self):
        for role in ("N", "N_repeat", "S", "R"):
            with self.subTest(role=role):
                recorder = self.recorder(role)
                self.assertEqual([r["planned_query_index"] for r in recorder.rows], list(range(12)))
                recorder.next = 12
                recorder.finish(finish_reason(12, 12, False))
                self.assertTrue(recorder.closed)

    def test_incomplete_windows_never_relabelled_complete(self):
        for terminal in (False, True):
            recorder = self.recorder("R")
            recorder.next = 11
            reason = finish_reason(11, 12, terminal)
            self.assertNotEqual(reason, "window_complete")
            result = recorder.finish(reason)
            self.assertEqual(result["rows"][-1]["status"], "unreached")


if __name__ == "__main__":
    unittest.main()
