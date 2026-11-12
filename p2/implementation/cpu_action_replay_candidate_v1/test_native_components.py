"""Fakes only: never imports or constructs MuJoCo / LIBERO / RoboSuite."""
import gzip
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch
import numpy as np
import native_budget as B
from native_facade import NativeReplay
from physical_writer import Events, public_tree, write_new


def fake_modules():
    calls = []
    sentinel = object()
    def lower_reset():
        calls.append("lower_reset")
        return sentinel
    class Control:
        def __init__(self):
            self.noise = 0
            self.env = NS(control_freq=20, control_timestep=.05, model_timestep=.002,
                          sim=NS(model=NS(_model=NS(opt=NS(timestep=.002)))),
                          reset=lower_reset)
        def reset(self):
            raise AssertionError("old_retry_loop_must_never_run")
        def set_init_state(self, x):
            calls.append("set_init_state")
            return x
    def engine(name):
        def call(*args, **kwargs):
            calls.append(name)
            return name
        return call
    mujoco = NS(**{k: engine(k) for k in ("mj_step", "mj_step1", "mj_step2", "mj_forward")})
    pdb = NS(set_trace=engine("debugger"))
    return Control, mujoco, pdb, calls, sentinel


class NativeBudgetTests(unittest.TestCase):
    def setUp(self):
        self.control, self.m, self.p, self.calls, self.sentinel = fake_modules()
        self.budget = B.Budget(time.time()+10)
        self.saved = (self.control.__init__, self.control.reset, self.m.mj_step, self.p.set_trace)
        self.stack = B.install(self.control, self.m, self.p, self.budget,
                               role="N", emit=lambda *_: None)
        self.addCleanup(self.stack.close)

    def test_success_path_exact_return_one_reset_and_restoration(self):
        c = self.control()
        self.assertIs(c.reset(), self.sentinel)
        self.assertEqual(c.set_init_state("identity"), "identity")
        self.m.mj_step(1, 2)
        self.assertEqual(self.calls, ["lower_reset", "set_init_state", "mj_step"])
        self.assertEqual(self.budget.counts["reset"], 1)
        self.assertEqual(self.budget.resets.successful, 1)
        self.stack.close()
        self.assertEqual((self.control.__init__, self.control.reset, self.m.mj_step,
                          self.p.set_trace), self.saved)

    def test_native_cap_charged_before_call_and_sticky(self):
        self.budget.counts["mj_step"] = B.LIMITS["mj_step"]
        with self.assertRaisesRegex(B.NativeBudgetError, "native_cap_mj_step"):
            self.m.mj_step(1, 2)
        self.assertEqual(self.calls, [])
        self.assertEqual(self.budget.attempted["mj_step"], 1)
        with self.assertRaisesRegex(B.NativeBudgetError, "native_cap_mj_step"):
            self.m.mj_forward(1, 2)

    def test_split_integrator_forbidden_and_batch_ABI_rejected(self):
        with self.assertRaisesRegex(B.NativeBudgetError, "native_cap_mj_step2"):
            self.m.mj_step2(1, 2)
        self.assertEqual(self.calls, [])
        self.budget.aborted = None
        with self.assertRaisesRegex(B.NativeBudgetError, "batch_ABI"):
            self.m.mj_step(1, 2, nstep=5)
        self.assertEqual(self.calls, [])

    def test_clock_drift_stops_before_reset_and_dummies(self):
        c = self.control()
        c.env.sim.model._model.opt.timestep = .001
        # Clock is rechecked after the one explicitly authorized init/reset;
        # the caller cannot subsequently enter any engine step.
        with self.assertRaisesRegex(B.NativeBudgetError, "actual_native_clock"):
            c.set_init_state("x")
        with self.assertRaises(B.NativeBudgetError):
            self.m.mj_step(1, 2)
        self.assertNotIn("mj_step", self.calls)

    def test_failed_reset_not_retried_and_counts_entered_calls(self):
        c = self.control()
        c.env.reset = lambda: (_ for _ in ()).throw(KeyboardInterrupt())
        with self.assertRaises(KeyboardInterrupt):
            c.reset()
        self.assertEqual(self.budget.resets.attempted, 1)
        self.assertEqual(self.budget.resets.successful, 0)
        self.assertEqual(self.budget.counts["reset"], 1)

    def test_placement_failure_blocks_interaction(self):
        with self.assertRaisesRegex(RuntimeError, "placement_failed_no_debugger"):
            self.p.set_trace()
        self.assertNotIn("debugger", self.calls)
        self.assertEqual(self.budget.aborted, "placement_debugger_requested")

    def test_deadline_rejected_before_constructor(self):
        self.budget.deadline_epoch = time.time()-1
        with self.assertRaisesRegex(B.NativeBudgetError, "absolute_deadline"):
            self.control()
        self.assertEqual(self.budget.counts["construct"], 0)


class WriterTests(unittest.TestCase):
    def test_typed_numeric_roundtrip_and_signed_zero(self):
        value = np.array([-0., 1., 2.], dtype=">f8")
        doc = public_tree(value)
        rebuilt = np.array(doc["values"], dtype=doc["dtype"]).reshape(doc["shape"])
        self.assertEqual(hashlib.sha256(rebuilt.tobytes()).hexdigest(), doc["payload_sha256"])
        self.assertEqual(value.tobytes(), rebuilt.tobytes())

    def test_images_nonfinite_and_objects_rejected(self):
        for value in (np.zeros((224, 224, 3), np.uint8),
                      np.array([np.nan]), np.array([None], dtype=object)):
            with self.assertRaises(ValueError):
                public_tree(value)

    def test_concatenated_events_reopen_and_prewrite_caps(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td)/"events.gz"
            log = Events(path, check=lambda: None, max_compressed=1000)
            log.emit("A", {"x": 1})
            log.emit("B", {"x": 2})
            before = path.read_bytes()
            log.compressed_limit = len(before)
            with self.assertRaisesRegex(ValueError, "compressed_cap"):
                log.emit("C", {"x": 3})
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual(log.count, 2)
            log.close()
            rows = [json.loads(x) for x in gzip.decompress(path.read_bytes()).splitlines()]
            self.assertEqual([r["event"] for r in rows], ["A", "B"])

    def test_write_is_exclusive_and_limits_before_creation(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td)/"result.json"
            with self.assertRaisesRegex(ValueError, "output_cap"):
                write_new(path, {"x": "too_long"}, limit=1)
            self.assertFalse(path.exists())
            write_new(path, {"x": 1})
            with self.assertRaises(FileExistsError):
                write_new(path, {"x": 2})


class NativeFacadeTests(unittest.TestCase):
    def test_unapproved_open_fails_before_any_dynamic_module(self):
        runtime = NativeReplay({}, emit=lambda *_: None)
        with patch("native_facade.importlib.import_module",
                   side_effect=AssertionError("no_target_import")):
            with self.assertRaisesRegex(ValueError, "new_direct_environment_authority"):
                runtime.open(object(), check=lambda: None)
        runtime.close()
        self.assertEqual(runtime.env_step_calls, 0)

    def test_compiled_identity_excludes_auxiliary_receipt_metadata(self):
        a = NativeReplay({}, emit=lambda *_: None)
        b = NativeReplay({}, emit=lambda *_: None)
        metadata = {"role": "current_clean_parent_after_exact_selected_init",
                    "name": "compiled-model-fields", "receipt": "old"}
        a._capture_compiled("SETUP_COMPILED_ARRAYS", metadata, {"x": np.array([1.])})
        b._capture_compiled("SETUP_COMPILED_ARRAYS", {**metadata, "receipt": "new"},
                            {"x": np.array([1.])})
        self.assertEqual(a.compiled_parts, b.compiled_parts)
        with self.assertRaisesRegex(ValueError, "unique_compiled_snapshot"):
            a._capture_compiled("SETUP_COMPILED_ARRAYS", metadata, {})

    def test_model_callback_records_attempt_before_rejection(self):
        runtime = NativeReplay({}, emit=lambda *_: None)
        with self.assertRaisesRegex(RuntimeError, "no_model"):
            runtime.forbid_neural()
        self.assertEqual(runtime.model_attempts, 1)
        self.assertEqual(runtime.model_calls, 0)


if __name__ == "__main__":
    unittest.main()
