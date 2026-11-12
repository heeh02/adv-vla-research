"""Fake state-machine exercises only; no simulator or recorded arrays."""
import copy
import time
import unittest
import numpy as np
import campaign_core as C


def state(step):
    return {"flat_physics": np.full(123, step, np.float64),
            "sim_time": np.asarray(step / 20, np.float64),
            "returned_eef_pos": np.array([step, 0, 0], np.float64),
            "returned_eef_quat": np.array([0, 0, 0, 1], np.float64)}


class Source:
    root_id = C.ROOT
    def diagnostic(self, q):
        return {"before": {"controller.0": "same", "root.clock": str(q),
                           "environment": "source_with_other_pixels"}}
    def anchor(self, q):
        return {"physical_sim_state": state(q * 5)["flat_physics"],
                "observation": {"robot0_eef_pos": state(q * 5)["returned_eef_pos"],
                                "agentview_image": np.zeros((2, 2, 3), np.uint8)}}
    def parent_actions(self, q):
        return np.full((1, 10, 7), q / 100, np.float64)
    def issued(self, q, slot):
        step = 5 * q + slot
        return {"values": {"before": state(step), "after": state(step + 1)}}
    def outcome(self, q, slot):
        return {"done": False, "official_success": False}


class Runtime:
    synthetic = True
    model_calls = 0
    def __init__(self, role, fault=None):
        self.role, self.fault = role, fault
        self.env_step_calls, self.steps = 0, 0
        self.closed, self.q = False, None
    def open(self, source, check):
        check()
        self.source = source
        self.env_step_calls = 10
    def query(self, q):
        self.q = q
        if self.fault == "model" and q == 2:
            self.model_calls = 1
        obs = self.source.anchor(q)["observation"]
        obs["agentview_image"] = np.ones((2, 2, 3), np.uint8)
        return {"flat_physics": state(q * 5)["flat_physics"], "observation": obs}
    def group_ids(self):
        result = {**self.source.diagnostic(self.q)["before"], "environment": "same_fresh_CPU_pixels"}
        if self.fault == "source" and self.q == 2:
            result["controller.0"] = "mismatch"
        return result
    def critical_state(self):
        return {"gripper_current_action": np.array([1., -1.]) +
                (1 if self.fault == "start" and self.role == "N_repeat" else 0),
                "environment": {"model_timestep": .002, "control_timestep": .05},
                "controller": "same", "compiled_model": "same"}
    def bind_recorded_actions(self, chunk, q):
        self.cached = False
    def bind_cached_commands(self, actions, raw32, q, key):
        self.cached = True
        self.key = key
    def step(self, action, *, query, slot, steps):
        before, after = state(steps), state(steps + 1)
        self.env_step_calls += 1
        self.steps += 1
        done = self.fault == "terminal" and query == 11 and slot == 1 and self.role in ("N", "N_repeat")
        response = np.array([float(action[0]), steps, 0])
        if self.fault == "repeat" and self.role == "N_repeat" and query == 11:
            response[0] += .01
        return {"before": before, "after": after, "done": done, "official_success": False,
                "physical_outputs": {"synthetic_EEF": response}}
    def query_done(self, q):
        pass
    def close(self):
        self.closed = True


class Campaign(unittest.TestCase):
    def run_fake(self, fault=None, synthetic=True, authority=None):
        runtimes, events = [], []
        def factory(role):
            runtime = Runtime(role, fault)
            runtimes.append(runtime)
            return runtime
        cached = {arm: {"actions": np.full((1, 10, 7), i / 10, np.float64),
                        "raw32": np.zeros((1, 10, 32), np.float32)}
                  for i, arm in enumerate(("N", "S", "R"))}
        result = C.run_campaign(Source(), cached, factory, authority=authority,
                                emit=lambda *x: events.append(x),
                                deadline_epoch=time.time()+10, synthetic=synthetic)
        return result, runtimes, events

    def test_five_roles350_steps_and_old_pixel_failure_retained(self):
        result, runtimes, _ = self.run_fake()
        self.assertTrue(result["completed_all_roles"])
        self.assertEqual(result["total_explicit_env_steps"], 350)
        self.assertEqual([r["role"] for r in result["roles"]], list(C.ROLES))
        self.assertTrue(all(r.closed for r in runtimes))
        self.assertFalse(result["roles"][0]["queries"][0]["old_all_declared_groups_exact"])
        self.assertFalse(result["old_full_replay_qualification_changed"])

    def test_missing_direct_authority_rejected_before_runtime(self):
        with self.assertRaisesRegex(ValueError, "new_direct_environment_authority"):
            self.run_fake(synthetic=False)

    def test_source_failure_prevents_all_cached_branches(self):
        result, runtimes, _ = self.run_fake("source")
        self.assertEqual(len(runtimes), 1)
        self.assertEqual(result["roles"][0]["status"], "FAILED_NO_RETRY")
        self.assertTrue(all(r["status"].startswith("NOT_") for r in result["roles"][1:]))
        self.assertTrue(runtimes[0].closed)

    def test_gripper_start_mismatch_stops_before_S_R(self):
        result, runtimes, _ = self.run_fake("start")
        self.assertEqual(len(runtimes), 3)
        self.assertEqual(result["roles"][2]["failure"]["code"], "independent_rebuild_start_mismatch")
        self.assertTrue(all(r["status"].startswith("NOT_") for r in result["roles"][3:]))

    def test_repeat_mismatch_stops_before_S_R(self):
        result, runtimes, _ = self.run_fake("repeat")
        self.assertEqual(len(runtimes), 3)
        self.assertEqual(result["roles"][2]["failure"]["code"], "N_repeat_trajectory_mismatch")
        self.assertFalse(result["completed_all_roles"])

    def test_equal_early_terminal_prefix_preserved_not_filled(self):
        result, _, _ = self.run_fake("terminal")
        self.assertTrue(result["completed_all_roles"])
        self.assertEqual(result["total_explicit_env_steps"], 344)
        self.assertEqual(result["roles"][1]["observed_continuation_steps"], 2)
        self.assertEqual(result["roles"][2]["observed_continuation_steps"], 2)

    def test_typed_snapshot_distinguishes_dtype_and_signed_zero(self):
        self.assertNotEqual(C.digest(np.array([0.], np.float32)), C.digest(np.array([0.], np.float64)))
        self.assertNotEqual(C.digest(np.array([0.])), C.digest(np.array([-0.])))

    def test_cached_wrong_dtype_rejected(self):
        cached = {arm: {"actions": np.zeros((1, 10, 7), np.float32),
                        "raw32": np.zeros((1, 10, 32), np.float32)} for arm in ("N", "S", "R")}
        with self.assertRaisesRegex(ValueError, "cached_command_ABI"):
            C.run_campaign(Source(), cached, lambda role: self.fail("factory must not run"),
                           authority=None, emit=lambda *_: None,
                           deadline_epoch=time.time()+10, synthetic=True)

    def test_unexpected_model_counter_is_reported_not_overwritten_with_zero(self):
        result, runtimes, _ = self.run_fake("model")
        self.assertEqual(len(runtimes), 1)
        self.assertEqual(result["model_calls"], 1)
        self.assertEqual(result["roles"][0]["failure"]["code"], "unexpected_model_activity")
        self.assertFalse(result["completed_all_roles"])


if __name__ == "__main__":
    unittest.main()
