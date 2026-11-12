"""Synthetic arrays only; exercise the retained writer and actual hash domain."""
import copy
import unittest
import numpy as np
import pinned_campaign as P
import pinned_writer as W
import physical_codec as C
import validate as V


def state(step, offset=0.):
    return {"flat_physics": np.full(123, step+offset, np.float64),
            "sim_time": np.asarray(step*.05),
            "returned_eef_pos": np.array([step+offset, 0., 0.]),
            "returned_eef_quat": np.array([0., 0., 0., 1.]),
            "pose_source": "actual_returned_observation_not_resampled"}


def trajectory(offset):
    rows = []
    for i in range(5):
        before, after = state(55+i, i*offset), state(56+i, (i+1)*offset)
        rows.append({"before": before, "after": after, "done": False, "official_success": False,
            "physical_outputs": {"returned_nonimage_observations": {
                "robot0_eef_pos": after["returned_eef_pos"].copy(),
                "robot0_eef_quat": after["returned_eef_quat"].copy()},
                "gripper_current_action": np.array([-1., 1.]), "contacts": []}})
    return rows


def fixture():
    groups = sorted(P.OLD_MIXED)+["controller.0"]+["other."+str(i) for i in range(56)]
    statuses = {k: "DIFFERENT" if k in P.OLD_MIXED else "EXACT" for k in groups}
    obs = {k: {} for k in ("robot0_eef_pos", "robot0_eef_quat", *sorted(P.IMAGE_KEYS))}
    anchors = [{"query_index": q, "observations": obs} for q in (0, 11, 12)]
    steps = []
    for q in range(12):
        for slot in range(5):
            steps.append({"query_index": q, "slot": slot,
                "before": {k: C.array_identity(state(q*5+slot)[k]) for k in P.STEP_FIELDS},
                "after": {k: C.array_identity(state(q*5+slot+1)[k]) for k in P.STEP_FIELDS}})
    qualified = {"parent": {"root_id": P.ROOT, "all_source_continuity_exact": True,
        "q12_commands_used": 0, "steps": steps, "anchors": anchors,
        "protected_queries": [{"query_index": q, "source_groups_before": {k: "frozen" for k in groups}}
                              for q in range(13)]},
        "join": {"native_final72_qualification_bound": True, "actual_saved_input_body_decoded": True,
                 "cached_commands": {"N": {"entry": {"receiver": "T", "role": "baseline_none_1"}}}}}
    roles = []
    for name, count, offset in (("source_qualification", 13, 0.), ("N", 12, .01)):
        rows = trajectory(offset)
        queries = [{"query": q, "source_group_status": dict(statuses),
                    "old_all_declared_groups_exact": False,
                    "old_full_replay_qualification_relabelled": False,
                    "anchor": {"physics_exact": True, "observation_typed_exact": {
                        k: k not in P.IMAGE_KEYS for k in obs}} if q in (0, 11, 12) else None}
                   for q in range(count)]
        role = {"role": name, "status": "QUALIFIED_SOURCE_PATH" if name == "source_qualification" else "FAILED_NO_RETRY",
            "env_steps": 70, "model_calls": 0, "model_attempts": 0,
            "observed_continuation_steps": 5, "trajectory": rows, "queries": queries,
            "start_identity": "a"*64, "trajectory_identity": P.trajectory_identity(rows),
            "start_critical_witness": {"gripper_current_action": np.array([-1., 1.]),
                "controller": {"controller.0": "same"}, "compiled_model": {"model": "same"}}}
        if name == "N":
            role["close_failure"] = {"code": "one_window_finish", "type": "PairingError"}
        roles.append(role)
    roles.extend({"role": r, "status": "NOT_ATTEMPTED_PREVIOUS_GATE_FAILED", "env_steps": 0,
                  "model_calls": 0, "model_attempts": 0} for r in ("N_repeat", "S", "R"))
    campaign = {"schema": "paper2-q11-cached-command-campaign-v1", "synthetic": False,
        "completed_all_roles": False, "total_explicit_env_steps": 140, "GPU": 0,
        "model_calls": 0, "model_attempts": 0, "roles": roles}
    entered = {"construct": 4, "reset": 4, "set_init_state": 4, "mj_step": 3500,
               "mj_step1": 8, "mj_step2": 0, "mj_forward": 7038}
    native = {"total_explicit_env_steps": 140, "failure": None, "completed_all_roles": False,
        "GPU": 0, "new_model_calls": 0, "model_attempts": 0, "model_import_attempts": 0,
        "roles": [{k: v for k, v in r.items() if k not in ("trajectory", "queries", "start_critical_witness")}
                  for r in roles], "native_budget": {"limits": V.LIMITS, "aborted": None,
                  "entered_calls": entered, "attempted_calls": dict(entered),
                  "reset_attempts": 4, "reset_successes": 4}}
    return campaign, native, qualified


class Witness(unittest.TestCase):
    def test_actual_writer_roundtrip_recomputes_both_trajectory_identities(self):
        campaign, native, qualified = fixture()
        encoded = C.json_bytes(W.encoded(campaign))
        counts = {"arrays": 0, "numeric_bytes": 0}
        recovered = C.decode(encoded, counts)
        witness, checks = V.inspect_campaign(recovered, native, qualified)
        self.assertTrue(witness["retained_observation_input_eligible"])
        self.assertFalse(witness["new_environment_execution_authorized"])
        self.assertFalse(witness["original_attempt_relabelled_complete"])
        self.assertEqual(checks["old_unmixed_group_comparisons_exact"], 1425)
        self.assertEqual(witness["maximum_new_environment_steps"], 210)

    def test_payload_damage_rejected(self):
        wrapped = W.public_tree(np.array([-0., 1.]))
        wrapped["values"][1] = 2.
        with self.assertRaisesRegex(ValueError, "payload_SHA"):
            C.decode(wrapped, {"arrays": 0, "numeric_bytes": 0})

    def test_signed_zero_and_endianness_preserved(self):
        value = np.array([-0., 1.], dtype=">f8")
        decoded = C.decode(C.json_bytes(W.encoded(value)), {"arrays": 0, "numeric_bytes": 0})
        self.assertEqual(P.digest(value), P.digest(decoded))

    def test_other_close_error_not_reinterpreted_as_reusable(self):
        campaign, native, qualified = fixture()
        campaign["roles"][1]["close_failure"]["code"] = "unknown"
        with self.assertRaisesRegex(ValueError, "exact_post_observation_close_failure"):
            V.inspect_campaign(campaign, native, qualified)

    def test_unmixed_source_mismatch_remains_failure(self):
        campaign, native, qualified = fixture()
        campaign["roles"][0]["queries"][0]["source_group_status"]["controller.0"] = "DIFFERENT"
        with self.assertRaisesRegex(ValueError, "unmixed_groups"):
            V.inspect_campaign(campaign, native, qualified)

    def test_original_transition_payload_identity_rechecked(self):
        campaign, native, qualified = fixture()
        qualified["parent"]["steps"][55]["before"]["flat_physics"]["sha256"] = "0"*64
        with self.assertRaisesRegex(ValueError, "original_target_transition"):
            V.inspect_campaign(campaign, native, qualified)

    def test_missing_repeat_cannot_be_filled_with_source_role(self):
        campaign, native, qualified = fixture()
        campaign["roles"][2]["status"] = "QUALIFIED_SOURCE_PATH"
        native["roles"][2]["status"] = "QUALIFIED_SOURCE_PATH"
        with self.assertRaisesRegex(ValueError, "missing_roles"):
            V.inspect_campaign(campaign, native, qualified)

    def test_cached_N_must_keep_receiver_T_semantics(self):
        campaign, native, qualified = fixture()
        qualified["join"]["cached_commands"]["N"]["entry"]["receiver"] = "C"
        with self.assertRaisesRegex(ValueError, "cached_receiver_T"):
            V.inspect_campaign(campaign, native, qualified)


if __name__ == "__main__":
    unittest.main()
