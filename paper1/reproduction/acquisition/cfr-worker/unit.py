"""One unit (variant x init 0): E_T, E_C, warm-up with counterfactual tap, S0, pair gate.

New code (N2/N3/N4 of TECH-CONTRACT-v1 4.4) on top of the reused pieces named in each module.
Order per unit (chosen so that no E_C activity can perturb E_T or S0):
  1. E_C reset (compiled clean agentview fields are read here, before E_T's 10th step);
  2. E_T reset + 10 official dummy steps, the 10th with the counterfactual tap/transaction;
  3. S0 capture on E_T (q0 observations obs_T0 / obs_C0 carried inside S0);
  4. original-continuation replay record (T2) on E_T, then S0 restore (fingerprint checked);
  5. E_C official warm-up (own S0_C) and the physics pair gate.
"""

from __future__ import annotations

import hashlib
import time

import numpy as np

from .bases import identity_chain, stack_bases
from .call_log import identity_fields
from .common import TechnicalFailure, array_id, need
from .counterfactual import CameraCallbackCapture, SensorTap, camera_compatibility, camera_ids, clean_camera_fields
from .noise import bind_schedule, checked_noise
from .plus_env import pair_gate, variant_case, warmup
from .repair_math import gram_error
from .rng_guard import rng_fingerprint
from .selection import parse_variant
from . import snapshot as S

REPLAY_STEPS = 15


def replay_actions():
    """Frozen 15-step technical action sequence for T2 (no policy, no RNG)."""
    out = []
    for i in range(REPLAY_STEPS):
        out.append([0.2 * np.sin(0.7 * i), 0.2 * np.cos(0.5 * i), 0.1 * np.sin(0.3 * i), 0.0, 0.0,
                    0.05 * np.sin(1.0 * i), -1.0 if i < 8 else 1.0])
    return [list(map(float, a)) for a in out]


class Runtime:
    """Job-level handles: catalog, init reader, env classes, policy, bases, audit."""

    def __init__(self, *, classification, init_reader, mujoco, convention, variant_env_cls, base_env_cls,
                 policy, audit, split, bundle, tag, g_assigned="G1"):
        self.classification, self.init_reader, self.mujoco = classification, init_reader, mujoco
        self.convention, self.variant_env_cls, self.base_env_cls = convention, variant_env_cls, base_env_cls
        self.policy, self.audit, self.split, self.bundle, self.tag = policy, audit, split, bundle, tag
        self.g_assigned = g_assigned
        self.random_bases, self.random_proofs = bundle.random_bases() if bundle is not None else ({}, {})
        self.state_mode = S.integration_mode(mujoco)


class UnitContext:
    def __init__(self, rt, unit, *, noise_label=0, with_bases=True, keep_render_data=False, record_replay=True):
        self.rt, self.unit, self.noise_label = rt, unit, noise_label
        self.family = unit["family"]
        self.case = variant_case(unit, rt.classification)
        self.params = parse_variant(unit["variant_name"])
        self.timing = {}
        self.failures = []
        if with_bases:
            unit_bases, self.template, self.identity = identity_chain(unit, rt.split, rt.bundle)
            self.bases_stack, self.bases_by_name = stack_bases(unit_bases, rt.random_bases)
            self.gram = {name: gram_error(b) for name, b in self.bases_by_name.items()}
            self.identity_log = identity_fields(self.identity, rt.random_proofs, rt.g_assigned)
        t0 = time.monotonic()
        # 1. base env first: compiled clean camera fields
        self.E_C = rt.base_env_cls(rt.classification, protocol_mode="released_task_language", init_reader=rt.init_reader)
        self.E_C.reset(self.case)
        self.h_C = S.SimHandles(self.E_C.env, rt.mujoco)
        # 2. treated env with counterfactual tap on the 10th dummy step
        self.E_T = rt.variant_env_cls(rt.classification, protocol_mode="released_task_language", init_reader=rt.init_reader)
        self.E_T.reset(self.case)
        self.h_T = S.SimHandles(self.E_T.env, rt.mujoco)
        self.prompt = self.E_T.prompt
        self.capture = None
        self.camera_recoverability = None
        if self.family == "Camera Viewpoints":
            ids = camera_ids(rt.mujoco, self.h_T.raw_model)
            base_ids = camera_ids(rt.mujoco, self.h_C.raw_model)
            self.clean_fields = clean_camera_fields(self.h_C.sim.model, base_ids["agentview"])
            self.camera_recoverability = camera_compatibility(self.h_T.sim.model, self.clean_fields, ids["agentview"])
            need(self.camera_recoverability["recoverable_by_camera_only_transaction"],
                 "camera_variant_not_recoverable_by_camera_only_transaction")
            self.capture = CameraCallbackCapture(self.h_T, self.clean_fields, convention=rt.convention,
                                                 keep_render_data=keep_render_data)
        self.timing["environment_construct_s"] = time.monotonic() - t0
        t1 = time.monotonic()
        self.steps_T = 0
        obs_T0, obs_C0, receipt = self._warmup_treated()
        self.obs_T0, self.obs_C0, self.cf_receipt0 = obs_T0, obs_C0, receipt
        # 3. S0
        self.S0 = S.capture(self.h_T, mode=rt.state_mode)
        self.S0_fingerprint = S.fingerprint(self.S0)
        # 4. original continuation (T2) then restore
        self.original_replay = self.replay_record(self.E_T, self.h_T) if record_replay else None
        if record_replay:
            self.restore_T()
        # 5. base own warm-up and pair gate
        self.obs_C_own0 = warmup(self.E_C)
        self.S0_C = S.capture(self.h_C, mode=rt.state_mode)
        self.pair = pair_gate(self.E_T, self.E_C)
        self.timing["warmup_snapshot_pair_s"] = time.monotonic() - t1
        self.horizon_queries = (self.case["max_policy_steps"] + 4) // 5
        self.noise_schedule = bind_schedule(tag=rt.tag, family=self.family, suite=self.case["suite"],
                                            classification_id=self.case["classification_id"], init_index=0,
                                            noise_label=noise_label, horizon_queries=self.horizon_queries)

    # ------------------------------------------------------------- stepping
    def _warmup_treated(self):
        tap_record = {}

        def tapped(action):
            obs, done = self.tapped_step(self.E_T, self.h_T, action, step_index=10)
            tap_record["receipt"] = self._last_cf
            return obs, done
        obs = warmup(self.E_T, tapped_step=tapped)
        self.steps_T = 10
        return obs, self._last_obs_c, tap_record.get("receipt")

    def tapped_step(self, env, h, action, *, step_index):
        """One chunk-final wrapper step with the counterfactual capture; sets _last_obs_c/_last_cf."""
        started = time.monotonic()
        if self.family == "Sensor Noise":
            tap = SensorTap(env.env, lambda: env.env.get_sim_state(), raw_data=h.raw_data)
            obs, done = tap.step(action, invoke=env.step)
            clean, receipt = tap.clean_observation()
            receipt["query_physics_sha256"] = array_id(np.asarray(env.env.get_sim_state()))["sha256"]
            receipt["render_data"] = tap.record["render_data"]
            receipt["source_physics"] = tap.record["source_physics"]
        else:
            need(self.capture is not None, "camera_capture_not_installed")
            self.capture.step = step_index
            self.capture.enabled = True
            try:
                obs, done = env.step(action)
            finally:
                self.capture.enabled = False
            clean, receipt = self.capture.take(step_index, obs)
            receipt["query_physics_sha256"] = array_id(np.asarray(env.env.get_sim_state()))["sha256"]
            receipt["render_data"] = self.capture.records["agentview"]["render_data"]
            receipt["source_physics"] = self.capture.records["agentview"]["source_physics"]
        receipt["source_equals_query_state"] = receipt["source_physics_sha256"] == receipt["query_physics_sha256"]
        receipt["seconds"] = time.monotonic() - started
        self._last_obs_c, self._last_cf = clean, receipt
        return obs, done

    def restore_T(self):
        info = S.restore(self.h_T, self.S0)
        check = S.compare(self.S0_fingerprint, S.fingerprint(S.recapture(self.h_T, self.S0)))
        if not check["equal"]:
            raise TechnicalFailure("S0_restore_fingerprint_differs_E_T", check)
        self.steps_T = 10
        return info, check

    def restore_C(self):
        info = S.restore(self.h_C, self.S0, into_base=True)
        check = S.compare(self.S0_fingerprint, S.fingerprint(S.recapture(self.h_C, self.S0)), exclude_prefixes=S.BASE_EXCLUDE)
        if not check["equal"]:
            raise TechnicalFailure("S0_restore_fingerprint_differs_E_C", check)
        return info, check

    def step_fingerprint(self, h, obs):
        flat = np.asarray(h.wrapper.get_sim_state())
        robots = [S.tree_digest(S.capture_tree(r)) for r in h.env.robots]
        return {"physics": array_id(flat)["sha256"],
                "controller": hashlib.sha256(repr([sorted(r.items()) for r in robots]).encode()).hexdigest(),
                "observation": {k: array_id(v)["sha256"] for k, v in obs.items() if isinstance(v, np.ndarray)},
                "rng": hashlib.sha256(repr(sorted(rng_fingerprint().items())).encode()).hexdigest()}

    def replay_record(self, env, h):
        rows = []
        for action in replay_actions():
            obs, done = env.step(action)
            rows.append(self.step_fingerprint(h, obs))
            if done:
                break
        return rows

    def noise(self, query):
        return checked_noise(self.noise_schedule, query)

    def close(self):
        try:
            if self.capture is not None:
                self.capture.close()
        finally:
            for env in (self.E_T, self.E_C):
                try:
                    env.close()
                except Exception:  # noqa: BLE001
                    pass
