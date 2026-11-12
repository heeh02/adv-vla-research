"""Loaded pi05 policy wrapper: canonical, record-all and repair routes (cloud only).

Adapts NSA/execution/closed_loop/native_adapter.py:177-247 (``NativeCampaignQuery``):
the stock ``CampaignPolicy.sample`` graph serves N_C/N_T (no hook), separate
``module_jit`` graphs serve the record-all and repair routes, encode/decode are the
original CampaignPolicy methods, and actual BF16 writes are accounted from cast bits
(native_adapter.py:222-230). Every neural call runs inside the RNG audit.
"""
# ruff: noqa: PLC0415, SLF001

from __future__ import annotations

import copy
import time
from types import MethodType

import numpy as np

from .common import TechnicalFailure, array_id, need
from .repair_math import ROUTE_CODE

ARM_ROUTES = {
    "R_resp": ("natural", "B"), "R_clean": ("natural", "C"), "R_rand": ("natural", "G"),
    "R_full": ("full", None), "hook_no_write": ("hook_no_write", "B"), "alpha0": ("alpha_fp32_no_rule", "B"),
    "eq_B": ("equal_norm", "B"), "eq_C": ("equal_norm", "C"), "eq_G": ("equal_norm", "G"),
}
BASIS_INDEX = {"B": 0, "C": 1, "G1": 2, "G2": 3, "G3": 4}


def arm_config(arm, *, g_assigned="G1", basis_override=None, alpha=None):
    route, basis = ARM_ROUTES[arm]
    if basis == "G":
        basis = g_assigned
    if basis_override is not None:
        basis = basis_override
    if alpha is None:
        alpha = 0.0 if arm == "alpha0" else 1.0
    return {"arm": arm, "route": route, "route_code": ROUTE_CODE[route],
            "basis": basis, "arm_index": BASIS_INDEX[basis] if basis else 0,
            "eq_names": ["B", "C", g_assigned], "eq_index": [0, 1, BASIS_INDEX[g_assigned]],
            "alpha": float(alpha)}


class LoadedPolicy:
    def __init__(self, campaign, audit, *, c_noop=1.0):
        import jax

        from openpi.eval.campaign_policy import CampaignPolicy
        from openpi.models.pi0 import Pi0
        from openpi.shared.nnx_utils import module_jit

        import cfr.repair_graph as G
        need(isinstance(campaign, CampaignPolicy) and isinstance(campaign.policy._model, Pi0),
             "original_loaded_CampaignPolicy_Pi0")
        self.campaign, self.audit, self.c_noop = campaign, audit, float(c_noop)
        model = campaign.policy._model
        self.model = model
        self.record_jit = module_jit(MethodType(G.record_query, model))
        # One jitted function; clean_layers is static (9 = variant A, 18 = variant B).
        self.repair_jit = module_jit(MethodType(G.repair_query, model), static_argnames=("clean_layers",))
        self.compiled = {}
        self.jax = jax
        self.precision = {"jax_default_matmul_precision": str(jax.config.jax_default_matmul_precision),
                          "write_projection_precision": "jax.lax.Precision.HIGHEST"}
        self.query_budget = None  # Compilation is synthetic; owner sets approved 1A call caps afterwards.
        self.query_counts = {"native": 0, "repair": 0, "episode_native": 0, "episode_repair": 0}
        self.episode_mode = False

    def set_query_budget(self, *, native, repair, episode_native, episode_repair):
        budget = {"native": native, "repair": repair, "episode_native": episode_native,
                  "episode_repair": episode_repair}
        need(all(type(v) is int and v >= 0 for v in budget.values()), "explicit_nonnegative_query_caps")
        self.query_budget = budget

    def count_query(self, route):
        if self.query_budget is None:
            return
        key = ("episode_" if self.episode_mode else "") + route
        need(self.query_counts[key] < self.query_budget[key], "approved_query_cap_reached:" + key)
        self.query_counts[key] += 1

    # ------------------------------------------------------------- inputs
    def encode(self, observation, prompt):
        encoded, batch = self.campaign.encode([copy.deepcopy(observation)], [prompt])
        return encoded, batch

    def encode_pair(self, obs_t, obs_c, prompt):
        enc_t, batch_t = self.encode(obs_t, prompt)
        enc_c, batch_c = self.encode(obs_c, prompt)
        same = {}
        for key in ("state", "tokenized_prompt", "tokenized_prompt_mask"):
            if key in batch_t:
                same[key] = array_id(batch_t[key]) == array_id(batch_c[key])
        for key in ("image_mask",):
            if key in batch_t:
                same[key] = all(array_id(batch_t[key][k]) == array_id(batch_c[key][k]) for k in batch_t[key])
        need(all(same.values()), "encode_pair_state_prompt_or_mask_differs")
        return enc_t, enc_c, batch_t, {"equal_fields": same}

    def decode(self, raw, batch):
        return np.asarray(self.campaign.decode(np.asarray(raw), np.asarray(batch["state"])))

    # ------------------------------------------------------------- routes
    def canonical(self, encoded, batch, noise, *, label="canonical"):
        self.count_query("native")
        jax, jnp = self.jax, self.jax.numpy

        def run():
            raw = self.campaign.sample(jax.random.key(0), encoded, noise=jnp.asarray(noise), num_steps=10)
            jax.block_until_ready(raw)
            return np.asarray(raw)
        start = time.monotonic()
        raw = self.audit.call(label, run)
        seconds = time.monotonic() - start
        return {"raw32": raw, "adapter7": self.decode(raw, batch), "model_seconds": seconds, "route": "canonical"}

    def record(self, encoded, batch, noise, *, label="record_all"):
        self.count_query("native")
        jax, jnp = self.jax, self.jax.numpy

        def run():
            raw, trace = self.record_jit(encoded, jnp.asarray(noise))
            jax.block_until_ready((raw, trace))
            return np.asarray(raw), jax.tree.map(np.asarray, trace)
        start = time.monotonic()
        raw, trace = self.audit.call(label, run)
        seconds = time.monotonic() - start
        need(int(trace["completed_updates"]) == 10 and np.array_equal(trace["flow_x"][0], noise),
             "record_full_native_flow")
        return {"raw32": raw, "adapter7": self.decode(raw, batch), "trace": trace, "model_seconds": seconds,
                "route": "record_all"}

    def repair(self, enc_t, enc_c, batch, noise, bases_stack, cfg, *, clean_layers, schedule=None,
               teacher=None, label="repair"):
        self.count_query("repair")
        jax, jnp = self.jax, self.jax.numpy
        schedule = np.ones(10, bool) if schedule is None else np.asarray(schedule, bool)
        need(schedule.shape == (10,), "schedule_mask_ten_calls")
        if teacher is None:
            tf, tx, tt = False, np.zeros((10, 1, 10, 32), np.float32), np.zeros((10,), np.float32)
        else:
            tf, tx, tt = True, np.asarray(teacher["x"], np.float32), np.asarray(teacher["t"], np.float32)
            need(tx.shape == (10, 1, 10, 32) and tt.shape == (10,), "teacher_path_ABI")

        def run():
            raw, trace = self.repair_jit(
                enc_t, enc_c, jnp.asarray(noise), jnp.asarray(bases_stack),
                jnp.asarray(cfg["route_code"], jnp.int32), jnp.asarray(cfg["alpha"], jnp.float32),
                jnp.asarray(cfg["arm_index"], jnp.int32), jnp.asarray(cfg["eq_index"], jnp.int32),
                jnp.asarray(self.c_noop, jnp.float32), jnp.asarray(schedule), jnp.asarray(tf),
                jnp.asarray(tx), jnp.asarray(tt), clean_layers=clean_layers)
            jax.block_until_ready((raw, trace))
            return np.asarray(raw), jax.tree.map(np.asarray, trace)
        start = time.monotonic()
        raw, trace = self.audit.call(label, run)
        seconds = time.monotonic() - start
        need(int(trace["completed_updates"]) == 10, "repair_ten_flow_updates")
        need(np.array_equal(trace["xs"][0], noise), "repair_noise_is_first_flow_state")
        for key in ("H_T", "H_C", "H_P"):
            need(trace[key].shape == (10, 1, 10, 1024) and trace[key].dtype.name == "bfloat16",
                 "repair_native_BF16_all10_L9_H")
        proof = {"completed_flow_updates": 10, "edited_flow_calls": [int(i) for i in np.flatnonzero(schedule)]
                 if cfg["route"] not in ("hook_no_write",) else [],
                 "schedule_mask": schedule.tolist(), "teacher_forced": bool(tf), "clean_variant": "A" if clean_layers == 9 else "B",
                 "clean_layers": clean_layers, "route": cfg["route"], "arm": cfg["arm"], "basis": cfg["basis"],
                 "alpha": cfg["alpha"], "eq_names": cfg["eq_names"], "c_noop": self.c_noop,
                 "edited_layer": "L9 (zero-based 8) post-block", "edited_action_positions": 10,
                 **self.precision}
        return {"raw32": raw, "adapter7": self.decode(raw, batch), "trace": trace, "model_seconds": seconds,
                "route": cfg["route"], "generation": proof}

    def compile_all(self, prompt, *, clean_variants=(9, 18)):
        """Compile every graph on a synthetic observation BEFORE any S0 (TECH-CONTRACT-v1 3.5)."""
        dummy = {"agentview_image": np.zeros((256, 256, 3), np.uint8),
                 "robot0_eye_in_hand_image": np.zeros((256, 256, 3), np.uint8),
                 "robot0_eef_pos": np.zeros(3), "robot0_eef_quat": np.asarray([0.0, 0.0, 0.0, 1.0]),
                 "robot0_gripper_qpos": np.zeros(2)}
        noise = np.zeros((1, 10, 32), np.float32)
        enc, batch = self.encode(dummy, prompt)
        bases = np.zeros((5, 1024, 16), np.float32)
        bases[:, :16, :] = np.eye(16, dtype=np.float32)
        out = {}
        start = time.monotonic()
        self.canonical(enc, batch, noise, label="compile_canonical")
        out["canonical_seconds"] = time.monotonic() - start
        start = time.monotonic()
        self.record(enc, batch, noise, label="compile_record")
        out["record_seconds"] = time.monotonic() - start
        for layers in clean_variants:
            start = time.monotonic()
            self.repair(enc, enc, batch, noise, bases, arm_config("hook_no_write"), clean_layers=layers,
                        label="compile_repair_%d" % layers)
            out["repair_%d_seconds" % layers] = time.monotonic() - start
        out["graphs"] = ["canonical", "record_all"] + ["repair_clean_layers_%d" % layers for layers in clean_variants]
        out["synthetic_input"] = True
        out["science_value"] = False
        self.compiled = out
        return out


def write_accounting(trace, call):
    """Actual BF16 write of one call from cast bits (native_adapter.py:222-230 extended per token)."""
    before = trace["H_T"][call].astype(np.float64)
    after = trace["H_P"][call].astype(np.float64)
    delta = after - before
    return {"l2": float(np.linalg.norm(delta)), "l2_per_token": np.linalg.norm(delta[0], axis=1).tolist(),
            "zero_edit": bool(not np.any(delta))}


def require_route_bits(trace, route):
    """Route invariants that must hold bitwise (hook/alpha0 rule-free checks are tests, not here)."""
    if route == "hook_no_write":
        if not np.array_equal(trace["H_P"].view(np.uint16), trace["H_T"].view(np.uint16)):
            raise TechnicalFailure("hook_no_write_changed_H")
    if route == "full":
        active = np.asarray(trace["applied"], bool)
        if not np.array_equal(trace["H_P"][active].view(np.uint16), trace["H_C"][active].view(np.uint16)):
            raise TechnicalFailure("R_full_write_not_H_C_bits")
