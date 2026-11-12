"""Stage 1A test drivers T1-T12 (TECH-CONTRACT-v1 section 7; TECH-1A-SIZING-v1 jobs J1-J4).

New code. Rules that apply everywhere (7.0): exact bytes unless a tolerance was frozen
before the job (only T3(d)'s 1-LSB/0.1% alternative and T6/T7's deterministic bounds are
non-exact, both from the contract); a failed identity test is retained as data and never
repeated; 1A may not analyse repair-arm command recovery or task outcomes, so every edited
plan and every T9 outcome is written under SEALED/ or in a ``sealed`` key.
"""

from __future__ import annotations

import platform
import sys
import time

import numpy as np

from .common import TechnicalFailure, array_id, bf16_bits, difference, jsonable, need, safe_error, within
from .episode import NATIVE_ARMS, REPAIR_ARMS, query_arm, rollout_native, run_episode
from .plus_env import image_stats, render_base_frame, render_base_frame_exact_source
from .policy import arm_config
from .repair_math import equal_norm_verdict, identity_verdict, call_measurements, BASIS_ORDER
from .noise import query_noise
from . import snapshot as S

ARM_ORDER_1A = ("N_C", "N_T", "R_resp", "R_clean", "R_rand", "R_full")
T8_ARMS = ("R_resp", "R_clean", "R_rand", "R_full", "eq_B")
T8_ANCHORS = (0, 2, 5, 10)
T9_STOP_SECONDS_PER_QUERY = 2.0


def entry(test_id, name, *, samples, passed, measured, tolerances=None, inputs=None, notes=None, sealed=None):
    return jsonable({"id": test_id, "name": name, "sample_ids": samples, "pass": passed, "measured": measured,
                     "tolerances": tolerances or {}, "inputs": inputs or {}, "notes": notes or [],
                     "sealed_refs": sealed or []})


def failed_entry(test_id, name, error, samples=None):
    return entry(test_id, name, samples=samples or [], passed=False, measured={"technical_failure": safe_error(error)},
                 notes=["technical failure retained as data; not repeated"])


# ---------------------------------------------------------------------- T11
def t11_abi(mujoco_module, jax_module, *, expected=None):
    import os
    probe = {"python": sys.version.split()[0], "platform": platform.platform()}
    for name in ("numpy", "jax", "flax", "mujoco", "robosuite"):
        try:
            module = __import__(name)
            probe[name + "_version"] = getattr(module, "__version__", None)
        except Exception as error:  # noqa: BLE001
            probe[name + "_version"] = "unavailable:" + type(error).__name__
    probe["mj_getState_available"] = S.integration_mode(mujoco_module) == "mj_getState"
    probe["integration_mode"] = S.integration_mode(mujoco_module)
    probe["jax_default_matmul_precision"] = str(jax_module.config.jax_default_matmul_precision)
    probe["XLA_FLAGS"] = os.environ.get("XLA_FLAGS")
    probe["devices"] = [str(d) for d in jax_module.devices()]
    required = {"mjtObj": hasattr(mujoco_module, "mjtObj"), "mj_camlight": hasattr(mujoco_module, "mj_camlight"),
                "mj_forward": hasattr(mujoco_module, "mj_forward"), "mj_name2id": hasattr(mujoco_module, "mj_name2id")}
    probe["required_mujoco_functions"] = required
    passed = all(required.values()) and probe["jax_default_matmul_precision"] == "highest"
    if expected:
        probe["expected"] = expected
        mismatch = {k: (probe.get(k), v) for k, v in expected.items() if probe.get(k) != v}
        probe["expected_mismatch"] = mismatch
        passed = passed and not mismatch
    return entry("T11", "ABI probe", samples=[], passed=passed, measured=probe,
                 notes=["camera extended-field presence and headlight types are recorded per unit in T3"])


def model_abi(h):
    """Per-unit camera/headlight ABI (part of T11, recorded where a live model exists)."""
    from .counterfactual import EXTENDED_OPTICS, HEADLIGHT
    m = h.sim.model
    return {"extended_optics_present": {name: (str(np.asarray(getattr(m, name)).dtype), list(np.asarray(getattr(m, name)).shape))
                                        for name in EXTENDED_OPTICS if hasattr(m, name)},
            "headlight_types": {name: type(getattr(m.vis.headlight, name)).__name__ for name in HEADLIGHT},
            "cam_fields": {name: str(np.asarray(getattr(m, name)).dtype) for name in ("cam_pos", "cam_quat", "cam_fovy",
                                                                                      "cam_bodyid", "cam_mode", "cam_targetbodyid")
                           if hasattr(m, name)}}


# ---------------------------------------------------------------------- T2
def t2_snapshot(ctx):
    samples = [ctx.unit["row_id"]]
    measured = {"state_mode": ctx.rt.state_mode, "unobserved": ctx.S0["unobserved"], "restores": []}
    ok = True
    for arm in ARM_ORDER_1A:
        try:
            info, check = ctx.restore_C() if arm == "N_C" else ctx.restore_T()
            measured["restores"].append({"arm": arm, "equal": check["equal"], "compared": check["compared"],
                                         "replaced_paths": info["replaced_paths"][:50]})
        except TechnicalFailure as error:
            ok = False
            measured["restores"].append({"arm": arm, "equal": False, "failure": safe_error(error), "detail": error.detail})
    copies = []
    for _ in range(2):
        ctx.restore_T()
        copies.append(ctx.replay_record(ctx.E_T, ctx.h_T))
    def same(a, b):
        return len(a) == len(b) and all(x == y for x, y in zip(a, b, strict=True))
    measured["replay_copies_equal"] = same(copies[0], copies[1])
    measured["replay_equals_original_continuation"] = same(copies[0], ctx.original_replay or [])
    first_diff = next((i for i, (x, y) in enumerate(zip(copies[0], ctx.original_replay or [])) if x != y), None)
    measured["first_differing_step_vs_original"] = first_diff
    if first_diff is not None:
        x, y = copies[0][first_diff], ctx.original_replay[first_diff]
        measured["differing_components"] = sorted(k for k in x if x[k] != y.get(k))
    shared = S.compare(S.fingerprint(ctx.S0), S.fingerprint(ctx.S0_C),
                       exclude_prefixes=S.SHARED_EXCLUDE + S.BASE_EXCLUDE)
    shared_no_rng = S.compare(S.fingerprint(ctx.S0), S.fingerprint(ctx.S0_C),
                              exclude_prefixes=S.SHARED_EXCLUDE + S.BASE_EXCLUDE + ("rng/",))
    measured["E_C_shared_state_equal_including_rng"] = shared
    measured["E_C_shared_state_equal_excluding_rng"] = shared_no_rng
    measured["pair_gate"] = ctx.pair
    ok = ok and measured["replay_copies_equal"] and measured["replay_equals_original_continuation"] and shared["equal"] \
        and ctx.pair["qualified"]
    ctx.restore_T()
    return entry("T2", "S0 restore equality and replay", samples=samples, passed=ok, measured=measured,
                 tolerances={"all": "exact_bytes"},
                 inputs={"replay_steps": len(ctx.original_replay or []), "arm_order": list(ARM_ORDER_1A)},
                 notes=["E_C comparison excludes derived camera fields, observation caches and the variant view "
                        "configuration attributes; RNG equality is reported separately as well"])


# ---------------------------------------------------------------------- T4
def t4_hc_parity(ctxs, out):
    samples, measured = [], {"A": [], "B": []}
    ok = {"A": True, "B": True}
    for ctx in ctxs:
        samples.append(ctx.unit["row_id"])
        ctx.restore_C()
        env = ctx.E_C
        obs = ctx.obs_C0
        for q in range(2):
            noise = ctx.noise(q)
            enc, batch = ctx.rt.policy.encode(obs, ctx.prompt)
            native = ctx.rt.policy.record(enc, batch, noise, label="T4_native_record")
            teacher = {"x": native["trace"]["flow_x"][:10], "t": native["trace"]["flow_time"]}
            for variant, layers in (("A", 9), ("B", 18)):
                res = query_arm(ctx, "hook_no_write", obs, obs, noise, clean_layers=layers, teacher=teacher)
                per_call = []
                for j in range(10):
                    h_c = difference(res["trace"]["H_C"][j], native["trace"]["L9_H"][j])
                    h_t = difference(res["trace"]["H_T"][j], native["trace"]["L9_H"][j])
                    per_call.append({"call": j, "H_C_vs_native_exact": h_c["exact"], "H_C_max_abs": h_c["max_abs"],
                                     "treated_hook_H_vs_native_exact": h_t["exact"],
                                     "t_j_equal": float(res["trace"]["t_in"][j]) == float(native["trace"]["flow_time"][j])})
                exact = all(row["H_C_vs_native_exact"] for row in per_call)
                ok[variant] = ok[variant] and exact
                measured[variant].append({"unit": ctx.unit["row_id"], "query": q, "all_calls_exact": exact, "per_call": per_call})
                out.shard(f"T4-{ctx.unit['row_id']}-q{q}-{variant}", {
                    "native_L9_H": native["trace"]["L9_H"], "teacher_forced_H_C": res["trace"]["H_C"],
                    "teacher_forced_H_T": res["trace"]["H_T"]})
            # advance the N_C path with the record route's own actions (record == canonical by T1)
            for action in native["adapter7"][0, :5]:
                obs, done = env.step(action)
                if done or env.check_success():
                    break
    selected = "A" if ok["A"] else ("B" if ok["B"] else None)
    measured["variant_pass"] = ok
    measured["selected_clean_variant"] = selected
    return entry("T4", "H_C vs native clean H (teacher-forced)", samples=samples, passed=selected is not None,
                 measured=measured, tolerances={"H_C": "exact_bytes_per_call"},
                 notes=["selection A->B is technical and allowed in 1A; both failing is a technical stop"]), selected


# ---------------------------------------------------------------------- T1 / T5 / T6 / T7 shared observations
def qualification_observations(ctxs):
    """q0 and q3 of each unit (N_T canonical rollout to q3; counterfactual at every chunk-final)."""
    rows = []
    for ctx in ctxs:
        rows.append({"ctx": ctx, "query": 0, "obs_t": ctx.obs_T0, "obs_c": ctx.obs_C0, "noise": ctx.noise(0)})
        roll = rollout_native(ctx, 3)
        last = roll[-1]
        need(not last.get("terminal_before_query"), "episode_terminated_before_q3")
        rows.append({"ctx": ctx, "query": 3, "obs_t": last["obs_t"], "obs_c": last["obs_c"], "noise": last["noise"]})
    for row in rows:
        row["sample_id"] = f"{row['ctx'].unit['row_id']}:q{row['query']}"
    return rows


def t1_identity(obs_rows, out, *, clean_layers, repeats=2):
    measured, samples, ok = [], [], True
    for row in obs_rows:
        ctx = row["ctx"]
        samples.append(row["sample_id"])
        runs = {}
        for r in range(repeats):
            enc, batch = ctx.rt.policy.encode(row["obs_t"], ctx.prompt)
            runs[("canonical", r)] = ctx.rt.policy.canonical(enc, batch, row["noise"], label="T1_canonical")
            runs[("record", r)] = ctx.rt.policy.record(enc, batch, row["noise"], label="T1_record")
            runs[("hook", r)] = query_arm(ctx, "hook_no_write", row["obs_t"], row["obs_c"], row["noise"], clean_layers=clean_layers)
            runs[("alpha0", r)] = query_arm(ctx, "alpha0", row["obs_t"], row["obs_c"], row["noise"], clean_layers=clean_layers)
        ref = runs[("canonical", 0)]
        comparisons = {}
        for key, run in runs.items():
            if key == ("canonical", 0):
                continue
            label = f"{key[0]}#{key[1]}_vs_canonical#0"
            comparisons[label] = {"raw32": difference(run["raw32"], ref["raw32"]),
                                  "adapter7": difference(run["adapter7"], ref["adapter7"])}
        rec = runs[("record", 0)]["trace"]
        for name in ("hook", "alpha0"):
            for r in range(repeats):
                tr = runs[(name, r)]["trace"]
                comparisons[f"{name}#{r}_vs_record#0_flow"] = {
                    "x_j": difference(tr["xs"], rec["flow_x"]), "v_j": difference(tr["velocity"], rec["flow_velocity"]),
                    "L9_H_T": difference(tr["H_T"], rec["L9_H"]), "L9_H_after": difference(tr["H_P"], rec["L9_H"]),
                    "L10": difference(tr["L10"], rec["L10_H"]), "L18": difference(tr["L18"], rec["L18_H"])}
        rec1 = runs[("record", 1)]["trace"]
        comparisons["record#1_vs_record#0_flow"] = {
            key: difference(rec1[key], rec[key])
            for key in ("flow_x", "flow_velocity", "flow_time", "L9_H", "L10_H", "L18_H")}
        exact = all(stat["exact"] for comp in comparisons.values() for stat in comp.values())
        ok = ok and exact
        measured.append({"sample": row["sample_id"], "all_exact": exact, "comparisons": comparisons})
        out.shard("T1-" + row["sample_id"].replace(":", "-"), {
            "record_L9_H": rec["L9_H"], "hook_H_T": runs[("hook", 0)]["trace"]["H_T"],
            "hook_H_C": runs[("hook", 0)]["trace"]["H_C"], "alpha0_H_P": runs[("alpha0", 0)]["trace"]["H_P"],
            "record_L10_H": rec["L10_H"], "record_L18_H": rec["L18_H"]})
    return entry("T1", "no hook / hook-no-write / alpha=0 identity", samples=samples, passed=ok,
                 measured=measured, tolerances={"all": "exact_bytes"},
                 inputs={"routes": ["canonical", "record_all", "hook_no_write", "alpha0"], "repeats": repeats,
                         "clean_variant": "A" if clean_layers == 9 else "B"},
                 notes=["signed_zero_only_elements is diagnostic; the verdict is exact bytes"])


def t5_rfull(obs_rows, out, *, clean_layers):
    measured, samples, ok, sealed = [], [], True, []
    for row in obs_rows:
        ctx = row["ctx"]
        samples.append(row["sample_id"])
        res = query_arm(ctx, "R_full", row["obs_t"], row["obs_c"], row["noise"], clean_layers=clean_layers, out=out,
                        log_base={"job_test": "T5", "sample_id": row["sample_id"], "query_index": row["query"]})
        tr = res["trace"]
        per_call = [bool(np.array_equal(bf16_bits(tr["H_P"][j]), bf16_bits(tr["H_C"][j]))) for j in range(10)]
        ok = ok and all(per_call)
        measured.append({"sample": row["sample_id"], "H_prime_equals_H_C_bits_per_call": per_call})
        sealed.append(out.sealed_arrays("T5-plan-" + row["sample_id"].replace(":", "-"),
                                        {"raw32": res["raw32"], "adapter7": res["adapter7"]},
                                        reason="edited plan (R_full); 1A must not analyse command change")["relative"])
        out.shard("T5-" + row["sample_id"].replace(":", "-"), {"H_T": tr["H_T"], "H_C": tr["H_C"], "H_P": tr["H_P"]})
    return entry("T5", "R_full write equals H_C", samples=samples, passed=ok, measured=measured,
                 tolerances={"H_prime_vs_H_C": "exact_bits_all_10_calls"}, sealed=sealed)


def t6_identities(obs_rows, out, *, clean_layers):
    measured, samples, ok, sealed = [], [], True, []
    for row in obs_rows:
        ctx = row["ctx"]
        samples.append(row["sample_id"])
        for basis in BASIS_ORDER:
            cfg = arm_config("R_resp", g_assigned=ctx.rt.g_assigned, basis_override=basis)
            cfg["arm"] = "T6_natural_" + basis
            res = query_arm(ctx, cfg["arm"], row["obs_t"], row["obs_c"], row["noise"], clean_layers=clean_layers,
                            out=out, cfg=cfg, log_base={"job_test": "T6", "sample_id": row["sample_id"],
                                                         "query_index": row["query"]})
            tr = res["trace"]
            verdicts = []
            for j in range(10):
                m = call_measurements(h_t=tr["H_T"][j, 0], h_c=tr["H_C"][j, 0], h_p=tr["H_P"][j, 0], w=tr["w"][j, 0],
                                      declared_noop=tr["declared_noop"][j, 0], route="natural", arm_name=basis,
                                      bases=ctx.bases_by_name, template=ctx.template, gram=ctx.gram, tau=tr["tau"][j, 0])
                verdicts.append(identity_verdict(m))
            passed = all(v["pass"] for v in verdicts)
            ok = ok and passed
            measured.append({"sample": row["sample_id"], "basis": basis, "pass": passed,
                             "gram_spectral_fp32": ctx.gram[basis],
                             "max_ratio_in": max((v["max_ratio_in"] for v in verdicts if v["max_ratio_in"] is not None), default=None),
                             "max_ratio_out": max((v["max_ratio_out"] for v in verdicts if v["max_ratio_out"] is not None), default=None),
                             "failing_tokens": sum(v["failing_tokens"] for v in verdicts),
                             "noop_tokens": int(np.asarray(tr["declared_noop"]).sum()),
                             "noop_bit_failures": sum(v["declared_noop_bit_failures"] for v in verdicts),
                             "cast_failures": sum(v["cast_failures"] for v in verdicts)})
            sealed.append(out.sealed_arrays(f"T6-plan-{row['sample_id'].replace(':', '-')}-{basis}",
                                            {"raw32": res["raw32"], "adapter7": res["adapter7"]},
                                            reason="edited plan (natural alpha=1); 1A must not analyse")["relative"])
    return entry("T6", "projection identities (B, C, G1-G3)", samples=samples, passed=ok, measured=measured,
                 tolerances={"r_in": "u*||H'|| + 2*g_k*||D|| + eps32", "r_out": "u*||H'|| + 2*g_k*||p_k|| + eps32",
                             "u": 2.0 ** -8, "eps32": "1024*2^-24*(||H_T||+||H_C||)"},
                 notes=["passing does not prove the right basis was chosen; that is the identity chain"], sealed=sealed)


def t7_equal_norm(obs_rows, out, *, clean_layers):
    measured, samples, ok, sealed = [], [], True, []
    for row in obs_rows:
        ctx = row["ctx"]
        samples.append(row["sample_id"])
        s_j0 = {}
        for arm in ("eq_B", "eq_C", "eq_G"):
            res = query_arm(ctx, arm, row["obs_t"], row["obs_c"], row["noise"], clean_layers=clean_layers, out=out,
                            log_base={"job_test": "T7", "sample_id": row["sample_id"], "query_index": row["query"]})
            tr = res["trace"]
            verdict = equal_norm_verdict(w=tr["w"][0, 0], s=tr["s"][0, 0], tau=tr["tau"][0, 0],
                                         h_t=tr["H_T"][0, 0], h_c=tr["H_C"][0, 0], h_p=tr["H_P"][0, 0])
            s_j0[arm] = np.asarray(tr["s"][0]).copy()
            ok = ok and verdict["pass"]
            measured.append({"sample": row["sample_id"], "arm": arm, "j0": verdict})
            sealed.append(out.sealed_arrays(f"T7-plan-{row['sample_id'].replace(':', '-')}-{arm}",
                                            {"raw32": res["raw32"], "adapter7": res["adapter7"]},
                                            reason="equal-norm full-generation plan; sealed in 1A")["relative"])
        same_s = all(np.array_equal(s_j0["eq_B"], s_j0[a]) for a in ("eq_C", "eq_G"))
        ok = ok and same_s
        measured.append({"sample": row["sample_id"], "j0_candidate_norm_s_identical_across_arms": same_s})
    return entry("T7", "equal-norm candidate and actual write", samples=samples, passed=ok, measured=measured,
                 tolerances={"candidate": "|‖s p̂‖ − s| <= 1e-6 s + 1e-12", "actual": "‖W + s p̂‖ <= u‖H'‖ + eps32",
                             "excluded": "tokens with s <= tau counted, not ratio-checked"}, sealed=sealed)


# ---------------------------------------------------------------------- T3
def t3_dual_render(rt, units, out, *, unit_factory, later_states=3):
    measured, samples, ok = [], [], True
    for unit in units:
        samples.append(unit["row_id"])
        try:
            ctx = unit_factory(unit)
        except Exception as error:  # noqa: BLE001
            ok = False
            measured.append({"unit": unit["row_id"], "technical_failure": safe_error(error),
                             "detail": getattr(error, "detail", None)})
            continue
        try:
            states = []
            q0 = ctx.cf_receipt0
            states.append(_render_state(ctx, 0, ctx.obs_T0, ctx.obs_C0, q0, out))
            states[-1]["q0_native_base_sensor_vs_clean"] = image_stats(ctx.obs_C0["agentview_image"],
                                                                      ctx.obs_C_own0["agentview_image"])
            ctx.restore_T()
            env, h = ctx.E_T, ctx.h_T
            obs_t, steps = ctx.obs_T0, 10
            for k in range(1, later_states + 1):
                noise = ctx.noise(k - 1)
                enc, batch = rt.policy.encode(obs_t, ctx.prompt)
                native = rt.policy.canonical(enc, batch, noise, label="T3_canonical")
                actions = native["adapter7"][0, :5]
                pre = S.capture(h, mode=rt.state_mode)
                off = _chunk(ctx, env, h, actions, steps, capture=False)
                S.restore(h, pre)
                on = _chunk(ctx, env, h, actions, steps, capture=True)
                steps += len(on["rows"])
                e_check = {"observations_equal": [a["obs"] == b["obs"] for a, b in zip(on["rows"], off["rows"], strict=False)],
                           "rng_equal": [a["rng"] == b["rng"] for a, b in zip(on["rows"], off["rows"], strict=False)],
                           "physics_equal": [a["physics"] == b["physics"] for a, b in zip(on["rows"], off["rows"], strict=False)],
                           "same_length": len(on["rows"]) == len(off["rows"])}
                if on.get("terminal") or on.get("clean") is None:
                    states.append({"state": k, "terminal_before_capture": True, "e_capture_on_off": e_check})
                    break
                obs_t = on["obs"]
                row = _render_state(ctx, k, on["obs"], on["clean"], on["receipt"], out)
                row["e_capture_on_off"] = e_check
                states.append(row)
            unit_ok = all(s.get("pass", False) for s in states if "pass" in s) and \
                all(all(v) if isinstance(v, list) else v for s in states for v in s.get("e_capture_on_off", {}).values())
            ok = ok and unit_ok
            measured.append({"unit": unit["row_id"], "family": ctx.family, "params": ctx.params,
                             "problem_name": getattr(ctx.E_T.env, "problem_name", None),
                             "camera_recoverability": ctx.camera_recoverability, "abi": model_abi(ctx.h_T),
                             "states": states, "unit_pass": unit_ok})
        except Exception as error:  # noqa: BLE001
            ok = False
            measured.append({"unit": unit["row_id"], "technical_failure": safe_error(error),
                             "detail": getattr(error, "detail", None)})
        finally:
            ctx.close()
    return entry("T3", "dual render and RNG-pollution stream", samples=samples, passed=ok, measured=measured,
                 tolerances={"a,b,c,e": "exact_bytes", "d": "exact, or frozen max|du8|<=1 on <=0.1% values (reported separately)",
                             "d_prime": "diagnostic only (exact source data, no forward)"})


def _chunk(ctx, env, h, actions, steps, *, capture):
    rows, obs, clean, receipt, terminal = [], None, None, None, False
    for i, action in enumerate(actions):
        last = i == len(actions) - 1
        if last and capture:
            obs, done = ctx.tapped_step(env, h, action, step_index=steps + i + 1)
            clean, receipt = ctx._last_obs_c, ctx._last_cf
        else:
            obs, done = env.step(action)
        fp = ctx.step_fingerprint(h, obs)
        rows.append({"obs": fp["observation"], "rng": fp["rng"], "physics": fp["physics"]})
        if done or env.check_success():
            terminal = True
            break
    return {"rows": rows, "obs": obs, "clean": clean, "receipt": receipt, "terminal": terminal}


def _render_state(ctx, k, obs_t, obs_c, receipt, out):
    rt = ctx.rt
    checks = dict(receipt.get("checks", {})) if ctx.family == "Camera Viewpoints" else {
        "b_wrist_unchanged_by_transaction": bool(receipt.get("wrist_pre_equals_post")),
        "nonagentview_keys_equal": bool(receipt.get("nonagentview_keys_equal")),
        "a_restored_equals_native": "not_applicable_sensor_tap_never_rerenders",
        "c_physics_controller_rng_unchanged": "measured_by_e_capture_on_off_for_later_states"}
    base = render_base_frame(ctx.E_C, receipt["source_physics"], convention=rt.convention)
    d = image_stats(obs_c["agentview_image"], base)
    d_prime = None
    if receipt.get("render_data"):
        exact_src = render_base_frame_exact_source(ctx.E_C, receipt["render_data"], rt.mujoco, convention=rt.convention)
        d_prime = image_stats(obs_c["agentview_image"], exact_src)
    out.shard(f"T3-{ctx.unit['row_id']}-s{k}", {"treated_agentview": obs_t["agentview_image"],
                                                "clean_agentview": obs_c["agentview_image"], "base_render": base,
                                                "wrist": obs_t["robot0_eye_in_hand_image"]})
    gate = [v for key, v in checks.items() if key.startswith(("a_", "b_", "c_")) and isinstance(v, bool)]
    passed = all(gate) and (d["exact"] or d["within_frozen_1lsb_0p1pct"])
    return {"state": k, "checks": checks, "d_base_render": d, "d_prime_exact_source_diagnostic": d_prime,
            "f_source_equals_query_state": receipt.get("source_equals_query_state"),
            "source_physics_sha256": receipt.get("source_physics_sha256"),
            "query_physics_sha256": receipt.get("query_physics_sha256"), "pass": passed,
            "d_exact": d["exact"]}


# ---------------------------------------------------------------------- T8
def t8_dose(ctxs, out, *, clean_layers):
    measured, samples, sealed, released = [], [], [], []
    for ctx in ctxs:
        samples.append(ctx.unit["row_id"])
        roll = rollout_native(ctx, max(T8_ANCHORS))
        by_query = {row["query"]: row for row in roll if not row.get("terminal_before_query")}
        for q in T8_ANCHORS:
            row = by_query.get(q)
            if row is None:
                measured.append({"unit": ctx.unit["row_id"], "anchor": q, "missing": "episode_terminated_before_anchor"})
                continue
            sid = f"{ctx.unit['row_id']}:q{q}"
            # Unedited commands released for the scale rule (N_T plan from the rollout; a_C on obs_C).
            enc_c, batch_c = ctx.rt.policy.encode(row["obs_c"], ctx.prompt)
            a_c = ctx.rt.policy.canonical(enc_c, batch_c, row["noise"], label="T8_clean_full_generation")
            clean_resample_noise = query_noise(
                tag=ctx.noise_schedule["tag"], family=ctx.family, suite=ctx.case["suite"],
                classification_id=ctx.case["classification_id"], init_index=0, noise_label=1, query=q)
            a_c_prime = ctx.rt.policy.canonical(enc_c, batch_c, clean_resample_noise,
                                               label="T8_clean_resample_full_generation")
            n_t = row.get("native")
            if n_t is None:
                enc_t, batch_t = ctx.rt.policy.encode(row["obs_t"], ctx.prompt)
                n_t = ctx.rt.policy.canonical(enc_t, batch_t, row["noise"], label="T8_N_T_at_anchor")
            released.append(out.shard("T8-unedited-" + sid.replace(":", "-"),
                                      {"N_T_raw32": n_t["raw32"], "N_T_adapter7": n_t["adapter7"],
                                       "N_C_raw32": a_c["raw32"], "N_C_adapter7": a_c["adapter7"],
                                       "N_C_prime_raw32": a_c_prime["raw32"],
                                       "N_C_prime_adapter7": a_c_prime["adapter7"],
                                       "noise_label0": row["noise"], "noise_label1": clean_resample_noise})["relative"])
            summary = {"unit": ctx.unit["row_id"], "anchor": q, "arms": {}}
            for arm in T8_ARMS:
                res = query_arm(ctx, arm, row["obs_t"], row["obs_c"], row["noise"], clean_layers=clean_layers, out=out,
                                log_base={"job_test": "T8", "sample_id": sid, "query_index": q})
                tr = res["trace"]
                summary["arms"][arm] = _dose_summary(ctx, tr, res["cfg"])
                sealed.append(out.sealed_arrays(f"T8-plan-{sid.replace(':', '-')}-{arm}",
                                                {"raw32": res["raw32"], "adapter7": res["adapter7"]},
                                                reason="edited plan at a dose-precheck anchor; sealed in 1A")["relative"])
            measured.append(summary)
    return entry("T8", "dose precheck (hidden-state only)", samples=samples, passed=None, measured=measured,
                 inputs={"anchors": list(T8_ANCHORS), "arms": list(T8_ARMS)},
                 notes=["descriptive, no gate; unedited N_T/N_C commands at anchors are released in SHARDS/T8-unedited-*",
                        "edited plans are sealed"], sealed=sealed + [{"released_unedited": released}])


def _dose_summary(ctx, tr, cfg):
    rows = []
    for j in range(10):
        m = call_measurements(h_t=tr["H_T"][j, 0], h_c=tr["H_C"][j, 0], h_p=tr["H_P"][j, 0], w=tr["w"][j, 0],
                              declared_noop=tr["declared_noop"][j, 0], route=cfg["route"], arm_name=cfg["basis"],
                              bases=ctx.bases_by_name, template=ctx.template, gram=ctx.gram,
                              s=tr["s"][j, 0], argmin=tr["argmin"][j, 0], eq_names=cfg["eq_names"], tau=tr["tau"][j, 0])
        rows.append({"call": j, "W_actual_norm_median": float(np.median(m["W_actual_norm"])),
                     "declared_noop_tokens": int(np.sum(m["declared_noop"])),
                     "p_norm_median": {b: float(np.median(m["bases"][b]["p_norm"])) for b in m["bases"]},
                     "pM_norm_median": {b: float(np.median(m["bases"][b]["pM_norm"])) for b in m["bases"]},
                     "pR_norm_median": {b: float(np.median(m["bases"][b]["pR_norm"])) for b in m["bases"]},
                     "energy_share_median": {b: float(np.nanmedian(m["bases"][b]["energy_share"])) for b in m["bases"]},
                     "equal_norm": ({"s_median": float(np.median(m["equal_norm"]["s"])),
                                     "f_median": float(np.nanmedian(m["equal_norm"]["f"])),
                                     "argmin_counts": {n: m["equal_norm"]["argmin"].count(n) for n in set(m["equal_norm"]["argmin"])}}
                                    if "equal_norm" in m else None)})
    return rows


# ---------------------------------------------------------------------- T9
def t9_throughput(ctx, out, *, clean_layers, deadline):
    samples = [ctx.unit["row_id"]]
    episodes, throughput, stopped = [], {}, None
    order = list(ARM_ORDER_1A) + ["N_T_repeat"]
    for index, arm in enumerate(order):
        if time.time() > deadline - 120:
            stopped = "deadline_reserve"
            break
        base_arm = "N_T" if arm == "N_T_repeat" else arm
        ctx.rt.policy.episode_mode = True
        try:
            record = run_episode(ctx, base_arm, clean_layers=clean_layers, out=out, sample_id=f"{ctx.unit['row_id']}:{arm}",
                                 test_id="T9", deadline=deadline)
        finally:
            ctx.rt.policy.episode_mode = False
        record["arm_label"] = arm
        out.episode(f"T9-{ctx.unit['row_id']}-{arm}", record)
        per_query = [q["timing"] for q in record["queries"]]
        total = [sum(v for k, v in t.items() if k.endswith("_s")) for t in per_query]
        throughput[arm] = {"queries": len(per_query), "episode_length_policy_steps": record["episode_length_policy_steps"],
                           "mean_seconds_per_query": float(np.mean(total)) if total else None,
                           "components_mean": {k: float(np.mean([t.get(k, 0.0) for t in per_query]))
                                               for k in ("encode_s", "model_s", "log_s", "env_steps_s", "counterfactual_capture_s")}
                           if per_query else {}, "execution_status": record["execution_status"]}
        episodes.append({"arm": arm, "status": record["execution_status"],
                         "technical_missing_reason": record["technical_missing_reason"]})
        if index == 0 and total and float(np.mean(total)) > T9_STOP_SECONDS_PER_QUERY:
            stopped = "first_episode_exceeded_2s_per_query"
            break
    passed = stopped is None and all(e["status"] == "complete" for e in episodes)
    return entry("T9", "throughput (outcomes sealed)", samples=samples, passed=passed,
                 measured={"episodes": episodes, "throughput": throughput, "stopped": stopped},
                 notes=["episode outcomes and issued commands are in EPISODES/*.json under the sealed key",
                        "in-graph prefill/suffix/clean-pass split is not separable inside one compiled query graph; "
                        "model_s is the whole repair or canonical query graph"]), throughput


# ---------------------------------------------------------------------- T10 / T12
def t10_bridge(bridge_inputs):
    if not bridge_inputs:
        return entry("T10", "cross-cloud L9 bridge (descriptive)", samples=[], passed=None,
                     measured={"status": "not_run_inputs_absent"},
                     notes=["no historical Baidu observation with stored L9 H bits was supplied in the source artifact"])
    raise TechnicalFailure("T10_inputs_present_but_driver_requires_schema_binding")


def t12_noise_rng(audit, schedules):
    summary = audit.summary()
    bound = all(s.get("noise_sha256_by_query") for s in schedules)
    return entry("T12", "noise binding and RNG isolation", samples=[], passed=summary["pass"] and bound,
                 measured={"rng_audit": summary, "noise_schedules_bound_before_policy": bound,
                           "schedules": [{k: s[k] for k in ("family", "suite", "classification_id", "noise_label",
                                                            "horizon_queries")} for s in schedules]},
                 tolerances={"rng": "exact"})
