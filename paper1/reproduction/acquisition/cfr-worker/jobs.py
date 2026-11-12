"""Stage 1A job definitions J1-J4 (TECH-1A-SIZING-v1.json) and their in-job orchestration.

New code. J1/J2: qualification (T11, T2, T4, T1, T5, T6, T7, T3, T10, T12) for Camera/Sensor.
J3/J4: dose precheck + throughput (T11, T8, T9, T12) using the clean variant chosen by the
J1/J2 T4 readout (a JOB field set by the owner before submission). Technical units are dev
units with their own identity; they never enter the confirmation population.
"""

from __future__ import annotations

import time

from .common import TechnicalFailure, need, safe_error
from .selection import RULES_VERSION, camera_render_sample, core_units, sensor_render_sample
from . import tests_1a as T

JOB_SCHEMA = "cf-repair-1a-job-v1"
JOBS = {
    "J1": {"family": "Camera Viewpoints", "kind": "qualification",
           "tests": ["T11", "T2", "T4", "T1", "T5", "T6", "T7", "T3", "T10", "T12"]},
    "J2": {"family": "Sensor Noise", "kind": "qualification",
           "tests": ["T11", "T2", "T4", "T1", "T5", "T6", "T7", "T3", "T10", "T12"]},
    "J3": {"family": "Camera Viewpoints", "kind": "dose_throughput", "tests": ["T11", "T8", "T9", "T12"]},
    "J4": {"family": "Sensor Noise", "kind": "dose_throughput", "tests": ["T11", "T8", "T9", "T12"]},
}
LAYERS = {"A": 9, "B": 18}


def validate_job(job, *, now):
    need(job.get("schema") == JOB_SCHEMA and job.get("job_id") in JOBS, "cf_repair_1a_job_schema")
    spec = JOBS[job["job_id"]]
    need(job.get("family") == spec["family"] and job.get("tests") == spec["tests"], "job_family_and_tests_frozen")
    need(job.get("role") == "stage_1A_technical_calibration" and job.get("population") == "dev_units_only",
         "stage_1A_technical_role")
    times = (job.get("science_deadline_epoch"), job.get("allocation_deadline_epoch"))
    need(all(type(t) in (int, float) for t in times) and now < times[0] <= times[1] - 600
         and times[1] - now <= 2 * 3600 + 900, "bounded_absolute_deadlines_2h_allocation")
    need(type(job.get("temporary_results_budget_bytes")) is int
         and 64 * 1024 ** 2 <= job["temporary_results_budget_bytes"] <= 2 * 1024 ** 3, "I3_byte_cap")
    if spec["kind"] == "dose_throughput":
        need(job.get("clean_variant") in LAYERS, "J3_J4_need_T4_selected_clean_variant")
    need(job.get("xla_gpu_autotune_level") in (None, 0), "autotune_pin_none_or_zero")
    need(job.get("c_noop", 1.0) == 1.0, "frozen_c_noop_must_be_one")
    if job["job_id"] != "J1":
        need(isinstance(job.get("expected_abi"), dict) and bool(job["expected_abi"]),
             "J2_J3_J4_need_J1_bound_ABI")
    return spec


def run_job(job, rt, out, *, unit_factory, units, t0_codes=None, mujoco=None, jax=None, bridge_inputs=None):
    """Execute the frozen test list; every test result (pass, fail, technical) is retained."""
    spec = JOBS[job["job_id"]]
    default_native = {"J1": 52, "J2": 43, "J3": 44, "J4": 44}[job["job_id"]]
    native_cap = job.get("approved_native_queries_non_episode", default_native)
    max_native = 76 if spec["kind"] == "dose_throughput" else default_native
    need(type(native_cap) is int and default_native <= native_cap <= max_native, "native_cap_within_frozen_support")
    if native_cap > default_native:
        need(job.get("precal_extra_queries_owner_confirmation") is True,
             "PRECAL_additional_32_native_queries_need_owner_confirmation")
    rt.policy.set_query_budget(native=native_cap, repair=60 if spec["kind"] == "qualification" else 80,
                               episode_native=0 if spec["kind"] == "qualification" else 3 * 104,
                               episode_repair=0 if spec["kind"] == "qualification" else 4 * 104)
    deadline = job["science_deadline_epoch"]
    family = spec["family"]
    results, failures, schedules, timing = {}, [], [], {}
    selection = {"rules_version": RULES_VERSION}
    stopped = {"reason": None}

    def record(test_entry):
        out.test(test_entry)
        results[test_entry["id"]] = test_entry["pass"]
        if test_entry["pass"] is False:
            stopped["reason"] = "failed_qualification:" + test_entry["id"]

    def guarded(test_id, name, fn):
        start = time.monotonic()
        if stopped["reason"] is not None:
            return None
        try:
            need(time.time() < deadline - 60, "science_deadline_reserve")
            return fn()
        except TechnicalFailure as error:
            stopped["reason"] = error.code
            failures.append({"test": test_id, **safe_error(error), "detail": error.detail})
            record(T.failed_entry(test_id, name, error))
        except Exception as error:  # noqa: BLE001
            stopped["reason"] = "technical_exception:" + test_id
            failures.append({"test": test_id, **safe_error(error)})
            record(T.failed_entry(test_id, name, error))
        finally:
            timing[test_id] = time.monotonic() - start
        return None

    if "T11" in spec["tests"]:
        guarded("T11", "ABI probe", lambda: record(T.t11_abi(mujoco, jax, expected=job.get("expected_abi"))))
    if stopped["reason"] is not None:
        return {"tests": results, "technical_failures": failures, "selection": selection, "timing": timing,
                "technical_gate_passed": False, "stopped_reason": stopped["reason"]}

    if spec["kind"] == "qualification":
        chosen, selection["core"] = core_units(units, rt.split, family, 2, t0_codes=t0_codes)
        ctxs = []
        for unit in chosen:
            try:
                ctxs.append(unit_factory(unit))
            except Exception as error:  # noqa: BLE001
                failures.append({"test": "unit_setup", "unit": unit["row_id"], **safe_error(error),
                                 "detail": getattr(error, "detail", None)})
                stopped["reason"] = "unit_setup_failed"
                break
        for ctx in ctxs:
            schedules.append(ctx.noise_schedule)
            guarded("T2", "S0 restore equality and replay", lambda c=ctx: record(T.t2_snapshot(c)))
        variant = None
        if ctxs:
            got = guarded("T4", "H_C vs native clean H", lambda: T.t4_hc_parity(ctxs, out))
            if got is not None:
                record(got[0])
                variant = got[1]
        if variant is None:
            failures.append({"test": "T4", "code": "technical_stop_T4_no_qualified_clean_variant"})
        else:
            layers = LAYERS[variant]
            rows = guarded("T1_obs", "qualification observations", lambda: T.qualification_observations(ctxs))
            if rows:
                guarded("T1", "identity routes", lambda: record(T.t1_identity(rows, out, clean_layers=layers)))
                guarded("T5", "R_full write", lambda: record(T.t5_rfull(rows, out, clean_layers=layers)))
                guarded("T6", "projection identities", lambda: record(T.t6_identities(rows, out, clean_layers=layers)))
                guarded("T7", "equal-norm writes", lambda: record(T.t7_equal_norm(rows, out, clean_layers=layers)))
        selection["selected_clean_variant"] = variant
        for ctx in ctxs:
            ctx.close()
        if family == "Camera Viewpoints":
            sample, selection["T3"] = camera_render_sample(units, rt.split, t0_codes=t0_codes)
        else:
            sample, selection["T3"] = sensor_render_sample(units, rt.split, t0_codes=t0_codes)
        guarded("T3", "dual render and RNG stream", lambda: record(
            T.t3_dual_render(rt, sample, out, unit_factory=lambda u: unit_factory(u, with_bases=False, keep_render_data=True,
                                                                record_replay=False))))
        guarded("T10", "cross-cloud L9 bridge", lambda: record(T.t10_bridge(bridge_inputs)))
    else:
        layers = LAYERS[job["clean_variant"]]
        chosen, selection["T8"] = core_units(units, rt.split, family, 4, t0_codes=t0_codes)
        ctxs = []
        for unit in chosen:
            try:
                ctxs.append(unit_factory(unit, record_replay=False))
            except Exception as error:  # noqa: BLE001
                failures.append({"test": "unit_setup", "unit": unit["row_id"], **safe_error(error)})
                stopped["reason"] = "unit_setup_failed"
                break
        schedules += [c.noise_schedule for c in ctxs]
        guarded("T8", "dose precheck", lambda: record(T.t8_dose(ctxs, out, clean_layers=layers)))
        throughput = {}
        if ctxs:
            selection["T9"] = {"rule": "first_T8_unit", "row_id": ctxs[0].unit["row_id"]}
            got = guarded("T9", "throughput", lambda: T.t9_throughput(ctxs[0], out, clean_layers=layers, deadline=deadline))
            if got is not None:
                record(got[0])
                throughput = got[1]
        for ctx in ctxs:
            ctx.close()
        timing["throughput"] = throughput
    guarded("T12", "noise binding and RNG isolation", lambda: record(T.t12_noise_rng(rt.audit, schedules)))
    required = [test for test in spec["tests"] if test not in ("T8", "T10")]
    passed = not failures and stopped["reason"] is None and all(results.get(test) is True for test in required)
    return {"tests": results, "technical_failures": failures, "selection": selection, "timing": timing,
            "query_counts": dict(rt.policy.query_counts), "query_budget": dict(rt.policy.query_budget),
            "technical_gate_passed": passed, "stopped_reason": stopped["reason"]}
