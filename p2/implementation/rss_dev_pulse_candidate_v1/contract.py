"""A fixed 72-call candidate, with native execution separately unauthorized."""
import hashlib
import json

ROOTS = tuple("SDRU-CDEV-" + task + "-" + root
              for task in ("T00", "T01") for root in ("R00", "R01"))
KEYS = tuple((root, q) for root in ROOTS for q in (11, 22))
PHASES = (
    ("qualification", (
        ("baseline_none_1", "none", "N"),
        ("baseline_none_2", "none", "N"),
        ("V_own_null_pulse", "pulse", "own_null"),
        ("V_own_null_persistent", "persistent", "own_null"))),
    ("effect", (
        ("shared_pulse", "pulse", "S"),
        ("shared_persistent", "persistent", "P"),
        ("random_pulse", "pulse", "R"))),
    ("delayed_repeat", (
        ("shared_pulse_repeat", "pulse", "S"),
        ("shared_persistent_repeat", "persistent", "P"))),
)
CAPS = dict(full_queries=72, prefix_prefills=144, suffix_evaluations=720,
            environment_steps=0, training_updates=0, archive_bytes=256 * 1024**2)


def need(ok, code):
    if not ok:
        raise ValueError(code)


def encoded(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                       allow_nan=False) + "\n").encode()


def identity(value):
    return hashlib.sha256(encoded(value)).hexdigest()


def expected_entries():
    entries = []
    for phase, roles in PHASES:
        for root, query in KEYS:
            for role, schedule, arm in roles:
                active = schedule != "none"
                entries.append(dict(
                    call_id=f"{root}-Q{query:02d}-{role}", phase=phase,
                    root_id=root, query_index=query, role=role, schedule=schedule, arm=arm,
                    receiver="T", donor="C", prefix_prefills=2, suffix_evaluations=10,
                    num_steps=10, full_query=True, kv_mask=[False, active],
                    layer_mask=[i == 8 and active for i in range(18)],
                    flow_mask=[i == 5 if schedule == "pulse" else i >= 5
                               if schedule == "persistent" else False for i in range(10)],
                    use_vmod=active))
    return entries


def validate(plan):
    need(plan["schema"] == "DEV-pulse-native-plan-v1", "exact_DEV_pulse_plan_schema")
    need(plan["scope"] == "historical_DEV8_fixed16_pulse_observability_not_confirmatory",
         "exact_historical_observability_scope")
    need(plan["planned_queries"] == [list(k) for k in KEYS], "exact_eight_DEV_snapshots")
    need(plan["entries"] == expected_entries(), "exact72_calls_order_and_selectors")
    need(plan["caps"] == CAPS, "exact72_call_and_archive_caps")
    need(plan["qualification_barrier"] == "all32initial_checks_before_any_effect",
         "all_snapshot_qualification_barrier")
    need(plan["GPU_execution_authorized"] is False and
         plan["native_call_start_requires_separate_explicit_user_authority"] is True,
         "candidate_is_not_GPU_authority")
    need(plan["R_delayed_repeat"] is False and plan["physical_meaningful_threshold"] is None,
         "no_invented_repeat_or_physical_threshold")
    need(plan["no_automatic_retry"] is True and plan["no_automatic96orB"] is True,
         "bounded_no_retry_or_expansion")
    return KEYS


def require_native_authority(plan, authority):
    validate(plan)
    need(plan.get("source_roster_pending_binding") is False,
         "actual_input_roster_must_be_bound_before_native_execution")
    need(isinstance(authority, dict) and authority.get("GPU_execution_authorized") is True,
         "GPU0_requires_new_explicit_user_authorization")
    need(authority.get("schema") == "DEV-pulse-explicit-user-GPU-authority-v1" and
         authority.get("plan_sha256") == identity(plan) and
         authority.get("maximum_full_generations") == 72 and
         authority.get("sole_creator_thread_id") == "01a152e1-282c-7201-9af3-f8b1ea7444ba" and
         isinstance(authority.get("direct_user_authorization"), str) and
         len(authority["direct_user_authorization"]) > 0,
         "exact_reviewed_native_authority_required")
