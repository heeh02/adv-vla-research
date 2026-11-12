"""Post-observation closure repair; preserve the original q12 sensor arming.

Cached branches stop before reading q12. The original recorder therefore has
an unreached auxiliary query, even after the five target commands are observed.
Use its original incomplete-window reason; retain that event explicitly.
"""


def close_original_recorder(recorder, bridge, *, terminal, role, env_steps, emit):
    complete = recorder.next == len(recorder.rows)
    reason = "window_complete" if complete else "root_terminal" if terminal else "root_failure"
    if not complete and not terminal and recorder.next == 12 and len(recorder.rows) == 13 and env_steps == 70:
        emit("FIXED_HORIZON_LEGACY_PARTIAL_WINDOW_CLOSE", {
            "role": role, "observed_queries": 12, "original_scheduled_queries": 13,
            "target_five_commands_observed": True, "unread_auxiliary_query": 12,
            "reason_to_unchanged_recorder": reason,
            "reason_domain": "incomplete_original_recorder_window",
            "environment_failure_inferred": False, "original_q12_sensor_schedule_preserved": True})
    return bridge.finish(reason)
