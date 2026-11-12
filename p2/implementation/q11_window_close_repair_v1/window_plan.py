"""Role-correct original recorder window. No native execution or state mutation."""
ROLES = ("source_qualification", "N", "N_repeat", "S", "R")


def query_indices(role):
    if role not in ROLES:
        raise ValueError("exact_pilot_role_required")
    # The q12 witness qualifies only the unmodified parent continuation.
    # Cached q11 command branches do not request an original-parent q12 input.
    return list(range(13 if role == "source_qualification" else 12))


def make_plan(root, role, root_id):
    indices = query_indices(role)
    return {"root_trajectory_id": root_id,
            "max_policy_steps": root["official_max_policy_steps"],
            "recorded_query_indices": indices,
            "tiers": {str(q): root["tiers"][str(q)] for q in indices}}


def finish_reason(next_query, planned_queries, terminal):
    if next_query == planned_queries:
        return "window_complete"
    return "root_terminal" if terminal else "root_failure"
