"""Infrastructure termination reasons for rollout truncation."""

TRUNCATION_REASONS = frozenset(
    {
        "step_cap",
        "sc2_crash",
        "process_restart",
        "manual_stop",
        "infra_abort",
    }
)
