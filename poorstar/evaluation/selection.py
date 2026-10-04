"""Evaluation: selection."""

import math
from dataclasses import asdict, dataclass

from poorstar.training.contracts import digest, is_sha


@dataclass(frozen=True)
class SelectionRule:
    metric: str
    direction: str
    candidate_policy_versions: tuple[int, ...]
    expected_evaluation_plan_sha256: str
    required_metrics: tuple[str, ...]
    purpose: str = "ENGINEERING"

    def __post_init__(self):
        if (
            not self.metric
            or self.direction not in ("maximize", "minimize")
            or self.purpose not in ("ENGINEERING", "FORMAL")
            or not is_sha(self.expected_evaluation_plan_sha256)
            or not self.candidate_policy_versions
            or tuple(sorted(set(self.candidate_policy_versions))) != self.candidate_policy_versions
            or any(type(v) is not int or v < 0 for v in self.candidate_policy_versions)
            or self.metric not in self.required_metrics
            or len(set(self.required_metrics)) != len(self.required_metrics)
        ):
            raise ValueError("complete prospective selection rule required")

    @property
    def sha256(self):
        return digest(asdict(self))


def select_checkpoint(reports, rule, identity):
    if identity.purpose != rule.purpose:
        raise ValueError("selection purpose drift")
    by_version = {}
    history = []
    expected = {
        "schema",
        "identity_sha256",
        "policy_version",
        "actor_sha256",
        "evaluation_plan_sha256",
        "split",
        "provenance",
        "status",
        "metrics",
        "all_registered_jobs_accounted",
        "content_digest",
    }
    for report in reports:
        payload = dict(report)
        content = payload.pop("content_digest", None)
        if (
            set(report) != expected
            or digest(payload) != content
            or report["schema"] != "poorstar_v3_selection_candidate_v1"
            or report["identity_sha256"] != identity.sha256
            or report["split"] != "development"
            or report["provenance"] != "evaluator"
            or not is_sha(report["actor_sha256"])
            or report["evaluation_plan_sha256"] != rule.expected_evaluation_plan_sha256
            or type(report["all_registered_jobs_accounted"]) is not bool
            or not report["all_registered_jobs_accounted"]
        ):
            raise ValueError("test/partial/unbound evaluation cannot select a checkpoint")
        version = report["policy_version"]
        if (
            type(version) is not int
            or version not in rule.candidate_policy_versions
            or version in by_version
        ):
            raise ValueError("unexpected or duplicate checkpoint candidate")
        if report["status"] not in ("COMPLETE", "ERROR", "INVALID"):
            raise ValueError("all selected and failed candidate statuses required")
        if report["status"] == "COMPLETE":
            if not set(rule.required_metrics) <= set(report["metrics"]) or any(
                type(report["metrics"][k]) not in (int, float)
                or not math.isfinite(report["metrics"][k])
                for k in rule.required_metrics
            ):
                raise ValueError("required development metrics unavailable")
        by_version[version] = report
        history.append(content)
    if set(by_version) != set(rule.candidate_policy_versions):
        raise ValueError("all preregistered checkpoints, including failures, must be accounted")
    candidates = [r for r in by_version.values() if r["status"] == "COMPLETE"]
    if not candidates:
        raise ValueError("no valid checkpoint candidate")
    sign = -1 if rule.direction == "maximize" else 1
    chosen = min(candidates, key=lambda r: (sign * r["metrics"][rule.metric], r["policy_version"]))
    result = dict(
        schema="poorstar_v3_checkpoint_selection_v1",
        identity_sha256=identity.sha256,
        rule_sha256=rule.sha256,
        selected_actor_sha256=chosen["actor_sha256"],
        policy_version=chosen["policy_version"],
        selected_report_sha256=chosen["content_digest"],
        candidate_report_content_digests=sorted(history),
        failed_policy_versions=sorted(
            r["policy_version"] for r in by_version.values() if r["status"] != "COMPLETE"
        ),
        tie_rule="EARLIEST_POLICY_VERSION",
        paper_test_read=False,
    )
    result["content_digest"] = digest(result)
    return result
