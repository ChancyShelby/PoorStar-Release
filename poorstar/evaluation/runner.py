"""Evaluation: runner."""

from poorstar.evaluation.confidence import ConfidenceTarget, confidence_calibration
from poorstar.evaluation.evaluation import (
    PairedResult,
    dose_and_cross_effects,
    intent_dynamics,
    learning_curve,
    paired_effects,
    population_coverage,
    restricted_response_gain,
    task_goal_metrics,
)
from poorstar.training.contracts import digest, is_sha


def evaluate_registered(plan, evidence):
    if (
        plan.get("schema") != "poorstar_v3_research_evaluation_plan_v1"
        or plan.get("purpose") not in ("ENGINEERING", "FORMAL")
        or not all(
            is_sha(plan.get(n))
            for n in (
                "source_sha256",
                "scope_sha256",
                "split_sha256",
                "metric_program_sha256",
                "matched_budget_sha256",
            )
        )
        or not isinstance(plan.get("analyses"), list)
        or not plan["analyses"]
    ):
        raise ValueError("complete source/data/metric/budget-bound evaluator plan required")
    body = dict(plan)
    expected = body.pop("content_digest", None)
    if digest(body) != expected:
        raise ValueError("frozen evaluator plan digest drift")
    if (
        evidence.get("schema") != "poorstar_v3_research_measurements_v1"
        or evidence.get("evaluation_plan_sha256") != expected
        or evidence.get("provenance") != "evaluator"
        or evidence.get("purpose") != plan["purpose"]
    ):
        raise ValueError("independent measured evaluator evidence must match the frozen plan")
    payload = dict(evidence)
    content = payload.pop("content_digest", None)
    if digest(payload) != content:
        raise ValueError("measurement content drift")
    expected_ids = [a["analysis_id"] for a in plan["analyses"]]
    if len(set(expected_ids)) != len(expected_ids) or set(evidence["analyses"]) != set(
        expected_ids
    ):
        raise ValueError("all frozen E1-E6 analyses, including failures, required")
    outputs = {}
    for analysis in plan["analyses"]:
        if set(analysis) != {"analysis_id", "kind", "parameters", "expected_cell_ids"}:
            raise ValueError("exact registered analysis contract required")
        values = evidence["analyses"][analysis["analysis_id"]]
        cells = analysis["expected_cell_ids"]
        if not cells or len(set(cells)) != len(cells) or set(values) != set(cells):
            raise ValueError("complete frozen seed/opponent/map/cell matrix required")
        parameters = analysis["parameters"]
        kind = analysis["kind"]
        rows = [values[k] for k in cells]
        if kind in ("paired", "dose"):
            records = [PairedResult(**v) for v in rows]
            outputs[analysis["analysis_id"]] = (
                paired_effects(records, **parameters)
                if kind == "paired"
                else dose_and_cross_effects(records, **parameters)
            )
        elif kind == "intent_dynamics":
            outputs[analysis["analysis_id"]] = intent_dynamics(rows, **parameters)
        elif kind == "task_goals":
            outputs[analysis["analysis_id"]] = task_goal_metrics(rows, **parameters)
        elif kind == "confidence":
            targets = [
                ConfidenceTarget(
                    **dict(v["target"], evidence_sha256=tuple(v["target"]["evidence_sha256"]))
                )
                for v in rows
            ]
            outputs[analysis["analysis_id"]] = confidence_calibration(
                [v["prediction"] for v in rows], targets, **parameters
            )
        elif kind == "learning_curve":
            outputs[analysis["analysis_id"]] = learning_curve(
                [tuple(v) for v in rows], **parameters
            )
        elif kind in ("population_coverage", "restricted_response_gain"):
            payoff = {k: v for k, v in zip(cells, rows)}
            outputs[analysis["analysis_id"]] = (
                population_coverage(payoff, **parameters)
                if kind == "population_coverage"
                else restricted_response_gain(payoff, **parameters)
            )
        else:
            raise ValueError("unsupported unregistered paper analysis")
    result = dict(
        schema="poorstar_v3_registered_research_report_v1",
        purpose=plan["purpose"],
        evaluation_plan_sha256=expected,
        measurements_content_digest=content,
        analyses=outputs,
        full_registered_input_panel_accounted=True,
        optimizer_started=False,
        engine_qualified_by_this_report=False,
        paper_hypotheses_proven=False,
    )
    result["content_digest"] = digest(result)
    return result
