"""Tests: test evaluation pipeline."""

import json
from copy import deepcopy
from dataclasses import replace

import pytest
import torch

from poorstar.evaluation.confidence import (
    ConfidenceWindow,
    build_confidence_target,
    confidence_auxiliary_loss,
    confidence_calibration,
    fit_temperature,
)
from poorstar.evaluation.evaluation import (
    PairedResult,
    dose_and_cross_effects,
    learning_curve,
    paired_effects,
    population_coverage,
    restricted_response_gain,
)
from poorstar.evaluation.selection import SelectionRule, select_checkpoint
from poorstar.evaluation.teachers import build_control_sidecar, verify_annotation_document
from poorstar.training.checkpoint import file_hash
from poorstar.training.config import ControlSidecar
from poorstar.training.contracts import digest


@pytest.fixture(autouse=True)
def threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def window():
    return ConfidenceWindow("episode", 1, 0, 4, "a" * 64)


def event(status, loop=2, revision=1):
    return dict(
        episode="episode",
        parent_revision=revision,
        loop=loop,
        task_id="task",
        status=status,
        evidence_sha256="b" * 64,
    )


def coverage(begin=0, end=4):
    return dict(
        episode="episode",
        parent_revision=1,
        start_loop=begin,
        end_loop=end,
        complete=True,
        evidence_sha256="c" * 64,
    )


def test_confidence_acceptance_and_wrong_parent_do_not_manufacture_success_or_negative():
    target = build_confidence_target(
        window(),
        confirmations=(event("accepted"), event("confirmed", revision=2)),
        cutoff_evidence_sha256="d" * 64,
    )
    assert not target.mask and target.value is None
    negative = build_confidence_target(
        window(), confirmations=(event("accepted"),), coverage=(coverage(),)
    )
    assert negative.mask and negative.value == 0
    positive = build_confidence_target(window(), confirmations=(event("confirmed"),), cutoff_loop=2)
    assert positive.mask and positive.value == 1


def test_confidence_end_boundary_gap_and_future_remain_censored():
    boundary = build_confidence_target(
        window(), confirmations=(event("confirmed", loop=4),), cutoff_evidence_sha256="d" * 64
    )
    assert not boundary.mask
    gap = build_confidence_target(window(), coverage=(coverage(0, 1), coverage(2, 4)))
    assert not gap.mask
    with pytest.raises(ValueError, match="future"):
        build_confidence_target(window(), confirmations=(event("confirmed"),), cutoff_loop=1)


def test_confidence_censored_loss_zero_gradient_and_calibration_denominator():
    positive = build_confidence_target(window(), confirmations=(event("confirmed"),))
    negative = build_confidence_target(window(), coverage=(coverage(),))
    censored = build_confidence_target(window(), cutoff_loop=2, cutoff_evidence_sha256="d" * 64)
    predictions = torch.tensor([0.75, 0.25, 0.5], requires_grad=True)
    loss, denominator = confidence_auxiliary_loss(predictions, (positive, negative, censored))
    loss.backward()
    assert (
        denominator == 2
        and predictions.grad[0] < 0 < predictions.grad[1]
        and predictions.grad[2] == 0
    )
    report = confidence_calibration(
        [0.75, 0.25, 0.5], (positive, negative, censored), bin_edges=(0, 0.5, 1)
    )
    assert report["denominator"] == 2 and report["censored"] == 1 and report["brier"] == 0.0625
    with pytest.raises(ValueError, match="test"):
        fit_temperature([0.75], (positive,), split="test", evidence_sha256="e" * 64)


def paired(treatment="base", seed=1, value=0, status="COMPLETE", checkpoint="a" * 64):
    return PairedResult(
        "E1",
        treatment,
        seed,
        "b" * 64,
        "map",
        "c" * 64,
        checkpoint,
        10,
        "d" * 64,
        "e" * 64,
        {"army": value},
        "f" * 64,
        status=status,
    )


def test_same_checkpoint_causal_pair_and_cross_arm_foundation_are_explicit():
    records = [paired(seed=1), paired("army", 1, 0.2), paired(seed=2), paired("army", 2, 0.4)]
    report = paired_effects(records, baseline="base", treatment="army", metric="army", direction=1)
    assert report["seed_clusters"] == 2 and report["mean_directed_effect"] == pytest.approx(0.3)
    assert not report["hypothesis_proven"]
    changed = [paired(), paired("army", checkpoint="9" * 64)]
    with pytest.raises(ValueError, match="complete"):
        paired_effects(changed, baseline="base", treatment="army", metric="army", direction=1)
    matched = [replace(r, foundation_sha256="8" * 64) for r in changed]
    result = paired_effects(
        matched,
        baseline="base",
        treatment="army",
        metric="army",
        direction=1,
        pairing_mode="TRAINED_ARM_MATCHED_SCENARIO",
    )
    assert result["pairing_mode"] == "TRAINED_ARM_MATCHED_SCENARIO"
    with pytest.raises(ValueError, match="foundation"):
        paired_effects(
            changed,
            baseline="base",
            treatment="army",
            metric="army",
            direction=1,
            pairing_mode="TRAINED_ARM_MATCHED_SCENARIO",
        )


def test_failed_pair_not_dropped_or_used_for_noninferiority_and_dose():
    rows = [paired(), paired("low", value=0.1), paired("high", value=0.2, status="TRUNCATED")]
    result = dose_and_cross_effects(
        rows,
        baseline="base",
        treatments={"low": ("army", 0.25), "high": ("army", 0.75)},
        metric_directions={"army": 1},
    )
    assert result["directed_dose_monotonicity"]["army:army"] is None
    report = paired_effects(
        rows[:2] + [paired(seed=2), paired("low", seed=2, status="ERROR")],
        baseline="base",
        treatment="low",
        metric="army",
        direction=1,
        noninferiority_margin=0.05,
    )
    assert (
        report["status"] == "INCOMPLETE_OR_FAILED_MATRIX" and report["noninferiority_pass"] is None
    )
    assert len(report["all_evidence_sha256"]) == 4


def test_matched_learning_curve_and_registered_population_metrics_keep_limits():
    curve = learning_curve(
        [(0, 0), (20, 0.4), (100, 0.6)], frame_budget=100, attainment_threshold=0.5
    )
    assert curve["first_measured_threshold_frames"] == 100 and not curve["interpolated_attainment"]
    with pytest.raises(ValueError, match="complete"):
        learning_curve([(10, 0.4), (100, 0.6)], frame_budget=100, attainment_threshold=0.5)
    payoff = {"a": {"x": 1.0, "y": -1.0}, "b": {"x": -1.0, "y": 1.0}}
    report = population_coverage(payoff, opponent_ids=("x", "y"), cluster_count=2)
    assert report["occupied_clusters"] == 2 and not report["full_game_exploitability_measured"]
    response = restricted_response_gain(payoff, {"a": 0.5, "b": 0.5}, opponent_ids=("x", "y"))
    assert response["worst_response_payoff"] == 0 and "NOT_SC2" in response["interpretation"]
    with pytest.raises(ValueError, match="complete"):
        population_coverage({"a": {"x": 1}}, opponent_ids=("x", "y"), cluster_count=1)


def candidate(identity, version, metric=1, status="COMPLETE", split="development"):
    report = dict(
        schema="poorstar_v3_selection_candidate_v1",
        identity_sha256=identity.sha256,
        policy_version=version,
        actor_sha256=digest(version),
        evaluation_plan_sha256="a" * 64,
        split=split,
        provenance="evaluator",
        status=status,
        metrics={"score": metric},
        all_registered_jobs_accounted=True,
    )
    report["content_digest"] = digest(report)
    return report


def test_selection_accounts_failures_and_earliest_tie_without_test_tuning():
    from tests.test_training_system_foundation import system

    learner, *_ = system("flat")
    rule = SelectionRule("score", "maximize", (1, 2, 3), "a" * 64, ("score",))
    result = select_checkpoint(
        [
            candidate(learner.identity, 1),
            candidate(learner.identity, 2),
            candidate(learner.identity, 3, status="ERROR"),
        ],
        rule,
        learner.identity,
    )
    assert (
        result["policy_version"] == 1
        and result["failed_policy_versions"] == [3]
        and not result["paper_test_read"]
    )
    with pytest.raises(ValueError, match="accounted"):
        select_checkpoint([candidate(learner.identity, 1)], rule, learner.identity)
    with pytest.raises(ValueError, match="test"):
        select_checkpoint([candidate(learner.identity, 1, split="test")], rule, learner.identity)


def sidecar_inputs(tmp_path):
    from tests.test_coarse_bc import decision_rows
    from tests.test_training_system_foundation import system

    learner, profile, *_ = system("flat")
    rows, *_ = decision_rows()
    annotations = []
    for row in rows:
        dependency = lambda f: [
            dict(
                loop=f.key.loop,
                player_id=f.key.player_id,
                observation_sha256=f.key.observation_sha256,
                evidence_sha256="e" * 64,
                availability="ACTOR_VISIBLE_CURRENT_OR_PAST",
            )
        ]
        annotations.append(
            dict(
                row_id=ControlSidecar.row_id(row),
                label={"supervision_kind": "weak_visible_task_labels"},
                scheduler_evidence=dependency(row.scheduler),
                body_evidence=[] if row.body is None else dependency(row.body),
            )
        )
    evidence = dict(
        schema="poorstar_v3_control_annotation_evidence_v1",
        annotation_policy_sha256="f" * 64,
        human_intent_truth=False,
        future_or_privileged_inputs=False,
        annotation_sha256=digest(annotations),
        initial_control_states_sha256=digest({}),
        row_order=[a["row_id"] for a in annotations],
    )
    path = tmp_path / "annotation_evidence.json"
    path.write_text(json.dumps(evidence))
    return learner, profile, rows, annotations, evidence, path


def test_typed_sidecar_compiles_actual_teacher_and_reader_rejects_label_tamper(tmp_path):
    learner, profile, rows, annotations, evidence, path = sidecar_inputs(tmp_path)
    out = tmp_path / "labels.json"
    result = build_control_sidecar(
        rows,
        annotations,
        initial_control_state=None,
        model=learner.model,
        profile=profile,
        identity=learner.identity,
        evidence_path=path,
        output_path=out,
        annotation_policy_sha256="f" * 64,
    )
    identity = replace(learner.identity, control_sidecar_sha256=result["sha256"])
    reader = ControlSidecar(out, identity, learner.model.hierarchy_config)
    sequence = reader.sequence(rows)
    assert len(sequence.records) == len(rows) and result["optimizer_steps"] == 0
    doc = json.loads(out.read_text())
    doc["records"][annotations[0]["row_id"]]["supervision_id"] = "forged"
    doc.pop("content_digest")
    doc["content_digest"] = digest(doc)
    out.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="no longer match"):
        ControlSidecar(
            out,
            replace(identity, control_sidecar_sha256=file_hash(out)),
            learner.model.hierarchy_config,
        )


@pytest.mark.parametrize("failure", ["future", "order"])
def test_sidecar_future_dependency_and_wrong_order_fail_before_publication(tmp_path, failure):
    learner, profile, rows, annotations, evidence, path = sidecar_inputs(tmp_path)
    if failure == "future":
        annotations[0]["scheduler_evidence"][0]["loop"] = rows[0].scheduler.key.loop + 1
        evidence["annotation_sha256"] = digest(annotations)
    else:
        evidence["row_order"] = list(reversed(evidence["row_order"]))
    path.write_text(json.dumps(evidence))
    out = tmp_path / "labels.json"
    with pytest.raises(ValueError):
        build_control_sidecar(
            rows,
            annotations,
            initial_control_state=None,
            model=learner.model,
            profile=profile,
            identity=learner.identity,
            evidence_path=path,
            output_path=out,
            annotation_policy_sha256="f" * 64,
        )
    assert not out.exists()


def test_registered_evaluator_rejects_missing_seed_cells_and_pooled_comparisons():
    from dataclasses import asdict

    from poorstar.evaluation.runner import evaluate_registered

    analysis = dict(
        analysis_id="E1-army",
        kind="paired",
        parameters=dict(baseline="base", treatment="army", metric="army", direction=1),
        expected_cell_ids=["seed1-base", "seed1-army", "seed2-base", "seed2-army"],
    )
    plan = dict(
        schema="poorstar_v3_research_evaluation_plan_v1",
        purpose="ENGINEERING",
        source_sha256="1" * 64,
        scope_sha256="2" * 64,
        split_sha256="3" * 64,
        metric_program_sha256="4" * 64,
        matched_budget_sha256="5" * 64,
        analyses=[analysis],
    )
    plan["content_digest"] = digest(plan)
    rows = [paired(seed=1), paired("army", 1, 0.2), paired(seed=2), paired("army", 2, 0.2)]
    evidence = dict(
        schema="poorstar_v3_research_measurements_v1",
        purpose="ENGINEERING",
        evaluation_plan_sha256=plan["content_digest"],
        provenance="evaluator",
        analyses={"E1-army": dict(zip(analysis["expected_cell_ids"], map(asdict, rows)))},
    )
    evidence["content_digest"] = digest(evidence)
    report = evaluate_registered(plan, evidence)
    assert report["full_registered_input_panel_accounted"] and not report["paper_hypotheses_proven"]
    evidence.pop("content_digest")
    evidence["analyses"]["E1-army"].pop("seed2-army")
    evidence["content_digest"] = digest(evidence)
    with pytest.raises(ValueError, match="complete frozen"):
        evaluate_registered(plan, evidence)
    with pytest.raises(ValueError, match="one explicit comparison"):
        paired_effects(
            [paired(), replace(paired("army"), comparison="different")],
            baseline="base",
            treatment="army",
            metric="army",
            direction=1,
        )
    with pytest.raises(ValueError, match="two distinct"):
        paired_effects([paired()], baseline="base", treatment="base", metric="army", direction=1)


def test_midgame_intervention_actual_collector_records_identical_visible_prefix():
    from copy import deepcopy

    from poorstar.evaluation.interventions import ScheduledPreferenceInterventions
    from poorstar.hierarchy.interventions import PreferenceIntervention
    from poorstar.training.actor import collect_pair
    from tests.test_training_system_actor import ProtocolPair, policies

    learner, opponent, profile = policies(0)
    vectors = [(0.0,) * 9, (0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0)]
    receipts = []
    for vector in vectors:
        program = ScheduledPreferenceInterventions(
            learner.model.hierarchy_config,
            episode="matched-prefix",
            interventions=(PreferenceIntervention(2, vector, 4),),
        )
        torch.manual_seed(81)
        _, report = collect_pair(
            ProtocolPair(terminal_loop=4),
            (deepcopy(learner.model), deepcopy(opponent)),
            (profile, profile),
            learner.identity,
            maximum_loops=8,
            episode_id="matched-prefix",
            provenance="evaluator",
            external_programs=(program, None),
        )
        assert len(report["registered_interventions"]) == 1
        receipts.append(report["registered_interventions"][0])
        clone = ScheduledPreferenceInterventions(
            learner.model.hierarchy_config,
            episode="matched-prefix",
            interventions=(PreferenceIntervention(2, vector, 4),),
        )
        clone.load_state_dict(program.state_dict())
        assert clone.state_dict() == program.state_dict()
    assert receipts[0]["actual_scheduler_loop"] == receipts[1]["actual_scheduler_loop"] == 2
    assert receipts[0]["visible_input_sha256"] == receipts[1]["visible_input_sha256"]
    assert (
        receipts[0]["pre_intervention_action_prefix_sha256"]
        == receipts[1]["pre_intervention_action_prefix_sha256"]
    )
    assert receipts[0]["intervention"]["preference"] != receipts[1]["intervention"]["preference"]


def test_goal_metrics_do_not_turn_unobserved_tasks_into_failure():
    from poorstar.evaluation.evaluation import task_goal_metrics

    events = [
        dict(task_id="a", loop=2, outcome="confirmed", evidence_id="a" * 64),
        dict(task_id="b", loop=3, outcome="censored", evidence_id="b" * 64),
    ]
    report = task_goal_metrics(events, declared_task_ids=("a", "b", "c"))
    assert report["observed_confirmed_tasks"] == 1 and report["unresolved_tasks"] == ["b", "c"]
    assert report["complete_window_success_rate"] is None and not report["absence_is_failure"]
    with pytest.raises(ValueError, match="duplicate"):
        task_goal_metrics(events + events[:1], declared_task_ids=("a", "b", "c"))
