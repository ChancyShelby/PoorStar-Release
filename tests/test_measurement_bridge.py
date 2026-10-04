"""Tests: test measurement bridge."""

import json
import subprocess
import sys
from copy import copy, deepcopy
from dataclasses import asdict, replace
from pathlib import Path

import pytest
import torch

from poorstar.common.integrity import source_integrity
from poorstar.evaluation.measurement_bridge import assemble_measurements, scenario_sha256
from poorstar.evaluation.runner import evaluate_registered
from poorstar.league.ownership import write_document
from poorstar.league.registry import SnapshotRegistry
from poorstar.training.checkpoint import file_hash
from poorstar.training.contracts import digest
from poorstar.training.export import export_actor, load_actor
from poorstar.training.job_worker import actor_parameter_counts
from poorstar.training.stage_actor import FIXTURE_SCOPE, StageActorPlan, execute_stage_actor
from tests.test_training_system_actor import ProtocolPair, policies

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


class OwnedProtocolPair(ProtocolPair):
    qualification_scope = FIXTURE_SCOPE
    owned_process_results = []
    closed = False

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        self.closed = True


class GoalProtocolPair(OwnedProtocolPair):
    def __init__(self, terminal_loop):
        super().__init__(terminal_loop)
        from tests.test_hierarchy_goals import visible_frame

        _, catalog = visible_frame()
        self.catalogs = (catalog, deepcopy(catalog))
        from s2clientprotocol import query_pb2
        from s2clientprotocol import sc2api_pb2 as sc

        from tests.test_training_system_actor import ProtocolController

        class GoalController(ProtocolController):
            def observe(self):
                frame, _ = visible_frame(
                    self.pair.loop, tags=(10, 20) if self.pair.loop == 0 else (10, 20, 30)
                )
                raw = sc.ResponseObservation()
                raw.observation.ParseFromString(frame._observation_wire)
                raw.observation.player_common.player_id = self.player
                if self.pair.loop >= self.pair.terminal_loop:
                    raw.player_result.add(
                        player_id=self.player, result=sc.Victory if self.player == 1 else sc.Defeat
                    )
                return raw

            def query(self, request):
                frame, _ = visible_frame(
                    self.pair.loop, tags=(10, 20) if self.pair.loop == 0 else (10, 20, 30)
                )
                assert tuple(a.unit_tag for a in request.abilities) == frame.slot_tags
                response = query_pb2.ResponseQuery()
                response.ParseFromString(frame._query_response_wire)
                return response

        self.controllers = tuple(GoalController(self, pid) for pid in (1, 2))


def experiment(tmp_path, *, paired=True, truncated=False, goal_mode=False):
    metric = ROOT / "poorstar/evaluation/metrics.py"
    metric_sha = file_hash(metric)
    manifest = tmp_path / "source.json"
    manifest.write_text(
        json.dumps(
            dict(
                files=[
                    dict(
                        path=str(metric.relative_to(ROOT)),
                        sha256=metric_sha,
                        bytes=metric.stat().st_size,
                    )
                ]
            )
        )
    )
    learner, opponent, profile = policies(0)
    if goal_mode:
        from poorstar.training.config import model_config_sha256
        from tests.test_hierarchy_goals import grounded_model

        learner.model, _, profile = grounded_model(arm="persistent_intent")
        learner.identity = replace(
            learner.identity,
            model_config_sha256=model_config_sha256(
                learner.model.config, learner.model.hierarchy_config
            ),
            profile_sha256=profile.sha256,
        )
        with torch.no_grad():
            learner.model.act_head.bias.copy_(torch.tensor([30.0, -30.0]))
            learner.model.delay_head.bias.fill_(-30.0)
            learner.model.delay_head.bias[0] = 30.0
            learner.model.task_operation_head.bias.copy_(torch.tensor([-30.0, 30.0, -30.0]))
            learner.model.task_goal_kind_head.bias.fill_(-30.0)
            learner.model.task_goal_kind_head.bias[0] = 30.0
            learner.model.task_goal_unit_head.bias.fill_(-30.0)
            learner.model.task_goal_unit_head.bias[learner.model.config.unit_types.index(100)] = (
                30.0
            )
            learner.model.task_goal_quantity_head.bias.fill_(-30.0)
            learner.model.task_goal_quantity_head.bias[0] = 30.0
        opponent = deepcopy(learner.model)
    # A declared probability fixture keeps every eligible operation represented;
    # it supplies no action script or claim that Preference has learned meaning.
    with torch.no_grad():
        learner.model.manager_operation_head.bias.copy_(torch.tensor([-30.0, 30.0, -30.0]))
    learner.identity = replace(
        learner.identity, source_sha256=file_hash(manifest), initializer_sha256="e" * 64
    )
    registry = SnapshotRegistry(tmp_path / "registry")
    export_actor(tmp_path / "learner.pt", learner)
    learner_snapshot = registry.register(
        tmp_path / "learner.pt", identity=learner.identity, policy_version=0, role="main"
    )
    other = copy(learner)
    other.model = opponent
    export_actor(tmp_path / "opponent.pt", other)
    opponent_snapshot = registry.register(
        tmp_path / "opponent.pt", identity=learner.identity, policy_version=0, role="anchor"
    )
    counts = actor_parameter_counts(learner.model, learner.phase)
    budget = dict(
        counts,
        maximum_parameters=counts["parameter_count"],
        maximum_trainable_parameters=counts["trainable_parameter_count"],
    )
    bindings = {}
    reports = {}
    for index in range(2 if paired else 1):
        execution = dict(
            unroll_steps=4,
            maximum_resident_steps=20,
            maximum_sources=63,
            burn_in_steps=0,
            deterministic_learner=goal_mode,
            deterministic_opponent=True,
            measurement_period=1,
            device="cpu",
            version="protocol-fixture",
            sc2_path="/unused-protocol-fixture",
            source_manifest=str(manifest),
            metric_program_sha256=metric_sha,
        )
        if paired:
            execution["episode_id"] = "registered-common-causal-fixture"
            preference = [0.0] * 9
            preference[3] = float(index)
            execution["preference_interventions"] = {
                "learner": [
                    dict(loop=0, preference=preference, commitment_horizon=4, confidence=0.5)
                ]
            }
        plan = StageActorPlan(
            asdict(learner.identity),
            learner.phase,
            "evaluator",
            digest(["collection", index]),
            learner_snapshot["snapshot_id"],
            opponent_snapshot["snapshot_id"],
            "ProtocolMap",
            11,
            4,
            60000,
            budget,
            execution,
        )
        output = tmp_path / f"run{index}"
        backend = (
            GoalProtocolPair(terminal_loop=3)
            if goal_mode
            else OwnedProtocolPair(terminal_loop=None if truncated else 3)
        )
        report = execute_stage_actor(plan, registry, output, protocol_fixture=backend)
        assert report["status"] == ("TRUNCATED" if truncated else "COMPLETE"), report.get("error")
        bindings[str(index)] = dict(
            stage_report=dict(
                path=str(output / "stage_report.json"),
                sha256=file_hash(output / "stage_report.json"),
            ),
            stage_plan=dict(
                path=str(output / "stage_plan.json"), sha256=file_hash(output / "stage_plan.json")
            ),
            identity=asdict(learner.identity),
            stage=learner.phase,
            binding=report["binding"],
        )
        reports[str(index)] = json.loads((output / "stage_report.json").read_text())
    plan = dict(
        schema="poorstar_v3_research_evaluation_plan_v1",
        purpose="ENGINEERING",
        source_sha256=file_hash(manifest),
        scope_sha256=learner.identity.scope_sha256,
        split_sha256=learner.identity.split_sha256,
        metric_program_sha256=metric_sha,
        matched_budget_sha256="f" * 64,
        analyses=[],
    )
    cells = {}
    if paired:
        analysis = dict(
            analysis_id="E1",
            kind="paired",
            parameters=dict(baseline="base", treatment="army", metric="resources", direction=1),
            expected_cell_ids=["base", "army"],
        )
        for index, treatment in enumerate(("base", "army")):
            r = reports[str(index)]
            actors = r["actors"]
            scenario = scenario_sha256(dict(report=r, actors=actors))
            cells[treatment] = dict(
                kind="paired",
                run_id=str(index),
                comparison="E1",
                treatment=treatment,
                scenario_sha256=scenario,
                anchor=dict(kind="intervention", index=0),
                foundation_sha256=None,
                metrics={"resources": dict(kind="visible_scalar", name="minerals", loop=1)},
            )
    else:
        metrics = json.loads(
            Path(reports["0"]["artifacts"]["evaluator_metrics"]["path"]).read_text()
        )
        predictions = metrics["confidence_predictions"]
        assert predictions
        analysis = dict(
            analysis_id="E3-confidence",
            kind="confidence",
            parameters=dict(bin_edges=[0, 0.5, 1]),
            expected_cell_ids=[f"prediction{i}" for i in range(len(predictions))],
        )
        cells = {
            f"prediction{i}": dict(
                kind="confidence", run_id="0", index=i, window_sha256=p["window_sha256"]
            )
            for i, p in enumerate(predictions)
        }
    plan["analyses"] = [analysis]
    plan_path = tmp_path / "plan.json"
    write_document(plan_path, plan, immutable=True)
    plan = json.loads(plan_path.read_text())
    mapping = dict(
        schema="poorstar_v3_measurement_bridge_mapping_v1",
        purpose=plan["purpose"],
        evaluation_plan_sha256=plan["content_digest"],
        evaluation_plan_file_sha256=file_hash(plan_path),
        source=dict(
            code_root=str(ROOT), manifest=dict(path=str(manifest), sha256=file_hash(manifest))
        ),
        metric_program=dict(path=str(metric), sha256=metric_sha),
        matched_budget_sha256=plan["matched_budget_sha256"],
        runs=bindings,
        cells={analysis["analysis_id"]: cells},
    )
    mapping_path = tmp_path / "mapping.json"
    write_document(mapping_path, mapping, immutable=True)
    return plan_path, mapping_path, reports


def rewrite(path, value):
    value = dict(value)
    value.pop("content_digest", None)
    write_document(path, value)


def change_analysis(plan_path, mapping_path, analysis, cells):
    plan = json.loads(plan_path.read_text())
    plan["analyses"] = [analysis]
    rewrite(plan_path, plan)
    plan = json.loads(plan_path.read_text())
    mapping = json.loads(mapping_path.read_text())
    mapping["evaluation_plan_sha256"] = plan["content_digest"]
    mapping["evaluation_plan_file_sha256"] = file_hash(plan_path)
    mapping["cells"] = {analysis["analysis_id"]: cells}
    rewrite(mapping_path, mapping)


def test_actual_producer_artifact_matrix_assembles_and_existing_analyzer_consumes(tmp_path):
    plan, mapping, reports = experiment(tmp_path)
    result = assemble_measurements(plan, mapping)
    assert result["assembly_status"] == "ASSEMBLED", (
        result["failed_runs"] or result["blocked_cells"]
    )
    analyzed = evaluate_registered(json.loads(plan.read_text()), result)
    assert analyzed["analyses"]["E1"]["pairs"] == 1
    assert (
        not result["causal_pairing_inferred_from_seed"]
        and not result["fixed_clock_samples_are_continuous_truth"]
    )
    assert not analyzed["paper_hypotheses_proven"]


def test_actual_prechoice_confidence_ledger_preserves_censored_denominator(tmp_path):
    plan, mapping, _ = experiment(tmp_path, paired=False)
    result = assemble_measurements(plan, mapping)
    assert result["assembly_status"] == "ASSEMBLED", (
        result["failed_runs"] or result["blocked_cells"]
    )
    calibration = evaluate_registered(json.loads(plan.read_text()), result)["analyses"][
        "E3-confidence"
    ]
    assert calibration["status"] == "NO_ELIGIBLE_TARGETS" and calibration["denominator"] == 0
    assert calibration["censored"] > 0 and calibration["brier"] is None


@pytest.mark.parametrize(
    "problem",
    [None, "missing", "file_drift", "stage_drift", "execution_drift", "gate_evidence_drift"],
)
def test_formal_bridge_rereads_exact_existing_phase_gate_proof_without_launching(tmp_path, problem):
    from poorstar.evaluation.measurement_bridge import _validate_launch_proof
    from poorstar.training import PPOConfig
    from poorstar.training.contracts import TrainingIdentity
    from poorstar.training.gates import REQUIRED
    from tests.test_training_system import identity

    # Synthetic protocol authorization documents exercise readback only. No
    # exporter, engine, optimizer, or actual capability evidence is produced.
    model_identity = identity(PPOConfig(), purpose="FORMAL")
    assert isinstance(model_identity, TrainingIdentity)
    stage = "BC"
    execution = "1" * 64
    evidence = {}
    for name in REQUIRED[stage]:
        path = tmp_path / (name + ".json")
        write_document(
            path,
            dict(
                status="PASS",
                gate=name,
                binding=asdict(model_identity),
                qualification_scope="SYNTHETIC_NO_LAUNCH_GATE_READBACK",
            ),
        )
        evidence[name] = dict(
            path=str(path), sha256=file_hash(path), binding=asdict(model_identity)
        )
    path = tmp_path / "preflight.json"
    write_document(
        path,
        dict(
            schema="poorstar_v3_training_launch_preflight_v1",
            identity_sha256=model_identity.sha256,
            stage=stage,
            status="READY",
            execution_config_sha256=execution,
            gates={name: "PASS" for name in REQUIRED[stage]},
            evidence=evidence,
        ),
    )
    proof = dict(
        path=str(path),
        sha256=file_hash(path),
        content_digest=json.loads(path.read_text())["content_digest"],
        stage=stage,
    )
    report = dict(stage=stage, launch_preflight=proof)
    if problem == "missing":
        report["launch_preflight"] = None
    elif problem == "file_drift":
        path.write_text("{}")
    elif problem == "stage_drift":
        proof["stage"] = "VALUE_ONLY"
    elif problem == "execution_drift":
        execution = "2" * 64
    elif problem == "gate_evidence_drift":
        Path(next(iter(evidence.values()))["path"]).write_text("{}")
    if problem is None:
        _validate_launch_proof(report, model_identity, execution)
    else:
        with pytest.raises(ValueError):
            _validate_launch_proof(report, model_identity, execution)


def test_actual_control_ledger_complete_dynamics_and_missing_event_block(tmp_path):
    plan, mapping, reports = experiment(tmp_path, paired=False)
    actual = json.loads(Path(reports["0"]["artifacts"]["evaluator_metrics"]["path"]).read_text())
    controls = [
        r for r in actual["controls"] if r["phase"] == "scheduler" and r["operation"] is not None
    ]
    actor = reports["0"]["actors"]["learner"]
    model, *_ = load_actor(actor["weights_path"], sha256=actor["weights_sha256"])
    cells = {f"control{i}": dict(kind="control", run_id="0", index=i) for i in range(len(controls))}
    analysis = dict(
        analysis_id="E2-dynamics",
        kind="intent_dynamics",
        expected_cell_ids=list(cells),
        parameters=dict(
            preference_names=list(model.hierarchy_config.preference_names),
            probability_bin_edges=[0, 0.5, 1],
            oscillation_distance=0.1,
        ),
    )
    change_analysis(plan, mapping, analysis, cells)
    result = assemble_measurements(plan, mapping)
    assert result["assembly_status"] == "ASSEMBLED", (
        result["blocked_cells"] or result["failed_runs"]
    )
    measured = evaluate_registered(json.loads(plan.read_text()), result)["analyses"]["E2-dynamics"]
    assert measured["learned_updates"] == sum(
        c["operation"] == "update" and c["manager_decision"] for c in controls
    )
    bad = json.loads(mapping.read_text())
    bad["cells"]["E2-dynamics"].pop(next(iter(cells)))
    # Removing a registered cell is invalid even after recomputing the mapping hash.
    rewrite(mapping, bad)
    with pytest.raises(ValueError, match="complete frozen"):
        assemble_measurements(plan, mapping)


def test_actual_typed_goal_observer_to_metric_artifact_to_full_task_ledger_analysis(tmp_path):
    plan, mapping, reports = experiment(tmp_path, paired=False, goal_mode=True)
    actual = json.loads(Path(reports["0"]["artifacts"]["evaluator_metrics"]["path"]).read_text())
    assert any(g["outcome"] == "confirmed" for g in actual["goal_observations"])

    tasks = sorted({t for control in actual["controls"] for t in control["tasks"]})
    cells = {
        f"goal{i}": dict(kind="goal", run_id="0", index=i)
        for i in range(len(actual["goal_observations"]))
    }
    analysis = dict(
        analysis_id="E3-task",
        kind="task_goals",
        expected_cell_ids=list(cells),
        parameters=dict(declared_task_ids=tasks),
    )
    change_analysis(plan, mapping, analysis, cells)
    result = assemble_measurements(plan, mapping)
    assert result["assembly_status"] == "ASSEMBLED", (
        result["blocked_cells"] or result["failed_runs"]
    )
    measured = evaluate_registered(json.loads(plan.read_text()), result)["analyses"]["E3-task"]
    assert (
        measured["observed_confirmed_tasks"] > 0
        and measured["complete_window_success_rate"] is None
    )
    assert not measured["absence_is_failure"] and not measured["command_causation_proven"]


def test_actual_grounded_collector_confidence_positive_and_censored_predictions_reach_calibration(
    tmp_path,
):
    plan, mapping, _ = experiment(tmp_path, paired=False, goal_mode=True)
    result = assemble_measurements(plan, mapping)
    assert result["assembly_status"] == "ASSEMBLED", (
        result["failed_runs"] or result["blocked_cells"]
    )
    measured = evaluate_registered(json.loads(plan.read_text()), result)["analyses"][
        "E3-confidence"
    ]
    assert measured["denominator"] > 0 and measured["status"] == "MEASURED"
    assert not measured["calibrated_claim"]


@pytest.mark.parametrize("kind", ["population_coverage", "restricted_response_gain"])
def test_complete_actual_natural_terminal_payoff_panels_feed_population_analyzers(tmp_path, kind):
    plan, mapping, reports = experiment(tmp_path)
    policy = reports["0"]["actors"]["learner"]["snapshot_id"]
    opponent = reports["0"]["actors"]["opponent"]["snapshot_id"]
    parameters = (
        dict(opponent_ids=[opponent], cluster_count=1)
        if kind == "population_coverage"
        else dict(opponent_ids=[opponent], mixture={policy: 1.0})
    )
    analysis = dict(analysis_id="E6", kind=kind, expected_cell_ids=[policy], parameters=parameters)
    change_analysis(
        plan,
        mapping,
        analysis,
        {
            policy: dict(
                kind="payoff_row", policy_snapshot_id=policy, opponents={opponent: ["0", "1"]}
            )
        },
    )
    result = assemble_measurements(plan, mapping)
    assert result["assembly_status"] == "ASSEMBLED", (
        result["blocked_cells"] or result["failed_runs"]
    )
    measured = evaluate_registered(json.loads(plan.read_text()), result)["analyses"]["E6"]
    assert (
        measured["worst_fixed_opponent_payoff"][policy]
        if kind == "population_coverage"
        else measured["worst_response_payoff"]
    ) == 1.0
    bad = json.loads(mapping.read_text())
    bad["cells"]["E6"][policy]["opponents"][opponent] = ["0", "0", "1"]
    rewrite(mapping, bad)
    assert assemble_measurements(plan, mapping)["assembly_status"] == "BLOCKED"


def test_learning_curve_cannot_supply_missing_trained_frame_endpoint(tmp_path):
    plan, mapping, _ = experiment(tmp_path)
    analysis = dict(
        analysis_id="E4",
        kind="learning_curve",
        expected_cell_ids=["initial", "final"],
        parameters=dict(frame_budget=3, attainment_threshold=0.5),
    )
    change_analysis(
        plan,
        mapping,
        analysis,
        {
            "initial": dict(kind="outcome_panel", run_ids=["0"]),
            "final": dict(kind="outcome_panel", run_ids=["1"]),
        },
    )
    result = assemble_measurements(plan, mapping)
    assert (
        result["assembly_status"] == "BLOCKED" and "registered_analysis" in result["blocked_cells"]
    )
    assert result["analyses"]["E4"]["initial"][0] == result["analyses"]["E4"]["final"][0] == 0


@pytest.mark.parametrize(
    "problem",
    ["metric_drift", "missing_cell", "unused_run", "duplicate_run", "wrong_source", "wrong_budget"],
)
def test_frozen_mapping_drift_and_incomplete_registration_fail_closed(tmp_path, problem):
    plan, mapping, _ = experiment(tmp_path)
    value = json.loads(mapping.read_text())
    if problem == "metric_drift":
        value["metric_program"]["sha256"] = "0" * 64
    elif problem == "missing_cell":
        value["cells"]["E1"].pop("army")
    elif problem == "unused_run":
        value["runs"]["unused"] = value["runs"]["0"]
    elif problem == "duplicate_run":
        value["runs"]["1"]["stage_report"] = value["runs"]["0"]["stage_report"]
    elif problem == "wrong_source":
        value["source"]["manifest"]["sha256"] = "0" * 64
    else:
        value["matched_budget_sha256"] = "0" * 64
    rewrite(mapping, value)
    with pytest.raises(ValueError):
        assemble_measurements(plan, mapping)


@pytest.mark.parametrize(
    "problem",
    [
        "truncated",
        "missing_anchor",
        "missing_metric",
        "off_clock",
        "wrong_identity",
        "wrong_stage",
        "open_process",
        "artifact_drift",
        "wrong_context",
        "wrong_full_context",
    ],
)
def test_actual_failures_are_retained_in_non_analyzable_blocked_panel(tmp_path, problem):
    plan, mapping, reports = experiment(tmp_path, truncated=problem == "truncated")
    value = json.loads(mapping.read_text())
    if problem in ("missing_anchor", "missing_metric", "off_clock"):
        cell = value["cells"]["E1"]["army"]
        if problem == "missing_anchor":
            cell["anchor"]["index"] = 9
        elif problem == "missing_metric":
            cell["metrics"]["resources"]["name"] = "unmeasured_enemy_losses"
        else:
            cell["metrics"]["resources"]["loop"] = 99
    else:
        report = reports["1"]
        path = Path(value["runs"]["1"]["stage_report"]["path"])
        if problem == "wrong_identity":
            report["identity"]["seed"] = 19
        elif problem == "wrong_stage":
            report["stage"] = "VALUE_ONLY"
        elif problem == "open_process":
            report["all_owned_sessions_closed"] = False
        elif problem == "artifact_drift":
            metric = Path(report["artifacts"]["evaluator_metrics"]["path"])
            metric.write_text(metric.read_text() + " ")
        elif problem in ("wrong_context", "wrong_full_context"):
            artifact = Path(report["artifacts"]["collector_report"]["path"])
            collector = json.loads(artifact.read_text())
            collector["registered_interventions"][0][
                "visible_input_sha256" if problem == "wrong_context" else "policy_context_sha256"
            ] = "9" * 64
            rewrite(artifact, collector)
            report["artifacts"]["collector_report"]["sha256"] = file_hash(artifact)
        if problem != "truncated":
            rewrite(path, report)
            value["runs"]["1"]["stage_report"]["sha256"] = file_hash(path)
    rewrite(mapping, value)
    result = assemble_measurements(plan, mapping)
    assert result["assembly_status"] == "BLOCKED" and (
        result["failed_runs"] or result["blocked_cells"]
    )
    assert set(result["all_registered_run_ids"]) == {"0", "1"}
    with pytest.raises(ValueError, match="independent measured"):
        evaluate_registered(json.loads(plan.read_text()), result)


def test_assemble_cli_writes_only_fresh_external_artifact(tmp_path):
    plan, mapping, _ = experiment(tmp_path)
    output = tmp_path / "assembled.json"
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "poorstar/cli/evaluation/run.py"),
            "assemble",
            "--plan",
            str(plan),
            "--mapping",
            str(mapping),
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(output.read_text())["assembly_status"] == "ASSEMBLED"
    again = subprocess.run(
        [
            sys.executable,
            str(ROOT / "poorstar/cli/evaluation/run.py"),
            "assemble",
            "--plan",
            str(plan),
            "--mapping",
            str(mapping),
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
    )
    assert again.returncode != 0


@pytest.mark.parametrize("parent", [1, 2])
def test_confidence_recorder_uses_observed_goal_same_parent_and_keeps_other_windows_censored(
    parent,
):
    from poorstar.evaluation.metrics import VisibleMetricRecorder
    from poorstar.hierarchy import HierarchyState, IntentState, TaskEffectObserver, TaskGoal
    from tests.test_hierarchy_goals import declared_task, goal_config, visible_frame

    frame, _ = visible_frame()
    later, _ = visible_frame(1, tags=(10, 20, 30))
    config = goal_config()
    observer = TaskEffectObserver(config, episode="episode", player_id=1)
    task = replace(
        declared_task(config, TaskGoal("own_completed_unit_delta", 100, 1)), parent_intent=parent
    )
    observer.register(frame, task, ())
    (observed,) = observer.observe(later)
    assert observed.outcome == "confirmed"
    intent = IntentState((0.0,) * 9, 8, 0, 0, 0.1, "update", "learned", 1)
    state = HierarchyState("episode", intent=intent)
    recorder = VisibleMetricRecorder()
    recorder.observe(frame)
    recorder.confidence(frame, state, 0.73)
    actor = dict(
        episode_id="episode",
        receipts=[],
        goal_observations=[dict(asdict(observed), player_id=1, parent_revision=parent)],
        terminated=True,
        reward=1.0,
        environment_loops=8,
    )
    ledger = recorder.report(actor)["confidence_predictions"]
    assert ledger[0]["prediction"] == 0.73 and ledger[0]["prediction"] != intent.confidence
    assert ledger[0]["target"]["mask"] == (parent == 1)
    assert ledger[0]["target"]["value"] == (1.0 if parent == 1 else None)
    with pytest.raises(ValueError, match="duplicate"):
        recorder.confidence(frame, state, 0.73)
    with pytest.raises(ValueError, match="actual current"):
        recorder.confidence(frame, replace(state, intent=replace(intent, source="external")), 0.73)
