"""Tests: test stage actor."""

import json
import subprocess
import sys
from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import Path

import pytest
import torch

from poorstar.league import SnapshotRegistry
from poorstar.league.ownership import write_document
from poorstar.training.actor import collect_pair
from poorstar.training.checkpoint import file_hash
from poorstar.training.contracts import digest
from poorstar.training.export import export_actor, load_actor
from poorstar.training.job_worker import actor_parameter_counts
from poorstar.training.stage_actor import (
    FIXTURE_SCOPE,
    StageActorPlan,
    execute_stage_actor,
    validate_phase_preflight,
)
from tests.test_training_system_actor import ProtocolPair, policies, replay_every_step
from tests.test_training_system_foundation import system

TREE = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def threads():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


class OwnedProtocolPair(ProtocolPair):
    qualification_scope = FIXTURE_SCOPE

    def __init__(self, terminal_loop=3):
        super().__init__(terminal_loop)
        self.owned_process_results = []
        self.closed = False
        self.enter_count = 0

    def __enter__(self):
        self.enter_count += 1
        return self

    def close(self):
        self.closed = True

    def __exit__(self, *_):
        self.close()


def stage_fixture(tmp_path, phase="VALUE_ONLY", provenance="learner", arm="persistent_intent"):
    learner, profile, _ = system(arm, "VALUE_ONLY" if phase == "BC" else phase)
    if provenance == "learner":
        learner.config = replace(
            learner.config, maximum_batch_sequences=2, maximum_sequence_steps=3
        )
        learner.identity = replace(learner.identity, budget_sha256=learner.config.sha256)
    source = tmp_path / "source.json"
    # A labelled subset manifest proves these fixture inputs only, never a
    # release-source or pipeline capability qualification.
    paths = [
        "poorstar/training/stage_actor.py",
        "poorstar/training/actor.py",
        "poorstar/evaluation/metrics.py",
    ]
    source.write_text(
        json.dumps(
            dict(
                schema="STAGE_ACTOR_ENGINEERING_SUBSET_NOT_RELEASE",
                files=[
                    dict(path=p, sha256=file_hash(TREE / p), bytes=(TREE / p).stat().st_size)
                    for p in paths
                ],
            ),
            sort_keys=True,
        )
    )
    learner.identity = replace(learner.identity, source_sha256=file_hash(source))
    if phase == "BC":
        from poorstar.data.bc.windows import BCRunPlan
        from poorstar.training.bc import HierarchicalBCLearner

        bp = BCRunPlan("ENGINEERING", 71, 1, hidden_size=32, unroll_length=2, burn_in_length=1)
        ident = replace(
            learner.identity,
            budget_sha256=digest(dict(plan=asdict(bp), head_weights={}, control_weights={})),
        )
        learner = HierarchicalBCLearner(learner.model, profile, ident, bp)
    registry = SnapshotRegistry(tmp_path / "registry")
    export = tmp_path / "initial_actor.pt"
    export_actor(export, learner)
    main = registry.register(
        export, identity=learner.identity, policy_version=learner.policy_version, role="main"
    )
    anchor = registry.register(
        export, identity=learner.identity, policy_version=learner.policy_version, role="anchor"
    )
    counts = actor_parameter_counts(learner.model, phase)
    budget = dict(
        counts,
        maximum_parameters=counts["parameter_count"],
        maximum_trainable_parameters=counts["trainable_parameter_count"],
    )
    execution = dict(
        device="cpu",
        version="protocol-fixture",
        sc2_path="UNUSED_FIXTURE_NO_SC2",
        source_manifest=str(source),
        unroll_steps=2,
        burn_in_steps=1,
        maximum_resident_steps=4,
        maximum_sources=63,
        deterministic_learner=False,
        deterministic_opponent=True,
    )
    if provenance == "learner":
        execution["learner_config"] = asdict(learner.config)
    if provenance == "evaluator":
        execution.update(
            measurement_period=1,
            metric_program_sha256=file_hash(TREE / "poorstar/evaluation/metrics.py"),
        )
    plan = StageActorPlan(
        asdict(learner.identity),
        phase,
        provenance,
        digest(["FIXTURE", phase, provenance]),
        main["snapshot_id"],
        anchor["snapshot_id"],
        "FIXTURE_NOT_A_MAP",
        71,
        6,
        100000,
        budget,
        execution,
    )
    return learner, registry, plan


def test_value_stage_zero_publishes_exact_sampled_raw_windows_without_future_gates(tmp_path):
    learner, registry, plan = stage_fixture(tmp_path)
    assert (
        learner.updates == learner.optimizer_steps == learner.environment_loops == 0
        and not learner.optimizer.state
    )
    before = learner.parameter_sha256
    backend = OwnedProtocolPair()
    result = execute_stage_actor(plan, registry, tmp_path / "output", protocol_fixture=backend)
    assert result["status"] == "COMPLETE" and result["all_owned_sessions_closed"]
    assert result["actual_engine_frames"] == 0 and result["environment_loops"] == 3
    assert result["fixture_scope"] == FIXTURE_SCOPE and not result["engine_launch_attempted"]
    assert learner.parameter_sha256 == before and learner.optimizer_steps == 0
    assert not any(
        result[k]
        for k in (
            "optimizer_created",
            "selfplay_started",
            "league_started",
            "capability_evaluated",
            "pipeline_stage_unlocked",
        )
    )
    assert (
        result["source_preflight"] == result["source_postflight"]
        and not result["source_postflight"]["problems"]
    )
    assert result["actors"]["learner"]["training_phase"] == "VALUE_ONLY"
    index = json.loads(Path(result["artifacts"]["rollout_index"]["path"]).read_text())
    assert (
        index["physical_environment_loops"] == 3 and index["collection_plan_sha256"] == plan.sha256
    )
    assert len(index["shards"]) == 2 and index["update_groups"] == [
        [s["sha256"] for s in index["shards"]]
    ]
    from poorstar.training.storage import RolloutReader

    recovered = []
    for shard in index["shards"]:
        reader = RolloutReader(shard["path"], sha256=shard["sha256"], identity=learner.identity)
        try:
            seq = reader.sequence()
            recovered.append(seq)
            seq.validate(learner.identity, 0, 0, learning=True)
            hidden = seq.initial_hidden
            for step in seq.steps:
                assert step.behavior == "sampled"
                scored = learner.scorer(step, hidden)
                hidden = scored.next_hidden
                assert float(scored.log_prob) == pytest.approx(step.old_log_prob, abs=1e-5)
                assert float(scored.value) == pytest.approx(step.old_value, abs=1e-6)
        finally:
            reader.close()
    assert backend.closed and backend.enter_count == 1
    # A declared mathematical outcome from the protocol fixture tests the actual
    # receiving logical-batch interface, never SC2 gameplay or formal RL.
    update = learner.update(recovered)
    assert update["optimizer_steps"] == 1 and update["environment_loops"] == 3
    assert set(update["changed_parameters"]) <= {n for n, _ in learner.groups["value"]}


@pytest.mark.parametrize("phase", ["BC", "VALUE_ONLY", "MANAGER_RL", "JOINT_RL"])
def test_existing_exported_phase_evaluator_produces_no_training_index(tmp_path, phase):
    _, registry, plan = stage_fixture(tmp_path, phase, "evaluator")
    result = execute_stage_actor(
        plan, registry, tmp_path / "output", protocol_fixture=OwnedProtocolPair()
    )
    assert result["status"] == "COMPLETE" and result["stage"] == phase
    assert "evaluator_metrics" in result["artifacts"] and "rollout_index" not in result["artifacts"]
    assert not (tmp_path / "output" / "rollouts").exists()
    collector = json.loads(Path(result["artifacts"]["collector_report"]["path"]).read_text())
    metrics = json.loads(Path(result["artifacts"]["evaluator_metrics"]["path"]).read_text())
    assert metrics["provenance"] == "evaluator" and metrics["physical_environment_loops"] == 3
    assert metrics["terminal_outcome"] == 1
    assert (
        collector["deterministic_players"] == [False, True]
        and len(collector["initial_inputs"]) == 2
    )
    assert {a["player_id"] for a in collector["initial_inputs"]} == {1, 2}
    assert all(
        a["loop"] == 0 and a["pre_intervention_action_prefix_sha256"] == digest([])
        for a in collector["initial_inputs"]
    )
    assert all(len(a["policy_context_sha256"]) == 64 for a in collector["initial_inputs"])


def test_sampler_opponent_greedy_and_old_default_are_exact_replayable():
    learner, opponent, profile = policies(0)
    torch.manual_seed(88)
    seq, report = collect_pair(
        ProtocolPair(3),
        (learner.model, opponent),
        (profile, profile),
        learner.identity,
        maximum_loops=5,
        episode_id="per-player",
        deterministic_players=(False, True),
    )
    assert report["deterministic_players"] == [False, True] and all(
        s.behavior == "sampled" for s in seq.steps
    )
    replay_every_step(learner, seq)
    torch.manual_seed(88)
    a, ar = collect_pair(
        ProtocolPair(3),
        (learner.model, opponent),
        (profile, profile),
        learner.identity,
        maximum_loops=5,
        episode_id="old-default",
    )
    torch.manual_seed(88)
    b, br = collect_pair(
        ProtocolPair(3),
        (learner.model, opponent),
        (profile, profile),
        learner.identity,
        maximum_loops=5,
        episode_id="old-default",
        deterministic_players=(False, False),
    )
    assert ar == br and [s.payload.trace for s in a.steps] == [s.payload.trace for s in b.steps]


@pytest.mark.parametrize("flags", [(False,), [False, True], (False, 1), ("no", True)])
def test_per_player_sampling_mode_strict_booleans(flags):
    learner, opponent, profile = policies(0)
    backend = ProtocolPair(3)
    with pytest.raises(ValueError, match="two explicit boolean"):
        collect_pair(
            backend,
            (learner.model, opponent),
            (profile, profile),
            learner.identity,
            maximum_loops=4,
            deterministic_players=flags,
        )
    assert not backend.advances and not backend.dispatches


def test_bc_learner_greedy_learner_and_intervened_ppo_are_rejected_before_backend(tmp_path):
    _, _, bc = stage_fixture(tmp_path, "BC", "evaluator")
    with pytest.raises(ValueError, match="evaluator-only"):
        replace(bc, provenance="learner")
    p = dict(bc.execution, deterministic_learner=True)
    with pytest.raises(ValueError, match="greedy learner"):
        replace(bc, phase="VALUE_ONLY", provenance="learner", execution=p)
    p = dict(bc.execution, preference_interventions={"learner": [dict(loop=0)]})
    with pytest.raises(ValueError, match="paper interventions"):
        replace(bc, phase="VALUE_ONLY", provenance="learner", execution=p)


@pytest.mark.parametrize("changed", ["identity_budget", "logical_batch", "sequence", "frames"])
def test_receiving_ppo_numeric_capacity_cannot_be_silently_relaxed(tmp_path, changed):
    _, _, plan = stage_fixture(tmp_path)
    if changed == "identity_budget":
        with pytest.raises(ValueError, match="identity budget"):
            replace(plan, identity=dict(plan.identity, budget_sha256="0" * 64))
        return
    execution = deepcopy(plan.execution)
    if changed == "logical_batch":
        execution["maximum_resident_steps"] = 5
    elif changed == "sequence":
        execution["burn_in_steps"] = 2
    if changed == "frames":
        with pytest.raises(ValueError, match="capacity"):
            replace(plan, maximum_loops=101)
    else:
        with pytest.raises(ValueError, match="capacity"):
            replace(plan, execution=execution)


def test_engineering_dispatch_and_formal_fixture_do_not_authorize_engine(tmp_path):
    _, registry, plan = stage_fixture(tmp_path)
    result = execute_stage_actor(plan, registry, tmp_path / "actual_unapproved")
    assert result["status"] == "ERROR" and not result["engine_launch_attempted"]
    assert result["environment_loops"] is None and "FORMAL" in result["error"]["error"]
    formal = replace(plan, identity=dict(plan.identity, purpose="FORMAL"))
    backend = OwnedProtocolPair()
    with pytest.raises(ValueError, match="cannot authorize"):
        execute_stage_actor(formal, registry, tmp_path / "formal_fixture", protocol_fixture=backend)
    assert backend.enter_count == 0 and not (tmp_path / "formal_fixture").exists()


@pytest.mark.parametrize(
    "changed", ["phase", "parameter", "source", "metric_program", "actor_version"]
)
def test_static_export_execution_drift_is_pre_engine_failure(tmp_path, changed):
    _, registry, plan = stage_fixture(tmp_path, "VALUE_ONLY", "evaluator")
    backend = OwnedProtocolPair()
    if changed == "phase":
        plan = replace(plan, phase="MANAGER_RL")
    elif changed == "parameter":
        plan = replace(
            plan,
            parameter_budget=dict(
                plan.parameter_budget, parameter_count=plan.parameter_budget["parameter_count"] - 1
            ),
        )
    elif changed == "source":
        plan = replace(plan, identity=dict(plan.identity, source_sha256="0" * 64))
    elif changed == "metric_program":
        plan = replace(plan, execution=dict(plan.execution, metric_program_sha256="0" * 64))
    else:
        actor = registry.weights_path(plan.learner_snapshot_id)
        payload = torch.load(actor, weights_only=False)
        payload["policy_version"] = 99
        torch.save(payload, actor)
    result = execute_stage_actor(plan, registry, tmp_path / "output", protocol_fixture=backend)
    assert (
        result["status"] == "ERROR"
        and backend.enter_count == 0
        and result["environment_loops"] is None
    )
    assert result["all_owned_sessions_closed"] and "failure" in result["artifacts"]


def test_full_known_frame_receipt_preserved_when_source_postflight_fails(tmp_path):
    _, registry, plan = stage_fixture(tmp_path, "VALUE_ONLY", "evaluator")

    class ChangeExternalManifest(OwnedProtocolPair):
        def close(self):
            super().close()
            source = Path(plan.execution["source_manifest"])
            value = json.loads(source.read_text())
            value["postflight_drift_fixture"] = True
            source.write_text(json.dumps(value))

    result = execute_stage_actor(
        plan, registry, tmp_path / "output", protocol_fixture=ChangeExternalManifest()
    )
    assert (
        result["status"] == "ERROR"
        and result["environment_loops"] == 3
        and result["all_owned_sessions_closed"]
    )
    assert "collector_report" in result["artifacts"] and "rollout_index" not in result["artifacts"]
    assert result["source_postflight"]["manifest_sha256"] != result["identity"]["source_sha256"]
    failure = json.loads(Path(result["artifacts"]["failure"]["path"]).read_text())
    assert failure["observed_environment_loops"] == 3


def test_unresolved_midcollection_failure_preserves_unknown_frames_and_cleanup(tmp_path):
    _, registry, plan = stage_fixture(tmp_path)

    class BrokenAdvance(OwnedProtocolPair):
        def advance(self, loops):
            raise RuntimeError("owned protocol interrupted; no complete collector receipt")

    result = execute_stage_actor(
        plan, registry, tmp_path / "output", protocol_fixture=BrokenAdvance()
    )
    assert (
        result["status"] == "ERROR"
        and result["environment_loops"] is None
        and result["all_owned_sessions_closed"]
    )
    assert (
        "rollout_index" not in result["artifacts"] and "collector_report" not in result["artifacts"]
    )


@pytest.mark.parametrize("phase", ["BC", "VALUE_ONLY"])
def test_existing_phase_gate_no_future_selfplay_or_league_requirement(tmp_path, phase):
    from poorstar.training.gates import REQUIRED

    _, _, plan = stage_fixture(tmp_path, phase, "evaluator")
    plan = replace(plan, identity=dict(plan.identity, purpose="FORMAL"))
    identity = plan.identity
    refs = {}
    # Only pure gate verification is invoked. These explicit mock reports never
    # enter execute_stage_actor or create an engine.
    for gate in REQUIRED[phase]:
        p = tmp_path / (gate + ".json")
        p.write_text(json.dumps(dict(status="PASS", gate=gate, binding=identity)))
        refs[gate] = dict(path=str(p), sha256=file_hash(p), binding=identity)
    preflight = tmp_path / "pure_mock_preflight.json"
    write_document(
        preflight,
        dict(
            schema="poorstar_v3_training_launch_preflight_v1",
            identity_sha256=digest(identity),
            stage=phase,
            status="READY",
            execution_config_sha256=plan.sha256,
            evidence=refs,
            gates={k: "PASS" for k in refs},
        ),
        immutable=True,
    )
    assert validate_phase_preflight(plan, preflight)["stage"] == phase
    assert not set(refs) & {
        "outcome_rl",
        "selfplay_capability",
        "intent_causal_gate",
        "value_pretraining",
    }
    missing = json.loads(preflight.read_text())
    missing.pop("content_digest")
    missing["gates"][REQUIRED[phase][0]] = "FAIL"
    rejected = tmp_path / "pure_mock_blocked.json"
    write_document(rejected, missing, immutable=True)
    with pytest.raises(ValueError, match="launch blocked"):
        validate_phase_preflight(plan, rejected)


@pytest.mark.parametrize("grounded", [False, True])
def test_confidence_measurement_adds_only_prior_context_receipts_without_sampling_changes(grounded):
    from poorstar.evaluation.metrics import VisibleMetricRecorder

    learner, opponent, profile = policies(0)
    if grounded:
        from poorstar.hierarchy.model import HierarchicalPolicy
        from poorstar.training.config import model_config_sha256
        from poorstar.training.scorer import HierarchyScorer

        learner.model = HierarchicalPolicy(
            learner.model.config,
            replace(
                learner.model.hierarchy_config,
                grounded_tasks=True,
                common_receipts_mode="shared_v1",
            ),
        ).eval()
        learner.scorer = HierarchyScorer(learner.model, profile)
        learner.identity = replace(
            learner.identity,
            model_config_sha256=model_config_sha256(
                learner.model.config, learner.model.hierarchy_config
            ),
        )
        opponent = deepcopy(learner.model)
    with torch.no_grad():
        for model in (learner.model, opponent):
            model.manager_operation_head.bias.copy_(torch.tensor([-30.0, 30.0, -30.0]))
            # Keep this declared protocol clock at one-loop events; this fixture
            # does not emulate SC2's stop-at-terminal within a larger step call.
            model.act_head.bias.copy_(torch.tensor([30.0, -30.0]))
            model.delay_head.bias.fill_(-30.0)
            model.delay_head.bias[0] = 30.0
    before = learner.parameter_sha256
    torch.manual_seed(777)
    a_backend = ProtocolPair(4)
    a, ar = collect_pair(
        a_backend,
        (learner.model, opponent),
        (profile, profile),
        learner.identity,
        maximum_loops=6,
        provenance="evaluator",
        episode_id="confidence-measurement",
        deterministic_players=(False, True),
    )
    rng_a = torch.get_rng_state().clone()
    sink = VisibleMetricRecorder()
    torch.manual_seed(777)
    b_backend = ProtocolPair(4)
    b, br = collect_pair(
        b_backend,
        (learner.model, opponent),
        (profile, profile),
        learner.identity,
        maximum_loops=6,
        provenance="evaluator",
        episode_id="confidence-measurement",
        deterministic_players=(False, True),
        measurement_sink=sink,
        measurement_period=1,
    )
    assert torch.equal(rng_a, torch.get_rng_state()) and before == learner.parameter_sha256
    assert (
        a_backend.dispatches == b_backend.dispatches
        and a_backend.advances == b_backend.advances
        and ar == br
    )
    assert [s.payload.trace for s in a.steps] == [s.payload.trace for s in b.steps]
    assert all(
        torch.equal(x.payload.memory_before, y.payload.memory_before)
        for x, y in zip(a.steps, b.steps)
        if x.payload.memory_before is not None
    )
    predictions = sink.report(br)["confidence_predictions"]
    assert predictions and all(
        p["prediction_kind"] == "PURE_PRECHOICE_MANAGER_BETA_MEAN" for p in predictions
    )
    from poorstar.hierarchy.state import ExecutionFeedback, HierarchyState

    for prediction in predictions:
        step = next(s for s in b.steps if s.game_loop == prediction["window"]["start_loop"])
        trace = step.payload.trace
        from poorstar.hierarchy.state import DispatchObservation

        dispatch = trace.get("scheduler_dispatch_observation")
        value = learner.model.confidence_prediction(
            step.payload.scheduler_frame.policy_observation(),
            step.payload.memory_before,
            HierarchyState.from_dict(trace["prior_control_state"], learner.model.hierarchy_config),
            feedback=tuple(ExecutionFeedback(**f) for f in trace["feedback"]),
            dispatch_observation=None if dispatch is None else DispatchObservation(**dispatch),
            task_allocations=trace.get("scheduler_task_allocations"),
        )
        assert float(value) == pytest.approx(prediction["prediction"], abs=1e-7)
        assert prediction["target"]["mask"] == 0  # no invented absence/negative coverage


@pytest.mark.parametrize("arm", ["flat", "direct_task"])
def test_no_confidence_head_is_never_called_by_metric_collection(arm):
    learner, profile, _ = system(arm)
    opponent = deepcopy(learner.model)

    class NoConfidenceCalls:
        def observe(self, *_):
            pass

        def control(self, *_):
            pass

        def confidence(self, *_):
            raise AssertionError("a headless arm cannot produce a prediction")

    _, report = collect_pair(
        ProtocolPair(3),
        (learner.model, opponent),
        (profile, profile),
        learner.identity,
        maximum_loops=4,
        provenance="evaluator",
        measurement_sink=NoConfidenceCalls(),
        measurement_period=1,
    )
    assert report["terminated"]


def test_registered_midgame_pair_preserves_actual_context_and_prefix_with_shared_episode(tmp_path):
    _, registry, base = stage_fixture(tmp_path, "BC", "evaluator")
    reports = []
    for i, vector in enumerate(((0.0,) * 9, (0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0))):
        execution = dict(
            base.execution,
            episode_id="REGISTERED_PROTOCOL_SAME_SCENARIO",
            preference_interventions={
                "learner": [dict(loop=2, preference=list(vector), commitment_horizon=4)]
            },
        )
        plan = replace(base, execution=execution, collection_id=digest(["H1_PROTOCOL_PAIR", i]))
        result = execute_stage_actor(
            plan, registry, tmp_path / ("output" + str(i)), protocol_fixture=OwnedProtocolPair(4)
        )
        assert result["status"] == "COMPLETE", result["error"]
        reports.append(
            json.loads(Path(result["artifacts"]["collector_report"]["path"]).read_text())
        )
    assert (
        reports[0]["episode_id"] == reports[1]["episode_id"] == "REGISTERED_PROTOCOL_SAME_SCENARIO"
    )
    a, b = [r["registered_interventions"][0] for r in reports]
    assert a["actual_scheduler_loop"] == b["actual_scheduler_loop"] == 2
    for name in (
        "visible_input_sha256",
        "policy_context_sha256",
        "pre_intervention_action_prefix_sha256",
    ):
        assert a[name] == b[name]
    assert a["intervention"]["preference"] != b["intervention"]["preference"]


@pytest.mark.parametrize(
    "changed", ["hidden", "prior_state", "feedback", "receipt", "allocation", "parameters"]
)
def test_same_c3_input_does_not_conflate_different_prior_policy_context(changed):
    from poorstar.evaluation.interventions import visible_input_digest
    from poorstar.hierarchy.state import DispatchObservation, ExecutionFeedback, HierarchyState
    from poorstar.training.actor import policy_context_digest
    from tests.test_coarse_bc import setup

    learner, _, _ = system("persistent_intent")
    frame, *_ = setup()
    obs = frame.policy_observation()
    prior = HierarchyState("REGISTERED_PROTOCOL_SAME_SCENARIO")
    state = prior
    hidden = None
    feedback = ()
    dispatch = None
    allocations = None
    original = policy_context_digest(obs, learner.model, None, prior)
    if changed == "hidden":
        hidden = torch.zeros(learner.model.config.hidden_size)
    elif changed == "prior_state":
        state = replace(prior, last_loop=0)
    elif changed == "feedback":
        feedback = (
            ExecutionFeedback("actual-fixture-event", "task-fixture", "accepted", 0, "a" * 64),
        )
    elif changed == "receipt":
        dispatch = DispatchObservation(0, (1, 1, 0, 1), "a" * 64)
    elif changed == "allocation":
        allocations = {"task-fixture": (0,)}
    else:
        with torch.no_grad():
            next(learner.model.parameters()).add_(0.001)
    changed_context = policy_context_digest(
        obs,
        learner.model,
        hidden,
        state,
        feedback=feedback,
        dispatch_observation=dispatch,
        task_allocations=allocations,
    )
    assert changed_context != original
    assert visible_input_digest(obs) == visible_input_digest(frame.policy_observation())


def test_cached_context_weight_binding_is_proven_for_the_fixed_opponent_too():
    learner, opponent, profile = policies(0)

    class MutatingOpponent(ProtocolPair):
        def advance(self, loops):
            super().advance(loops)
            if self.loop == 1:
                with torch.no_grad():
                    opponent.act_head.bias.add_(0.01)

    with pytest.raises(ValueError, match="actor parameters changed"):
        collect_pair(
            MutatingOpponent(3),
            (learner.model, opponent),
            (profile, profile),
            learner.identity,
            maximum_loops=4,
            episode_id="immutable-opponent-proof",
        )


def test_cli_plan_only_is_fixed_pair_and_dispatch_remains_blocked(tmp_path):
    _, registry, plan = stage_fixture(tmp_path)
    planfile = tmp_path / "plan.json"
    write_document(planfile, asdict(plan), immutable=True)
    command = [
        sys.executable,
        str(TREE / "poorstar/cli/training/collect.py"),
        "--plan",
        str(planfile),
        "--registry",
        str(registry.root),
    ]
    process = subprocess.run(
        command + ["--output-dir", str(tmp_path / "planned")], capture_output=True, text=True
    )
    assert process.returncode == 0, process.stderr
    report = json.loads((tmp_path / "planned/stage_report.json").read_text())
    assert (
        report["status"] == "PLAN_ONLY"
        and not report["engine_started"]
        and not report["capability_evaluated"]
    )
    dispatched = subprocess.run(
        command + ["--output-dir", str(tmp_path / "blocked"), "--dispatch"],
        capture_output=True,
        text=True,
    )
    assert (
        dispatched.returncode == 1
        and json.loads((tmp_path / "blocked/stage_report.json").read_text())["status"] == "ERROR"
    )
