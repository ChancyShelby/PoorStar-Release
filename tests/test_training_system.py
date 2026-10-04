"""Tests: test training system."""

import copy
import json
import random
from dataclasses import replace

import numpy as np
import pytest
import torch
from torch import nn
from torch.distributions import Categorical

from poorstar.league import LeagueConfig, LeagueScheduler, SnapshotRegistry
from poorstar.training import (
    PPOConfig,
    PPOLearner,
    RolloutSequence,
    RolloutStep,
    StepScore,
    TrainingIdentity,
)
from poorstar.training.checkpoint import capture_rng, load_checkpoint, save_checkpoint
from poorstar.training.contracts import digest
from poorstar.training.gates import validate_launch
from poorstar.training.returns import targets


@pytest.fixture(autouse=True)
def single_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def identity(config, **kwargs):
    values = dict(
        experiment="engineering-only",
        arm="persistent_intent",
        seed=7,
        source_sha256="a" * 64,
        dataset_sha256="b" * 64,
        split_sha256="c" * 64,
        scope_sha256="d" * 64,
        model_config_sha256="e" * 64,
        profile_sha256="f" * 64,
        budget_sha256=config.sha256,
    )
    values.update(kwargs)
    return TrainingIdentity(**values)


class MechanismPolicy(nn.Module):
    """Small mathematical fixture to independently verify optimizer plumbing."""

    def __init__(self):
        super().__init__()
        self.backbone = nn.Linear(2, 4)
        self.decoder = nn.Linear(4, 2)
        self.manager = nn.Linear(4, 2)
        self.task = nn.Linear(4, 2)
        self.value = nn.Linear(4, 2)

    def parameter_groups(self):
        return {
            k: tuple((n, p) for n, p in self.named_parameters() if n.startswith(k + "."))
            for k in ("backbone", "decoder", "manager", "task", "value")
        }


def scorer(model):
    def score(step, hidden):
        feature = torch.tanh(
            model.backbone(torch.tensor(step.payload["x"]))
            + (0 if hidden is None else 0.1 * hidden)
        )
        action = Categorical(logits=model.decoder(feature))
        task = Categorical(logits=model.task(feature))
        manager = Categorical(logits=model.manager(feature))
        values = model.value(feature)
        choice = torch.tensor(step.payload["action"])
        mlp = manager.log_prob(choice) if step.manager_decision else values.sum() * 0
        ment = manager.entropy() if step.manager_decision else values.sum() * 0
        return StepScore(
            action.log_prob(choice) + task.log_prob(choice) + mlp,
            action.entropy() + task.entropy() + ment,
            values[0],
            feature,
            mlp,
            ment,
            values[1],
            step.observation_sha256,
            step.mask_sha256,
        )

    return score


def sequence(learner, episode="mechanism", *, truncated=False):
    steps = []
    hidden = None
    clock = 0
    with torch.no_grad():
        for i, duration in enumerate((3, 8, 4)):
            step = RolloutStep(
                episode,
                i,
                clock,
                clock + duration,
                learner.policy_version,
                learner.parameter_sha256,
                digest(["observation", i]),
                digest(["mask", i]),
                dict(x=[i * 0.1, 0.5], action=i % 2),
                manager_decision=i in (0, 2),
            )
            score = learner.scorer(step, hidden)
            hidden = score.next_hidden
            steps.append(
                replace(
                    step,
                    old_log_prob=float(score.log_prob),
                    old_manager_log_prob=float(score.manager_log_prob),
                    old_value=float(score.value),
                    old_manager_value=float(score.manager_value),
                )
            )
            clock += duration
    for i, step in enumerate(steps[:-1]):
        steps[i] = replace(
            step,
            next_value=steps[i + 1].old_value,
            next_manager_value=steps[i + 1].old_manager_value,
        )
    steps[-1] = replace(
        steps[-1],
        terminated=not truncated,
        truncated=truncated,
        boundary_reason="step_cap" if truncated else "engine_terminal",
        reward=0.0 if truncated else 1.0,
        reward_loop=None if truncated else clock,
        next_value=0.5 if truncated else 0.0,
        next_manager_value=0.25 if truncated else 0.0,
    )
    return RolloutSequence(tuple(steps), learner.identity.sha256)


def learner(phase="JOINT_RL", **kwargs):
    torch.manual_seed(7)
    config = PPOConfig(maximum_updates=5, maximum_environment_loops=1000, **kwargs)
    model = MechanismPolicy()
    return PPOLearner(model, scorer(model), identity(config), config, phase=phase)


@pytest.mark.parametrize(
    "phase,allowed",
    [
        ("VALUE_ONLY", {"value"}),
        ("MANAGER_RL", {"manager", "value"}),
        ("JOINT_RL", {"backbone", "decoder", "manager", "task", "value"}),
    ],
)
def test_actual_optimizer_and_frozen_stage_groups(phase, allowed):
    train = learner(phase)
    report = train.update([sequence(train)])
    changed = {n.split(".")[0] for n in report["changed_parameters"]}
    assert changed and changed <= allowed
    assert report["optimizer_steps"] == 1 and report["environment_loops"] == 15
    assert report["post_update"]["samples"] == (2 if phase == "MANAGER_RL" else 3)
    assert not report["capability_evaluated"] and not report["rl_ready"]
    assert train.optimizer.state and all(torch.isfinite(v).all() for v in train.model.parameters())


def test_semi_markov_returns_follow_real_duration_and_reward_time():
    train = learner(normalize_advantage=False, gamma_per_loop=0.9, lambda_per_loop=1.0)
    seq = sequence(train)
    for manager in (False, True):
        ret = targets(seq, train.config, manager=manager)
        expected = [0.9 ** (15 - seq.steps[i].game_loop) for i in ret.indices]
        assert torch.allclose(ret.returns, torch.tensor(expected), atol=1e-6)
    assert targets(seq, train.config, manager=True).indices == (0, 2)


def test_truncation_bootstraps_and_never_receives_outcome_reward():
    train = learner(gamma_per_loop=0.9, lambda_per_loop=1.0)
    seq = sequence(train, truncated=True)
    assert targets(seq, train.config).returns[-1] == pytest.approx(0.9**4 * 0.5)
    assert targets(seq, train.config, manager=True).returns[-1] == pytest.approx(0.9**4 * 0.25)
    with pytest.raises(ValueError, match="shaped"):
        replace(seq.steps[0], reward=0.1).validate()


@pytest.mark.parametrize(
    "change",
    [
        dict(provenance="evaluator"),
        dict(behavior="deterministic"),
        dict(policy_version=99),
        dict(checkpoint_sha256="1" * 64),
    ],
)
def test_refuse_evaluator_argmax_future_and_foreign_checkpoint(change):
    train = learner()
    seq = sequence(train)
    bad = replace(seq, steps=tuple(replace(s, **change) for s in seq.steps))
    with pytest.raises(ValueError):
        train.update([bad])
    assert train.optimizer_steps == 0 and train.environment_loops == 0


def test_replay_probability_and_mask_are_checked_before_update():
    train = learner()
    seq = sequence(train)
    altered = replace(seq.steps[0], old_log_prob=seq.steps[0].old_log_prob + 0.2)
    with pytest.raises(ValueError, match="probability mismatch"):
        train.update([replace(seq, steps=(altered,) + seq.steps[1:])])
    original = train.scorer
    train.scorer = lambda s, h: replace(original(s, h), mask_sha256="1" * 64)
    with pytest.raises(ValueError, match="domain drift"):
        train.update([seq])


def test_post_update_kl_failure_restores_parameters_adam_and_rng():
    train = learner(learning_rate=0.1, max_kl=1e-12)
    seq = sequence(train)
    before = train.parameter_sha256
    rng = torch.get_rng_state().clone()
    with pytest.raises(ValueError, match="KL"):
        train.update([seq])
    assert train.parameter_sha256 == before and not train.optimizer.state
    assert torch.equal(rng, torch.get_rng_state())
    assert train.optimizer_steps == train.policy_version == train.environment_loops == 0


def test_checkpoint_rebuild_reproduces_next_sample_update_and_rng(tmp_path):
    train = learner()
    train.update([sequence(train)])
    save_checkpoint(
        tmp_path / "checkpoint",
        train,
        sampler_state={"cursor": 3},
        actor_states={"pending": "kept"},
    )
    rng_expected = random.random(), np.random.rand(), torch.rand(3)
    expected = train.update([sequence(train, episode="next")])
    restored = learner()
    recovered = load_checkpoint(tmp_path / "checkpoint", restored)
    assert recovered["sampler_state"] == {"cursor": 3} and recovered["actor_states"] == {
        "pending": "kept"
    }
    assert random.random() == rng_expected[0] and np.random.rand() == rng_expected[1]
    assert torch.equal(torch.rand(3), rng_expected[2])
    actual = restored.update([sequence(restored, episode="next")])
    assert actual == expected and restored.parameter_sha256 == train.parameter_sha256
    with pytest.raises(FileExistsError):
        save_checkpoint(tmp_path / "checkpoint", train)
    with (tmp_path / "checkpoint" / "state.pt").open("ab") as f:
        f.write(b"corrupt")
    with pytest.raises(ValueError, match="corruption"):
        load_checkpoint(tmp_path / "checkpoint", restored)


def test_environment_budget_and_duplicate_consumption_are_closed():
    train = learner(maximum_policy_lag=2)
    seq = sequence(train)
    train.update([seq])
    with pytest.raises(ValueError, match="already consumed"):
        train.update([seq])
    limited = learner()
    limited.config = replace(limited.config, maximum_environment_loops=14)
    with pytest.raises(ValueError, match="budget"):
        limited.update([sequence(limited)])


def test_formal_engine_launch_is_closed_without_hashed_capability_evidence():
    train = learner()
    with pytest.raises(ValueError, match="FORMAL"):
        validate_launch(None, train.identity, "SELFPLAY")
    with pytest.raises(ValueError, match="missing"):
        validate_launch(None, replace(train.identity, purpose="FORMAL"), "SELFPLAY")


def test_immutable_snapshots_owned_jobs_and_separate_evaluation_payoff(tmp_path):
    train = learner()
    weights = tmp_path / "weights.pt"
    torch.save(train.model.state_dict(), weights)
    registry = SnapshotRegistry(tmp_path / "snapshots")
    first = registry.register(weights, identity=train.identity, policy_version=0, role="main")
    second = registry.register(
        weights, identity=train.identity, policy_version=0, role="historical"
    )
    assert registry.read(first["snapshot_id"]) == first
    cfg = LeagueConfig(7, 100, 50, 20, 2)
    scheduler = LeagueScheduler(tmp_path / "league.sqlite", registry, train.identity, cfg)
    try:
        job = scheduler.schedule(
            first["snapshot_id"],
            [second["snapshot_id"]],
            provenance="evaluator",
            map_name="AbyssalReef",
        )
        assert scheduler.claim("actor1") == job and scheduler.claim("actor2") is None
        with pytest.raises(ValueError, match="owner"):
            scheduler.finish(
                job["job_id"],
                "actor2",
                verdict="win",
                environment_loops=12,
                evidence_sha256="a" * 64,
            )
        scheduler.finish(
            job["job_id"], "actor1", verdict="win", environment_loops=12, evidence_sha256="a" * 64
        )
        assert scheduler.payoff(first["snapshot_id"], second["snapshot_id"])["games"] == 0
        assert (
            scheduler.payoff(first["snapshot_id"], second["snapshot_id"], provenance="evaluator")[
                "win_rate"
            ]
            == 1
        )
        training = scheduler.schedule(
            first["snapshot_id"], [second["snapshot_id"]], map_name="AbyssalReef"
        )
        scheduler.claim("actor1")
        scheduler.finish(training["job_id"], "actor1", verdict="error", evidence_sha256="b" * 64)
        assert scheduler.report()["budgets"]["learner"] == {"spent": 20, "reserved": 0}
        assert scheduler.payoff(first["snapshot_id"], second["snapshot_id"])["games"] == 0
        with registry.weights_path(first["snapshot_id"]).open("ab") as f:
            f.write(b"drift")
        with pytest.raises(ValueError, match="snapshot drift"):
            registry.read(first["snapshot_id"])
    finally:
        scheduler.close()


def test_global_population_limit_cannot_rotate_through_additional_snapshots(tmp_path):
    train = learner()
    weights = tmp_path / "weights.pt"
    torch.save(train.model.state_dict(), weights)
    registry = SnapshotRegistry(tmp_path / "snapshots")
    main = registry.register(weights, identity=train.identity, policy_version=0, role="main")
    first = registry.register(weights, identity=train.identity, policy_version=0, role="historical")
    second = registry.register(
        weights, identity=train.identity, policy_version=1, role="historical"
    )
    scheduler = LeagueScheduler(
        tmp_path / "league.sqlite", registry, train.identity, LeagueConfig(7, 100, 50, 20, 2)
    )
    try:
        job = scheduler.schedule(
            main["snapshot_id"], [first["snapshot_id"]], map_name="AbyssalReef"
        )
        scheduler.claim("owner")
        scheduler.finish(
            job["job_id"], "owner", verdict="win", environment_loops=12, evidence_sha256="a" * 64
        )
        with pytest.raises(ValueError, match="global registered"):
            scheduler.schedule(main["snapshot_id"], [second["snapshot_id"]], map_name="AbyssalReef")
        with pytest.raises(ValueError, match="self-match"):
            scheduler.schedule(main["snapshot_id"], [main["snapshot_id"]], map_name="AbyssalReef")
        assert scheduler.payoff(main["snapshot_id"], first["snapshot_id"])["games"] == 1
    finally:
        scheduler.close()


def test_initial_burn_in_is_charged_and_reused_burn_in_is_paid_once():
    train = learner("VALUE_ONLY", maximum_policy_lag=2, maximum_batch_sequences=2)
    seq = sequence(train)
    # First active row follows actual fresh collection, so the preceding burn-in
    # loops must count even though they contribute no training target.
    first = replace(
        seq,
        steps=(
            replace(seq.steps[0], burn_in=True, loss=False),
            replace(seq.steps[1], truncated=True, boundary_reason="unroll_cut"),
        ),
    )
    train.update([first])
    assert train.environment_loops == 11
    # Causal replay starts at game zero; these two burn-in rows are already paid.
    tail = replace(
        seq,
        steps=(
            replace(seq.steps[0], burn_in=True, loss=False),
            replace(seq.steps[1], burn_in=True, loss=False),
            seq.steps[2],
        ),
    )
    train.update([tail])
    assert train.environment_loops == 15
    assert train.loss_intervals == {"mechanism": [(3, 15)]}
    overlapping = replace(
        seq, steps=seq.steps[1:], starts_at_episode_boundary=False, initial_hidden=torch.zeros(4)
    )
    with pytest.raises(ValueError, match="overlapping active"):
        train.update([overlapping])


def test_high_level_label_artifact_cannot_swap_under_same_training_identity(tmp_path):
    from poorstar.hierarchy import HierarchyConfig
    from poorstar.training.checkpoint import file_hash
    from poorstar.training.config import ControlSidecar

    cfg = HierarchyConfig()
    ident = identity(PPOConfig())
    evidence = tmp_path / "provenance.json"
    evidence.write_text('{"kind":"explicit-mechanism-fixture"}\n')
    document = dict(
        schema="poorstar_v3_hierarchy_bc_sidecar_v1",
        binding=dict(
            dataset_sha256=ident.dataset_sha256,
            split_sha256=ident.split_sha256,
            scope_sha256=ident.scope_sha256,
            hierarchy_config_sha256=cfg.sha256,
        ),
        provenance_evidence_path=str(evidence),
        provenance_evidence_sha256=file_hash(evidence),
        records={},
        initial_control_states={},
    )
    document["content_digest"] = digest(document)
    path = tmp_path / "controls.json"
    path.write_text(json.dumps(document))
    bound = replace(ident, control_sidecar_sha256=file_hash(path))
    ControlSidecar(path, bound, cfg)
    changed = copy.deepcopy(document)
    changed.pop("content_digest")
    changed["records"]["another"] = {}
    changed["content_digest"] = digest(changed)
    other = tmp_path / "other.json"
    other.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="frozen in training identity"):
        ControlSidecar(other, bound, cfg)
    evidence.write_text('{"kind":"changed"}\n')
    with pytest.raises(ValueError, match="provenance evidence"):
        ControlSidecar(path, bound, cfg)
