"""Tests: test training system foundation."""

from copy import deepcopy
from dataclasses import replace

import pytest
import torch

from poorstar.actions.coarse import coarse_sample
from poorstar.hierarchy.model import HierarchicalPolicy
from poorstar.hierarchy.session import HierarchyPolicySession
from poorstar.hierarchy.state import HierarchyConfig
from poorstar.models.policy import BCModelConfig
from poorstar.training import PPOConfig, PPOLearner, RolloutSequence, RolloutStep, TrainingIdentity
from poorstar.training.contracts import digest
from poorstar.training.scorer import HierarchyScorer, payload_from_session
from poorstar.training.storage import RolloutReader, RolloutWriter
from tests.test_coarse_bc import setup


@pytest.fixture(autouse=True)
def threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def system(arm, phase="JOINT_RL"):
    f, profile, catalog, info = setup()
    cfg = BCModelConfig.from_catalog(
        catalog, profile, hidden_size=32, maximum_delay=8, position_grid=(2, 2)
    )
    hcfg = HierarchyConfig(
        arm=arm,
        manager_period=2,
        task_period=1,
        commitment_horizons=(2, 4),
        task_horizons=(1, 2),
        task_slots=2,
    )
    torch.manual_seed(71)
    model = HierarchicalPolicy(cfg, hcfg)
    model.eval()
    # Explicit mechanics-fixture probabilities, never a strategic scriptedpolicy.
    with torch.no_grad():
        model.act_head.bias.copy_(torch.tensor([30.0, -30.0]))
        model.delay_head.bias.fill_(-30.0)
        model.delay_head.bias[0] = 30.0
        model.fixed_heads["bundle"].bias.copy_(torch.tensor([-30.0, 30.0]))
    config = PPOConfig(maximum_updates=3, maximum_environment_loops=100, max_kl=0.1)
    ident = TrainingIdentity(
        "c3-mechanism",
        arm,
        71,
        "a" * 64,
        "b" * 64,
        "c" * 64,
        "d" * 64,
        digest([cfg.sha256, hcfg.sha256]),
        profile.sha256,
        config.sha256,
    )
    learner = PPOLearner(model, HierarchyScorer(model, profile), ident, config, phase=phase)
    return learner, profile, catalog


def sampled_sequence(learner, profile, *, episode="episode"):
    session = HierarchyPolicySession(learner.model, profile)
    session.reset(episode)
    steps = []
    for i in range(3):
        f, _, _, _ = setup(loop=i, episode=episode)
        memory_before = None if session.hidden is None else session.hidden.detach().clone()
        decision, delay = session.choose_schedule(f.policy_observation(), 8)
        assert decision == "ACT" and delay == 0
        machine = coarse_sample(
            f, lambda h, o, c: session.choose_body(f.policy_observation(), h, o, c), profile
        )
        assert machine.head == "done"
        payload = payload_from_session(
            session,
            f,
            f,
            dispatch_results=(1,) * len(machine.commands),
            memory_before=memory_before,
        )
        t = session.audit_trace
        steps.append(
            RolloutStep(
                episode,
                i,
                i,
                i + 1,
                learner.policy_version,
                learner.parameter_sha256,
                payload.observation_sha256,
                payload.mask_sha256,
                payload,
                t["action_log_prob"],
                t["manager_log_prob"],
                t["value"],
                t["manager_value"],
                manager_decision=t["manager_decision"],
            )
        )
    for i in range(2):
        steps[i] = replace(
            steps[i],
            next_value=steps[i + 1].old_value,
            next_manager_value=steps[i + 1].old_manager_value,
        )
    # A declared mathematical terminal fixture, not an engine outcome.
    steps[-1] = replace(
        steps[-1], terminated=True, reward=1.0, reward_loop=3, boundary_reason="engine_terminal"
    )
    return RolloutSequence(tuple(steps), learner.identity.sha256)


@pytest.mark.parametrize("arm", ["flat", "direct_task", "persistent_intent"])
def test_sampled_hierarchy_c3_joint_probability_replays_exactly(arm):
    learner, profile, _ = system(arm)
    seq = sampled_sequence(learner, profile)
    hidden = None
    for step in seq.steps:
        score = learner.scorer(step, hidden)
        hidden = score.next_hidden
        assert float(score.log_prob) == pytest.approx(step.old_log_prob, abs=1e-5)
        assert float(score.manager_log_prob) == pytest.approx(step.old_manager_log_prob, abs=1e-6)
        assert float(score.value) == pytest.approx(step.old_value, abs=1e-6)
        assert torch.isfinite(score.entropy)
    report = learner.update([seq])
    assert report["optimizer_steps"] == 1 and report["changed_parameters"]


@pytest.mark.parametrize("phase", ["VALUE_ONLY", "MANAGER_RL"])
def test_c3_stage_freezing_uses_real_hierarchy_parameters(phase):
    learner, profile, _ = system("persistent_intent", phase)
    seq = sampled_sequence(learner, profile)
    report = learner.update([seq])
    allowed = {
        name
        for group, members in learner.groups.items()
        if group in ({"value"} if phase == "VALUE_ONLY" else {"value", "manager"})
        for name, _ in members
    }
    assert set(report["changed_parameters"]) <= allowed


def test_rollout_shard_readback_retains_full_trace_and_masks(tmp_path):
    learner, profile, catalog = system("persistent_intent")
    seq = sampled_sequence(learner, profile)
    writer = RolloutWriter(tmp_path, learner.identity, catalog, maximum_sources=63, maximum_delay=8)
    for step in seq.steps:
        writer.append(step)
    artifact = writer.close()
    reader = RolloutReader(artifact["path"], sha256=artifact["sha256"], identity=learner.identity)
    try:
        recovered = reader.sequence()
        hidden = None
        for original, step in zip(seq.steps, recovered.steps):
            assert (
                step.old_log_prob == original.old_log_prob
                and step.mask_sha256 == original.mask_sha256
            )
            score = learner.scorer(step, hidden)
            hidden = score.next_hidden
            assert float(score.log_prob) == pytest.approx(original.old_log_prob, abs=1e-5)
    finally:
        reader.close()


def test_action_token_domain_mutation_cannot_train():
    learner, profile, _ = system("persistent_intent")
    seq = sampled_sequence(learner, profile)
    trace = deepcopy(seq.steps[0].payload.trace)
    trace["body_tokens"][0]["options"] = (999999,)
    payload = replace(seq.steps[0].payload, trace=trace)
    bad = replace(seq.steps[0], payload=payload, mask_sha256=payload.mask_sha256)
    with pytest.raises(ValueError, match="mask mismatch"):
        learner.update([replace(seq, steps=(bad,) + seq.steps[1:])])


def test_external_override_has_no_manager_density_or_manager_training_event():
    from poorstar.hierarchy.state import ExternalOverride

    learner, profile, _ = system("persistent_intent")
    f, _, _, _ = setup()
    session = HierarchyPolicySession(learner.model, profile)
    session.reset("episode")
    session.intervene(ExternalOverride((0.5,) * 9, 2, 0.5, "explicit-mechanism-intervention"))
    session.choose_schedule(f.policy_observation(), 8)
    coarse_sample(f, lambda h, o, c: session.choose_body(f.policy_observation(), h, o, c), profile)
    assert (
        session.audit_trace["manager_decision"] is False
        and session.audit_trace["manager_log_prob"] == 0
    )
    payload = payload_from_session(session, f, f)
    step = RolloutStep(
        "episode",
        0,
        0,
        1,
        0,
        learner.parameter_sha256,
        payload.observation_sha256,
        payload.mask_sha256,
        payload,
        old_log_prob=session.action_log_prob,
    )
    score = learner.scorer(step)
    assert float(score.manager_log_prob) == 0 and float(score.log_prob) == pytest.approx(
        session.action_log_prob, abs=1e-5
    )


def test_midgame_burn_in_windows_roundtrip_and_bounded_logical_update(tmp_path):
    from poorstar.training.windows import sequence_windows

    learner, profile, catalog = system("persistent_intent")
    learner.config = replace(learner.config, maximum_batch_sequences=3, maximum_sequence_steps=2)
    # Rebind the engineering identity to these explicit capacity limits.
    learner.identity = replace(learner.identity, budget_sha256=learner.config.sha256)
    seq = sampled_sequence(learner, profile)
    windows = sequence_windows(seq, unroll_steps=1, burn_in_steps=1)
    assert [len(w.steps) for w in windows] == [1, 2, 2]
    assert windows[0].steps[-1].boundary_reason == "unroll_cut"
    assert not windows[-1].starts_at_episode_boundary and windows[-1].initial_hidden is not None
    assert windows[-1].steps[0].burn_in and not windows[-1].steps[0].loss
    recovered = []
    for window in windows:
        writer = RolloutWriter(
            tmp_path,
            learner.identity,
            catalog,
            maximum_sources=63,
            maximum_delay=8,
            initial_hidden=window.initial_hidden,
            starts_at_episode_boundary=window.starts_at_episode_boundary,
        )
        for step in window.steps:
            writer.append(step)
        artifact = writer.close()
        reader = RolloutReader(
            artifact["path"], sha256=artifact["sha256"], identity=learner.identity
        )
        try:
            item = reader.sequence()
            recovered.append(item)
            hidden = item.initial_hidden
            for step in item.steps:
                scored = learner.scorer(step, hidden)
                hidden = scored.next_hidden
                assert float(scored.log_prob) == pytest.approx(step.old_log_prob, abs=1e-5)
                assert float(scored.value) == pytest.approx(step.old_value, abs=1e-6)
        finally:
            reader.close()
    report = learner.update(recovered)
    assert report["optimizer_steps"] == 1 and report["environment_loops"] == 3
    assert report["post_update"]["samples"] == 3
    assert learner.loss_intervals == {"episode": [(0, 3)]}


@pytest.mark.parametrize("arm", ["flat", "direct_task", "persistent_intent"])
def test_actual_bc_optimizer_value_freeze_and_exact_recovery(arm, tmp_path):
    from dataclasses import asdict

    from poorstar.data.bc.windows import BCRunPlan
    from poorstar.hierarchy import HierarchyBCRecord, HierarchyBCSequence
    from poorstar.training.bc import HierarchicalBCLearner
    from poorstar.training.checkpoint import load_checkpoint, save_checkpoint
    from poorstar.training.config import model_config_sha256
    from tests.test_coarse_bc import decision_rows
    from tests.test_hierarchy_bc import records_fixture
    from tests.test_hierarchy_runtime import model_fixture

    model, _, _ = model_fixture(arm)
    if arm == "flat":
        rows, profile, *_ = decision_rows()
        records = tuple(HierarchyBCRecord(r) for r in rows)
    else:
        records, profile = records_fixture(model)
    seq = HierarchyBCSequence(records)
    plan = BCRunPlan(
        "ENGINEERING",
        71,
        3,
        hidden_size=model.config.hidden_size,
        autoregressive=model.config.autoregressive,
        unroll_length=2,
        burn_in_length=0,
    )
    budget = digest(dict(plan=asdict(plan), head_weights={}, control_weights={}))
    ident = TrainingIdentity(
        "bc-mechanism",
        arm,
        71,
        "a" * 64,
        "b" * 64,
        "c" * 64,
        "d" * 64,
        model_config_sha256(model.config, model.hierarchy_config),
        profile.sha256,
        budget,
    )
    learner = HierarchicalBCLearner(model, profile, ident, plan)
    frozen = {n: p.detach().clone() for n, p in learner.groups["value"]}
    before = learner.parameter_sha256
    first = learner.update([seq])
    assert learner.parameter_sha256 != before and first["optimizer_steps"] == 1
    assert all(torch.equal(v, dict(model.named_parameters())[n]) for n, v in frozen.items())
    save_checkpoint(tmp_path / "bc", learner, sampler_state={"cursor": 1})
    expected = learner.update([seq])
    restored_model, _, _ = model_fixture(arm)
    restored = HierarchicalBCLearner(restored_model, profile, ident, plan)
    assert load_checkpoint(tmp_path / "bc", restored)["sampler_state"] == {"cursor": 1}
    actual = restored.update([seq])
    assert actual == expected and restored.parameter_sha256 == learner.parameter_sha256


def test_bc_plan_seed_scheduler_and_vocabulary_drift_fail_before_adam():
    from dataclasses import asdict

    from poorstar.data.bc.windows import BCRunPlan
    from poorstar.training.bc import validate_bc_contract

    learner, profile, catalog = system("flat")
    plan = BCRunPlan("ENGINEERING", 71, 2, hidden_size=32)
    identity = replace(
        learner.identity,
        budget_sha256=digest(dict(plan=asdict(plan), head_weights={}, control_weights={})),
    )
    validate_bc_contract(learner.model, profile, identity, plan, catalog=catalog, maximum_delay=8)
    with pytest.raises(ValueError, match="seed"):
        validate_bc_contract(learner.model, profile, replace(identity, seed=72), plan)
    with pytest.raises(ValueError, match="horizon"):
        validate_bc_contract(learner.model, profile, identity, plan, maximum_delay=32)
    bad_catalog = deepcopy(catalog)
    bad_catalog.units.add(unit_id=999)
    with pytest.raises(ValueError, match="vocabulary"):
        validate_bc_contract(learner.model, profile, identity, plan, catalog=bad_catalog)


def test_forced_commitment_keep_is_recorded_without_resetting_manager_return_span():
    from poorstar.training.config import model_config_sha256
    from poorstar.training.returns import targets

    learner, profile, _ = system("persistent_intent", "MANAGER_RL")
    learner.model.hierarchy_config = replace(learner.model.hierarchy_config, manager_period=1)
    with torch.no_grad():
        learner.model.manager_horizon_head.bias.copy_(torch.tensor([-30.0, 30.0]))
        learner.model.task_operation_head.bias.copy_(torch.tensor([30.0, -30.0, -30.0]))
    learner.identity = replace(
        learner.identity,
        model_config_sha256=model_config_sha256(
            learner.model.config, learner.model.hierarchy_config
        ),
    )
    learner.behavior_registry = {0: learner.parameter_sha256}
    sequence = sampled_sequence(learner, profile)
    assert sequence.steps[0].manager_decision
    assert all(not s.manager_decision for s in sequence.steps[1:])
    assert all(
        s.payload.trace["scheduler_control"]["manager"] == {"operation": "keep"}
        for s in sequence.steps[1:]
    )
    assert targets(sequence, learner.config, manager=True).indices == (0,)
    score = learner.scorer(sequence.steps[0])
    hidden = score.next_hidden
    for step in sequence.steps[1:]:
        score = learner.scorer(step, hidden)
        hidden = score.next_hidden
        assert float(score.manager_log_prob) == 0.0
        assert float(score.log_prob) == pytest.approx(step.old_log_prob, abs=1e-5)
    result = learner.update([sequence])
    assert result["post_update"]["samples"] == 1
