"""Tests: test training system comparators."""

import json
from copy import deepcopy
from dataclasses import asdict, replace

import pytest
import torch

from poorstar.data.bc.windows import BCRunPlan
from poorstar.hierarchy import HierarchyBCRecord, HierarchyBCSequence
from poorstar.training import PPOConfig, PPOLearner, TrainingIdentity
from poorstar.training.bc import HierarchicalBCLearner
from poorstar.training.checkpoint import load_checkpoint, save_checkpoint
from poorstar.training.config import model_config_sha256, read_model_config, specialist_override
from poorstar.training.contracts import digest
from poorstar.training.export import export_actor, load_actor
from poorstar.training.scorer import HierarchyScorer
from poorstar.training.storage import RolloutReader, RolloutWriter
from poorstar.training.windows import sequence_windows
from tests.test_coarse_bc import decision_rows, setup
from tests.test_hierarchy_runtime import model_fixture, update_control
from tests.test_training_system_foundation import sampled_sequence


@pytest.fixture(autouse=True)
def threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def comparator(kind, phase="JOINT_RL"):
    arm = "flat" if kind == "feedforward" else "persistent_intent"
    options = (
        dict(memory_mode="feedforward")
        if kind == "feedforward"
        else dict(commitment_mode="fixed_dwell", fixed_commitment_horizon=2)
    )
    model, _, profile = model_fixture(arm, **options)
    with torch.no_grad():
        model.act_head.bias.copy_(torch.tensor([30.0, -30.0]))
        model.delay_head.bias.fill_(-30.0)
        model.delay_head.bias[0] = 30.0
        model.fixed_heads["bundle"].bias.copy_(torch.tensor([-30.0, 30.0]))
    config = PPOConfig(
        maximum_updates=3,
        maximum_environment_loops=100,
        maximum_batch_sequences=3,
        maximum_sequence_steps=2,
        max_kl=0.1,
    )
    identity = TrainingIdentity(
        "comparator-mechanism",
        arm,
        71,
        "a" * 64,
        "b" * 64,
        "c" * 64,
        "d" * 64,
        model_config_sha256(model.config, model.hierarchy_config),
        profile.sha256,
        config.sha256,
    )
    learner = PPOLearner(model, HierarchyScorer(model, profile), identity, config, phase=phase)
    return learner, profile


@pytest.mark.parametrize(
    "kind,phase",
    [
        ("feedforward", "VALUE_ONLY"),
        ("feedforward", "JOINT_RL"),
        ("fixed_dwell", "MANAGER_RL"),
        ("fixed_dwell", "JOINT_RL"),
    ],
)
def test_comparator_windows_shards_export_config_and_exact_next_update(kind, phase, tmp_path):
    learner, profile = comparator(kind, phase)
    seq = sampled_sequence(learner, profile)
    if kind == "fixed_dwell":
        assert [s.manager_decision for s in seq.steps] == [True, False, True]
        assert seq.steps[1].payload.trace["scheduler_control"]["manager"] == {"operation": "keep"}
        assert set(seq.steps[2].payload.trace["scheduler_control"]["manager"]) == {
            "operation",
            "preference",
        }
    windows = sequence_windows(seq, unroll_steps=1, burn_in_steps=1)
    _, _, catalog, _ = setup()
    recovered = []
    for item in windows:
        writer = RolloutWriter(
            tmp_path / "shards",
            learner.identity,
            catalog,
            maximum_sources=63,
            maximum_delay=8,
            initial_hidden=item.initial_hidden,
            starts_at_episode_boundary=item.starts_at_episode_boundary,
        )
        for step in item.steps:
            writer.append(step)
        artifact = writer.close()
        reader = RolloutReader(
            artifact["path"], sha256=artifact["sha256"], identity=learner.identity
        )
        try:
            window = reader.sequence()
            recovered.append(window)
            hidden = window.initial_hidden
            for step in window.steps:
                score = learner.scorer(step, hidden)
                hidden = score.next_hidden
                assert float(score.log_prob) == pytest.approx(step.old_log_prob, abs=1e-5)
                assert float(score.value) == pytest.approx(step.old_value, abs=1e-6)
        finally:
            reader.close()
    first = learner.update(recovered)
    assert first["optimizer_steps"] == 1 and first["environment_loops"] == 3
    assert first["post_update"]["samples"] == (2 if phase == "MANAGER_RL" else 3)
    save_checkpoint(tmp_path / "checkpoint", learner, sampler_state={"cursor": 1})
    exported = export_actor(tmp_path / "actor.pt", learner)
    before_loading = torch.get_rng_state().clone()
    actor, actor_identity, actor_record = load_actor(exported["path"], sha256=exported["sha256"])
    assert torch.equal(torch.get_rng_state(), before_loading)
    assert (
        actor_identity == learner.identity
        and actor_record["policy_version"] == learner.policy_version
    )
    assert actor.hierarchy_config == learner.model.hierarchy_config
    config_path = tmp_path / "model.json"
    config_path.write_text(
        json.dumps(dict(bc=asdict(actor.config), hierarchy=asdict(actor.hierarchy_config)))
    )
    low, high = read_model_config(config_path)
    assert low == actor.config and high == actor.hierarchy_config
    assert model_config_sha256(low, high) == actor_identity.model_config_sha256
    expected = learner.update(
        sequence_windows(
            sampled_sequence(learner, profile, episode="next"), unroll_steps=1, burn_in_steps=1
        )
    )
    rebuilt, _ = comparator(kind, phase)
    assert load_checkpoint(tmp_path / "checkpoint", rebuilt)["sampler_state"] == {"cursor": 1}
    actual = rebuilt.update(
        sequence_windows(
            sampled_sequence(rebuilt, profile, episode="next"), unroll_steps=1, burn_in_steps=1
        )
    )
    assert actual == expected and rebuilt.parameter_sha256 == learner.parameter_sha256


@pytest.mark.parametrize("kind", ["feedforward", "fixed_dwell"])
def test_comparator_bc_uses_existing_c3_supervision_and_explicit_control_schema(kind, tmp_path):
    learner, profile = comparator(kind)
    model = learner.model
    rows, profile, *_ = decision_rows()
    if kind == "feedforward":
        records = tuple(HierarchyBCRecord(r) for r in rows)
    else:
        # Explicit mechanics-only control targets; never labeled human Intent.
        records = (
            HierarchyBCRecord(
                rows[0],
                scheduler_control=update_control(model),
                body_control={"manager": None, "tasks": None},
                body_execution_task="episode:task:1",
                supervision_id="fixed-dwell-mechanism-0",
                supervision_kind="actor_control_trace",
            ),
            HierarchyBCRecord(
                rows[1],
                scheduler_control={
                    "manager": {"operation": "update", "preference": [0.5] * 9},
                    "tasks": None,
                },
                supervision_id="fixed-dwell-mechanism-1",
                supervision_kind="actor_control_trace",
            ),
        )
    sequence = HierarchyBCSequence(records)
    plan = BCRunPlan("ENGINEERING", 71, 2, hidden_size=32, unroll_length=2, burn_in_length=0)
    identity = replace(
        learner.identity,
        budget_sha256=digest(dict(plan=asdict(plan), head_weights={}, control_weights={})),
    )
    bc = HierarchicalBCLearner(model, profile, identity, plan)
    first = bc.update([sequence])
    save_checkpoint(tmp_path / "bc", bc)
    expected = bc.update([sequence])
    rebuilt, _ = comparator(kind)
    restored = HierarchicalBCLearner(rebuilt.model, profile, identity, plan)
    load_checkpoint(tmp_path / "bc", restored)
    assert restored.update([sequence]) == expected
    assert first["denominators"]["act"] == first["denominators"]["delay"] == 2
    if kind == "fixed_dwell":
        assert first["denominators"]["manager_control"] == 2


@pytest.mark.parametrize("unsupported", ["fixed_dwell", "stateless"])
def test_specialist_external_renewal_modes_fail_before_engine(unsupported):
    learner, _ = comparator("fixed_dwell")
    config = replace(learner.model.hierarchy_config, manager_mode="external")
    if unsupported == "stateless":
        config = replace(config, commitment_mode="minimum_hold", persistence=False)
    profile = {name: 0.5 for name in config.preference_names}
    with pytest.raises(ValueError, match="renewal policy unavailable"):
        specialist_override(config, profile, "qualified-job-profile")
    supported = replace(config, commitment_mode="minimum_hold", persistence=True)
    override = specialist_override(supported, profile, "qualified-job-profile")
    assert override.preference == (0.5,) * 9 and override.intervention_id == "qualified-job-profile"
