"""Tests: test training system outcome."""

import json
from copy import deepcopy
from dataclasses import asdict, replace

import pytest
import torch

from poorstar.training import PPOConfig, PPOLearner, RolloutSequence
from poorstar.training.checkpoint import capture_rng, file_hash, load_checkpoint, save_checkpoint
from poorstar.training.contracts import digest
from poorstar.training.export import export_actor, load_actor
from poorstar.training.publication import publish_candidate, publish_candidate_index
from poorstar.training.returns import group_targets, targets
from poorstar.training.stages import (
    cumulative_counters,
    initialize_stage,
    read_transition,
    transition_document,
    validate_transition,
)
from poorstar.training.windows import sequence_windows
from tests.test_coarse_bc import decision_rows
from tests.test_hierarchy_runtime import update_control
from tests.test_training_system_comparators import comparator
from tests.test_training_system_foundation import sampled_sequence, system


@pytest.fixture(autouse=True)
def threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def outcome_system(phase="JOINT_RL", *, normalized=True):
    old, profile, catalog = system("persistent_intent", phase)
    config = replace(
        old.config,
        weight_decay=0.0,
        value_coefficient=0.0,
        entropy_coefficient=0.0,
        learning_rate=1e-5,
        maximum_batch_sequences=6,
        maximum_sequence_steps=2,
        normalize_advantage=normalized,
        gamma_per_loop=0.97,
        lambda_per_loop=1.0,
    )
    identity = replace(old.identity, budget_sha256=config.sha256)
    learner = PPOLearner(old.model, old.scorer, identity, config, phase=phase)
    return learner, profile, catalog


def outcome_windows(learner, profile, *, prefix="math-outcome"):
    result = []
    physical = []
    for label, reward in (("positive", 1.0), ("negative", -1.0)):
        sequence = sampled_sequence(learner, profile, episode=prefix + ":" + label)
        sequence = replace(
            sequence, steps=sequence.steps[:-1] + (replace(sequence.steps[-1], reward=reward),)
        )
        physical.append(sequence)
        result.extend(sequence_windows(sequence, unroll_steps=1, burn_in_steps=1))
    return tuple(result), tuple(physical)


@pytest.mark.parametrize("phase", ["VALUE_ONLY", "MANAGER_RL", "JOINT_RL"])
def test_actual_c3_multi_outcome_policy_only_gradient_excludes_critic_entropy_decay(phase):
    learner, profile, _ = outcome_system(phase)
    windows, _ = outcome_windows(learner, profile)
    before = deepcopy(learner.model.state_dict())
    rng = capture_rng()
    audit = learner.policy_gradient_audit(windows)
    assert (
        not audit["critic_included"]
        and not audit["entropy_included"]
        and not audit["weight_decay_included"]
    )
    assert learner.optimizer_steps == audit["optimizer_steps"] == 0 and not learner.optimizer.state
    assert torch.equal(rng["torch"], capture_rng()["torch"])
    assert all(
        torch.equal(value, learner.model.state_dict()[name]) for name, value in before.items()
    )
    norms = audit["group_l2_norms"]
    if phase == "VALUE_ONLY":
        assert (
            all(value == 0 for value in norms.values())
            and audit["effective_advantage_nonzero"] == 0
        )
    else:
        assert audit["effective_advantage_nonzero"] >= 2 and audit["effective_advantage_std"] > 0
        assert norms["manager"] > 0 and norms["value"] == 0
        if phase == "MANAGER_RL":
            assert all(norms[name] == 0 for name in ("backbone", "decoder", "task"))
        else:
            assert all(norms[name] > 0 for name in ("backbone", "decoder", "task"))
    update = learner.update(windows)
    changed = {name.split(".")[0] for name in update["changed_parameters"]}
    assert changed and update["environment_loops"] == 6
    if phase != "VALUE_ONLY":
        # With value/entropy/decay all zero, these actual weight changes can only
        # arise from policy-loss gradients, not the former canary's confounders.
        assert any(name.startswith("manager_") for name in update["changed_parameters"])
        assert not any(
            name.startswith(("value_head.", "manager_value_head.", "value_context."))
            for name in update["changed_parameters"]
        )


@pytest.mark.parametrize("manager", [False, True])
def test_cross_unroll_outcome_returns_equal_uncut_episode_targets(manager):
    learner, profile, _ = outcome_system("MANAGER_RL" if manager else "JOINT_RL")
    windows, physical = outcome_windows(learner, profile)
    grouped = group_targets(windows, learner.config, manager=manager)
    by_episode = {}
    for window, result in zip(windows, grouped):
        for j, local in enumerate(result.indices):
            step = window.steps[local]
            by_episode.setdefault(step.episode_id, []).append(
                (step.index, result.returns[j], result.advantages[j])
            )
    for original in physical:
        expected = targets(original, learner.config, manager=manager)
        actual = sorted(by_episode[original.steps[0].episode_id])
        assert tuple(v[0] for v in actual) == expected.indices
        assert torch.equal(torch.stack([v[1] for v in actual]), expected.returns)
        assert torch.equal(torch.stack([v[2] for v in actual]), expected.advantages)
        assert torch.allclose(
            expected.returns,
            torch.tensor(
                [
                    original.steps[-1].reward
                    * learner.config.gamma_per_loop ** (3 - original.steps[i].game_loop)
                    for i in expected.indices
                ]
            ),
            atol=1e-6,
        )


def test_cross_unroll_keeps_true_truncation_bootstrap_and_rejects_gaps_duplicates():
    learner, profile, _ = outcome_system()
    sequence = sampled_sequence(learner, profile)
    sequence = replace(
        sequence,
        steps=sequence.steps[:-1]
        + (
            replace(
                sequence.steps[-1],
                terminated=False,
                truncated=True,
                reward=0.0,
                reward_loop=None,
                boundary_reason="resident_step_cap",
                next_value=0.25,
                next_manager_value=0.5,
            ),
        ),
    )
    windows = sequence_windows(sequence, unroll_steps=1, burn_in_steps=0)
    grouped = group_targets(windows, learner.config, manager=True)
    expected = targets(sequence, learner.config, manager=True)
    actual = torch.cat([item.returns for item in grouped])
    assert torch.equal(actual, expected.returns)
    with pytest.raises(ValueError, match="gap"):
        group_targets((windows[0], windows[2]), learner.config)
    with pytest.raises(ValueError, match="duplicate active"):
        group_targets((windows[0], windows[0]), learner.config)
    bad = replace(windows[1].steps[0], old_value=windows[1].steps[0].old_value + 0.2)
    with pytest.raises(ValueError, match="bootstrap"):
        group_targets((windows[0], replace(windows[1], steps=(bad,)), windows[2]), learner.config)


def test_single_normalized_advantage_truthfully_reports_zero_policy_gradient():
    learner, profile, _ = outcome_system()
    sequence = sampled_sequence(learner, profile)
    one = sequence_windows(sequence, unroll_steps=1, burn_in_steps=0)[0]
    audit = learner.policy_gradient_audit((one,))
    assert audit["samples"] == 1 and audit["effective_advantage_nonzero"] == 0
    assert all(value == 0 for value in audit["group_l2_norms"].values())


@pytest.mark.parametrize("phase", ["VALUE_ONLY", "MANAGER_RL", "JOINT_RL"])
def test_nonzero_multi_outcome_update_and_next_sample_exact_checkpoint_restore(phase, tmp_path):
    learner, profile, _ = outcome_system(phase)
    windows, _ = outcome_windows(learner, profile)
    first = learner.update(windows)
    save_checkpoint(tmp_path / "checkpoint", learner, sampler_state={"logical_group": 1})
    next_windows, _ = outcome_windows(learner, profile, prefix="next")
    expected_audit = learner.policy_gradient_audit(next_windows)
    expected = learner.update(next_windows)
    rebuilt, profile, _ = outcome_system(phase)
    recovered = load_checkpoint(tmp_path / "checkpoint", rebuilt)
    actual_windows, _ = outcome_windows(rebuilt, profile, prefix="next")
    assert rebuilt.policy_gradient_audit(actual_windows) == expected_audit
    assert rebuilt.update(actual_windows) == expected
    assert rebuilt.parameter_sha256 == learner.parameter_sha256
    assert recovered["sampler_state"] == {"logical_group": 1}
    assert first["optimizer_steps"] == 1 and rebuilt.optimizer_steps == 2


def make_bc_predecessor(tmp_path, *, frozen_inputs=None):
    from poorstar.data.bc.windows import BCRunPlan
    from poorstar.hierarchy import HierarchyBCRecord, HierarchyBCSequence
    from poorstar.training.bc import HierarchicalBCLearner

    prior, profile = comparator("fixed_dwell")
    rows, profile, *_ = decision_rows()
    records = (
        HierarchyBCRecord(
            rows[0],
            scheduler_control=update_control(prior.model),
            body_control={"manager": None, "tasks": None},
            body_execution_task="episode:task:1",
            supervision_id="mechanism-only-0",
            supervision_kind="actor_control_trace",
        ),
        HierarchyBCRecord(
            rows[1],
            scheduler_control={
                "manager": {"operation": "update", "preference": [0.5] * 9},
                "tasks": None,
            },
            supervision_id="mechanism-only-1",
            supervision_kind="actor_control_trace",
        ),
    )
    plan = BCRunPlan("ENGINEERING", 71, 1, hidden_size=32, unroll_length=2, burn_in_length=0)
    identity = replace(
        prior.identity,
        budget_sha256=digest(dict(plan=asdict(plan), head_weights={}, control_weights={})),
        **({} if frozen_inputs is None else frozen_inputs),
    )
    bc = HierarchicalBCLearner(prior.model, profile, identity, plan)
    bc.update([HierarchyBCSequence(records)])
    actor = export_actor(tmp_path / "bc.pt", bc)
    model, identity, payload = load_actor(actor["path"], sha256=actor["sha256"])
    return model, identity, payload, actor, profile


def next_stage(tmp_path, model, prior_identity, prior_actor, artifact, phase, *, suffix=""):
    from poorstar.training.scorer import HierarchyScorer

    config = PPOConfig(
        learning_rate=1e-5,
        maximum_updates=3,
        maximum_environment_loops=100,
        maximum_batch_sequences=3,
        maximum_sequence_steps=2,
        max_kl=0.1,
        weight_decay=0.0,
    )
    identity = replace(
        prior_identity,
        budget_sha256=config.sha256,
        initializer_sha256=artifact["sha256"],
        stage_transition_sha256="0" * 64,
    )
    document = transition_document(
        identity,
        phase,
        prior_identity,
        prior_actor,
        artifact["sha256"],
        config,
        manager_available=bool(model.parameter_groups()["manager"]),
    )
    path = tmp_path / (phase + suffix + ".transition.json")
    path.write_text(json.dumps(document, indent=2) + "\n")
    identity = replace(identity, stage_transition_sha256=file_hash(path))
    read = read_transition(path, identity, phase, config)
    validate_transition(
        read,
        identity,
        phase,
        prior_identity,
        prior_actor,
        artifact["sha256"],
        config,
        manager_available=bool(model.parameter_groups()["manager"]),
    )
    learner = PPOLearner(
        model, HierarchyScorer(model, model._test_profile), identity, config, phase=phase
    )
    initialize_stage(learner, read)
    return learner, read


def test_real_model_bc_value_manager_joint_phase_transfer_freezing_and_cumulative_budget(tmp_path):
    model, prior_identity, prior_actor, artifact, profile = make_bc_predecessor(tmp_path)
    model._test_profile = profile
    previous_version = prior_actor["policy_version"]
    for phase in ("VALUE_ONLY", "MANAGER_RL", "JOINT_RL"):
        learner, document = next_stage(
            tmp_path, model, prior_identity, prior_actor, artifact, phase
        )
        assert not learner.optimizer.state and learner.updates == learner.environment_loops == 0
        assert learner.policy_version == previous_version
        allowed = (
            {"value"}
            if phase == "VALUE_ONLY"
            else {"manager", "value"}
            if phase == "MANAGER_RL"
            else set(learner.groups)
        )
        for group, params in learner.groups.items():
            assert all(parameter.requires_grad == (group in allowed) for _, parameter in params)
        seq = sampled_sequence(learner, profile, episode=phase)
        update = learner.update(sequence_windows(seq, unroll_steps=1, burn_in_steps=1))
        assert update["policy_version"] == previous_version + 1
        actual_totals = cumulative_counters(learner)
        assert (
            actual_totals["optimizer_steps"]
            == 1 + ("VALUE_ONLY", "MANAGER_RL", "JOINT_RL").index(phase) + 1
        )
        artifact = export_actor(tmp_path / (phase + ".pt"), learner)
        model, prior_identity, prior_actor = load_actor(artifact["path"], sha256=artifact["sha256"])
        model._test_profile = profile
        previous_version = prior_actor["policy_version"]
        assert (
            prior_actor["training_phase"] == phase
            and prior_actor["cumulative_counters"] == actual_totals
        )
        assert document["fresh_optimizer"] and document["capability_inferred"] is False
    assert prior_actor["cumulative_counters"] == {"optimizer_steps": 4, "environment_loops": 9}


def test_stage_transfer_rejects_cross_experiment_skipped_phase_unbound_budget_and_counter_drift(
    tmp_path,
):
    model, identity, payload, artifact, profile = make_bc_predecessor(tmp_path)
    model._test_profile = profile
    config = PPOConfig()
    target = replace(identity, initializer_sha256=artifact["sha256"], budget_sha256=config.sha256)
    with pytest.raises(ValueError, match="skipped"):
        transition_document(
            target,
            "MANAGER_RL",
            identity,
            payload,
            artifact["sha256"],
            config,
            manager_available=True,
        )
    with pytest.raises(ValueError, match="experiment"):
        transition_document(
            replace(target, experiment="other"),
            "VALUE_ONLY",
            identity,
            payload,
            artifact["sha256"],
            config,
            manager_available=True,
        )
    value, document = next_stage(tmp_path, model, identity, payload, artifact, "VALUE_ONLY")
    changed = deepcopy(document)
    changed["stage_budget"]["optimizer_steps"] += 1
    with pytest.raises(ValueError, match="predecessor"):
        validate_transition(
            changed,
            value.identity,
            "VALUE_ONLY",
            identity,
            payload,
            artifact["sha256"],
            value.config,
            manager_available=True,
        )
    state = value.state_dict()
    state["policy_version"] += 1
    with pytest.raises(ValueError, match="counter drift"):
        value.load_state_dict(state)


def test_checkpoint_candidate_publication_contains_full_identity_and_never_selects(tmp_path):
    learner, profile, _ = outcome_system()
    windows, _ = outcome_windows(learner, profile)
    learner.update(windows)
    item = publish_candidate(tmp_path, learner, sampler_state={"cursor": 1})
    result = publish_candidate_index(
        tmp_path / "index.json", learner.identity, learner.phase, [item]
    )
    document = json.loads((tmp_path / "index.json").read_text())
    candidate = document["candidates"][0]["candidate"]
    assert (
        candidate["identity"] == asdict(learner.identity)
        and not candidate["evaluated"]
        and not candidate["selected"]
    )
    assert candidate["actor"]["sha256"] == file_hash(candidate["actor"]["path"])
    assert result["sha256"] == file_hash(tmp_path / "index.json")
    with pytest.raises(ValueError, match="unique"):
        publish_candidate_index(
            tmp_path / "bad.json", learner.identity, learner.phase, [item, item]
        )


def test_static_real_frame_qualification_is_explicit_counterfactual_and_has_nonzero_policy_gradients(
    tmp_path,
):
    from poorstar.training.qualification import qualify_static_outcome_gradients

    rows, profile, catalog, _ = decision_rows()
    model, _, _ = system("persistent_intent")
    reports = qualify_static_outcome_gradients(
        rows,
        profile,
        model.model.config,
        device="cpu",
        output_dir=tmp_path,
        source_sha256="a" * 64,
        dataset_sha256="b" * 64,
        split_sha256="c" * 64,
    )
    assert [report["phase"] for report in reports] == ["VALUE_ONLY", "MANAGER_RL", "JOINT_RL"]
    for report in reports:
        assert report["status"] == "PASS" and report["exact_next_sample_update"]
        assert report["synthetic_rewards"] == [1.0, -1.0] and not report["actual_engine_outcomes"]
        assert not report["live_on_policy_qualified"] and report["new_environment_frames"] == 0
        assert report["dispatched_commands"] == 0 and not report["human_high_level_labels"]
    assert reports[1]["policy_only_gradient_audit"]["group_l2_norms"]["manager"] > 0
    assert reports[2]["policy_only_gradient_audit"]["group_l2_norms"]["decoder"] > 0


def test_semantically_invalid_checkpoint_rolls_back_model_optimizer_counters_and_rng(tmp_path):
    learner, profile, _ = outcome_system()
    windows, _ = outcome_windows(learner, profile)
    learner.update(windows)
    save_checkpoint(tmp_path / "state", learner)
    path = tmp_path / "state" / "state.pt"
    payload = torch.load(path, weights_only=False)
    version = payload["learner"]["policy_version"]
    payload["learner"]["behavior_registry"][version] = "0" * 64
    torch.save(payload, path)
    manifest_path = tmp_path / "state" / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest.pop("content_digest")
    manifest["state_sha256"] = file_hash(path)
    manifest["content_digest"] = digest(manifest)
    manifest_path.write_text(json.dumps(manifest))
    restored, _, _ = outcome_system()
    before, rng = restored.parameter_sha256, capture_rng()
    with pytest.raises(ValueError, match="current policy identity"):
        load_checkpoint(tmp_path / "state", restored)
    assert restored.parameter_sha256 == before and not restored.optimizer.state
    assert (
        restored.updates
        == restored.optimizer_steps
        == restored.policy_version
        == restored.environment_loops
        == 0
    )
    assert torch.equal(rng["torch"], capture_rng()["torch"])


@pytest.mark.parametrize("arm", ["flat", "direct_task", "persistent_intent"])
def test_named_foundation_migration_uses_original_c3_checkpoint_binding_without_rng_or_upper_change(
    arm, tmp_path
):
    from poorstar.data.bc.windows import BCRunPlan, WindowOrder, freeze_binding
    from poorstar.data.bc.windows import save_checkpoint as save_c3
    from poorstar.models.policy import RecurrentBCPolicy
    from poorstar.training.bc import load_foundation_initialization
    from poorstar.training.foundation import BCTrainer

    destination, profile, _ = system(arm)
    source = RecurrentBCPolicy(destination.model.config)
    trainer = BCTrainer(source, profile, learning_rate=1e-4)
    rows, *_ = decision_rows()
    trainer.step([rows])
    plan = BCRunPlan("ENGINEERING", 71, 1, hidden_size=32)
    binding = freeze_binding(trainer, plan, source_sha256="e" * 64, dataset_sha256="b" * 64)
    report = save_c3(tmp_path / "foundation", trainer, WindowOrder(1, 71), binding)
    old = deepcopy(destination.model.state_dict())
    rng = capture_rng()
    result = load_foundation_initialization(
        tmp_path / "foundation" / "state.pt",
        destination.model,
        expected_sha256=report["state_sha256"],
        source_report=tmp_path / "foundation" / "report.json",
        expected_binding=binding,
    )
    assert torch.equal(rng["torch"], capture_rng()["torch"])
    assert all(
        torch.equal(destination.model.state_dict()[n], p) for n, p in source.state_dict().items()
    )
    assert all(
        torch.equal(destination.model.state_dict()[n], old[n])
        for n in result["new_upper_parameters"]
    )
    before = destination.parameter_sha256
    with pytest.raises(ValueError, match="configuration/report"):
        load_foundation_initialization(
            tmp_path / "foundation" / "state.pt",
            destination.model,
            expected_sha256=report["state_sha256"],
            source_report=tmp_path / "foundation" / "report.json",
            expected_binding=binding | {"profile_sha256": "f" * 64},
        )
    assert destination.parameter_sha256 == before


@pytest.mark.parametrize("phase", ["VALUE_ONLY", "MANAGER_RL", "JOINT_RL"])
def test_explicit_stage_transition_exact_resume_preserves_inherited_version_and_cumulative_budgets(
    phase, tmp_path
):
    from poorstar.training.scorer import HierarchyScorer

    model, identity, actor, artifact, profile = make_bc_predecessor(tmp_path)
    model._test_profile = profile
    for stage in ("VALUE_ONLY", "MANAGER_RL", "JOINT_RL"):
        learner, transition = next_stage(tmp_path, model, identity, actor, artifact, stage)
        first = sampled_sequence(learner, profile, episode=stage + ":before")
        learner.update(sequence_windows(first, unroll_steps=1, burn_in_steps=1))
        if stage == phase:
            save_checkpoint(tmp_path / "resume", learner, sampler_state={"groups": 1})
            following = sampled_sequence(learner, profile, episode=stage + ":after")
            expected = learner.update(sequence_windows(following, unroll_steps=1, burn_in_steps=1))
            with torch.random.fork_rng(devices=[]):
                rebuilt_model = type(model)(model.config, model.hierarchy_config)
            rebuilt = PPOLearner(
                rebuilt_model,
                HierarchyScorer(rebuilt_model, profile),
                learner.identity,
                learner.config,
                phase=phase,
            )
            rebuilt.stage_transition = deepcopy(transition)
            loaded = load_checkpoint(tmp_path / "resume", rebuilt)
            sampled = sampled_sequence(rebuilt, profile, episode=stage + ":after")
            actual = rebuilt.update(sequence_windows(sampled, unroll_steps=1, burn_in_steps=1))
            assert expected == actual and rebuilt.parameter_sha256 == learner.parameter_sha256
            assert cumulative_counters(rebuilt) == cumulative_counters(learner)
            assert rebuilt.policy_version == transition["predecessor_policy_version"] + 2
            assert loaded["sampler_state"] == {"groups": 1}
            break
        artifact = export_actor(tmp_path / (stage + "-prior.pt"), learner)
        model, identity, actor = load_actor(artifact["path"], sha256=artifact["sha256"])
        model._test_profile = profile


def test_real_cli_initializes_stage_zero_then_consumes_only_fresh_behavior_and_exact_empty_prefix(
    tmp_path,
):
    """Actual CLI/Adam/storage on protocol fixture; no SC2 or formal execution."""
    import os
    import subprocess
    import sys
    from pathlib import Path

    from poorstar.data.bc.records import BCShardWriter, freeze_split
    from poorstar.data.bc.windows import BCWindowDataset
    from poorstar.league.ownership import read_document
    from poorstar.training.storage import RolloutWriter

    tree = Path(__file__).resolve().parents[1]
    cli = tree / "poorstar/cli/training/train.py"
    rows, profile, catalog, info = decision_rows()
    split = freeze_split([dict(replay_sha256=rows[0].replay_sha256, match_group="match")], seed=1)
    assert split["assignments"][rows[0].replay_sha256] == "train"
    split_path = tmp_path / "split.json"
    split_path.write_text(json.dumps(split))
    writer = BCShardWriter(
        tmp_path / "bc_rows" / "train",
        split_manifest=split,
        split="train",
        catalog=catalog,
        game_info=info,
        source_manifest_sha256="b" * 64,
        extractor_sha256="c" * 64,
        maximum_delay=8,
    )
    for row in rows:
        writer.append(row)
    artifact = writer.close()
    build = dict(
        schema="poorstar_c3_fixture_build_v1", split_sha256=split["sha256"], shards=[artifact]
    )
    build["content_digest"] = digest(build)
    build_path = tmp_path / "build.json"
    build_path.write_text(json.dumps(build))
    data = BCWindowDataset(build_path, split_path, unroll_length=2, burn_in_length=0)
    dataset_sha = data.sha256
    data.close()
    source_path = tmp_path / "mechanism_source.json"
    source_path.write_text(
        json.dumps(
            dict(
                files=[
                    dict(
                        path=str(cli.relative_to(tree)),
                        sha256=file_hash(cli),
                        bytes=cli.stat().st_size,
                    )
                ],
                scope="CLI_PROTOCOL_FIXTURE_SUBSET_NOT_RELEASE_QUALIFICATION",
            )
        )
    )
    model, prior_identity, prior_actor, prior_artifact, profile = make_bc_predecessor(
        tmp_path,
        frozen_inputs=dict(
            dataset_sha256=dataset_sha,
            split_sha256=split["sha256"],
            source_sha256=file_hash(source_path),
        ),
    )
    model._test_profile = profile
    learner, transition = next_stage(
        tmp_path, model, prior_identity, prior_actor, prior_artifact, "VALUE_ONLY"
    )
    identity_path = tmp_path / "identity.json"
    identity_path.write_text(json.dumps(asdict(learner.identity)))
    config_path = tmp_path / "model_config.json"
    config_path.write_text(
        json.dumps(dict(bc=asdict(model.config), hierarchy=asdict(model.hierarchy_config)))
    )
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(asdict(learner.config)))
    base = [
        sys.executable,
        str(cli),
        "--stage",
        "VALUE_ONLY",
        "--identity",
        str(identity_path),
        "--model-config",
        str(config_path),
        "--plan",
        str(plan_path),
        "--source-manifest",
        str(source_path),
        "--stage-transition",
        str(tmp_path / "VALUE_ONLY.transition.json"),
        "--device",
        "cpu",
    ]
    env = os.environ | {"PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION": "python"}
    boot = subprocess.run(
        base
        + [
            "--initialize-stage-only",
            "--initial-actor",
            prior_artifact["path"],
            "--initial-actor-sha256",
            prior_artifact["sha256"],
            "--build-report",
            str(build_path),
            "--split-manifest",
            str(split_path),
            "--output-dir",
            str(tmp_path / "bootstrap"),
        ],
        cwd=tree,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert boot.returncode == 0, boot.stdout + boot.stderr
    report = read_document(tmp_path / "bootstrap" / "report.json")
    assert (
        report["updates"]
        == report["optimizer_steps"]
        == report["optimizer_moments"]
        == report["environment_loops"]
        == 0
    )
    assert (
        not report["engine_started"] and report["policy_version"] == prior_actor["policy_version"]
    )
    empty = read_document(report["accumulated_rollout_index"]["path"])
    assert (
        empty["shards"] == empty["update_groups"] == [] and empty["physical_environment_loops"] == 0
    )
    stage_model, stage_identity, stage_actor = load_actor(
        report["actor"]["path"], sha256=report["actor"]["sha256"]
    )
    assert stage_identity == learner.identity and stage_actor["training_phase"] == "VALUE_ONLY"
    assert stage_actor["parameter_sha256"] == prior_actor["parameter_sha256"]
    state = torch.load(
        tmp_path / "bootstrap" / "checkpoint_00000000" / "state.pt", weights_only=False
    )
    assert (
        not state["learner"]["optimizer"]["state"]
        and state["sampler_state"]["consumed_groups"] == []
    )
    # Fresh target-stage trajectories share exactly this now-published version.
    from poorstar.training.scorer import HierarchyScorer

    behavior = PPOLearner(
        stage_model,
        HierarchyScorer(stage_model, profile),
        stage_identity,
        learner.config,
        phase="VALUE_ONLY",
    )
    behavior.stage_transition = transition
    behavior.policy_version = stage_actor["policy_version"]
    physical = sampled_sequence(behavior, profile, episode="STAGE_ZERO_PROTOCOL_FIXTURE")
    shards = []
    for window in sequence_windows(physical, unroll_steps=1, burn_in_steps=1):
        out = RolloutWriter(
            tmp_path / "fresh_rollouts",
            stage_identity,
            catalog,
            maximum_sources=63,
            maximum_delay=8,
            initial_hidden=window.initial_hidden,
            starts_at_episode_boundary=window.starts_at_episode_boundary,
        )
        for step in window.steps:
            out.append(step)
        shards.append(out.close())
    index = dict(
        schema="poorstar_v3_rollout_index_v1",
        identity_sha256=stage_identity.sha256,
        shards=shards,
        update_groups=[[s["sha256"] for s in shards]],
        physical_environment_loops=3,
    )
    index["content_digest"] = digest(index)
    index_path = tmp_path / "fresh_index.json"
    index_path.write_text(json.dumps(index))
    run = subprocess.run(
        base
        + [
            "--resume-from",
            str(tmp_path / "bootstrap" / "checkpoint_00000000"),
            "--rollout-index",
            str(index_path),
            "--output-dir",
            str(tmp_path / "continued"),
        ],
        cwd=tree,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert run.returncode == 0, run.stdout + run.stderr
    continued = read_document(tmp_path / "continued" / "report.json")
    assert (
        continued["updates"] == continued["optimizer_steps"] == 1
        and continued["waiting_for_rollouts"]
    )
    assert continued["cumulative_counters"] == {"optimizer_steps": 2, "environment_loops": 3}
    assert read_document(report["checkpoint_manifest"]["path"])["updates"] == 0


def test_actual_bc_all_censored_confidence_keeps_zero_denominator_without_nan_or_auxiliary_contribution():
    from poorstar.data.bc.windows import BCRunPlan
    from poorstar.evaluation.confidence import ConfidenceTarget, ConfidenceWindow
    from poorstar.hierarchy import HierarchicalPolicy, HierarchyBCRecord, HierarchyBCSequence
    from poorstar.training.bc import HierarchicalBCLearner
    from poorstar.training.config import model_config_sha256

    old, profile, _ = system("persistent_intent")
    high = replace(
        old.model.hierarchy_config,
        manager_period=8,
        task_period=4,
        commitment_horizons=(8,),
        task_horizons=(8,),
    )
    model = HierarchicalPolicy(old.model.config, high)
    rows, *_ = decision_rows()
    window = ConfidenceWindow(
        rows[0].scheduler.key.episode, 1, 0, 8, rows[0].scheduler.key.observation_sha256
    )
    target = ConfidenceTarget(
        window.sha256, None, False, "CENSORED_ENGINEERING_FIXTURE", ("a" * 64,)
    )
    records = (
        HierarchyBCRecord(
            rows[0],
            scheduler_control=update_control(model),
            body_control={"manager": None, "tasks": None},
            body_execution_task="episode:task:1",
            supervision_id="CENSORED_MATH_0",
            supervision_kind="actor_control_trace",
            confidence_target=target,
        ),
        HierarchyBCRecord(
            rows[1],
            scheduler_control={"manager": None, "tasks": None},
            supervision_id="CENSORED_MATH_1",
            supervision_kind="actor_control_trace",
        ),
    )
    plan = BCRunPlan("ENGINEERING", 71, 1, hidden_size=32, unroll_length=2, burn_in_length=0)
    controls = {"confidence_aux": 1.0}
    identity = replace(
        old.identity,
        model_config_sha256=model_config_sha256(model.config, high),
        budget_sha256=digest(dict(plan=asdict(plan), head_weights={}, control_weights=controls)),
    )
    learner = HierarchicalBCLearner(model, profile, identity, plan, control_weights=controls)
    baseline_model = HierarchicalPolicy(old.model.config, high)
    baseline_model.load_state_dict(model.state_dict())
    baseline = HierarchicalBCLearner(
        baseline_model, profile, identity, plan, control_weights=controls
    )
    baseline_report = baseline.update(
        [HierarchyBCSequence(tuple(replace(record, confidence_target=None) for record in records))]
    )
    report = learner.update([HierarchyBCSequence(records)])
    assert (
        report["denominators"]["confidence_aux"] == 0
        and "confidence_aux" in report["unsupervised_heads"]
    )
    assert "confidence_aux" not in report["head_loss"] and torch.isfinite(
        torch.tensor(report["loss"])
    )
    # Confidence density shares a head with Manager BC. It may still learn that
    # independently declared objective; censorship contributes exactly nothing.
    assert (
        report["loss"] == baseline_report["loss"]
        and report["gradient_norm"] == baseline_report["gradient_norm"]
    )
    assert learner.parameter_sha256 == baseline.parameter_sha256
    aggregate = learner.aggregate.report(required_heads=("act", "delay"))
    assert (
        aggregate["denominators"]["confidence_aux"] == 0
        and "confidence_aux" not in aggregate["head_loss"]
    )
    assert "confidence_aux" in aggregate["unsupervised_heads"]
