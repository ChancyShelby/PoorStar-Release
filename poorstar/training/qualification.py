"""Training: qualification."""

from dataclasses import replace

import torch

from poorstar.training.checkpoint import load_checkpoint, save_checkpoint
from poorstar.training.config import model_config_sha256
from poorstar.training.contracts import (
    PPOConfig,
    RolloutSequence,
    RolloutStep,
    TrainingIdentity,
    digest,
)
from poorstar.training.learner import PPOLearner
from poorstar.training.scorer import HierarchyScorer, payload_from_session


def qualify_static_outcome_gradients(
    rows, profile, low, *, device, output_dir, source_sha256, dataset_sha256, split_sha256
):
    from poorstar.actions.coarse import coarse_sample
    from poorstar.hierarchy import HierarchicalPolicy, HierarchyConfig, HierarchyPolicySession

    selected = next(
        (
            i
            for i, row in enumerate(rows[:-1])
            if row.decision == "ACT" and row.body is not None and row.body.ability_options()
        ),
        None,
    )
    if selected is None:
        raise ValueError(
            "fixed TRAIN window has no ACT plus successor; mechanism cannot silently enlarge selection"
        )
    row, successor = rows[selected], rows[selected + 1]
    if successor.scheduler.key.loop != row.scheduler.key.loop + row.delay + 1:
        raise ValueError("static mechanism original ACT successor clock changed")
    results = []
    for ordinal, phase in enumerate(("VALUE_ONLY", "MANAGER_RL", "JOINT_RL")):
        high = HierarchyConfig(arm="persistent_intent")
        plan = PPOConfig(
            learning_rate=1e-5,
            weight_decay=0.0,
            value_coefficient=0.0,
            entropy_coefficient=0.0,
            maximum_updates=2,
            maximum_environment_loops=100000,
            maximum_sequence_steps=1,
            maximum_batch_sequences=2,
            max_kl=0.1,
        )
        identity = TrainingIdentity(
            "STATIC_COUNTERFACTUAL_OUTCOME_" + phase,
            high.arm,
            713 + ordinal,
            source_sha256,
            dataset_sha256,
            split_sha256,
            digest(["STATIC_FRAME_ONLY", row.replay_sha256]),
            model_config_sha256(low, high),
            profile.sha256,
            plan.sha256,
        )

        def fresh():
            model = HierarchicalPolicy(low, high).to(device)
            # This changes the declared ENGINEERING model before recording its
            # behavior hash; the same immutable weights then sample both scenes.
            with torch.no_grad():
                model.act_head.weight.zero_()
                model.act_head.bias.copy_(torch.tensor([30.0, -30.0], device=device))
                model.delay_head.weight.zero_()
                model.delay_head.bias.fill_(-30.0)
                model.delay_head.bias[row.delay] = 30.0
                model.fixed_heads["bundle"].bias.copy_(torch.tensor([-30.0, 30.0], device=device))
                model.end_head.bias.fill_(30.0)
            return PPOLearner(model, HierarchyScorer(model, profile), identity, plan, phase=phase)

        def sample(train, group):
            sequences, traces = [], []
            for scene, reward in enumerate((1.0, -1.0)):
                episode = (
                    "STATIC_COUNTERFACTUAL_OUTCOME:"
                    + row.scheduler.key.episode
                    + ":"
                    + phase
                    + ":"
                    + str(group)
                    + ":"
                    + str(scene)
                )
                scheduler = replace(row.scheduler, key=replace(row.scheduler.key, episode=episode))
                body = replace(row.body, key=replace(row.body.key, episode=episode))
                session = HierarchyPolicySession(train.model, profile)
                session.reset(episode)
                memory = train.model.zero_state().detach().clone()
                session.hidden = memory
                decision, delay = session.choose_schedule(
                    scheduler.policy_observation(), low.maximum_delay
                )
                if (decision, delay) != ("ACT", row.delay):
                    raise ValueError("declared static mechanism schedule sample differs; no retry")
                bundle = coarse_sample(
                    body,
                    lambda h, o, c: session.choose_body(body.policy_observation(), h, o, c),
                    profile,
                )
                if bundle.head != "done":
                    raise ValueError("static mechanism sampled bundle incomplete")
                payload = payload_from_session(session, scheduler, body, memory_before=memory)
                trace = payload.trace
                end = successor.scheduler.key.loop
                step = RolloutStep(
                    episode,
                    0,
                    scheduler.key.loop,
                    end,
                    train.policy_version,
                    train.parameter_sha256,
                    payload.observation_sha256,
                    payload.mask_sha256,
                    payload,
                    trace["action_log_prob"],
                    trace["manager_log_prob"],
                    trace["value"],
                    trace["manager_value"],
                    reward=reward,
                    reward_loop=end,
                    terminated=True,
                    boundary_reason="engine_terminal",
                    manager_decision=trace["manager_decision"],
                )
                # engine_terminal is the validator's mathematical boundary
                # fixture, not an assertion that SC2 ran or this scene won/lost.
                sequences.append(RolloutSequence((step,), identity.sha256, memory, False))
                traces.append(digest(trace))
            return tuple(sequences), traces

        torch.manual_seed(identity.seed)
        learner = fresh()
        sequences, traces = sample(learner, 0)
        audit = learner.policy_gradient_audit(sequences)
        norms = audit["group_l2_norms"]
        if phase == "VALUE_ONLY":
            if any(norms.values()):
                raise ValueError("value-only qualification has policy gradients")
        elif (
            audit["effective_advantage_nonzero"] < 2 or norms["manager"] <= 0 or norms["value"] != 0
        ):
            raise ValueError("multi-outcome pure Manager policy gradient not proved")
        if phase == "MANAGER_RL" and any(norms[n] for n in ("backbone", "task", "decoder")):
            raise ValueError("Manager qualification changed lower policy gradient groups")
        if phase == "JOINT_RL" and any(norms[n] <= 0 for n in ("backbone", "task", "decoder")):
            raise ValueError("joint pure policy gradient not proved")
        first = learner.update(sequences)
        checkpoint = output_dir / ("outcome_" + phase)
        save_checkpoint(checkpoint, learner, sampler_state={"scene_group": 1})
        next_sequences, expected_traces = sample(learner, 1)
        expected_audit = learner.policy_gradient_audit(next_sequences)
        expected = learner.update(next_sequences)
        restored = fresh()
        recovered = load_checkpoint(checkpoint, restored)
        actual_sequences, actual_traces = sample(restored, 1)
        actual_audit = restored.policy_gradient_audit(actual_sequences)
        actual = restored.update(actual_sequences)
        if (
            actual_traces != expected_traces
            or actual_audit != expected_audit
            or actual != expected
            or restored.parameter_sha256 != learner.parameter_sha256
            or recovered["sampler_state"] != {"scene_group": 1}
        ):
            raise ValueError("multi-outcome exact next sample/update recovery failed")
        results.append(
            dict(
                phase=phase,
                status="PASS",
                policy_only_gradient_audit=audit,
                first_update=first,
                second_update=actual,
                exact_next_sample_update=True,
                original_row=selected,
                original_replay_sha256=row.replay_sha256,
                trace_digests=traces,
                scope="STATIC_ORIGINAL_TRAIN_FRAME_SYNTHETIC_OUTCOME_MATH",
                schedule_conditioning="DECLARED_BIASED_ACT_RETAINED_DELAY_MODEL",
                retries=0,
                hidden_initialization="ZERO_NEW_STATIC_SCENE_NOT_REPLAY_HIDDEN",
                synthetic_rewards=[1.0, -1.0],
                actual_engine_outcomes=False,
                new_environment_frames=0,
                dispatched_commands=0,
                human_high_level_labels=False,
                live_on_policy_qualified=False,
                capability_evaluated=False,
                rl_ready=False,
            )
        )
    return results
