"""Training: gradient audit."""

import math

import torch

from poorstar.training.checkpoint import capture_rng, restore_rng
from poorstar.training.returns import group_targets


def policy_gradient_audit(learner, sequences):
    sequences = tuple(sequences)
    if (
        not sequences
        or len(sequences) > learner.config.maximum_batch_sequences
        or any(len(s.steps) > learner.config.maximum_sequence_steps for s in sequences)
    ):
        raise ValueError("policy-gradient audit exceeds bounded logical batch")
    for sequence in sequences:
        sequence.validate(
            learner.identity, learner.policy_version, learner.config.maximum_policy_lag
        )
        for step in sequence.steps:
            if (
                step.valid
                and learner.behavior_registry.get(step.policy_version) != step.checkpoint_sha256
            ):
                raise ValueError("policy-gradient audit has unregistered behavior")
    device = next(learner.model.parameters()).device
    main = group_targets(
        sequences, learner.config, manager=learner.phase == "MANAGER_RL", device=device
    )
    strategic = group_targets(sequences, learner.config, manager=True, device=device)
    selected = [(s, t, m) for s, t, m in zip(sequences, main, strategic) if t.indices]
    if not selected:
        raise ValueError("no eligible policy-gradient audit samples")
    advantages = torch.cat([t.advantages for _, t, _ in selected])
    normalization = advantages.mean(), advantages.std(unbiased=False)
    if learner.config.normalize_advantage and learner.phase != "VALUE_ONLY":
        effective = (advantages - normalization[0]) / (normalization[1] + 1e-8)
    else:
        effective = advantages
    denominator = len(advantages), max(1, sum(len(m.indices) for _, _, m in selected))
    parameters = [(n, p) for n, p in learner.model.named_parameters() if p.requires_grad]
    # Sum gradients before norms; microbatch gradients may cancel each other.
    accumulated = {n: torch.zeros_like(p) for n, p in parameters}
    mode, rng = learner.model.training, capture_rng()
    learner.model.eval()
    try:
        with torch.enable_grad():
            for sequence, target, manager_target in selected:
                _, _, parts = learner._batch(
                    [sequence],
                    check_behavior=True,
                    normalization=normalization,
                    denominators=denominator,
                    target_overrides=(target,),
                    manager_target_overrides=(manager_target,),
                    return_components=True,
                )
                grads = torch.autograd.grad(
                    parts["policy"], [p for _, p in parameters], allow_unused=True
                )
                for (name, _), grad in zip(parameters, grads):
                    if grad is not None:
                        if not torch.isfinite(grad).all():
                            raise ValueError("nonfinite policy-only gradient")
                        accumulated[name].add_(grad.detach())
    finally:
        learner.model.train(mode)
        restore_rng(rng)
    norms = {
        group: math.sqrt(
            sum(
                float(accumulated[n].double().square().sum())
                for n, _ in members
                if n in accumulated
            )
        )
        for group, members in learner.groups.items()
    }
    return dict(
        schema="poorstar_v3_policy_only_gradient_audit_v1",
        phase=learner.phase,
        samples=len(advantages),
        manager_samples=sum(len(m.indices) for _, _, m in selected),
        effective_advantage_nonzero=int(torch.count_nonzero(effective))
        if learner.phase != "VALUE_ONLY"
        else 0,
        effective_advantage_std=float(effective.std(unbiased=False)),
        group_l2_norms=norms,
        critic_included=False,
        entropy_included=False,
        weight_decay_included=False,
        optimizer_steps=0,
        newly_collected_environment_frames=0,
        capability_evaluated=False,
    )
