"""Staged PPO updates over causal recurrent and SMDP rollout windows."""

import copy
from dataclasses import asdict

import torch

from poorstar.common.parameters import parameter_digest, parameter_digests
from poorstar.training.contracts import digest
from poorstar.training.returns import group_targets, targets

PHASES = ("VALUE_ONLY", "MANAGER_RL", "JOINT_RL")


def merge_intervals(values):
    merged = []
    for start, end in sorted(values):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    return merged


def interval_size(values):
    return sum(end - start for start, end in merge_intervals(values))


def groups_for(model):
    groups = model.parameter_groups()
    named = dict(model.named_parameters())
    resolved = {}
    seen = set()
    if set(groups) != {"backbone", "decoder", "manager", "task", "value"}:
        raise ValueError("explicit exhaustive policy/value parameter groups required")
    for group, members in groups.items():
        items = list(members.items()) if isinstance(members, dict) else list(members)
        values = []
        for name, parameter in items:
            if name not in named or parameter is not named[name] or name in seen:
                raise ValueError("parameter group overlap or identity drift")
            seen.add(name)
            values.append((name, parameter))
        resolved[group] = tuple(values)
    if seen != set(named) or not resolved["value"]:
        raise ValueError("parameter groups must exhaust model with a real value head")
    return resolved


def detach_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach()
    if isinstance(value, tuple):
        return tuple(detach_tree(v) for v in value)
    if isinstance(value, list):
        return [detach_tree(v) for v in value]
    if isinstance(value, dict):
        return {k: detach_tree(v) for k, v in value.items()}
    return value


class PPOLearner:
    def __init__(self, model, scorer, identity, config, *, phase):
        if phase not in PHASES or identity.budget_sha256 != config.sha256:
            raise ValueError("phase or frozen numeric budget identity drift")
        self.model, self.scorer, self.identity, self.config, self.phase = (
            model,
            scorer,
            identity,
            config,
            phase,
        )
        self.groups = groups_for(model)
        allowed = (
            {"value"}
            if phase == "VALUE_ONLY"
            else {"manager", "value"}
            if phase == "MANAGER_RL"
            else set(self.groups)
        )
        if phase == "MANAGER_RL" and not self.groups["manager"]:
            raise ValueError("manager RL requires actual Manager parameters")
        self.trainable_names = tuple(
            n for g, items in self.groups.items() for n, _ in items if g in allowed
        )
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(name in self.trainable_names)
        self.optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self.updates = self.optimizer_steps = self.policy_version = self.environment_loops = 0
        self.consumed = set()
        self.environment_intervals = {}
        self.loss_intervals = {}
        self.behavior_registry = {0: self.parameter_sha256}
        self.stage_transition = None

    @property
    def parameter_sha256(self):
        return parameter_digest(self.model.state_dict())

    def _scores(self, sequence):
        hidden = sequence.initial_hidden
        scores = {}
        for i, step in enumerate(sequence.steps):
            if not step.valid:
                continue
            with torch.set_grad_enabled(torch.is_grad_enabled() and not step.burn_in):
                score = self.scorer(step, hidden)
                score.validate(step)
            hidden = detach_tree(score.next_hidden) if step.burn_in else score.next_hidden
            scores[i] = score
        return scores

    def _batch(
        self,
        sequences,
        *,
        check_behavior=False,
        normalization=None,
        denominators=None,
        target_overrides=None,
        manager_target_overrides=None,
        return_components=False,
    ):
        manager = self.phase == "MANAGER_RL"
        logps, entropies, values, olds, advantages, returns, old_values = [], [], [], [], [], [], []
        manager_critic_values, manager_critic_targets, manager_critic_old_values = [], [], []
        for sequence_index, sequence in enumerate(sequences):
            scored = self._scores(sequence)
            target = (
                targets(
                    sequence,
                    self.config,
                    manager=manager,
                    device=next(self.model.parameters()).device,
                )
                if target_overrides is None
                else target_overrides[sequence_index]
            )
            for j, i in enumerate(target.indices):
                score = scored[i]
                step = sequence.steps[i]
                lp = score.manager_log_prob if manager else score.log_prob
                entropy = score.manager_entropy if manager else score.entropy
                value = score.manager_value if manager else score.value
                if check_behavior and step.policy_version == self.policy_version:
                    # The tolerance only accommodates float serialization; it is
                    # not a game-capability threshold and cannot hide domain drift.
                    if abs(float(lp.detach()) - float(target.old_log_probs[j])) > 1e-5:
                        raise ValueError("behavior/replay probability mismatch before update")
                    if abs(float(value.detach()) - float(target.old_values[j])) > 1e-5:
                        raise ValueError("behavior/replay value mismatch before update")
                logps.append(lp.reshape(()))
                entropies.append(entropy.reshape(()))
                values.append(value.reshape(()))
            olds.append(target.old_log_probs)
            advantages.append(target.advantages)
            returns.append(target.returns)
            old_values.append(target.old_values)
            if self.phase in ("VALUE_ONLY", "JOINT_RL"):
                strategic = (
                    targets(
                        sequence,
                        self.config,
                        manager=True,
                        device=next(self.model.parameters()).device,
                    )
                    if manager_target_overrides is None
                    else manager_target_overrides[sequence_index]
                )
                for j, i in enumerate(strategic.indices):
                    manager_critic_values.append(scored[i].manager_value.reshape(()))
                    manager_critic_targets.append(strategic.returns[j])
                    manager_critic_old_values.append(strategic.old_values[j])
        if not logps:
            raise ValueError("no loss decisions for selected training phase")
        new_lp, entropy, value = (torch.stack(v) for v in (logps, entropies, values))
        old_lp, advantage, ret, old_value = (
            torch.cat(v) for v in (olds, advantages, returns, old_values)
        )
        if self.config.normalize_advantage and self.phase != "VALUE_ONLY":
            mean, std = (
                (advantage.mean(), advantage.std(unbiased=False))
                if normalization is None
                else normalization
            )
            advantage = (advantage - mean) / (std + 1e-8)
        advantage = advantage.detach()
        ret = ret.detach()
        old_lp = old_lp.detach()
        old_value = old_value.detach()
        log_ratio = new_lp - old_lp
        if torch.any(log_ratio.abs() > 50):
            raise ValueError("policy ratio overflow")
        ratio = log_ratio.exp()
        clipped = ratio.clamp(1 - self.config.clip_ratio, 1 + self.config.clip_ratio)
        denominator = len(logps) if denominators is None else denominators[0]
        policy_loss = -torch.minimum(ratio * advantage, clipped * advantage).sum() / denominator
        clipped_value = old_value + (value - old_value).clamp(
            -self.config.value_clip, self.config.value_clip
        )
        value_loss = (
            0.5
            * torch.maximum((value - ret).square(), (clipped_value - ret).square()).sum()
            / denominator
        )
        entropy_mean = entropy.sum() / denominator
        if self.phase == "VALUE_ONLY":
            policy_loss = value.sum() * 0.0  # no policy objective or policy-only gradient
            # Critic warm-up is plain regression; clipping a frozen old critic
            # would cap learning before its targets become useful.
            value_loss = 0.5 * (value - ret).square().sum() / denominator
            if manager_critic_values:
                manager_denominator = (
                    len(manager_critic_values) if denominators is None else denominators[1]
                )
                value_loss = (
                    value_loss
                    + 0.5
                    * (torch.stack(manager_critic_values) - torch.stack(manager_critic_targets))
                    .square()
                    .sum()
                    / manager_denominator
                )
            loss = value_loss
        else:
            if self.phase == "JOINT_RL" and manager_critic_values:
                manager_denominator = (
                    len(manager_critic_values) if denominators is None else denominators[1]
                )
                mv = torch.stack(manager_critic_values)
                mr = torch.stack(manager_critic_targets).detach()
                mo = torch.stack(manager_critic_old_values).detach()
                mc = mo + (mv - mo).clamp(-self.config.value_clip, self.config.value_clip)
                value_loss = (
                    value_loss
                    + 0.5
                    * torch.maximum((mv - mr).square(), (mc - mr).square()).sum()
                    / manager_denominator
                )
            loss = (
                policy_loss
                + self.config.value_coefficient * value_loss
                - self.config.entropy_coefficient * entropy_mean
            )
        kl = ((ratio - 1) - log_ratio).clamp_min(0)
        if not torch.isfinite(loss):
            raise ValueError("nonfinite PPO objective")
        report = dict(
            policy_loss=float(policy_loss.detach()),
            value_loss=float(value_loss.detach()),
            entropy=float(entropy_mean.detach()),
            max_kl=float(kl.max().detach()),
            mean_kl=float(kl.mean().detach()),
            samples=len(logps),
        )
        if return_components:
            return loss, report, dict(policy=policy_loss, value=value_loss, entropy=entropy_mean)
        return loss, report

    def policy_gradient_audit(self, sequences):
        """Bounded pure policy-loss differentiation; no Adam/critic/entropy step."""
        from poorstar.training.gradient_audit import policy_gradient_audit

        return policy_gradient_audit(self, sequences)

    def update(self, sequences):
        sequences = tuple(sequences)
        if self.updates >= self.config.maximum_updates or not sequences:
            raise ValueError("optimizer budget exhausted or no trajectories")
        if len(sequences) > self.config.maximum_batch_sequences or any(
            len(s.steps) > self.config.maximum_sequence_steps for s in sequences
        ):
            raise ValueError("registered bounded recurrent batch/unroll capacity exceeded")
        keys = []
        loops = 0
        pending_environment = copy.deepcopy(self.environment_intervals)
        pending_loss = copy.deepcopy(self.loss_intervals)
        for sequence in sequences:
            sequence.validate(self.identity, self.policy_version, self.config.maximum_policy_lag)
            real = [s for s in sequence.steps if s.valid]
            if self.behavior_registry.get(real[0].policy_version) != real[0].checkpoint_sha256:
                raise ValueError("unregistered behavior checkpoint")
            key = digest(
                [
                    real[0].episode_id,
                    real[0].policy_version,
                    real[0].index,
                    real[-1].index,
                    real[0].checkpoint_sha256,
                ]
            )
            if key in self.consumed or key in keys:
                raise ValueError("trajectory already consumed by this learner")
            keys.append(key)
            episode = real[0].episode_id
            active = [(s.game_loop, s.next_game_loop) for s in real if s.loss and not s.burn_in]
            existing = pending_loss.setdefault(episode, [])
            if any(max(a, c) < min(b, d) for a, b in active for c, d in existing):
                raise ValueError("overlapping active trajectory windows already consumed")
            pending_loss[episode] = merge_intervals(existing + active)
            existing_environment = pending_environment.setdefault(episode, [])
            # Count physical newly observed loops, including initial burn-in;
            # reuse of already paid burn-in snapshots adds no new environment.
            combined = merge_intervals(
                existing_environment + [(real[0].game_loop, real[-1].next_game_loop)]
            )
            loops += interval_size(combined) - interval_size(existing_environment)
            pending_environment[episode] = combined
        if self.environment_loops + loops > self.config.maximum_environment_loops:
            raise ValueError("environment game-loop budget exhausted")
        # Snapshot model, Adam moments AND RNG so rejected updates are exactly
        # recoverable, including failure after optimizer.step().
        from poorstar.training.checkpoint import capture_rng, restore_rng

        snapshot = (
            copy.deepcopy(self.model.state_dict()),
            copy.deepcopy(self.optimizer.state_dict()),
            capture_rng(),
        )
        before = parameter_digests(dict(self.model.named_parameters()))
        self.model.eval()  # same actor/scorer dropout semantics, gradients remain enabled
        device = next(self.model.parameters()).device
        per_sequence = group_targets(
            sequences, self.config, manager=self.phase == "MANAGER_RL", device=device
        )
        manager_targets = group_targets(sequences, self.config, manager=True, device=device)
        selected = [
            (s, t, m) for s, t, m in zip(sequences, per_sequence, manager_targets) if t.indices
        ]
        if not selected:
            raise ValueError("no loss decisions for selected training phase")
        count = sum(len(t.indices) for _, t, _ in selected)
        manager_count = sum(len(m.indices) for _, _, m in selected)
        all_advantages = torch.cat([t.advantages for _, t, _ in selected])
        normalization = all_advantages.mean(), all_advantages.std(unbiased=False)
        denominators = count, max(1, manager_count)

        def evaluate_batch(check_behavior=False):
            reports = [
                self._batch(
                    [s],
                    check_behavior=check_behavior,
                    normalization=normalization,
                    denominators=denominators,
                    target_overrides=(t,),
                    manager_target_overrides=(m,),
                )[1]
                for s, t, m in selected
            ]
            return dict(
                policy_loss=sum(r["policy_loss"] for r in reports),
                value_loss=sum(r["value_loss"] for r in reports),
                entropy=sum(r["entropy"] for r in reports),
                max_kl=max(r["max_kl"] for r in reports),
                mean_kl=sum(r["mean_kl"] * r["samples"] for r in reports) / count,
                samples=count,
            )

        try:
            with torch.no_grad():
                initial = evaluate_batch(check_behavior=True)
            if self.phase != "VALUE_ONLY" and initial["max_kl"] > self.config.max_kl:
                raise ValueError("pre-update KL gate failed")
            epoch_reports = []
            for _ in range(self.config.epochs):
                self.optimizer.zero_grad(set_to_none=True)
                losses = []
                reports = []
                for sequence, target, strategic in selected:
                    loss, one_report = self._batch(
                        [sequence],
                        normalization=normalization,
                        denominators=denominators,
                        target_overrides=(target,),
                        manager_target_overrides=(strategic,),
                    )
                    if self.phase != "VALUE_ONLY" and one_report["max_kl"] > self.config.max_kl:
                        raise ValueError("KL gate failed before epoch")
                    loss.backward()
                    losses.append(float(loss.detach()))
                    reports.append(one_report)
                    del loss
                grads = [
                    p for p in self.model.parameters() if p.requires_grad and p.grad is not None
                ]
                if not grads:
                    raise ValueError("no differentiable gradients in phase")
                norm = torch.nn.utils.clip_grad_norm_(
                    grads, self.config.gradient_clip, error_if_nonfinite=True
                )
                self.optimizer.step()
                if any(not torch.isfinite(p).all() for p in self.model.parameters()):
                    raise ValueError("nonfinite parameter after update")
                report = dict(
                    policy_loss=sum(r["policy_loss"] for r in reports),
                    value_loss=sum(r["value_loss"] for r in reports),
                    entropy=sum(r["entropy"] for r in reports),
                    max_kl=max(r["max_kl"] for r in reports),
                    mean_kl=sum(r["mean_kl"] * r["samples"] for r in reports) / count,
                    samples=count,
                    loss=sum(losses),
                    gradient_norm=float(norm),
                )
                epoch_reports.append(report)
            with torch.no_grad():
                final = evaluate_batch()
            if self.phase != "VALUE_ONLY" and final["max_kl"] > self.config.max_kl:
                raise ValueError("post-update KL gate failed")
            after = parameter_digests(dict(self.model.named_parameters()))
            changed = sorted(n for n in before if before[n] != after[n])
            if set(changed) - set(self.trainable_names):
                raise ValueError("frozen policy parameters changed")
        except BaseException:
            self.model.load_state_dict(snapshot[0])
            self.optimizer.load_state_dict(snapshot[1])
            restore_rng(snapshot[2])
            self.optimizer.zero_grad(set_to_none=True)
            raise
        self.optimizer_steps += self.config.epochs
        self.updates += 1
        self.policy_version += 1
        self.environment_loops += loops
        self.consumed.update(keys)
        self.behavior_registry[self.policy_version] = self.parameter_sha256
        self.environment_intervals = pending_environment
        self.loss_intervals = pending_loss
        return dict(
            schema="poorstar_v3_ppo_update_v1",
            identity_sha256=self.identity.sha256,
            phase=self.phase,
            update=self.updates,
            policy_version=self.policy_version,
            optimizer_steps=self.optimizer_steps,
            environment_loops=self.environment_loops,
            epochs=epoch_reports,
            post_update=final,
            changed_parameters=changed,
            parameter_sha256=self.parameter_sha256,
            capability_evaluated=False,
            rl_ready=False,
        )

    def state_dict(self):
        return dict(
            model=self.model.state_dict(),
            optimizer=self.optimizer.state_dict(),
            identity=asdict(self.identity),
            config=asdict(self.config),
            phase=self.phase,
            trainable_names=self.trainable_names,
            updates=self.updates,
            optimizer_steps=self.optimizer_steps,
            policy_version=self.policy_version,
            environment_loops=self.environment_loops,
            consumed=sorted(self.consumed),
            behavior_registry=self.behavior_registry,
            environment_intervals=self.environment_intervals,
            loss_intervals=self.loss_intervals,
            stage_transition=self.stage_transition,
        )

    def load_state_dict(self, state):
        from poorstar.training.stages import validate_recovered_counters, validate_transition_state

        validate_recovered_counters(state, self.config, self.phase)
        validate_transition_state(
            state.get("stage_transition"),
            self.identity,
            self.phase,
            self.config,
            manager_available=bool(self.groups["manager"]),
        )
        if (
            state["identity"] != asdict(self.identity)
            or state["config"] != asdict(self.config)
            or state["phase"] != self.phase
            or tuple(state["trainable_names"]) != self.trainable_names
        ):
            raise ValueError("checkpoint learner identity/config/group drift")
        if (
            not 0 <= state["updates"] <= self.config.maximum_updates
            or not 0 <= state["environment_loops"] <= self.config.maximum_environment_loops
        ):
            raise ValueError("checkpoint counters outside registered budget")
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        for name in ("updates", "optimizer_steps", "policy_version", "environment_loops"):
            setattr(self, name, state[name])
        self.consumed = set(state["consumed"])
        self.environment_intervals = {
            k: [tuple(v) for v in values] for k, values in state["environment_intervals"].items()
        }
        self.loss_intervals = {
            k: [tuple(v) for v in values] for k, values in state["loss_intervals"].items()
        }
        self.behavior_registry = {int(k): v for k, v in state["behavior_registry"].items()}
        if self.stage_transition is not None and self.stage_transition != state.get(
            "stage_transition"
        ):
            raise ValueError("checkpoint stage transition drift")
        self.stage_transition = state.get("stage_transition")
        if self.behavior_registry.get(self.policy_version) != self.parameter_sha256:
            raise ValueError("checkpoint current policy identity drift")
