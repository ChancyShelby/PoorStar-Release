"""Supervised objectives and updates for the shared recurrent policy."""

import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field

import torch

from poorstar.actions.coarse import coarse_teacher_force
from poorstar.actions.contract import TargetDomainRejection
from poorstar.data.bc.supervision import STRICT, prepare_teacher


@dataclass
class BCBatchResult:
    loss: torch.Tensor
    head_sums: dict
    denominators: dict
    metrics: dict
    rejections: dict
    counts: dict
    calibration_bins: dict = field(default_factory=dict)


def bc_objective(
    model,
    sequences,
    profile,
    *,
    head_weights=None,
    collect_metrics=True,
    observation_cache=None,
    supervision_policy=STRICT,
    feature_batch_size=1,
):
    if type(feature_batch_size) is not int or feature_batch_size < 1:
        raise ValueError("positive feature batch size required")
    weights = {} if head_weights is None else head_weights
    from poorstar.models.policy import HEADS

    if set(weights) - set(HEADS) - {"act", "delay"} or any(
        not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0 for v in weights.values()
    ):
        raise ValueError("finite nonnegative objective weights required")
    losses = defaultdict(list)
    metrics = defaultdict(list)
    rejections = Counter()
    counts = Counter()
    calibration = {}

    def observation(frame):
        return (
            frame.policy_observation()
            if observation_cache is None
            else observation_cache.get(frame)
        )

    for sequence in sequences:
        sequence = tuple(sequence)
        features = {}
        if feature_batch_size > 1:
            from poorstar.models.encoder import advance_features, encode_features_batch

            groups = {}
            for i, row in enumerate(sequence):
                row.validate(model.config.maximum_delay)
                if not row.valid:
                    continue
                for phase, frame in [("scheduler", row.scheduler)] + (
                    [("body", row.body)] if row.decision == "ACT" else []
                ):
                    obs = observation(frame)
                    groups.setdefault((row.burn_in, tuple(obs["map_size"])), []).append(
                        ((i, phase), obs)
                    )
            for (burn, size), entries in groups.items():
                with torch.set_grad_enabled(torch.is_grad_enabled() and not burn):
                    for start in range(0, len(entries), feature_batch_size):
                        chunk = entries[start : start + feature_batch_size]
                        values = encode_features_batch(model, [obs for key, obs in chunk])
                        features.update((key, value) for (key, obs), value in zip(chunk, values))

        def encode_frame(i, phase, frame, hidden):
            return (
                model.encode(observation(frame), hidden)
                if feature_batch_size == 1
                else advance_features(model, features[(i, phase)], hidden)
            )

        hidden = None
        episode = None
        next_loop = None
        for row_index, row in enumerate(sequence):
            row.validate(model.config.maximum_delay)
            if not row.valid:
                counts["padding_rows"] += 1
                continue
            identity = (row.replay_sha256, row.scheduler.key.episode, row.scheduler.key.player_id)
            if episode != identity:
                hidden = None
                episode = identity
                next_loop = None
            if next_loop is not None and row.scheduler.key.loop != next_loop:
                raise ValueError("recurrent sequence has a clock gap or overlap")
            next_loop = row.scheduler.key.loop + row.delay + (row.decision == "ACT")
            counts["valid_rows"] += 1
            counts[row.decision] += 1
            active = row.loss and not row.burn_in
            if active:
                counts["active_" + row.decision] += 1
            # Burn-in advances exactly the same phase clocks but contributes no gradient.
            with torch.set_grad_enabled(torch.is_grad_enabled() and not row.burn_in):
                scheduler = encode_frame(row_index, "scheduler", row.scheduler, hidden)
                act, delay = model.scheduler(scheduler, row.decision)
                decision = ("ACT", "WAIT").index(row.decision)
                if active:
                    losses["act"].append(-act.log_prob(torch.tensor(decision, device=model.device)))
                    losses["delay"].append(
                        -delay.log_prob(torch.tensor(row.delay, device=model.device))
                    )
                    if collect_metrics:
                        ok = int(act.logits.argmax().item() == decision)
                        metrics["act_correct"].append(ok)
                        metrics[row.decision + "_correct"].append(ok)
                        metrics["delay_correct"].append(
                            int(delay.logits.argmax().item() == row.delay)
                        )
                        one = torch.zeros_like(act.probs)
                        one[decision] = 1
                        metrics["act_brier"].append(float(((act.probs - one) ** 2).sum().detach()))
                        confidence = float(act.probs.max().detach())
                        bins = calibration.setdefault("act", [[0, 0.0, 0.0] for _ in range(10)])
                        b = bins[min(9, int(confidence * 10))]
                        b[0] += 1
                        b[1] += confidence
                        b[2] += ok
                    counts["loss_rows"] += 1
                hidden = model.consume_schedule(scheduler, row.delay)
                if row.decision == "ACT":
                    counts["requested_commands"] += len(row.targets)
                    counts["requested_source_references"] += sum(
                        len(t.raw.source_slots if hasattr(t, "raw") else t.source_slots)
                        for t in row.targets
                    )
                    counts["original_native_commands"] += len(row.native_event_wires)
                    if active:
                        counts["active_requested_commands"] += len(row.targets)
                    body = encode_frame(row_index, "body", row.body, hidden)
                    hidden = body.hidden
                    # Validation is outside model execution: internal model errors cannot
                    # be reclassified as data that was rejected by its action domain.
                    try:
                        prepared = prepare_teacher(
                            row.body,
                            row.targets,
                            profile,
                            policy=supervision_policy,
                            body_rejection=row.body_rejection,
                        )
                        normalized = prepared.targets
                    except TargetDomainRejection as error:
                        counts["rejected_bundles"] += 1
                        rejections[error.code] += 1
                        counts["rejected_body_commands"] += len(row.targets)
                        if active:
                            counts["active_rejected_bundles"] += 1
                            counts["active_rejected_body_commands"] += len(row.targets)
                    else:
                        counts["skipped_commands"] += len(prepared.skipped)
                        for _, reason in prepared.skipped:
                            rejections["COMMAND_SKIPPED/" + reason] += 1
                        if active:
                            counts["active_skipped_commands"] += len(prepared.skipped)
                        if not normalized:
                            counts["rejected_bundles"] += 1
                            counts["rejected_body_commands"] += len(row.targets)
                            if active:
                                counts["active_rejected_bundles"] += 1
                                counts["active_rejected_body_commands"] += len(row.targets)
                            if row.burn_in:
                                counts["burn_in_rows"] += 1
                                hidden = hidden.detach()
                            continue
                        if prepared.skipped:
                            counts["partially_supervised_bundles"] += 1
                        if collect_metrics:
                            terms, correct, hidden, machine, normalized = model.teacher_body(
                                body, row.body, normalized, profile
                            )
                        else:
                            terms, correct, hidden, machine, normalized = model.teacher_body(
                                body, row.body, normalized, profile, collect_metrics=False
                            )
                        counts["admissible_bundles"] += 1
                        counts["commands"] += len(machine.commands)
                        counts["discarded_sources"] += sum(
                            t.discarded_source_count for t in normalized
                        )
                        counts["converted_commands"] += sum(
                            "OPERATION_COARSENED" in t.reasons for t in normalized
                        )
                        if active:
                            for head, values in terms.items():
                                losses[head].extend(values)
                            for head, values in correct.items():
                                metrics[head + "_teacher_conditioned"].extend(values)
                            categorical = [
                                v
                                for h, values in correct.items()
                                if h in HEADS and h != "target_position"
                                for v in values
                            ]
                            if collect_metrics:
                                metrics["teacher_conditioned_discrete_bundle_exact"].append(
                                    int(all(categorical))
                                )
                            for h in HEADS:
                                if h == "target_position" or h not in correct:
                                    continue
                                bins = calibration.setdefault(h, [[0, 0.0, 0.0] for _ in range(10)])
                                for ok, confidence in zip(correct[h], correct[h + "_confidence"]):
                                    b = bins[min(9, int(confidence * 10))]
                                    b[0] += 1
                                    b[1] += confidence
                                    b[2] += ok
                            counts["body_loss_bundles"] += 1
                if row.burn_in:
                    counts["burn_in_rows"] += 1
                    hidden = hidden.detach()
    head_sums = {h: torch.stack(values).sum() for h, values in losses.items()}
    denominators = {h: len(values) for h, values in losses.items()}
    # Differentiable zero permits a normal sparse batch. Aggregate requirements are
    # checked separately; an empty batch is not evidence that training is complete.
    total = sum(
        (value / denominators[h] * weights.get(h, 1.0) for h, value in head_sums.items()),
        next(model.parameters()).sum() * 0.0,
    )
    if not torch.isfinite(total):
        raise ValueError("nonfinite BC objective")
    return BCBatchResult(
        total,
        head_sums,
        denominators,
        {h: dict(sum=sum(v), count=len(v), mean=sum(v) / len(v)) for h, v in metrics.items()},
        dict(rejections),
        dict(counts),
        calibration,
    )


class BCAggregate:
    def __init__(self):
        self.denominators = Counter()
        self.head_sums = Counter()
        self.counts = Counter()
        self.rejections = Counter()
        self.metric_sums = Counter()
        self.metric_counts = Counter()
        self.calibration_bins = {}

    def add(self, result):
        self.denominators.update(result.denominators)
        self.counts.update(result.counts)
        self.rejections.update(result.rejections)
        self.head_sums.update({h: float(v.detach()) for h, v in result.head_sums.items()})
        for h, value in result.metrics.items():
            self.metric_sums[h] += value["sum"]
            self.metric_counts[h] += value["count"]
        for head, rows in result.calibration_bins.items():
            bins = self.calibration_bins.setdefault(head, [[0, 0.0, 0.0] for _ in range(10)])
            for a, b in zip(bins, rows):
                for i in range(3):
                    a[i] += b[i]

    def state_dict(self):
        return dict(
            denominators=dict(self.denominators),
            head_sums=dict(self.head_sums),
            counts=dict(self.counts),
            rejections=dict(self.rejections),
            metric_sums=dict(self.metric_sums),
            metric_counts=dict(self.metric_counts),
            calibration_bins={
                h: [list(b) for b in bins] for h, bins in self.calibration_bins.items()
            },
        )

    def load_state_dict(self, state):
        if set(state) != set(self.state_dict()):
            raise ValueError("aggregate checkpoint schema drift")
        for key in (
            "denominators",
            "head_sums",
            "counts",
            "rejections",
            "metric_sums",
            "metric_counts",
        ):
            setattr(self, key, Counter(state[key]))
        self.calibration_bins = {
            h: [list(b) for b in bins] for h, bins in state["calibration_bins"].items()
        }

    def report(self, *, required_heads=()):
        missing = [h for h in required_heads if self.denominators[h] == 0]
        if missing:
            raise ValueError("zero aggregate supervision denominator: " + ",".join(missing))
        return dict(
            head_loss={h: self.head_sums[h] / n for h, n in self.denominators.items()},
            denominators=dict(self.denominators),
            counts=dict(self.counts),
            rejections=dict(self.rejections),
            metrics={h: self.metric_sums[h] / n for h, n in self.metric_counts.items()},
            teacher_conditioned_ece={
                h: sum(abs(b[1] - b[2]) for b in bins) / sum(b[0] for b in bins)
                for h, bins in self.calibration_bins.items()
            },
            capability_verdict="NOT_EVALUATED",
            rl_ready=False,
        )


class BCTrainer:
    """Instantiate explicitly only after a formal run's data/budget/gates are frozen."""

    def __init__(
        self,
        model,
        profile,
        *,
        learning_rate,
        gradient_clip=10.0,
        observation_cache=None,
        supervision_policy=STRICT,
        feature_batch_size=1,
    ):
        if (
            not math.isfinite(learning_rate)
            or not math.isfinite(gradient_clip)
            or learning_rate <= 0
            or gradient_clip <= 0
        ):
            raise ValueError("finite positive optimizer bounds required")
        self.model = model
        self.profile = profile
        self.gradient_clip = gradient_clip
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        self.steps = 0
        self.observation_cache = observation_cache
        from poorstar.data.bc.supervision import POLICIES

        if supervision_policy not in POLICIES:
            raise ValueError("unknown supervision policy")
        self.supervision_policy = supervision_policy
        if type(feature_batch_size) is not int or feature_batch_size < 1:
            raise ValueError("positive feature batch size required")
        self.feature_batch_size = feature_batch_size

    def step(self, sequences):
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        result = bc_objective(
            self.model,
            sequences,
            self.profile,
            collect_metrics=False,
            observation_cache=self.observation_cache,
            supervision_policy=self.supervision_policy,
            feature_batch_size=self.feature_batch_size,
        )
        if not result.counts.get("loss_rows"):
            raise ValueError("no supervised rows; optimizer step forbidden")
        result.loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.gradient_clip, error_if_nonfinite=True
        )
        if not torch.isfinite(norm):
            raise ValueError("nonfinite gradient")
        self.optimizer.step()
        self.steps += 1
        return result

    def step_microbatched(self, sequences):
        """One logical AdamW update, exact per-head denominators across windows.

        Backpropagate one window at a time to bound activation memory. Averaging
        microbatch means would overweight sparse entity/position heads; sum each
        head against the full logical-batch token count instead.
        """
        sequences = tuple(sequences)
        denominators = Counter()
        for sequence in sequences:
            for row in sequence:
                row.validate(self.model.config.maximum_delay)
                if not row.valid or row.burn_in or not row.loss:
                    continue
                denominators.update(("act", "delay"))
                if row.decision != "ACT" or row.body_rejection:
                    continue
                try:
                    trace = prepare_teacher(
                        row.body, row.targets, self.profile, policy=self.supervision_policy
                    ).trace
                except TargetDomainRejection:
                    continue
                denominators.update(h for h, o, v in trace)
        if not denominators["act"]:
            raise ValueError("no supervised rows; optimizer step forbidden")
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        sums = Counter()
        counts = Counter()
        rejections = Counter()
        actual_denominators = Counter()
        total = 0.0
        for sequence in sequences:
            result = bc_objective(
                self.model,
                [sequence],
                self.profile,
                collect_metrics=False,
                observation_cache=self.observation_cache,
                supervision_policy=self.supervision_policy,
                feature_batch_size=self.feature_batch_size,
            )
            objective = sum(
                (value / denominators[h] for h, value in result.head_sums.items()),
                next(self.model.parameters()).sum() * 0.0,
            )
            objective.backward()
            total += float(objective.detach())
            sums.update({h: float(v.detach()) for h, v in result.head_sums.items()})
            counts.update(result.counts)
            rejections.update(result.rejections)
            actual_denominators.update(result.denominators)
        if actual_denominators != denominators:
            raise ValueError("microbatch supervision denominator drift")
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.gradient_clip, error_if_nonfinite=True
        )
        self.optimizer.step()
        self.steps += 1
        return BCBatchResult(
            torch.tensor(total, device=self.model.device),
            {h: torch.tensor(v, device=self.model.device) for h, v in sums.items()},
            dict(denominators),
            {},
            dict(rejections),
            dict(counts),
        )

    @torch.no_grad()
    def evaluate(self, sequences):
        self.model.eval()
        return bc_objective(
            self.model,
            sequences,
            self.profile,
            observation_cache=self.observation_cache,
            supervision_policy=self.supervision_policy,
            feature_batch_size=self.feature_batch_size,
        )
