"""Evaluation: bc."""

import math
from collections import Counter

import torch

from poorstar.actions.coarse import coarse_sample, coarse_teacher_force
from poorstar.actions.contract import TargetDomainRejection, Token
from poorstar.data.bc.supervision import STRICT, prepare_teacher


def _sample_encoded_body(model, encoded, frame, profile, *, token_watchdog):
    prefix = encoded.hidden

    def choose(head, options, completed):
        nonlocal prefix
        dist = model.distribution(encoded, prefix, head, options)
        value = (
            dist.choose(True) if head == "target_position" else options[int(dist.logits.argmax())]
        )
        prefix = model.consume(encoded, prefix, Token(head, value))
        return value

    return coarse_sample(frame, choose, profile, token_watchdog=token_watchdog), prefix


@torch.no_grad()
def evaluate_sequence_bodies(
    model,
    sequences,
    profile,
    *,
    position_tolerance=1.0,
    token_watchdog=100000,
    supervision_policy=STRICT,
):
    """Predicted body prefixes propagate to later rows; expert clocks remain explicit.

    ACT/delay predictions are scored, but replay rows cannot supply observations at
    alternative clocks. This is therefore not an online scheduler capability test.
    """
    if model.training:
        raise ValueError("evaluation mode required")
    counts = Counter()
    position_errors = []
    identity = None
    hidden = None
    next_loop = None
    for sequence in sequences:
        hidden = None
        identity = None
        next_loop = None
        for row in sequence:
            if not row.valid:
                continue
            key = (row.replay_sha256, row.scheduler.key.episode, row.scheduler.key.player_id)
            if identity != key:
                hidden = None
                identity = key
                next_loop = None
            if next_loop is not None and row.scheduler.key.loop != next_loop:
                raise ValueError("evaluation clock gap")
            next_loop = row.scheduler.key.loop + row.delay + (row.decision == "ACT")
            encoded = model.encode(row.scheduler.policy_observation(), hidden)
            act, delay = model.scheduler(encoded, row.decision)
            active = row.loss and not row.burn_in
            if active:
                counts[row.decision + "_rows"] += 1
                counts[row.decision + "_correct"] += int(
                    int(act.logits.argmax()) == ("ACT", "WAIT").index(row.decision)
                )
                counts["delay_rows"] += 1
                counts["delay_correct"] += int(int(delay.logits.argmax()) == row.delay)
            hidden = model.consume_schedule(encoded, row.delay)
            if row.decision != "ACT":
                continue
            body = model.encode(row.body.policy_observation(), hidden)
            hidden = body.hidden
            try:
                prepared = prepare_teacher(
                    row.body,
                    row.targets,
                    profile,
                    policy=supervision_policy,
                    body_rejection=row.body_rejection,
                )
                if active:
                    counts["skipped_reference_commands"] += len(prepared.skipped)
                if not prepared.targets:
                    raise TargetDomainRejection(
                        "NO_REPRESENTABLE_COMMAND", "empty filtered reference"
                    )
                teacher, _, _ = coarse_teacher_force(row.body, prepared.targets, profile)
            except TargetDomainRejection:
                if active:
                    counts["rejected_bodies"] += 1
                continue
            try:
                prediction, hidden = _sample_encoded_body(
                    model, body, row.body, profile, token_watchdog=token_watchdog
                )
            except RuntimeError as error:
                if "explicit sampling watchdog exceeded" not in str(error):
                    raise
                counts["watchdog_failures_all_phases"] += 1
                if active:
                    counts["watchdog_failures"] += 1
                    counts["admissible_bodies"] += 1
                # No invented STOP. Reset this failed diagnostic sequence instead
                # of treating an expert hidden state as a sampled continuation.
                break
            if not active:
                continue
            counts["admissible_bodies"] += 1
            joint = len(teacher.commands) == len(prediction.commands)
            counts["expected_commands"] += len(teacher.commands)
            counts["predicted_commands"] += len(prediction.commands)
            for reference, actual in zip(teacher.commands, prediction.commands):
                discrete = reference[:4] == actual[:4] and reference[5] == actual[5]
                counts["command_discrete_exact"] += int(discrete)
                exact = discrete
                if reference[2] == "position":
                    if actual[2] == "position":
                        error = math.dist(reference[4], actual[4])
                        position_errors.append(error)
                        exact = discrete and error <= position_tolerance
                    else:
                        exact = False
                joint = joint and exact
            counts["bundle_within_tolerance"] += int(joint)
    accuracies = [
        counts[d + "_correct"] / counts[d + "_rows"] for d in ("ACT", "WAIT") if counts[d + "_rows"]
    ]
    return dict(
        counts=dict(counts),
        scheduler_balanced_accuracy=sum(accuracies) / len(accuracies) if accuracies else None,
        bundle_within_tolerance_fraction=counts["bundle_within_tolerance"]
        / max(1, counts["admissible_bodies"]),
        position_errors=position_errors,
        position_tolerance=position_tolerance,
        actor_mode="FREE_RUNNING_RECURRENT_BODY_EXPERT_SCHEDULE",
        supervision_policy=supervision_policy,
        reference_scope="DECLARED_REPRESENTABLE_COMMAND_PROJECTION",
        online_scheduler_evaluated=False,
        rl_ready=False,
    )


@torch.no_grad()
def evaluate_body(
    model,
    frame,
    targets,
    profile,
    *,
    hidden=None,
    position_tolerance=1.0,
    token_watchdog=100000,
    supervision_policy=STRICT,
):
    if model.training:
        raise ValueError("evaluation mode required")
    prepared = prepare_teacher(frame, targets, profile, policy=supervision_policy)
    if not prepared.targets:
        raise TargetDomainRejection("NO_REPRESENTABLE_COMMAND", "empty filtered reference")
    teacher, trace, normalized = coarse_teacher_force(frame, prepared.targets, profile)
    encoded = model.encode(frame.policy_observation(), hidden)
    prefix = encoded.hidden
    decisions = []

    def choose(head, options, completed):
        nonlocal prefix
        dist = model.distribution(encoded, prefix, head, options)
        if head == "target_position":
            value = dist.choose(True)
            confidence = None
        else:
            i = int(dist.logits.argmax())
            value = options[i]
            confidence = float(dist.probs[i])
        decisions.append(dict(head=head, confidence=confidence, value=value))
        prefix = model.consume(encoded, prefix, Token(head, value))
        return value

    predicted = coarse_sample(frame, choose, profile, token_watchdog=token_watchdog)
    expected = teacher.commands
    actual = predicted.commands
    counts = Counter(expected_commands=len(expected), predicted_commands=len(actual))
    position_errors = []
    joint = len(expected) == len(actual)
    for reference, prediction in zip(expected, actual):
        exact = reference == prediction
        discrete = reference[:4] == prediction[:4] and reference[5] == prediction[5]
        counts["command_discrete_exact"] += int(discrete)
        if reference[2] == "position" and prediction[2] == "position":
            error = math.dist(reference[4], prediction[4])
            position_errors.append(error)
            exact = discrete and error <= position_tolerance
        counts["command_within_position_tolerance"] += int(exact)
        joint = joint and exact
    return dict(
        schema="poorstar_c3_free_running_body_diagnostic_v1",
        counts=dict(counts),
        normalized_bundle_within_tolerance=joint,
        position_errors=position_errors,
        position_tolerance=position_tolerance,
        decisions=decisions,
        actor_mode="FREE_RUNNING_BODY_EXPERT_SCHEDULE",
        supervision_policy=supervision_policy,
        skipped_reference_commands=len(prepared.skipped),
        reference_scope="DECLARED_REPRESENTABLE_COMMAND_PROJECTION",
        scheduler_capability_evaluated=False,
        engine_legality_evaluated=False,
        rl_ready=False,
    )
