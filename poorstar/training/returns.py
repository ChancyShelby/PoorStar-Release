"""Training: returns."""

from dataclasses import dataclass, replace

import torch


@dataclass(frozen=True)
class ReturnTargets:
    indices: tuple[int, ...]
    returns: torch.Tensor
    advantages: torch.Tensor
    old_log_probs: torch.Tensor
    old_values: torch.Tensor


def targets(sequence, config, *, manager=False, device="cpu"):
    real = [(i, s) for i, s in enumerate(sequence.steps) if s.valid]
    if manager:
        starts = [j for j, (_, s) in enumerate(real) if s.manager_decision]
    else:
        starts = list(range(len(real)))
    segments = []
    for k, start in enumerate(starts):
        end = starts[k + 1] if k + 1 < len(starts) else len(real)
        i, first = real[start]
        last = real[end - 1][1]
        reward = sum(
            config.gamma_per_loop
            ** (
                (s.reward_loop if s.reward_loop is not None else s.next_game_loop) - first.game_loop
            )
            * s.reward
            for _, s in real[start:end]
        )
        old_value = first.old_manager_value if manager else first.old_value
        next_value = last.next_manager_value if manager else last.next_value
        duration = last.next_game_loop - first.game_loop
        segments.append((i, first, last, old_value, next_value, reward, duration))
    carry = 0.0
    result = []
    for i, first, last, old_value, next_value, reward, duration in reversed(segments):
        discount = config.gamma_per_loop**duration
        delta = reward + (0 if last.terminated else discount * next_value) - old_value
        continuation = 0.0 if last.terminated or last.truncated else 1.0
        carry = delta + continuation * discount * config.lambda_per_loop**duration * carry
        result.append(
            (
                i,
                carry + old_value,
                carry,
                first.old_manager_log_prob if manager else first.old_log_prob,
                old_value,
                first.loss and not first.burn_in,
            )
        )
    kept = list(reversed([r for r in result if r[-1]]))
    return ReturnTargets(
        tuple(r[0] for r in kept),
        *(
            torch.tensor([r[j] for r in kept], device=device, dtype=torch.float32)
            for j in range(1, 5)
        ),
    )


def group_targets(sequences, config, *, manager=False, device="cpu"):
    """Compute targets across contiguous windows of one frozen behavior episode.

    Only training targets are stitched; no future observation enters the actor
    or recurrent scorer. Burn-in duplicates remain gradient-free, internal
    unroll cuts disappear only when their exact successor is in this group.
    A genuine final truncation keeps its measured visible bootstrap.
    """
    from poorstar.training.contracts import RolloutSequence

    sequences = tuple(sequences)
    groups = {}
    destinations = {}
    fields = (
        "episode_id",
        "index",
        "game_loop",
        "next_game_loop",
        "policy_version",
        "checkpoint_sha256",
        "observation_sha256",
        "mask_sha256",
        "old_log_prob",
        "old_manager_log_prob",
        "old_value",
        "old_manager_value",
        "next_value",
        "next_manager_value",
        "reward",
        "reward_loop",
        "manager_decision",
        "provenance",
        "behavior",
        "hidden_fallback_count",
    )
    for seq_index, sequence in enumerate(sequences):
        for local, step in enumerate(sequence.steps):
            if not step.valid:
                continue
            key = (step.episode_id, step.policy_version, step.checkpoint_sha256)
            members = groups.setdefault(key, {})
            old = members.get(step.index)
            if old is not None and any(getattr(old, n) != getattr(step, n) for n in fields):
                raise ValueError("duplicated burn-in/active row has different frozen behavior")
            if old is not None and (
                old.terminated != step.terminated
                or (
                    old.truncated
                    and old.boundary_reason != "unroll_cut"
                    and (not step.truncated or old.boundary_reason != step.boundary_reason)
                )
            ):
                raise ValueError("duplicated physical termination/truncation differs")
            if step.loss and not step.burn_in:
                destination = (key, step.index)
                if destination in destinations:
                    raise ValueError("duplicate active row in logical update group")
                destinations[destination] = (seq_index, local)
                members[step.index] = replace(step, burn_in=False, loss=True)
            elif old is None:
                members[step.index] = replace(step, burn_in=True, loss=False)
    rows_by_sequence = [[] for _ in sequences]
    if len({key[0] for key in groups}) != len(groups):
        raise ValueError("one physical episode has multiple behavior identities in logical group")
    for key, members in groups.items():
        ordered = [members[i] for i in sorted(members)]
        for i, row in enumerate(ordered[:-1]):
            successor = ordered[i + 1]
            if row.index + 1 != successor.index or row.next_game_loop != successor.game_loop:
                raise ValueError("logical return group has an episode/index/clock gap")
            if row.terminated or (row.truncated and row.boundary_reason != "unroll_cut"):
                raise ValueError("logical return group crosses physical episode end")
            if (
                row.next_value != successor.old_value
                or row.next_manager_value != successor.old_manager_value
            ):
                raise ValueError("logical return group bootstrap discontinuity")
            if row.truncated:
                ordered[i] = replace(row, truncated=False, boundary_reason="")
        virtual = RolloutSequence(tuple(ordered), sequences[0].identity_sha256)
        combined = targets(virtual, config, manager=manager, device=device)
        for position, virtual_index in enumerate(combined.indices):
            row = ordered[virtual_index]
            seq_index, local = destinations[(key, row.index)]
            rows_by_sequence[seq_index].append(
                (
                    local,
                    *(
                        getattr(combined, n)[position]
                        for n in ("returns", "advantages", "old_log_probs", "old_values")
                    ),
                )
            )
    result = []
    for values in rows_by_sequence:
        values.sort(key=lambda v: v[0])
        result.append(
            ReturnTargets(
                tuple(v[0] for v in values),
                *(
                    torch.stack([v[j] for v in values])
                    if values
                    else torch.empty(0, device=device, dtype=torch.float32)
                    for j in range(1, 5)
                ),
            )
        )
    return tuple(result)
