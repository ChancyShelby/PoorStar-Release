"""Training: windows."""

from dataclasses import replace

import torch

from poorstar.training.contracts import RolloutSequence


def sequence_windows(sequence, *, unroll_steps, burn_in_steps):
    if (
        type(unroll_steps) is not int
        or unroll_steps < 1
        or type(burn_in_steps) is not int
        or burn_in_steps < 0
    ):
        raise ValueError("explicit bounded recurrent unroll/burn-in required")
    real = tuple(s for s in sequence.steps if s.valid)
    if any(s.burn_in or not s.loss for s in real):
        raise ValueError("windowing requires original physical episode, not already windowed masks")
    output = []
    for active in range(0, len(real), unroll_steps):
        first = max(0, active - burn_in_steps)
        end = min(len(real), active + unroll_steps)
        values = [
            replace(s, burn_in=i < active, loss=i >= active)
            for i, s in enumerate(real[first:end], first)
        ]
        if end != len(real):
            values[-1] = replace(values[-1], truncated=True, boundary_reason="unroll_cut")
        if first == 0:
            memory = sequence.initial_hidden
        else:
            memory = getattr(real[first].payload, "memory_before", None)
            if memory is None:
                raise ValueError("midgame window lacks its immutable actor recurrent state")
            if not isinstance(memory, torch.Tensor) or not torch.isfinite(memory).all():
                raise ValueError("finite bound memory snapshot required")
            memory = memory.detach().clone()
        output.append(
            RolloutSequence(
                tuple(values),
                sequence.identity_sha256,
                memory,
                sequence.starts_at_episode_boundary and first == 0,
            )
        )
    return tuple(output)
