"""Actions: recoverability."""

from __future__ import annotations

import dataclasses
import math
from collections import Counter
from typing import Mapping, Sequence


@dataclasses.dataclass(frozen=True)
class DecisionFrame:
    loop: int
    slots: Mapping[int, int] = dataclasses.field(repr=False)
    source_candidates: frozenset[int] = dataclasses.field(repr=False)
    map_size: tuple[int, int] | None = None
    overflow: int = 0


class RecoverabilityAudit:
    def __init__(self, *, grid: int) -> None:
        if type(grid) is not int or grid < 1:
            raise ValueError("a positive explicit decision grid is required")
        self.grid = grid
        self.counts: Counter = Counter()
        self.by_ability: dict[int, Counter] = {}
        self.selection_lengths: Counter = Counter()
        self.bundles: Counter = Counter()

    def command(
        self,
        *,
        ability: int,
        sources: Sequence[int],
        target_tag: int | None,
        position: tuple[float, float] | None,
        action_loop: int | None,
        response_loop: int,
        frames: Mapping[int, DecisionFrame],
    ) -> None:
        row = self.by_ability.setdefault(ability, Counter())

        def count(name: str) -> None:
            self.counts[name] += 1
            row[name] += 1

        count("commands")
        self.selection_lengths[len(sources)] += 1
        if len(set(sources)) != len(sources):
            count("duplicate_source_identifier_commands")
        mode = (
            "ambiguous"
            if target_tag and position is not None
            else ("entity" if target_tag else "position" if position is not None else "none")
        )
        count("target_mode_" + mode)
        if action_loop is None:
            count("missing_protocol_action_timestamp")
            return
        if action_loop < 0 or action_loop > response_loop:
            count("invalid_protocol_action_timestamp")
            return
        self.bundles[action_loop] += 1
        anchor = ((action_loop - 1) // self.grid) * self.grid
        frame = frames.get(anchor)
        if frame is None:
            count("no_strictly_earlier_grid_observation")
            return
        if frame.loop != anchor or frame.loop >= action_loop:
            raise ValueError("decision frame must be the exact strictly earlier grid anchor")
        count("timestamp_bound_commands")
        if frame.overflow:
            count("commands_on_overflow_frames")
        if not sources:
            count("empty_source_commands")
        else:
            count("source_set_commands")
            if all(tag in frame.slots and tag in frame.source_candidates for tag in sources):
                count("entire_source_set_recoverable")
            else:
                count("entire_source_set_unrecoverable")
        if mode == "entity":
            count(
                "entity_target_recoverable"
                if target_tag in frame.slots
                else "entity_target_unrecoverable"
            )
        elif mode == "position":
            if frame.map_size is None:
                count("position_map_frame_unknown")
            elif (
                all(math.isfinite(value) for value in position)
                and 0 <= position[0] < frame.map_size[0]
                and 0 <= position[1] < frame.map_size[1]
            ):
                count("position_inside_declared_world_map")
            else:
                count("position_outside_declared_world_map")

    def report(self) -> dict:
        return {
            "schema": "poorstar_v3_c3_command_recoverability_audit_v1",
            "decision_grid": self.grid,
            "source_mask_candidate": "OWN_VISIBLE_ONLY_AUDIT_CANDIDATE_V1",
            "counts": dict(sorted(self.counts.items())),
            "by_ability": {
                str(k): dict(sorted(v.items())) for k, v in sorted(self.by_ability.items())
            },
            "complete_selection_length_histogram": {
                str(k): v for k, v in sorted(self.selection_lengths.items())
            },
            "same_protocol_loop_bundle_length_histogram": {
                str(k): v for k, v in sorted(Counter(self.bundles.values()).items())
            },
            "timestamp_policy": "PROTOCOL_ACTION_GAME_LOOP_ONLY_NO_RESPONSE_TIME_SUBSTITUTE",
            "world_target_semantics": "RAW_WORLD_ONLY_LIVE_DISPATCH_NOT_VERIFIED",
            "training_ready": False,
            "rl_ready": False,
        }
