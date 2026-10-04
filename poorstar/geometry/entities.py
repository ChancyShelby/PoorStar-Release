"""Deterministically order visible entities and assign stable frame-local slots."""

from __future__ import annotations

import dataclasses
import math
from typing import Any, Iterable, Mapping, Sequence

ALLIANCE_OWN = 1
ALLIANCE_NEUTRAL = 3
ALLIANCE_ENEMY = 4

#: Order of the alliance groups in the entity tensor.  Unknown alliances sort last rather than being
#: dropped, so an unexpected value is visible in the counts.
ALLIANCE_ORDER: dict[int, int] = {ALLIANCE_OWN: 0, ALLIANCE_NEUTRAL: 1, ALLIANCE_ENEMY: 2}
UNKNOWN_ALLIANCE_RANK = 3

DISPLAY_VISIBLE = 1
DISPLAY_SNAPSHOT = 2

#: World-position quantisation used for the third sort key, in world units.  Four units keeps a
#: mineral line's patches in a stable order without merging distinct clusters.
POSITION_QUANTUM = 4.0

#: Candidate tensor widths.  None is chosen by the audit; each is reported with its overflow rate.
CANDIDATE_MAX_ENTITIES: tuple[int, ...] = (128, 256, 512)


@dataclasses.dataclass(frozen=True)
class Entity:
    """One visible-or-owned unit in one frame, in protocol world coordinates."""

    tag: int
    alliance: int
    unit_type: int
    display_type: int
    world_x: float
    world_y: float
    fields: Mapping[str, Any]

    @property
    def quantized_position(self) -> tuple[int, int]:
        return (
            int(math.floor(self.world_x / POSITION_QUANTUM)),
            int(math.floor(self.world_y / POSITION_QUANTUM)),
        )


def alliance_rank(alliance: int) -> int:
    return ALLIANCE_ORDER.get(int(alliance), UNKNOWN_ALLIANCE_RANK)


def sort_key(entity: Entity) -> tuple:
    """Alliance, then unit type, then quantized position, then tag -- and nothing else."""

    return (
        alliance_rank(entity.alliance),
        int(entity.unit_type),
        entity.quantized_position,
        int(entity.tag),
    )


def is_visible(entity: Entity) -> bool:
    return int(entity.display_type) == DISPLAY_VISIBLE


def in_player_frame(entity: Entity) -> bool:
    """Whether the entity belongs in the player's frame at all."""

    return int(entity.alliance) == ALLIANCE_OWN or is_visible(entity)


def order_entities(entities: Iterable[Entity]) -> list[Entity]:
    return sorted(entities, key=sort_key)


def assign_slots(entities: Iterable[Entity], *, max_entities: int) -> dict[str, Any]:
    """Give every entity a slot index, truncating deterministically and counting the overflow."""

    ordered = order_entities(entity for entity in entities if in_player_frame(entity))
    kept = ordered[: max(0, int(max_entities))]
    return {
        "slots": {entity.tag: index for index, entity in enumerate(kept)},
        "kept": kept,
        "overflow": len(ordered) - len(kept),
        "candidates": len(ordered),
    }


def frame_summary(entities: Sequence[Entity]) -> dict[str, Any]:
    """Counts the report needs per frame, without any tag value."""

    own = [entity for entity in entities if int(entity.alliance) == ALLIANCE_OWN]
    visible_enemy = [
        entity
        for entity in entities
        if int(entity.alliance) == ALLIANCE_ENEMY and is_visible(entity)
    ]
    snapshot_enemy = [
        entity
        for entity in entities
        if int(entity.alliance) == ALLIANCE_ENEMY and not is_visible(entity)
    ]
    neutral = [entity for entity in entities if int(entity.alliance) == ALLIANCE_NEUTRAL]
    return {
        "raw_units": len(entities),
        "own": len(own),
        "neutral": len(neutral),
        "visible_enemy": len(visible_enemy),
        "snapshot_enemy": len(snapshot_enemy),
        "in_player_frame": len([e for e in entities if in_player_frame(e)]),
        "visible_total": len([e for e in entities if is_visible(e)]),
    }


def ordering_hash(slots: Mapping[int, int]) -> str:
    """A stable digest of one frame's tag->slot assignment, for the determinism check.

    Tags appear here and nowhere else in an artifact summary; the digest is what crosses runs, not the
    mapping.
    """

    import hashlib

    payload = ";".join(f"{tag}:{slot}" for tag, slot in sorted(slots.items()))
    return hashlib.sha256(payload.encode()).hexdigest()


__all__ = [
    "ALLIANCE_ENEMY",
    "ALLIANCE_NEUTRAL",
    "ALLIANCE_ORDER",
    "ALLIANCE_OWN",
    "CANDIDATE_MAX_ENTITIES",
    "DISPLAY_SNAPSHOT",
    "DISPLAY_VISIBLE",
    "Entity",
    "POSITION_QUANTUM",
    "UNKNOWN_ALLIANCE_RANK",
    "alliance_rank",
    "assign_slots",
    "frame_summary",
    "in_player_frame",
    "is_visible",
    "order_entities",
    "ordering_hash",
    "sort_key",
]
