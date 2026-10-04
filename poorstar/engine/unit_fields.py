"""Convert raw units with explicit field presence and world coordinates."""

from __future__ import annotations

import dataclasses
from typing import Any, Mapping

from poorstar.geometry import entities as frames

INT_FIELDS = (
    "owner",
    "alliance",
    "display_type",
    "unit_type",
    "cloak",
    "is_burrowed",
    "is_flying",
    "is_selected",
    "assigned_harvesters",
    "ideal_harvesters",
    "mineral_contents",
    "vespene_contents",
)
FLOAT_FIELDS = (
    "health",
    "health_max",
    "shield",
    "energy",
    "build_progress",
    "facing",
    "radius",
    "weapon_cooldown",
)


@dataclasses.dataclass(frozen=True)
class UnitRow:
    """One protocol unit with explicit field presence."""

    tag: int
    alliance: int
    unit_type: int
    display_type: int
    world_x: float
    world_y: float
    values: Mapping[str, Any]
    orders: tuple[Mapping[str, Any], ...]
    buff_ids: tuple[int, ...]
    present: frozenset[str]

    def as_slot_dict(self, slots: Mapping[int, int]) -> dict[str, Any]:
        row = dict(self.values)
        row.update({"world_x": self.world_x, "world_y": self.world_y, "buff_ids": self.buff_ids})
        resolved = []
        for order in self.orders:
            target_tag = int(order.get("target_unit_tag", 0))
            resolved.append(
                {
                    "ability_id": int(order["ability_id"]),
                    "target_slot": int(slots[target_tag]) if target_tag in slots else -1,
                    "has_target_pos": bool(order.get("has_target_pos")),
                    "target_x": float(order.get("target_x", 0.0)),
                    "target_y": float(order.get("target_y", 0.0)),
                    "progress": float(order.get("progress", 0.0)),
                }
            )
        row["orders"] = tuple(resolved)
        return row


def _unit_row(unit: Any, *, include_orders: bool = True) -> UnitRow:
    present = frozenset(field.name for field, _ in unit.ListFields())
    values: dict[str, Any] = {}
    for name in INT_FIELDS:
        values[name] = int(getattr(unit, name)) if name in present else 0
    for name in FLOAT_FIELDS:
        values[name] = float(getattr(unit, name)) if name in present else 0.0
    position = unit.pos if "pos" in present else None
    orders = []
    if include_orders and "orders" in present:
        for order in unit.orders:
            has_pos = order.HasField("target_world_space_pos")
            orders.append(
                {
                    "ability_id": int(order.ability_id),
                    "target_unit_tag": int(order.target_unit_tag),
                    "has_target_pos": bool(has_pos),
                    "target_x": float(order.target_world_space_pos.x) if has_pos else 0.0,
                    "target_y": float(order.target_world_space_pos.y) if has_pos else 0.0,
                    "progress": float(order.progress) if order.HasField("progress") else 0.0,
                }
            )
    buffs = tuple(int(value) for value in unit.buff_ids) if "buff_ids" in present else ()
    return UnitRow(
        tag=int(unit.tag),
        alliance=int(unit.alliance),
        unit_type=int(unit.unit_type),
        display_type=int(unit.display_type),
        world_x=float(position.x) if position is not None else 0.0,
        world_y=float(position.y) if position is not None else 0.0,
        values=values,
        orders=tuple(orders),
        buff_ids=buffs,
        present=present,
    )


def _entity(row: UnitRow) -> frames.Entity:
    return frames.Entity(
        tag=row.tag,
        alliance=row.alliance,
        unit_type=row.unit_type,
        display_type=row.display_type,
        world_x=row.world_x,
        world_y=row.world_y,
        fields=row.values,
    )
