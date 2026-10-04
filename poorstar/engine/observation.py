"""Extract player-visible entities, scalars and native spatial layers."""

from __future__ import annotations

import dataclasses
import hashlib
import math
from typing import Any

import numpy as np

from poorstar.engine import unit_fields as client
from poorstar.geometry import entities as frames

SENSOR_VERSION = "poorstar_v3_c3_sensor_snapshot_candidate_v1"
SCALAR_FIELDS = (
    "minerals",
    "vespene",
    "food_cap",
    "food_used",
    "food_army",
    "food_workers",
    "idle_worker_count",
    "army_count",
    "warp_gate_count",
    "larva_count",
)
EXTRA_ENTITY_FIELDS = (
    "shield_max",
    "energy_max",
    "cargo_space_taken",
    "cargo_space_max",
    "is_powered",
    "is_active",
    "attack_upgrade_level",
    "armor_upgrade_level",
    "shield_upgrade_level",
    "buff_duration_remain",
    "buff_duration_max",
)
NATIVE_RASTER_FRAME = "SC2_PROTOCOL_IMAGE_NATIVE_YX_NO_WORLD_TRANSFORM"


class ObservationExtractionError(ValueError):
    pass


def has(message: Any, name: str) -> bool:
    """Repeated-field emptiness is an observed empty list, not an absent scalar."""
    field = message.DESCRIPTOR.fields_by_name.get(name)
    if field is None:
        return False
    if field.label == field.LABEL_REPEATED:
        return True
    return bool(message.HasField(name))


def decode_image(image: Any) -> np.ndarray:
    """SC2 ImageData -> native (height,width) raster; preserve rectangular and packed images.

    Protocol/PySC2 use high-bit-first packed 1-bit pixels. Integer 16/32-bit images are
    little endian. Reject inconsistent sizes rather than silently reshape or pad.
    """
    width, height, bits = int(image.size.x), int(image.size.y), int(image.bits_per_pixel)
    if width <= 0 or height <= 0 or bits not in (1, 8, 16, 32):
        raise ObservationExtractionError(f"invalid ImageData shape/depth: {width}x{height}/{bits}")
    expected = (width * height * bits + 7) // 8
    raw = bytes(image.data)
    if len(raw) != expected:
        raise ObservationExtractionError(f"ImageData bytes {len(raw)} != expected {expected}")
    if bits == 1:
        pixels = np.unpackbits(np.frombuffer(raw, dtype=np.uint8), bitorder="big")[: width * height]
    else:
        pixels = np.frombuffer(raw, dtype={8: "u1", 16: "<u2", 32: "<i4"}[bits])
    return pixels.reshape(height, width).copy()


@dataclasses.dataclass(frozen=True)
class ExtractionContext:
    player_id: int
    sc2_version: str
    source: str
    fog_disabled: bool

    def validate(self) -> None:
        if type(self.player_id) is not int or self.player_id < 1:
            raise ObservationExtractionError("a controlled player ID is required")
        if not self.sc2_version or self.source not in ("REPLAY_PLAYER", "LIVE_PLAYER"):
            raise ObservationExtractionError(
                "SC2 version and player-perspective source are required"
            )
        if self.fog_disabled is not False:
            raise ObservationExtractionError("observer/full-map inputs are forbidden")


@dataclasses.dataclass(frozen=True)
class SensorSnapshot:
    observation_loop: int
    context: ExtractionContext
    scalars: dict[str, int]
    scalar_present: dict[str, bool]
    upgrades: tuple[int, ...] | None
    entities: tuple[dict[str, Any], ...]
    overflow: int
    spatial: dict[str, np.ndarray]
    spatial_present: dict[str, bool]
    map_size: tuple[int, int] | None
    schema_version: str = SENSOR_VERSION
    raster_frame: str = NATIVE_RASTER_FRAME

    def metadata(self) -> dict[str, Any]:
        """Small persistable header; pixel arrays belong in separate binary shard members."""
        return {
            "schema_version": self.schema_version,
            "observation_loop": self.observation_loop,
            "context": dataclasses.asdict(self.context),
            "scalars": self.scalars,
            "scalar_present": self.scalar_present,
            "upgrades": self.upgrades,
            "entities": self.entities,
            "overflow": self.overflow,
            "map_size": self.map_size,
            "raster_frame": self.raster_frame,
            "world_to_raster": "NOT_ENGINE_VERIFIED",
            "spatial_present": self.spatial_present,
            "spatial": {
                name: {
                    "shape": list(array.shape),
                    "dtype": str(array.dtype),
                    "sha256": hashlib.sha256(array.tobytes()).hexdigest(),
                }
                for name, array in sorted(self.spatial.items())
            },
        }


def snapshot_from_protocol(
    response: Any, game_info: Any | None, *, context: ExtractionContext, entity_cap: int = 512
) -> SensorSnapshot:
    """Replay and live call this identical extraction path; callers retain tags only for dispatch."""
    context.validate()
    if type(entity_cap) is not int or entity_cap <= 0:
        raise ObservationExtractionError("entity cap must be a positive declared integer")
    observation = getattr(response, "observation", response)
    if not has(observation, "game_loop") or not has(observation, "raw_data"):
        raise ObservationExtractionError("raw player observation and game loop are required")
    if not has(observation, "player_common"):
        raise ObservationExtractionError("player_common is absent; no scalar substitute is allowed")
    common = observation.player_common
    if not has(common, "player_id") or int(common.player_id) != context.player_id:
        raise ObservationExtractionError("observation belongs to a different/unknown player")
    scalars = {name: int(getattr(common, name)) for name in SCALAR_FIELDS}
    scalar_present = {name: has(common, name) for name in SCALAR_FIELDS}
    raw = observation.raw_data
    # Actor snapshots never consume these legacy order/target dictionaries;
    # own visible orders use the separate audited operation adapter below.
    units = [client._unit_row(unit, include_orders=False) for unit in raw.units]
    proto_by_identifier = {int(unit.tag): unit for unit in raw.units}
    if len(proto_by_identifier) != len(units) or any(unit.tag <= 0 for unit in units):
        raise ObservationExtractionError("duplicate or invalid entity identifiers")
    assignment = frames.assign_slots(
        (client._entity(unit) for unit in units), max_entities=entity_cap
    )
    slots = assignment["slots"]
    by_identifier = {unit.tag: unit for unit in units}
    entities = []
    for identifier, slot in sorted(slots.items(), key=lambda item: item[1]):
        unit = by_identifier[identifier]
        proto = proto_by_identifier[identifier]
        values = {name: value for name, value in unit.values.items() if name != "is_selected"}
        present = {name: name in unit.present for name in values}
        for name in EXTRA_ENTITY_FIELDS:
            present[name] = name in unit.present
            descriptor = proto.DESCRIPTOR.fields_by_name[name]
            convert = (
                float if descriptor.type in (descriptor.TYPE_FLOAT, descriptor.TYPE_DOUBLE) else int
            )
            values[name] = convert(getattr(proto, name)) if present[name] else 0
        values.update(world_x=unit.world_x, world_y=unit.world_y)
        present.update(world_x="pos" in unit.present, world_y="pos" in unit.present)
        if any(not math.isfinite(float(value)) for value in values.values()):
            raise ObservationExtractionError("non-finite entity attribute")
        entities.append(
            {
                "slot": int(slot),
                "values": values,
                "present": present,
                "buff_ids": list(unit.buff_ids),
            }
        )
    spatial = {}
    spatial_present = {
        name: False for name in ("visibility", "creep", "pathing", "placement", "terrain_height")
    }
    if has(raw, "map_state"):
        for name in ("visibility", "creep"):
            spatial_present[name] = has(raw.map_state, name)
            if spatial_present[name]:
                spatial[name] = decode_image(getattr(raw.map_state, name))
    map_size = None
    if game_info is not None and has(game_info, "start_raw"):
        start = game_info.start_raw
        if has(start, "map_size"):
            map_size = (int(start.map_size.x), int(start.map_size.y))
            if min(map_size) <= 0:
                raise ObservationExtractionError("invalid declared map size")
        for name, field in (
            ("pathing", "pathing_grid"),
            ("placement", "placement_grid"),
            ("terrain_height", "terrain_height"),
        ):
            spatial_present[name] = has(start, field)
            if spatial_present[name]:
                spatial[name] = decode_image(getattr(start, field))
    if map_size is not None:
        for name, array in spatial.items():
            if array.shape != (map_size[1], map_size[0]):
                raise ObservationExtractionError(f"{name} dimensions disagree with raw map size")
    upgrades = tuple(int(value) for value in raw.player.upgrade_ids) if has(raw, "player") else None
    return SensorSnapshot(
        int(observation.game_loop),
        context,
        scalars,
        scalar_present,
        upgrades,
        tuple(entities),
        int(assignment["overflow"]),
        spatial,
        spatial_present,
        map_size,
    )
