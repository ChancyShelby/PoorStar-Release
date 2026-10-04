"""Engine: own operations."""

import math
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Mapping

from poorstar.engine.observation import ExtractionContext, has

SCHEMA = "poorstar_c3_current_own_operation_candidate_v2"
PASSENGER_FIELDS = (
    "unit_type",
    "health",
    "health_max",
    "shield",
    "shield_max",
    "energy",
    "energy_max",
)


class OwnOperationExtractionError(ValueError):
    pass


@dataclass(frozen=True)
class OwnOperationFrame:
    observation_loop: int
    parents: tuple[dict, ...]
    passenger_references: Mapping[tuple[int, int], int | None] = field(repr=False)

    def actor_candidate_payload(self):
        # Provenance, raw IDs and possible dispatch aliases are never this payload's features.
        return dict(
            schema=SCHEMA, observation_loop=self.observation_loop, parents=deepcopy(self.parents)
        )


def _number(message, name, *, integer=False):
    present = has(message, name)
    value = (
        (int if integer else float)(getattr(message, name)) if present else (0 if integer else 0.0)
    )
    if not math.isfinite(value):
        raise OwnOperationExtractionError("non-finite current own operation value")
    return value, present


def own_operations_from_protocol(
    response, *, context: ExtractionContext, entity_slots: Mapping[int, int]
) -> OwnOperationFrame:
    context.validate()
    observation = getattr(response, "observation", response)
    if (
        not has(observation, "game_loop")
        or not has(observation, "raw_data")
        or not has(observation, "player_common")
        or not has(observation.player_common, "player_id")
        or observation.player_common.player_id != context.player_id
    ):
        raise OwnOperationExtractionError("exact controlled current player/loop/raw frame required")
    units = tuple(observation.raw_data.units)
    if any(
        not all(has(u, k) for k in ("tag", "alliance", "display_type")) or u.tag <= 0 for u in units
    ):
        raise OwnOperationExtractionError(
            "current entity identity/ownership/visibility unavailable"
        )
    if len({int(u.tag) for u in units}) != len(units):
        raise OwnOperationExtractionError("duplicate current entity identifier")
    eligible = {int(u.tag) for u in units if u.alliance == 1 or u.display_type == 1}
    if (
        any(
            type(k) is not int or k <= 0 or type(v) is not int or v < 0
            for k, v in entity_slots.items()
        )
        or not set(entity_slots).issubset(eligible)
        or set(entity_slots.values()) != set(range(len(entity_slots)))
    ):
        raise OwnOperationExtractionError(
            "bijective current eligible contiguous entity slots required"
        )
    own = [u for u in units if u.alliance == 1 and u.display_type == 1]
    if any(int(u.tag) not in entity_slots or not has(u, "unit_type") for u in own):
        raise OwnOperationExtractionError(
            "complete uncapped own-visible parent slots/types required"
        )
    parents, references = [], {}
    for unit in sorted(own, key=lambda u: entity_slots[int(u.tag)]):
        slot = entity_slots[int(unit.tag)]
        orders = []
        for index, order in enumerate(unit.orders):
            ability, ability_present = _number(order, "ability_id", integer=True)
            progress, progress_present = _number(order, "progress")
            orders.append(
                dict(
                    local_index=index,
                    ability_id=ability,
                    ability_present=ability_present,
                    progress=progress,
                    progress_present=progress_present,
                )
            )
        passengers = []
        # The measured Nydus pool appears under multiple current own parents. References are
        # parent-local edges, not a globally unique list or globally dispatchable unit sources.
        seen_passengers = set()
        for index, passenger in enumerate(unit.passengers):
            tag = int(passenger.tag) if has(passenger, "tag") else None
            if tag is not None:
                if tag <= 0 or tag == int(unit.tag) or tag in seen_passengers:
                    raise OwnOperationExtractionError(
                        "invalid/duplicated current passenger reference"
                    )
                seen_passengers.add(tag)
            references[(slot, index)] = tag
            values, present = {}, {}
            for name in PASSENGER_FIELDS:
                values[name], present[name] = _number(passenger, name, integer=name == "unit_type")
            passengers.append(dict(local_index=index, values=values, present=present))
        parents.append(
            dict(
                parent_slot=slot,
                unit_type=int(unit.unit_type),
                orders=orders,
                passengers=passengers,
            )
        )
    return OwnOperationFrame(int(observation.game_loop), tuple(parents), references)
