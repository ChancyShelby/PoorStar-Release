"""Hierarchy: goals."""

import math
from dataclasses import asdict, dataclass

GOAL_KINDS = ("own_completed_unit_delta", "upgrade_present", "allocated_region", "region_visible")


@dataclass(frozen=True)
class TaskGoal:
    kind: str
    subject: int | None = None
    quantity: int | None = None
    region: int | None = None

    def validate(self, config, *, unit_types=None, upgrades=None):
        if self.kind not in config.goal_kinds:
            raise ValueError("goal outside declared predicate vocabulary")
        if self.kind == "own_completed_unit_delta":
            if type(self.subject) is not int or self.subject <= 0:
                raise ValueError("explicit engine unit type required")
            if unit_types is not None and self.subject not in unit_types:
                raise ValueError("goal unit type absent from measured catalog")
            if self.quantity not in config.goal_quantities or type(self.quantity) is not int:
                raise ValueError("declared positive unit-count delta required")
            if self.region is not None:
                raise ValueError("unit-count goal cannot contain a hidden position")
        elif self.kind == "upgrade_present":
            if type(self.subject) is not int or self.subject <= 0:
                raise ValueError("explicit engine upgrade identity required")
            if upgrades is not None and self.subject not in upgrades:
                raise ValueError("goal upgrade absent from measured catalog")
            if self.quantity is not None or self.region is not None:
                raise ValueError("upgrade predicate has no count/position parameter")
        else:
            if self.subject is not None or self.quantity is not None:
                raise ValueError("regional predicates have no implicit unit type/count")
            if type(self.region) is not int or not 0 <= self.region < math.prod(
                config.target_regions
            ):
                raise ValueError("goal region outside declared tactical grid")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict) or set(value) != set(cls.__dataclass_fields__):
            raise ValueError("typed goal schema drift")
        return cls(**value)


def current_own_slots(observation):
    """Allocation domain uses the exact current actor tensor, never old slots."""
    result = []
    for entity in observation["entities"]:
        values, present = entity["values"], entity["present"]
        if (
            present.get("alliance", False)
            and present.get("display_type", False)
            and values["alliance"] == 1
            and values["display_type"] == 1
        ):
            result.append(entity["slot"])
    return tuple(sorted(result))
