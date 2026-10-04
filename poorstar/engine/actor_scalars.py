"""Engine: actor scalars."""

import hashlib
from dataclasses import dataclass

from poorstar.engine.observation import SCALAR_FIELDS, ExtractionContext

COMMON_FIELDS = tuple(name for name in SCALAR_FIELDS if name != "larva_count")


@dataclass(frozen=True)
class EngineLarvaIdentity:
    unit_type: int
    base_build: int
    catalog_sha256: str

    @classmethod
    def from_catalog(cls, catalog, *, base_build):
        if type(base_build) is not int or base_build != 55958:
            raise ValueError("this field candidate is qualified only for Base55958")
        identities = [int(u.unit_id) for u in catalog.units if u.name == "Larva"]
        if (
            len({int(u.unit_id) for u in catalog.units}) != len(catalog.units)
            or len(identities) != 1
        ):
            raise ValueError("unique engine Larva catalog identity required")
        return cls(
            identities[0], base_build, hashlib.sha256(catalog.SerializeToString()).hexdigest()
        )


def actor_scalar_candidate(response, *, context: ExtractionContext, identity: EngineLarvaIdentity):
    context.validate()
    if context.sc2_version != "3.16.1" or identity.base_build != 55958:
        raise ValueError("scalar candidate engine/version mismatch")
    obs = response.observation
    if (
        not obs.HasField("game_loop")
        or not obs.HasField("player_common")
        or not obs.HasField("raw_data")
    ):
        raise ValueError("current player observation blocks required")
    common = obs.player_common
    if not common.HasField("player_id") or common.player_id != context.player_id:
        raise ValueError("current controlled player identity required")
    count = 0
    for unit in obs.raw_data.units:
        if not unit.HasField("alliance"):
            raise ValueError("unit alliance absent")
        if unit.alliance != 1:
            continue
        if not unit.HasField("display_type") or not unit.HasField("unit_type"):
            raise ValueError("own unit visibility/type absent")
        count += unit.display_type == 1 and unit.unit_type == identity.unit_type
    values = {name: int(getattr(common, name)) for name in COMMON_FIELDS}
    present = {name: common.HasField(name) for name in COMMON_FIELDS}
    values["own_visible_larva_count"] = count
    present["own_visible_larva_count"] = True
    return dict(
        schema="poorstar_c3_actor_scalars_candidate_v1",
        observation_loop=int(obs.game_loop),
        values=values,
        present=present,
        catalog_sha256=identity.catalog_sha256,
        base_build=identity.base_build,
    )
