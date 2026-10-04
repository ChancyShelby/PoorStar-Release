"""Tests: test actor scalars."""

import pytest
from s2clientprotocol import sc2api_pb2 as sc

from poorstar.engine.actor_scalars import EngineLarvaIdentity, actor_scalar_candidate
from poorstar.engine.observation import ExtractionContext


def identity():
    d = sc.ResponseData()
    d.units.add(unit_id=999, name="Larva")
    return EngineLarvaIdentity.from_catalog(d, base_build=55958)


def frame():
    r = sc.ResponseObservation()
    r.observation.game_loop = 8
    r.observation.player_common.player_id = 1
    r.observation.player_common.minerals = 50
    r.observation.raw_data.SetInParent()
    return r


def candidate(r, **kwargs):
    return actor_scalar_candidate(
        r,
        context=ExtractionContext(1, "3.16.1", "REPLAY_PLAYER", False),
        identity=kwargs.get("identity", identity()),
    )


def test_ui_larva_presence_value_selection_and_orders_do_not_change_candidate():
    r = frame()
    u = r.observation.raw_data.units.add(unit_type=999, alliance=1, display_type=1)
    before = candidate(r)
    r.observation.player_common.larva_count = 777
    u.is_selected = True
    u.orders.add(ability_id=123)
    u.tag = 999999999
    assert candidate(r) == before
    assert "larva_count" not in before["values"] and "larva_count" not in before["present"]
    assert before["values"]["own_visible_larva_count"] == 1


def test_count_is_uncapped_owned_visible_and_observed_zero_is_distinct_from_missing_common():
    r = frame()
    for _ in range(513):
        r.observation.raw_data.units.add(unit_type=999, alliance=1, display_type=1)
    for alliance, display in ((4, 1), (4, 2), (1, 2)):
        r.observation.raw_data.units.add(unit_type=999, alliance=alliance, display_type=display)
    assert candidate(r)["values"]["own_visible_larva_count"] == 513
    r.observation.raw_data.ClearField("units")
    d = candidate(r)
    assert d["values"]["own_visible_larva_count"] == 0 and d["present"]["own_visible_larva_count"]
    assert not d["present"]["vespene"]


def test_missing_own_unit_visibility_fails_and_engine_mismatch_fails():
    r = frame()
    r.observation.raw_data.units.add(unit_type=999, alliance=1)
    with pytest.raises(ValueError, match="visibility/type"):
        candidate(r)
    with pytest.raises(ValueError, match="Base55958"):
        EngineLarvaIdentity.from_catalog(sc.ResponseData(), base_build=55505)
