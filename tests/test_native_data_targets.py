import json

import pytest

from poorstar.engine.native_data_targets import census_data_families


def command(uid, kind, value, ability, flags=0, loop=5):
    return dict(
        _event="NNet.Game.SCmdEvent",
        _gameloop=loop,
        _userid={"m_userId": uid},
        m_data={kind: value},
        m_abil=ability,
        m_cmdFlags=flags,
    )


def test_all_views_and_nondata_denominators_preserved_without_opaque_payload_embedding():
    ability = dict(m_abilLink=211, m_abilCmdIndex=3)
    events = [
        command(5, "Data", 7654321, ability, 2),
        command(5, "Data", 0, ability),
        command(2, "Data", -1, None),
        command(2, "None", None, ability),
        command(8, "Data", 999, ability),
    ]
    r = census_data_families(events, controlled_user_ids=[5, 2])
    assert r["views"] == {
        "5": {"all_direct_commands": 2, "data_commands": 2},
        "2": {"all_direct_commands": 2, "data_commands": 1},
    }
    assert r["families"]["5"]["211:3"]["commands"] == 2
    assert r["families"]["5"]["211:3"]["queue_flag_set"] == 1
    assert r["families"]["2"]["ABILITY_LINK_UNAVAILABLE"]["opaque_payload_negative"] == 1
    assert "7654321" not in json.dumps(r) and "999" not in json.dumps(r)
    assert not r["native_data_semantics_verified"] and not r["actor_inputs_generated"]


def test_invalid_union_payload_or_identity_fails_without_no_target_conversion():
    with pytest.raises(ValueError, match="identities"):
        census_data_families([], controlled_user_ids=[True])
    for value in (None, True, "0"):
        with pytest.raises(ValueError, match="payload"):
            census_data_families([command(1, "Data", value, None)], controlled_user_ids=[1])
    e = command(1, "Data", 0, None)
    e["m_data"]["None"] = None
    with pytest.raises(ValueError, match="union"):
        census_data_families([e], controlled_user_ids=[1])


def test_missing_ability_remains_a_family_and_event_time_reversal_fails():
    r = census_data_families([command(1, "Data", 0, None)], controlled_user_ids=[1])
    assert r["families"]["1"]["ABILITY_LINK_UNAVAILABLE"]["commands"] == 1
    with pytest.raises(ValueError, match="out of order"):
        census_data_families(
            [command(1, "Data", 0, None, loop=5), command(1, "Data", 0, None, loop=4)],
            controlled_user_ids=[1],
        )
