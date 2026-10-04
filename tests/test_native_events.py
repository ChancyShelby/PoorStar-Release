import json

import pytest

from poorstar.engine.native_events import census_native_events, complete_candidate_entries


def event(loop, uid, name, **fields):
    return dict(_gameloop=loop, _userid={"m_userId": uid}, _event="NNet.Game." + name, **fields)


def test_all_players_bundles_manager_commands_and_unknown_fields_preserved_without_tags():
    events = [
        event(0, 5, "SSelectionDeltaEvent", m_addUnitTags=[99999999]),
        event(
            3,
            5,
            "SCmdEvent",
            m_data={"TargetUnit": {"m_tag": 99999999}},
            m_cmdFlags=2,
            m_abil={"m_abilLink": 7, "m_abilCmdIndex": 2},
        ),
        event(3, 5, "SCommandManagerStateEvent", m_state=1),
        event(
            4,
            2,
            "SCmdEvent",
            m_data={"TargetPoint": {"x": 7654321, "y": 1234567}},
            m_cmdFlags=0,
            m_abil=None,
        ),
        event(5, 8, "SCmdEvent", m_data={"None": None}, m_cmdFlags=0, m_abil=None),
    ]
    result = census_native_events(events, controlled_user_ids=[5, 2])
    assert result["counts"]["SCmdEvent"] == 3 and result["views"]["5"]["direct_commands"] == 1
    assert result["views"]["5"]["SSelectionDeltaEvent"] == 1
    assert result["views"]["5"]["manager_state_1_commands"] == 1
    assert result["same_loop_command_lengths"]["5"] == {"2": 1}
    assert result["physical_command_fields"]["2"]["ability_link_unavailable"] == 1
    assert result["physical_command_fields"]["2"]["target_point_xy_present"] == 1
    assert result["native_ability_link_index_counts"]["5"] == {"7:2": 1}
    assert all(
        s not in json.dumps(result)
        for s in ("99999999", "7654321", "1234567", "m_addUnitTags", "m_tag")
    )
    assert not result["engine_ability_mapping_verified"] and not result["actor_inputs_generated"]


def test_unknown_identity_or_invalid_time_and_target_fail_closed():
    with pytest.raises(ValueError, match="identities"):
        census_native_events([], controlled_user_ids=[1, 1])
    with pytest.raises(ValueError, match="out of order"):
        census_native_events([event(5, 1, "X"), event(4, 1, "X")], controlled_user_ids=[1])
    with pytest.raises(ValueError, match="malformed"):
        census_native_events(
            [event(0, 1, "SCmdEvent", m_data={"TargetPoint": {}, "TargetUnit": {}})],
            controlled_user_ids=[1],
        )


def test_explicit_damaged_scope_entry_does_not_abort_or_invent_a_candidate():
    candidate = dict(scope_candidate=True, base_build=55958, controlled_zerg_player_ids=[1, 2])
    scope = dict(
        entries=[
            {"problems": ["METADATA_ERROR"]},
            candidate,
            dict(scope_candidate=True, base_build=55505, controlled_zerg_player_ids=[1]),
        ],
        counts=dict(compatible_zerg_candidate_matches=1, compatible_zerg_candidate_perspectives=2),
    )
    assert complete_candidate_entries(scope) == [candidate]
    scope["counts"]["compatible_zerg_candidate_matches"] = 2
    with pytest.raises(ValueError, match="denominator"):
        complete_candidate_entries(scope)


def test_legitimate_native_data_target_is_retained_without_guessing_an_api_target():
    e = event(
        5,
        1,
        "SCmdEvent",
        m_data={"Data": 987654321},
        m_cmdFlags=800,
        m_abil={"m_abilLink": 211, "m_abilCmdIndex": 3},
    )
    r = census_native_events([e], controlled_user_ids=[1])
    assert r["views"]["1"]["direct_commands"] == r["views"]["1"]["target_Data"] == 1
    assert r["physical_command_fields"]["1"]["opaque_data_target_present"] == 1
    assert "target_None" not in r["views"]["1"] and "target_TargetUnit" not in r["views"]["1"]
    assert "987654321" not in json.dumps(r)
    assert not r["opaque_data_target_semantics_verified"] and not r["actor_inputs_generated"]
