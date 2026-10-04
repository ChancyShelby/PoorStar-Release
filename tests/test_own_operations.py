import json

import pytest
from s2clientprotocol import raw_pb2, sc2api_pb2

from poorstar.engine.observation import ExtractionContext
from poorstar.engine.own_operations import OwnOperationExtractionError, own_operations_from_protocol

CONTEXT = ExtractionContext(1, "3.16.1", "REPLAY_PLAYER", False)


def frame():
    response = sc2api_pb2.ResponseObservation()
    response.observation.game_loop = 7
    response.observation.player_common.player_id = 1
    own = response.observation.raw_data.units.add(
        tag=987654321, alliance=raw_pb2.Self, display_type=raw_pb2.Visible, unit_type=893
    )
    own.orders.add(ability_id=1632, progress=0.25)
    own.passengers.add(tag=876543210, unit_type=105, health=35, health_max=35)
    return response, own


def test_current_own_orders_and_passengers_without_ids_selection_enemy_or_response_action_features():
    response, own = frame()
    enemy = response.observation.raw_data.units.add(
        tag=765432109, alliance=raw_pb2.Enemy, display_type=raw_pb2.Visible, unit_type=48
    )
    enemy.orders.add(ability_id=55555555)
    slots = {int(own.tag): 0, int(enemy.tag): 1}
    before = own_operations_from_protocol(response, context=CONTEXT, entity_slots=slots)
    own.is_selected = True
    response.actions.add().action_raw.unit_command.ability_id = 55555555
    response.observation.ui_data.cargo.slots_available = 8
    after = own_operations_from_protocol(response, context=CONTEXT, entity_slots=slots)
    payload = before.actor_candidate_payload()
    assert payload == after.actor_candidate_payload()
    assert len(payload["parents"]) == 1 and payload["parents"][0]["orders"][0]["ability_id"] == 1632
    assert before.passenger_references == {(0, 0): 876543210}
    assert all(
        x not in json.dumps(payload)
        for x in ("987654321", "876543210", "765432109", "55555555", "is_selected")
    )
    assert "876543210" not in repr(before)
    assert "world_x" not in json.dumps(payload) and "world_y" not in json.dumps(payload)


def test_missing_maxima_progress_and_reference_are_explicit_without_guessing_values():
    response, own = frame()
    own.orders[0].ClearField("progress")
    own.passengers[0].ClearField("tag")
    state = own_operations_from_protocol(response, context=CONTEXT, entity_slots={int(own.tag): 0})
    parent = state.actor_candidate_payload()["parents"][0]
    assert not parent["orders"][0]["progress_present"]
    assert not parent["passengers"][0]["present"]["shield_max"]
    assert not parent["passengers"][0]["present"]["energy_max"]
    assert state.passenger_references == {(0, 0): None}
    parent["orders"][0]["ability_id"] = 0
    assert state.actor_candidate_payload()["parents"][0]["orders"][0]["ability_id"] == 1632


def test_all_513_parents_and_65_passengers_are_preserved_without_distar_caps():
    response, own = frame()
    own.ClearField("passengers")
    for i in range(65):
        own.passengers.add(tag=10000 + i, unit_type=105, health=35)
    for i in range(512):
        response.observation.raw_data.units.add(
            tag=100 + i, alliance=raw_pb2.Self, display_type=raw_pb2.Visible, unit_type=104
        )
    slots = {int(u.tag): i for i, u in enumerate(response.observation.raw_data.units)}
    state = own_operations_from_protocol(response, context=CONTEXT, entity_slots=slots)
    assert len(state.parents) == 513 and len(state.parents[0]["passengers"]) == 65


def test_shared_current_cargo_keeps_both_parent_edges_without_global_dedup_or_identifier_features():
    response, own = frame()
    second = response.observation.raw_data.units.add(
        tag=123456789, alliance=raw_pb2.Self, display_type=raw_pb2.Visible, unit_type=142
    )
    second.passengers.add().CopyFrom(own.passengers[0])
    state = own_operations_from_protocol(
        response, context=CONTEXT, entity_slots={int(own.tag): 0, int(second.tag): 1}
    )
    assert state.passenger_references == {(0, 0): 876543210, (1, 0): 876543210}
    payload = state.actor_candidate_payload()
    assert len(payload["parents"]) == 2
    assert all(len(parent["passengers"]) == 1 for parent in payload["parents"])
    assert "876543210" not in json.dumps(payload)
    # Sharing across parents does not make duplicate entries within one parent acceptable.
    second.passengers.add().CopyFrom(own.passengers[0])
    with pytest.raises(OwnOperationExtractionError, match="passenger"):
        own_operations_from_protocol(
            response, context=CONTEXT, entity_slots={int(own.tag): 0, int(second.tag): 1}
        )


def test_unknown_future_slots_wrong_player_hidden_parent_and_invalid_passenger_fail_closed():
    response, own = frame()
    with pytest.raises(OwnOperationExtractionError, match="slots"):
        own_operations_from_protocol(
            response, context=CONTEXT, entity_slots={int(own.tag): 0, 444: 1}
        )
    with pytest.raises(OwnOperationExtractionError, match="complete"):
        own_operations_from_protocol(response, context=CONTEXT, entity_slots={})
    response.observation.player_common.player_id = 2
    with pytest.raises(OwnOperationExtractionError, match="controlled"):
        own_operations_from_protocol(response, context=CONTEXT, entity_slots={int(own.tag): 0})
    response.observation.player_common.player_id = 1
    own.passengers.add(tag=876543210, unit_type=105)
    with pytest.raises(OwnOperationExtractionError, match="passenger"):
        own_operations_from_protocol(response, context=CONTEXT, entity_slots={int(own.tag): 0})
    own.ClearField("passengers")
    own.display_type = raw_pb2.Snapshot
    assert (
        own_operations_from_protocol(
            response, context=CONTEXT, entity_slots={int(own.tag): 0}
        ).parents
        == ()
    )
