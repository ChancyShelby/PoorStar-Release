from __future__ import annotations

import pytest

from poorstar.actions.recoverability import DecisionFrame, RecoverabilityAudit


def run(
    *, sources=(100, 101), target=200, position=None, action_loop=9, response_loop=10, frames=None
):
    audit = RecoverabilityAudit(grid=8)
    audit.command(
        ability=23,
        sources=sources,
        target_tag=target,
        position=position,
        action_loop=action_loop,
        response_loop=response_loop,
        frames=frames
        if frames is not None
        else {8: DecisionFrame(8, {100: 0, 101: 1, 200: 2}, frozenset({100, 101}), (20, 30))},
    )
    return audit.report()


def test_full_selection_and_entity_target_bind_only_to_earlier_frame():
    report = run()
    assert report["counts"]["entire_source_set_recoverable"] == 1
    assert report["counts"]["entity_target_recoverable"] == 1
    assert report["complete_selection_length_histogram"] == {"2": 1}
    assert not report["training_ready"] and not report["rl_ready"]
    assert "100" not in str(report) and "200" not in str(report)


def test_one_missing_unit_fails_the_entire_set_without_projection():
    report = run(sources=tuple(range(100, 166)))
    assert report["complete_selection_length_histogram"] == {"66": 1}
    assert report["counts"]["entire_source_set_unrecoverable"] == 1


@pytest.mark.parametrize(
    "action_loop,key",
    [
        (None, "missing_protocol_action_timestamp"),
        (11, "invalid_protocol_action_timestamp"),
        (8, "no_strictly_earlier_grid_observation"),
    ],
)
def test_response_or_same_frame_timestamp_is_never_substituted(action_loop, key):
    report = run(action_loop=action_loop)
    assert report["counts"][key] == 1 and "timestamp_bound_commands" not in report["counts"]


def test_future_or_wrong_frame_under_the_anchor_is_refused():
    with pytest.raises(ValueError, match="earlier"):
        run(frames={8: DecisionFrame(9, {100: 0}, frozenset({100}), (20, 30))})


@pytest.mark.parametrize(
    "position,key",
    [
        ((19.5, 29.5), "position_inside_declared_world_map"),
        ((20.0, 30.0), "position_outside_declared_world_map"),
        ((float("nan"), 1.0), "position_outside_declared_world_map"),
    ],
)
def test_position_denominators_keep_invalid_coordinates(position, key):
    report = run(target=None, position=position)
    assert report["counts"][key] == 1


def test_empty_and_duplicate_selection_commands_are_retained():
    assert run(sources=())["counts"]["empty_source_commands"] == 1
    assert run(sources=(100, 100))["counts"]["duplicate_source_identifier_commands"] == 1


def test_full_bundle_is_counted_even_if_target_binding_fails():
    audit = RecoverabilityAudit(grid=8)
    for _ in range(30):
        audit.command(
            ability=23,
            sources=(100,),
            target_tag=None,
            position=None,
            action_loop=9,
            response_loop=10,
            frames={},
        )
    report = audit.report()
    assert report["counts"]["commands"] == 30
    assert report["same_protocol_loop_bundle_length_histogram"] == {"30": 1}
