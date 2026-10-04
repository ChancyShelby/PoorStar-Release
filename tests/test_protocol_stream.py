from pathlib import Path
from types import SimpleNamespace as S

import pytest

from poorstar.engine.protocol_stream import protocol_ticks


def reader(responses, *, player_id=2):
    calls = []
    responses = iter(responses)
    controller = S(
        start_replay=lambda request: calls.append(("start", request.observed_player_id)),
        observe=lambda: next(responses),
        step=lambda n: calls.append(("step", n)),
        game_info=lambda: S(start_raw="game_info"),
    )
    r = S(
        _sc_pb=S(RequestStartReplay=lambda **kw: S(**kw)),
        _options="options",
        _cfg=S(replay_data=lambda path: b"replay"),
        _controller=controller,
        replay_info=lambda path: S(
            map_name="map", player_info=[S(player_info=S(player_id=player_id))]
        ),
        maps_relative_path=lambda name: None,
        _leave=lambda: calls.append(("leave",)),
    )
    return r, calls


def response(loop, *, terminal=False, player=2, actions=()):
    return S(
        observation=S(game_loop=loop, player_common=S(player_id=player)),
        player_result=["terminal"] if terminal else [],
        actions=actions,
    )


def test_initial_and_terminal_commands_are_retained():
    initial, terminal = (
        response(0, actions=["initial"]),
        response(2, terminal=True, actions=["terminal"]),
    )
    r, calls = reader([initial, response(1), terminal])
    ticks = list(protocol_ticks(r, Path("replay"), player_id=2))
    assert ticks[0].raw is initial and ticks[0].initial
    assert ticks[-1].raw is terminal and ticks[-1].terminal
    assert [x.response_loop for x in ticks] == [0, 1, 2]
    assert calls == [("start", 2), ("step", 1), ("step", 1), ("leave",)]


@pytest.mark.parametrize(
    "responses,message",
    [
        ([response(1)], "loop zero"),
        ([response(0), response(2, terminal=True)], "gap or duplicate"),
        ([response(0), response(0, terminal=True)], "gap or duplicate"),
        ([response(0, player=1)], "controlled player"),
    ],
)
def test_invalid_lifecycle_fails_and_leaves(responses, message):
    r, calls = reader(responses)
    with pytest.raises(ValueError, match=message):
        list(protocol_ticks(r, Path("replay"), player_id=2))
    assert calls[-1] == ("leave",)


def test_early_consumer_close_leaves_replay():
    r, calls = reader([response(0), response(1)])
    stream = protocol_ticks(r, Path("replay"), player_id=2)
    next(stream)
    stream.close()
    assert calls[-1] == ("leave",)


def test_controller_failure_is_not_normal_terminal():
    r, calls = reader([response(0)])

    def fail():
        raise OSError("engine unavailable")

    r._controller.observe = fail
    with pytest.raises(OSError, match="engine unavailable"):
        list(protocol_ticks(r, Path("replay"), player_id=2))
    assert calls[-1] == ("leave",)


def test_unknown_participant_does_not_start():
    r, calls = reader([], player_id=1)
    with pytest.raises(ValueError, match="participant"):
        list(protocol_ticks(r, Path("replay"), player_id=2))
    assert calls == []
