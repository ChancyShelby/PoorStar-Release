"""Engine: protocol stream."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator


@dataclass(frozen=True)
class ProtocolTick:
    response_loop: int
    raw: Any
    terminal: bool
    initial: bool


def protocol_ticks(reader: Any, replay: Path, *, player_id: int) -> Iterator[ProtocolTick]:
    """Yield initial, all consecutive responses and terminal; fail on missing lifecycle evidence."""
    info = reader.replay_info(replay)
    participants = {int(p.player_info.player_id) for p in info.player_info}
    if player_id not in participants:
        raise ValueError("controlled player is not a replay participant")
    request = reader._sc_pb.RequestStartReplay(
        replay_data=reader._cfg.replay_data(str(replay)),
        options=reader._options,
        observed_player_id=player_id,
    )
    local = reader.maps_relative_path(info.map_name)
    if local:
        request.map_data = reader._cfg.map_data(local)
    started = False
    try:
        reader._controller.start_replay(request)
        started = True
        reader.game_info = reader._controller.game_info()
        previous = None
        while True:
            raw = reader._controller.observe()
            loop = int(raw.observation.game_loop)
            terminal = bool(raw.player_result)
            if int(raw.observation.player_common.player_id) != player_id:
                raise ValueError("observation player differs from controlled player")
            if previous is None:
                if loop != 0:
                    raise ValueError("initial replay observation is not loop zero")
            elif loop != previous + 1:
                raise ValueError("one-loop protocol stream has a gap or duplicate")
            yield ProtocolTick(loop, raw, terminal, previous is None)
            if terminal:
                return
            previous = loop
            reader._controller.step(1)
    finally:
        if started:
            reader._leave()
