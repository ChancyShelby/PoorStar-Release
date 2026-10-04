"""Bind replay player, engine version and visible sensor snapshots."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Iterator

from poorstar.engine.observation import ExtractionContext, SensorSnapshot, snapshot_from_protocol


class ReplaySensorAdapter:
    def __init__(self, *, sc2_path: Path, version: str, player_id: int, entity_cap: int) -> None:
        self.sc2_path = Path(sc2_path).resolve()
        self.context = ExtractionContext(player_id, version, "REPLAY_PLAYER", False)
        self.context.validate()
        self.entity_cap = entity_cap
        self.reader: Any = None
        self.engine_version: str | None = None
        self.engine_base_build: int | None = None

    def __enter__(self) -> "ReplaySensorAdapter":
        if not self.sc2_path.is_dir():
            raise FileNotFoundError(f"SC2 engine installation not found: {self.sc2_path}")
        os.environ["SC2PATH"] = str(self.sc2_path)
        from poorstar.engine.replay_reader import ReplayReader

        self.reader = ReplayReader(
            sc2_path=self.sc2_path, version=self.context.sc2_version, commands_only=True
        )
        try:
            self.reader.__enter__()
            ping = self.reader._controller.ping()
            self.engine_version = str(ping.game_version)
            self.engine_base_build = int(ping.base_build)
            requested = self.context.sc2_version.split(".")
            if self.engine_version.split(".")[: len(requested)] != requested:
                raise ValueError(
                    "launched SC2 version does not match the declared extraction version"
                )
        except BaseException:
            self.reader.close()
            raise
        return self

    def __exit__(self, *args: Any) -> None:
        if self.reader is not None:
            self.reader.close()

    def snapshots(self, replay: Path) -> Iterator[SensorSnapshot]:
        if self.reader is None:
            raise RuntimeError("adapter is not open")
        info = self.reader.replay_info(replay)
        player_ids = {int(player.player_info.player_id) for player in info.player_info}
        if self.context.player_id not in player_ids:
            raise ValueError("declared controlled player is not a replay participant")
        if int(info.base_build) != self.engine_base_build:
            raise ValueError("replay and engine base builds differ; no cross-version fallback")
        for tick in self.reader.ticks(replay, player_id=self.context.player_id, step_mul=1):
            yield snapshot_from_protocol(
                tick.observation,
                self.reader.game_info,
                context=self.context,
                entity_cap=self.entity_cap,
            )
