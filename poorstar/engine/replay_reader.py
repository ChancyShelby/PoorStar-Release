#!/usr/bin/env python3
"""Read replay observations and raw commands through the SC2 client."""

from __future__ import annotations

import dataclasses
import os
import unicodedata
from pathlib import Path
from typing import Any, Iterator

DEFAULT_SC2_PATH = ""
DEFAULT_VERSION = "3.16.1"
LOOPS_PER_SECOND = 22.398436

#: Maps whose archive the client can only resolve through the cache carry no ``Maps/`` file;
#: ``maps_relative_path`` copes through the map index, and a map absent from both is reported
#: by name rather than silently skipped.
RACE_BY_ENUM = {0: "None", 1: "Terran", 2: "Zerg", 3: "Protoss", 4: "Random"}


class ReplayUnavailable(Exception):
    """The replay cannot be decoded here, with the reason already formatted."""


def normalise_map_name(name: str) -> str:
    """Lower-case, strip accents, and drop punctuation -- keeping every script.

    Note that a map's identity is its **content hash**, not its name: the corpus spells one
    map as ``Odyssey LE``, ``奥德赛 - 天梯版``, ``오딧세이 - 래더`` and ``Odyssée EC``, and only the
    hash says they are the same ladder map.  This function exists to look a name up in the
    installed ``Maps/`` tree, and the protocol reports an English ``map_name`` for localised
    replays, so the localised spellings rarely need to resolve at all.
    """

    decomposed = unicodedata.normalize("NFKD", str(name or "")).lower()
    return "".join(character for character in decomposed if character.isalnum())


@dataclasses.dataclass(frozen=True)
class Tick:
    """One decoded step: what the player could see, and what they ordered."""

    game_loop: int
    observation: Any  # raw protocol or requested converted observation
    commands: tuple[dict[str, Any], ...]
    seconds: float


class ReplayReader:
    """One SC2 process, many replays.

    Reuse is the point: launching the binary costs more than decoding a replay, and
    ``start_replay`` is valid from every state a finished replay leaves behind.
    """

    def __init__(
        self,
        *,
        sc2_path: str | os.PathLike[str] | None = None,
        version: str = DEFAULT_VERSION,
        screen: int = 64,
        minimap: int = 64,
        raw_resolution: int = 64,
        want_rgb: bool = False,
        commands_only: bool = True,
    ) -> None:
        self.sc2_path = Path(sc2_path or os.environ.get("SC2PATH") or DEFAULT_SC2_PATH)
        self.version = version
        self.screen = screen
        self.minimap = minimap
        self.raw_resolution = raw_resolution
        self.want_rgb = want_rgb
        # Raw mode bypasses PySC2 feature conversion.
        self.commands_only = commands_only
        self._cfg: Any = None
        self._proc: Any = None
        self._controller: Any = None
        self._interface: Any = None
        self._options: Any = None
        self._features: Any = None
        self._sc_pb: Any = None
        self.maps: dict[str, str] = {}
        self.info_cache: dict[str, Any] = {}
        #: The live replay's game_info, set by :meth:`ticks`.  The coordinate frame is built
        #: from it rather than from a map name, so the frame follows the replay's own map.
        self.game_info: Any = None

    # -- lifecycle ---------------------------------------------------------------------

    def __enter__(self) -> "ReplayReader":
        os.environ["SC2PATH"] = str(self.sc2_path)
        from absl import flags as absl_flags

        if not absl_flags.FLAGS.is_parsed():
            absl_flags.FLAGS(["poorstar_replay_reader"])

        from pysc2 import run_configs
        from pysc2.lib import features as features_module
        from s2clientprotocol import sc2api_pb2 as sc_pb

        self._features = features_module
        self._sc_pb = sc_pb
        self._interface = features_module.AgentInterfaceFormat(
            feature_dimensions=features_module.Dimensions(screen=self.screen, minimap=self.minimap),
            raw_resolution=self.raw_resolution,
            use_raw_units=True,
            use_raw_actions=True,
            use_unit_counts=True,
        )
        options = sc_pb.InterfaceOptions(raw=True, score=True)
        if not self.commands_only:
            # The feature layer is what the live extractor reads; a caller that only wants
            # commands leaves it off, because the conversion is where the time goes.
            options.feature_layer.width = self._interface.camera_width_world_units
            options.feature_layer.resolution.x = self.screen
            options.feature_layer.resolution.y = self.screen
            options.feature_layer.minimap_resolution.x = self.minimap
            options.feature_layer.minimap_resolution.y = self.minimap
            options.feature_layer.crop_to_playable_area = self._interface.crop_to_playable_area
        self._options = options
        self._cfg = run_configs.get(version=self.version)
        self._proc = self._cfg.start(want_rgb=self.want_rgb)
        self._controller = self._proc.controller
        self._controller.ping()
        self.maps = self._build_map_index()
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()

    def close(self) -> None:
        for step in (
            lambda: self._controller.quit() if self._controller else None,
            lambda: self._proc.close() if self._proc else None,
        ):
            try:
                step()
            except Exception:  # noqa: BLE001 - teardown must not mask the real error
                pass
        self._controller = None
        self._proc = None

    def _build_map_index(self) -> dict[str, str]:
        """Installed archives, keyed by normalised name, valued by ``Maps/``-relative path.

        Season 3 and Season 4 both ship some of these names; the first match wins and the
        caller can see which file was used through :meth:`maps_relative_path`.
        """

        index: dict[str, str] = {}
        for path in sorted((self.sc2_path / "Maps").glob("**/*")):
            if path.suffix.lower() in (".sc2map", ".s2ma"):
                index.setdefault(
                    normalise_map_name(path.stem),
                    path.relative_to(self.sc2_path / "Maps").as_posix(),
                )
        return index

    # -- metadata ----------------------------------------------------------------------

    def replay_info(self, replay_path: str | os.PathLike[str]) -> Any:
        key = str(replay_path)
        if key not in self.info_cache:
            self.info_cache[key] = self._controller.replay_info(self._cfg.replay_data(key))
        return self.info_cache[key]

    def races(self, info: Any) -> tuple[str, str]:
        races = [RACE_BY_ENUM.get(p.player_info.race_actual, "?") for p in info.player_info]
        return (races + ["?", "?"])[0], (races + ["?", "?"])[1]

    def matchup(self, info: Any) -> str:
        first, second = self.races(info)
        return "{0}v{1}".format(first[0], second[0])

    def mmrs(self, info: Any) -> tuple[int, int]:
        values = [int(getattr(p, "player_mmr", 0) or 0) for p in info.player_info]
        return (values + [0, 0])[0], (values + [0, 0])[1]

    def maps_relative_path(self, map_name: str) -> str | None:
        return self.maps.get(normalise_map_name(map_name))

    # -- decoding ----------------------------------------------------------------------

    def ticks(
        self,
        replay_path: str | os.PathLike[str],
        *,
        player_id: int = 1,
        step_mul: int = 1,
        max_loops: int | None = None,
    ) -> Iterator[Tick]:
        """Yield decoded observations for one replay, oldest first.

        Raises :class:`ReplayUnavailable` with the reason when the client refuses the file,
        after leaving the current game so the process stays usable for the next replay.
        """

        sc_pb = self._sc_pb
        path = str(replay_path)
        try:
            info = self.replay_info(path)
        except Exception as error:  # noqa: BLE001 - reported as a skip, not a crash
            raise ReplayUnavailable(
                "replay_info: {0}: {1}".format(type(error).__name__, error)
            ) from error

        local = self.maps_relative_path(info.map_name)
        request = sc_pb.RequestStartReplay(
            replay_data=self._cfg.replay_data(path),
            options=self._options,
            observed_player_id=int(player_id),
        )
        if local:
            # map_data is a bytes field, and cfg.map_data resolves relative to Maps/.
            request.map_data = self._cfg.map_data(local)
        try:
            self._controller.start_replay(request)
        except Exception as error:  # noqa: BLE001
            self._leave()
            raise ReplayUnavailable(
                "start_replay ({0}, map file {1!r}): {2}: {3}".format(
                    info.map_name, local, type(error).__name__, error
                )
            ) from error

        game_info = self._controller.game_info()
        self.game_info = game_info
        converter = None
        if not self.commands_only:
            converter = self._features.features_from_game_info(
                game_info, agent_interface_format=self._interface
            )

        loops = 0
        while True:
            if max_loops is not None and loops >= max_loops:
                break
            try:
                self._controller.step(int(step_mul))
                raw = self._controller.observe()
            except Exception as error:  # noqa: BLE001
                self._leave()
                raise ReplayUnavailable(
                    "observe at loop {0}: {1}: {2}".format(loops, type(error).__name__, error)
                ) from error
            if getattr(raw, "player_result", None):
                break
            game_loop = int(raw.observation.game_loop)
            observation = raw if converter is None else converter.transform_obs(raw)
            commands = tuple(self._commands(raw, game_loop=game_loop))
            loops += 1
            yield Tick(
                game_loop=game_loop,
                observation=observation,
                commands=commands,
                seconds=game_loop / LOOPS_PER_SECOND,
            )

    def _commands(self, raw: Any, *, game_loop: int) -> Iterator[dict[str, Any]]:
        """The replaying player's raw unit commands in this step.

        ``ResponseObservation.actions`` carries the recorded commands; only raw unit
        commands are macro-relevant, and camera moves are counted by the caller as micro.
        """

        for action in list(getattr(raw, "actions", []) or []):
            if not action.HasField("action_raw"):
                continue
            action_raw = action.action_raw
            if not action_raw.HasField("unit_command"):
                continue
            command = action_raw.unit_command
            position = None
            if command.HasField("target_world_space_pos"):
                position = (command.target_world_space_pos.x, command.target_world_space_pos.y)
            yield dict(
                ability_id=int(command.ability_id),
                unit_tags=list(command.unit_tags),
                queue_command=bool(command.queue_command),
                target_unit_tag=int(command.target_unit_tag),
                target_world_space_pos=position,
                game_loop=game_loop,
            )

    def _leave(self) -> None:
        try:
            self._controller.leave()
        except Exception:  # noqa: BLE001
            pass


def ensure_sc2_path(path: str | os.PathLike[str] | None = None) -> str:
    """Set ``SC2PATH`` before anything imports run_configs; returns what was set."""

    resolved = str(path or os.environ.get("SC2PATH") or DEFAULT_SC2_PATH)
    os.environ["SC2PATH"] = resolved
    return resolved


__all__ = [
    "ReplayReader",
    "ReplayUnavailable",
    "Tick",
    "normalise_map_name",
    "ensure_sc2_path",
    "DEFAULT_SC2_PATH",
    "DEFAULT_VERSION",
    "LOOPS_PER_SECOND",
    "RACE_BY_ENUM",
]
