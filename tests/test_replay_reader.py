"""Raw replay command decoding, map names and process cleanup."""

import unicodedata
from types import SimpleNamespace

from s2clientprotocol import sc2api_pb2

from poorstar.engine.replay_reader import ReplayReader, normalise_map_name


def test_map_names_preserve_unicode_and_normalize_accents():
    assert normalise_map_name("Odyssée EC") == "odysseeec"
    assert normalise_map_name("奥德赛 - 天梯版") == "奥德赛天梯版"
    assert unicodedata.normalize("NFC", normalise_map_name("오딧세이 - 래더")) == "오딧세이래더"


def test_raw_commands_preserve_order_membership_and_targets():
    raw = sc2api_pb2.ResponseObservation()
    first = raw.actions.add().action_raw.unit_command
    first.ability_id = 23
    first.unit_tags.extend([40, 41])
    first.target_unit_tag = 99
    first.queue_command = True
    raw.actions.add().action_ui.select_army.selection_add = False
    second = raw.actions.add().action_raw.unit_command
    second.ability_id = 16
    second.unit_tags.append(42)
    second.target_world_space_pos.x = 3.5
    second.target_world_space_pos.y = 8.25
    reader = ReplayReader(sc2_path="/unused")
    commands = list(reader._commands(raw, game_loop=18))
    assert commands == [
        dict(
            ability_id=23,
            unit_tags=[40, 41],
            queue_command=True,
            target_unit_tag=99,
            target_world_space_pos=None,
            game_loop=18,
        ),
        dict(
            ability_id=16,
            unit_tags=[42],
            queue_command=False,
            target_unit_tag=0,
            target_world_space_pos=(3.5, 8.25),
            game_loop=18,
        ),
    ]


def test_cleanup_closes_process_even_if_controller_quit_fails():
    events = []

    def quit_controller():
        events.append("quit")
        raise RuntimeError("controller already stopped")

    reader = ReplayReader(sc2_path="/unused")
    reader._controller = SimpleNamespace(quit=quit_controller)
    reader._proc = SimpleNamespace(close=lambda: events.append("close"))
    reader.close()
    reader.close()
    assert events == ["quit", "close"]
    assert reader._controller is None and reader._proc is None
