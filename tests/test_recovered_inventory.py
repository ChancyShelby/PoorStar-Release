"""Tests: test recovered inventory."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from poorstar.data.corpus.inventory import inventory, replay_metadata


def fixture(root, *, duplicate=False, outside=False):
    replay = root / "batch_0001/replay_pilot/replays/ZvP/alias.SC2Replay"
    replay.parent.mkdir(parents=True)
    replay.write_bytes(b"fixture bytes")
    row = {
        "source_replay_id": "alias",
        "matchup": "ZvP",
        "target_pids": [2],
        "extracted_path": str(replay),
        "sha256": "a" * 64,
        "file_size": 13,
    }
    if outside:
        row["extracted_path"] = str(root.parent / "outside.SC2Replay")
    path = root / "batch_0001/replay_pilot/pilot_replay_manifest.json"
    path.write_text(json.dumps({"records": [row, row] if duplicate else [row]}))
    return path


def metadata(path):
    return {
        "file_sha256": "a" * 64,
        "bytes": 13,
        "semantic_sha256": "b" * 64,
        "version": "3.16.1.55958",
        "base_build": 55958,
        "frames": 100,
        "map_hash": "map",
        "map_name": "map",
        "players": [
            {"player_id": 1, "race": "Protoss", "account_digest": "account1"},
            {"player_id": 2, "race": "Zerg", "account_digest": "account2"},
        ],
    }


def test_alias_is_not_assumed_to_be_a_content_hash_and_no_training_is_claimed(tmp_path):
    report = inventory([fixture(tmp_path)], allowed_replay_root=tmp_path, inspect=metadata)
    assert report["records"] == report["unique_file_contents"] == 1
    assert not report["failure_counts"]
    assert not report["training_ready"] and not report["split_frozen"] and not report["rl_ready"]
    assert report["historical_exposure"] == "PREVIOUS_V2_RECOVERY_PIPELINE_NO_UNSEEN_DATA_CLAIM"
    assert report["engine_decodability"] == "NOT_VERIFIED"


def test_duplicate_files_and_matches_are_separate_from_record_counts(tmp_path):
    report = inventory(
        [fixture(tmp_path, duplicate=True)], allowed_replay_root=tmp_path, inspect=metadata
    )
    assert report["records"] == 2 and report["unique_semantic_matches"] == 1
    assert report["duplicate_byte_groups"] == report["duplicate_semantic_groups"] == [[0, 1]]
    assert report["duplicate_source_aliases"] == {"alias": 2}
    assert report["accounts_in_multiple_matches"] == 0


@pytest.mark.parametrize("problem", ["hash", "size", "player", "race", "matchup", "parse"])
def test_bad_recovery_inputs_are_explicit_failures(tmp_path, problem):
    path = fixture(tmp_path)

    def inspect(replay):
        result = metadata(replay)
        if problem == "hash":
            result["file_sha256"] = "c" * 64
        elif problem == "size":
            result["bytes"] = 9
        elif problem == "player":
            result["players"][-1]["player_id"] = 3
        elif problem == "race":
            result["players"][-1]["race"] = "Terran"
        elif problem == "matchup":
            result["players"][0]["race"] = "Terran"
        else:
            raise ValueError("damaged archive")
        return result

    report = inventory([path], allowed_replay_root=tmp_path, inspect=inspect)
    assert report["failure_counts"]
    assert report["records"] == 1 and not report["training_ready"]


def test_manifest_cannot_redirect_inventory_outside_declared_root(tmp_path):
    path = fixture(tmp_path, outside=True)
    report = inventory(
        [path], allowed_replay_root=tmp_path, inspect=lambda path: pytest.fail("must not read")
    )
    assert report["failure_counts"] == {"REPLAY_OUTSIDE_DECLARED_ROOT": 1}


@pytest.mark.parametrize("tracker", [None, b"", b"real tracker"])
def test_backup_details_and_tracker_absence_are_explicit_not_fabricated(
    tmp_path, monkeypatch, tracker
):
    path = tmp_path / "fixture.SC2Replay"
    path.write_bytes(b"raw replay bytes")
    members = {
        "replay.details.backup": b"details",
        "replay.game.events": b"commands",
        "replay.tracker.events": tracker,
    }
    player = SimpleNamespace(region_id=1, subregion=1, toon_id=10, pid=1, play_race="Zerg")
    replay = SimpleNamespace(
        players=[player],
        load_players=lambda: None,
        release_string="3.16.1",
        base_build=55958,
        frames=100,
        map_name="map",
        map_hash="hash",
    )
    monkeypatch.setitem(
        sys.modules, "sc2reader", SimpleNamespace(load_replay=lambda *args, **kwargs: replay)
    )
    monkeypatch.setitem(
        sys.modules,
        "mpyq",
        SimpleNamespace(
            MPQArchive=lambda *args: SimpleNamespace(read_file=lambda name: members.get(name))
        ),
    )
    result = replay_metadata(path)
    assert result["details_member"] == "replay.details.backup"
    assert result["tracker_available"] is (tracker is not None)
    assert result["players"][0]["account_digest"] is not None
    assert "raw replay bytes" not in json.dumps(result)
    members["replay.game.events"] = None
    with pytest.raises(ValueError, match="game event"):
        replay_metadata(path)
