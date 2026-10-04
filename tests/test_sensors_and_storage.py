"""Tests: test sensors and storage."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from s2clientprotocol import common_pb2, sc2api_pb2

from poorstar.common.integrity import label_census, payload_hash
from poorstar.data.sensors import SensorShardReader, SensorShardWriter, SensorStorageError
from poorstar.engine.observation import (
    SCALAR_FIELDS,
    ExtractionContext,
    ObservationExtractionError,
    decode_image,
    snapshot_from_protocol,
)


def protocol_frame(loop=8):
    response = sc2api_pb2.ResponseObservation()
    observation = response.observation
    observation.game_loop = loop
    observation.player_common.player_id = 1
    for name in SCALAR_FIELDS:
        setattr(observation.player_common, name, 0)
    observation.player_common.minerals = 500
    observation.raw_data.player.upgrade_ids.extend([1, 2])
    for tag, alliance, display in ((123, 1, 1), (456, 4, 1), (789, 4, 2)):
        unit = observation.raw_data.units.add()
        unit.tag, unit.alliance, unit.display_type, unit.unit_type = tag, alliance, display, 104
        unit.pos.x, unit.pos.y = 1.25, 2.5
        unit.health = 40.5
        unit.shield_max = 10.25
        unit.energy_max = 75.5
        unit.is_selected = True
        unit.buff_ids.extend([4, 5])
    info = sc2api_pb2.ResponseGameInfo()
    info.start_raw.map_size.x, info.start_raw.map_size.y = 3, 2
    for image in (
        observation.raw_data.map_state.visibility,
        observation.raw_data.map_state.creep,
        info.start_raw.pathing_grid,
        info.start_raw.placement_grid,
        info.start_raw.terrain_height,
    ):
        image.size.x, image.size.y, image.bits_per_pixel, image.data = 3, 2, 8, bytes(range(6))
    return response, info


def snapshot(loop=8, source="REPLAY_PLAYER", cap=512):
    response, info = protocol_frame(loop)
    return snapshot_from_protocol(
        response, info, context=ExtractionContext(1, "3.16.1", source, False), entity_cap=cap
    )


def provenance():
    return {
        "purpose": "FIELD_AUDIT",
        "split_hash": None,
        "replay_sha256": "a" * 64,
        "source_manifest_sha256": "b" * 64,
        "extractor_revision": "c" * 64,
    }


def write_shard(root):
    with SensorShardWriter(root, provenance=provenance()) as writer:
        writer.append(snapshot())
        writer.append(snapshot(16))
        manifest = writer.finalize(completion={"status": "STREAM_EXHAUSTED"})
    return root / f"{manifest['sha256']}.manifest.json", manifest


@pytest.mark.parametrize(
    "bits,values,dtype",
    [
        (1, [1, 0, 1, 0, 1, 1], np.uint8),
        (8, [0, 1, 2, 3, 4, 255], np.uint8),
        (16, [0, 1, 2, 3, 4, 65535], np.uint16),
        (32, [-1, 0, 2, 3, 4, 2147483647], np.int32),
    ],
)
def test_native_rectangular_raster_matches_installed_pysc2(bits, values, dtype):
    from pysc2.lib.features import Feature

    expected = np.array(values, dtype=dtype).reshape(2, 3)
    image = common_pb2.ImageData(bits_per_pixel=bits)
    image.size.x, image.size.y = 3, 2
    image.data = np.packbits(expected.ravel()).tobytes() if bits == 1 else expected.tobytes()
    actual = decode_image(image)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(actual, Feature.unpack_layer(image))
    assert actual.dtype == dtype


@pytest.mark.parametrize("data", [b"", b"12345", b"1234567"])
def test_bad_image_bytes_are_rejected(data):
    image = common_pb2.ImageData(bits_per_pixel=8, data=data)
    image.size.x, image.size.y = 3, 2
    with pytest.raises(ObservationExtractionError, match="bytes"):
        decode_image(image)


def test_current_sensors_are_lossless_and_tag_free():
    frame = snapshot()
    assert len(frame.entities) == 2  # hidden enemy snapshot never reaches inputs
    assert [entity["slot"] for entity in frame.entities] == [0, 1]
    assert frame.entities[0]["values"]["health"] == 40.5
    assert frame.entities[0]["values"]["shield_max"] == 10.25
    assert frame.entities[0]["values"]["energy_max"] == 75.5
    assert frame.upgrades == (1, 2)
    assert all(frame.scalar_present.values()) and all(frame.spatial_present.values())
    header = json.dumps(frame.metadata())
    assert "tag" not in header and "is_selected" not in header and "orders" not in header
    assert frame.metadata()["world_to_raster"] == "NOT_ENGINE_VERIFIED"


def test_replay_and_live_use_identical_candidate_values():
    replay, live = snapshot().metadata(), snapshot(source="LIVE_PLAYER").metadata()
    replay.pop("context")
    live.pop("context")
    assert replay == live


def test_response_actions_orders_selection_and_hidden_values_do_not_enter_inputs():
    response, info = protocol_frame()
    context = ExtractionContext(1, "3.16.1", "REPLAY_PLAYER", False)
    before = snapshot_from_protocol(response, info, context=context).metadata()
    response.actions.add().action_raw.unit_command.ability_id = 999
    response.observation.raw_data.units[0].orders.add().target_unit_tag = 555
    response.observation.raw_data.units[0].is_selected = False
    hidden = response.observation.raw_data.units[-1]
    hidden.health, hidden.pos.x, hidden.unit_type = 999, 900, 999
    hidden.buff_ids.append(999)
    assert snapshot_from_protocol(response, info, context=context).metadata() == before


def test_missing_fields_remain_explicitly_missing_not_observed_zero():
    response, info = protocol_frame()
    response.observation.player_common.ClearField("vespene")
    response.observation.raw_data.map_state.ClearField("creep")
    response.observation.raw_data.ClearField("player")
    frame = snapshot_from_protocol(
        response, info, context=ExtractionContext(1, "3.16.1", "REPLAY_PLAYER", False)
    )
    assert frame.scalars["vespene"] == 0 and not frame.scalar_present["vespene"]
    assert not frame.spatial_present["creep"] and "creep" not in frame.spatial
    assert frame.upgrades is None


@pytest.mark.parametrize(
    "context",
    [
        ExtractionContext(1, "3.16.1", "REPLAY_PLAYER", True),
        ExtractionContext(2, "3.16.1", "REPLAY_PLAYER", False),
        ExtractionContext(1, "", "REPLAY_PLAYER", False),
        ExtractionContext(1, "3.16.1", "OBSERVER", False),
    ],
)
def test_invalid_perspective_is_rejected(context):
    response, info = protocol_frame()
    with pytest.raises(ObservationExtractionError):
        snapshot_from_protocol(response, info, context=context)


@pytest.mark.parametrize("problem", ["duplicate", "invalid_tag", "nonfinite", "bad_size", "no_raw"])
def test_bad_sensor_frame_fails_closed(problem):
    response, info = protocol_frame()
    if problem == "duplicate":
        response.observation.raw_data.units[1].tag = 123
    elif problem == "invalid_tag":
        response.observation.raw_data.units[1].tag = 0
    elif problem == "nonfinite":
        response.observation.raw_data.units[0].health = float("nan")
    elif problem == "bad_size":
        info.start_raw.map_size.x = 4
    else:
        response.observation.ClearField("raw_data")
    with pytest.raises(ObservationExtractionError):
        snapshot_from_protocol(
            response, info, context=ExtractionContext(1, "3.16.1", "REPLAY_PLAYER", False)
        )


def test_entity_overflow_is_reported():
    assert snapshot(cap=1).overflow == 1


def test_binary_shards_rebuild_identically_and_deduplicate_static_maps(tmp_path):
    first_path, first = write_shard(tmp_path / "first")
    _, second = write_shard(tmp_path / "second")
    assert first == second
    assert len([n for n in first["members"] if n.endswith(".npy")]) == 1
    with SensorShardReader(first_path) as reader:
        reader.verify()
        for loop in (8, 16):
            assert reader.frame(loop).metadata() == snapshot(loop).metadata()
        assert reader.index["training_ready"] is False


@pytest.mark.parametrize("problem", ["duplicate_loop", "backward_loop", "version", "player", "map"])
def test_episode_drift_does_not_produce_complete_shard(tmp_path, problem):
    with pytest.raises(SensorStorageError):
        with SensorShardWriter(tmp_path, provenance=provenance()) as writer:
            writer.append(snapshot())
            next_frame = snapshot(16)
            if problem in ("duplicate_loop", "backward_loop"):
                next_frame = dataclasses.replace(
                    next_frame, observation_loop=8 if problem == "duplicate_loop" else 7
                )
            elif problem == "map":
                next_frame = dataclasses.replace(next_frame, map_size=(4, 2))
            else:
                context = dataclasses.replace(
                    next_frame.context,
                    **({"sc2_version": "5.0"} if problem == "version" else {"player_id": 2}),
                )
                next_frame = dataclasses.replace(next_frame, context=context)
            writer.append(next_frame)
    assert not list(tmp_path.glob("*.manifest.json"))
    assert len(list(tmp_path.glob("*.partial"))) == 1
    assert len(list(tmp_path.glob("*.error.json"))) == 1


def test_interruption_is_preserved_as_incomplete(tmp_path):
    with SensorShardWriter(tmp_path, provenance=provenance()) as writer:
        writer.append(snapshot())
    records = list(tmp_path.glob("*.aborted.json"))
    assert json.loads(records[0].read_text())["state"] == "ABORTED"
    assert not list(tmp_path.glob("*.manifest.json"))


def test_limited_stream_and_empty_stream_cannot_be_finalized(tmp_path):
    with SensorShardWriter(tmp_path / "limited", provenance=provenance()) as writer:
        writer.append(snapshot())
        with pytest.raises(SensorStorageError, match="truncated"):
            writer.finalize(completion={"status": "MAX_LOOPS"})
    with SensorShardWriter(tmp_path / "empty", provenance=provenance()) as writer:
        with pytest.raises(SensorStorageError, match="empty"):
            writer.finalize(completion={"status": "STREAM_EXHAUSTED"})


def test_existing_artifacts_are_never_overwritten(tmp_path):
    path, manifest = write_shard(tmp_path)
    original = path.read_bytes()
    with pytest.raises(FileExistsError):
        write_shard(tmp_path)
    assert path.read_bytes() == original
    with SensorShardReader(path) as reader:
        reader.verify()
    assert len(list(tmp_path.glob("*.error.json"))) == 1


def test_shard_corruption_is_refused(tmp_path):
    path, manifest = write_shard(tmp_path)
    shard = tmp_path / manifest["shard"]
    content = bytearray(shard.read_bytes())
    content[50] ^= 1
    shard.write_bytes(content)
    with pytest.raises(SensorStorageError, match="hash"):
        SensorShardReader(path)


def test_manifest_corruption_is_refused(tmp_path):
    path, manifest = write_shard(tmp_path)
    manifest["frame_count"] = 999
    path.write_text(json.dumps(manifest))
    with pytest.raises(SensorStorageError, match="manifest digest"):
        SensorShardReader(path)


def test_census_measures_full_command_bundle_and_multi_unit_labels():
    labels = [
        {
            "command_index": index,
            "response_loop": 9 + index,
            "command_loop": 8 + index,
            "observation_loop": 8,
            "source_indices": [0, 1, 2],
            "ability_id": 80,
            "target_mode": "NONE",
            "command_mask": 1,
            "field_masks": {"ability": 1, "source": 1},
        }
        for index in (0, 1)
    ]
    row = {
        "split": "train",
        "replay_id": "r",
        "decision_loop": 8,
        "act_wait_class": "ACT_SUPPORTED",
        "command_count": 2,
        "unsupported_commands": 0,
        "labels": labels,
    }
    census = label_census([row])
    assert census["counts"]["commands_beyond_first"] == 1
    assert census["counts"]["first_command_sources_beyond_first"] == 2
    assert census["all_command_field_denominators"]["source"] == 2
    assert census["c2a2_projection_field_denominators"]["source"] == 1
    assert not census["failures"]
    labels[1]["response_loop"] = 8
    assert label_census([row])["failures"]["causal_window"] == 1
    with pytest.raises(ValueError, match="test supervision"):
        label_census([{**row, "split": "test"}])


@pytest.mark.parametrize("problem", [None, "engine_version", "replay_build", "participant"])
def test_replay_adapter_records_versions_refuses_mismatch_and_closes(
    tmp_path, monkeypatch, problem
):
    import os

    from poorstar.engine import replay_reader
    from poorstar.engine.replay_adapter import ReplaySensorAdapter

    instances = []

    class Reader:
        def __init__(self, *, sc2_path, version, commands_only):
            assert os.environ["SC2PATH"] == str(tmp_path.resolve())
            assert commands_only is True and version == "3.16.1"
            self.closed = False
            self._controller = SimpleNamespace(
                ping=lambda: SimpleNamespace(
                    game_version="5.0.0" if problem == "engine_version" else "3.16.1.55958",
                    base_build=55958,
                )
            )
            instances.append(self)

        def __enter__(self):
            return self

        def close(self):
            self.closed = True

        def replay_info(self, replay):
            return SimpleNamespace(
                base_build=999 if problem == "replay_build" else 55958,
                player_info=[
                    SimpleNamespace(
                        player_info=SimpleNamespace(player_id=2 if problem == "participant" else 1)
                    )
                ],
            )

        def ticks(self, replay, *, player_id, step_mul):
            assert player_id == 1 and step_mul == 1
            response, self.game_info = protocol_frame()
            yield SimpleNamespace(observation=response)

    monkeypatch.setattr(replay_reader, "ReplayReader", Reader)
    adapter = ReplaySensorAdapter(sc2_path=tmp_path, version="3.16.1", player_id=1, entity_cap=512)
    if problem:
        with pytest.raises(ValueError):
            with adapter:
                list(adapter.snapshots(tmp_path / "replay.SC2Replay"))
    else:
        with adapter:
            snapshots = list(adapter.snapshots(tmp_path / "replay.SC2Replay"))
            assert snapshots[0].metadata() == snapshot().metadata()
            assert adapter.engine_base_build == 55958 and adapter.engine_version == "3.16.1.55958"
    assert instances[0].closed
