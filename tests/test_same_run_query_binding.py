"""Tests: test same run query binding."""

import gzip
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from s2clientprotocol import query_pb2, sc2api_pb2

from poorstar.common.integrity import file_hash, payload_hash
from poorstar.data.trajectory import encode_bytes
from poorstar.engine.protocol_wire import ProtocolWireWriter


def report(root, **fields):
    fields["artifacts"] = {p.name: file_hash(p) for p in root.iterdir()}
    digest = payload_hash(fields)
    fields["content_digest"] = digest
    (root / "report.json").write_text(json.dumps(fields))
    return digest


def inputs(tmp_path, *, bad_digest=False):
    clock, query = tmp_path / "old_clock", tmp_path / "query"
    clock.mkdir()
    query.mkdir()
    with gzip.open(clock / "original_events.jsonl.gz", "wt") as f:
        f.write(
            json.dumps(dict(_event="NNet.Game.SCmdEvent", _gameloop=0, _userid={"m_userId": 0}))
            + "\n"
        )
    cd = report(
        clock,
        replay_sha256="a" * 64,
        controlled_player_id=1,
        controlled_user_id=0,
        counts={"observations": 2, "unit_commands": 1},
        all_response_wire_verified=True,
        timestamp_alignment_verified=False,
        training_ready=False,
        rl_ready=False,
    )
    initial = sc2api_pb2.ResponseObservation()
    initial.observation.game_loop = 0
    initial.observation.player_common.player_id = 1
    initial.observation.raw_data.units.add(tag=12, alliance=1, display_type=1, unit_type=151)
    terminal = sc2api_pb2.ResponseObservation()
    terminal.CopyFrom(initial)
    terminal.observation.game_loop = 1
    terminal.player_result.add(player_id=1, result=1)
    terminal.actions.add().action_raw.unit_command.unit_tags.append(12)
    terminal.actions[0].action_raw.unit_command.ability_id = 11
    writer = ProtocolWireWriter(query / "responses.protocol.gz")
    for raw in [initial, terminal]:
        writer.append(raw.SerializeToString())
    writer.close(complete=True)
    absent = 1 << 63
    row = dict(
        loop=0,
        own_tags=[12],
        absent_control=absent,
        gone_control=None,
        queries=[],
        observation_sha256="bad"
        if bad_digest
        else hashlib.sha256(initial.SerializeToString()).hexdigest(),
    )
    for mode in ("default", "ignore_resources", "repeat_default"):
        request = query_pb2.RequestQuery(ignore_resource_requirements=mode == "ignore_resources")
        response = query_pb2.ResponseQuery()
        for tag in [12, absent]:
            request.abilities.add(unit_tag=tag)
            a = response.abilities.add(unit_tag=tag)
            if tag == 12:
                a.abilities.add(ability_id=11)
        row["queries"].append(
            dict(
                mode=mode,
                request=encode_bytes(request.SerializeToString()),
                response=encode_bytes(response.SerializeToString()),
            )
        )
    with gzip.open(query / "queries.jsonl.gz", "wt") as f:
        f.write(json.dumps(row) + "\n")
    counts = dict(
        observations=2, query_frames=1, query_rpc=3, repeat_comparisons=1, repeat_mismatch=0
    )
    report(
        query,
        input_clock_digest=cd,
        replay_sha256="a" * 64,
        all_response_wire_verified=True,
        actual_unit_command_count=1,
        counts=counts,
        complete_unit_command_stream_identical=False,
    )
    return clock, query


def run(tmp_path, *, bad_digest=False):
    clock, query = inputs(tmp_path, bad_digest=bad_digest)
    output = tmp_path / "bound"
    process = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve().parents[1] / "poorstar/cli/engine/bind_clock.py"),
            "--clock-trajectory",
            str(clock),
            "--query-trajectory",
            str(query),
            "--output-dir",
            str(output),
        ],
        capture_output=True,
        text=True,
    )
    return process, output


def test_terminal_actions_and_same_run_pointer_identity_are_preserved(tmp_path):
    process, output = run(tmp_path)
    assert process.returncode == 0, process.stderr
    r = json.loads((output / "report.json").read_text())
    assert r["same_run_query_binding_verified"]
    assert r["counts"]["observations"] == 2 and r["counts"]["unit_commands"] == 1
    # One static frame cannot certify changing-state/disappearance controls.
    assert not r["query_channel_fidelity_verified"] and not r["training_ready"]
    with gzip.open(output / "engine_actions.jsonl.gz", "rt") as f:
        action = json.loads(next(f))
    assert action["terminal"] and action["sources"] == [12] and action["response_loop"] == 1


def test_query_observation_digest_mismatch_refuses_report(tmp_path):
    process, output = run(tmp_path, bad_digest=True)
    assert process.returncode != 0 and "exact observation wire" in process.stderr
    assert not (output / "report.json").exists()
