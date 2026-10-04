"""Tests: test query horizon."""

import gzip
import json
import subprocess
import sys
from pathlib import Path

from s2clientprotocol import query_pb2, sc2api_pb2

from poorstar.common.integrity import file_hash, payload_hash
from poorstar.data.trajectory import encode_bytes


def artifact(root, **report):
    report["artifacts"] = {p.name: file_hash(p) for p in root.iterdir()}
    report["content_digest"] = payload_hash(report)
    (root / "report.json").write_text(json.dumps(report))
    return report["content_digest"]


def fixture(tmp_path, *, include_event=True):
    clock, query = tmp_path / "clock", tmp_path / "query"
    clock.mkdir()
    query.mkdir()
    with gzip.open(clock / "engine_actions.jsonl.gz", "wt") as f:
        for ability in [11, 99]:
            f.write(
                json.dumps(
                    dict(kind="unit_command", response_loop=8, ability_id=ability, sources=[1, 2])
                )
                + "\n"
            )
    cd = artifact(clock, counts={"unit_commands": 2})
    data = sc2api_pb2.ResponseData()
    data.abilities.add(ability_id=11, remaps_to_ability_id=21)
    data.abilities.add(ability_id=21)
    (query / "ability_catalog.protocol").write_bytes(data.SerializeToString())
    with gzip.open(query / "queries.jsonl.gz", "wt") as f:
        for loop, own in [(0, [1])] + ([(7, [1, 2])] if include_event else []):
            tags = own + [1 << 63]
            row = dict(
                loop=loop, own_tags=own, absent_control=1 << 63, gone_control=None, queries=[]
            )
            for mode in ("default", "ignore_resources", "repeat_default"):
                request = query_pb2.RequestQuery(
                    ignore_resource_requirements=mode == "ignore_resources"
                )
                response = query_pb2.ResponseQuery()
                for tag in tags:
                    request.abilities.add(unit_tag=tag)
                    answer = response.abilities.add(unit_tag=tag)
                    if tag in own:
                        answer.abilities.add(ability_id=11 if loop == 0 else 21)
                row["queries"].append(
                    dict(
                        mode=mode,
                        request=encode_bytes(request.SerializeToString()),
                        response=encode_bytes(response.SerializeToString()),
                    )
                )
            f.write(json.dumps(row) + "\n")
    artifact(
        query,
        input_clock_digest=cd,
        complete_unit_command_stream_identical=True,
        query_channel_fidelity_verified=True,
    )
    output = tmp_path / "audit.json"
    process = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve().parents[1] / "poorstar/cli/engine/check_horizon.py"),
            "--clock-trajectory",
            str(clock),
            "--query-trajectory",
            str(query),
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
    )
    return process, output


def test_grid_horizon_conflict_and_remapping_keep_every_source(tmp_path):
    process, output = fixture(tmp_path)
    assert process.returncode == 0, process.stderr
    r = json.loads(output.read_text())
    old = r["counts"]["legacy_grid_floor_f_by_8_default"]
    now = r["counts"]["event_frame_f_default"]
    assert old["commands"] == now["commands"] == 2
    assert old["source_units"] == 4 and old["source_units_absent"] == 2
    assert now["commands_all_sources_exact_available"] == 0
    assert now["commands_all_sources_catalog_available"] == 1
    assert r["transitions"]["default_all_available_at_event_but_not_grid"] == 1
    assert r["unknown_catalog_ids"] == [99]
    assert not r["label_filtering_applied"] and not r["training_ready"]


def test_missing_event_query_cannot_silently_drop_commands(tmp_path):
    process, output = fixture(tmp_path, include_event=False)
    assert process.returncode != 0
    assert "denominator not fully exhausted" in process.stderr
    assert not output.exists()
