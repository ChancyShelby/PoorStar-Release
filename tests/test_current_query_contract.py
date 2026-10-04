"""Tests: test current query contract."""

import gzip
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest
from s2clientprotocol import query_pb2, sc2api_pb2

from poorstar.common.integrity import file_hash, payload_hash
from poorstar.data.trajectory import encode_bytes
from poorstar.engine.protocol_wire import ProtocolWireWriter
from poorstar.engine.query_masks import (
    absent_control_class,
    availability_counts,
    choose_absent_own_control,
    current_own_answers,
)


def test_global_any_is_distinct_from_selected_any_and_selected_all():
    answers = {10: (), 20: ((21, False),), 30: ((11, True),)}
    remaps = {11: 21, 21: 0}
    a = availability_counts(11, [10], answers, remaps)
    assert a["commands_global_current_own_available"] == 1
    assert a["commands_selected_any_available"] == a["commands_selected_all_available"] == 0
    assert a["commands_global_available_but_selected_none"] == 1
    a = availability_counts(11, [20, 30, 99], answers, remaps)
    assert a["commands_selected_any_available"] == 1 and a["commands_selected_all_available"] == 0
    assert (
        a["source_units"] == 3
        and a["source_units_available"] == 2
        and a["source_units_absent"] == 1
    )
    assert a["source_units_present_unavailable"] == 0


def test_controls_and_other_players_cannot_grant_current_source_masks():
    observation = sc2api_pb2.Observation()
    observation.raw_data.units.add(tag=10, alliance=1, display_type=1, unit_type=104)
    observation.raw_data.units.add(tag=20, alliance=1, display_type=2, unit_type=104)
    observation.raw_data.units.add(tag=30, alliance=4, display_type=1, unit_type=104)
    request, response = query_pb2.RequestQuery(), query_pb2.ResponseQuery()
    for tag in (10, 20, 99):
        request.abilities.add(unit_tag=tag)
        answer = response.abilities.add(unit_tag=tag, unit_type_id=104)
        if tag != 10:
            answer.abilities.add(ability_id=11)
    answers = current_own_answers(observation, request, response, control_tags=[99])
    assert answers == {10: ()}
    assert not availability_counts(11, [10], answers, {11: 0})[
        "commands_global_current_own_available"
    ]
    response.abilities[0].unit_type_id = 105
    with pytest.raises(ValueError, match="type identity"):
        current_own_answers(observation, request, response, control_tags=[99])
    response.abilities[0].ClearField("unit_type_id")
    with pytest.raises(ValueError, match="type identity"):
        current_own_answers(observation, request, response, control_tags=[99])


def test_no_guessing_unknown_abilities_or_partial_source_sets():
    with pytest.raises(ValueError, match="catalog"):
        availability_counts(99, [10], {10: ()}, {11: 0})
    with pytest.raises(ValueError, match="unique"):
        availability_counts(11, [10, 10], {10: ()}, {11: 0})
    with pytest.raises(ValueError, match="nonempty"):
        availability_counts(11, [], {10: ()}, {11: 0})


def test_temporary_absence_is_not_confirmed_death():
    assert (
        absent_control_class(observed_dead_before=False, later_reappearance=True)
        == "TEMPORARILY_ABSENT"
    )
    assert (
        absent_control_class(observed_dead_before=True, later_reappearance=False)
        == "OBSERVED_DEAD_NO_REAPPEARANCE"
    )
    assert (
        absent_control_class(observed_dead_before=False, later_reappearance=False)
        == "UNCLASSIFIED_ABSENCE"
    )
    assert (
        absent_control_class(observed_dead_before=True, later_reappearance=True)
        == "DEAD_THEN_REAPPEARED"
    )


def test_dead_control_preference_is_causal_and_preserves_v1_fallback():
    seen, raw, deaths = {10, 20, 30}, {20}, {20, 30, 99}
    assert choose_absent_own_control(seen, raw, deaths) == 10
    assert choose_absent_own_control(seen, raw, deaths, prefer_observed_dead=True) == 30
    assert choose_absent_own_control(seen, raw, {99}, prefer_observed_dead=True) == 10
    assert choose_absent_own_control(seen, seen, deaths, prefer_observed_dead=True) is None


def artifact(root, **fields):
    fields["artifacts"] = {p.name: file_hash(p) for p in root.iterdir()}
    fields["content_digest"] = payload_hash(fields)
    (root / "report.json").write_text(json.dumps(fields))
    return fields["content_digest"]


def run_audit(tmp_path, *, bad_type=False, omit_dead=False):
    clock, query = tmp_path / "clock", tmp_path / "query"
    clock.mkdir()
    query.mkdir()
    catalog = sc2api_pb2.ResponseData()
    catalog.abilities.add(ability_id=11, remaps_to_ability_id=21, target=1)
    catalog.abilities.add(ability_id=21, target=1)
    (query / "ability_catalog.protocol").write_bytes(catalog.SerializeToString())
    writer = ProtocolWireWriter(query / "responses.protocol.gz")
    digest = hashlib.sha256()
    rows = []
    absent = 1 << 63
    for loop in range(4):
        raw = sc2api_pb2.ResponseObservation()
        raw.observation.game_loop = loop
        raw.observation.player_common.player_id = 1
        own = [10, 20, 40] if loop == 0 else [20] if loop == 2 else [10, 20]
        for tag in own:
            raw.observation.raw_data.units.add(tag=tag, alliance=1, display_type=1, unit_type=104)
        if loop == 1 and not omit_dead:
            raw.observation.raw_data.event.dead_units.append(40)
        if loop in (1, 2):
            command = raw.actions.add().action_raw.unit_command
            command.ability_id = 11
            command.unit_tags.append(10)
        if loop == 3:
            raw.player_result.add(player_id=1, result=1)
        wire = raw.SerializeToString()
        writer.append(wire)
        digest.update(wire)
        if loop == 3:
            continue
        gone = None if loop == 0 else 40 if loop == 1 else 10
        row = dict(
            loop=loop,
            own_tags=own,
            absent_control=absent,
            gone_control=gone,
            observation_sha256=hashlib.sha256(wire).hexdigest(),
            queries=[],
        )
        tags = own + [absent] + ([] if gone is None else [gone])
        for mode in ("default", "ignore_resources", "repeat_default"):
            request, response = (
                query_pb2.RequestQuery(ignore_resource_requirements=mode == "ignore_resources"),
                query_pb2.ResponseQuery(),
            )
            for tag in tags:
                request.abilities.add(unit_tag=tag)
                answer = response.abilities.add(
                    unit_tag=tag, unit_type_id=105 if bad_type and tag == 10 else 104
                )
                if tag == 20 or (tag == 10 and loop > 0):
                    answer.abilities.add(ability_id=21)
            row["queries"].append(
                dict(
                    mode=mode,
                    request=encode_bytes(request.SerializeToString()),
                    response=encode_bytes(response.SerializeToString()),
                )
            )
        rows.append(row)
    writer.close(complete=True)
    with gzip.open(query / "queries.jsonl.gz", "wt") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    qd = artifact(query, counts=dict(observations=4, query_frames=3))
    # Binding is an independently tested prerequisite; this fixture supplies its digest.
    (clock / "responses.protocol.gz").write_bytes((query / "responses.protocol.gz").read_bytes())
    artifact(
        clock,
        input_query_digest=qd,
        same_run_query_binding_verified=True,
        controlled_player_id=1,
        counts=dict(unit_commands=2),
        response_wire_sha256=digest.hexdigest(),
        query_channel_fidelity_verified=False,
    )
    output = tmp_path / "report.json"
    process = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve().parents[1] / "poorstar/cli/engine/check_queries.py"),
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


def test_complete_wire_new_qualification_preserves_original_failure_and_every_command(tmp_path):
    process, output = run_audit(tmp_path)
    assert process.returncode == 0, process.stderr
    report = json.loads(output.read_text())
    assert report["counts"]["unit_commands"] == 2 and report["counts"]["observations"] == 4
    assert (
        report["lifecycle_qualified_current_query_fidelity"]
        and not report["original_strict_query_fidelity"]
    )
    assert report["absent_control_lifecycle"]["TEMPORARILY_ABSENT"]["default_nonempty"] == 1
    assert (
        report["absent_control_lifecycle"]["OBSERVED_DEAD_NO_REAPPEARANCE"]["default_queries"] == 1
    )
    assert report["masks"]["event_frame_f_default"]["commands_global_current_own_available"] == 2
    assert report["masks"]["event_frame_f_default"]["commands_selected_all_available"] == 1
    assert report["masks"]["grid_floor_f_by_8_default"]["commands_selected_all_available"] == 0
    assert report["transitions"]["default_selected_all_event_available_grid_unavailable"] == 1
    assert report["transitions"]["default_global_current_own_event_available_grid_unavailable"] == 0
    assert (
        not report["current_mask_adopted"]
        and not report["label_filtering_applied"]
        and not report["training_ready"]
    )


def test_no_dead_controls_is_not_a_pass(tmp_path):
    process, output = run_audit(tmp_path, omit_dead=True)
    assert process.returncode == 0, process.stderr
    assert not json.loads(output.read_text())["lifecycle_qualified_current_query_fidelity"]


def test_wrong_current_unit_type_refuses_report(tmp_path):
    process, output = run_audit(tmp_path, bad_type=True)
    assert process.returncode != 0 and "type identity mismatch" in process.stderr
    assert not output.exists()
