#!/usr/bin/env python
"""Cli: engine: bind clock."""

import argparse
import gzip
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path

from poorstar.common.paths import source_root

TREE = source_root()
sys.path.insert(0, str(TREE))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--clock-trajectory", type=Path, required=True)
    p.add_argument("--query-trajectory", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    if args.output_dir.exists() or args.output_dir.resolve().is_relative_to(TREE):
        p.error("fresh external RAW_TRAJECTORY required")
    from s2clientprotocol import query_pb2, sc2api_pb2

    from poorstar.cli.engine.check_horizon import verified_report
    from poorstar.common.integrity import file_hash, payload_hash
    from poorstar.data.trajectory import decode_bytes, encode_bytes
    from poorstar.engine.protocol_wire import read_protocol_wire
    from poorstar.engine.query_evidence import ability_answers

    clock, cd = verified_report(args.clock_trajectory)
    query, qd = verified_report(args.query_trajectory)
    if (
        query["input_clock_digest"] != cd
        or query["replay_sha256"] != clock["replay_sha256"]
        or not query["all_response_wire_verified"]
    ):
        raise ValueError("original replay/clock/query provenance mismatch")
    with gzip.open(args.clock_trajectory / "original_events.jsonl.gz", "rt") as f:
        events = [json.loads(s) for s in f]
    event_loops = {
        e["_gameloop"]
        for e in events
        if e.get("_userid", {}).get("m_userId") == clock["controlled_user_id"]
        and (
            e["_event"].endswith("SCmdEvent")
            or "SCmdUpdateTarget" in e["_event"]
            or (e["_event"].endswith("SCommandManagerStateEvent") and e["m_state"] == 1)
        )
    }
    args.output_dir.mkdir(parents=True)
    for name, origin in [
        ("responses.protocol.gz", args.query_trajectory),
        ("original_events.jsonl.gz", args.clock_trajectory),
    ]:
        os.link(origin / name, args.output_dir / name)
    counts, controls = Counter(), Counter()
    seen = set()
    prior = {}
    changed = set()
    wire_digest = hashlib.sha256()
    with (
        gzip.open(args.query_trajectory / "queries.jsonl.gz", "rt") as f,
        gzip.open(args.output_dir / "engine_actions.jsonl.gz", "xt") as out,
    ):
        rows = (json.loads(s, object_hook=decode_bytes) for s in f)
        row = next(rows, None)
        for wire in read_protocol_wire(args.output_dir / "responses.protocol.gz"):
            raw = sc2api_pb2.ResponseObservation()
            raw.ParseFromString(wire)
            loop = int(raw.observation.game_loop)
            if (
                loop != counts["observations"]
                or raw.observation.player_common.player_id != clock["controlled_player_id"]
            ):
                raise ValueError("observation loop/player mismatch")
            counts["observations"] += 1
            wire_digest.update(wire)
            own = sorted(int(u.tag) for u in raw.observation.raw_data.units if u.alliance == 1)
            all_tags = {int(u.tag) for u in raw.observation.raw_data.units}
            seen.update(own)
            required = not raw.player_result and (loop % 8 == 0 or loop in event_loops)
            if required:
                if (
                    row is None
                    or row["loop"] != loop
                    or row["observation_sha256"] != hashlib.sha256(wire).hexdigest()
                ):
                    raise ValueError(
                        "required query missing or not bound to exact observation wire"
                    )
                if row["own_tags"] != own or row["absent_control"] in all_tags:
                    raise ValueError("query own/absent source mismatch")
                gone = row["gone_control"]
                if gone is not None and (gone not in seen or gone in all_tags):
                    raise ValueError("gone control lifecycle mismatch")
                control_tags = [row["absent_control"]] + ([gone] if gone is not None else [])
                modes = {}
                for item in row["queries"]:
                    request, response = query_pb2.RequestQuery(), query_pb2.ResponseQuery()
                    request.ParseFromString(item["request"])
                    response.ParseFromString(item["response"])
                    tags = [int(a.unit_tag) for a in request.abilities]
                    if tags != own + control_tags or item["mode"] in modes:
                        raise ValueError("complete source/request or duplicate mode mismatch")
                    if request.ignore_resource_requirements != (item["mode"] == "ignore_resources"):
                        raise ValueError("query resource mode mismatch")
                    modes[item["mode"]] = ability_answers(response, tags, control_tags=control_tags)
                    controls["query_rpc"] += 1
                    controls[item["mode"] + "_absent_nonempty"] += bool(
                        modes[item["mode"]][row["absent_control"]]
                    )
                    if gone is not None:
                        controls[item["mode"] + "_gone_queries"] += 1
                        controls[item["mode"] + "_gone_nonempty"] += bool(modes[item["mode"]][gone])
                if set(modes) != {"default", "ignore_resources", "repeat_default"}:
                    raise ValueError("required query mode absent")
                controls["repeat_comparisons"] += 1
                controls["repeat_mismatch"] += modes["default"] != modes["repeat_default"]
                for tag in own:
                    if tag in prior and prior[tag] != modes["default"][tag]:
                        changed.add(tag)
                    prior[tag] = modes["default"][tag]
                controls["query_frames"] += 1
                row = next(rows, None)
            elif row is not None and row["loop"] <= loop:
                raise ValueError("extra or terminal query row")
            for index, action in enumerate(raw.actions):
                counts["all_actions"] += 1
                record = dict(
                    response_loop=loop,
                    response_action_index=index,
                    terminal=bool(raw.player_result),
                    protocol_game_loop=int(action.game_loop)
                    if action.HasField("game_loop")
                    else None,
                    protocol_action_wire=encode_bytes(action.SerializeToString()),
                    kind="other",
                )
                if action.HasField("action_raw") and action.action_raw.HasField("unit_command"):
                    cmd = action.action_raw.unit_command
                    tags = list(map(int, cmd.unit_tags))
                    record.update(
                        kind="unit_command",
                        ability_id=int(cmd.ability_id),
                        sources=tags,
                        queue=bool(cmd.queue_command),
                        sources_in_current_own_frame=sum(t in own for t in tags),
                        target_tag=int(cmd.target_unit_tag)
                        if cmd.HasField("target_unit_tag")
                        else None,
                        position=[cmd.target_world_space_pos.x, cmd.target_world_space_pos.y]
                        if cmd.HasField("target_world_space_pos")
                        else None,
                    )
                    counts["unit_commands"] += 1
                    counts["commands_with_all_sources_currently_own"] += bool(tags) and all(
                        t in own for t in tags
                    )
                else:
                    counts["other_actions"] += 1
                out.write(json.dumps(record, sort_keys=True) + "\n")
        if (
            row is not None
            or not raw.player_result
            or counts["observations"] != query["counts"]["observations"]
        ):
            raise ValueError("full query/terminal lifecycle not exhausted")
    if counts["unit_commands"] != query["actual_unit_command_count"]:
        raise ValueError("unit command denominator mismatch")
    if any(controls[k] != query["counts"].get(k, 0) for k in controls):
        raise ValueError("independent control census differs")
    fidelity = (
        bool(changed)
        and controls["default_gone_queries"] > 0
        and not any(
            v for k, v in controls.items() if k.endswith("_nonempty") or k == "repeat_mismatch"
        )
    )
    result = {
        k: v
        for k, v in clock.items()
        if k not in ("artifacts", "counts", "source_manifest", "response_wire_sha256")
    }
    result.update(
        schema="poorstar_c3_same_run_query_clock_v1",
        input_query_digest=qd,
        input_original_clock_digest=cd,
        clock_origin="SAME_QUERY_RUN_FULL_RESPONSE_PROTOCOL_WIRE",
        counts=dict(counts),
        source_manifest={
            n: file_hash(TREE / n)
            for n in (
                "poorstar/cli/engine/bind_clock.py",
                "poorstar/engine/protocol_wire.py",
                "poorstar/engine/query_evidence.py",
            )
        },
        same_run_query_binding_verified=True,
        query_channel_fidelity_verified=fidelity,
        query_control_counts=dict(controls),
        changed_own_source_count=len(changed),
        cross_process_target_differences_resolved=False,
        response_wire_sha256=wire_digest.hexdigest(),
        artifacts={
            n: file_hash(args.output_dir / n)
            for n in (
                "original_events.jsonl.gz",
                "engine_actions.jsonl.gz",
                "responses.protocol.gz",
            )
        },
    )
    result["content_digest"] = payload_hash(result)
    with (args.output_dir / "report.json").open("x") as f:
        json.dump(result, f, indent=2, sort_keys=True)
    print(
        json.dumps(
            {
                k: v
                for k, v in result.items()
                if k
                not in ("source_manifest", "player_mapping", "original_event_counts", "artifacts")
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
