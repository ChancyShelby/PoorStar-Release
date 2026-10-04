#!/usr/bin/env python
"""Cli: engine: check queries."""

import argparse
import gzip
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from poorstar.common.paths import source_root

TREE = source_root()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clock-trajectory", type=Path, required=True)
    parser.add_argument("--query-trajectory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.resolve().is_relative_to(TREE):
        parser.error("fresh external evidence output required")
    from s2clientprotocol import query_pb2, sc2api_pb2

    from poorstar.cli.engine.check_horizon import verified_report
    from poorstar.common.integrity import file_hash, payload_hash
    from poorstar.data.trajectory import decode_bytes
    from poorstar.engine.protocol_wire import read_protocol_wire
    from poorstar.engine.query_evidence import ability_answers
    from poorstar.engine.query_masks import (
        absent_control_class,
        availability_counts,
        current_own_answers,
    )

    names = (
        "poorstar/cli/engine/check_queries.py",
        "poorstar/engine/query_masks.py",
        "poorstar/engine/query_evidence.py",
        "poorstar/actions/abilities.py",
        "poorstar/engine/protocol_wire.py",
        "poorstar/cli/engine/check_horizon.py",
    )
    source = {name: file_hash(TREE / name) for name in names}
    clock, cd = verified_report(args.clock_trajectory)
    query, qd = verified_report(args.query_trajectory)
    if (
        clock.get("input_query_digest") != qd
        or not clock.get("same_run_query_binding_verified")
        or clock["artifacts"]["responses.protocol.gz"]
        != query["artifacts"]["responses.protocol.gz"]
    ):
        raise ValueError("verified exact SAME-RUN clock/query binding required")
    catalog = sc2api_pb2.ResponseData()
    catalog.ParseFromString((args.query_trajectory / "ability_catalog.protocol").read_bytes())
    remaps = {int(x.ability_id): int(x.remaps_to_ability_id) for x in catalog.abilities}
    abilities = {int(x.ability_id): x for x in catalog.abilities}
    if len(remaps) != len(catalog.abilities):
        raise ValueError("duplicate engine catalog ability")
    counters = Counter()
    masks, per_ability = defaultdict(Counter), defaultdict(Counter)
    transitions, lengths, bundles = Counter(), Counter(), Counter()
    seen_own, last_own, death_loop = set(), {}, {}
    absent_controls, gone_records = set(), []
    prior, changed = {}, set()
    grid = previous_query = None
    previous_loop = None
    wire_digest = hashlib.sha256()
    with gzip.open(args.query_trajectory / "queries.jsonl.gz", "rt") as f:
        rows = (json.loads(line, object_hook=decode_bytes) for line in f)
        row = next(rows, None)
        for wire in read_protocol_wire(args.query_trajectory / "responses.protocol.gz"):
            raw = sc2api_pb2.ResponseObservation()
            raw.ParseFromString(wire)
            loop = int(raw.observation.game_loop)
            if (
                loop != counters["observations"]
                or raw.observation.player_common.player_id != clock["controlled_player_id"]
            ):
                raise ValueError("full observation loop/player mismatch")
            counters["observations"] += 1
            wire_digest.update(wire)
            units = {int(u.tag): u for u in raw.observation.raw_data.units}
            if len(units) != len(raw.observation.raw_data.units):
                raise ValueError("duplicate current raw units")
            own = {tag for tag, u in units.items() if u.alliance == 1}
            seen_own.update(own)
            for tag in own:
                last_own[tag] = loop
            for tag in raw.observation.raw_data.event.dead_units:
                if tag in seen_own:
                    death_loop.setdefault(int(tag), loop)
            command_count = 0
            for action in raw.actions:
                counters["all_actions"] += 1
                if not action.HasField("action_raw") or not action.action_raw.HasField(
                    "unit_command"
                ):
                    counters["other_actions"] += 1
                    continue
                command = action.action_raw.unit_command
                if (
                    previous_loop != loop - 1
                    or previous_query is None
                    or grid is None
                    or grid[0] != (loop - 1) // 8 * 8
                ):
                    raise ValueError("command lacks exact candidate event/grid query")
                command_count += 1
                counters["unit_commands"] += 1
                sources = list(map(int, command.unit_tags))
                lengths[len(sources)] += 1
                ability = int(command.ability_id)
                values = {}
                for anchor, frame in [
                    ("event_frame_f", previous_query),
                    ("grid_floor_f_by_8", grid[1]),
                ]:
                    for mode in ("default", "ignore_resources"):
                        metrics = availability_counts(ability, sources, frame[mode], remaps)
                        masks[anchor + "_" + mode].update(metrics)
                        per_ability[ability].update(
                            {
                                anchor + "_" + mode + "_" + key: value
                                for key, value in metrics.items()
                            }
                        )
                        values[anchor, mode] = metrics
                per_ability[ability]["commands"] += 1
                for mode in ("default", "ignore_resources"):
                    now, old = values["event_frame_f", mode], values["grid_floor_f_by_8", mode]
                    for level in ("global_current_own", "selected_any", "selected_all"):
                        key = "commands_" + level + "_available"
                        transitions[mode + "_" + level + "_event_available_grid_unavailable"] += (
                            bool(now[key] and not old[key])
                        )
                target = abilities.get(ability)
                kind = (
                    "entity"
                    if command.HasField("target_unit_tag")
                    else ("position" if command.HasField("target_world_space_pos") else "none")
                )
                allowed = {"none": (1, 5), "position": (2, 4, 5), "entity": (3, 4)}[kind]
                counters["target_kind_" + kind] += 1
                if target is None or not target.HasField("target"):
                    counters["catalog_target_type_absent"] += 1
                elif target.target not in allowed:
                    counters["catalog_target_type_incompatible"] += 1
                    per_ability[ability]["catalog_target_type_incompatible_" + kind] += 1
            if command_count:
                bundles[command_count] += 1
            current = None
            if row is not None and row["loop"] == loop:
                if (
                    raw.player_result
                    or row["observation_sha256"] != hashlib.sha256(wire).hexdigest()
                ):
                    raise ValueError("query not bound to exact nonterminal observation")
                if row["own_tags"] != sorted(own):
                    raise ValueError("stored complete current own identity mismatch")
                absent, gone = row["absent_control"], row["gone_control"]
                absent_controls.add(absent)
                controls = [absent] + ([gone] if gone is not None else [])
                if absent in units or (
                    gone is not None and (gone not in seen_own or gone in units)
                ):
                    raise ValueError("negative control lifecycle mismatch")
                current, full = {}, {}
                for item in row["queries"]:
                    mode = item["mode"]
                    request, response = query_pb2.RequestQuery(), query_pb2.ResponseQuery()
                    request.ParseFromString(item["request"])
                    response.ParseFromString(item["response"])
                    if mode in current or request.ignore_resource_requirements != (
                        mode == "ignore_resources"
                    ):
                        raise ValueError("query mode duplicate or resource mismatch")
                    current[mode] = current_own_answers(
                        raw.observation, request, response, control_tags=controls
                    )
                    tags = [int(a.unit_tag) for a in request.abilities]
                    full[mode] = ability_answers(response, tags, control_tags=controls)
                    counters["query_rpc"] += 1
                    counters["current_own_type_checks"] += len(own)
                    counters[mode + "_nonexistent_queries"] += 1
                    counters[mode + "_nonexistent_nonempty"] += bool(full[mode][absent])
                    if gone is not None:
                        gone_records.append(
                            (
                                loop,
                                gone,
                                mode,
                                bool(full[mode][gone]),
                                death_loop.get(gone, loop + 1) <= loop,
                            )
                        )
                if set(current) != {"default", "ignore_resources", "repeat_default"}:
                    raise ValueError("full three-mode query required")
                counters["repeat_comparisons"] += 1
                counters["repeat_mismatch"] += full["default"] != full["repeat_default"]
                for tag, skills in current["default"].items():
                    if tag in prior and prior[tag] != skills:
                        changed.add(tag)
                    prior[tag] = skills
                counters["query_frames"] += 1
                row = next(rows, None)
            elif row is not None and row["loop"] < loop:
                raise ValueError("persisted query schedule skipped")
            if loop % 8 == 0 and not raw.player_result:
                if current is None:
                    raise ValueError("grid query missing")
                grid = (loop, current)
            previous_query, previous_loop = current, loop
        if (
            row is not None
            or not raw.player_result
            or counters["observations"] != query["counts"]["observations"]
        ):
            raise ValueError("terminal/full query lifecycle incomplete")
    if (
        counters["unit_commands"] != clock["counts"]["unit_commands"]
        or counters["query_frames"] != query["counts"]["query_frames"]
        or wire_digest.hexdigest() != clock["response_wire_sha256"]
    ):
        raise ValueError("full independent command/query/wire denominator mismatch")
    if absent_controls & seen_own:
        raise ValueError("nonexistent control later became a current own unit")
    lifecycle = defaultdict(Counter)
    for loop, tag, mode, nonempty, dead in gone_records:
        category = absent_control_class(
            observed_dead_before=dead, later_reappearance=last_own.get(tag, -1) > loop
        )
        lifecycle[category][mode + "_queries"] += 1
        lifecycle[category][mode + "_nonempty"] += nonempty
    dead = lifecycle["OBSERVED_DEAD_NO_REAPPEARANCE"]
    qualified = (
        bool(changed)
        and counters["repeat_comparisons"] > 0
        and not counters["repeat_mismatch"]
        and all(
            counters[mode + "_nonexistent_queries"] > 0
            and not counters[mode + "_nonexistent_nonempty"]
            and dead[mode + "_queries"] > 0
            and not dead[mode + "_nonempty"]
            for mode in ("default", "ignore_resources", "repeat_default")
        )
        and not lifecycle["DEAD_THEN_REAPPEARED"]
    )
    for name, sha in source.items():
        if file_hash(TREE / name) != sha:
            raise ValueError("source changed during audit")
    verified_report(args.clock_trajectory)
    verified_report(args.query_trajectory)
    result = dict(
        schema="poorstar_c3_current_query_contract_audit_v1",
        clock_digest=cd,
        query_digest=qd,
        counts=dict(counters),
        masks={k: dict(v) for k, v in masks.items()},
        by_ability={str(k): dict(v) for k, v in per_ability.items()},
        full_selection_lengths={str(k): v for k, v in sorted(lengths.items())},
        full_bundle_lengths={str(k): v for k, v in sorted(bundles.items())},
        transitions=dict(transitions),
        absent_control_lifecycle={k: dict(v) for k, v in lifecycle.items()},
        changed_current_own_source_count=len(changed),
        original_strict_query_fidelity=clock["query_channel_fidelity_verified"],
        lifecycle_qualified_current_query_fidelity=qualified,
        all_current_own_answer_types_verified=True,
        same_run_wire_binding_verified=True,
        full_command_denominators_preserved=True,
        label_filtering_applied=False,
        future_mask_adopted=False,
        current_mask_adopted=False,
        target_queue_legality_verified=False,
        training_ready=False,
        rl_ready=False,
        source_manifest=source,
    )
    result["content_digest"] = payload_hash(result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as f:
        json.dump(result, f, indent=2, sort_keys=True)
    print(
        json.dumps(
            {k: v for k, v in result.items() if k not in ("by_ability", "source_manifest")},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
