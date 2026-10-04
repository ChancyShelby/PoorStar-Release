#!/usr/bin/env python
"""Cli: engine: check horizon."""

import argparse
import gzip
import json
from collections import Counter, defaultdict
from pathlib import Path

from poorstar.common.paths import source_root

TREE = source_root()


def verified_report(root):
    from poorstar.common.integrity import file_hash, payload_hash

    report = json.loads((root / "report.json").read_text())
    digest = report.pop("content_digest")
    if payload_hash(report) != digest:
        raise ValueError("input report digest mismatch")
    for name, sha in report["artifacts"].items():
        if file_hash(root / name) != sha:
            raise ValueError("input artifact drift")
    return report, digest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--clock-trajectory", type=Path, required=True)
    p.add_argument("--query-trajectory", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists() or args.output.resolve().is_relative_to(TREE):
        p.error("fresh external evidence output required")
    from s2clientprotocol import query_pb2, sc2api_pb2

    from poorstar.actions.abilities import canonical_ability
    from poorstar.common.integrity import file_hash, payload_hash
    from poorstar.data.trajectory import decode_bytes
    from poorstar.engine.query_evidence import ability_answers

    clock, cd = verified_report(args.clock_trajectory)
    query, qd = verified_report(args.query_trajectory)
    independent = (
        query["complete_unit_command_stream_identical"] and query["input_clock_digest"] == cd
    )
    same_run = (
        clock.get("input_query_digest") == qd
        and clock.get("same_run_query_binding_verified") is True
        and clock["artifacts"]["responses.protocol.gz"]
        == query["artifacts"]["responses.protocol.gz"]
    )
    if not (independent or same_run):
        raise ValueError("query and clock do not belong to same complete command stream")
    catalog = sc2api_pb2.ResponseData()
    catalog.ParseFromString((args.query_trajectory / "ability_catalog.protocol").read_bytes())
    remaps = {int(x.ability_id): int(x.remaps_to_ability_id) for x in catalog.abilities}
    if len(remaps) != len(catalog.abilities):
        raise ValueError("duplicate catalog abilities")
    unknown = set()

    def identity(ability):
        try:
            return canonical_ability(ability, remaps)
        except ValueError:
            unknown.add(ability)
            return None

    commands = defaultdict(list)
    with gzip.open(args.clock_trajectory / "engine_actions.jsonl.gz", "rt") as f:
        for line in f:
            command = json.loads(line)
            if command["kind"] == "unit_command":
                commands[command["response_loop"] - 1].append(command)
    counters = defaultdict(Counter)
    per_ability = defaultdict(Counter)
    transition = Counter()
    grid, grid_loop = None, None
    processed = 0
    previous = -1
    with gzip.open(args.query_trajectory / "queries.jsonl.gz", "rt") as f:
        for line in f:
            row = json.loads(line, object_hook=decode_bytes)
            loop = row["loop"]
            if loop <= previous:
                raise ValueError("query frames are not strictly ordered")
            previous = loop
            frame = {}
            for item in row["queries"]:
                request, response = query_pb2.RequestQuery(), query_pb2.ResponseQuery()
                request.ParseFromString(item["request"])
                response.ParseFromString(item["response"])
                tags = [int(x.unit_tag) for x in request.abilities]
                if tags[: len(row["own_tags"])] != row["own_tags"]:
                    raise ValueError("stored own/request identity mismatch")
                answers = ability_answers(
                    response, tags, control_tags=[row["absent_control"], row["gone_control"]]
                )
                if request.ignore_resource_requirements != (item["mode"] == "ignore_resources"):
                    raise ValueError("query mode mismatch")
                frame[item["mode"]] = {
                    t: ({a for a, _ in answers[t]}, {identity(a) for a, _ in answers[t]} - {None})
                    for t in row["own_tags"]
                }
            if loop % 8 == 0:
                grid, grid_loop = frame, loop
            for command in commands.pop(loop, []):
                if grid is None or grid_loop != loop // 8 * 8:
                    raise ValueError("required exact grid query missing")
                processed += 1
                ability = command["ability_id"]
                canonical = identity(ability)
                source = command["sources"]
                if not source or len(set(source)) != len(source):
                    raise ValueError("empty or duplicated raw command source set")
                metrics = {}
                for anchor, current in [
                    ("event_frame_f", frame),
                    ("legacy_grid_floor_f_by_8", grid),
                ]:
                    for mode in ("default", "ignore_resources"):
                        count = counters[anchor + "_" + mode]
                        answers = current[mode]
                        count["commands"] += 1
                        count["source_units"] += len(source)
                        present = [t in answers for t in source]
                        exact = [t in answers and ability in answers[t][0] for t in source]
                        mapped = [
                            t in answers and canonical is not None and canonical in answers[t][1]
                            for t in source
                        ]
                        count["commands_all_sources_present"] += all(present)
                        count["source_units_absent"] += sum(not x for x in present)
                        count["commands_all_sources_exact_available"] += all(exact)
                        count["commands_any_source_exact_available"] += any(exact)
                        count["source_units_exact_unavailable"] += sum(not x for x in exact)
                        count["commands_all_sources_catalog_available"] += all(mapped)
                        count["commands_any_source_catalog_available"] += any(mapped)
                        count["commands_without_any_catalog_available_source"] += not any(mapped)
                        count["source_units_catalog_unavailable"] += sum(not x for x in mapped)
                        count["commands_catalog_unknown"] += canonical is None
                        per_ability[ability][anchor + "_" + mode + "_all_available"] += all(mapped)
                        per_ability[ability][anchor + "_" + mode + "_no_available"] += not any(
                            mapped
                        )
                        metrics[(anchor, mode)] = mapped
                per_ability[ability]["commands"] += 1
                transition["commands"] += 1
                for mode in ("default", "ignore_resources"):
                    old, now = (
                        metrics[("legacy_grid_floor_f_by_8", mode)],
                        metrics[("event_frame_f", mode)],
                    )
                    transition[mode + "_any_missing_at_grid_available_at_event"] += any(
                        b and not a for a, b in zip(old, now)
                    )
                    transition[mode + "_all_available_at_event_but_not_grid"] += all(
                        now
                    ) and not all(old)
    if commands or processed != clock["counts"]["unit_commands"]:
        raise ValueError("command denominator not fully exhausted")
    result = dict(
        schema="poorstar_c3_query_horizon_audit_v1",
        clock_digest=cd,
        query_digest=qd,
        counts={k: dict(v) for k, v in counters.items()},
        transitions=dict(transition),
        by_ability={str(k): dict(v) for k, v in per_ability.items()},
        catalog_ability_count=len(remaps),
        unknown_catalog_ids=sorted(unknown),
        label_filtering_applied=False,
        causal_observation_phase_verified=False,
        query_channel_fidelity_verified=(
            clock["query_channel_fidelity_verified"]
            if same_run
            else query["query_channel_fidelity_verified"]
        ),
        query_binding="SAME_RUN_FULL_WIRE" if same_run else "INDEPENDENT_EXACT_ACTION_WIRE",
        current_or_future_mask_adopted=False,
        target_queue_legality_verified=False,
        training_ready=False,
        rl_ready=False,
        source_manifest={
            n: file_hash(TREE / n)
            for n in (
                "poorstar/cli/engine/check_horizon.py",
                "poorstar/actions/abilities.py",
                "poorstar/engine/query_evidence.py",
            )
        },
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
