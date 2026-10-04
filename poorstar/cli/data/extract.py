#!/usr/bin/env python
"""Extract one replay player into causal BC data shards."""

import argparse
import gzip
import hashlib
import io
import json
import os
from pathlib import Path

from poorstar.common.paths import source_root

TREE = source_root()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--replay", type=Path, required=True)
    p.add_argument("--player-id", type=int, required=True)
    p.add_argument("--sc2-path", type=Path, required=True)
    p.add_argument("--split-manifest", type=Path, required=True)
    p.add_argument("--source-manifest-sha256", required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument(
        "--terminal-audit",
        type=Path,
        help="optional verified full audit; otherwise discover actual engine terminal in a separate full replay pass",
    )
    p.add_argument("--maximum-delay", type=int, default=32)
    p.add_argument("--shard-decisions", type=int, default=1024)
    p.add_argument(
        "--solid-compression",
        action="store_true",
        help="lossless outer compression; identical index and protocol member schema",
    )
    p.add_argument("--solid-codec", choices=("xz", "zstd"), default="xz")
    args = p.parse_args()
    if args.output_dir.exists() or args.output_dir.resolve().is_relative_to(TREE):
        p.error("fresh external output required")
    if args.shard_decisions < 1 or args.maximum_delay < 1:
        p.error("positive shard size and delay required")
    os.environ["SC2PATH"] = str(args.sc2_path)
    import mpyq
    import sc2reader
    from s2protocol import versions

    from poorstar.cli.engine.check_horizon import verified_report
    from poorstar.common.integrity import file_hash, payload_hash
    from poorstar.data.bc.builder import (
        decisions_from_controller,
        discover_replay_terminal,
        partition_native_commands,
    )
    from poorstar.data.bc.records import BCShardWriter, validate_split
    from poorstar.data.corpus.scope import RACE_TRANSLATIONS
    from poorstar.data.trajectory import encode_bytes
    from poorstar.engine.replay_adapter import ReplaySensorAdapter

    content = args.replay.read_bytes()
    replay_sha = hashlib.sha256(content).hexdigest()
    terminal_loop = None
    terminal_digest = None
    if args.terminal_audit:
        terminal_report, terminal_digest = verified_report(args.terminal_audit)
        if (
            terminal_report["replay_sha256"] != replay_sha
            or terminal_report["controlled_player_id"] != args.player_id
            or terminal_report["base_build"] != 55958
            or not terminal_report.get("terminal_observation_retained")
            or terminal_report.get("completion") != "TERMINAL_OBSERVATION_REACHED"
        ):
            raise ValueError("same replay/player complete terminal audit required")
        terminal_loop = int(terminal_report["terminal_loop"])
    meta = sc2reader.load_replay(
        io.BytesIO(content), load_level=4, engine=None, do_tracker_events=False
    )
    if meta.base_build != 55958:
        raise ValueError("unqualified replay build")
    controlled = [x for x in meta.players if x.pid == args.player_id]
    if len(controlled) != 1 or RACE_TRANSLATIONS.get(str(controlled[0].play_race)) != "Zerg":
        raise ValueError("explicit Zerg player identity required")
    original = list(
        versions.build(55958).decode_replay_game_events(
            mpyq.MPQArchive(io.BytesIO(content)).read_file("replay.game.events")
        )
    )
    native = [e for e in original if e.get("_userid", {}).get("m_userId") == controlled[0].uid]
    split = json.loads(args.split_manifest.read_text())
    validate_split(split)
    assignment = split["assignments"][replay_sha]
    names = (
        "poorstar/cli/data/extract.py",
        "poorstar/data/bc/builder.py",
        "poorstar/data/bc/records.py",
        "poorstar/actions/coarse.py",
        "poorstar/actions/frame.py",
        "poorstar/engine/observation.py",
        "poorstar/engine/own_operations.py",
        "poorstar/actions/timing.py",
        "poorstar/engine/replay_adapter.py",
        "poorstar/data/corpus/scope.py",
        "poorstar/data/bc/storage.py",
        "poorstar/engine/replay_reader.py",
    )
    sources = {n: file_hash(TREE / n) for n in names}
    extractor = payload_hash(sources)
    args.output_dir.mkdir(parents=True)
    writer = None
    artifacts = []
    counts = {}
    n = 0
    try:
        with ReplaySensorAdapter(
            sc2_path=args.sc2_path, version="3.16.1", player_id=args.player_id, entity_cap=512
        ) as adapter:
            if adapter.engine_base_build != 55958:
                raise ValueError("engine build mismatch")
            reader = adapter.reader
            info = reader.replay_info(args.replay)
            request = reader._sc_pb.RequestStartReplay(
                replay_data=reader._cfg.replay_data(str(args.replay)),
                options=reader._options,
                observed_player_id=args.player_id,
            )
            local = reader.maps_relative_path(info.map_name)
            if local:
                request.map_data = reader._cfg.map_data(local)
            reader._controller.start_replay(request)
            try:
                if terminal_loop is None:

                    def restart_discovery():
                        reader._leave()
                        reader._controller.start_replay(request)

                    terminal, discovery = discover_replay_terminal(
                        reader._controller,
                        player_id=args.player_id,
                        metadata_duration_loops=int(info.game_duration_loops),
                        step_chunk=2048,
                        restart=restart_discovery,
                        terminal_probe_margin=64,
                    )
                    terminal_loop = discovery["terminal_loop"]
                    (args.output_dir / "discovery_terminal.protocol").write_bytes(
                        terminal.SerializeToString()
                    )
                    discovery.update(replay_sha256=replay_sha, engine_base_build=55958)
                    terminal_digest = payload_hash(discovery)
                    discovery["content_digest"] = terminal_digest
                    (args.output_dir / "terminal_discovery.json").write_text(
                        json.dumps(discovery, indent=2) + "\n"
                    )
                    reader._leave()
                    reader._controller.start_replay(request)
                active_native, terminal_native = partition_native_commands(
                    native, terminal_loop=terminal_loop
                )
                tail_path = args.output_dir / "terminal_native_commands.jsonl.gz"
                with tail_path.open("xb") as tail_file:
                    with gzip.GzipFile(
                        filename="", mode="wb", fileobj=tail_file, mtime=0
                    ) as tail_wire:
                        for event in terminal_native:
                            tail_wire.write(
                                (
                                    json.dumps(event, sort_keys=True, default=encode_bytes) + "\n"
                                ).encode()
                            )
                terminal_boundary = dict(
                    rule="FIRST_ENGINE_RESULT_NONTERMINAL_BODY_INTERVAL",
                    terminal_loop=terminal_loop,
                    original_command_events=len(active_native) + len(terminal_native),
                    playable_command_events=len(active_native),
                    commands_at_terminal=sum(
                        e["_gameloop"] == terminal_loop for e in terminal_native
                    ),
                    commands_after_terminal=sum(
                        e["_gameloop"] > terminal_loop for e in terminal_native
                    ),
                    artifact=tail_path.name,
                    sha256=file_hash(tail_path),
                )
                geometry = reader._controller.game_info()
                catalog = reader._controller.data_raw()
                for row in decisions_from_controller(
                    reader._controller,
                    geometry,
                    catalog,
                    context=adapter.context,
                    episode=replay_sha + ":player" + str(args.player_id),
                    replay_sha256=replay_sha,
                    native_events=active_native,
                    terminal_loop=terminal_loop,
                    maximum_delay=args.maximum_delay,
                ):
                    if writer is None:
                        writer = BCShardWriter(
                            args.output_dir / assignment,
                            split_manifest=split,
                            split=assignment,
                            catalog=catalog,
                            game_info=geometry,
                            source_manifest_sha256=args.source_manifest_sha256,
                            extractor_sha256=extractor,
                            maximum_delay=args.maximum_delay,
                            solid_compression=args.solid_compression,
                            solid_codec=args.solid_codec,
                        )
                    writer.append(row)
                    n += 1
                    for key in (
                        "all_decisions",
                        row.decision,
                        row.body_rejection or "body_not_structurally_rejected",
                    ):
                        counts[key] = counts.get(key, 0) + 1
                    if n % args.shard_decisions == 0:
                        artifacts.append(writer.close())
                        writer = None
                if writer is not None:
                    artifacts.append(writer.close())
                    writer = None
                terminal = reader._controller.observe()
                if (
                    not terminal.player_result
                    or int(terminal.observation.game_loop) != terminal_loop
                ):
                    raise ValueError(
                        "actual extraction terminal does not match independent discovery"
                    )
                (args.output_dir / "extraction_terminal.protocol").write_bytes(
                    terminal.SerializeToString()
                )
            finally:
                reader._leave()
        if any(file_hash(TREE / n) != h for n, h in sources.items()):
            raise ValueError("source drift")
        result = dict(
            schema="poorstar_c3_direct_decision_extraction_v1",
            purpose="ENGINEERING",
            registered_views=1,
            replay_sha256=replay_sha,
            controlled_player_id=args.player_id,
            counts=counts,
            split_sha256=split["sha256"],
            source_manifest_sha256=args.source_manifest_sha256,
            extractor_sha256=extractor,
            engine_base_build=55958,
            terminal_loop=terminal_loop,
            terminal_audit_digest=terminal_digest,
            terminal_source="VERIFIED_PRIOR_AUDIT"
            if args.terminal_audit
            else "INDEPENDENT_FULL_ENGINE_PASS",
            terminal_observation_sha256=file_hash(args.output_dir / "extraction_terminal.protocol"),
            duration_metadata_loops=int(info.game_duration_loops),
            terminal_native_boundary=terminal_boundary,
            shards=artifacts,
            optimizer_steps=0,
            full_game=True,
            intermediate_simulation_skipped=False,
            lossless_solid_compression=args.solid_compression,
            solid_codec=args.solid_codec if args.solid_compression else None,
            observation_rpc_scope="EXACT_SCHEDULER_BODY_AND_ONE_STEP_ECHO",
            rl_ready=False,
        )
        result["content_digest"] = payload_hash(result)
        (args.output_dir / "report.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result), flush=True)
    except BaseException as error:
        if writer is not None:
            writer.abort()
        (args.output_dir / "failure.json").write_text(
            json.dumps(
                dict(error_type=type(error).__name__, error=str(error), completed_shards=artifacts),
                indent=2,
            )
        )
        raise


if __name__ == "__main__":
    main()
