#!/usr/bin/env python
"""Build a verified replay extraction plan."""

import argparse
import concurrent.futures
import gzip
import hashlib
import io
import json
import multiprocessing
import time
from collections import Counter, defaultdict
from pathlib import Path

from poorstar.common.paths import source_root

TREE = source_root()


def inspect(entry):
    import mpyq
    import sc2reader
    from s2protocol import versions

    from poorstar.common.integrity import payload_hash
    from poorstar.engine.native_events import census_native_events

    path = Path(entry["path"])
    result = {
        k: entry[k]
        for k in (
            "alias",
            "file_sha256",
            "map_hash",
            "matchup",
            "controlled_zerg_player_ids",
            "history",
        )
    }
    try:
        content = path.read_bytes()
        if (
            len(content) != entry["bytes"]
            or hashlib.sha256(content).hexdigest() != entry["file_sha256"]
        ):
            raise ValueError("native census input replay bytes drift")
        replay = sc2reader.load_replay(
            io.BytesIO(content), load_level=1, engine=None, do_tracker_events=False
        )
        replay.load_players()
        if int(replay.base_build) != 55958 or str(replay.map_hash) != entry["map_hash"]:
            raise ValueError("native census source build/map mismatch")
        players = {int(x.pid): x for x in replay.players}
        ids = entry["controlled_zerg_player_ids"]
        users = [int(players[pid].uid) for pid in ids]
        translations = {"Zerg": "Zerg", "Зерги": "Zerg"}
        if len(set(users)) != len(users) or any(
            translations.get(str(players[pid].play_race)) != "Zerg" for pid in ids
        ):
            raise ValueError("unique controlled player/race/user mapping unavailable")
        mpq = mpyq.MPQArchive(io.BytesIO(content))
        native = mpq.read_file("replay.game.events")
        if native is None:
            raise ValueError("complete native game events absent")
        native_sha = hashlib.sha256(native).hexdigest()
        census = census_native_events(
            versions.build(55958).decode_replay_game_events(native), controlled_user_ids=users
        )
        if hashlib.sha256(path.read_bytes()).hexdigest() != entry["file_sha256"]:
            raise ValueError("source replay changed during native census")
        result.update(
            completion="COMPLETE_NATIVE_EVENT_CENSUS",
            census=census,
            player_user_mapping={str(pid): int(players[pid].uid) for pid in ids},
            native_game_events_sha256=native_sha,
            native_content_match_key=payload_hash([55958, entry["map_hash"], native_sha]),
            metadata_provisional_match_key=entry["provisional_match_key"],
        )
    except Exception as error:
        result.update(
            completion="FAILED_NATIVE_EVENT_CENSUS", error=f"{type(error).__name__}: {error}"
        )
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scope", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()
    if (
        args.output_dir.exists()
        or args.output_dir.resolve().is_relative_to(TREE)
        or not 1 <= args.workers <= 8
    ):
        p.error("fresh external output and 1..8 parser workers required")
    from poorstar.common.integrity import file_hash, payload_hash
    from poorstar.engine.native_events import complete_candidate_entries

    names = (
        "poorstar/cli/data/plan.py",
        "poorstar/engine/native_events.py",
        "poorstar/common/integrity.py",
    )
    sources = {n: file_hash(TREE / n) for n in names}
    scope_sha = file_hash(args.scope)
    scope = json.loads(args.scope.read_text())
    sd = scope.pop("content_digest")
    if payload_hash(scope) != sd or scope["engine_base_build"] != 55958:
        raise ValueError("verified candidate scope required")
    entries = complete_candidate_entries(scope)
    args.output_dir.mkdir(parents=True)
    start = time.monotonic()
    counts, view_totals = Counter(), Counter()
    strata, abilities = defaultdict(Counter), Counter()
    groups = defaultdict(list)
    chunks, pending = [], []
    failures = []

    def save_chunk():
        if not pending:
            return
        path = args.output_dir / ("native_census_" + str(len(chunks)).zfill(4) + ".jsonl.gz")
        with gzip.open(path, "xt") as f:
            for item in pending:
                f.write(json.dumps(item, sort_keys=True) + "\n")
        chunks.append(
            dict(
                path=path.name,
                records=len(pending),
                bytes=path.stat().st_size,
                sha256=file_hash(path),
            )
        )
        pending.clear()

    with concurrent.futures.ProcessPoolExecutor(
        max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        for result in pool.map(inspect, entries, chunksize=8):
            pending.append(result)
            counts["records"] += 1
            counts["declared_perspectives"] += len(result["controlled_zerg_player_ids"])
            counts[result["completion"]] += 1
            key = result["matchup"] + "/" + result["map_hash"]
            strata[key]["records"] += 1
            strata[key][result["completion"]] += 1
            if result["completion"] == "FAILED_NATIVE_EVENT_CENSUS":
                failures.append(dict(alias=result["alias"], error=result["error"]))
            else:
                census = result["census"]
                groups[result["native_content_match_key"]].append(result["alias"])
                counts["native_events_all_players"] += census["counts"]["events"]
                counts["complete_perspectives"] += len(census["views"])
                for uid, view in census["views"].items():
                    view_totals.update(view)
                    abilities.update(census["native_ability_link_index_counts"][uid])
                    strata[key]["direct_commands"] += view.get("direct_commands", 0)
                    strata[key]["manager_state_1_commands"] += view.get(
                        "manager_state_1_commands", 0
                    )
            if len(pending) == 1000:
                save_chunk()
                print(
                    json.dumps(dict(counts=dict(counts), elapsed_seconds=time.monotonic() - start)),
                    flush=True,
                )
    save_chunk()
    readback = Counter()
    for chunk in chunks:
        path = args.output_dir / chunk["path"]
        if file_hash(path) != chunk["sha256"]:
            raise ValueError("native chunk drift")
        local = 0
        with gzip.open(path, "rt") as f:
            for line in f:
                item = json.loads(line)
                local += 1
                readback["records"] += 1
                readback["declared_perspectives"] += len(item["controlled_zerg_player_ids"])
                readback[item["completion"]] += 1
                if item["completion"] == "COMPLETE_NATIVE_EVENT_CENSUS":
                    readback["native_events_all_players"] += item["census"]["counts"]["events"]
                    readback["complete_perspectives"] += len(item["census"]["views"])
        if local != chunk["records"]:
            raise ValueError("native chunk record count mismatch")
    if readback != counts or counts["records"] != len(entries):
        raise ValueError("complete native chunk readback denominator mismatch")
    if file_hash(args.scope) != scope_sha or any(
        file_hash(TREE / n) != sha for n, sha in sources.items()
    ):
        raise ValueError("source/scope drift during complete census")
    report = dict(
        schema="poorstar_c3_full_native_corpus_census_v1",
        source_scope_digest=sd,
        source_scope_counts=scope["counts"],
        source_scope_sha256=scope_sha,
        source_manifest=sources,
        workers=args.workers,
        counts=dict(counts),
        controlled_view_totals=dict(view_totals),
        native_ability_link_index_counts=dict(abilities),
        strata={k: dict(v) for k, v in strata.items()},
        native_content_duplicate_groups=[v for v in groups.values() if len(v) > 1],
        unique_native_content_match_groups=len(groups),
        failures=failures,
        chunks=chunks,
        full_chunk_readback_verified=True,
        engine_decodability_verified=False,
        native_to_engine_ability_mapping_verified=False,
        historical_exposure="MINIMUM_KNOWN_REMAINDER_UNVERIFIED",
        opaque_data_target_semantics_verified=False,
        player_disjoint_verified=False,
        split_frozen=False,
        training_ready=False,
        rl_ready=False,
        elapsed_seconds=time.monotonic() - start,
    )
    report["content_digest"] = payload_hash(report)
    with (args.output_dir / "report.json").open("x") as f:
        json.dump(report, f, indent=2, sort_keys=True)
    print(
        json.dumps(
            {
                k: v
                for k, v in report.items()
                if k
                not in ("strata", "native_ability_link_index_counts", "source_manifest", "chunks")
            },
            indent=2,
        )
    )
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
