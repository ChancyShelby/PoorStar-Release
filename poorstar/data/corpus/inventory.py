"""Data: corpus: inventory."""

from __future__ import annotations

import hashlib
import io
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

from poorstar.common.integrity import file_hash, payload_hash


def replay_metadata(path: Path) -> dict[str, Any]:
    import mpyq
    import sc2reader

    content = path.read_bytes()
    replay = sc2reader.load_replay(io.BytesIO(content), load_level=1, engine=None)
    # Basic details alone do not construct player objects; no message/game events are needed.
    replay.load_players()
    players = []
    for player in replay.players:
        account = (int(player.region_id), int(player.subregion), int(player.toon_id))
        identity = payload_hash(account) if account[-1] > 0 else None
        players.append(
            {
                "player_id": int(player.pid),
                "race": str(player.play_race),
                "account_digest": identity,
            }
        )
    archive = mpyq.MPQArchive(io.BytesIO(content))
    members = {}
    details_name = "replay.details"
    details = archive.read_file(details_name)
    if details is None:
        details_name = "replay.details.backup"
        details = archive.read_file(details_name)
    if details is None:
        raise ValueError("neither primary nor backup replay details are available")
    members["replay.details"] = hashlib.sha256(details).hexdigest()
    for name in ("replay.game.events", "replay.tracker.events"):
        value = archive.read_file(name)
        if value is None and name == "replay.game.events":
            raise ValueError("game event stream is absent")
        # Absence stays in the key; it is not treated as an empty or observed tracker stream.
        members[name] = hashlib.sha256(value).hexdigest() if value is not None else None
    return {
        "file_sha256": hashlib.sha256(content).hexdigest(),
        "bytes": len(content),
        "version": replay.release_string,
        "base_build": int(replay.base_build),
        "frames": int(replay.frames),
        "map_name": str(replay.map_name),
        "map_hash": str(replay.map_hash),
        "players": players,
        "details_member": details_name,
        "tracker_available": members["replay.tracker.events"] is not None,
        "semantic_key_schema": "DETAILS_GAME_EVENTS_AND_EXPLICIT_TRACKER_AVAILABILITY_V1",
        "semantic_sha256": payload_hash(members),
    }


def inventory(
    manifests: list[Path],
    *,
    allowed_replay_root: Path,
    inspect: Callable[[Path], dict[str, Any]] = replay_metadata,
    progress: Callable[[int], None] | None = None,
) -> dict[str, Any]:
    if not manifests:
        raise ValueError("no recovery manifests supplied")
    allowed = allowed_replay_root.resolve()
    entries = []
    source = {}
    bytes_groups: dict[str, list[int]] = defaultdict(list)
    semantic_groups: dict[str, list[int]] = defaultdict(list)
    account_replays: dict[str, set[str]] = defaultdict(set)
    failure_counts: Counter = Counter()
    versions: Counter = Counter()
    matchups: Counter = Counter()
    maps: Counter = Counter()
    aliases: Counter = Counter()
    for manifest in sorted(manifests):
        source[str(manifest)] = file_hash(manifest)
        document = json.loads(manifest.read_text(encoding="utf-8"))
        for item in document["records"]:
            path = Path(item["extracted_path"])
            entry: dict[str, Any] = {
                "source_alias": item["source_replay_id"],
                "path": str(path),
                "manifest": str(manifest),
                "declared_matchup": item["matchup"],
                "target_pids": item["target_pids"],
                "problems": [],
            }
            aliases[entry["source_alias"]] += 1
            if not path.resolve().is_relative_to(allowed):
                entry["problems"].append("REPLAY_OUTSIDE_DECLARED_ROOT")
            elif not path.is_file():
                entry["problems"].append("MISSING_PHYSICAL_REPLAY")
            else:
                try:
                    measured = inspect(path)
                    entry["metadata"] = measured
                    if (
                        measured["file_sha256"] != item["sha256"]
                        or measured["bytes"] != item["file_size"]
                    ):
                        entry["problems"].append("RECOVERY_MANIFEST_BYTE_MISMATCH")
                    pids = item["target_pids"]
                    player_table = {p["player_id"]: p for p in measured["players"]}
                    if not pids or any(
                        type(pid) is not int or pid not in player_table for pid in pids
                    ):
                        entry["problems"].append("CONTROLLED_PLAYER_UNRECOVERABLE")
                    elif any(player_table[pid]["race"] != "Zerg" for pid in pids):
                        entry["problems"].append("CONTROLLED_PLAYER_NOT_ZERG")
                    races = sorted(p["race"] for p in measured["players"])
                    expected_races = {
                        "ZvP": ["Protoss", "Zerg"],
                        "ZvT": ["Terran", "Zerg"],
                        "ZvZ": ["Zerg", "Zerg"],
                    }.get(item["matchup"])
                    if races != expected_races:
                        entry["problems"].append("DECLARED_MATCHUP_MISMATCH")
                    index = len(entries)
                    bytes_groups[measured["file_sha256"]].append(index)
                    semantic_groups[measured["semantic_sha256"]].append(index)
                    versions[measured["version"]] += 1
                    maps[measured["map_hash"]] += 1
                    matchups[item["matchup"]] += 1
                    for player in measured["players"]:
                        if player["account_digest"]:
                            account_replays[player["account_digest"]].add(
                                measured["semantic_sha256"]
                            )
                except Exception as error:
                    entry["problems"].append("REPLAY_METADATA_PARSE_ERROR")
                    entry["error"] = f"{type(error).__name__}: {error}"
            failure_counts.update(entry["problems"])
            entries.append(entry)
            if progress is not None and len(entries) % 500 == 0:
                progress(len(entries))
    report = {
        "schema": "poorstar_v3_c3_recovered_replay_inventory_v1",
        "purpose": "METADATA_AUDIT_ONLY",
        "training_ready": False,
        "split_frozen": False,
        "historical_exposure": "PREVIOUS_V2_RECOVERY_PIPELINE_NO_UNSEEN_DATA_CLAIM",
        "source_manifest_sha256": payload_hash(source),
        "source_manifests": source,
        "records": len(entries),
        "entries": entries,
        "failure_counts": dict(sorted(failure_counts.items())),
        "unique_file_contents": len(bytes_groups),
        "unique_semantic_matches": len(semantic_groups),
        "duplicate_byte_groups": [v for v in bytes_groups.values() if len(v) > 1],
        "duplicate_semantic_groups": [v for v in semantic_groups.values() if len(v) > 1],
        "duplicate_source_aliases": {k: n for k, n in aliases.items() if n > 1},
        "versions": dict(sorted(versions.items())),
        "matchups": dict(sorted(matchups.items())),
        "map_counts": dict(sorted(maps.items())),
        "account_identities": len(account_replays),
        "accounts_in_multiple_matches": sum(len(v) > 1 for v in account_replays.values()),
        "account_repeat_max_matches": max(map(len, account_replays.values()), default=0),
        "tracker_unavailable_records": sum(
            e.get("metadata", {}).get("tracker_available") is False for e in entries
        ),
        "backup_details_records": sum(
            e.get("metadata", {}).get("details_member") == "replay.details.backup" for e in entries
        ),
        "semantic_dedup_verdict": "PROVISIONAL_REQUIRES_MATCH_ID_AND_EXPOSURE_AUDIT",
        "engine_decodability": "NOT_VERIFIED",
        "rl_ready": False,
    }
    report["content_digest"] = payload_hash(report)
    return report
