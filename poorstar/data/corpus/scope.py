"""Data: corpus: scope."""

from __future__ import annotations

from collections import Counter
from typing import Mapping

from poorstar.common.integrity import payload_hash

RACE_TRANSLATIONS = {
    "Zerg": "Zerg",
    "Protoss": "Protoss",
    "Terran": "Terran",
    "Зерги": "Zerg",
    "Протоссы": "Protoss",
    "Терраны": "Terran",
}


def candidate_scope(original: Mapping, recovered: Mapping, *, engine_build: int) -> dict:
    if len(original["entries"]) != original["records"]:
        raise ValueError("original inventory count mismatch")
    recovered_hashes = {
        r["metadata"]["file_sha256"]
        for r in recovered["entries"]
        if not r.get("problems") and "metadata" in r
    }
    counts, matchups, versions = Counter(), Counter(), Counter()
    entries = []
    for row in original["entries"]:
        counts["all_records"] += 1
        entry = {
            "alias": row["alias"],
            "path": row["path"],
            "problems": [],
            "history": "UNKNOWN_NOT_CONFIRMATORY",
            "training_ready": False,
        }
        if "metadata" not in row:
            entry["problems"].append("METADATA_PARSE_FAILURE")
            entry["error"] = row.get("error", "unknown metadata failure")
            counts["metadata_parse_failure"] += 1
            entries.append(entry)
            continue
        meta = row["metadata"]
        entry.update(
            file_sha256=meta["file_sha256"],
            bytes=meta["bytes"],
            provisional_match_key=meta["semantic_sha256"],
            base_build=meta["base_build"],
            map_hash=meta["map_hash"],
            header_loops=meta["frames"],
        )
        players = [
            {
                "player_id": p["player_id"],
                "original_race": p["race"],
                "canonical_race": RACE_TRANSLATIONS.get(p["race"]),
                "account_digest": p["account_digest"],
            }
            for p in meta["players"]
        ]
        entry["players"] = players
        if len(players) != 2 or len({p["player_id"] for p in players}) != 2:
            entry["problems"].append("INVALID_ONE_V_ONE_PARTICIPANTS")
            counts["invalid_participants"] += 1
        versions[str(meta["base_build"])] += 1
        if any(p["canonical_race"] is None for p in players):
            entry["problems"].append("UNKNOWN_RACE_LABEL")
            counts["unknown_race_label"] += 1
        races = sorted(p["canonical_race"] for p in players if p["canonical_race"] is not None)
        matchup = {
            ("Zerg", "Zerg"): "ZvZ",
            ("Protoss", "Zerg"): "ZvP",
            ("Terran", "Zerg"): "ZvT",
            ("Protoss", "Protoss"): "PvP",
            ("Protoss", "Terran"): "PvT",
            ("Terran", "Terran"): "TvT",
        }.get(tuple(races))
        entry["matchup"] = matchup
        matchups[matchup or "UNKNOWN"] += 1
        if meta["file_sha256"] in recovered_hashes:
            entry["history"] = "KNOWN_HISTORICAL_RECOVERY_AT_LEAST"
            counts["known_historical_recovery"] += 1
        perspectives = [p["player_id"] for p in players if p["canonical_race"] == "Zerg"]
        entry["controlled_zerg_player_ids"] = perspectives
        if meta["base_build"] != engine_build:
            entry["problems"].append("ENGINE_BASE_BUILD_UNAVAILABLE")
            counts["engine_base_build_unavailable"] += 1
        if perspectives:
            counts["zerg_matches_all_versions"] += 1
            counts["zerg_player_perspectives_all_versions"] += len(perspectives)
            if not entry["problems"]:
                counts["compatible_zerg_candidate_matches"] += 1
                counts["compatible_zerg_candidate_perspectives"] += len(perspectives)
        else:
            counts["non_zerg_matches"] += 1
        entry["scope_candidate"] = bool(perspectives) and not entry["problems"]
        entries.append(entry)
    report = dict(
        schema="poorstar_v3_c3_original_pack_candidate_scope_v1",
        purpose="FULL_INVENTORY_SCOPE_AND_EXPOSURE_AUDIT_ONLY",
        source_inventory_digest=original["content_digest"],
        recovered_inventory_digest=recovered["content_digest"],
        engine_base_build=engine_build,
        race_translations=RACE_TRANSLATIONS,
        counts=dict(counts),
        matchups=dict(matchups),
        base_builds=dict(versions),
        entries=entries,
        zvz_split_policy="BOTH_PERSPECTIVES_SHARE_ONE_PROVISIONAL_MATCH_KEY",
        history_policy="RECOVERY_IS_MINIMUM_KNOWN_EXPOSURE_REMAINDER_IS_UNKNOWN",
        match_dedup="PROVISIONAL_REQUIRES_FURTHER_AUDIT",
        split_frozen=False,
        training_ready=False,
        rl_ready=False,
    )
    report["content_digest"] = payload_hash(report)
    return report
