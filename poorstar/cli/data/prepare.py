#!/usr/bin/env python
"""Aggregate verified BC shards into a complete indexed corpus."""

import argparse
import gzip
import json
import shutil
from collections import Counter
from pathlib import Path

from poorstar.common.paths import source_root

TREE = source_root()


def verified(path):
    from poorstar.common.integrity import payload_hash

    x = json.loads(path.read_text())
    digest = x.pop("content_digest")
    if payload_hash(x) != digest:
        raise ValueError("input report drift: " + str(path))
    return x, digest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scope", type=Path, required=True)
    p.add_argument("--native-census", type=Path, required=True)
    p.add_argument("--pilot-build-report", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--seed", type=int, default=20261003)
    a = p.parse_args()
    if a.output_dir.exists() or a.output_dir.resolve().is_relative_to(TREE):
        p.error("fresh external corpus plan required")
    from poorstar.common.integrity import file_hash, payload_hash
    from poorstar.data.bc.records import freeze_split

    scope, sd = verified(a.scope)
    census, cd = verified(a.native_census)
    pilot, pd = verified(a.pilot_build_report)
    if (
        scope["engine_base_build"] != 55958
        or census["source_scope_digest"] != sd
        or census.get("failures")
    ):
        raise ValueError("complete compatible scope/native census required")
    if not census.get("full_chunk_readback_verified"):
        raise ValueError("native census readback missing")
    entries = {e["alias"]: e for e in scope["entries"] if e.get("scope_candidate", False)}
    seen = set()
    groups = []
    jobs = {}
    counts = Counter()
    strata = Counter()
    for chunk in census["chunks"]:
        path = a.native_census.parent / chunk["path"]
        if file_hash(path) != chunk["sha256"] or path.stat().st_size != chunk["bytes"]:
            raise ValueError("native chunk drift")
        local = 0
        with gzip.open(path, "rt") as f:
            for line in f:
                x = json.loads(line)
                local += 1
                alias = x["alias"]
                if alias not in entries or alias in seen:
                    raise ValueError("native scope duplicate/unexpected alias")
                seen.add(alias)
                e = entries[alias]
                if (
                    x["completion"] != "COMPLETE_NATIVE_EVENT_CENSUS"
                    or x["file_sha256"] != e["file_sha256"]
                ):
                    raise ValueError("incomplete native/source identity")
                if x["controlled_zerg_player_ids"] != e["controlled_zerg_player_ids"]:
                    raise ValueError("player identity drift")
                groups.append(
                    dict(replay_sha256=e["file_sha256"], match_group=x["native_content_match_key"])
                )
                for player in e["controlled_zerg_player_ids"]:
                    counts["candidate_alias_views"] += 1
                    key = (e["file_sha256"], player)
                    if key in jobs:
                        jobs[key]["aliases"].append(alias)
                        continue
                    jobs[key] = dict(
                        job_id=payload_hash(list(key)),
                        replay_sha256=e["file_sha256"],
                        player_id=player,
                        replay=e["path"],
                        replay_bytes=e["bytes"],
                        aliases=[alias],
                        map_hash=e["map_hash"],
                        matchup=e["matchup"],
                        header_loops=e["header_loops"],
                        history=e["history"],
                        native_content_group=x["native_content_match_key"],
                    )
                    strata[e["matchup"]] += 1
        if local != chunk["records"]:
            raise ValueError("native chunk denominator drift")
    if (
        seen != set(entries)
        or counts["candidate_alias_views"]
        != scope["counts"]["compatible_zerg_candidate_perspectives"]
    ):
        raise ValueError("full candidate scope does not close")
    split = freeze_split(groups, seed=a.seed)
    a.output_dir.mkdir(parents=True)
    (a.output_dir / "split.json").write_text(json.dumps(split, indent=2) + "\n")
    split_counts = Counter()
    with gzip.open(a.output_dir / "jobs.jsonl.gz", "xt") as f:
        for key, x in sorted(jobs.items()):
            x["split"] = split["assignments"][x["replay_sha256"]]
            split_counts[x["split"]] += 1
            f.write(json.dumps(x, sort_keys=True) + "\n")
    total_pilot_bytes = sum(e["bytes"] for e in pilot["shards"])
    estimated_bytes_per_view = total_pilot_bytes / pilot["registered_views"]
    projected = int(estimated_bytes_per_view * len(jobs))
    free = shutil.disk_usage(a.output_dir).free
    result = dict(
        schema="poorstar_c3_full_corpus_extraction_plan_v1",
        purpose="FULL_COMPATIBLE_SCOPE_NO_REDUCTION",
        scope_digest=sd,
        native_census_digest=cd,
        split_sha256=split["sha256"],
        seed=a.seed,
        counts=dict(
            candidate_matches=len(entries),
            candidate_alias_views=counts["candidate_alias_views"],
            unique_byte_identical_views=len(jobs),
            byte_identical_alias_duplicates=counts["candidate_alias_views"] - len(jobs),
        ),
        split_view_counts=dict(split_counts),
        matchup_view_counts=dict(strata),
        jobs_sha256=file_hash(a.output_dir / "jobs.jsonl.gz"),
        jobs_bytes=(a.output_dir / "jobs.jsonl.gz").stat().st_size,
        pilot_build_digest=pd,
        pilot_bytes_per_view=estimated_bytes_per_view,
        projected_materialized_shard_bytes=projected,
        free_disk_bytes=free,
        projection_scope="LINEAR_PILOT_ESTIMATE_NOT_A_MEASURED_FULL_CORPUS_SIZE",
        storage_gate="PROJECTION_FITS_WITH_20_PERCENT_FREE_RESERVE"
        if projected <= 0.8 * free
        else "COMPRESSION_OR_STORAGE_REQUIRED",
        SC2_workers_maximum=2,
        learner_GPU_count=1,
        history_quarantine=False,
        native_content_groups_used_for_split_not_view_deletion=True,
        complete_engine_extraction=False,
        completed_views=0,
        optimizer_steps=0,
        formal_bc_ready=False,
        rl_ready=False,
    )
    result["content_digest"] = payload_hash(result)
    (a.output_dir / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
