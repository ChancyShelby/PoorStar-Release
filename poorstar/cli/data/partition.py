#!/usr/bin/env python
"""Assign verified replay views to disjoint extraction owners."""

import argparse
from pathlib import Path

from poorstar.common.paths import source_root

TREE = source_root()


def main():
    from poorstar.cli.data.build import freeze_jobs, write_report
    from poorstar.common.integrity import file_hash
    from poorstar.data.corpus.partition import partition_jobs

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan-dir", type=Path, required=True)
    p.add_argument("--extractor-manifest", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--cpu-weight", type=int, default=2)
    p.add_argument("--gpu-weight", type=int, default=1)
    p.add_argument("--maximum-delay", type=int, default=32)
    p.add_argument("--shard-decisions", type=int, default=1024)
    p.add_argument("--solid-codec", choices=("xz", "zstd"), default="zstd")
    a = p.parse_args()
    if a.output_dir.exists() or a.output_dir.resolve().is_relative_to(TREE):
        p.error("fresh external partition output required")
    if min(a.maximum_delay, a.shard_decisions) < 1:
        p.error("positive data parameters required")
    plan, digest, jobs = freeze_jobs(a.plan_dir)
    identity = dict(
        schema="poorstar_c3_full_corpus_coordinator_v1",
        plan_digest=digest,
        source_manifest_sha256=file_hash(a.extractor_manifest),
        split_sha256=plan["split_sha256"],
        maximum_delay=a.maximum_delay,
        shard_decisions=a.shard_decisions,
        solid_compression=True,
        solid_codec=a.solid_codec,
    )
    lanes, report = partition_jobs(
        identity, jobs, {"cpu": a.cpu_weight, "gpu": a.gpu_weight}, seed=a.seed
    )
    a.output_dir.mkdir(parents=True)
    for lane, assignment in lanes.items():
        write_report(a.output_dir / (lane + ".json"), assignment)
    report["assignment_files"] = {
        lane: dict(path=lane + ".json", sha256=file_hash(a.output_dir / (lane + ".json")))
        for lane in lanes
    }
    write_report(a.output_dir / "report.json", report)
    print("PARTITION_READY", report["lane_view_counts"])


if __name__ == "__main__":
    main()
