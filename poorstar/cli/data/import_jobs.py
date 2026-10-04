#!/usr/bin/env python
"""Cli: data: import jobs."""

import argparse
import json
from pathlib import Path

from poorstar.common.paths import source_root

TREE = source_root()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-dir", type=Path, required=True)
    parser.add_argument("--reuse-data-root", type=Path, required=True, action="append")
    parser.add_argument("--extractor-source-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--maximum-delay", type=int, default=32)
    parser.add_argument("--shard-decisions", type=int, default=1024)
    parser.add_argument("--solid-codec", choices=("xz", "zstd"), default="zstd")
    parser.add_argument("--require-source-complete", action="store_true")
    args = parser.parse_args()
    if args.output.resolve().is_relative_to(TREE):
        parser.error("external import ledger required")
    from poorstar.cli.data.build import freeze_jobs
    from poorstar.common.integrity import file_hash
    from poorstar.data.corpus.subset import import_completed_jobs

    plan, digest, jobs = freeze_jobs(args.plan_dir)
    split = json.loads((args.plan_dir / "split.json").read_text())
    identity = dict(
        schema="poorstar_c3_full_corpus_coordinator_v1",
        plan_digest=digest,
        source_manifest_sha256=file_hash(args.extractor_source_manifest),
        split_sha256=plan["split_sha256"],
        maximum_delay=args.maximum_delay,
        shard_decisions=args.shard_decisions,
        solid_compression=True,
        solid_codec=args.solid_codec,
    )
    _, ledger = import_completed_jobs(
        plan,
        digest,
        jobs,
        split,
        identity,
        args.reuse_data_root,
        args.output,
        require_source_complete=args.require_source_complete,
    )
    print(
        json.dumps(
            dict(
                imported_views=ledger["imported_views"],
                imported_rows=ledger["imported_rows"],
                incomplete=len(ledger["incomplete"]),
                formal_bc_ready=False,
            )
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
