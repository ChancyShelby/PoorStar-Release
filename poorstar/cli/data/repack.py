#!/usr/bin/env python
"""Cli: data: repack."""

import argparse
import json
import time
from pathlib import Path

from poorstar.common.paths import source_root

TREE = source_root()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--build-report", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    a = p.parse_args()
    if a.output_dir.exists() or a.output_dir.resolve().is_relative_to(TREE):
        p.error("fresh external output required")
    from poorstar.common.integrity import file_hash, payload_hash
    from poorstar.data.bc.storage import solid_repack

    report = json.loads(a.build_report.read_text())
    digest = report.pop("content_digest")
    if payload_hash(report) != digest:
        raise ValueError("build report drift")
    a.output_dir.mkdir(parents=True)
    shards = []
    start = time.monotonic()
    try:
        for entry in report["shards"]:
            path = Path(entry["path"])
            result = solid_repack(
                path, sha256=entry["sha256"], output_dir=a.output_dir / path.parent.name
            )
            shards.append(result)
            print(
                json.dumps(
                    dict(
                        completed=len(shards),
                        original_bytes=entry["bytes"],
                        packed_bytes=result["bytes"],
                    )
                ),
                flush=True,
            )
        result = dict(
            report,
            shards=shards,
            storage_migration_schema="LOSSLESS_SOLID_ZIP_XZ_V1",
            input_build_digest=digest,
            input_build_report_sha256=file_hash(a.build_report),
            original_shard_bytes=sum(e["bytes"] for e in report["shards"]),
            solid_shard_bytes=sum(e["bytes"] for e in shards),
            all_member_wires_and_indices_byte_identical=True,
            storage_elapsed_seconds=time.monotonic() - start,
        )
        result["content_digest"] = payload_hash(result)
        (a.output_dir / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    except BaseException as error:
        (a.output_dir / "failure.json").write_text(
            json.dumps(
                dict(error_type=type(error).__name__, error=str(error), completed_shards=shards),
                indent=2,
            )
        )
        raise


if __name__ == "__main__":
    main()
