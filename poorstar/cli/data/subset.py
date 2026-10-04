#!/usr/bin/env python
"""Cli: data: subset."""

import argparse
import json
from pathlib import Path

from poorstar.common.paths import source_root

TREE = source_root()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-plan-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.resolve().is_relative_to(TREE):
        parser.error("fresh external subset plan required")
    from poorstar.data.corpus.subset import write_zvz_plan

    plan, digest, jobs = write_zvz_plan(args.parent_plan_dir, args.output_dir)
    print(
        json.dumps(
            dict(
                plan_digest=digest,
                counts=plan["counts"],
                split_view_counts=plan["split_view_counts"],
                parent=plan["scope"]["parent_plan_digest"],
            )
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
