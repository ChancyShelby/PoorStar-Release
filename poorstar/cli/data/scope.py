#!/usr/bin/env python
"""Select replay scope from verified inventory metadata."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from poorstar.common.paths import source_root

TREE = source_root()
from poorstar.common.integrity import file_hash, payload_hash
from poorstar.data.corpus.scope import candidate_scope


def verified(path):
    result = json.loads(path.read_text())
    body = {k: v for k, v in result.items() if k != "content_digest"}
    if payload_hash(body) != result["content_digest"]:
        raise ValueError("inventory content digest mismatch")
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--original-inventory", type=Path, required=True)
    p.add_argument("--recovered-inventory", type=Path, required=True)
    p.add_argument("--engine-build", type=int, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists() or args.output.resolve().is_relative_to(TREE):
        p.error("fresh external output required")
    r = candidate_scope(
        verified(args.original_inventory),
        verified(args.recovered_inventory),
        engine_build=args.engine_build,
    )
    r["input_file_sha256"] = {
        str(path): file_hash(path) for path in (args.original_inventory, args.recovered_inventory)
    }
    r["source_sha256"] = {
        name: file_hash(TREE / name)
        for name in ("poorstar/data/corpus/scope.py", "poorstar/cli/data/scope.py")
    }
    r.pop("content_digest")
    r["content_digest"] = payload_hash(r)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as f:
        json.dump(r, f, indent=2, sort_keys=True)
    print(json.dumps({k: v for k, v in r.items() if k != "entries"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
