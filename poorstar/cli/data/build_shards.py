#!/usr/bin/env python
"""Cli: data: build shards."""

import argparse
import json
from pathlib import Path

from poorstar.common.paths import source_root

TREE = source_root()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--registry", type=Path, required=True)
    p.add_argument("--split-manifest", type=Path, required=True)
    p.add_argument("--unit-catalog", type=Path, required=True)
    p.add_argument("--unit-catalog-sha256", required=True)
    p.add_argument("--source-manifest-sha256", required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--maximum-delay", type=int, default=32)
    p.add_argument("--shard-decisions", type=int, default=1024)
    args = p.parse_args()
    if args.output_dir.exists() or args.output_dir.resolve().is_relative_to(TREE):
        p.error("fresh external output required")
    if args.shard_decisions < 1:
        p.error("positive shard size required")
    from s2clientprotocol import sc2api_pb2 as sc

    from poorstar.cli.engine.check_horizon import verified_report
    from poorstar.common.integrity import file_hash, payload_hash
    from poorstar.data.bc.builder import decisions_from_audit
    from poorstar.data.bc.records import BCShardWriter, validate_split

    if file_hash(args.unit_catalog) != args.unit_catalog_sha256:
        raise ValueError("unit catalog hash mismatch")
    catalog = sc.ResponseData()
    catalog.ParseFromString(args.unit_catalog.read_bytes())
    split = json.loads(args.split_manifest.read_text())
    validate_split(split)
    registry = json.loads(args.registry.read_text())
    if not registry["views"] or registry.get("purpose") not in ("ENGINEERING", "FORMAL_CORPUS"):
        raise ValueError("nonempty explicitly scoped registry required")
    if registry.get("maximum_delay", args.maximum_delay) != args.maximum_delay:
        raise ValueError("registry scheduler horizon mismatch")
    source_names = (
        "poorstar/cli/data/build_shards.py",
        "poorstar/data/bc/builder.py",
        "poorstar/data/bc/records.py",
        "poorstar/actions/coarse.py",
        "poorstar/actions/frame.py",
        "poorstar/actions/contract.py",
        "poorstar/engine/observation.py",
        "poorstar/engine/own_operations.py",
        "poorstar/actions/timing.py",
    )
    sources = {n: file_hash(TREE / n) for n in source_names}
    extractor = payload_hash(sources)
    args.output_dir.mkdir(parents=True)
    artifacts = []
    counts = {}
    writer = None
    try:
        for view in registry["views"]:
            trajectory, query, geometry = (
                Path(view[k]) for k in ("trajectory", "query", "geometry")
            )
            report, _ = verified_report(trajectory)
            digest = report["replay_sha256"]
            if view["replay_sha256"] != digest:
                raise ValueError("registry replay identity drift")
            assignment = split["assignments"][digest]
            game = sc.ResponseGameInfo()
            game.ParseFromString((geometry / "game_info.protocol").read_bytes())
            n = 0
            chunk = 0
            for decision in decisions_from_audit(
                trajectory, query, geometry, catalog, maximum_delay=args.maximum_delay
            ):
                if writer is None:
                    writer = BCShardWriter(
                        args.output_dir / assignment,
                        split_manifest=split,
                        split=assignment,
                        catalog=catalog,
                        game_info=game,
                        source_manifest_sha256=args.source_manifest_sha256,
                        extractor_sha256=extractor,
                        maximum_delay=args.maximum_delay,
                    )
                writer.append(decision)
                n += 1
                chunk += 1
                for key in (
                    "all_decisions",
                    decision.decision,
                    decision.body_rejection or "body_not_structurally_rejected",
                ):
                    counts[key] = counts.get(key, 0) + 1
                if chunk == args.shard_decisions:
                    artifacts.append(writer.close())
                    writer = None
                    chunk = 0
            if writer is not None:
                artifacts.append(writer.close())
                writer = None
            if n == 0:
                raise ValueError("registered view yielded no complete decisions")
            print(json.dumps(dict(view=view["name"], decisions=n, split=assignment)), flush=True)
        if any(file_hash(TREE / n) != sha for n, sha in sources.items()):
            raise ValueError("extractor source drift")
        result = dict(
            schema="poorstar_c3_bc_build_v1",
            purpose=registry["purpose"],
            registered_views=len(registry["views"]),
            registry_sha256=file_hash(args.registry),
            split_sha256=split["sha256"],
            source_manifest_sha256=args.source_manifest_sha256,
            extractor_sha256=extractor,
            unit_catalog_sha256=args.unit_catalog_sha256,
            counts=counts,
            shards=artifacts,
            registered_scope_complete=True,
            paper_corpus_gate_passed=False,
            optimizer_steps=0,
            bc_capability_evaluated=False,
            rl_ready=False,
        )
        result["content_digest"] = payload_hash(result)
        (args.output_dir / "report.json").write_text(json.dumps(result, indent=2) + "\n")
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
