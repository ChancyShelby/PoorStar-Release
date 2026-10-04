#!/usr/bin/env python
"""Collect a fixed-stage actor rollout with verified input and output identities."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from poorstar.common.paths import source_root

TREE = source_root()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--preflight", type=Path)
    parser.add_argument("--dispatch", action="store_true")
    args = parser.parse_args()
    from poorstar.league import SnapshotRegistry
    from poorstar.league.ownership import write_document
    from poorstar.training.contracts import digest
    from poorstar.training.stage_actor import StageActorPlan, execute_stage_actor

    value = json.loads(args.plan.read_text())
    content = value.pop("content_digest", None)
    if content is not None and content != digest(value):
        raise ValueError("stage plan content drift")
    plan = StageActorPlan(**value)
    registry = SnapshotRegistry(args.registry)
    if args.output_dir.exists() or args.output_dir.resolve().is_relative_to(TREE):
        parser.error("fresh external output required")
    if not args.dispatch:
        for identifier in (plan.learner_snapshot_id, plan.opponent_snapshot_id):
            registry.read(identifier)
        args.output_dir.mkdir(parents=True)
        write_document(args.output_dir / "stage_plan.json", asdict(plan), immutable=True)
        report = dict(
            status="PLAN_ONLY",
            stage=plan.phase,
            provenance=plan.provenance,
            execution_config_sha256=plan.sha256,
            engine_started=False,
            optimizer_created=False,
            capability_evaluated=False,
            pipeline_stage_unlocked=False,
        )
        write_document(args.output_dir / "stage_report.json", report, immutable=True)
        print(json.dumps(report, sort_keys=True))
        return 0
    report = execute_stage_actor(plan, registry, args.output_dir, preflight=args.preflight)
    print(json.dumps(report, sort_keys=True))
    return (
        0
        if report["status"] in ("COMPLETE", "TRUNCATED") and report["all_owned_sessions_closed"]
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
