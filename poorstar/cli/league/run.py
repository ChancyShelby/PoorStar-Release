#!/usr/bin/env python
"""Coordinate actor, learner and independent evaluator league jobs."""

import argparse
import json
from pathlib import Path

from poorstar.common.paths import source_root

TREE = source_root()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("identity", "run-plan", "registry", "output-dir"):
        p.add_argument("--" + name, type=Path, required=True)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--fixture-run", action="store_true")
    mode.add_argument("--dispatch", action="store_true")
    p.add_argument("--maximum-operations", type=int, default=1)
    p.add_argument("--actor-worker", type=Path)
    p.add_argument("--actor-worker-sha256")
    a = p.parse_args()
    if a.maximum_operations < 1 or a.output_dir.resolve().is_relative_to(TREE):
        p.error("positive bounded operation count and external host-local output root required")
    if a.dispatch and (not a.actor_worker or not a.actor_worker_sha256):
        p.error("qualified source-bound actor worker required for dispatch")
    from poorstar.league import SnapshotRegistry
    from poorstar.league.contracts import LeagueRunPlan
    from poorstar.league.coordinator import LeagueCoordinator
    from poorstar.league.ownership import write_document
    from poorstar.training import TrainingIdentity

    identity = TrainingIdentity(**json.loads(a.identity.read_text()))
    plan = LeagueRunPlan.from_dict(json.loads(a.run_plan.read_text()))
    if a.fixture_run and identity.purpose != "ENGINEERING":
        p.error("fixture requires ENGINEERING identity and separate outputs")
    registry = SnapshotRegistry(a.registry)
    # Planning is a labelled separate SQLite identity; it cannot be resumed as
    # a real experiment, even if someone later supplies gate documents.
    fixture = not a.dispatch
    coordinator = LeagueCoordinator(a.output_dir, registry, identity, plan, fixture=fixture)
    try:
        if a.dispatch:
            from poorstar.league.dispatch import SubprocessDispatcher

            dispatcher = SubprocessDispatcher(
                a.actor_worker, actor_worker_sha256=a.actor_worker_sha256
            )
        elif a.fixture_run:
            from poorstar.league.fixture import FixtureDispatcher

            dispatcher = FixtureDispatcher()
        else:
            report = coordinator.report()
            report["status"] = "PLAN_ONLY"
            report["engine_started"] = False
            print(json.dumps(report, sort_keys=True))
            return
        recovered = coordinator.recover()
        count = 0
        while count < a.maximum_operations:
            result = coordinator.run_next(dispatcher)
            if result is None:
                break
            count += 1
        report = coordinator.report()
        report.update(
            recovered_attempts=recovered,
            operations_this_invocation=count,
            status="FIXTURE_ONLY" if a.fixture_run else "FORMAL_DISPATCH",
            formal_bc_started=False,
        )
        index = len(tuple(a.output_dir.glob("report_*.json")))
        write_document(
            a.output_dir / ("report_" + str(index).zfill(6) + ".json"), report, immutable=True
        )
        print(json.dumps(report, sort_keys=True))
    finally:
        coordinator.close()


if __name__ == "__main__":
    main()
