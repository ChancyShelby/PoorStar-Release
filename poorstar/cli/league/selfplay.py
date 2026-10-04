#!/usr/bin/env python
"""Cli: league: selfplay."""

import argparse
import json
import os
from pathlib import Path

from poorstar.common.paths import source_root

TREE = source_root()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("identity", "league-config", "registry", "database", "output-dir"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--learner", required=True)
    p.add_argument("--opponent", action="append", required=True)
    p.add_argument("--map", required=True)
    p.add_argument("--owner", required=True)
    p.add_argument("--provenance", choices=("learner", "evaluator"), default="learner")
    p.add_argument("--dispatch", action="store_true")
    p.add_argument("--preflight", type=Path)
    p.add_argument("--source-manifest", type=Path)
    p.add_argument("--sc2-path", type=Path)
    p.add_argument("--version", default="3.16.1")
    p.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    p.add_argument("--unroll-steps", type=int, required=True)
    p.add_argument("--burn-in-steps", type=int, required=True)
    p.add_argument("--maximum-resident-steps", type=int, required=True)
    a = p.parse_args()
    if (
        a.output_dir.exists()
        or a.output_dir.resolve().is_relative_to(TREE)
        or a.database.resolve().is_relative_to(TREE)
    ):
        p.error("fresh external outputs and host-local external league database required")
    if a.unroll_steps < 1 or a.burn_in_steps < 0 or a.maximum_resident_steps < 1:
        p.error("positive unroll and nonnegative causal burn-in required")
    if a.dispatch and (not a.sc2_path or not a.source_manifest):
        p.error("dispatch requires qualified SC2 and an immutable source manifest")
    os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
    if a.sc2_path:
        os.environ["SC2PATH"] = str(a.sc2_path)
    from poorstar.league import LeagueConfig, LeagueScheduler, SnapshotRegistry
    from poorstar.training import TrainingIdentity
    from poorstar.training.actor import SC2PairBackend, collect_pair
    from poorstar.training.contracts import digest
    from poorstar.training.export import load_actor
    from poorstar.training.gates import validate_launch
    from poorstar.training.storage import RolloutWriter
    from poorstar.training.windows import sequence_windows

    identity = TrainingIdentity(**json.loads(a.identity.read_text()))
    config = LeagueConfig(**json.loads(a.league_config.read_text()))
    registry = SnapshotRegistry(a.registry)
    if a.dispatch:
        from poorstar.common.integrity import file_hash, source_integrity

        if (
            source_integrity(TREE, a.source_manifest)["problems"]
            or file_hash(a.source_manifest) != identity.source_sha256
        ):
            raise ValueError("actor source identity preflight failed")
        validate_launch(
            a.preflight,
            identity,
            "SELFPLAY",
            execution_config_sha256=digest(
                dict(
                    league=__import__("dataclasses").asdict(config),
                    unroll_steps=a.unroll_steps,
                    burn_in_steps=a.burn_in_steps,
                    maximum_resident_steps=a.maximum_resident_steps,
                )
            ),
        )
        from poorstar.training.runtime import configure_bc_runtime

        configure_bc_runtime(a.device)
    scheduler = LeagueScheduler(a.database, registry, identity, config)
    job = None
    try:
        queued = scheduler.schedule(a.learner, a.opponent, provenance=a.provenance, map_name=a.map)
        a.output_dir.mkdir(parents=True)
        (a.output_dir / "queued.json").write_text(json.dumps(queued, indent=2) + "\n")
        if not a.dispatch:
            print(json.dumps(dict(status="QUEUED_ONLY", job=queued, engine_started=False)))
            return
        job = scheduler.claim(a.owner, job_id=queued["job_id"])
        if job is None:
            raise ValueError("no owned league job available")
        # The job seed binds both engine and policy sampling, rather than only
        # SC2 map randomness. Loading immutable actors preserves this stream.
        import random

        import numpy as np
        import torch

        random.seed(job["seed"])
        np.random.seed(job["seed"])
        torch.manual_seed(job["seed"])
        torch.set_num_threads(1)
        policies = []
        versions = []
        programs = []
        episode = "selfplay-job:" + job["job_id"]
        for player, snapshot_id in enumerate((job["learner"], job["opponent"])):
            snapshot = registry.read(snapshot_id)
            model, actor_identity, actor = load_actor(
                registry.weights_path(snapshot_id),
                sha256=snapshot["weights_sha256"],
                device=a.device,
            )
            if actor_identity.model_config_sha256 != snapshot["model_config_sha256"]:
                raise ValueError("snapshot actor architecture mismatch")
            if (
                actor_identity.sha256 != snapshot["identity_sha256"]
                or actor["policy_version"] != snapshot["policy_version"]
            ):
                raise ValueError("snapshot export identity/version mismatch")
            if player == 0 and actor_identity != identity:
                raise ValueError("full learner training identity must match self-play run")
            common = ("scope_sha256", "profile_sha256")
            learner_fields = (
                (
                    "experiment",
                    "arm",
                    "seed",
                    "source_sha256",
                    "model_config_sha256",
                    "dataset_sha256",
                    "split_sha256",
                )
                if player == 0
                else ()
            )
            for name in common + learner_fields:
                if getattr(actor_identity, name) != getattr(identity, name):
                    raise ValueError("registered actor/run identity mismatch: " + name)
            policies.append(model)
            versions.append(actor["policy_version"])
            if snapshot["role"] == "intent_specialist":
                from poorstar.training.config import specialist_program

                programs.append(
                    specialist_program(model.hierarchy_config, snapshot["intent_profile"], episode)
                )
            else:
                programs.append(None)
        if (
            policies[0].hierarchy_config.arm != identity.arm
            or policies[0].config.sha256 != policies[1].config.sha256
        ):
            raise ValueError("learner identity or shared lower action architecture mismatch")
        from poorstar.actions.coarse import CoarseActionProfile

        with SC2PairBackend(
            map_name=job["map_name"], version=a.version, seed=job["seed"]
        ) as backend:
            profiles = [
                CoarseActionProfile.from_catalog(c, maximum_sources=63) for c in backend.catalogs
            ]
            if any(profile.sha256 != identity.profile_sha256 for profile in profiles):
                raise ValueError("actual engine action profile differs from frozen actor domain")
            sequence, report = collect_pair(
                backend,
                policies,
                profiles,
                identity,
                policy_versions=tuple(versions),
                maximum_loops=job["maximum_loops"],
                provenance=job["provenance"],
                episode_id=episode,
                external_programs=tuple(programs),
                maximum_resident_steps=a.maximum_resident_steps,
            )
            catalog = backend.catalogs[0]
        report["owned_process_results"] = backend.owned_process_results
        report["actor_sampling_seed"] = job["seed"]
        if source_integrity(TREE, a.source_manifest)["problems"]:
            raise ValueError("actor source identity postflight failed")
        artifacts = []
        for window in (
            sequence_windows(sequence, unroll_steps=a.unroll_steps, burn_in_steps=a.burn_in_steps)
            if job["provenance"] == "learner"
            else ()
        ):
            writer = RolloutWriter(
                a.output_dir / "rollouts",
                identity,
                catalog,
                maximum_sources=63,
                maximum_delay=policies[0].config.maximum_delay,
                initial_hidden=window.initial_hidden,
                starts_at_episode_boundary=window.starts_at_episode_boundary,
            )
            for step in window.steps:
                writer.append(step)
            artifacts.append(writer.close())
        if job["provenance"] == "learner":
            index = dict(
                schema="poorstar_v3_rollout_index_v1",
                identity_sha256=identity.sha256,
                job_id=job["job_id"],
                physical_environment_loops=report["environment_loops"],
                shards=artifacts,
                update_groups=[[artifact["sha256"] for artifact in artifacts]],
            )
            index["content_digest"] = digest(index)
            (a.output_dir / "rollout_index.json").write_text(json.dumps(index, indent=2) + "\n")
        report["content_digest"] = digest(report)
        (a.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        from poorstar.training.checkpoint import file_hash

        verdict = (
            {-1.0: "loss", 0.0: "tie", 1.0: "win"}[report["reward"]]
            if report["terminated"]
            else "truncated"
        )
        scheduler.finish(
            job["job_id"],
            a.owner,
            verdict=verdict,
            environment_loops=report["environment_loops"],
            evidence_sha256=file_hash(a.output_dir / "report.json"),
        )
        print(
            json.dumps(
                dict(
                    status="COMPLETE" if report["terminated"] else "TRUNCATED",
                    actor=report,
                    training_started=False,
                )
            )
        )
    except BaseException as error:
        if a.output_dir.exists():
            failure = dict(error_type=type(error).__name__, error=str(error), job=job)
            (a.output_dir / "failure.json").write_text(json.dumps(failure, indent=2) + "\n")
            if job is not None:
                from poorstar.training.checkpoint import file_hash

                scheduler.finish(
                    job["job_id"],
                    a.owner,
                    verdict="error",
                    evidence_sha256=file_hash(a.output_dir / "failure.json"),
                    error=str(error),
                )
        raise
    finally:
        scheduler.close()


if __name__ == "__main__":
    main()
