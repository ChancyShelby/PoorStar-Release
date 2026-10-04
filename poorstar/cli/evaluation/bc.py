#!/usr/bin/env python
"""Evaluate supervised and free-running BC bodies on development data."""

import argparse
import json
import math
import time
from pathlib import Path

from poorstar.common.paths import source_root

TREE = source_root()


def evaluate(model, dataset, profile, settings, *, supervision_policy, feature_batch_size):
    from collections import Counter

    import torch

    from poorstar.data.bc.windows import PolicyObservationCache
    from poorstar.evaluation.bc import evaluate_sequence_bodies
    from poorstar.training.foundation import BCAggregate, bc_objective

    if dataset.allowed_split != "development":
        raise ValueError("development split required")
    if set(settings) != {
        "schema",
        "unroll_length",
        "burn_in_length",
        "position_tolerance",
        "token_watchdog",
    }:
        raise ValueError("frozen evaluator settings schema drift")
    if settings["schema"] != "poorstar_c3_development_evaluator_v1":
        raise ValueError("evaluator version drift")
    if (
        not math.isfinite(settings["position_tolerance"])
        or settings["position_tolerance"] <= 0
        or any(
            type(settings[k]) is not int
            for k in ("unroll_length", "burn_in_length", "token_watchdog")
        )
        or min(settings["unroll_length"], settings["token_watchdog"]) < 1
        or settings["burn_in_length"] < 0
    ):
        raise ValueError("positive finite evaluator bounds required")
    model.eval()
    aggregate = BCAggregate()
    free_counts = Counter()
    errors = []
    active = 0
    cache = PolicyObservationCache(max(64, settings["unroll_length"] * 4))
    with torch.no_grad():
        for i in range(len(dataset.windows)):
            rows = dataset.window(i)
            active += sum(r.valid and r.loss and not r.burn_in for r in rows)
            aggregate.add(
                bc_objective(
                    model,
                    [rows],
                    profile,
                    observation_cache=cache,
                    supervision_policy=supervision_policy,
                    feature_batch_size=feature_batch_size,
                )
            )
            free = evaluate_sequence_bodies(
                model,
                [rows],
                profile,
                supervision_policy=supervision_policy,
                position_tolerance=settings["position_tolerance"],
                token_watchdog=settings["token_watchdog"],
            )
            free_counts.update(free["counts"])
            errors.extend(free["position_errors"])
    if active != dataset.rows:
        raise ValueError("development active-row denominator does not close")
    accuracies = [
        free_counts[d + "_correct"] / free_counts[d + "_rows"]
        for d in ("ACT", "WAIT")
        if free_counts[d + "_rows"]
    ]
    # Sparse/missing development heads are reported, never invented or a fit gate.
    from poorstar.models.policy import HEADS

    return dict(
        teacher=aggregate.report(),
        active_unique_rows=active,
        windows=len(dataset.windows),
        missing_development_heads=[
            h for h in ("act", "delay") + HEADS if aggregate.denominators[h] == 0
        ],
        free_body=dict(
            counts=dict(free_counts),
            position_errors=errors,
            evaluated_scheduler_rows=sum(free_counts[d + "_rows"] for d in ("ACT", "WAIT")),
            scheduler_row_coverage_fraction=sum(free_counts[d + "_rows"] for d in ("ACT", "WAIT"))
            / active,
            scheduler_balanced_accuracy=sum(accuracies) / len(accuracies) if accuracies else None,
            bundle_within_tolerance_fraction=free_counts["bundle_within_tolerance"]
            / max(1, free_counts["admissible_bodies"]),
            actor_mode="WINDOWED_FREE_RUNNING_BODY_EXPERT_SCHEDULE",
            watchdog_rows_may_truncate_diagnostic_sequence=True,
            online_scheduler_evaluated=False,
        ),
        settings=settings,
        supervision_policy=supervision_policy,
        reference_scope="DECLARED_REPRESENTABLE_COMMAND_PROJECTION",
        optimizer_steps=0,
        test_read=False,
        capability_verdict="METRICS_ONLY_NO_PROMOTION",
        rl_ready=False,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for n in (
        "build-report",
        "split-manifest",
        "source-manifest",
        "run-plan",
        "checkpoint",
        "evaluator-plan",
        "output",
    ):
        p.add_argument("--" + n, type=Path, required=True)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    a = p.parse_args()
    if a.output.exists() or a.output.resolve().is_relative_to(TREE):
        p.error("fresh external output required")
    import torch

    from poorstar.common.integrity import file_hash, payload_hash, source_integrity
    from poorstar.data.bc.windows import BCRunPlan, BCWindowDataset, load_policy_checkpoint
    from poorstar.models.policy import BCModelConfig, RecurrentBCPolicy

    pre = source_integrity(TREE, a.source_manifest)
    if pre["problems"]:
        raise ValueError("source preflight failed")
    plan = BCRunPlan(**json.loads(a.run_plan.read_text()))
    settings = json.loads(a.evaluator_plan.read_text())
    from poorstar.training.runtime import configure_bc_runtime

    runtime = configure_bc_runtime(a.device, plan.runtime_mode)
    train = BCWindowDataset(
        a.build_report,
        a.split_manifest,
        unroll_length=plan.unroll_length,
        burn_in_length=plan.burn_in_length,
    )
    development = None
    try:
        cfg = BCModelConfig.from_catalog(
            train.catalog,
            train.profile,
            hidden_size=plan.hidden_size,
            maximum_delay=train.maximum_delay,
            autoregressive=plan.autoregressive,
        )
        binding = dict(
            plan_sha256=plan.sha256,
            source_sha256=file_hash(a.source_manifest),
            dataset_sha256=train.sha256,
            model_config_sha256=cfg.sha256,
            profile_sha256=train.profile.sha256,
        )
        model = RecurrentBCPolicy(cfg).to(a.device)
        torch.set_num_threads(1)
        checkpoint = load_policy_checkpoint(
            a.checkpoint,
            model,
            binding,
            supervision_policy=plan.supervision_policy,
            feature_batch_size=plan.feature_batch_size,
        )
        train.close()
        development = BCWindowDataset(
            a.build_report,
            a.split_manifest,
            allowed_split="development",
            unroll_length=settings["unroll_length"],
            burn_in_length=settings["burn_in_length"],
        )
        if development.profile.sha256 != binding["profile_sha256"]:
            raise ValueError("development actor profile drift")
        start = time.monotonic()
        report = evaluate(
            model,
            development,
            development.profile,
            settings,
            supervision_policy=plan.supervision_policy,
            feature_batch_size=plan.feature_batch_size,
        )
        post = source_integrity(TREE, a.source_manifest)
        if post["problems"]:
            raise ValueError("source postflight failed")
        report.update(
            schema="poorstar_c3_development_evaluation_v1",
            binding=binding,
            dataset_sha256=development.sha256,
            checkpoint_report_digest=checkpoint["content_digest"],
            evaluator_plan_sha256=file_hash(a.evaluator_plan),
            source_pre=pre,
            source_post=post,
            device=a.device,
            seconds=time.monotonic() - start,
            runtime=runtime,
        )
        report["content_digest"] = payload_hash(report)
        a.output.parent.mkdir(parents=True, exist_ok=True)
        with a.output.open("x") as f:
            f.write(json.dumps(report, indent=2) + "\n")
    finally:
        train.close()
        if development is not None:
            development.close()


if __name__ == "__main__":
    main()
