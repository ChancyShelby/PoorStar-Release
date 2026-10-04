#!/usr/bin/env python
"""Train the shared recurrent policy with supervised BC."""

import argparse
import json
import random
import time
from dataclasses import asdict
from pathlib import Path

from poorstar.common.paths import source_root

TREE = source_root()


def formal_guard(plan, dataset, binding, preflight):
    from poorstar.common.integrity import file_hash, payload_hash

    if plan.purpose != "FORMAL":
        return
    if dataset.report.get("purpose") != "FORMAL_CORPUS" or not dataset.report.get(
        "paper_corpus_gate_passed"
    ):
        raise ValueError(
            "formal launch refused: engineering/pilot corpus is not the full registered corpus"
        )
    if preflight is None:
        raise ValueError("formal launch requires frozen matched-arm launch preflight")
    gate = json.loads(Path(preflight).read_text())
    digest = gate.pop("content_digest")
    if (
        payload_hash(gate) != digest
        or gate.get("schema") != "poorstar_c3_formal_launch_preflight_v1"
    ):
        raise ValueError("preflight drift")
    required = (
        "full_corpus",
        "source_identity",
        "cpu_complete_tests",
        "cuda_complete_tests",
        "domain_accounting",
        "trainability",
        "resume",
        "matched_arms_and_budget",
        "frozen_evaluator",
        "actor_parity",
        "storage_throughput",
    )
    if (
        gate.get("status") != "READY"
        or gate.get("binding") != binding
        or any(gate.get("gates", {}).get(k) != "PASS" for k in required)
    ):
        raise ValueError("formal launch prerequisites do not close")
    for name in required:
        evidence = gate["evidence"][name]
        if file_hash(Path(evidence["path"])) != evidence["sha256"]:
            raise ValueError("formal preflight evidence drift: " + name)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("build-report", "split-manifest", "run-plan", "source-manifest", "output-dir"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--preflight", type=Path)
    p.add_argument("--resume-from", type=Path)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    p.add_argument("--checkpoint-every", type=int, default=250)
    a = p.parse_args()
    if a.output_dir.exists() or a.output_dir.resolve().is_relative_to(TREE):
        p.error("fresh external output namespace required, including resume attempts")
    if a.checkpoint_every < 1:
        p.error("positive checkpoint cadence required")
    import numpy as np
    import torch

    from poorstar.common.integrity import file_hash, payload_hash, source_integrity
    from poorstar.data.bc.windows import (
        BCRunPlan,
        BCWindowDataset,
        PolicyObservationCache,
        WindowOrder,
        load_checkpoint,
        save_checkpoint,
    )
    from poorstar.models.policy import HEADS, BCModelConfig, RecurrentBCPolicy
    from poorstar.training.foundation import BCAggregate, BCTrainer

    if source_integrity(TREE, a.source_manifest)["problems"]:
        raise ValueError("source preflight failed")
    plan = BCRunPlan(**json.loads(a.run_plan.read_text()))
    from poorstar.training.runtime import configure_bc_runtime

    runtime = configure_bc_runtime(a.device, plan.runtime_mode)
    random.seed(plan.seed)
    np.random.seed(plan.seed)
    torch.manual_seed(plan.seed)
    torch.set_num_threads(1)
    ds = BCWindowDataset(
        a.build_report,
        a.split_manifest,
        unroll_length=plan.unroll_length,
        burn_in_length=plan.burn_in_length,
    )
    try:
        cfg = BCModelConfig.from_catalog(
            ds.catalog,
            ds.profile,
            hidden_size=plan.hidden_size,
            maximum_delay=ds.maximum_delay,
            autoregressive=plan.autoregressive,
        )
        model = RecurrentBCPolicy(cfg).to(a.device)
        # No optimizer before formal gating. Binding needs only model/profile config.
        binding = dict(
            plan_sha256=plan.sha256,
            source_sha256=file_hash(a.source_manifest),
            dataset_sha256=ds.sha256,
            model_config_sha256=cfg.sha256,
            profile_sha256=ds.profile.sha256,
        )
        formal_guard(plan, ds, binding, a.preflight)
        trainer = BCTrainer(
            model,
            ds.profile,
            learning_rate=plan.learning_rate,
            gradient_clip=plan.gradient_clip,
            observation_cache=PolicyObservationCache(max(64, plan.unroll_length * 4)),
            supervision_policy=plan.supervision_policy,
            feature_batch_size=plan.feature_batch_size,
        )
        order = WindowOrder(len(ds.windows), plan.seed)
        aggregate = BCAggregate()
        if a.resume_from:
            load_checkpoint(a.resume_from, trainer, order, binding, aggregate=aggregate)
        if trainer.steps >= plan.total_steps:
            raise ValueError("recovered run already exhausted optimizer budget")
        a.output_dir.mkdir(parents=True)
        identity = dict(
            schema="poorstar_c3_bc_run_identity_v1",
            binding=binding,
            plan=asdict(plan),
            original_train_rows=ds.rows,
            windows=len(ds.windows),
            device=a.device,
            torch_version=torch.__version__,
            resume_from=str(a.resume_from) if a.resume_from else None,
            optimizer_steps_at_start=trainer.steps,
        )
        identity["runtime"] = runtime
        (a.output_dir / "identity.json").write_text(json.dumps(identity, indent=2) + "\n")
        start = time.monotonic()
        with (a.output_dir / "steps.jsonl").open("x") as log:
            while trainer.steps < plan.total_steps:
                indices = order.take(plan.batch_windows)
                sequences = [ds.window(i) for i in indices]
                for group in trainer.optimizer.param_groups:
                    group["lr"] = plan.lr_at(trainer.steps)
                result = trainer.step_microbatched(sequences)
                aggregate.add(result)
                record = dict(
                    step=trainer.steps,
                    window_indices=indices,
                    sampler_cursor=order.cursor,
                    head_losses={
                        h: float(v) / result.denominators[h] for h, v in result.head_sums.items()
                    },
                    denominators=result.denominators,
                    counts=result.counts,
                    rejections=result.rejections,
                    lr=trainer.optimizer.param_groups[0]["lr"],
                    elapsed_seconds=time.monotonic() - start,
                )
                log.write(json.dumps(record) + "\n")
                log.flush()
                if trainer.steps % a.checkpoint_every == 0 or trainer.steps == plan.total_steps:
                    save_checkpoint(
                        a.output_dir / ("checkpoint_" + str(trainer.steps).zfill(8)),
                        trainer,
                        order,
                        binding,
                        aggregate=aggregate,
                    )
                if trainer.steps % 20 == 0:
                    print(json.dumps(record), flush=True)
        if source_integrity(TREE, a.source_manifest)["problems"]:
            raise ValueError("source postflight failed")
        report = dict(
            schema="poorstar_c3_bc_optimizer_run_v1",
            identity=identity,
            aggregate=aggregate.report(required_heads=("act", "delay") + HEADS),
            aggregate_scope="WHOLE_RUN_INCLUDING_RESUMED_STEPS",
            optimizer_steps=trainer.steps,
            elapsed_seconds=time.monotonic() - start,
            budget_complete=True,
            capability_evaluated=False,
            rl_ready=False,
        )
        report["content_digest"] = payload_hash(report)
        (a.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    except BaseException as error:
        if a.output_dir.exists():
            (a.output_dir / "failure.json").write_text(
                json.dumps(dict(error_type=type(error).__name__, error=str(error)), indent=2)
            )
        raise
    finally:
        ds.close()


if __name__ == "__main__":
    main()
