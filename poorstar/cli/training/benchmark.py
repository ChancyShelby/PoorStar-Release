#!/usr/bin/env python
"""Measure BC memory, throughput and checkpoint recovery."""

import argparse
import json
import random
import resource
import sys
import time
from dataclasses import asdict
from pathlib import Path

from poorstar.common.paths import source_root

TREE = source_root()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("build-report", "split-manifest", "source-manifest", "output-dir"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    p.add_argument("--steps", type=int, default=3)
    p.add_argument("--batch-windows", type=int, default=8)
    p.add_argument("--unroll-length", type=int, default=32)
    p.add_argument("--burn-in-length", type=int, default=16)
    p.add_argument("--feature-batch-size", type=int, default=1)
    p.add_argument(
        "--supervision-policy",
        choices=("STRICT_WHOLE_BODY_V1", "DISTAR_COARSE_SKIP_V1"),
        default="STRICT_WHOLE_BODY_V1",
    )
    a = p.parse_args()
    if a.output_dir.exists() or a.output_dir.resolve().is_relative_to(TREE):
        p.error("fresh external output required")
    import numpy as np
    import torch

    from poorstar.common.integrity import file_hash, payload_hash, source_integrity
    from poorstar.data.bc.windows import (
        BCRunPlan,
        BCWindowDataset,
        PolicyObservationCache,
        WindowOrder,
    )
    from poorstar.models.policy import BCModelConfig, RecurrentBCPolicy
    from poorstar.training.foundation import BCTrainer

    plan = BCRunPlan(
        "ENGINEERING",
        20261003,
        a.steps,
        batch_windows=a.batch_windows,
        unroll_length=a.unroll_length,
        burn_in_length=a.burn_in_length,
        feature_batch_size=a.feature_batch_size,
        supervision_policy=a.supervision_policy,
    )
    if source_integrity(TREE, a.source_manifest)["problems"]:
        raise ValueError("source preflight failed")
    from poorstar.training.runtime import configure_bc_runtime

    runtime = configure_bc_runtime(a.device, plan.runtime_mode)
    torch.set_num_threads(1)
    a.output_dir.mkdir(parents=True)
    started = time.monotonic()
    ds = BCWindowDataset(
        a.build_report,
        a.split_manifest,
        unroll_length=plan.unroll_length,
        burn_in_length=plan.burn_in_length,
    )
    indexing = time.monotonic() - started
    order = WindowOrder(len(ds.windows), plan.seed)
    batches = [order.take(plan.batch_windows) for _ in range(plan.total_steps)]
    prereg = dict(
        plan=asdict(plan),
        batches=batches,
        dataset_sha256=ds.sha256,
        source_manifest_sha256=file_hash(a.source_manifest),
        arms=["C3_CONDITIONAL", "PARALLEL_WITHIN_COMMAND_CONTROL"],
        purpose="ENGINEERING_RESOURCE_MEASUREMENT",
        selection="SEEDED_TRAIN_WINDOWS_BEFORE_MODEL_CREATION",
        formal_numeric_budget_frozen=False,
        formal_bc_started=False,
        rl_ready=False,
        runtime=runtime,
    )
    (a.output_dir / "prereg.json").write_text(json.dumps(prereg, indent=2) + "\n")
    arms = []
    try:
        for autoregressive, name in [
            (True, "C3_CONDITIONAL"),
            (False, "PARALLEL_WITHIN_COMMAND_CONTROL"),
        ]:
            random.seed(plan.seed)
            np.random.seed(plan.seed)
            torch.manual_seed(plan.seed)
            cfg = BCModelConfig.from_catalog(
                ds.catalog,
                ds.profile,
                hidden_size=plan.hidden_size,
                maximum_delay=ds.maximum_delay,
                autoregressive=autoregressive,
            )
            model = RecurrentBCPolicy(cfg).to(a.device)
            trainer = BCTrainer(
                model,
                ds.profile,
                learning_rate=plan.learning_rate,
                observation_cache=PolicyObservationCache(
                    plan.batch_windows * (plan.unroll_length + plan.burn_in_length) * 2
                ),
                supervision_policy=plan.supervision_policy,
                feature_batch_size=plan.feature_batch_size,
            )
            if a.device == "cuda":
                torch.cuda.reset_peak_memory_stats()
            measurements = []
            for indices in batches:
                start = time.monotonic()
                sequences = [ds.window(i) for i in indices]
                load_seconds = time.monotonic() - start
                if a.device == "cuda":
                    torch.cuda.synchronize()
                start = time.monotonic()
                result = trainer.step_microbatched(sequences)
                if a.device == "cuda":
                    torch.cuda.synchronize()
                measurements.append(
                    dict(
                        load_seconds=load_seconds,
                        optimizer_seconds=time.monotonic() - start,
                        active_rows=sum(not r.burn_in and r.loss for s in sequences for r in s),
                        burn_in_rows=sum(r.burn_in for s in sequences for r in s),
                        denominators=result.denominators,
                        counts=result.counts,
                        rejections=result.rejections,
                        loss=float(result.loss.detach()),
                    )
                )
                print(
                    json.dumps(dict(arm=name, step=trainer.steps, measurement=measurements[-1])),
                    flush=True,
                )
            elapsed = sum(x["load_seconds"] + x["optimizer_seconds"] for x in measurements)
            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (
                1 if sys.platform == "darwin" else 1024
            )
            arms.append(
                dict(
                    arm=name,
                    parameter_count=sum(p.numel() for p in model.parameters()),
                    steps=measurements,
                    active_rows_per_second=sum(x["active_rows"] for x in measurements) / elapsed,
                    cumulative_process_peak_rss_bytes=rss,
                    cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated()
                    if a.device == "cuda"
                    else None,
                    cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved()
                    if a.device == "cuda"
                    else None,
                    compute_note="SOURCE_PREFIX_GRU_CALLS_DIFFER_BETWEEN_ARMS;_EQUAL_PARAMETER_COUNT_IS_NOT_EQUAL_COMPUTE",
                )
            )
            del result, trainer, model
            if a.device == "cuda":
                torch.cuda.empty_cache()
        if source_integrity(TREE, a.source_manifest)["problems"]:
            raise ValueError("source postflight failed")
        if [x["denominators"] for x in arms[0]["steps"]] != [
            x["denominators"] for x in arms[1]["steps"]
        ]:
            raise ValueError("matched engineering batches changed head denominators")
        report = dict(
            prereg,
            indexing_seconds=indexing,
            arms=arms,
            device=a.device,
            torch_version=torch.__version__,
            matched_input_and_denominators=True,
            measurement_complete=True,
            full_corpus_throughput_certified=False,
            formal_bc_ready=False,
        )
        report["content_digest"] = payload_hash(report)
        (a.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    except BaseException as e:
        (a.output_dir / "failure.json").write_text(
            json.dumps(dict(error=f"{type(e).__name__}: {e}", completed_arms=arms), indent=2)
        )
        raise
    finally:
        ds.close()


if __name__ == "__main__":
    main()
