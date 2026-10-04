#!/usr/bin/env python
"""Check bounded BC updates and checkpoint recovery against TRAIN data."""

import argparse
import json
import os
import random
from dataclasses import asdict
from pathlib import Path

from poorstar.common.paths import source_root

TREE = source_root()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("build-report", "split-manifest", "output-dir"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument(
        "--outcome-mechanisms",
        action="store_true",
        help="explicit static real-frame mathematical outcome gradients; never live on-policy qualification",
    )
    a = p.parse_args()
    if a.output_dir.exists() or a.output_dir.resolve().is_relative_to(TREE):
        p.error("fresh external engineering evidence directory required")
    os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
    import numpy as np
    import torch

    from poorstar.common.integrity import file_hash
    from poorstar.data.bc.windows import BCRunPlan, BCWindowDataset
    from poorstar.hierarchy import HierarchicalPolicy, HierarchyConfig
    from poorstar.hierarchy.bc import HierarchyBCRecord, HierarchyBCSequence
    from poorstar.models.policy import BCModelConfig
    from poorstar.training import TrainingIdentity
    from poorstar.training.bc import HierarchicalBCLearner
    from poorstar.training.checkpoint import load_checkpoint, save_checkpoint
    from poorstar.training.config import model_config_sha256
    from poorstar.training.contracts import digest
    from poorstar.training.runtime import configure_bc_runtime

    source_paths = tuple((TREE / "poorstar").rglob("*.py")) + (Path(__file__).resolve(),)
    sources = {str(p.relative_to(TREE)): file_hash(p) for p in source_paths}
    source_sha = digest(sources)
    configure_bc_runtime(a.device)
    torch.set_num_threads(1)
    random.seed(71)
    np.random.seed(71)
    torch.manual_seed(71)
    dataset = BCWindowDataset(a.build_report, a.split_manifest, unroll_length=8, burn_in_length=4)
    a.output_dir.mkdir(parents=True)
    try:
        # Existing dataset iteration is TRAIN-only. First fixed window is a
        # declared bounded engineering rule, never a validation-based selection.
        rows = dataset.window(0)
        if any(dataset.split["assignments"][r.replay_sha256] != "train" for r in rows):
            raise ValueError("engineering check may not read held-out data")
        low = BCModelConfig.from_catalog(
            dataset.catalog,
            dataset.profile,
            hidden_size=32,
            maximum_delay=dataset.maximum_delay,
            position_grid=(4, 4),
        )
        high = HierarchyConfig(arm="flat")
        plan = BCRunPlan(
            "ENGINEERING",
            71,
            2,
            hidden_size=32,
            unroll_length=8,
            burn_in_length=4,
            supervision_policy="DISTAR_COARSE_SKIP_V1",
        )
        identity = TrainingIdentity(
            "real-train-bc-engineering",
            "flat",
            71,
            source_sha,
            dataset.sha256,
            dataset.split["sha256"],
            digest(["engineering-data-reader", dataset.sha256]),
            model_config_sha256(low, high),
            dataset.profile.sha256,
            digest(dict(plan=asdict(plan), head_weights={}, control_weights={})),
        )
        model = HierarchicalPolicy(low, high).to(a.device)
        learner = HierarchicalBCLearner(model, dataset.profile, identity, plan)
        sequence = HierarchyBCSequence(tuple(HierarchyBCRecord(r) for r in rows))
        first = learner.update([sequence])
        save_checkpoint(a.output_dir / "checkpoint_first", learner, sampler_state={"cursor": 1})
        expected = learner.update([sequence])
        expected_digest = learner.parameter_sha256
        restored = HierarchicalBCLearner(
            HierarchicalPolicy(low, high).to(a.device), dataset.profile, identity, plan
        )
        recovered = load_checkpoint(a.output_dir / "checkpoint_first", restored)
        actual = restored.update([sequence])
        if (
            actual != expected
            or restored.parameter_sha256 != expected_digest
            or recovered["sampler_state"] != {"cursor": 1}
        ):
            raise ValueError("actual hierarchical BC next-update recovery mismatch")
        outcome_mechanisms = None
        if a.outcome_mechanisms:
            from poorstar.training.qualification import qualify_static_outcome_gradients

            outcome_mechanisms = qualify_static_outcome_gradients(
                rows,
                dataset.profile,
                low,
                device=a.device,
                output_dir=a.output_dir,
                source_sha256=source_sha,
                dataset_sha256=dataset.sha256,
                split_sha256=dataset.split["sha256"],
            )
        if any(file_hash(TREE / name) != value for name, value in sources.items()):
            raise ValueError("source changed during engineering check")
        report = dict(
            schema="poorstar_v3_training_system_real_train_canary_v1",
            status="PASS",
            purpose="ENGINEERING",
            identity=asdict(identity),
            source_files=sources,
            rows=len(rows),
            original_train_replays=sorted({r.replay_sha256 for r in rows}),
            first_update=first,
            second_update=actual,
            exact_rebuilt_next_update=True,
            formal_optimizer_steps=0,
            sc2_started=False,
            held_out_read=False,
            high_level_replay_labels_available=False,
            capability_evaluated=False,
            rl_ready=False,
            outcome_mechanisms=outcome_mechanisms,
            live_on_policy_qualified=False,
            scope="ONE_FIXED_TRAIN_WINDOW_NOT_FULL_CORPUS_OR_GENERALIZATION",
        )
        report["content_digest"] = digest(report)
        (a.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(
            json.dumps(
                dict(
                    status="PASS",
                    report=str(a.output_dir / "report.json"),
                    rows=len(rows),
                    exact_recovery=True,
                    formal_started=False,
                )
            )
        )
    except BaseException as error:
        (a.output_dir / "failure.json").write_text(
            json.dumps(dict(error_type=type(error).__name__, error=str(error)), indent=2) + "\n"
        )
        raise
    finally:
        dataset.close()


if __name__ == "__main__":
    main()
