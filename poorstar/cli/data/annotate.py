#!/usr/bin/env python
"""Build typed Intent and Task supervision bound to BC windows."""

import argparse
import json
import os
from pathlib import Path

from poorstar.common.paths import source_root

TREE = source_root()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "identity",
        "model-config",
        "plan",
        "source-manifest",
        "build-report",
        "split-manifest",
        "components",
        "output-dir",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument(
        "--storage",
        choices=("indexed_v3", "merged_v2"),
        default="indexed_v3",
        help="bounded indexed windows by default; resident merged v2 is an explicit compatibility option",
    )
    args = parser.parse_args()
    if args.output_dir.exists() or args.output_dir.resolve().is_relative_to(TREE):
        parser.error("fresh external corpus annotation output required")
    os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
    import torch

    from poorstar.common.integrity import source_integrity
    from poorstar.data.bc.windows import BCRunPlan, BCWindowDataset
    from poorstar.evaluation.sidecar_corpus import (
        COMPONENT_SCHEMA,
        compose_control_sidecar_corpus,
        compose_indexed_control_sidecar_corpus,
    )
    from poorstar.hierarchy import HierarchicalPolicy
    from poorstar.training import TrainingIdentity
    from poorstar.training.checkpoint import file_hash
    from poorstar.training.config import model_config_sha256, read_model_config
    from poorstar.training.contracts import digest
    from poorstar.training.runtime import configure_bc_runtime

    read = lambda path: json.loads(path.read_text())
    identity = TrainingIdentity(**read(args.identity))
    low, high = read_model_config(args.model_config)
    if (
        file_hash(args.source_manifest) != identity.source_sha256
        or source_integrity(TREE, args.source_manifest)["problems"]
        or model_config_sha256(low, high) != identity.model_config_sha256
        or high.arm != identity.arm
    ):
        raise ValueError("source/model identity differs from frozen annotation inputs")
    plan_doc = read(args.plan)
    if (
        set(plan_doc) != {"plan", "head_weights", "control_weights"}
        or digest(plan_doc) != identity.budget_sha256
    ):
        raise ValueError("BC window/objective plan must be frozen in annotation identity")
    plan = BCRunPlan(**plan_doc["plan"])
    if plan.seed != identity.seed or plan.purpose != identity.purpose:
        raise ValueError("annotation seed/purpose differs from frozen BC plan")
    components = read(args.components)
    check = dict(components)
    claimed = check.pop("content_digest", None)
    if (
        digest(check) != claimed
        or components.get("schema") != COMPONENT_SCHEMA
        or components.get("dataset_sha256") != identity.dataset_sha256
    ):
        raise ValueError("complete frozen window/component mapping required")
    configure_bc_runtime(args.device)
    torch.set_num_threads(1)
    with torch.random.fork_rng(
        devices=list(range(torch.cuda.device_count())) if args.device == "cuda" else []
    ):
        torch.manual_seed(identity.seed)
        model = HierarchicalPolicy(low, high).to(args.device).eval()
    dataset = BCWindowDataset(
        args.build_report,
        args.split_manifest,
        unroll_length=plan.unroll_length,
        burn_in_length=plan.burn_in_length,
    )
    try:
        compose = (
            compose_indexed_control_sidecar_corpus
            if args.storage == "indexed_v3"
            else compose_control_sidecar_corpus
        )
        result = compose(
            dataset,
            components["windows"],
            model=model,
            identity=identity,
            output_dir=args.output_dir,
            head_weights=plan_doc["head_weights"],
            control_weights=plan_doc["control_weights"],
            supervision_policy=plan.supervision_policy,
        )
        if source_integrity(TREE, args.source_manifest)["problems"]:
            raise ValueError("source changed during corpus annotation qualification")
        print(json.dumps(result))
    finally:
        dataset.close()


if __name__ == "__main__":
    main()
