#!/usr/bin/env python
"""Assemble verified measurements and evaluate registered research comparisons."""

import argparse
import json
from pathlib import Path

from poorstar.common.paths import source_root

TREE = source_root()


def read(path):
    return json.loads(Path(path).read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    analyze = sub.add_parser("analyze")
    analyze.add_argument("--plan", type=Path, required=True)
    analyze.add_argument("--measurements", type=Path, required=True)
    assemble = sub.add_parser("assemble")
    assemble.add_argument("--plan", type=Path, required=True)
    assemble.add_argument("--mapping", type=Path, required=True)
    selection = sub.add_parser("select")
    selection.add_argument("--identity", type=Path, required=True)
    selection.add_argument("--rule", type=Path, required=True)
    selection.add_argument("--candidate", type=Path, action="append", required=True)
    calibration = sub.add_parser("calibrate")
    calibration.add_argument("--inputs", type=Path, required=True)
    annotation = sub.add_parser("compile-sidecar")
    for name in (
        "identity",
        "model-config",
        "build-report",
        "split-manifest",
        "annotations",
        "annotation-evidence",
    ):
        annotation.add_argument("--" + name, type=Path, required=True)
    annotation.add_argument("--window", type=int, required=True)
    annotation.add_argument("--unroll-length", type=int, required=True)
    annotation.add_argument("--burn-in-length", type=int, required=True)
    annotation.add_argument("--annotation-policy-sha256", required=True)
    for command in (analyze, assemble, selection, calibration, annotation):
        command.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.resolve().is_relative_to(TREE):
        parser.error("fresh external evidence output required")
    from poorstar.league.ownership import write_document

    if args.command == "analyze":
        from poorstar.evaluation.runner import evaluate_registered

        result = evaluate_registered(read(args.plan), read(args.measurements))
    elif args.command == "assemble":
        from poorstar.evaluation.measurement_bridge import assemble_measurements

        result = assemble_measurements(args.plan, args.mapping)
    elif args.command == "select":
        from poorstar.evaluation.selection import SelectionRule, select_checkpoint
        from poorstar.training import TrainingIdentity

        value = read(args.rule)
        for name in ("candidate_policy_versions", "required_metrics"):
            value[name] = tuple(value[name])
        result = select_checkpoint(
            [read(p) for p in args.candidate],
            SelectionRule(**value),
            TrainingIdentity(**read(args.identity)),
        )
    elif args.command == "calibrate":
        from poorstar.evaluation.confidence import ConfidenceTarget, fit_temperature

        value = read(args.inputs)
        result = fit_temperature(
            value["predictions"],
            [
                ConfidenceTarget(**dict(t, evidence_sha256=tuple(t["evidence_sha256"])))
                for t in value["targets"]
            ],
            split=value["split"],
            evidence_sha256=value["evidence_sha256"],
            iterations=value["iterations"],
        )
    else:
        import torch

        from poorstar.data.bc.windows import BCWindowDataset
        from poorstar.evaluation.teachers import build_control_sidecar
        from poorstar.hierarchy import HierarchicalPolicy, HierarchyState
        from poorstar.training import TrainingIdentity
        from poorstar.training.config import model_config_sha256, read_model_config

        identity = TrainingIdentity(**read(args.identity))
        low, high = read_model_config(args.model_config)
        dataset = BCWindowDataset(
            args.build_report,
            args.split_manifest,
            unroll_length=args.unroll_length,
            burn_in_length=args.burn_in_length,
        )
        try:
            if (
                identity.dataset_sha256 != dataset.sha256
                or identity.split_sha256 != dataset.split["sha256"]
                or identity.model_config_sha256 != model_config_sha256(low, high)
            ):
                raise ValueError("annotation data/model/split differs from frozen identity")
            value = read(args.annotations)
            state = value.get("initial_control_state")
            rows = dataset.window(args.window)
            if any(dataset.split["assignments"][r.replay_sha256] != "train" for r in rows):
                raise ValueError("BC annotation compiler must use TRAIN data only")
            with torch.random.fork_rng(devices=[]):
                model = HierarchicalPolicy(low, high).eval()
            result = build_control_sidecar(
                rows,
                value["annotations"],
                initial_control_state=None
                if state is None
                else HierarchyState.from_dict(state, high),
                model=model,
                profile=dataset.profile,
                identity=identity,
                evidence_path=args.annotation_evidence,
                output_path=args.output,
                annotation_policy_sha256=args.annotation_policy_sha256,
            )
        finally:
            dataset.close()
        print(json.dumps(result))
        return
    result = dict(result)
    result.pop("content_digest", None)
    write_document(args.output, result, immutable=True)
    print(
        json.dumps(
            dict(status="EVIDENCE_WRITTEN", path=str(args.output), formal_training_started=False)
        )
    )


if __name__ == "__main__":
    main()
