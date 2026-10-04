#!/usr/bin/env python
"""Train hierarchical BC or staged VALUE_ONLY, MANAGER_RL and JOINT_RL PPO."""

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
    p.add_argument("--stage", choices=("BC", "VALUE_ONLY", "MANAGER_RL", "JOINT_RL"), required=True)
    for name in ("identity", "model-config", "plan", "source-manifest", "output-dir"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--build-report", type=Path)
    p.add_argument("--split-manifest", type=Path)
    p.add_argument("--control-sidecar", type=Path)
    p.add_argument("--rollout-index", type=Path)
    p.add_argument("--corpus-qualification", type=Path)
    p.add_argument("--preflight", type=Path)
    p.add_argument("--resume-from", type=Path)
    p.add_argument("--initial-actor", type=Path)
    p.add_argument("--initial-actor-sha256")
    p.add_argument("--initial-foundation-checkpoint", type=Path)
    p.add_argument("--initial-foundation-binding", type=Path)
    p.add_argument("--stage-transition", type=Path)
    p.add_argument(
        "--initialize-stage-only",
        action="store_true",
        help="publish verified stage actor/checkpoint0/empty consumed prefix; zero updates and no engine",
    )
    p.add_argument("--policy-gradient-audit", action="store_true")
    p.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    p.add_argument("--checkpoint-every", type=int, default=250)
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    if (
        a.output_dir.exists()
        or a.output_dir.resolve().is_relative_to(TREE)
        or a.checkpoint_every < 1
    ):
        p.error("fresh external output namespace and positive checkpoint cadence required")
    if sum(bool(v) for v in (a.resume_from, a.initial_actor, a.initial_foundation_checkpoint)) > 1:
        p.error("initialization and exact resume are mutually exclusive")
    if a.initial_foundation_checkpoint and (a.stage != "BC" or not a.initial_foundation_binding):
        p.error("named foundation initialization requires BC and its complete frozen C3 binding")
    if a.initial_foundation_binding and not a.initial_foundation_checkpoint:
        p.error("C3 binding requires the original foundation checkpoint directory")
    if a.initialize_stage_only and (
        a.stage == "BC"
        or not a.initial_actor
        or a.rollout_index
        or not a.build_report
        or not a.split_manifest
    ):
        p.error(
            "stage initialization requires RL phase, predecessor actor, BC build/split metadata and no rollout index"
        )
    os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
    import numpy as np
    import torch

    from poorstar.common.integrity import file_hash, source_integrity
    from poorstar.hierarchy import HierarchicalPolicy
    from poorstar.training import PPOConfig, PPOLearner, TrainingIdentity
    from poorstar.training.bc import (
        HierarchicalBCLearner,
        load_foundation_initialization,
        validate_bc_contract,
    )
    from poorstar.training.checkpoint import load_checkpoint
    from poorstar.training.config import ControlSidecar, model_config_sha256, read_model_config
    from poorstar.training.contracts import digest
    from poorstar.training.export import export_actor, load_actor
    from poorstar.training.gates import validate_launch
    from poorstar.training.publication import publish_candidate, publish_candidate_index
    from poorstar.training.runtime import configure_bc_runtime
    from poorstar.training.scorer import HierarchyScorer
    from poorstar.training.stages import (
        ZERO_SHA,
        cumulative_counters,
        initialize_stage,
        read_transition,
        validate_transition,
    )
    from poorstar.training.storage import RolloutReader

    identity = TrainingIdentity(**json.loads(a.identity.read_text()))
    low, high = read_model_config(a.model_config)
    if high.arm != identity.arm or model_config_sha256(low, high) != identity.model_config_sha256:
        raise ValueError("model/arm identity drift")
    if (
        source_integrity(TREE, a.source_manifest)["problems"]
        or file_hash(a.source_manifest) != identity.source_sha256
    ):
        raise ValueError("source identity preflight failed")
    configure_bc_runtime(a.device)
    torch.set_num_threads(1)
    random.seed(identity.seed)
    np.random.seed(identity.seed)
    torch.manual_seed(identity.seed)
    model = HierarchicalPolicy(low, high).to(a.device)
    initial = None
    foundation = None
    if a.initial_actor:
        if identity.initializer_sha256 != a.initial_actor_sha256:
            raise ValueError("initial actor must be frozen in the training identity")
        initial_model, initial_identity, initial = load_actor(
            a.initial_actor, sha256=a.initial_actor_sha256, device=a.device
        )
        for name in (
            "experiment",
            "arm",
            "seed",
            "dataset_sha256",
            "split_sha256",
            "scope_sha256",
            "model_config_sha256",
            "profile_sha256",
        ):
            if getattr(initial_identity, name) != getattr(identity, name):
                raise ValueError("initial actor experiment/data/architecture mismatch: " + name)
        model.load_state_dict(initial_model.state_dict())
    if a.initial_foundation_checkpoint:
        binding = json.loads(a.initial_foundation_binding.read_text())
        for name, expected in (
            ("dataset_sha256", identity.dataset_sha256),
            ("profile_sha256", identity.profile_sha256),
            ("model_config_sha256", low.sha256),
        ):
            if binding.get(name) != expected:
                raise ValueError("foundation initializer data/model/profile drift: " + name)
        foundation = load_foundation_initialization(
            a.initial_foundation_checkpoint / "state.pt",
            model,
            expected_sha256=identity.initializer_sha256,
            source_report=a.initial_foundation_checkpoint / "report.json",
            expected_binding=binding,
        )
    dataset = None
    readers = []
    try:
        plan_document = json.loads(a.plan.read_text())
        if a.stage == "BC":
            from poorstar.data.bc.windows import (
                BCRunPlan,
                BCWindowDataset,
                PolicyObservationCache,
                WindowOrder,
            )
            from poorstar.hierarchy.bc import HierarchyBCRecord, HierarchyBCSequence

            if not a.build_report or not a.split_manifest:
                p.error("BC requires frozen build report and split manifest")
            if set(plan_document) != {"plan", "head_weights", "control_weights"}:
                raise ValueError("BC numeric plan and both objective dictionaries required")
            plan = BCRunPlan(**plan_document["plan"])
            if (
                digest(plan_document) != identity.budget_sha256
                or plan.seed != identity.seed
                or plan.purpose != identity.purpose
            ):
                raise ValueError("BC budget/objective identity drift")
            dataset = BCWindowDataset(
                a.build_report,
                a.split_manifest,
                unroll_length=plan.unroll_length,
                burn_in_length=plan.burn_in_length,
            )
            if (
                dataset.sha256 != identity.dataset_sha256
                or dataset.split["sha256"] != identity.split_sha256
                or dataset.profile.sha256 != identity.profile_sha256
            ):
                raise ValueError("BC data/split/action profile drift")
            if identity.purpose == "FORMAL":
                # Extraction reports retain their original source/scope role.
                # A separate frozen full-corpus audit binds this indexed data.
                if (
                    not a.corpus_qualification
                    or file_hash(a.corpus_qualification) != identity.corpus_qualification_sha256
                ):
                    raise ValueError(
                        "formal BC requires its frozen complete ZvZ qualification artifact"
                    )
                qualification = json.loads(a.corpus_qualification.read_text())
                check = dict(qualification)
                claimed = check.pop("content_digest", None)
                binding = dict(
                    dataset_sha256=dataset.sha256,
                    split_sha256=identity.split_sha256,
                    scope_sha256=identity.scope_sha256,
                    build_report_sha256=file_hash(a.build_report),
                    profile_sha256=identity.profile_sha256,
                    maximum_delay=dataset.maximum_delay,
                )
                if (
                    digest(check) != claimed
                    or qualification.get("schema") != "poorstar_v3_full_zvz_corpus_qualification_v1"
                    or qualification.get("status") != "PASS"
                    or qualification.get("binding") != binding
                    or qualification.get("full_registered_views")
                    != qualification.get("completed_views")
                    or qualification.get("full_registered_views", 0) < 1
                    or not qualification.get("full_train_census")
                ):
                    raise ValueError("formal BC full corpus/index/census qualification drift")
            sidecar = None
            if high.arm != "flat":
                if not a.control_sidecar:
                    raise ValueError(
                        "higher-level labels unavailable: explicit causal sidecar required"
                    )
                sidecar = ControlSidecar(a.control_sidecar, identity, high)

            def sequence_at(i):
                rows = dataset.window(i)
                return (
                    sidecar.sequence(rows, window_index=i)
                    if sidecar is not None
                    else HierarchyBCSequence(tuple(HierarchyBCRecord(r) for r in rows))
                )

            profile = dataset.profile
            validate_bc_contract(
                model,
                profile,
                identity,
                plan,
                catalog=dataset.catalog,
                maximum_delay=dataset.maximum_delay,
                head_weights=plan_document["head_weights"],
                control_weights=plan_document["control_weights"],
            )
            # High-level labels must cover ALL actual windows and their causal
            # burn-in snapshots before Adam, including in --dry-run. Plain flat
            # BC retains the bounded first-window engineering check.
            from poorstar.hierarchy.bc import hierarchy_bc_objective

            with torch.no_grad():
                if sidecar is not None:
                    sample_contract = sidecar.validate_dataset(
                        dataset,
                        model,
                        head_weights=plan_document["head_weights"],
                        control_weights=plan_document["control_weights"],
                        supervision_policy=plan.supervision_policy,
                    )
                else:
                    sample = sequence_at(0)
                    sample_result = hierarchy_bc_objective(
                        model,
                        [sample],
                        profile,
                        head_weights=plan_document["head_weights"],
                        control_weights=plan_document["control_weights"],
                        collect_metrics=False,
                        supervision_policy=plan.supervision_policy,
                    )
                    sample_contract = dict(
                        windows=1,
                        original_rows=len(sample.records),
                        denominators=sample_result.denominators,
                    )
        else:
            if not a.rollout_index and not a.initialize_stage_only:
                p.error("RL phases require immutable rollout index")
            plan = PPOConfig(**plan_document)
            if plan.sha256 != identity.budget_sha256:
                raise ValueError("RL numeric budget identity drift")
            if a.initialize_stage_only:
                from poorstar.data.bc.windows import BCWindowDataset
                from poorstar.training.stages import validate_dataset_window

                window = initial.get("dataset_window")
                validate_dataset_window(window)
                dataset = BCWindowDataset(a.build_report, a.split_manifest, **window)
                if (
                    dataset.sha256 != identity.dataset_sha256
                    or dataset.split["sha256"] != identity.split_sha256
                ):
                    raise ValueError("stage bootstrap original BC data/split/window identity drift")
                profile = dataset.profile
                if (
                    profile.sha256 != identity.profile_sha256
                    or dataset.maximum_delay != low.maximum_delay
                ):
                    raise ValueError("stage bootstrap original action profile/horizon drift")
                sample_contract = dict(
                    metadata_only=True,
                    original_rows_read=0,
                    original_dataset_window=window,
                    catalog_profile_checked=True,
                )
                groups = []
            else:
                from poorstar.training.admission import validate_rollout_admission

                validate_rollout_admission(a.rollout_index, identity)
                index = json.loads(a.rollout_index.read_text())
                check = dict(index)
                claimed = check.pop("content_digest", None)
                if (
                    digest(check) != claimed
                    or index.get("identity_sha256") != identity.sha256
                    or not index.get("shards")
                ):
                    raise ValueError("rollout index identity/content drift")
                for artifact in index["shards"]:
                    readers.append(
                        RolloutReader(
                            artifact["path"], sha256=artifact["sha256"], identity=identity
                        )
                    )
                # Every frozen behavior group updates once. Windows retain bounded
                # individual graphs, while all group gradients share denominators.
                groups = index.get("update_groups")
                if (
                    not isinstance(groups, list)
                    or not groups
                    or any(not isinstance(g, list) or not g for g in groups)
                    or [v for g in groups for v in g] != [v["sha256"] for v in index["shards"]]
                    or any(len(g) > plan.maximum_batch_sequences for g in groups)
                ):
                    raise ValueError("explicit bounded logical rollout update groups required")
                profile = readers[0].profile
                if any(r.profile.sha256 != profile.sha256 for r in readers):
                    raise ValueError("mixed rollout action profiles")
                if profile.sha256 != identity.profile_sha256 or any(
                    r.index["maximum_delay"] != low.maximum_delay for r in readers
                ):
                    raise ValueError("RL model/action profile or scheduler horizon drift")
        transition = None
        if a.initial_actor:
            transition = read_transition(
                a.stage_transition,
                identity,
                a.stage,
                plan,
                manager_available=bool(model.parameter_groups()["manager"]),
            )
            validate_transition(
                transition,
                identity,
                a.stage,
                initial_identity,
                initial,
                a.initial_actor_sha256,
                plan,
                manager_available=bool(model.parameter_groups()["manager"]),
            )
        elif identity.stage_transition_sha256 != ZERO_SHA:
            if not a.resume_from:
                raise ValueError("declared stage transition has no initializer or exact resume")
            transition = read_transition(
                a.stage_transition,
                identity,
                a.stage,
                plan,
                manager_available=bool(model.parameter_groups()["manager"]),
            )
        elif a.stage_transition:
            raise ValueError("stage manifest is not frozen in training identity")
        elif (
            identity.initializer_sha256 != ZERO_SHA
            and not a.resume_from
            and not a.initial_foundation_checkpoint
        ):
            raise ValueError("declared initializer has no verified source artifact")
        if identity.purpose == "FORMAL":
            if a.stage != "BC" and not a.initial_actor and not a.resume_from:
                raise ValueError(
                    "formal RL needs its frozen supervised/preceding-stage actor initialization"
                )
            validate_launch(a.preflight, identity, a.stage)
        if a.dry_run:
            print(
                json.dumps(
                    dict(
                        stage=a.stage,
                        identity_sha256=identity.sha256,
                        parameters=sum(p.numel() for p in model.parameters()),
                        sample_contract=sample_contract
                        if a.stage == "BC" or a.initialize_stage_only
                        else None,
                        stage_transition=None
                        if transition is None
                        else transition["content_digest"],
                        foundation_initialization=foundation,
                        optimizer_created=False,
                        formal_started=False,
                        capability_evaluated=False,
                    )
                )
            )
            return
        a.output_dir.mkdir(parents=True)
        if a.stage == "BC":
            learner = HierarchicalBCLearner(
                model,
                profile,
                identity,
                plan,
                head_weights=plan_document["head_weights"],
                control_weights=plan_document["control_weights"],
                observation_cache=PolicyObservationCache(max(64, plan.unroll_length * 4)),
            )
            order = WindowOrder(len(dataset.windows), plan.seed)
        else:
            learner = PPOLearner(
                model, HierarchyScorer(model, profile), identity, plan, phase=a.stage
            )
            if initial is not None:
                initialize_stage(learner, transition)
            order = None
        if a.initialize_stage_only:
            (a.output_dir / "identity.json").write_text(
                json.dumps(
                    dict(
                        identity=asdict(identity),
                        stage=a.stage,
                        model_config=dict(bc=asdict(low), hierarchy=asdict(high)),
                        plan=plan_document,
                        initial_actor=str(a.initial_actor),
                        initial_actor_sha256=a.initial_actor_sha256,
                        stage_transition=transition,
                        build_report_sha256=file_hash(a.build_report),
                        split_manifest_sha256=file_hash(a.split_manifest),
                        initialize_stage_only=True,
                    ),
                    indent=2,
                )
                + "\n"
            )
            empty = dict(
                schema="poorstar_v3_rollout_index_v1",
                identity_sha256=identity.sha256,
                shards=[],
                update_groups=[],
                physical_environment_loops=0,
            )
            empty["content_digest"] = digest(empty)
            empty_path = a.output_dir / "accumulated_rollout_index.json"
            empty_path.write_text(json.dumps(empty, indent=2) + "\n")
            sampler = dict(cursor=0, rollout_index_sha256=file_hash(empty_path), consumed_groups=[])
            item = publish_candidate(a.output_dir, learner, sampler_state=sampler)
            candidate_index = publish_candidate_index(
                a.output_dir / "candidate_index.json", identity, a.stage, [item]
            )
            checkpoint = Path(item["candidate"]["checkpoint_path"])
            result = dict(
                schema="poorstar_v3_stage_bootstrap_report_v1",
                identity_sha256=identity.sha256,
                stage=a.stage,
                updates=0,
                optimizer_steps=0,
                environment_loops=0,
                policy_version=learner.policy_version,
                actor=item["candidate"]["actor"],
                cumulative_counters=cumulative_counters(learner),
                candidate_index=candidate_index,
                checkpoint_manifest=dict(
                    path=str(checkpoint / "manifest.json"),
                    sha256=file_hash(checkpoint / "manifest.json"),
                ),
                accumulated_rollout_index=dict(path=str(empty_path), sha256=file_hash(empty_path)),
                sample_contract=sample_contract,
                optimizer_moments=0,
                engine_started=False,
                capability_evaluated=False,
                rl_ready=False,
            )
            if learner.optimizer.state or source_integrity(TREE, a.source_manifest)["problems"]:
                raise ValueError("stage bootstrap created optimizer moments or source drift")
            result["content_digest"] = digest(result)
            (a.output_dir / "report.json").write_text(json.dumps(result, indent=2) + "\n")
            print(json.dumps(result), flush=True)
            return
        cursor = 0
        if a.resume_from:
            learner.stage_transition = transition
            recovered = load_checkpoint(a.resume_from, learner)
            state = recovered["sampler_state"]
            if a.stage == "BC":
                if (
                    state["windows"] != order.windows
                    or state["seed"] != order.seed
                    or type(state["cursor"]) is not int
                    or state["cursor"] != learner.updates * plan.batch_windows
                ):
                    raise ValueError("BC recovered sampler drift")
                order.cursor = state["cursor"]
            else:
                cursor = state["cursor"]
                if (
                    type(cursor) is not int
                    or not 0 <= cursor <= len(groups)
                    or cursor != learner.updates
                    or state["consumed_groups"] != groups[:cursor]
                ):
                    raise ValueError("resumed rollout index is not an immutable prefix extension")
        (a.output_dir / "identity.json").write_text(
            json.dumps(
                dict(
                    identity=asdict(identity),
                    stage=a.stage,
                    model_config=dict(bc=asdict(low), hierarchy=asdict(high)),
                    plan=plan_document,
                    initial_actor=None if initial is None else str(a.initial_actor),
                    foundation_initialization=foundation,
                    initial_foundation_checkpoint=None
                    if foundation is None
                    else str(a.initial_foundation_checkpoint),
                    resumed_from=None if a.resume_from is None else str(a.resume_from),
                ),
                indent=2,
            )
            + "\n"
        )
        candidates = []
        with (a.output_dir / "steps.jsonl").open("x") as log:
            limit = plan.total_steps if a.stage == "BC" else plan.maximum_updates
            while learner.updates < limit:
                if a.stage == "BC":
                    indices = order.take(plan.batch_windows)
                    report = learner.update([sequence_at(i) for i in indices])
                    sampler = dict(cursor=order.cursor, windows=order.windows, seed=order.seed)
                else:
                    if cursor >= len(groups):
                        break  # explicit await-new-rollouts, never recycle old experience
                    offset = sum(len(g) for g in groups[:cursor])
                    batch = [r.sequence() for r in readers[offset : offset + len(groups[cursor])]]
                    audit = (
                        learner.policy_gradient_audit(batch) if a.policy_gradient_audit else None
                    )
                    report = learner.update(batch)
                    cursor += 1
                    if audit is not None:
                        report["policy_only_gradient_audit"] = audit
                    sampler = dict(
                        cursor=cursor,
                        rollout_index_sha256=file_hash(a.rollout_index),
                        consumed_groups=groups[:cursor],
                    )
                log.write(json.dumps(report, sort_keys=True) + "\n")
                log.flush()
                if (
                    learner.updates % a.checkpoint_every == 0
                    or learner.updates == limit
                    or (a.stage != "BC" and cursor == len(groups))
                ):
                    candidates.append(
                        publish_candidate(a.output_dir, learner, sampler_state=sampler)
                    )
                print(json.dumps(report), flush=True)
        actor = export_actor(a.output_dir / "actor.pt", learner)
        candidate_index = (
            publish_candidate_index(
                a.output_dir / "candidate_index.json", identity, a.stage, candidates
            )
            if candidates
            else None
        )
        result = dict(
            schema="poorstar_v3_training_run_report_v1",
            identity_sha256=identity.sha256,
            stage=a.stage,
            optimizer_steps=learner.optimizer_steps,
            updates=learner.updates,
            budget_complete=learner.updates == limit,
            actor=actor,
            waiting_for_rollouts=a.stage != "BC" and learner.updates < limit,
            cumulative_counters=cumulative_counters(learner),
            candidate_index=candidate_index,
            capability_evaluated=False,
            rl_ready=False,
        )
        if a.stage == "BC":
            result["aggregate"] = learner.aggregate.report(required_heads=("act", "delay"))
        if source_integrity(TREE, a.source_manifest)["problems"]:
            raise ValueError("source postflight failed")
        result["content_digest"] = digest(result)
        (a.output_dir / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    except BaseException as error:
        if a.output_dir.exists():
            (a.output_dir / "failure.json").write_text(
                json.dumps(dict(error_type=type(error).__name__, error=str(error)), indent=2) + "\n"
            )
        raise
    finally:
        if dataset is not None:
            dataset.close()
        for reader in readers:
            reader.close()


if __name__ == "__main__":
    main()
