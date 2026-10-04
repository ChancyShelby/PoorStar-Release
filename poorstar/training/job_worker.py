"""Training: job worker."""

import math
import random
import time
from dataclasses import asdict
from pathlib import Path

from poorstar.league.contracts import WorkerResult
from poorstar.league.ownership import authorize_worker, verify_worker_lease, write_document
from poorstar.training.checkpoint import file_hash
from poorstar.training.contracts import TrainingIdentity


def actor_parameter_counts(model, phase):
    from poorstar.training.learner import groups_for

    allowed = (
        {"backbone", "decoder", "manager", "task"}
        if phase == "BC"
        else (
            {"value"}
            if phase == "VALUE_ONLY"
            else {"manager", "value"}
            if phase == "MANAGER_RL"
            else {"backbone", "decoder", "manager", "task", "value"}
            if phase == "JOINT_RL"
            else None
        )
    )
    if allowed is None:
        raise ValueError("actor export requires an explicit training phase")
    groups = groups_for(model)
    return dict(
        parameter_count=sum(p.numel() for p in model.parameters()),
        trainable_parameter_count=sum(
            p.numel() for g, items in groups.items() if g in allowed for _, p in items
        ),
    )


def execute_actor_request(request):
    """Public real-worker entry point. Lease/gates cannot be opted out by CLI."""
    import os

    import numpy as np
    import torch

    from poorstar.actions.coarse import CoarseActionProfile
    from poorstar.common.integrity import source_integrity
    from poorstar.training.actor import SC2PairBackend, collect_pair
    from poorstar.training.config import specialist_program
    from poorstar.training.export import load_actor
    from poorstar.training.runtime import configure_bc_runtime
    from poorstar.training.storage import RolloutReader, RolloutWriter
    from poorstar.training.windows import sequence_windows

    if request.kind != "actor":
        raise ValueError("actor worker cannot execute a learner request")
    started = time.monotonic()
    backend = None
    report = None
    artifacts = {}
    closed = True
    directory = Path(request.output_dir) / "actor"
    tree = Path(__file__).resolve().parents[2]
    # Authorization failures cannot create a process or a training artifact.
    identity = authorize_worker(request)
    execution = request.execution
    job = request.payload["job"]
    for name in ("unroll_steps", "maximum_resident_steps", "maximum_sources"):
        if type(execution.get(name)) is not int or execution[name] < 1:
            raise ValueError("explicit positive actor option required: " + name)
    if type(execution.get("burn_in_steps")) is not int or execution["burn_in_steps"] < 0:
        raise ValueError("explicit nonnegative burn-in required")
    if (
        execution.get("device") not in ("cpu", "cuda")
        or not execution.get("sc2_path")
        or not execution.get("version")
    ):
        raise ValueError("qualified device/SC2 runtime/version required")
    manifest = Path(execution["source_manifest"])
    if (
        file_hash(manifest) != identity.source_sha256
        or source_integrity(tree, manifest)["problems"]
    ):
        raise ValueError("owned actor source preflight failed")
    if job["provenance"] != request.payload["provenance"]:
        raise ValueError("actor job/worker provenance differs")
    if job["provenance"] == "learner" and execution.get("deterministic", False):
        raise ValueError("greedy evaluator actions cannot become PPO behavior trajectories")
    if execution.get("learner_phase") not in ("BC", "VALUE_ONLY", "MANAGER_RL", "JOINT_RL"):
        raise ValueError("frozen learner actor phase required")
    if job["provenance"] == "evaluator" and (
        type(execution.get("measurement_period")) is not int or execution["measurement_period"] < 1
    ):
        raise ValueError("frozen evaluator physical measurement clock required")
    if execution.get("preference_interventions") is not None and job["provenance"] != "evaluator":
        raise ValueError("paper interventions cannot alter PPO learner collection")
    directory.mkdir(exist_ok=False)
    try:
        os.environ["SC2PATH"] = execution["sc2_path"]
        configure_bc_runtime(execution["device"])
        random.seed(job["seed"])
        np.random.seed(job["seed"])
        torch.manual_seed(job["seed"])
        torch.set_num_threads(1)
        models = []
        versions = []
        programs = []
        counts = []
        episode = "league-job:" + job["job_id"]
        for name in ("learner", "opponent"):
            snapshot = request.payload["snapshots"][name]
            if snapshot["snapshot_id"] != job[name]:
                raise ValueError("actor snapshot differs from exact claimed opponent job")
            model, actor_identity, payload = load_actor(
                snapshot["weights_path"],
                sha256=snapshot["weights_sha256"],
                device=execution["device"],
            )
            if (
                asdict(actor_identity) != snapshot["identity"]
                or payload["policy_version"] != snapshot["policy_version"]
            ):
                raise ValueError("owned immutable actor identity/version differs")
            for key in ("scope_sha256", "profile_sha256", "matchup"):
                if getattr(actor_identity, key) != getattr(identity, key):
                    raise ValueError("paired actor scope/action domain differs: " + key)
            if name == "learner" and actor_identity != identity:
                raise ValueError("learner actor does not match owned experiment member")
            if name == "learner" and payload.get("training_phase") != execution["learner_phase"]:
                raise ValueError("learner actor phase differs from frozen worker execution")
            count = actor_parameter_counts(model, payload.get("training_phase"))
            limit = request.payload["parameter_budget"]
            if count["parameter_count"] > limit["maximum_parameters"] or (
                name == "learner"
                and count["trainable_parameter_count"] > limit["maximum_trainable_parameters"]
            ):
                raise ValueError("paired actor exceeds frozen parameter budget")
            if name == "learner" and any(count[k] != limit[k] for k in count):
                raise ValueError("owned member parameter counts differ from frozen population plan")
            counts.append(
                dict(
                    player=name,
                    phase=payload["training_phase"],
                    trainable_in_this_worker=0,
                    **count,
                )
            )
            models.append(model)
            versions.append(payload["policy_version"])
            programs.append(
                None
                if snapshot["intent_profile"] is None
                else specialist_program(model.hierarchy_config, snapshot["intent_profile"], episode)
            )
        if execution.get("preference_interventions") is not None:
            from poorstar.evaluation.interventions import ScheduledPreferenceInterventions
            from poorstar.hierarchy.interventions import PreferenceIntervention

            schedules = execution["preference_interventions"]
            if set(schedules) - {"learner", "opponent"}:
                raise ValueError("explicit intervention player name required")
            for index, name in enumerate(("learner", "opponent")):
                if schedules.get(name):
                    if programs[index] is not None:
                        raise ValueError(
                            "specialist renewal and causal intervention require separate registered cells"
                        )
                    programs[index] = ScheduledPreferenceInterventions(
                        models[index].hierarchy_config,
                        episode=episode,
                        interventions=tuple(
                            PreferenceIntervention(**dict(v, preference=tuple(v["preference"])))
                            for v in schedules[name]
                        ),
                    )
        if (
            models[0].config.sha256 != models[1].config.sha256
            or models[0].hierarchy_config.arm != identity.arm
        ):
            raise ValueError("paired shared structured action architecture or learner arm differs")
        from poorstar.evaluation.metrics import VisibleMetricRecorder

        measurements = VisibleMetricRecorder() if job["provenance"] == "evaluator" else None
        backend = SC2PairBackend(
            map_name=job["map_name"], version=execution["version"], seed=job["seed"]
        )
        closed = False
        with backend:
            profiles = [
                CoarseActionProfile.from_catalog(c, maximum_sources=execution["maximum_sources"])
                for c in backend.catalogs
            ]
            if any(p.sha256 != identity.profile_sha256 for p in profiles):
                raise ValueError("actual SC2 profile differs from frozen actor domain")
            sequence, report = collect_pair(
                backend,
                models,
                profiles,
                identity,
                policy_versions=tuple(versions),
                maximum_loops=job["maximum_loops"],
                episode_id=episode,
                provenance=job["provenance"],
                external_programs=tuple(programs),
                deterministic=bool(execution.get("deterministic", False)),
                maximum_resident_steps=execution["maximum_resident_steps"],
                measurement_sink=measurements,
                measurement_period=execution.get("measurement_period"),
            )
            catalog = backend.catalogs[0]
        closed = all(p["exit_code"] is not None for p in backend.owned_process_results)
        if not closed:
            raise ValueError("owned SC2 processes not closed")
        if source_integrity(tree, manifest)["problems"]:
            raise ValueError("owned actor source postflight failed")
        verify_worker_lease(request)
        report.update(
            job_id=job["job_id"],
            request_sha256=request.sha256,
            actor_sampling_seed=job["seed"],
            owned_process_results=backend.owned_process_results,
            parameter_counts=counts,
            external_program_sha256=[None if p is None else p.sha256 for p in programs],
        )
        if measurements is not None:
            write_document(
                directory / "evaluator_metrics.json", measurements.report(report), immutable=True
            )
            artifacts["evaluator_metrics"] = dict(
                path=str(directory / "evaluator_metrics.json"),
                sha256=file_hash(directory / "evaluator_metrics.json"),
            )
        # Only the learner writer can publish optimizer inputs. Payoff games
        # remain independent even when they run exactly the same actor API.
        if job["provenance"] == "learner":
            shards = []
            for window in sequence_windows(
                sequence,
                unroll_steps=execution["unroll_steps"],
                burn_in_steps=execution["burn_in_steps"],
            ):
                writer = RolloutWriter(
                    directory / "rollouts",
                    identity,
                    catalog,
                    maximum_sources=execution["maximum_sources"],
                    maximum_delay=models[0].config.maximum_delay,
                    initial_hidden=window.initial_hidden,
                    starts_at_episode_boundary=window.starts_at_episode_boundary,
                )
                for step in window.steps:
                    writer.append(step)
                artifact = writer.close()
                reader = RolloutReader(
                    artifact["path"], sha256=artifact["sha256"], identity=identity
                )
                try:
                    reader.sequence()
                finally:
                    reader.close()
                shards.append(artifact)
            index = dict(
                schema="poorstar_v3_rollout_index_v1",
                identity_sha256=identity.sha256,
                job_id=job["job_id"],
                physical_environment_loops=report["environment_loops"],
                shards=shards,
                update_groups=[[s["sha256"] for s in shards]],
            )
            write_document(directory / "rollout_index.json", index, immutable=True)
            artifacts["rollout_index"] = dict(
                path=str(directory / "rollout_index.json"),
                sha256=file_hash(directory / "rollout_index.json"),
            )
        write_document(directory / "report.json", report, immutable=True)
        artifacts["report"] = dict(
            path=str(directory / "report.json"), sha256=file_hash(directory / "report.json")
        )
        if execution["device"] == "cuda":
            torch.cuda.synchronize()
        status = "COMPLETE" if report["terminated"] else "TRUNCATED"
    except BaseException as error:
        if backend is not None:
            try:
                backend.close()
                closed = all(p["exit_code"] is not None for p in backend.owned_process_results)
            except BaseException:
                closed = False
        failure = dict(
            schema="poorstar_v3_owned_actor_failure_v1",
            request_sha256=request.sha256,
            error_type=type(error).__name__,
            error=str(error),
            owned_sessions_closed=closed,
            observed_report=report,
            owned_process_results=[] if backend is None else backend.owned_process_results,
        )
        write_document(directory / "failure.json", failure, immutable=True)
        artifacts["failure"] = dict(
            path=str(directory / "failure.json"), sha256=file_hash(directory / "failure.json")
        )
        status = "ERROR"
    result = WorkerResult(
        request.sha256,
        request.operation_id,
        request.attempt_id,
        request.claim_token,
        "actor",
        status,
        math.ceil((time.monotonic() - started) * 1000),
        None if report is None else report["environment_loops"],
        closed,
        artifacts,
    )
    # An unresolved process remains quarantine evidence, never a valid payoff.
    if closed:
        result.validate(request)
    return result
