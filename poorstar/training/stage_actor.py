"""Training: stage actor."""

import math
import os
import random
import time
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path

from poorstar.league.ownership import write_document
from poorstar.training.actor import SC2PairBackend, collect_pair
from poorstar.training.checkpoint import file_hash
from poorstar.training.contracts import PPOConfig, TrainingIdentity, digest, is_sha
from poorstar.training.deadline import CollectionDeadline
from poorstar.training.gates import validate_launch

FIXTURE_SCOPE = "PROTOCOL_FIXTURE_NO_ENGINE_NOT_PHASE_AUTHORIZATION"


@dataclass(frozen=True)
class StageActorPlan:
    identity: dict
    phase: str
    provenance: str
    collection_id: str
    learner_snapshot_id: str
    opponent_snapshot_id: str
    map_name: str
    seed: int
    maximum_loops: int
    maximum_compute_millis: int
    parameter_budget: dict
    execution: dict
    schema: str = "poorstar_v3_stage_actor_plan_v1"

    def __post_init__(self):
        identity = TrainingIdentity(**self.identity)
        if self.schema != "poorstar_v3_stage_actor_plan_v1" or self.phase not in (
            "BC",
            "VALUE_ONLY",
            "MANAGER_RL",
            "JOINT_RL",
        ):
            raise ValueError("explicit supported actor phase/schema required")
        if (
            self.provenance not in ("learner", "evaluator")
            or self.phase == "BC"
            and self.provenance == "learner"
        ):
            raise ValueError("BC uses replay targets; BC actor collection is evaluator-only")
        if (
            not all(
                is_sha(h)
                for h in (self.collection_id, self.learner_snapshot_id, self.opponent_snapshot_id)
            )
            or not self.map_name
        ):
            raise ValueError("content-bound collection and two fixed immutable actors required")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("explicit nonnegative engine/policy seed required")
        for name in ("maximum_loops", "maximum_compute_millis"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError("positive collection budget required: " + name)
        names = {
            "parameter_count",
            "trainable_parameter_count",
            "maximum_parameters",
            "maximum_trainable_parameters",
        }
        if set(self.parameter_budget) != names or any(
            type(v) is not int or v < 1 for v in self.parameter_budget.values()
        ):
            raise ValueError("complete explicit positive parameter budget required")
        p = self.parameter_budget
        if (
            p["parameter_count"] > p["maximum_parameters"]
            or p["trainable_parameter_count"] > p["maximum_trainable_parameters"]
            or p["trainable_parameter_count"] > p["parameter_count"]
            or p["maximum_trainable_parameters"] > p["maximum_parameters"]
        ):
            raise ValueError("actor parameter allocation exceeds frozen budget")
        e = self.execution
        for name in ("unroll_steps", "maximum_resident_steps", "maximum_sources"):
            if type(e.get(name)) is not int or e[name] < 1:
                raise ValueError("explicit positive actor option required: " + name)
        if type(e.get("burn_in_steps")) is not int or e["burn_in_steps"] < 0:
            raise ValueError("explicit causal burn-in required")
        for name in ("deterministic_learner", "deterministic_opponent"):
            if type(e.get(name)) is not bool:
                raise ValueError("explicit per-player sampling mode required")
        if self.provenance == "learner" and e["deterministic_learner"]:
            raise ValueError("greedy learner cannot publish PPO behavior trajectories")
        if self.provenance == "evaluator" and (
            type(e.get("measurement_period")) is not int or e["measurement_period"] < 1
        ):
            raise ValueError("evaluator requires a frozen positive physical measurement clock")
        if self.provenance == "evaluator" and not is_sha(e.get("metric_program_sha256")):
            raise ValueError("evaluator requires a frozen visible metric implementation")
        if self.provenance == "learner" and e.get("preference_interventions"):
            raise ValueError("paper interventions cannot alter PPO behavior collection")
        if "episode_id" in e and (
            self.provenance != "evaluator"
            or not isinstance(e["episode_id"], str)
            or not e["episode_id"]
        ):
            raise ValueError("explicit shared scenario episode is evaluator-only")
        if self.provenance == "learner":
            if not isinstance(e.get("learner_config"), dict):
                raise ValueError("full frozen receiving PPO configuration required")
            config = PPOConfig(**e["learner_config"])
            if config.sha256 != identity.budget_sha256:
                raise ValueError("receiving PPO configuration differs from identity budget")
            if (
                e["maximum_resident_steps"] > e["unroll_steps"] * config.maximum_batch_sequences
                or e["unroll_steps"] + e["burn_in_steps"] > config.maximum_sequence_steps
                or self.maximum_loops > config.maximum_environment_loops
            ):
                raise ValueError(
                    "collection would exceed the receiving logical batch/frame capacity"
                )
        if e.get("device") not in ("cpu", "cuda") or not all(
            e.get(k) for k in ("version", "sc2_path", "source_manifest")
        ):
            raise ValueError("explicit source/device/SC2 runtime required")
        if identity.matchup != "ZvZ":
            raise ValueError("stage actor retains the frozen ZvZ domain")

    @property
    def sha256(self):
        # The preflight reference is a separate CLI/API argument. All scientific
        # controls, identity, actors, clocks and budgets participate in this hash.
        return digest(asdict(self))


def validate_phase_preflight(plan, preflight):
    """Existing complete phase gate, with no future self-play/league requirement."""
    identity = TrainingIdentity(**plan.identity)
    return validate_launch(preflight, identity, plan.phase, execution_config_sha256=plan.sha256)


def _prepare_actors(plan, registry):
    from poorstar.evaluation.interventions import ScheduledPreferenceInterventions
    from poorstar.hierarchy.interventions import PreferenceIntervention
    from poorstar.training.config import specialist_program
    from poorstar.training.export import load_actor
    from poorstar.training.job_worker import actor_parameter_counts
    from poorstar.training.stages import actor_counters

    identity = TrainingIdentity(**plan.identity)
    models = []
    versions = []
    programs = []
    actors = {}
    episode = plan.execution.get("episode_id", "stage-actor:" + plan.sha256)
    for name, snapshot_id in (
        ("learner", plan.learner_snapshot_id),
        ("opponent", plan.opponent_snapshot_id),
    ):
        record = registry.read(snapshot_id)
        path = registry.weights_path(snapshot_id).resolve()
        model, actor_identity, payload = load_actor(
            path, sha256=record["weights_sha256"], device=plan.execution["device"]
        )
        actor_counters(payload)
        if (
            asdict(actor_identity) != record.get("training_identity")
            or payload["policy_version"] != record["policy_version"]
        ):
            raise ValueError("fixed snapshot/export full identity or version drift")
        for key in ("scope_sha256", "profile_sha256", "matchup"):
            if getattr(actor_identity, key) != getattr(identity, key):
                raise ValueError("fixed actors have incompatible scope/action domain: " + key)
        if name == "learner" and (
            actor_identity != identity or payload.get("training_phase") != plan.phase
        ):
            raise ValueError("learner exact exported identity/phase differs from collection plan")
        if name == "learner" and plan.provenance == "learner":
            limit = plan.execution["learner_config"]["maximum_environment_loops"]
            if payload["stage_counters"]["environment_loops"] + plan.maximum_loops > limit:
                raise ValueError(
                    "collection exceeds the remaining exported-stage physical frame budget"
                )
        counts = actor_parameter_counts(model, payload.get("training_phase"))
        budget = plan.parameter_budget
        if counts["parameter_count"] > budget["maximum_parameters"]:
            raise ValueError("fixed actor exceeds total parameter budget")
        if name == "learner" and any(counts[k] != budget[k] for k in counts):
            raise ValueError("learner effective phase parameter allocation differs")
        actors[name] = dict(
            snapshot_id=snapshot_id,
            weights_path=str(path),
            weights_sha256=record["weights_sha256"],
            identity=asdict(actor_identity),
            policy_version=payload["policy_version"],
            parameter_sha256=payload["parameter_sha256"],
            training_phase=payload["training_phase"],
            cumulative_counters=payload["cumulative_counters"],
            parameter_counts=counts,
            trainable_in_this_collector=0,
            intent_profile=record["intent_profile"],
        )
        models.append(model)
        versions.append(payload["policy_version"])
        programs.append(
            None
            if record["intent_profile"] is None
            else specialist_program(model.hierarchy_config, record["intent_profile"], episode)
        )
    if (
        models[0].config.sha256 != models[1].config.sha256
        or models[0].hierarchy_config.arm != identity.arm
    ):
        raise ValueError("fixed actors must share the structured action architecture")
    schedules = plan.execution.get("preference_interventions", {})
    if not isinstance(schedules, dict) or set(schedules) - {"learner", "opponent"}:
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
    return identity, episode, models, tuple(versions), tuple(programs), actors


def _closed(backend, fixture):
    if backend is None:
        return True
    results = getattr(backend, "owned_process_results", None)
    if fixture:
        return results == [] and getattr(backend, "closed", False) is True
    return (
        isinstance(results, list)
        and len(results) == 2
        and len({r.get("pid") for r in results}) == 2
        and all(
            type(r.get("pid")) is int and r["pid"] > 0 and type(r.get("exit_code")) is int
            for r in results
        )
    )


def execute_stage_actor(plan, registry, output_dir, *, preflight=None, protocol_fixture=None):
    """Collect one immutable fixed pair. CLI dispatch cannot inject a fixture.

    The optional protocol_fixture is an explicitly ENGINEERING-only arithmetic
    seam. It rejects a real backend and never provides phase authorization.
    Public real execution always consumes the complete existing phase preflight.
    """
    import numpy as np
    import torch

    from poorstar.actions.coarse import CoarseActionProfile
    from poorstar.common.integrity import source_integrity
    from poorstar.evaluation.metrics import VisibleMetricRecorder
    from poorstar.training.runtime import configure_bc_runtime
    from poorstar.training.storage import RolloutReader, RolloutWriter
    from poorstar.training.windows import sequence_windows

    output = Path(output_dir)
    tree = Path(__file__).resolve().parents[2]
    if output.exists() or output.resolve().is_relative_to(tree):
        raise ValueError("fresh external collection output required")
    fixture = protocol_fixture is not None
    identity = TrainingIdentity(**plan.identity)
    if fixture and (
        identity.purpose != "ENGINEERING"
        or isinstance(protocol_fixture, SC2PairBackend)
        or getattr(protocol_fixture, "qualification_scope", None) != FIXTURE_SCOPE
        or plan.execution["device"] != "cpu"
    ):
        raise ValueError("protocol fixture cannot authorize a FORMAL identity or real engine")
    # dataclass frozen=True does not make nested caller dictionaries immutable.
    # Execute an independent frozen payload and reject external plan mutation.
    supplied_plan = plan
    frozen_sha = plan.sha256
    plan = StageActorPlan(**deepcopy(asdict(plan)))
    output.mkdir(parents=True)
    write_document(output / "stage_plan.json", asdict(plan), immutable=True)
    started = time.monotonic()
    backend = None
    engine_attempted = False
    collector = None
    artifacts = {}
    actors = {}
    source_pre = None
    source_post = None
    status = "ERROR"
    error = None
    closure = True
    manifest = None
    launch_preflight = None
    deadline = CollectionDeadline(plan.maximum_compute_millis)

    def artifact(path):
        return dict(path=str(path.resolve()), sha256=file_hash(path))

    def check_budget():
        deadline.check()
        if isinstance(backend, SC2PairBackend):
            backend.check_budget()

    try:
        deadline.arm(required=not fixture)
        if not fixture:
            qualified = validate_phase_preflight(plan, preflight)
            launch_preflight = dict(
                path=str(Path(preflight).resolve()),
                sha256=file_hash(preflight),
                content_digest=qualified["content_digest"],
                stage=qualified["stage"],
            )
        manifest = Path(plan.execution["source_manifest"])
        source_pre = source_integrity(tree, manifest)
        if file_hash(manifest) != identity.source_sha256 or source_pre["problems"]:
            raise ValueError("stage actor source preflight failed")
        if (
            plan.provenance == "evaluator"
            and file_hash(tree / "poorstar/evaluation/metrics.py")
            != plan.execution["metric_program_sha256"]
        ):
            raise ValueError("stage evaluator metric program content drift")
        if not fixture:
            os.environ["SC2PATH"] = plan.execution["sc2_path"]
        configure_bc_runtime(plan.execution["device"])
        torch.set_num_threads(1)
        random.seed(plan.seed)
        np.random.seed(plan.seed)
        torch.manual_seed(plan.seed)
        identity, episode, models, versions, programs, actors = _prepare_actors(plan, registry)
        measurements = VisibleMetricRecorder() if plan.provenance == "evaluator" else None
        check_budget()
        backend = (
            protocol_fixture
            if fixture
            else SC2PairBackend(
                map_name=plan.map_name,
                version=plan.execution["version"],
                seed=plan.seed,
                deadline=deadline,
            )
        )
        if backend.version != plan.execution["version"]:
            raise ValueError("actual backend version differs from frozen execution")
        engine_attempted = not fixture
        with backend:
            profiles = [
                CoarseActionProfile.from_catalog(
                    c, maximum_sources=plan.execution["maximum_sources"]
                )
                for c in backend.catalogs
            ]
            if any(p.sha256 != identity.profile_sha256 for p in profiles):
                raise ValueError("actual backend catalog/action profile differs")
            sequence, collector = collect_pair(
                backend,
                models,
                profiles,
                identity,
                policy_versions=versions,
                maximum_loops=plan.maximum_loops,
                provenance=plan.provenance,
                episode_id=episode,
                external_programs=programs,
                maximum_resident_steps=plan.execution["maximum_resident_steps"],
                measurement_sink=measurements,
                measurement_period=plan.execution.get("measurement_period"),
                deterministic_players=(
                    plan.execution["deterministic_learner"],
                    plan.execution["deterministic_opponent"],
                ),
                check_budget=check_budget,
            )
            catalog = backend.catalogs[0]
        deadline.disarm()
        closure = _closed(backend, fixture)
        write_document(output / "collector_report.json", collector, immutable=True)
        artifacts["collector_report"] = artifact(output / "collector_report.json")
        if not closure:
            raise ValueError("stage actor owned process closure is unresolved")
        deadline.arm(required=not fixture)
        source_post = source_integrity(tree, manifest)
        if source_post["problems"] or file_hash(manifest) != identity.source_sha256:
            raise ValueError("stage actor source postflight failed")
        for name, snapshot in (
            ("learner", plan.learner_snapshot_id),
            ("opponent", plan.opponent_snapshot_id),
        ):
            current = registry.read(snapshot)
            if current["weights_sha256"] != actors[name]["weights_sha256"]:
                raise ValueError("registered immutable actor changed during collection")
        if supplied_plan.sha256 != frozen_sha:
            raise ValueError("caller collection plan drift")
        if (
            launch_preflight is not None
            and file_hash(launch_preflight["path"]) != launch_preflight["sha256"]
        ):
            raise ValueError("launch phase preflight content changed during collection")
        if measurements is not None:
            write_document(
                output / "evaluator_metrics.json", measurements.report(collector), immutable=True
            )
            artifacts["evaluator_metrics"] = artifact(output / "evaluator_metrics.json")
        else:
            shards = []
            for window in sequence_windows(
                sequence,
                unroll_steps=plan.execution["unroll_steps"],
                burn_in_steps=plan.execution["burn_in_steps"],
            ):
                check_budget()
                writer = RolloutWriter(
                    output / "rollouts",
                    identity,
                    catalog,
                    maximum_sources=plan.execution["maximum_sources"],
                    maximum_delay=models[0].config.maximum_delay,
                    initial_hidden=window.initial_hidden,
                    starts_at_episode_boundary=window.starts_at_episode_boundary,
                )
                for step in window.steps:
                    writer.append(step)
                shard = writer.close()
                reader = RolloutReader(shard["path"], sha256=shard["sha256"], identity=identity)
                try:
                    reader.sequence().validate(identity, versions[0], 0, learning=True)
                finally:
                    reader.close()
                shards.append(shard)
            index = dict(
                schema="poorstar_v3_rollout_index_v1",
                identity_sha256=identity.sha256,
                physical_environment_loops=collector["environment_loops"],
                shards=shards,
                update_groups=[[s["sha256"] for s in shards]],
                collection_plan_sha256=plan.sha256,
                collection_id=plan.collection_id,
                stage_report_path=str((output / "stage_report.json").resolve()),
            )
            check_budget()
            write_document(output / "rollout_index.json", index, immutable=True)
            artifacts["rollout_index"] = artifact(output / "rollout_index.json")
        status = "COMPLETE" if collector["terminated"] else "TRUNCATED"
    except BaseException as caught:
        deadline.disarm()
        error = dict(error_type=type(caught).__name__, error=str(caught))
        if backend is not None:
            try:
                backend.close()
            except BaseException as close_error:
                error["close_error"] = dict(
                    error_type=type(close_error).__name__, error=str(close_error)
                )
            closure = _closed(backend, fixture)
        if collector is not None and "collector_report" not in artifacts:
            write_document(output / "collector_report.json", collector, immutable=True)
            artifacts["collector_report"] = artifact(output / "collector_report.json")
        if manifest is not None:
            try:
                source_post = source_integrity(tree, manifest)
            except BaseException as post_error:
                source_post = dict(problems=["SOURCE_POSTFLIGHT_UNRESOLVED"], error=str(post_error))
        failure = dict(
            schema="poorstar_v3_stage_actor_failure_v1",
            plan_sha256=plan.sha256,
            **error,
            observed_environment_loops=None
            if collector is None
            else collector["environment_loops"],
            owned_process_results=[]
            if backend is None
            else getattr(backend, "owned_process_results", None),
            all_owned_sessions_closed=closure,
        )
        write_document(output / "failure.json", failure, immutable=True)
        artifacts["failure"] = artifact(output / "failure.json")
    finally:
        deadline.disarm()
    elapsed = math.ceil((time.monotonic() - started) * 1000)
    if elapsed > plan.maximum_compute_millis:
        status = "ERROR"
        error = dict(
            error_type="CollectionComputeOverspend",
            error="actual wall time exceeded frozen collection cap",
            previous_error=error,
        )
        write_document(
            output / "compute_overspend.json",
            dict(
                plan_sha256=plan.sha256,
                compute_millis=elapsed,
                maximum_compute_millis=plan.maximum_compute_millis,
            ),
            immutable=True,
        )
        artifacts["compute_overspend"] = artifact(output / "compute_overspend.json")
    report = dict(
        schema="poorstar_v3_stage_actor_collection_report_v1",
        purpose=identity.purpose,
        status=status,
        provenance=plan.provenance,
        stage=plan.phase,
        identity=asdict(identity),
        identity_sha256=identity.sha256,
        plan_sha256=plan.sha256,
        execution_config_sha256=plan.sha256,
        binding=dict(
            map_name=plan.map_name,
            seed=plan.seed,
            collection_id=plan.collection_id,
            maximum_loops=plan.maximum_loops,
            maximum_compute_millis=plan.maximum_compute_millis,
            parameter_budget=plan.parameter_budget,
            execution=plan.execution,
        ),
        actors=actors,
        source_preflight=source_pre,
        source_postflight=source_post,
        launch_preflight=launch_preflight,
        artifacts=artifacts,
        environment_loops=None if collector is None else collector["environment_loops"],
        compute_millis=elapsed,
        owned_process_results=[]
        if backend is None
        else getattr(backend, "owned_process_results", None),
        all_owned_sessions_closed=closure,
        engine_launch_attempted=engine_attempted,
        fixture_scope=FIXTURE_SCOPE if fixture else None,
        actual_engine_frames=0
        if fixture or not engine_attempted
        else None
        if collector is None
        else collector["environment_loops"],
        error=error,
        optimizer_created=False,
        formal_bc_started=False,
        rl_optimizer_started=False,
        selfplay_started=False,
        league_started=False,
        capability_evaluated=False,
        pipeline_stage_unlocked=False,
    )
    write_document(output / "stage_report.json", report, immutable=True)
    return report
