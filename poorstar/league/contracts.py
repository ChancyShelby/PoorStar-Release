"""League: contracts."""

import math
from dataclasses import asdict, dataclass, field

from poorstar.training.contracts import TrainingIdentity, digest, is_sha


def positive(value, name):
    if type(value) is not int or value < 1:
        raise ValueError("positive integer required: " + name)


@dataclass(frozen=True)
class LeagueBudget:
    training_game_loops: int
    evaluation_game_loops: int
    actor_compute_millis: int
    evaluation_compute_millis: int
    learner_compute_millis: int
    maximum_job_loops: int
    maximum_actor_compute_millis: int
    maximum_learner_compute_millis: int
    maximum_updates: int
    population_limit: int
    maximum_parameters: int
    maximum_trainable_parameters: int
    hardware_sha256: str
    learner_contract_sha256: str
    actor_protocol_sha256: str
    compute_unit: str = "single_reserved_worker_wall_millisecond"

    def __post_init__(self):
        for name, value in asdict(self).items():
            if name not in (
                "hardware_sha256",
                "learner_contract_sha256",
                "actor_protocol_sha256",
                "compute_unit",
            ):
                positive(value, name)
        if (
            not all(
                is_sha(getattr(self, n))
                for n in ("hardware_sha256", "learner_contract_sha256", "actor_protocol_sha256")
            )
            or self.compute_unit != "single_reserved_worker_wall_millisecond"
        ):
            raise ValueError("frozen hardware and explicitly charged compute unit required")
        if self.maximum_trainable_parameters > self.maximum_parameters:
            raise ValueError("trainable parameter budget exceeds total parameters")


@dataclass(frozen=True)
class PromotionRules:
    minimum_anchor_win_rate: float
    minimum_history_win_rate: float
    anchor_noninferiority_margin: float
    wilson_z: float
    minimum_games_per_cell: int

    def __post_init__(self):
        for name in (
            "minimum_anchor_win_rate",
            "minimum_history_win_rate",
            "anchor_noninferiority_margin",
        ):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("explicit bounded promotion threshold required: " + name)
        if not math.isfinite(self.wilson_z) or self.wilson_z <= 0:
            raise ValueError("positive frozen confidence coefficient required")
        positive(self.minimum_games_per_cell, "minimum_games_per_cell")


@dataclass(frozen=True)
class LeagueMember:
    name: str
    snapshot_id: str
    identity: dict
    parameter_count: int
    trainable_parameter_count: int

    def __post_init__(self):
        if not self.name or not is_sha(self.snapshot_id):
            raise ValueError("named immutable population member required")
        TrainingIdentity(**self.identity)
        positive(self.parameter_count, "parameter_count")
        positive(self.trainable_parameter_count, "trainable_parameter_count")
        if self.trainable_parameter_count > self.parameter_count:
            raise ValueError("trainable member count exceeds total parameters")


@dataclass(frozen=True)
class LeagueRunPlan:
    kind: str
    seed: int
    members: tuple[LeagueMember, ...]
    anchors: tuple[str, ...]
    historical: tuple[str, ...]
    maps: tuple[str, ...]
    evaluation_seeds: tuple[int, ...]
    rounds: int
    training_games_per_member: int
    updates_per_member: int
    maximum_infrastructure_attempts: int
    budget: LeagueBudget
    promotion: PromotionRules
    actor_execution: dict = field(default_factory=dict)
    learner_execution: dict = field(default_factory=dict)
    schema: str = "poorstar_v3_league_run_plan_v1"

    def __post_init__(self):
        if (
            self.kind not in ("ordinary", "intent_specialized")
            or type(self.seed) is not int
            or self.seed < 0
        ):
            raise ValueError("declared league kind/seed required")
        for name in (
            "rounds",
            "training_games_per_member",
            "updates_per_member",
            "maximum_infrastructure_attempts",
        ):
            positive(getattr(self, name), name)
        if self.updates_per_member != 1:
            raise ValueError(
                "one fresh frozen behavior group updates once per member/round; add rounds for further updates"
            )
        if not self.members or len({m.name for m in self.members}) != len(self.members):
            raise ValueError("unique nonempty population members required")
        if not self.anchors or not self.historical or set(self.anchors) & set(self.historical):
            raise ValueError("separate fixed anchors and historical opponents required")
        for values in (self.anchors, self.historical):
            if len(set(values)) != len(values) or not all(is_sha(v) for v in values):
                raise ValueError("unique content-addressed fixed opponent pool required")
        if not self.maps or len(set(self.maps)) != len(self.maps) or not all(self.maps):
            raise ValueError("frozen nonempty evaluation maps required")
        if (
            not self.evaluation_seeds
            or len(set(self.evaluation_seeds)) != len(self.evaluation_seeds)
            or any(type(s) is not int or s < 0 for s in self.evaluation_seeds)
        ):
            raise ValueError("unique frozen evaluator seeds required")
        if len(self.evaluation_seeds) < self.promotion.minimum_games_per_cell:
            raise ValueError("evaluation matrix below frozen per-cell sample count")
        # History is retained. The budget counts every registered version, not
        # just simultaneous actors, so rotating snapshots cannot hide compute.
        maximum_snapshots = len(set(self.anchors + self.historical)) + len(self.members) * (
            self.rounds + 1
        )
        if maximum_snapshots > self.budget.population_limit:
            raise ValueError("all immutable versions exceed registered population budget")
        if any(
            m.parameter_count > self.budget.maximum_parameters
            or m.trainable_parameter_count > self.budget.maximum_trainable_parameters
            for m in self.members
        ):
            raise ValueError("declared member parameter budget exceeded")
        updates = len(self.members) * self.rounds * self.updates_per_member
        if updates > self.budget.maximum_updates:
            raise ValueError("matched learner update budget exceeded")

    @property
    def sha256(self):
        return digest(asdict(self))

    @property
    def matched_budget_sha256(self):
        # Snapshot content, Intent settings and arm labels differ by treatment.
        # Numeric budgets, map/seed panel, rules and member parameter counts do not.
        return digest(
            dict(
                budget=asdict(self.budget),
                maps=self.maps,
                evaluation_seeds=self.evaluation_seeds,
                rounds=self.rounds,
                training_games_per_member=self.training_games_per_member,
                updates_per_member=self.updates_per_member,
                maximum_infrastructure_attempts=self.maximum_infrastructure_attempts,
                member_parameters=sorted(
                    (m.parameter_count, m.trainable_parameter_count) for m in self.members
                ),
                anchors=self.anchors,
                historical=self.historical,
                promotion=asdict(self.promotion),
            )
        )

    @classmethod
    def from_dict(cls, value):
        value = dict(value)
        value["members"] = tuple(LeagueMember(**m) for m in value["members"])
        for name in ("anchors", "historical", "maps", "evaluation_seeds"):
            value[name] = tuple(value[name])
        value["budget"] = LeagueBudget(**value["budget"])
        value["promotion"] = PromotionRules(**value["promotion"])
        return cls(**value)


@dataclass(frozen=True)
class WorkerRequest:
    kind: str
    run_plan_sha256: str
    identity: dict
    operation_id: str
    attempt_id: str
    claim_token: str
    owner: dict
    output_dir: str
    maximum_compute_millis: int
    payload: dict
    execution: dict
    schema: str = "poorstar_v3_league_worker_request_v1"

    def __post_init__(self):
        if (
            self.kind not in ("actor", "learner")
            or self.schema != "poorstar_v3_league_worker_request_v1"
        ):
            raise ValueError("explicit worker request kind/schema required")
        if not all(
            is_sha(getattr(self, n))
            for n in ("run_plan_sha256", "operation_id", "attempt_id", "claim_token")
        ):
            raise ValueError("content-bound operation/attempt/token required")
        TrainingIdentity(**self.identity)
        if (
            not self.output_dir
            or not self.owner.get("host")
            or type(self.owner.get("pid")) is not int
            or not self.owner.get("starttime")
        ):
            raise ValueError("registered local process owner and external output required")
        positive(self.maximum_compute_millis, "maximum_compute_millis")
        if self.kind == "actor":
            job = self.payload.get("job", {})
            if job.get("provenance") not in ("learner", "evaluator") or not is_sha(
                job.get("job_id")
            ):
                raise ValueError("exact owned actor job/provenance required")
        elif self.payload.get("provenance") != "learner" or not self.payload.get("rollout_indices"):
            raise ValueError("learner requires training-only rollout input")

    @property
    def sha256(self):
        return digest(asdict(self))


@dataclass(frozen=True)
class WorkerResult:
    request_sha256: str
    operation_id: str
    attempt_id: str
    claim_token: str
    kind: str
    status: str
    compute_millis: int
    environment_loops: int | None
    owned_sessions_closed: bool
    artifacts: dict
    schema: str = "poorstar_v3_league_worker_result_v1"

    def validate(self, request):
        if self.schema != "poorstar_v3_league_worker_result_v1" or self.kind != request.kind:
            raise ValueError("worker result kind/schema mismatch")
        if (
            any(
                getattr(self, n) != getattr(request, n)
                for n in ("operation_id", "attempt_id", "claim_token")
            )
            or self.request_sha256 != request.sha256
        ):
            raise ValueError("worker result request/attempt ownership mismatch")
        if (
            self.status not in ("COMPLETE", "TRUNCATED", "ERROR")
            or type(self.compute_millis) is not int
            or self.compute_millis < 0
        ):
            raise ValueError("explicit worker outcome and measured compute required")
        if self.owned_sessions_closed is not True:
            raise ValueError("worker owned sessions not proven closed")
        if self.environment_loops is not None and (
            type(self.environment_loops) is not int or self.environment_loops < 0
        ):
            raise ValueError("actual nonnegative environment frames required")
        if request.kind == "learner" and self.environment_loops != 0:
            raise ValueError("learner cannot double-count actor environment frames")
        if (
            request.kind == "actor"
            and request.payload["job"]["provenance"] == "evaluator"
            and "rollout_index" in self.artifacts
        ):
            raise ValueError("independent evaluator trajectories cannot train policy")
        for artifact in self.artifacts.values():
            if (
                not isinstance(artifact, dict)
                or not artifact.get("path")
                or not is_sha(artifact.get("sha256"))
            ):
                raise ValueError("immutable worker artifacts required")


def assert_matched_leagues(ordinary, specialized):
    if (
        ordinary.kind != "ordinary"
        or specialized.kind != "intent_specialized"
        or ordinary.matched_budget_sha256 != specialized.matched_budget_sha256
    ):
        raise ValueError(
            "E6 requires matched frames/compute/parameters/population/evaluation rules"
        )
    return ordinary.matched_budget_sha256
