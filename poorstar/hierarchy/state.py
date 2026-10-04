"""Preference, persistent Intent and typed Task control state."""

import hashlib
import json
import math
from dataclasses import asdict, dataclass, replace

from poorstar.hierarchy.goals import GOAL_KINDS, TaskGoal

PREFERENCES = (
    "economy",
    "expansion",
    "technology",
    "army",
    "defense",
    "attack",
    "harassment",
    "scouting",
    "recovery",
)
TASK_TYPES = (
    "economy",
    "production",
    "technology",
    "expansion",
    "movement",
    "rally",
    "scouting",
    "harassment",
    "defense",
    "recovery",
)
FEEDBACK_STATUSES = ("dispatched", "accepted", "confirmed", "rejected", "timeout")
TASK_STATUSES = (
    "active",
    "dispatched",
    "accepted",
    "confirmed",
    "rejected",
    "timeout",
    "terminated",
)


def _integer(value, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError("nonnegative integer state clock/count required")


@dataclass(frozen=True)
class HierarchyConfig:
    arm: str = "flat"
    preference_names: tuple[str, ...] = PREFERENCES
    preference_levels: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)
    manager_period: int = 224
    task_period: int = 32
    commitment_horizons: tuple[int, ...] = (224, 448, 896)
    task_horizons: tuple[int, ...] = (32, 112, 224)
    task_slots: int = 4
    task_types: tuple[str, ...] = TASK_TYPES
    target_regions: tuple[int, int] = (4, 4)
    use_feedback: bool = True
    persistence: bool = True
    manager_mode: str = "learned"
    memory_mode: str = "recurrent"
    commitment_mode: str = "minimum_hold"
    fixed_commitment_horizon: int = 448
    confidence_event: str = "parent_task_confirmed_before_commitment_end"
    grounded_tasks: bool = False
    goal_kinds: tuple[str, ...] = GOAL_KINDS
    goal_quantities: tuple[int, ...] = (1, 2, 4, 8)
    maximum_task_units: int = 63
    common_receipts_mode: str = "legacy"
    schema_version: str = "poorstar_v3_intent_task_runtime_v1"

    def __post_init__(self):
        if self.arm not in ("flat", "direct_task", "persistent_intent"):
            raise ValueError("declared hierarchy arm required")
        if self.manager_mode not in ("learned", "external"):
            raise ValueError("explicit learned/external manager mode required")
        if self.memory_mode not in ("recurrent", "feedforward"):
            raise ValueError("declared recurrent/feedforward decision memory required")
        if self.memory_mode == "feedforward" and self.arm != "flat":
            raise ValueError("feedforward comparator is the flat foundation arm")
        if self.commitment_mode not in ("minimum_hold", "fixed_dwell"):
            raise ValueError("declared minimum-hold/fixed-dwell commitment mode required")
        _integer(self.fixed_commitment_horizon, 1)
        if self.commitment_mode == "fixed_dwell" and (
            self.arm != "persistent_intent" or not self.persistence
        ):
            raise ValueError("fixed dwell requires the persistent Intent arm")
        if (
            not self.preference_names
            or len(set(self.preference_names)) != len(self.preference_names)
            or any(not isinstance(n, str) or not n for n in self.preference_names)
        ):
            raise ValueError("unique named preferences required")
        if (
            not self.preference_levels
            or tuple(sorted(set(self.preference_levels))) != self.preference_levels
            or any(not math.isfinite(x) or not 0 <= x <= 1 for x in self.preference_levels)
        ):
            raise ValueError("declared nonnegative bounded preference levels required")
        for n in (self.manager_period, self.task_period, self.task_slots):
            _integer(n, 1)
        if self.manager_period < self.task_period:
            raise ValueError("manager clock must not be faster than task clock")
        for sequence in (self.commitment_horizons, self.task_horizons):
            if not sequence or tuple(sorted(set(sequence))) != sequence:
                raise ValueError("ordered unique clock horizons required")
            for n in sequence:
                _integer(n, 1)
        if not self.task_types or len(set(self.task_types)) != len(self.task_types):
            raise ValueError("declared unique task vocabulary required")
        for n in self.target_regions:
            _integer(n, 1)
        if len(self.target_regions) != 2:
            raise ValueError("declared tactical region grid required")
        if type(self.use_feedback) is not bool or type(self.persistence) is not bool:
            raise ValueError("explicit lifecycle and feedback ablations required")
        if self.confidence_event != "parent_task_confirmed_before_commitment_end":
            raise ValueError("undeclared confidence prediction event")
        if type(self.grounded_tasks) is not bool or (self.grounded_tasks and self.arm == "flat"):
            raise ValueError("explicit typed-goal Task arm required")
        if (
            not self.goal_kinds
            or len(set(self.goal_kinds)) != len(self.goal_kinds)
            or set(self.goal_kinds) - set(GOAL_KINDS)
        ):
            raise ValueError("declared typed-goal vocabulary required")
        if (
            not self.goal_quantities
            or tuple(sorted(set(self.goal_quantities))) != self.goal_quantities
        ):
            raise ValueError("ordered positive goal quantity vocabulary required")
        for n in self.goal_quantities:
            _integer(n, 1)
        _integer(self.maximum_task_units, 1)
        if self.common_receipts_mode not in ("legacy", "shared_v1"):
            raise ValueError("declared common dispatch observation mode required")

    @property
    def sha256(self):
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()


@dataclass(frozen=True)
class IntentState:
    preference: tuple[float, ...]
    commitment_horizon: int
    start_loop: int
    age: int
    confidence: float
    update_or_termination: str
    source: str
    revision: int
    active: bool = True

    def validate(self, config):
        if len(self.preference) != len(config.preference_names) or any(
            not math.isfinite(z) or not 0 <= z <= 1 for z in self.preference
        ):
            raise ValueError("bounded named Preference vector required")
        _integer(self.commitment_horizon, 1)
        _integer(self.start_loop)
        _integer(self.age)
        _integer(self.revision, 1)
        if not math.isfinite(self.confidence) or not 0 <= self.confidence <= 1:
            raise ValueError("bounded confidence prediction required")
        if self.update_or_termination not in ("keep", "update", "terminate", "external"):
            raise ValueError("explicit Intent lifecycle required")
        if self.source not in ("learned", "external") or type(self.active) is not bool:
            raise ValueError("explicit Intent source and activity required")


@dataclass(frozen=True)
class TaskState:
    task_id: str
    slot: int
    task_type: str
    start_loop: int
    horizon: int
    age: int
    resource_priority: float
    target_region: int | None
    parent_intent: int | None
    status: str = "active"
    progress: float = 0.0
    last_feedback_loop: int | None = None
    goal: TaskGoal | None = None
    squad_id: str | None = None
    allocation_size: int = 0

    @property
    def active(self):
        return self.status not in ("confirmed", "rejected", "timeout", "terminated")

    def validate(self, config):
        if not isinstance(self.task_id, str) or not self.task_id:
            raise ValueError("stable task ID required")
        _integer(self.slot)
        if self.slot >= config.task_slots or self.task_type not in config.task_types:
            raise ValueError("task outside declared slots/vocabulary")
        for n in (self.start_loop, self.age):
            _integer(n)
        _integer(self.horizon, 1)
        if self.last_feedback_loop is not None:
            _integer(self.last_feedback_loop)
            if not self.start_loop <= self.last_feedback_loop <= self.start_loop + self.age:
                raise ValueError("task feedback clock outside its visible lifetime")
        if self.parent_intent is not None:
            _integer(self.parent_intent, 1)
        if self.target_region is not None:
            _integer(self.target_region)
            if self.target_region >= math.prod(config.target_regions):
                raise ValueError("task region outside declared grid")
        if (
            self.status not in TASK_STATUSES
            or not math.isfinite(self.resource_priority)
            or not 0 <= self.resource_priority <= 1
            or not math.isfinite(self.progress)
            or not 0 <= self.progress <= 1
        ):
            raise ValueError("invalid task lifecycle/priority/progress")
        _integer(self.allocation_size)
        if self.allocation_size > config.maximum_task_units:
            raise ValueError("unit allocation outside declared task bound")
        if config.grounded_tasks:
            if (
                not isinstance(self.goal, TaskGoal)
                or not isinstance(self.squad_id, str)
                or not self.squad_id
            ):
                raise ValueError("grounded task requires typed goal and explicit squad identity")
            self.goal.validate(config)
            if self.goal.kind == "allocated_region" and not self.allocation_size:
                raise ValueError("allocated-region goal requires a nonempty learned allocation")
        elif self.goal is not None or self.squad_id is not None or self.allocation_size:
            raise ValueError("undeclared goal/allocation in legacy Task arm")


@dataclass(frozen=True)
class ExecutionFeedback:
    event_id: str
    task_id: str
    status: str
    loop: int
    evidence_id: str
    progress: float | None = None

    def validate(self, current_loop):
        if any(
            not isinstance(v, str) or not v for v in (self.event_id, self.task_id, self.evidence_id)
        ):
            raise ValueError("feedback event/task/evidence IDs required")
        _integer(self.loop)
        if self.loop > current_loop:
            raise ValueError("future execution feedback is forbidden")
        if self.status not in (*FEEDBACK_STATUSES, "progress"):
            raise ValueError("unknown execution feedback status")
        if self.progress is not None and (
            not math.isfinite(self.progress) or not 0 <= self.progress <= 1
        ):
            raise ValueError("bounded evidenced task progress required")


@dataclass(frozen=True)
class ExternalOverride:
    preference: tuple[float, ...]
    commitment_horizon: int
    confidence: float
    intervention_id: str

    def validate(self, config):
        if not isinstance(self.intervention_id, str) or not self.intervention_id:
            raise ValueError("auditable external intervention ID required")
        IntentState(
            self.preference,
            self.commitment_horizon,
            0,
            0,
            self.confidence,
            "external",
            "external",
            1,
        ).validate(config)


DISPATCH_STATUSES = ("dispatched", "accepted", "rejected", "effect_unknown")


@dataclass(frozen=True)
class DispatchObservation:
    """Task-independent, actor-visible RPC receipt counts for all policy arms."""

    loop: int
    counts: tuple[int, ...]
    evidence_id: str

    def validate(self, current_loop):
        _integer(self.loop)
        if self.loop > current_loop:
            raise ValueError("future common dispatch observation forbidden")
        if len(self.counts) != len(DISPATCH_STATUSES):
            raise ValueError("common receipt status count schema drift")
        for n in self.counts:
            _integer(n)
        if not isinstance(self.evidence_id, str) or not self.evidence_id:
            raise ValueError("auditable common dispatch receipt evidence required")

    @classmethod
    def from_dict(cls, value):
        if set(value) != set(cls.__dataclass_fields__):
            raise ValueError("common receipt recovery schema drift")
        return cls(value["loop"], tuple(value["counts"]), value["evidence_id"])


@dataclass(frozen=True)
class HierarchyState:
    episode: str
    last_loop: int | None = None
    intent: IntentState | None = None
    tasks: tuple[TaskState, ...] = ()
    last_manager_loop: int | None = None
    last_task_loop: int | None = None
    task_revision: int = 0
    consumed_feedback: tuple[str, ...] = ()
    feedback_watermark_loop: int | None = None

    def validate(self, config):
        if not isinstance(self.episode, str) or not self.episode:
            raise ValueError("explicit episode identity required")
        for n in (self.last_loop, self.last_manager_loop, self.last_task_loop):
            if n is not None:
                _integer(n)
                if self.last_loop is None or n > self.last_loop:
                    raise ValueError("control state contains a future clock")
        _integer(self.task_revision)
        if self.intent is not None:
            self.intent.validate(config)
            if self.last_loop is None or self.intent.start_loop > self.last_loop:
                raise ValueError("future Intent start")
            if self.intent.age != self.last_loop - self.intent.start_loop:
                raise ValueError("Intent age must reflect actual elapsed loops")
        if len(set(self.consumed_feedback)) != len(self.consumed_feedback):
            raise ValueError("duplicate consumed feedback ID")
        if self.feedback_watermark_loop is not None:
            _integer(self.feedback_watermark_loop)
            if self.last_loop is None or self.feedback_watermark_loop > self.last_loop:
                raise ValueError("future feedback watermark")
        elif self.consumed_feedback:
            raise ValueError("feedback deduplication requires an event-clock watermark")
        if len(self.consumed_feedback) > max(64, config.task_slots * len(FEEDBACK_STATUSES)):
            raise ValueError("same-loop feedback denominator exceeds bounded scheduler contract")
        if len({t.slot for t in self.tasks}) != len(self.tasks) or len(
            {t.task_id for t in self.tasks}
        ) != len(self.tasks):
            raise ValueError("duplicate task slot/identity")
        for t in self.tasks:
            t.validate(config)
            if (
                self.last_loop is None
                or t.start_loop > self.last_loop
                or t.age != self.last_loop - t.start_loop
            ):
                raise ValueError("task age must reflect actual elapsed loops")
        if config.arm == "flat" and (self.intent is not None or self.tasks):
            raise ValueError("flat state cannot contain Intent/tasks")
        if config.arm == "direct_task" and self.intent is not None:
            raise ValueError("direct Task state cannot contain Intent")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value, config):
        if set(value) != set(cls.__dataclass_fields__):
            raise ValueError("hierarchy recovery schema drift")
        d = dict(value)
        if d["intent"] is not None:
            q = dict(d["intent"])
            q["preference"] = tuple(q["preference"])
            d["intent"] = IntentState(**q)
        tasks = []
        for t in d["tasks"]:
            q = dict(t)
            if q.get("goal") is not None:
                q["goal"] = TaskGoal.from_dict(q["goal"])
            tasks.append(TaskState(**q))
        d["tasks"] = tuple(tasks)
        d["consumed_feedback"] = tuple(d["consumed_feedback"])
        state = cls(**d)
        state.validate(config)
        return state


def advance_state(state, loop, config, feedback=()):
    state.validate(config)
    _integer(loop)
    if state.last_loop is not None and loop < state.last_loop:
        raise ValueError("control clock cannot move backwards")
    intent = state.intent
    if intent is not None:
        intent = replace(intent, age=loop - intent.start_loop)
    tasks = {t.task_id: replace(t, age=loop - t.start_loop) for t in state.tasks}
    consumed = set(state.consumed_feedback)
    watermark = state.feedback_watermark_loop
    current = []
    for event in feedback:
        if not isinstance(event, ExecutionFeedback):
            raise ValueError("typed evidenced execution feedback required")
        event.validate(loop)
        if watermark is not None and event.loop < watermark:
            raise ValueError("feedback event time precedes the consumed event-clock watermark")
        if watermark is None or event.loop > watermark:
            consumed = set()
            watermark = event.loop
        if event.event_id in consumed:
            raise ValueError("feedback cannot be consumed twice")
        if event.task_id not in tasks:
            raise ValueError("feedback references an unknown task")
        task = tasks[event.task_id]
        if event.loop < task.start_loop:
            raise ValueError("feedback predates its task")
        if task.last_feedback_loop is not None and event.loop < task.last_feedback_loop:
            raise ValueError("execution feedback regresses event time")
        if not task.active:
            raise ValueError("feedback cannot revive a terminal task")
        # Omitted-feedback arm receives no outcome, progress or event-driven
        # scheduling information. IDs alone remain in recovery bookkeeping.
        if config.use_feedback:
            rank = {
                "active": 0,
                "dispatched": 1,
                "accepted": 2,
                "confirmed": 3,
                "rejected": 3,
                "timeout": 3,
                "progress": 0,
            }
            if event.status == "progress" and event.progress is None:
                raise ValueError("progress receipt requires visible measured progress")
            if (
                event.status != "progress"
                and rank[event.status] < rank[task.status]
                and event.loop == task.last_feedback_loop
            ):
                raise ValueError("execution feedback regresses lifecycle")
            progress = task.progress if event.progress is None else event.progress
            if progress < task.progress:
                raise ValueError("confirmed task progress cannot regress")
            status = (
                event.status
                if event.status != "progress" and rank[event.status] >= rank[task.status]
                else task.status
            )
            tasks[event.task_id] = replace(
                task, status=status, progress=progress, last_feedback_loop=event.loop
            )
            current.append(event)
        consumed.add(event.event_id)
    # A declared task duration is an executor scheduling deadline, not evidence
    # that its gameplay goal succeeded or failed. Expire after processing already
    # visible receipts; never fabricate confirmation. This clock rule applies to
    # both feedback arms because their known task durations are identical.
    for task_id, task in tuple(tasks.items()):
        if task.active and task.age >= task.horizon:
            deadline = task.start_loop + task.horizon
            tasks[task_id] = replace(task, status="timeout")
            if config.use_feedback:
                current.append(
                    ExecutionFeedback(
                        f"{task_id}:clock_timeout:{deadline}",
                        task_id,
                        "timeout",
                        deadline,
                        "TASK_DECLARED_CLOCK_DEADLINE",
                    )
                )
    result = replace(
        state,
        last_loop=loop,
        intent=intent,
        tasks=tuple(sorted(tasks.values(), key=lambda t: t.slot)),
        consumed_feedback=tuple(sorted(consumed)),
        feedback_watermark_loop=watermark,
    )
    result.validate(config)
    return result, tuple(current)
