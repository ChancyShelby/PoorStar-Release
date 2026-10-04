"""Observe Task progress and confirmation using player-visible evidence."""

import hashlib
import json
import math
import secrets
from dataclasses import asdict, dataclass

import numpy as np

from poorstar.actions.frame import ActionFrame
from poorstar.hierarchy.goals import TaskGoal
from poorstar.hierarchy.state import ExecutionFeedback


def _digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def entity_region(entity, observation, grid):
    values, present = entity["values"], entity["present"]
    if not present.get("world_x") or not present.get("world_y"):
        return None
    x, y = values["world_x"], values["world_y"]
    width, height = observation["map_size"]
    if not (math.isfinite(x) and math.isfinite(y) and 0 <= x < width and 0 <= y < height):
        return None
    return min(grid[0] - 1, int(x * grid[0] / width)) + grid[0] * min(
        grid[1] - 1, int(y * grid[1] / height)
    )


def region_visibility(observation, region, grid):
    """Fraction of a native raster tile visible; NO world-region conversion."""
    if not observation["spatial_present"].get("visibility"):
        return None
    from poorstar.engine.observation import NATIVE_RASTER_FRAME

    if observation.get("raster_frame") != NATIVE_RASTER_FRAME:
        raise ValueError("task visibility requires the declared C3 native raster frame")
    array = observation["spatial_arrays"].get("visibility")
    width, height = observation["map_size"]
    if not isinstance(array, np.ndarray) or array.shape != (height, width):
        raise ValueError("current visibility/map shape mismatch")
    gx, gy = region % grid[0], region // grid[0]
    x0, x1 = math.ceil(gx * width / grid[0]), math.ceil((gx + 1) * width / grid[0])
    y0, y1 = math.ceil(gy * height / grid[1]), math.ceil((gy + 1) * height / grid[1])
    section = array[y0:y1, x0:x1]
    return None if not section.size else float(np.count_nonzero(section == 2) / section.size)


def completed_count(observation, unit_type):
    count = 0
    for entity in observation["entities"]:
        v, p = entity["values"], entity["present"]
        if not p.get("alliance") or not p.get("display_type") or not p.get("unit_type"):
            return None
        if v["alliance"] == 1 and v["display_type"] == 1 and v["unit_type"] == unit_type:
            if not p.get("build_progress"):
                return None
            count += v["build_progress"] >= 1.0
    return int(count)


@dataclass(frozen=True)
class TaskObservation:
    task_id: str
    loop: int
    outcome: str  # progress / confirmed / censored
    progress: float | None
    evidence_id: str
    reason: str

    def to_feedback(self):
        if self.outcome == "censored":
            return None
        return ExecutionFeedback(
            f"{self.task_id}:visible:{self.loop}:{self.evidence_id}",
            self.task_id,
            self.outcome,
            self.loop,
            self.evidence_id,
            self.progress,
        )


class TaskEffectObserver:
    schema_version = "poorstar_v3_visible_task_observer_v1"

    def __init__(self, config, *, episode, player_id):
        if not config.grounded_tasks:
            raise ValueError("task observer requires declared grounded Task configuration")
        if (
            not isinstance(episode, str)
            or not episode
            or type(player_id) is not int
            or player_id < 1
        ):
            raise ValueError("observer requires exact episode/player identity")
        self.config = config
        self.episode = episode
        self.player_id = player_id
        self._salt = secrets.token_hex(32)
        self._tasks = {}
        self.last_loop = None
        self.map_sha256 = None
        self.catalog_sha256 = None

    def _frame(self, frame):
        if (
            not isinstance(frame, ActionFrame)
            or frame.key.episode != self.episode
            or frame.key.player_id != self.player_id
        ):
            raise ValueError("task observer requires same controlled player ActionFrame")
        if self.last_loop is not None and frame.key.loop < self.last_loop:
            raise ValueError("task observer cannot inspect an older frame")
        if self.map_sha256 is not None and frame.key.map_sha256 != self.map_sha256:
            raise ValueError("task observer map identity drift")
        if self.catalog_sha256 is not None and frame.key.catalog_sha256 != self.catalog_sha256:
            raise ValueError("task observer engine catalog identity drift")
        obs = frame.policy_observation()
        if obs["observation_loop"] != frame.key.loop or len(obs["entities"]) != len(
            frame.slot_tags
        ):
            raise ValueError("task observer frame/tensor alignment mismatch")
        handles = {
            i: hashlib.sha256((self._salt + ":" + str(tag)).encode()).hexdigest()
            for i, tag in enumerate(frame.slot_tags)
            if i in frame.own_visible_slots
        }
        self.map_sha256 = frame.key.map_sha256
        self.last_loop = frame.key.loop
        self.catalog_sha256 = frame.key.catalog_sha256
        return obs, handles

    def synchronize(self, tasks):
        """Forget superseded/terminal IDs; a late event cannot bind a reused slot."""
        current = {t.task_id for t in tasks if t.active}
        self._tasks = {k: v for k, v in self._tasks.items() if k in current}

    def current_allocations(self, frame):
        """Rebind private members to this current frame; invisible members omitted."""
        _, handles = self._frame(frame)
        slots = {handle: slot for slot, handle in handles.items()}
        return {
            task_id: tuple(slots[h] for h in record["members"] if h in slots)
            for task_id, record in self._tasks.items()
        }

    def allocation_audit(self, frame, task_id, source_slots):
        """Diagnose actual emitted members; never mask, rewrite or repair them.

        Assignment is learned conditioning and ownership, not an extra T2 action
        legality constraint. A decoder mismatch remains visible as a non-claim.
        """
        allocations = self.current_allocations(frame)
        sources = tuple(source_slots)
        if any(type(s) is not int or s not in frame.own_visible_slots for s in sources):
            raise ValueError("allocation audit requires actual current dispatched source slots")
        if task_id is None:
            return dict(
                status="EXPLICIT_UNASSIGNED_BUNDLE", source_count=len(set(sources)), matched=0
            )
        if task_id not in allocations:
            raise ValueError("bundle cannot be attributed to an unknown task/squad")
        selected = set(allocations[task_id])
        other = {s for key, members in allocations.items() if key != task_id for s in members}
        sources = set(sources)
        return dict(
            status="MATCHED_DECLARED_ALLOCATION"
            if sources <= selected
            else "OTHER_TASK_MEMBER_EMITTED"
            if sources & other
            else "UNASSIGNED_MEMBER_EMITTED",
            source_count=len(sources),
            matched=len(sources & selected),
        )

    def register(self, frame, task, unit_slots):
        """Bind a newly declared goal and explicit learned frame-local allocation."""
        task.validate(self.config)
        if not task.active or task.start_loop != frame.key.loop:
            raise ValueError("register task on its exact current creation frame")
        if task.task_id in self._tasks:
            raise ValueError("task goal cannot be rebased or registered twice")
        if len(self._tasks) >= self.config.task_slots:
            raise ValueError("observer active task count exceeds declared slots")
        obs, handles = self._frame(frame)
        slots = tuple(unit_slots)
        if (
            len(set(slots)) != len(slots)
            or any(type(s) is not int or s not in handles for s in slots)
            or len(slots) != task.allocation_size
        ):
            raise ValueError("allocation must reference exact current own-visible entity slots")
        members = tuple(handles[s] for s in slots)
        reserved = {h for record in self._tasks.values() for h in record["members"]}
        if reserved.intersection(members):
            raise ValueError("a visible unit cannot belong to two active task squads")
        goal = task.goal
        baseline = (
            completed_count(obs, goal.subject) if goal.kind == "own_completed_unit_delta" else 0
        )
        record = dict(
            goal=goal.to_dict(),
            members=members,
            start_loop=task.start_loop,
            horizon=task.horizon,
            baseline=baseline,
            progress=0.0,
            confirmed=False,
            initial_satisfied=False,
            last_evidence=None,
        )
        satisfaction, _, _ = self._measure(record, obs, handles)
        record["initial_satisfied"] = satisfaction is True
        self._tasks[task.task_id] = record

    def _measure(self, record, obs, handles):
        goal = TaskGoal.from_dict(record["goal"])
        if goal.kind == "own_completed_unit_delta":
            count = completed_count(obs, goal.subject)
            if count is None or record["baseline"] is None:
                return None, None, "OWN_COMPLETION_FIELD_UNAVAILABLE"
            gain = max(0, count - record["baseline"])
            return (
                gain >= goal.quantity,
                min(1.0, gain / goal.quantity),
                ("own_count", count, record["baseline"]),
            )
        if goal.kind == "upgrade_present":
            if obs["upgrades"] is None:
                return None, None, "CURRENT_UPGRADES_UNAVAILABLE"
            present = goal.subject in obs["upgrades"]
            return present, float(present), ("upgrade_present", present)
        if goal.kind == "region_visible":
            fraction = region_visibility(obs, goal.region, self.config.target_regions)
            if fraction is None:
                return None, None, "CURRENT_VISIBILITY_UNAVAILABLE"
            return fraction == 1.0, fraction, ("visible_fraction", fraction)
        by_handle = {handles[e["slot"]]: e for e in obs["entities"] if e["slot"] in handles}
        if not record["members"] or any(h not in by_handle for h in record["members"]):
            return None, None, "ALLOCATED_MEMBER_NOT_CURRENTLY_VISIBLE"
        regions = [
            entity_region(by_handle[h], obs, self.config.target_regions) for h in record["members"]
        ]
        if any(r is None for r in regions):
            return None, None, "ALLOCATED_MEMBER_POSITION_UNAVAILABLE"
        fraction = sum(r == goal.region for r in regions) / len(regions)
        return fraction == 1.0, fraction, ("allocated_region_fraction", fraction)

    def observe(self, frame):
        """Observe only current player-visible state; no future/result/receipt argument."""
        obs, handles = self._frame(frame)
        result = []
        for task_id, record in sorted(self._tasks.items()):
            if record["confirmed"]:
                continue
            if frame.key.loop >= record["start_loop"] + record["horizon"]:
                measured, progress, fact = None, None, "DECLARED_TASK_WINDOW_CENSORED"
            elif record["initial_satisfied"]:
                measured, progress, fact = None, None, "GOAL_ALREADY_SATISFIED_AT_REGISTRATION"
            else:
                measured, progress, fact = self._measure(record, obs, handles)
            evidence = _digest(
                dict(
                    schema=self.schema_version,
                    task=task_id,
                    goal=record["goal"],
                    frame=asdict(frame.key),
                    fact=fact,
                    start=record["start_loop"],
                    horizon=record["horizon"],
                )
            )
            if measured is None:
                outcome, reported, reason = "censored", None, fact
            elif measured and frame.key.loop > record["start_loop"]:
                outcome, reported, reason = "confirmed", 1.0, "OBSERVED_DECLARED_GOAL_SATISFACTION"
                record["confirmed"] = True
            elif progress > record["progress"]:
                outcome, reported, reason = "progress", progress, "OBSERVED_VISIBLE_GOAL_PROGRESS"
            else:
                continue
            if record["last_evidence"] == evidence:
                continue
            record["last_evidence"] = evidence
            if reported is not None:
                record["progress"] = max(record["progress"], reported)
            result.append(
                TaskObservation(task_id, frame.key.loop, outcome, reported, evidence, reason)
            )
        return tuple(result)

    def state_dict(self):
        """Private environment-boundary recovery, not an actor/model input or report."""
        return dict(
            schema_version=self.schema_version,
            config_sha256=self.config.sha256,
            episode=self.episode,
            player_id=self.player_id,
            salt=self._salt,
            last_loop=self.last_loop,
            map_sha256=self.map_sha256,
            catalog_sha256=self.catalog_sha256,
            tasks=json.loads(json.dumps(self._tasks)),
        )

    def load_state_dict(self, value):
        if set(value) != set(self.state_dict()) or (
            value["schema_version"] != self.schema_version
            or value["config_sha256"] != self.config.sha256
            or value["episode"] != self.episode
            or value["player_id"] != self.player_id
        ):
            raise ValueError("task observer recovery identity/schema mismatch")
        if (
            not isinstance(value["salt"], str)
            or len(value["salt"]) != 64
            or len(value["tasks"]) > self.config.task_slots
        ):
            raise ValueError("bounded private allocation recovery required")
        try:
            bytes.fromhex(value["salt"])
            if value["last_loop"] is not None and (
                type(value["last_loop"]) is not int or value["last_loop"] < 0
            ):
                raise ValueError()
            members = []
            for key, record in value["tasks"].items():
                if (
                    not isinstance(key, str)
                    or not key
                    or set(record)
                    != {
                        "goal",
                        "members",
                        "start_loop",
                        "horizon",
                        "baseline",
                        "progress",
                        "confirmed",
                        "initial_satisfied",
                        "last_evidence",
                    }
                ):
                    raise ValueError()
                goal = TaskGoal.from_dict(record["goal"])
                goal.validate(self.config)
                if (
                    type(record["start_loop"]) is not int
                    or record["start_loop"] < 0
                    or type(record["horizon"]) is not int
                    or record["horizon"] < 1
                    or value["last_loop"] is None
                    or record["start_loop"] > value["last_loop"]
                    or type(record["confirmed"]) is not bool
                    or type(record["initial_satisfied"]) is not bool
                    or not math.isfinite(record["progress"])
                    or not 0 <= record["progress"] <= 1
                    or len(record["members"]) > self.config.maximum_task_units
                    or any(not isinstance(h, str) or len(h) != 64 for h in record["members"])
                ):
                    raise ValueError()
                if (
                    goal.kind == "own_completed_unit_delta"
                    and record["baseline"] is not None
                    and (type(record["baseline"]) is not int or record["baseline"] < 0)
                ):
                    raise ValueError()
                members.extend(record["members"])
            if len(set(members)) != len(members):
                raise ValueError()
        except (TypeError, KeyError, ValueError) as error:
            raise ValueError("invalid causal task observer recovery") from error
        self._salt = value["salt"]
        self._tasks = json.loads(json.dumps(value["tasks"]))
        self.last_loop = value["last_loop"]
        self.map_sha256 = value["map_sha256"]
        self.catalog_sha256 = value["catalog_sha256"]
