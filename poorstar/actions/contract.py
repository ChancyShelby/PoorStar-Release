"""Autoregressive action tokens, candidate domains and raw command targets."""

import math
from dataclasses import dataclass, field

from poorstar.actions.frame import ActionFrame, FrameKey

SELECT_END = "SELECT_END"
BUNDLE_NEXT = "BUNDLE_NEXT"
BUNDLE_STOP = "BUNDLE_STOP"
TARGET_MODES = {
    1: ("none",),
    2: ("position",),
    3: ("entity",),
    4: ("position", "entity"),
    5: ("position", "none"),
}

TARGET_ISSUES = frozenset(
    (
        "EMPTY_SOURCE_SET",
        "DUPLICATE_SOURCE_REFERENCE",
        "UNOBSERVABLE_SOURCE_REFERENCE",
        "SOURCE_NOT_CURRENT_OWN_VISIBLE",
        "UNOBSERVABLE_ENTITY_TARGET",
        "NONFINITE_WORLD_POINT",
        "WORLD_POINT_OUTSIDE_MAP",
    )
)


class TargetDomainRejection(ValueError):
    """Explicit expert-label rejection, never an internal validation failure.

    Audit callers may catch this type only around teacher forcing. The same exception
    from a sampler is a hard failure because its chosen token violated its own domain.
    """

    def __init__(self, code, message, *, issues=()):
        super().__init__(message)
        self.code = code
        self.issues = tuple(issues)


@dataclass(frozen=True)
class Token:
    head: str
    value: object


@dataclass(frozen=True)
class RawCommandTarget:
    frame_key: FrameKey
    ability: int
    source_slots: tuple[int | None, ...]
    target_mode: str
    target_slot: int | None
    world_position: tuple[float, float] | None
    queue: bool | None
    protocol_game_loop: int | None
    original_wire: bytes = field(repr=False)
    issues: tuple[str, ...] = ()

    @classmethod
    def from_protocol(cls, action, frame: ActionFrame):
        known = type(action)()
        known.CopyFrom(action)
        known.DiscardUnknownFields()
        if known.SerializeToString() != action.SerializeToString():
            raise ValueError("unknown protocol fields cannot be silently discarded")
        if not action.HasField("action_raw") or not action.action_raw.HasField("unit_command"):
            raise ValueError("not a RAW unit command; retain in the other-action inventory")
        if set(f.name for f, _ in action.ListFields()) - {"action_raw", "game_loop"}:
            raise ValueError("action has undeclared parallel interface fields")
        if set(f.name for f, _ in action.action_raw.ListFields()) != {"unit_command"}:
            raise ValueError("RAW action has parallel commands")
        c = action.action_raw.unit_command
        if not c.HasField("ability_id") or c.ability_id <= 0:
            raise ValueError("positive explicit RAW ability required")
        slots = {tag: i for i, tag in enumerate(frame.slot_tags)}
        src = tuple(slots.get(int(t)) for t in c.unit_tags)
        issues = []
        if not src:
            issues.append("EMPTY_SOURCE_SET")
        if len(set(c.unit_tags)) != len(c.unit_tags):
            issues.append("DUPLICATE_SOURCE_REFERENCE")
        if any(s is None for s in src):
            issues.append("UNOBSERVABLE_SOURCE_REFERENCE")
        if any(s is not None and s not in frame.own_visible_slots for s in src):
            issues.append("SOURCE_NOT_CURRENT_OWN_VISIBLE")
        kind = c.WhichOneof("target")
        mode = {"target_world_space_pos": "position", "target_unit_tag": "entity", None: "none"}[
            kind
        ]
        target = slots.get(int(c.target_unit_tag)) if mode == "entity" else None
        pos = None
        if mode == "entity" and target is None:
            issues.append("UNOBSERVABLE_ENTITY_TARGET")
        if mode == "position":
            p = c.target_world_space_pos
            if not p.HasField("x") or not p.HasField("y"):
                raise ValueError("world point coordinate presence required")
            pos = (float(p.x), float(p.y))
            if not all(math.isfinite(v) for v in pos):
                issues.append("NONFINITE_WORLD_POINT")
            elif not 0 <= pos[0] < frame.map_size[0] or not 0 <= pos[1] < frame.map_size[1]:
                issues.append("WORLD_POINT_OUTSIDE_MAP")
        return cls(
            frame.key,
            int(c.ability_id),
            src,
            mode,
            target,
            pos,
            bool(c.queue_command) if c.HasField("queue_command") else None,
            int(action.game_loop) if action.HasField("game_loop") else None,
            action.SerializeToString(),
            tuple(issues),
        )

    def tokens(self):
        if self.issues:
            if not set(self.issues).issubset(TARGET_ISSUES):
                raise ValueError("unknown complete-command issue classification")
            raise TargetDomainRejection(
                "UNREPRESENTABLE_ARGUMENT",
                "unrepresentable complete command: " + ",".join(self.issues),
                issues=self.issues,
            )
        tokens = [Token("ability", self.ability)]
        tokens.extend(Token("source", s) for s in self.source_slots)
        tokens.extend((Token("source", SELECT_END), Token("target_mode", self.target_mode)))
        if self.target_mode == "entity":
            tokens.append(Token("target_entity", self.target_slot))
        if self.target_mode == "position":
            tokens.append(Token("target_position", self.world_position))
        tokens.append(Token("queue", bool(self.queue)))
        return tuple(tokens)

    def protocol(self, frame: ActionFrame):
        """Exact wire rebuild uses metadata presence, not a lossy bool/default projection."""
        from s2clientprotocol import sc2api_pb2 as sc

        if self.frame_key != frame.key:
            raise ValueError("stale command frame/episode/query identity")
        if self.issues:
            raise ValueError("unrepresentable command must remain an audit record")
        action = sc.Action()
        c = action.action_raw.unit_command
        c.ability_id = self.ability
        c.unit_tags.extend(frame.slot_tags[s] for s in self.source_slots)
        if self.target_mode == "entity":
            c.target_unit_tag = frame.slot_tags[self.target_slot]
        elif self.target_mode == "position":
            c.target_world_space_pos.x, c.target_world_space_pos.y = self.world_position
        if self.queue is not None:
            c.queue_command = self.queue
        if self.protocol_game_loop is not None:
            action.game_loop = self.protocol_game_loop
        return action


class PrefixMachine:
    """Both callers choose tokens from this exact current frame and prefix.

    Sources preserve all current-own visible members of a mixed selection. END requires
    selected ANY availability, not ALL. Missing/currently illegal targets fail explicitly.
    """

    def __init__(self, frame: ActionFrame):
        if not isinstance(frame, ActionFrame):
            raise ValueError("query-qualified action frame required")
        if getattr(frame, "query_qualified", True) is False:
            raise ValueError("scheduler frame is not a query-qualified action body")
        if frame.terminal:
            raise ValueError("no action body on terminal observation")
        self.frame = frame
        self.head = "ability"
        self.commands = []
        self.ability = None
        self.sources = []
        self.target_mode = self.target_slot = self.position = None

    def options(self):
        if self.head == "ability":
            return self.frame.ability_options()
        if self.head == "source":
            choices = tuple(sorted(self.frame.own_visible_slots - set(self.sources)))
            if any(self.frame.source_can_issue(s, self.ability) for s in self.sources):
                choices += (SELECT_END,)
            return choices
        if self.head == "target_mode":
            return TARGET_MODES.get(self.frame.targets[self.ability], ())
        if self.head == "target_entity":
            return tuple(range(len(self.frame.slot_tags)))
        if self.head == "target_position":
            return self.frame.map_size
        if self.head == "queue":
            return (False, True)
        if self.head == "bundle":
            return (BUNDLE_NEXT, BUNDLE_STOP)
        return ()

    def accept(self, token: Token):
        if token.head != self.head:
            raise ValueError("wrong causal head or token after BUNDLE_STOP")
        value = token.value
        if self.head == "target_position":
            if (
                not isinstance(value, tuple)
                or len(value) != 2
                or any(type(v) not in (int, float) or not math.isfinite(v) for v in value)
                or not 0 <= value[0] < self.frame.map_size[0]
                or not 0 <= value[1] < self.frame.map_size[1]
            ):
                raise TargetDomainRejection(
                    "POSITION_OUTSIDE_CURRENT_DOMAIN", "position outside current world domain"
                )
        else:
            if self.head in ("ability", "target_entity") or (
                self.head == "source" and value != SELECT_END
            ):
                if type(value) is not int:
                    raise ValueError("pointer/ability must be an integer")
            if self.head == "queue" and type(value) is not bool:
                raise ValueError("queue token must be an explicit boolean")
            if value not in self.options():
                raise TargetDomainRejection(
                    "TOKEN_OUTSIDE_CURRENT_DOMAIN", "token outside shared current prefix domain"
                )
        if self.head == "ability":
            self.ability = value
            self.sources = []
            self.target_mode = self.target_slot = self.position = None
            self.head = "source"
        elif self.head == "source":
            if value == SELECT_END:
                self.head = "target_mode"
            else:
                self.sources.append(value)
        elif self.head == "target_mode":
            self.target_mode = value
            self.head = {"none": "queue", "entity": "target_entity", "position": "target_position"}[
                value
            ]
        elif self.head == "target_entity":
            self.target_slot = value
            self.head = "queue"
        elif self.head == "target_position":
            self.position = value
            self.head = "queue"
        elif self.head == "queue":
            self.commands.append(
                (
                    self.ability,
                    tuple(self.sources),
                    self.target_mode,
                    self.target_slot,
                    self.position,
                    value,
                )
            )
            self.head = "bundle"
        elif self.head == "bundle":
            self.head = "ability" if value == BUNDLE_NEXT else "done"

    def protocol_actions(self):
        from s2clientprotocol import sc2api_pb2 as sc

        if self.head != "done":
            raise ValueError("incomplete bundle; BUNDLE_STOP required")
        actions = []
        for a, sources, mode, target, point, queue in self.commands:
            action = sc.Action()
            c = action.action_raw.unit_command
            c.ability_id = a
            c.unit_tags.extend(self.frame.slot_tags[s] for s in sources)
            c.queue_command = queue
            if mode == "entity":
                c.target_unit_tag = self.frame.slot_tags[target]
            elif mode == "position":
                c.target_world_space_pos.x, c.target_world_space_pos.y = point
            actions.append(action)
        return tuple(actions)


def teacher_force(frame, targets):
    """All commands or an explicit failure: no first-command projection / silent cap."""
    if not targets:
        raise ValueError("ACT body requires a complete nonempty bundle")
    machine = PrefixMachine(frame)
    records = []
    for i, target in enumerate(targets):
        if target.frame_key != frame.key:
            raise ValueError("stale teacher-forcing frame")
        for token in target.tokens():
            records.append((token.head, machine.options(), token.value))
            machine.accept(token)
        token = Token("bundle", BUNDLE_STOP if i == len(targets) - 1 else BUNDLE_NEXT)
        records.append((token.head, machine.options(), token.value))
        machine.accept(token)
    return machine, tuple(records)


def sample(frame, choose):
    """No arbitrary length cap; own-source uniqueness bounds each selection naturally.

    The caller may cancel an unending bundle for an explicit runtime failure; it must not
    silently insert STOP or count truncation as a model decision.
    """
    machine = PrefixMachine(frame)
    while machine.head != "done":
        head, options = machine.head, machine.options()
        machine.accept(Token(head, choose(head, options, tuple(machine.commands))))
    return machine


@dataclass(frozen=True)
class CompleteRequestTarget:
    """Original native target union and echo projection coexist; neither overwrites the other."""

    raw: RawCommandTarget
    native_target_kind: str
    native_event_wire: bytes = field(repr=False)

    @property
    def frame_key(self):
        return self.raw.frame_key

    @classmethod
    def retain_native(cls, raw, event):
        import json

        from poorstar.data.trajectory import encode_bytes

        data = event.get("m_data")
        if not isinstance(data, dict) or len(data) != 1:
            raise ValueError("complete native target union required")
        kind = next(iter(data))
        if kind not in ("None", "TargetUnit", "TargetPoint", "Data"):
            raise ValueError("unknown native target union")
        if kind == "Data" and type(data[kind]) is not int:
            raise ValueError("native opaque Data must be preserved as an integer")
        return cls(raw, kind, json.dumps(event, sort_keys=True, default=encode_bytes).encode())

    def tokens(self):
        if self.native_target_kind == "Data":
            raise TargetDomainRejection(
                "NATIVE_OPERATION_UNRESOLVED",
                "native Data argument requires a verified operation resolver; echo None is not a label",
            )
        return self.raw.tokens()
