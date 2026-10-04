"""Coarse action profiles with shared sampling and teacher-forcing semantics."""

import hashlib
import json
from dataclasses import dataclass, replace
from types import MappingProxyType

from poorstar.actions.abilities import canonical_ability
from poorstar.actions.contract import (
    BUNDLE_NEXT,
    BUNDLE_STOP,
    SELECT_END,
    CompleteRequestTarget,
    PrefixMachine,
    RawCommandTarget,
    TargetDomainRejection,
    Token,
)
from poorstar.actions.frame import EngineActionCatalog

SCHEMA = "poorstar_c3_coarse_action_v1"


@dataclass(frozen=True)
class CoarseActionProfile:
    catalog: EngineActionCatalog
    conversions: object
    maximum_sources: int = 63

    def __post_init__(self):
        if type(self.maximum_sources) is not int or self.maximum_sources < 1:
            raise ValueError("positive declared selection limit required")

    @classmethod
    def from_catalog(cls, catalog, *, maximum_sources=63):
        engine = EngineActionCatalog.from_protocol(catalog)
        # Derive families from THIS engine, not a handwritten version-specific ID table.
        names = {}
        for a in catalog.abilities:
            if a.remaps_to_ability_id == 0:
                names.setdefault(a.link_name, []).append(int(a.ability_id))
        conversions = {}
        for original, replacement in (
            ("TransportUnloadUnit", "TransportUnloadAll"),
            ("CancelSlot", "CancelLast"),
        ):
            if len(names.get(original, ())) != 1 or len(names.get(replacement, ())) != 1:
                raise ValueError("unique catalog operation families required: " + original)
            old, new = names[original][0], names[replacement][0]
            if engine.targets[new] != 1:
                raise ValueError("coarse operation must be catalog targetless")
            conversions[old] = new
        return cls(engine, MappingProxyType(conversions), maximum_sources)

    @property
    def sha256(self):
        value = dict(
            schema=SCHEMA,
            catalog=self.catalog.sha256,
            conversions=sorted(self.conversions.items()),
            maximum_sources=self.maximum_sources,
        )
        return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()

    def check_frame(self, frame):
        if frame.remaps != self.catalog.remaps or frame.targets != self.catalog.targets:
            raise ValueError("normalization profile and current engine catalog differ")

    def ability(self, ability):
        canonical = canonical_ability(ability, self.catalog.remaps)
        return self.conversions.get(canonical, canonical)

    def ability_options(self, frame):
        self.check_frame(frame)
        # Query advertisements are already catalog-canonicalized. The old path
        # expanded every catalog alias and canonicalized thousands of entries
        # again on every command; this exact intersection only visits advertised
        # canonical abilities, preserving the sorted current-frame domain.
        available = set().union(*frame.advertised.values()) if frame.advertised else set()
        return tuple(
            sorted(
                a
                for a in available
                if a > 0
                and a in frame.targets
                and frame.targets[a] is not None
                and self.ability(a) == a
            )
        )


@dataclass(frozen=True)
class NormalizedTarget:
    raw: RawCommandTarget
    original: RawCommandTarget | CompleteRequestTarget
    profile_sha256: str
    reasons: tuple[str, ...]
    discarded_source_count: int

    @property
    def frame_key(self):
        return self.raw.frame_key

    def tokens(self):
        return self.raw.tokens()


def normalize_target(target, frame, profile):
    profile.check_frame(frame)
    original = target.raw if isinstance(target, CompleteRequestTarget) else target
    if not isinstance(original, RawCommandTarget) or original.frame_key != frame.key:
        raise ValueError("current complete target required")
    if original.ability not in profile.catalog.remaps:
        raise TargetDomainRejection("UNKNOWN_ENGINE_ABILITY", "ability absent from profile catalog")
    ability = profile.ability(original.ability)
    family = canonical_ability(original.ability, profile.catalog.remaps)
    converted = family in profile.conversions
    if (
        isinstance(target, CompleteRequestTarget)
        and target.native_target_kind == "Data"
        and not converted
    ):
        raise TargetDomainRejection(
            "UNKNOWN_NATIVE_DATA_FAMILY", "opaque Data family has no authorized conversion"
        )
    issues = original.issues
    # A passenger pointer is no longer applicable ONLY under explicit unload-all conversion.
    if converted:
        issues = tuple(i for i in issues if i != "UNOBSERVABLE_ENTITY_TARGET")
    reasons = []
    if converted:
        reasons.append("OPERATION_COARSENED")
    elif ability != original.ability:
        reasons.append("CATALOG_CANONICALIZED")
    discarded = max(0, len(original.source_slots) - profile.maximum_sources)
    if discarded:
        reasons.append("SELECTION_PREFIX_CAPPED")
    raw = replace(
        original,
        ability=ability,
        source_slots=original.source_slots[: profile.maximum_sources],
        target_mode="none" if converted else original.target_mode,
        target_slot=None if converted else original.target_slot,
        world_position=None if converted else original.world_position,
        issues=issues,
    )
    return NormalizedTarget(raw, target, profile.sha256, tuple(reasons), discarded)


class CoarsePrefixMachine(PrefixMachine):
    def __init__(self, frame, profile):
        profile.check_frame(frame)
        super().__init__(frame)
        self.profile = profile
        self._ability_options = profile.ability_options(frame)

    def options(self):
        if self.head == "ability":
            return self._ability_options
        if self.head == "source":
            supported = any(self.frame.source_can_issue(s, self.ability) for s in self.sources)
            if len(self.sources) >= self.profile.maximum_sources:
                return (SELECT_END,) if supported else ()
            if len(self.sources) == self.profile.maximum_sources - 1 and not supported:
                # Last source must make END legal; sampling cannot paint itself into
                # an empty domain by spending the cap on an entirely unsupported set.
                return tuple(
                    s
                    for s in super().options()
                    if s != SELECT_END and self.frame.source_can_issue(s, self.ability)
                )
        return super().options()


def coarse_teacher_force(frame, targets, profile):
    if not targets:
        raise ValueError("ACT requires a nonempty complete bundle")
    machine = CoarsePrefixMachine(frame, profile)
    trace = []
    normalized = []
    for i, original in enumerate(targets):
        target = (
            original
            if isinstance(original, NormalizedTarget)
            else normalize_target(original, frame, profile)
        )
        if target.profile_sha256 != profile.sha256 or target.frame_key != frame.key:
            raise ValueError("normalized target profile/frame drift")
        normalized.append(target)
        for token in target.tokens():
            trace.append((token.head, machine.options(), token.value))
            machine.accept(token)
        token = Token("bundle", BUNDLE_STOP if i == len(targets) - 1 else BUNDLE_NEXT)
        trace.append((token.head, machine.options(), token.value))
        machine.accept(token)
    return machine, tuple(trace), tuple(normalized)


def coarse_sample(frame, choose, profile, *, token_watchdog=100000):
    machine = CoarsePrefixMachine(frame, profile)
    for _ in range(token_watchdog):
        if machine.head == "done":
            return machine
        if not machine.options():
            raise RuntimeError("empty live coarse prefix domain")
        machine.accept(
            Token(machine.head, choose(machine.head, machine.options(), tuple(machine.commands)))
        )
    raise RuntimeError("explicit sampling watchdog exceeded; no hidden bundle STOP")


def validate_coarse_bundle(frame, targets, profile):
    teacher, trace, normalized = coarse_teacher_force(frame, targets, profile)
    cursor = iter(trace)

    def choose(head, options, completed):
        h, o, value = next(cursor)
        if (head, options) != (h, o):
            raise RuntimeError("coarse teacher/sampler domain drift")
        return value

    sampled = coarse_sample(frame, choose, profile)
    if next(cursor, None) is not None or sampled.commands != teacher.commands:
        raise RuntimeError("coarse complete bundle parity failure")
    if tuple(a.SerializeToString() for a in sampled.protocol_actions()) != tuple(
        a.SerializeToString() for a in teacher.protocol_actions()
    ):
        raise RuntimeError("coarse wire parity failure")
    return teacher, trace, normalized
