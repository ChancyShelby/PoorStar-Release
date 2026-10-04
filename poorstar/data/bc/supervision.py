"""Data: bc: supervision."""

from dataclasses import dataclass, replace

from poorstar.actions.coarse import NormalizedTarget, coarse_teacher_force, normalize_target
from poorstar.actions.contract import TargetDomainRejection

STRICT = "STRICT_WHOLE_BODY_V1"
DISTAR_SKIP = "DISTAR_COARSE_SKIP_V1"
POLICIES = (STRICT, DISTAR_SKIP)


@dataclass(frozen=True)
class TeacherPreparation:
    targets: tuple
    trace: tuple
    skipped: tuple


def prepare_teacher(frame, targets, profile, *, policy=STRICT, body_rejection=None):
    if policy not in POLICIES:
        raise ValueError("unknown supervision policy")
    if body_rejection:
        # Unassociated native Data/no-echo cannot be matched by command ordinal.
        raise TargetDomainRejection(body_rejection, "native body association unavailable")
    if policy == STRICT:
        _, trace, normalized = coarse_teacher_force(frame, targets, profile)
        return TeacherPreparation(normalized, trace, ())
    retained = []
    skipped = []
    for index, original in enumerate(targets):
        try:
            if isinstance(original, NormalizedTarget):
                raise ValueError("filtering requires original complete targets")
            raw = original.raw if hasattr(original, "raw") else original
            missing = sum(s is None for s in raw.source_slots)
            projected = original
            if missing:
                slots = tuple(s for s in raw.source_slots if s is not None)
                issues = tuple(i for i in raw.issues if i != "UNOBSERVABLE_SOURCE_REFERENCE")
                if not slots and "EMPTY_SOURCE_SET" not in issues:
                    issues += ("EMPTY_SOURCE_SET",)
                raw = replace(raw, source_slots=slots, issues=issues)
                projected = replace(original, raw=raw) if hasattr(original, "raw") else raw
            normalized = normalize_target(projected, frame, profile)
            if missing:
                normalized = replace(
                    normalized,
                    original=original,
                    reasons=normalized.reasons + ("UNOBSERVABLE_SOURCES_DROPPED",),
                    discarded_source_count=normalized.discarded_source_count + missing,
                )
            # Validate only known data-domain rejections; programming errors escape.
            coarse_teacher_force(frame, (normalized,), profile)
            retained.append(normalized)
        except TargetDomainRejection as error:
            skipped.append((index, error.code))
    if not retained:
        return TeacherPreparation((), (), tuple(skipped))
    # Preserve original order; NEXT/STOP describe the declared filtered sequence.
    _, trace, normalized = coarse_teacher_force(frame, retained, profile)
    return TeacherPreparation(normalized, trace, tuple(skipped))


def supervision_census(rows, profile, *, policy=DISTAR_SKIP):
    """Ordered original TRAIN episodes; memory scales with episodes, not rows.

    Contiguous episodes and strictly increasing original clocks establish unique
    rows without retaining millions of frame keys. Unordered input is rejected.
    Zero missingness is not a gate.
    """
    from collections import Counter

    from poorstar.models.policy import HEADS

    counts = Counter()
    heads = Counter()
    requested = Counter()
    retained = Counter()
    reasons = Counter()
    seen_episodes = set()
    episode = None
    previous_loop = None
    for row in rows:
        key = (row.replay_sha256, row.scheduler.key.player_id, row.scheduler.key.episode)
        loop = row.scheduler.key.loop
        if key != episode:
            if key in seen_episodes:
                raise ValueError("duplicate/discontiguous original episode in supervision census")
            seen_episodes.add(key)
            episode = key
            previous_loop = None
        if previous_loop is not None and loop <= previous_loop:
            raise ValueError("duplicate/nonmonotonic original row in supervision census")
        previous_loop = loop
        if not row.valid or row.burn_in or not row.loss:
            raise ValueError("original nonpadding archive rows required")
        counts["rows"] += 1
        counts[row.decision] += 1
        heads.update(("act", "delay"))
        if row.decision != "ACT":
            continue
        counts["requested_commands"] += len(row.targets)
        for t in row.targets:
            raw = t.raw if hasattr(t, "raw") else t
            requested[raw.ability] += 1
        try:
            prepared = prepare_teacher(
                row.body, row.targets, profile, policy=policy, body_rejection=row.body_rejection
            )
        except TargetDomainRejection as e:
            counts["whole_body_rejected"] += 1
            counts["skipped_commands"] += len(row.targets)
            reasons[e.code] += len(row.targets)
            continue
        counts["retained_commands"] += len(prepared.targets)
        counts["skipped_commands"] += len(prepared.skipped)
        counts["commands_with_missing_sources_dropped"] += sum(
            n.raw.source_slots
            != (n.original.raw if hasattr(n.original, "raw") else n.original).source_slots
            and "UNOBSERVABLE_SOURCES_DROPPED" in n.reasons
            for n in prepared.targets
        )
        counts["discarded_source_references"] += sum(
            n.discarded_source_count for n in prepared.targets
        )
        if prepared.targets:
            counts["supervised_bodies"] += 1
        else:
            counts["empty_filtered_bodies"] += 1
        if prepared.skipped and prepared.targets:
            counts["partially_supervised_bodies"] += 1
        heads.update(h for h, o, v in prepared.trace)
        retained.update(n.raw.ability for n in prepared.targets)
        reasons.update(r for i, r in prepared.skipped)
    if counts["requested_commands"] != counts["retained_commands"] + counts["skipped_commands"]:
        raise ValueError("command accounting does not close")
    missing = [h for h in ("act", "delay") + HEADS if heads[h] == 0]
    return dict(
        policy=policy,
        counts=dict(counts),
        head_denominators=dict(heads),
        # JSON object keys are strings. Canonicalize before hashing so numeric
        # sort order cannot change after a report's write/read round trip.
        requested_ability_counts={str(k): v for k, v in requested.items()},
        retained_ability_counts={str(k): v for k, v in retained.items()},
        skip_reasons=dict(reasons),
        skipped_command_fraction=counts["skipped_commands"] / max(1, counts["requested_commands"]),
        missing_applicable_heads=missing,
        all_heads_observed=not missing,
        uniqueness_method="CONTIGUOUS_EPISODES_STRICTLY_INCREASING_ORIGINAL_CLOCKS",
        unique_episodes=len(seen_episodes),
        causal_rejection_adjudication_required=False,
        causation="NOT_ADJUDICATED_NOT_ASSUMED_HUMAN_FAILURE",
        capability_evaluated=False,
        optimizer_steps=0,
    )
