"""Evaluation: action diagnostics."""

from collections import Counter

from poorstar.actions.coarse import CoarsePrefixMachine, normalize_target, validate_coarse_bundle
from poorstar.actions.contract import BUNDLE_NEXT, BUNDLE_STOP, TargetDomainRejection, Token


def diagnose_body(row, profile):
    if row.decision != "ACT":
        raise ValueError("ACT required")
    requested = len(row.targets)
    if row.body_rejection:
        return dict(
            admissible=False,
            reason=row.body_rejection,
            head="native",
            command_index=None,
            requested_commands=requested,
            issues=[],
            cause_status="NATIVE_ECHO_LIMITATION",
        )
    machine = CoarsePrefixMachine(row.body, profile)
    for index, original in enumerate(row.targets):
        raw = original.raw if hasattr(original, "raw") else original
        normalized = None
        token = None
        try:
            normalized = normalize_target(original, row.body, profile)
            for token in normalized.tokens():
                machine.accept(token)
            token = Token("bundle", BUNDLE_STOP if index == requested - 1 else BUNDLE_NEXT)
            machine.accept(token)
        except TargetDomainRejection as error:
            ability = normalized.raw.ability if normalized is not None else raw.ability
            sources = normalized.raw.source_slots if normalized is not None else raw.source_slots
            return dict(
                admissible=False,
                reason=error.code,
                head=token.head if token else "arguments",
                command_index=index,
                requested_commands=requested,
                issues=list(error.issues),
                ability=ability,
                original_ability=raw.ability,
                sources_requested=len(sources),
                sources_current_own=sum(s in row.body.own_visible_slots for s in sources),
                sources_with_current_availability=sum(
                    row.body.source_can_issue(s, ability) for s in sources if s is not None
                ),
                ability_currently_available=ability in profile.ability_options(row.body),
                cause_status="CURRENT_ACTOR_DOMAIN_MISMATCH_CAUSE_UNRESOLVED",
            )
    # This also checks the independent sampler/wire path. Its faults propagate.
    machine, trace, normalized = validate_coarse_bundle(row.body, row.targets, profile)
    return dict(
        admissible=True,
        requested_commands=requested,
        normalized_commands=len(machine.commands),
        head_denominators=dict(Counter(h for h, o, v in trace)),
        discarded_sources=sum(t.discarded_source_count for t in normalized),
    )


def domain_census(rows, profile, *, example_limit=64):
    counts = Counter()
    reasons = Counter()
    strata = Counter()
    heads = Counter()
    examples = []
    seen = set()
    for row in rows:
        key = (
            row.replay_sha256,
            row.scheduler.key.player_id,
            row.scheduler.key.episode,
            row.scheduler.key.loop,
        )
        if key in seen:
            raise ValueError("duplicate original decision in domain census")
        seen.add(key)
        if not row.valid or row.burn_in or not row.loss:
            raise ValueError("unique original archive rows required")
        counts["original_rows"] += 1
        counts[row.decision] += 1
        if row.decision != "ACT":
            continue
        item = diagnose_body(row, profile)
        counts["requested_commands"] += len(row.targets)
        if item["admissible"]:
            counts["admissible_bundles"] += 1
            counts["supervised_commands"] += item["normalized_commands"]
            counts["discarded_sources"] += item["discarded_sources"]
            heads.update(item["head_denominators"])
        else:
            counts["rejected_bundles"] += 1
            counts["body_commands_without_supervision"] += len(row.targets)
            reasons[item["reason"]] += 1
            strata[f"{item['reason']}/{item['head']}/{item.get('ability', 'native')}"] += 1
            if len(examples) < example_limit:
                examples.append(
                    dict(
                        item,
                        replay_sha256=row.replay_sha256,
                        player_id=row.scheduler.key.player_id,
                        scheduler_loop=row.scheduler.key.loop,
                        body_loop=row.body.key.loop,
                        observation_sha256=row.body.key.observation_sha256,
                        query_sha256=row.body.key.query_sha256,
                    )
                )
    if counts["admissible_bundles"] + counts["rejected_bundles"] != counts["ACT"]:
        raise ValueError("ACT denominator does not close")
    return dict(
        schema="poorstar_c3_unique_action_domain_diagnostic_v1",
        counts=dict(counts),
        reasons=dict(reasons),
        first_failure_head_ability_strata=dict(strata),
        head_denominators=dict(heads),
        examples=examples,
        rejected_ACT_fraction=counts["rejected_bundles"] / max(1, counts["ACT"]),
        lost_command_fraction=counts["body_commands_without_supervision"]
        / max(1, counts["requested_commands"]),
        treatment="WHOLE_BODY_REJECTED_ACT_DELAY_RETAINED",
        causation="UNRESOLVED_REQUIRES_NATIVE_ENGINE_EFFECT_AND_NEIGHBOR_FRAME_EVIDENCE",
        raw_tags_or_positions_in_report=False,
        optimizer_steps=0,
        rl_ready=False,
    )
