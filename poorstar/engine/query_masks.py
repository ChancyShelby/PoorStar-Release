"""Engine: query masks."""

from poorstar.actions.abilities import canonical_ability
from poorstar.engine.query_evidence import ability_answers


def current_own_answers(observation, request, response, *, control_tags=()):
    if not observation.HasField("raw_data"):
        raise ValueError("current raw player observation required")
    own = {}
    all_tags = set()
    for unit in observation.raw_data.units:
        if not unit.HasField("tag") or unit.tag <= 0 or unit.tag in all_tags:
            raise ValueError("current unit tag absent, invalid or duplicated")
        all_tags.add(int(unit.tag))
        if not unit.HasField("alliance"):
            raise ValueError("current unit alliance absent")
        if unit.alliance == 1:
            if not unit.HasField("unit_type") or not unit.HasField("display_type"):
                raise ValueError("current own unit type/visibility absent")
            own[int(unit.tag)] = unit
    controls = list(control_tags)
    tags = [int(x.unit_tag) for x in request.abilities]
    if tags != sorted(own) + controls or any(t in all_tags for t in controls):
        raise ValueError("query must cover exact current own units and absent controls")
    answers = ability_answers(response, tags, control_tags=controls)
    for tag, reply in zip(tags, response.abilities):
        if tag in own:
            if not reply.HasField("unit_type_id") or reply.unit_type_id != own[tag].unit_type:
                raise ValueError("current own query unit type identity mismatch")
    return {tag: answers[tag] for tag, unit in own.items() if unit.display_type == 1}


def availability_counts(ability, sources, current_answers, remaps):
    """Separate global-any, selected-any and selected-all denominators without filtering."""
    if not sources or len(set(sources)) != len(sources):
        raise ValueError("complete nonempty unique selected sources required")
    wanted = canonical_ability(ability, remaps)
    canonical = {
        tag: {canonical_ability(a, remaps) for a, _ in answers}
        for tag, answers in current_answers.items()
    }
    present = [tag in canonical for tag in sources]
    advertised = [tag in canonical and wanted in canonical[tag] for tag in sources]
    global_any = any(wanted in skills for skills in canonical.values())
    return dict(
        commands=1,
        source_units=len(sources),
        commands_global_current_own_available=int(global_any),
        commands_global_current_own_unavailable=int(not global_any),
        commands_selected_any_available=int(any(advertised)),
        commands_selected_all_available=int(all(advertised)),
        commands_selected_none_available=int(not any(advertised)),
        commands_selected_all_present=int(all(present)),
        source_units_absent=sum(not p for p in present),
        source_units_present_unavailable=sum(p and not a for p, a in zip(present, advertised)),
        source_units_available=sum(advertised),
        commands_global_available_but_selected_none=int(global_any and not any(advertised)),
        commands_selected_any_but_not_all=int(any(advertised) and not all(advertised)),
    )


def absent_control_class(*, observed_dead_before, later_reappearance):
    """Full-lifecycle audit classification only; later appearance must never drive a mask."""
    if later_reappearance:
        return "DEAD_THEN_REAPPEARED" if observed_dead_before else "TEMPORARILY_ABSENT"
    return "OBSERVED_DEAD_NO_REAPPEARANCE" if observed_dead_before else "UNCLASSIFIED_ABSENCE"


def choose_absent_own_control(
    seen_own, current_raw_tags, observed_dead, *, prefer_observed_dead=False
):
    """Causal audit-control selection; absent responses never populate actor candidates."""
    missing = set(seen_own) - set(current_raw_tags)
    if prefer_observed_dead:
        confirmed = missing & set(observed_dead)
        if confirmed:
            return min(confirmed)
    return min(missing, default=None)
