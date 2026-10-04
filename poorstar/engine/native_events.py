"""Engine: native events."""

from collections import Counter


def complete_candidate_entries(scope):
    """Retain scope failures as exclusions; never guess eligibility for a damaged record."""
    entries = [
        e
        for e in scope["entries"]
        if e.get("scope_candidate") is True and e.get("base_build") == 55958
    ]
    expected = scope["counts"]
    if (
        len(entries) != expected["compatible_zerg_candidate_matches"]
        or sum(len(e["controlled_zerg_player_ids"]) for e in entries)
        != expected["compatible_zerg_candidate_perspectives"]
    ):
        raise ValueError("complete native corpus scope denominator mismatch")
    return entries


def census_native_events(events, *, controlled_user_ids):
    users = tuple(controlled_user_ids)
    if (
        not users
        or len(set(users)) != len(users)
        or any(type(u) is not int or u < 0 for u in users)
    ):
        raise ValueError("unique explicit controlled native user identities required")
    counts = Counter()
    views = {u: Counter() for u in users}
    abilities = {u: Counter() for u in users}
    command_loops = {u: Counter() for u in users}
    physical = {u: Counter() for u in users}
    observed_users = set()
    previous = -1
    for event in events:
        loop = event.get("_gameloop")
        if type(loop) is not int or loop < previous:
            raise ValueError("native event frames absent, negative or out of order")
        previous = loop
        name = event["_event"].rsplit(".", 1)[-1]
        counts["events"] += 1
        counts[name] += 1
        uid = event.get("_userid", {}).get("m_userId")
        if uid is not None:
            observed_users.add(uid)
        if uid not in views:
            continue
        view = views[uid]
        view["events"] += 1
        view[name] += 1
        if name == "SCommandManagerStateEvent" and event["m_state"] == 1:
            view["manager_state_1_commands"] += 1
            command_loops[uid][loop] += 1
        if name != "SCmdEvent":
            continue
        view["direct_commands"] += 1
        command_loops[uid][loop] += 1
        data = event["m_data"]
        if len(data) != 1 or next(iter(data)) not in ("None", "TargetUnit", "TargetPoint", "Data"):
            raise ValueError("native command target union malformed")
        kind = next(iter(data))
        view["target_" + kind] += 1
        if kind == "TargetUnit":
            physical[uid]["target_unit_tag_present"] += "m_tag" in data[kind]
            physical[uid]["target_unit_tag_zero"] += data[kind].get("m_tag") == 0
        if kind == "TargetPoint":
            physical[uid]["target_point_xy_present"] += all(k in data[kind] for k in ("x", "y"))
        if kind == "Data":
            if type(data[kind]) is not int:
                raise ValueError("native Data target payload malformed")
            physical[uid]["opaque_data_target_present"] += 1
        flags = event.get("m_cmdFlags")
        physical[uid]["cmd_flags_present"] += type(flags) is int
        if type(flags) is int:
            view["queue_flag_set"] += bool(flags & 2)
        ability = event.get("m_abil")
        if ability is None:
            physical[uid]["ability_link_unavailable"] += 1
        elif all(type(ability.get(k)) is int for k in ("m_abilLink", "m_abilCmdIndex")):
            abilities[uid][str(ability["m_abilLink"]) + ":" + str(ability["m_abilCmdIndex"])] += 1
            physical[uid]["ability_link_index_present"] += 1
        else:
            raise ValueError("native ability link/index malformed")
    return dict(
        counts=dict(counts),
        last_event_loop=previous,
        observed_native_users=sorted(observed_users),
        views={str(u): dict(v) for u, v in views.items()},
        native_ability_link_index_counts={str(u): dict(v) for u, v in abilities.items()},
        physical_command_fields={str(u): dict(v) for u, v in physical.items()},
        same_loop_command_lengths={
            str(u): {str(k): v for k, v in Counter(command_loops[u].values()).items()}
            for u in users
        },
        command_frames={str(u): len(command_loops[u]) for u in users},
        engine_ability_mapping_verified=False,
        opaque_data_target_semantics_verified=False,
        actor_inputs_generated=False,
    )
