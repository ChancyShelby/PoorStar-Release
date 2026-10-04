"""Engine: native data targets."""

from collections import Counter, defaultdict


def census_data_families(events, *, controlled_user_ids):
    users = tuple(controlled_user_ids)
    if (
        not users
        or len(set(users)) != len(users)
        or any(type(u) is not int or u < 0 for u in users)
    ):
        raise ValueError("unique explicit controlled native identities required")
    views = {u: Counter() for u in users}
    families = {u: defaultdict(Counter) for u in users}
    previous = -1
    for event in events:
        loop = event.get("_gameloop")
        if type(loop) is not int or loop < previous:
            raise ValueError("native event frames absent, negative or out of order")
        previous = loop
        uid = event.get("_userid", {}).get("m_userId")
        if uid not in views or event["_event"].rsplit(".", 1)[-1] != "SCmdEvent":
            continue
        view = views[uid]
        view["all_direct_commands"] += 1
        data = event["m_data"]
        if len(data) != 1 or next(iter(data)) not in ("None", "TargetUnit", "TargetPoint", "Data"):
            raise ValueError("native command target union malformed")
        if "Data" not in data:
            continue
        if type(data["Data"]) is not int or type(event.get("m_cmdFlags")) is not int:
            raise ValueError("native Data payload/flags malformed")
        ability = event.get("m_abil")
        if ability is None:
            key = "ABILITY_LINK_UNAVAILABLE"
        elif all(type(ability.get(k)) is int for k in ("m_abilLink", "m_abilCmdIndex")):
            key = str(ability["m_abilLink"]) + ":" + str(ability["m_abilCmdIndex"])
        else:
            raise ValueError("native Data ability link/index malformed")
        view["data_commands"] += 1
        counts = families[uid][key]
        counts["commands"] += 1
        counts["queue_flag_set"] += bool(event["m_cmdFlags"] & 2)
        counts["opaque_payload_zero"] += data["Data"] == 0
        counts["opaque_payload_nonzero"] += data["Data"] != 0
        counts["opaque_payload_negative"] += data["Data"] < 0
    return dict(
        views={str(u): dict(v) for u, v in views.items()},
        families={str(u): {k: dict(v) for k, v in f.items()} for u, f in families.items()},
        native_data_semantics_verified=False,
        actor_inputs_generated=False,
    )
