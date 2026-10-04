"""Engine: clock evidence."""

from collections import Counter, defaultdict


def event_name(event):
    return event["_event"].rsplit(".", 1)[-1]


def original_signature(event):
    data = event["m_data"]
    if len(data) != 1:
        raise ValueError("original command target union is malformed")
    kind, target = next(iter(data.items()))
    modes = {
        "None": "none",
        "TargetUnit": "entity",
        "TargetPoint": "position",
        "Data": "opaque_data",
    }
    if kind not in modes or (kind == "Data" and type(target) is not int):
        raise ValueError("original command target union is malformed")
    mode = modes[kind]
    point = (target["x"] / 4096, target["y"] / 4096) if mode == "position" else None
    return mode, bool(event["m_cmdFlags"] & 2), point


def engine_signature(command):
    entity = command.get("target_tag") is not None
    point = command.get("position")
    if entity and point is not None:
        raise ValueError("engine command has both target modes")
    return (
        "entity" if entity else "position" if point is not None else "none",
        bool(command["queue"]),
        tuple(point) if point is not None else None,
    )


def reconcile(events, commands, *, user_id):
    """Retain every selected event/command and compare candidate frame sets without joins."""
    original = defaultdict(list)
    updates = Counter()
    counts = Counter()
    for event in events:
        if event.get("_userid", {}).get("m_userId") != user_id:
            continue
        name = event_name(event)
        counts["original_" + name] += 1
        if name == "SCmdEvent" or (name == "SCommandManagerStateEvent" and event["m_state"] == 1):
            original[event["_gameloop"]].append(event)
        if name.startswith("SCmdUpdateTarget"):
            updates[event["_gameloop"]] += 1
    api = defaultdict(list)
    for command in commands:
        if command["kind"] == "unit_command":
            # Measured candidate only; this is NOT a repaired Action.game_loop.
            api[command["response_loop"] - 1].append(command)
            counts["engine_unit_commands"] += 1
            counts["engine_missing_action_game_loop"] += command["protocol_game_loop"] is None
    mismatches, ambiguous, unmatched = [], [], []
    for loop in sorted(set(original) | set(api)):
        left, right = original[loop], api[loop]
        counts["candidate_event_commands"] += len(left)
        if len(left) != len(right):
            mismatches.append(
                {
                    "original_loop": loop,
                    "original_count": len(left),
                    "api_count": len(right),
                    "original_kinds": [event_name(x) for x in left],
                }
            )
        else:
            counts["exact_frame_multiplicity"] += 1
        # Only content compatibility, never assign an ordinal identity within a frame.
        for event in left:
            if event_name(event) != "SCmdEvent":
                continue
            signature = original_signature(event)
            matches = [x for x in right if engine_signature(x) == signature]
            counts["direct_commands"] += 1
            counts["direct_mode_" + signature[0]] += 1
            if (
                len(matches) == 1
                and sum(
                    original_signature(x) == signature for x in left if event_name(x) == "SCmdEvent"
                )
                == 1
            ):
                counts["direct_unique_content_candidate"] += 1
                counts["direct_unique_" + signature[0]] += 1
            elif matches:
                counts["direct_ambiguous_content_candidate"] += 1
                ambiguous.append(
                    {"loop": loop, "mode": signature[0], "candidate_count": len(matches)}
                )
            else:
                counts["direct_without_content_candidate"] += 1
                target = event["m_data"].get("TargetUnit")
                unmatched.append(
                    {
                        "loop": loop,
                        "mode": signature[0],
                        "original_target_tag_zero": bool(
                            target is not None and target["m_tag"] == 0
                        ),
                        "original_ability": event.get("m_abil"),
                        "api_count": len(right),
                        "opaque_data_argument_unresolved": signature[0] == "opaque_data",
                    }
                )
    return {
        "counts": dict(sorted(counts.items())),
        "frame_multiplicity_mismatches": mismatches,
        "ambiguous_direct_commands": ambiguous,
        "unmatched_direct_commands": unmatched,
        "target_update_frames": len(updates),
        "candidate_clock": "ORIGINAL_EVENT_F_TO_RESPONSE_F_PLUS_1",
        "ability_id_mapping_verified": False,
        "entity_tag_mapping_verified": False,
        "selection_mapping_verified": False,
        "observation_phase_causality_verified": False,
        "native_data_target_semantics_verified": False,
        "timestamp_alignment_verified": False,
        "training_ready": False,
        "rl_ready": False,
    }
