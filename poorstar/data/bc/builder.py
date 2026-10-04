"""Build BC decisions from verified protocol streams and command timing."""

import gzip
import hashlib
import json
from collections import defaultdict

from poorstar.actions.abilities import canonical_ability
from poorstar.actions.coarse import CoarseActionProfile
from poorstar.actions.contract import RawCommandTarget
from poorstar.actions.frame import ActionFrame, BoundQuery, SchedulerFrame
from poorstar.actions.timing import schedule_targets
from poorstar.data.bc.records import BCDecision, BCShardWriter
from poorstar.data.trajectory import decode_bytes, encode_bytes
from poorstar.engine.observation import ExtractionContext
from poorstar.engine.protocol_wire import read_protocol_wire


def partition_native_commands(native_events, *, terminal_loop):
    """Separate terminal tail commands from the measured playable interval.

    Replay files can retain winner input after the first engine result (including
    input at that exact loop). Such input has no nonterminal body observation.
    Preserve it in an external audit rather than changing the terminal or labels.
    Malformed clocks remain errors; this is not a general scheduling fallback.
    """
    if type(terminal_loop) is not int or terminal_loop < 1:
        raise ValueError("positive measured terminal loop required")
    active = []
    tail = []
    for event in native_events:
        if not (
            event["_event"].endswith("SCmdEvent")
            or (event["_event"].endswith("SCommandManagerStateEvent") and event.get("m_state") == 1)
        ):
            continue
        loop = event["_gameloop"]
        if type(loop) is not int or loop < 0:
            raise ValueError("invalid native command clock")
        (active if loop < terminal_loop else tail).append(event)
    return tuple(active), tuple(tail)


def discover_replay_terminal(
    controller,
    *,
    player_id,
    metadata_duration_loops,
    step_chunk=1,
    restart=None,
    terminal_probe_margin=None,
):
    """Measure actual engine termination without storing a full one-loop sidecar.

    Metadata controls a finite resource guard only. It is never the terminal label.
    A chunked response can be terminal AFTER the first terminal loop. Chunked
    discovery therefore requires replay restart and one-loop refinement of the
    last nonterminal interval. A fresh replay supplies the training observations.
    """
    if metadata_duration_loops < 1 or step_chunk < 1:
        raise ValueError("positive discovery guard required")
    if step_chunk > 1 and restart is None:
        raise ValueError("chunked terminal discovery requires restart for exact refinement")
    guard = metadata_duration_loops * 2 + 4096
    current = controller.observe()
    previous = int(current.observation.game_loop)
    requested = 0
    if previous != 0 or current.player_result:
        raise ValueError("fresh nonterminal replay at loop zero required")
    lower = previous
    probe_used = False
    probe_fallback = False
    if terminal_probe_margin is not None:
        if type(terminal_probe_margin) is not int or terminal_probe_margin < 1 or restart is None:
            raise ValueError("positive terminal probe margin and replay restart required")
        # Header duration chooses a probe, never a terminal label. Normally it
        # lies within one second of true termination, so this avoids replaying
        # hundreds of one-loop refinement RPCs after a large chunk overshoot.
        probe = max(0, metadata_duration_loops - terminal_probe_margin)
        if probe:
            controller.step(probe)
            current = controller.observe()
            probe_used = True
            loop = int(current.observation.game_loop)
            if current.observation.player_common.player_id != player_id or not 0 < loop <= probe:
                raise ValueError("terminal probe clock/player invalid")
            if current.player_result:
                # A bad/late header does not justify a long assumed interval.
                # Re-run ordinary engine discovery and retain this fallback.
                restart()
                raw, report = discover_replay_terminal(
                    controller,
                    player_id=player_id,
                    metadata_duration_loops=metadata_duration_loops,
                    step_chunk=step_chunk,
                    restart=restart,
                )
                report.update(
                    terminal_probe_margin=terminal_probe_margin,
                    terminal_probe_used=True,
                    terminal_probe_fallback=True,
                    probe_requested_simulation_loops=probe,
                )
                return raw, report
            if loop != probe:
                raise ValueError("nonterminal short terminal probe")
            previous = loop
            requested = probe
        step_chunk = terminal_probe_margin
    while not current.player_result:
        lower = previous
        if requested >= guard:
            raise ValueError(
                "terminal discovery resource guard exceeded; metadata is not terminal truth"
            )
        count = min(step_chunk, guard - requested)
        controller.step(count)
        requested += count
        current = controller.observe()
        loop = int(current.observation.game_loop)
        if (
            current.observation.player_common.player_id != player_id
            or loop <= previous
            or loop > previous + count
        ):
            raise ValueError("terminal discovery clock/player invalid")
        if not current.player_result and loop != previous + count:
            raise ValueError("nonterminal short engine step")
        previous = loop
    upper = previous
    refined = 0
    if step_chunk > 1:
        restart()
        current = controller.observe()
        if current.player_result or int(current.observation.game_loop) != 0:
            raise ValueError("terminal refinement needs fresh replay")
        if lower:
            controller.step(lower)
            current = controller.observe()
        if current.player_result or int(current.observation.game_loop) != lower:
            raise ValueError("refinement lower bound no longer nonterminal")
        previous = lower
        while not current.player_result:
            if previous >= upper:
                raise ValueError("terminal refinement exceeded measured upper bound")
            controller.step(1)
            current = controller.observe()
            refined += 1
            loop = int(current.observation.game_loop)
            if loop != previous + 1 or current.observation.player_common.player_id != player_id:
                raise ValueError("terminal refinement clock/player drift")
            previous = loop
    return current, dict(
        schema="poorstar_c3_actual_engine_terminal_discovery_v2",
        terminal_loop=previous,
        terminal_loop_exact=True,
        chunk_terminal_upper_bound=upper,
        last_nonterminal_chunk_loop=lower,
        terminal_probe_margin=terminal_probe_margin,
        terminal_probe_used=probe_used,
        terminal_probe_fallback=probe_fallback,
        refinement_one_loop_steps=refined,
        metadata_duration_loops=metadata_duration_loops,
        chunk_requested_simulation_loops=requested,
        requested_simulation_loops=requested + (lower + refined if step_chunk > 1 else 0),
        terminal_observation_sha256=hashlib.sha256(current.SerializeToString()).hexdigest(),
        completion="TERMINAL_OBSERVATION_REACHED",
        player_id=player_id,
    )


def decisions_from_controller(
    controller,
    game_info,
    catalog,
    *,
    context,
    episode,
    replay_sha256,
    native_events,
    terminal_loop,
    maximum_delay=32,
):
    """Decision-driven extraction: full game, exact action intervals, fewer observation RPCs.

    Intermediate simulation is still executed by SC2. Only irrelevant observation/storage
    RPCs are omitted. Every declared native command time receives a fresh body query and
    exactly one step to its echo; unseen command echoes in a long advance fail extraction.
    """
    from s2clientprotocol import query_pb2

    profile = CoarseActionProfile.from_catalog(catalog)
    events = defaultdict(list)
    for event in native_events:
        if event["_event"].endswith("SCmdEvent") or (
            event["_event"].endswith("SCommandManagerStateEvent") and event.get("m_state") == 1
        ):
            events[event["_gameloop"]].append(event)
    plans = schedule_targets(
        sorted(events), initial_loop=0, terminal_loop=terminal_loop, maximum_delay=maximum_delay
    )
    current = controller.observe()
    if current.observation.game_loop != 0 or current.player_result:
        raise ValueError("initial nonterminal loop-zero response required")

    def advance(loop, *, body_echo=False):
        nonlocal current
        delta = loop - int(current.observation.game_loop)
        if delta < 0:
            raise ValueError("extraction clock moved backwards")
        if delta:
            controller.step(delta)
            current = controller.observe()
        if (
            int(current.observation.game_loop) != loop
            or current.observation.player_common.player_id != context.player_id
        ):
            raise ValueError("actual replay clock/player differs from declared phase")
        units = tuple(
            a
            for a in current.actions
            if a.HasField("action_raw") and a.action_raw.HasField("unit_command")
        )
        if delta and units and not body_echo:
            raise ValueError("unregistered command echo in scheduler advance")
        return units

    for plan in plans:
        advance(plan.scheduler_loop)
        scheduler = SchedulerFrame.from_protocol(
            current, game_info, profile.catalog, context=context, episode=episode
        )
        if plan.decision == "WAIT":
            yield BCDecision(replay_sha256, scheduler, "WAIT", plan.delay)
            continue
        advance(plan.body_loop)
        if current.player_result:
            raise ValueError("terminal before registered action")
        req = query_pb2.RequestQuery()
        for tag in sorted(u.tag for u in current.observation.raw_data.units if u.alliance == 1):
            req.abilities.add(unit_tag=tag)
        receipt = BoundQuery.bind(current.observation, req, controller.query(req))
        body = ActionFrame.from_protocol(
            current, game_info, profile.catalog, receipt, context=context, episode=episode
        )
        actions = advance(plan.body_loop + 1, body_echo=True)
        targets = tuple(RawCommandTarget.from_protocol(a, body) for a in actions)
        original = events[plan.body_loop]
        data = [e for e in original if "Data" in e.get("m_data", {})]
        converted = sum(
            t.ability in profile.catalog.remaps
            and profile.ability(t.ability) != canonical_ability(t.ability, profile.catalog.remaps)
            for t in targets
        )
        rejection = (
            "NATIVE_COMMAND_WITHOUT_UNIT_ECHO"
            if not targets
            else ("UNQUALIFIED_NATIVE_DATA_FAMILY" if len(data) > converted else None)
        )
        wires = tuple(
            json.dumps(e, sort_keys=True, default=encode_bytes).encode() for e in original
        )
        yield BCDecision(
            replay_sha256,
            scheduler,
            "ACT",
            plan.delay,
            body,
            targets,
            body_rejection=rejection,
            native_event_wires=wires,
        )
    advance(terminal_loop)
    if not current.player_result:
        raise ValueError("declared engine terminal loop is not terminal")


def decisions_from_audit(trajectory, query_trajectory, geometry, catalog, *, maximum_delay=32):
    from s2clientprotocol import query_pb2
    from s2clientprotocol import sc2api_pb2 as sc

    from poorstar.cli.engine.check_horizon import verified_report

    report, digest = verified_report(trajectory)
    qr, qd = verified_report(query_trajectory)
    gr, gd = verified_report(geometry)
    if (
        not report.get("same_run_query_binding_verified")
        or report.get("input_query_digest") != qd
        or report["artifacts"]["responses.protocol.gz"] != qr["artifacts"]["responses.protocol.gz"]
    ):
        raise ValueError("same-run action/query observation identity required")
    if any(gr[k] != report[k] for k in ("replay_sha256", "base_build", "controlled_player_id")):
        raise ValueError("trajectory/map/source mismatch")
    if report["base_build"] != 55958:
        raise ValueError("unqualified engine clock")
    info = sc.ResponseGameInfo()
    info.ParseFromString((geometry / "game_info.protocol").read_bytes())
    engine = sc.ResponseData()
    engine.ParseFromString((query_trajectory / "ability_catalog.protocol").read_bytes())
    profile = CoarseActionProfile.from_catalog(catalog)
    if {a.ability_id: a.SerializeToString() for a in engine.abilities} != {
        a.ability_id: a.SerializeToString() for a in catalog.abilities
    }:
        raise ValueError("unit catalog and query catalog abilities differ")
    native = defaultdict(list)
    with gzip.open(trajectory / "original_events.jsonl.gz", "rt") as f:
        for line in f:
            e = json.loads(line, object_hook=decode_bytes)
            if (
                e["_event"].endswith("SCmdEvent")
                and e.get("_userid", {}).get("m_userId") == report["controlled_user_id"]
            ):
                native[e["_gameloop"]].append(e)
    command_times = set(native)
    with gzip.open(trajectory / "engine_actions.jsonl.gz", "rt") as f:
        for line in f:
            e = json.loads(line)
            if e["kind"] == "unit_command":
                command_times.add(e["response_loop"] - 1)
    # Terminal loop comes from the fully exhausted wire, not metadata's approximate duration.
    terminal = None
    for wire in read_protocol_wire(trajectory / "responses.protocol.gz"):
        raw = sc.ResponseObservation()
        raw.ParseFromString(wire)
        if terminal is not None:
            raise ValueError("response after terminal")
        if raw.player_result:
            terminal = int(raw.observation.game_loop)
    if terminal is None:
        raise ValueError("complete terminal observation required")
    plans = iter(
        schedule_targets(
            sorted(command_times),
            initial_loop=0,
            terminal_loop=terminal,
            maximum_delay=maximum_delay,
        )
    )
    plan = next(plans, None)
    scheduler = None
    body = None
    context = ExtractionContext(report["controlled_player_id"], "3.16.1", "REPLAY_PLAYER", False)
    episode = report["replay_sha256"] + ":player" + str(context.player_id)
    with gzip.open(query_trajectory / "queries.jsonl.gz", "rt") as queries:

        def advance_query():
            line = queries.readline()
            return json.loads(line, object_hook=decode_bytes) if line else None

        query = advance_query()
        previous = -1
        for wire in read_protocol_wire(trajectory / "responses.protocol.gz"):
            raw = sc.ResponseObservation()
            raw.ParseFromString(wire)
            loop = int(raw.observation.game_loop)
            if loop != previous + 1:
                raise ValueError("incomplete one-loop trajectory")
            previous = loop
            while query is not None and query["loop"] < loop:
                query = advance_query()

            def frame():
                if (
                    query is None
                    or query["loop"] != loop
                    or query["observation_sha256"] != hashlib.sha256(wire).hexdigest()
                ):
                    raise ValueError("decision frame lacks exact same-run query")
                item = next(q for q in query["queries"] if q["mode"] == "default")
                req, rep = query_pb2.RequestQuery(), query_pb2.ResponseQuery()
                req.ParseFromString(item["request"])
                rep.ParseFromString(item["response"])
                controls = (query["absent_control"],) + (
                    (query["gone_control"],) if query["gone_control"] is not None else ()
                )
                receipt = BoundQuery.bind(raw.observation, req, rep, control_tags=controls)
                return ActionFrame.from_protocol(
                    raw, info, profile.catalog, receipt, context=context, episode=episode
                )

            def scheduler_frame():
                return SchedulerFrame.from_protocol(
                    raw, info, profile.catalog, context=context, episode=episode
                )

            if plan is None:
                continue
            if loop == plan.scheduler_loop:
                scheduler = scheduler_frame()
                if plan.decision == "WAIT":
                    yield BCDecision(report["replay_sha256"], scheduler, "WAIT", plan.delay)
                    plan = next(plans, None)
                    scheduler = None
            if plan is not None and plan.decision == "ACT" and loop == plan.body_loop:
                body = frame()
            if plan is not None and plan.decision == "ACT" and loop == plan.body_loop + 1:
                targets = tuple(
                    RawCommandTarget.from_protocol(a, body)
                    for a in raw.actions
                    if a.HasField("action_raw") and a.action_raw.HasField("unit_command")
                )
                events = native.get(plan.body_loop, ())
                data = [e for e in events if "Data" in e.get("m_data", {})]
                converted = sum(
                    t.ability in profile.catalog.remaps
                    and profile.ability(t.ability)
                    != canonical_ability(t.ability, profile.catalog.remaps)
                    for t in targets
                )
                rejection = (
                    "NATIVE_COMMAND_WITHOUT_UNIT_ECHO"
                    if not targets
                    else ("UNQUALIFIED_NATIVE_DATA_FAMILY" if len(data) > converted else None)
                )
                original = tuple(
                    json.dumps(e, sort_keys=True, default=encode_bytes).encode() for e in events
                )
                yield BCDecision(
                    report["replay_sha256"],
                    scheduler,
                    "ACT",
                    plan.delay,
                    body,
                    targets,
                    body_rejection=rejection,
                    native_event_wires=original,
                )
                plan = next(plans, None)
                scheduler = None
                body = None
                if plan is not None and loop == plan.scheduler_loop:
                    scheduler = scheduler_frame()
                    if plan.decision == "WAIT":
                        yield BCDecision(report["replay_sha256"], scheduler, "WAIT", plan.delay)
                        plan = next(plans, None)
                        scheduler = None
                    elif loop == plan.body_loop:
                        body = frame()
        if plan is not None:
            raise ValueError("trajectory did not exhaust every scheduled command")
        # Exhaust query gzip too, including its CRC, without exposing future rows to the model.
        while query is not None:
            query = advance_query()
