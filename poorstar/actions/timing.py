"""Schedule structured commands against their causal observation frames."""

from dataclasses import dataclass, field

from poorstar.actions.frame import ActionFrame, BoundQuery, FrameKey


def _current_observation_matches(frame, observation):
    """Exact current state, allowing only a consumed, non-actor death-event list.

    SC2 can clear dead_units on a same-loop repeat. Never discard those events from
    the original frame/query wire, nor accept new events or accompanying state changes.
    """
    from s2clientprotocol import sc2api_pb2 as sc

    from poorstar.actions.frame import wire_hash

    expected = sc.Observation()
    expected.ParseFromString(frame._observation_wire)
    if wire_hash(expected) != frame.key.observation_sha256:
        raise ValueError("action frame observation hash is corrupt")
    if wire_hash(observation) == frame.key.observation_sha256:
        return True
    if not expected.raw_data.event.dead_units or observation.raw_data.event.dead_units:
        return False
    current = sc.Observation()
    current.CopyFrom(observation)
    expected.raw_data.event.ClearField("dead_units")
    # Serialize instead of ListFields so undeclared/unknown event fields stay present.
    for value in (expected, current):
        if value.raw_data.HasField("event") and not value.raw_data.event.SerializeToString():
            value.raw_data.ClearField("event")
    return wire_hash(expected) == wire_hash(current)


@dataclass(frozen=True)
class ScheduleTarget:
    scheduler_loop: int
    decision: str
    delay: int
    body_loop: int | None


def schedule_targets(command_loops, *, initial_loop, terminal_loop, maximum_delay):
    """Exhaust every distinct command time; long gaps become WAIT segments, not dropped actions.

    Commands at one time are an uncapped ordered bundle handled by the body contract.
    Future command times are labels only; this function returns no future actor features.
    """
    numbers = (initial_loop, terminal_loop, maximum_delay)
    if (
        any(type(n) is not int for n in numbers)
        or initial_loop < 0
        or terminal_loop < initial_loop
        or maximum_delay < 1
    ):
        raise ValueError("explicit valid interval and positive scheduling horizon required")
    loops = tuple(command_loops)
    if any(type(n) is not int or not initial_loop <= n < terminal_loop for n in loops):
        raise ValueError("command outside nonterminal observation interval")
    if tuple(sorted(set(loops))) != loops:
        raise ValueError("distinct ordered command times required; preserve bundles separately")
    cursor = initial_loop
    for loop in loops:
        while loop - cursor > maximum_delay:
            yield ScheduleTarget(cursor, "WAIT", maximum_delay, None)
            cursor += maximum_delay
        yield ScheduleTarget(cursor, "ACT", loop - cursor, loop)
        # Advance only one loop so a following command inside the old grid cannot be lost.
        cursor = loop + 1
    while cursor < terminal_loop:
        delay = min(maximum_delay, terminal_loop - cursor)
        yield ScheduleTarget(cursor, "WAIT", delay, None)
        cursor += delay


@dataclass(frozen=True)
class IntervalBinding:
    body_frame: FrameKey
    response_loop: int
    protocol_action_loop: int | None
    clock_evidence_id: str

    @classmethod
    def one_step(cls, frame, response, action, *, clock_evidence_id):
        if frame.terminal:
            raise ValueError("no command interval starts at terminal")
        if not isinstance(clock_evidence_id, str) or not clock_evidence_id:
            raise ValueError("declared measured clock evidence required")
        obs = response.observation
        if (
            obs.player_common.player_id != frame.key.player_id
            or obs.game_loop != frame.key.loop + 1
        ):
            raise ValueError("response is not the next same-player one-step observation")
        original = int(action.game_loop) if action.HasField("game_loop") else None
        return cls(frame.key, int(obs.game_loop), original, clock_evidence_id)


@dataclass(frozen=True)
class PreparedBody:
    scheduler_frame: FrameKey
    body_frame: ActionFrame | None
    elapsed_loops: int
    termination_loop: int | None = None
    requested_delay: int | None = None
    scheduler_check_wire: bytes | None = field(default=None, repr=False)


def prepare_live_body(
    controller, scheduler_frame, *, delay, maximum_delay, context, episode, game_info, catalog
):
    """Call BEFORE body sampling, never decode pointers then advance with old slots/masks."""
    from s2clientprotocol import query_pb2

    if (
        type(delay) is not int
        or type(maximum_delay) is not int
        or maximum_delay < 1
        or not 0 <= delay <= maximum_delay
    ):
        raise ValueError("invalid sampled delay")
    if scheduler_frame.key.episode != episode or scheduler_frame.key.player_id != context.player_id:
        raise ValueError("scheduler episode/player identity mismatch")
    if scheduler_frame.terminal:
        raise ValueError("terminal scheduler frame")
    before = controller.observe()
    if (
        before.player_result
        or before.observation.game_loop != scheduler_frame.key.loop
        or not _current_observation_matches(scheduler_frame, before.observation)
    ):
        raise ValueError("scheduler is stale before advancing sampled delay")
    if delay:
        controller.step(delay)
    fresh = controller.observe()
    if fresh.observation.player_common.player_id != context.player_id:
        raise ValueError("controlled player changed during sampled delay")
    if fresh.player_result:
        elapsed = int(fresh.observation.game_loop) - scheduler_frame.key.loop
        if not 0 <= elapsed <= delay:
            raise ValueError("terminal elapsed loops outside the sampled interval")
        return PreparedBody(
            scheduler_frame.key,
            None,
            elapsed,
            int(fresh.observation.game_loop),
            delay,
            before.SerializeToString(),
        )
    if fresh.observation.game_loop != scheduler_frame.key.loop + delay:
        raise ValueError("actual elapsed loops differ from sampled delay")
    request = query_pb2.RequestQuery()
    for tag in sorted(int(u.tag) for u in fresh.observation.raw_data.units if u.alliance == 1):
        request.abilities.add(unit_tag=tag)
    reply = controller.query(request)
    receipt = BoundQuery.bind(fresh.observation, request, reply)
    frame = ActionFrame.from_protocol(
        fresh, game_info, catalog, receipt, context=context, episode=episode
    )
    return PreparedBody(
        scheduler_frame.key,
        frame,
        delay,
        requested_delay=delay,
        scheduler_check_wire=before.SerializeToString(),
    )


@dataclass(frozen=True)
class DispatchReceipt:
    body_frame: FrameKey
    command_count: int
    request_wire: bytes
    response_wire: bytes
    results: tuple[int, ...]
    body_check_wire: bytes | None = field(default=None, repr=False)

    @property
    def all_accepted(self):
        return bool(self.results) and all(r == 1 for r in self.results)

    @property
    def accepted_count(self):
        return sum(r == 1 for r in self.results)


def dispatch_live_bundle(controller, prepared: PreparedBody, machine):
    """Dispatch the entire shared-machine bundle; never rename acceptance to effect."""
    from s2clientprotocol import sc2api_pb2 as sc

    frame = prepared.body_frame
    if frame is None:
        raise ValueError("episode terminated before body; dispatch is forbidden")
    if machine.frame.key != frame.key:
        raise ValueError("body machine belongs to another prepared observation/episode")
    actions = machine.protocol_actions()
    current = controller.observe()
    if current.player_result or not _current_observation_matches(frame, current.observation):
        raise ValueError("body observation changed before dispatch; stale arguments rejected")
    request = sc.RequestAction(actions=actions)
    response = controller.actions(request)
    if response is None or len(response.result) != len(actions):
        raise ValueError("action RPC result denominator differs from the complete bundle")
    return DispatchReceipt(
        frame.key,
        len(actions),
        request.SerializeToString(),
        response.SerializeToString(),
        tuple(int(r) for r in response.result),
        current.SerializeToString(),
    )


@dataclass(frozen=True)
class ScheduledDecisionReceipt:
    decision: str
    prepared: PreparedBody
    dispatch: DispatchReceipt | None
    next_frame: ActionFrame | None = None
    post_dispatch_elapsed_loops: int = 0
    post_dispatch_termination_loop: int | None = None
    post_dispatch_response_wire: bytes | None = field(default=None, repr=False)

    @property
    def elapsed_loops(self):
        return self.prepared.elapsed_loops + self.post_dispatch_elapsed_loops

    @property
    def terminated(self):
        return (
            self.prepared.termination_loop is not None
            or self.post_dispatch_termination_loop is not None
        )


def _next_frame_after_dispatch(controller, body_frame, *, context, episode, game_info, catalog):
    """Match schedule_targets' one-loop ACT interval, then query the actual next frame.

    The RPC result is acceptance only. The next response wire is retained for effects/
    errors and previous-action feedback; it is not silently added to actor features.
    """
    from s2clientprotocol import query_pb2

    controller.step(1)
    response = controller.observe()
    obs = response.observation
    if obs.player_common.player_id != context.player_id:
        raise ValueError("controlled player changed after action dispatch")
    elapsed = int(obs.game_loop) - body_frame.key.loop
    if response.player_result:
        if not 0 <= elapsed <= 1:
            raise ValueError("terminal elapsed loops outside the action interval")
        return None, elapsed, int(obs.game_loop), response.SerializeToString()
    if elapsed != 1:
        raise ValueError("post-dispatch response is not the next one-step observation")
    request = query_pb2.RequestQuery()
    for tag in sorted(int(u.tag) for u in obs.raw_data.units if u.alliance == 1):
        request.abilities.add(unit_tag=tag)
    receipt = BoundQuery.bind(obs, request, controller.query(request))
    frame = ActionFrame.from_protocol(
        response, game_info, catalog, receipt, context=context, episode=episode
    )
    return frame, elapsed, None, response.SerializeToString()


def run_scheduled_decision(
    controller,
    scheduler_frame,
    *,
    choose_schedule,
    choose_body,
    maximum_delay,
    context,
    episode,
    game_info,
    catalog,
    body_sampler=None,
):
    """Operational entry point: the two policy callbacks never receive future labels/state.

    Recurrent/Intent/Task clients advance age using receipt.elapsed_loops and reset on an
    episode change; this lower-level entry point does not replace their persistent state.
    """
    from poorstar.actions.contract import sample

    payload = scheduler_frame.policy_observation()
    decision, delay = choose_schedule(payload, maximum_delay)
    if decision not in ("ACT", "WAIT") or (
        decision == "WAIT" and (type(delay) is not int or delay < 1)
    ):
        raise ValueError("invalid ACT/WAIT schedule; WAIT must advance time")
    prepared = prepare_live_body(
        controller,
        scheduler_frame,
        delay=delay,
        maximum_delay=maximum_delay,
        context=context,
        episode=episode,
        game_info=game_info,
        catalog=catalog,
    )
    if prepared.body_frame is None:
        return ScheduledDecisionReceipt(decision, prepared, None)
    if decision == "WAIT":
        return ScheduledDecisionReceipt(decision, prepared, None, prepared.body_frame)
    body_payload = prepared.body_frame.policy_observation()
    machine = (body_sampler or sample)(
        prepared.body_frame,
        lambda head, options, completed: choose_body(body_payload, head, options, completed),
    )
    dispatch = dispatch_live_bundle(controller, prepared, machine)
    next_frame, elapsed, terminal, wire = _next_frame_after_dispatch(
        controller,
        prepared.body_frame,
        context=context,
        episode=episode,
        game_info=game_info,
        catalog=catalog,
    )
    return ScheduledDecisionReceipt(
        decision, prepared, dispatch, next_frame, elapsed, terminal, wire
    )
