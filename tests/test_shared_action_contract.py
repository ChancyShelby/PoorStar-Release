from dataclasses import replace
from types import SimpleNamespace

import pytest
from s2clientprotocol import query_pb2
from s2clientprotocol import sc2api_pb2 as sc

from poorstar.actions.contract import (
    BUNDLE_NEXT,
    BUNDLE_STOP,
    SELECT_END,
    CompleteRequestTarget,
    PrefixMachine,
    RawCommandTarget,
    Token,
    sample,
    teacher_force,
)
from poorstar.actions.frame import ActionFrame, ArgumentReferences, BoundQuery, EngineActionCatalog
from poorstar.actions.timing import (
    IntervalBinding,
    dispatch_live_bundle,
    prepare_live_body,
    run_scheduled_decision,
    schedule_targets,
)
from poorstar.engine.observation import ExtractionContext


def fixture(*, loop=0, tags=(10, 20), available=None, episode="episode", source="REPLAY_PLAYER"):
    raw = sc.ResponseObservation()
    obs = raw.observation
    obs.game_loop = loop
    obs.player_common.player_id = 1
    obs.player_common.minerals = 50
    obs.player_common.larva_count = 3
    for tag in tags:
        u = obs.raw_data.units.add(tag=tag, alliance=1, display_type=1, unit_type=100)
        u.pos.x = 5
        u.pos.y = 7
    info = sc.ResponseGameInfo()
    info.start_raw.map_size.x = 200
    info.start_raw.map_size.y = 160
    cat = sc.ResponseData()
    cat.units.add(unit_id=100, name="Larva")
    for a, target in ((11, 1), (21, 2), (31, 3), (41, 4), (51, 5)):
        cat.abilities.add(ability_id=a, target=target)
    cat.abilities.add(ability_id=12, target=1, remaps_to_ability_id=11)
    q, r = query_pb2.RequestQuery(), query_pb2.ResponseQuery()
    for tag in sorted(tags):
        q.abilities.add(unit_tag=tag)
        ans = r.abilities.add(unit_tag=tag, unit_type_id=100)
        for a in (available if available is not None else {}).get(tag, (11, 21, 31, 41, 51)):
            ans.abilities.add(ability_id=a)
    receipt = BoundQuery.bind(obs, q, r)
    context = ExtractionContext(1, "3.16.1", source, False)
    frame = ActionFrame.from_protocol(raw, info, cat, receipt, context=context, episode=episode)
    return frame, raw, info, cat, q, r, context


def command(frame, *, ability=11, tags=None, queue=False, point=None, target=None, timestamp=None):
    action = sc.Action()
    c = action.action_raw.unit_command
    c.ability_id = ability
    c.unit_tags.extend(frame.slot_tags if tags is None else tags)
    if queue is not None:
        c.queue_command = queue
    if point is not None:
        c.target_world_space_pos.x, c.target_world_space_pos.y = point
    if target is not None:
        c.target_unit_tag = target
    if timestamp is not None:
        action.game_loop = timestamp
    return action


@pytest.mark.parametrize("queue", [None, False, True])
@pytest.mark.parametrize("mode", ["none", "entity", "position"])
def test_complete_wire_roundtrip_preserves_optional_queue_and_timestamp(queue, mode):
    f, *_ = fixture()
    a = command(
        f,
        ability={"none": 11, "entity": 31, "position": 21}[mode],
        queue=queue,
        point=(24.125, 87.5) if mode == "position" else None,
        target=f.slot_tags[0] if mode == "entity" else None,
        timestamp=0,
    )
    target = RawCommandTarget.from_protocol(a, f)
    assert target.protocol(f).SerializeToString() == a.SerializeToString()
    assert target.protocol_game_loop == 0 and target.queue is queue
    machine, _ = teacher_force(f, (target,))
    sampled = machine.protocol_actions()[0]
    assert sampled.action_raw.unit_command.queue_command == bool(queue)
    assert not sampled.HasField("game_loop")  # no fabricated replay timestamp in a live request


def test_sixty_six_sources_and_thirty_commands_are_not_capped():
    f, *_ = fixture(tags=tuple(range(1, 67)))
    targets = tuple(
        RawCommandTarget.from_protocol(command(f, queue=bool(i % 2)), f) for i in range(30)
    )
    machine, records = teacher_force(f, targets)
    assert len(machine.commands) == 30
    assert all(len(c[1]) == 66 for c in machine.commands)
    assert sum(r[2] == SELECT_END for r in records) == 30
    assert sum(r[2] == BUNDLE_STOP for r in records) == 1
    assert sum(r[2] == BUNDLE_NEXT for r in records) == 29


def test_more_than_512_entities_keeps_every_source_and_target():
    f, *_ = fixture(tags=tuple(range(1, 514)))
    target = RawCommandTarget.from_protocol(command(f, target=513, ability=31), f)
    payload, _ = f.actor_candidate_payload()
    assert len(f.slot_tags) == len(payload["entities"]) == 513 and payload["overflow"] == 0
    assert len(target.source_slots) == 513 and target.target_slot == 512


def test_teacher_forcing_and_sampling_traverse_identical_prefix_domains():
    f, *_ = fixture()
    targets = tuple(
        RawCommandTarget.from_protocol(command(f, ability=41, point=(1.25, 2.5)), f)
        for _ in range(2)
    )
    teacher, rows = teacher_force(f, targets)
    cursor = iter(rows)

    def choose(head, options, completed):
        wanted_head, wanted_options, value = next(cursor)
        assert (head, options) == (wanted_head, wanted_options)
        return value

    sampled = sample(f, choose)
    assert sampled.commands == teacher.commands
    assert [a.SerializeToString() for a in sampled.protocol_actions()] == [
        a.SerializeToString() for a in teacher.protocol_actions()
    ]


def test_strict_validator_accepts_full_bundle_and_preserves_declared_rejection():
    from poorstar.actions.validation import validate_shared_bundle

    f, *_ = fixture()
    targets = tuple(RawCommandTarget.from_protocol(command(f), f) for _ in range(30))
    verdict = validate_shared_bundle(f, targets)
    assert verdict.status == "SHARED_PREFIX_EQUAL"
    assert len(verdict.sampled.commands) == 30
    missing = RawCommandTarget.from_protocol(command(f, tags=(999,)), f)
    verdict = validate_shared_bundle(f, (missing,))
    assert verdict.status == "TARGET_DOMAIN_REJECTED"
    assert verdict.rejection_code == "UNREPRESENTABLE_ARGUMENT"
    assert verdict.rejection_issues == ("UNOBSERVABLE_SOURCE_REFERENCE",)
    unavailable = RawCommandTarget.from_protocol(command(f, ability=999), f)
    assert (
        validate_shared_bundle(f, (unavailable,)).rejection_code == "TOKEN_OUTSIDE_CURRENT_DOMAIN"
    )


@pytest.mark.parametrize(
    "message",
    [
        "UNEXPECTED_INTERNAL_VALIDATOR_BUG",
        "token outside shared current prefix domain",
        "unrepresentable complete command: UNOBSERVABLE_SOURCE_REFERENCE",
    ],
)
def test_strict_validator_never_classifies_unknown_valueerror_by_message(monkeypatch, message):
    from poorstar.actions import validation

    f, *_ = fixture()
    target = RawCommandTarget.from_protocol(command(f), f)

    def broken_teacher(*args):
        raise ValueError(message)

    monkeypatch.setattr(validation, "teacher_force", broken_teacher)
    with pytest.raises(ValueError, match=message):
        validation.validate_shared_bundle(f, (target,))


@pytest.mark.parametrize("failure", ["stale", "empty", "unknown_issue"])
def test_strict_validator_contract_faults_are_hard_failures(failure):
    from poorstar.actions.validation import validate_shared_bundle

    f, *_ = fixture()
    target = RawCommandTarget.from_protocol(command(f), f)
    targets = (target,)
    if failure == "stale":
        targets = (replace(target, frame_key=replace(f.key, episode="other")),)
    if failure == "empty":
        targets = ()
    if failure == "unknown_issue":
        targets = (replace(target, issues=("INTERNAL_BUG",)),)
    with pytest.raises(ValueError):
        validate_shared_bundle(f, targets)


@pytest.mark.parametrize(
    "fault", ["domain", "command", "wire", "early_stop", "extra_token", "prefix"]
)
def test_fault_injection_runs_through_actual_strict_comparison(monkeypatch, fault):
    from poorstar.actions import validation
    from poorstar.actions.contract import TargetDomainRejection

    f, *_ = fixture()
    target = RawCommandTarget.from_protocol(command(f), f)
    real_sample = validation.sample

    def broken_sampler(frame, choose):
        if fault == "domain":
            raise TargetDomainRejection(
                "TOKEN_OUTSIDE_CURRENT_DOMAIN", "sampler selected illegal token"
            )
        if fault == "early_stop":
            return SimpleNamespace(commands=[])
        if fault == "prefix":
            choose("source", (), ())
        sampled = real_sample(frame, choose)
        if fault == "extra_token":
            choose("ability", (), ())
        if fault == "command":
            sampled.commands = []
        if fault == "wire":
            changed = sampled.protocol_actions()[0]
            changed.action_raw.unit_command.queue_command = True
            sampled.protocol_actions = lambda: (changed,)
        return sampled

    monkeypatch.setattr(validation, "sample", broken_sampler)
    expected = TargetDomainRejection if fault == "domain" else validation.SharedActionParityError
    with pytest.raises(expected):
        validation.validate_shared_bundle(f, (target,))


def test_selected_any_keeps_mixed_sources_but_does_not_allow_empty_or_unavailable_end():
    f, *_ = fixture(available={10: (11,), 20: ()})
    m = PrefixMachine(f)
    m.accept(Token("ability", 11))
    assert SELECT_END not in m.options()
    m.accept(Token("source", 1))
    assert SELECT_END not in m.options()
    m.accept(Token("source", 0))
    assert SELECT_END in m.options()
    m.accept(Token("source", SELECT_END))
    m.accept(Token("target_mode", "none"))
    m.accept(Token("queue", False))
    m.accept(Token("bundle", BUNDLE_STOP))
    assert tuple(m.protocol_actions()[0].action_raw.unit_command.unit_tags) == (20, 10)


@pytest.mark.parametrize("token", [Token("source", 0), Token("source", True), Token("source", 999)])
def test_duplicate_noninteger_and_invalid_sources_fail_shared_machine(token):
    f, *_ = fixture()
    m = PrefixMachine(f)
    m.accept(Token("ability", 11))
    m.accept(Token("source", 0))
    with pytest.raises(ValueError):
        m.accept(token)


def test_same_loop_different_episode_or_query_cannot_reuse_pointer_targets():
    f, *_ = fixture()
    target = RawCommandTarget.from_protocol(command(f), f)
    for key in (replace(f.key, episode="other"), replace(f.key, query_sha256="different")):
        other = replace(f, key=key)
        with pytest.raises(ValueError, match="stale"):
            target.protocol(other)
        with pytest.raises(ValueError, match="stale"):
            teacher_force(other, (target,))


def test_unknown_source_and_hidden_target_stay_complete_explicit_failures():
    f, *_ = fixture()
    a = command(f, tags=(10, 999), ability=31, target=888)
    target = RawCommandTarget.from_protocol(a, f)
    assert len(target.source_slots) == 2 and target.source_slots[1] is None
    assert target.original_wire == a.SerializeToString()
    assert set(target.issues) == {"UNOBSERVABLE_SOURCE_REFERENCE", "UNOBSERVABLE_ENTITY_TARGET"}
    with pytest.raises(ValueError, match="unrepresentable"):
        teacher_force(f, (target,))


def test_current_query_cannot_be_replaced_by_future_frame_or_ignore_resource_query():
    f, raw, info, cat, q, r, context = fixture()
    receipt = BoundQuery.bind(raw.observation, q, r)
    raw.observation.game_loop += 1
    with pytest.raises(ValueError, match="phase"):
        ActionFrame.from_protocol(raw, info, cat, receipt, context=context, episode="episode")
    q.ignore_resource_requirements = True
    receipt = BoundQuery.bind(raw.observation, q, r)
    with pytest.raises(ValueError, match="audit-only"):
        ActionFrame.from_protocol(raw, info, cat, receipt, context=context, episode="episode")


def test_actions_results_ui_selection_and_hud_larva_are_not_actor_features():
    f, raw, info, cat, q, r, context = fixture()
    expected, pixels = f.actor_candidate_payload()
    raw.observation.player_common.larva_count = 999
    raw.observation.raw_data.units[0].is_selected = True
    raw.observation.ui_data.single.unit.unit_type = 777
    raw.actions.add().CopyFrom(command(f))
    raw.action_errors.add(result=9)
    receipt = BoundQuery.bind(raw.observation, q, r)
    other = ActionFrame.from_protocol(raw, info, cat, receipt, context=context, episode="episode")
    actual, other_pixels = other.actor_candidate_payload()
    assert expected == actual and pixels.keys() == other_pixels.keys()
    assert "larva_count" not in actual["scalars"] and "larva_count" not in actual["scalar_present"]


def test_scheduler_preserves_every_command_inside_one_legacy_window():
    rows = list(schedule_targets((8, 9, 10, 25), initial_loop=0, terminal_loop=30, maximum_delay=8))
    assert [r.body_loop for r in rows if r.decision == "ACT"] == [8, 9, 10, 25]
    assert all(0 <= r.delay <= 8 for r in rows)
    assert rows[-1].scheduler_loop + rows[-1].delay == 30
    assert next(r for r in rows if r.body_loop == 9).delay == 0


@pytest.mark.parametrize("loops", [(1, 1), (2, 1), (-1,), (10,), (True,)])
def test_scheduler_does_not_sort_dedup_or_drop_invalid_times(loops):
    with pytest.raises(ValueError):
        list(schedule_targets(loops, initial_loop=0, terminal_loop=10, maximum_delay=8))


def test_interval_keeps_missing_action_timestamp_missing():
    f, *_ = fixture(loop=8)
    _, response, *_ = fixture(loop=9)
    action = command(f)
    binding = IntervalBinding.one_step(f, response, action, clock_evidence_id="measured-build55958")
    assert binding.body_frame.loop == 8 and binding.response_loop == 9
    assert binding.protocol_action_loop is None and not action.HasField("game_loop")
    response.observation.game_loop = 10
    with pytest.raises(ValueError, match="next"):
        IntervalBinding.one_step(f, response, action, clock_evidence_id="x")


def test_live_delay_refresh_replaces_morphed_sources_and_query_before_body_decoding():
    start, before, info, cat, q, r, context = fixture(tags=(10,), source="LIVE_PLAYER")
    fresh, after, _, _, fq, fr, _ = fixture(tags=(20,), loop=3, source="LIVE_PLAYER")
    state = {"raw": before, "steps": []}

    def step(n):
        state["steps"].append(n)
        state["raw"] = after

    ctl = SimpleNamespace(observe=lambda: state["raw"], step=step, query=lambda request: fr)
    prepared = prepare_live_body(
        ctl,
        start,
        delay=3,
        maximum_delay=8,
        context=context,
        episode="episode",
        game_info=info,
        catalog=cat,
    )
    assert prepared.elapsed_loops == 3 and prepared.body_frame.slot_tags == (20,)
    assert prepared.scheduler_frame.loop == 0 and prepared.body_frame.key.loop == 3
    old = RawCommandTarget.from_protocol(command(start), start)
    with pytest.raises(ValueError, match="stale"):
        teacher_force(prepared.body_frame, (old,))


@pytest.mark.parametrize("delay", [-1, 9, True])
def test_live_scheduler_rejects_invalid_delay_before_controller_calls(delay):
    f, _, info, cat, _, _, context = fixture()
    with pytest.raises(ValueError, match="delay"):
        prepare_live_body(
            None,
            f,
            delay=delay,
            maximum_delay=8,
            context=context,
            episode="episode",
            game_info=info,
            catalog=cat,
        )


def test_live_dispatch_sends_all_commands_and_keeps_result_denominator():
    f, raw, info, cat, q, r, context = fixture(source="LIVE_PLAYER")
    sent = []

    def actions(request):
        sent.append(request)
        return sc.ResponseAction(result=[1] * len(request.actions))

    ctl = SimpleNamespace(observe=lambda: raw, query=lambda request: r, actions=actions)
    prepared = prepare_live_body(
        ctl,
        f,
        delay=0,
        maximum_delay=8,
        context=context,
        episode="episode",
        game_info=info,
        catalog=cat,
    )
    body = prepared.body_frame
    machine, _ = teacher_force(
        body, tuple(RawCommandTarget.from_protocol(command(body), body) for _ in range(30))
    )
    receipt = dispatch_live_bundle(ctl, prepared, machine)
    assert receipt.command_count == len(sent[0].actions) == len(receipt.results) == 30
    raw.observation.player_common.minerals = 49
    with pytest.raises(ValueError, match="stale"):
        dispatch_live_bundle(ctl, prepared, machine)
    assert len(sent) == 1


def test_native_data_cannot_become_none_even_when_engine_echo_omits_argument():
    import json

    f, *_ = fixture()
    raw = RawCommandTarget.from_protocol(command(f), f)
    e = {"_gameloop": 0, "m_data": {"Data": 86769666}, "m_sequence": 42}
    target = CompleteRequestTarget.retain_native(raw, e)
    assert target.raw.target_mode == "none" and target.native_target_kind == "Data"
    assert json.loads(target.native_event_wire)["m_data"] == {"Data": 86769666}
    with pytest.raises(ValueError, match="operation resolver"):
        target.tokens()
    with pytest.raises(ValueError, match="operation resolver"):
        teacher_force(f, (target,))


def test_reference_only_frame_can_roundtrip_wire_but_never_supply_sampling_masks():
    f, raw, info, _, _, _, context = fixture()
    refs = ArgumentReferences.from_protocol(raw, info, context=context, episode="episode")
    action = command(f)
    target = RawCommandTarget.from_protocol(action, refs)
    assert target.protocol(refs) == action and refs.key.query_sha256 == "QUERY_NOT_COLLECTED"
    with pytest.raises(ValueError, match="query-qualified"):
        PrefixMachine(refs)


def test_unknown_protobuf_fields_cannot_be_lost_in_complete_codec():
    f, *_ = fixture()
    a = command(f)

    a.ParseFromString(a.SerializeToString() + b"\xa0\x06\x01")
    with pytest.raises(ValueError, match="unknown protocol"):
        RawCommandTarget.from_protocol(a, f)


def test_absent_query_controls_do_not_become_source_or_ability_candidates():
    f, raw, info, cat, q, r, context = fixture()
    q.abilities.add(unit_tag=999)
    ans = r.abilities.add(unit_tag=999)
    ans.abilities.add(ability_id=51)
    receipt = BoundQuery.bind(raw.observation, q, r, control_tags=(999,))
    other = ActionFrame.from_protocol(raw, info, cat, receipt, context=context, episode="episode")
    assert other.slot_tags == f.slot_tags and other.ability_options() == f.ability_options()


def test_actual_engine_null_catalog_placeholder_is_retained_but_never_executable():
    f, raw, info, cat, q, r, context = fixture()
    cat.abilities.add(ability_id=0, link_name="Null")
    profile = EngineActionCatalog.from_protocol(cat)
    assert 0 in profile.targets and 0 in profile.remaps
    frame = ActionFrame.from_protocol(
        raw,
        info,
        profile,
        BoundQuery.bind(raw.observation, q, r),
        context=context,
        episode="episode",
    )
    assert 0 not in frame.ability_options() and not frame.source_can_issue(0, 0)
    cat.abilities[-1].link_name = "invented"
    with pytest.raises(ValueError, match="Null placeholder"):
        EngineActionCatalog.from_protocol(cat)


@pytest.mark.parametrize("bad", ["absent_x", "nonfinite_y"])
def test_reference_position_presence_and_finiteness_cannot_be_guessed(bad):
    _, raw, info, _, _, _, context = fixture()
    if bad == "absent_x":
        raw.observation.raw_data.units[0].pos.ClearField("x")
    else:
        raw.observation.raw_data.units[0].pos.y = float("inf")
    with pytest.raises(ValueError):
        ArgumentReferences.from_protocol(raw, info, context=context, episode="episode")


def test_operational_wait_and_act_callbacks_receive_only_their_current_phase():
    start, before, info, cat, _, _, context = fixture(source="LIVE_PLAYER")
    _, after, _, _, _, reply, _ = fixture(loop=3, tags=(20,), source="LIVE_PLAYER")
    state = {"raw": before, "requests": [], "scheduler_loops": [], "body_loops": []}

    def step(n):
        _, state["raw"], *_ = fixture(
            loop=state["raw"].observation.game_loop + n, tags=(20,), source="LIVE_PLAYER"
        )

    def schedule(payload, horizon):
        state["scheduler_loops"].append(payload["observation_loop"])
        return ("ACT", 3)

    def body(payload, head, options, completed):
        state["body_loops"].append(payload["observation_loop"])
        return {
            "ability": 11,
            "source": 0 if 0 in options else SELECT_END,
            "target_mode": "none",
            "queue": False,
            "bundle": BUNDLE_STOP,
        }[head]

    def actions(request):
        state["requests"].append(request)
        return sc.ResponseAction(result=[1] * len(request.actions))

    ctl = SimpleNamespace(
        observe=lambda: state["raw"], step=step, query=lambda request: reply, actions=actions
    )
    receipt = run_scheduled_decision(
        ctl,
        start,
        choose_schedule=schedule,
        choose_body=body,
        maximum_delay=8,
        context=context,
        episode="episode",
        game_info=info,
        catalog=cat,
    )
    assert state["scheduler_loops"] == [0] and set(state["body_loops"]) == {3}
    assert receipt.dispatch.command_count == 1 and state["requests"][0].actions[
        0
    ].action_raw.unit_command.unit_tags == [20]
    assert receipt.elapsed_loops == 4 and receipt.post_dispatch_elapsed_loops == 1
    assert receipt.next_frame.key.loop == 4
    response = sc.ResponseObservation()
    response.ParseFromString(receipt.post_dispatch_response_wire)
    assert response.observation.game_loop == 4
    state["raw"] = before
    state["requests"].clear()
    wait = run_scheduled_decision(
        ctl,
        start,
        choose_schedule=lambda payload, horizon: ("WAIT", 3),
        choose_body=lambda *args: pytest.fail("WAIT must not sample a body"),
        maximum_delay=8,
        context=context,
        episode="episode",
        game_info=info,
        catalog=cat,
    )
    assert wait.dispatch is None and not state["requests"] and wait.prepared.elapsed_loops == 3
    assert wait.elapsed_loops == 3 and wait.next_frame.key.loop == 3


def test_replay_and_live_callbacks_receive_equal_actor_inputs_not_a_source_indicator():
    replay, *_ = fixture(source="REPLAY_PLAYER")
    live, *_ = fixture(source="LIVE_PLAYER")
    a, ar = replay.actor_candidate_payload()
    b, br = live.actor_candidate_payload()
    assert a == b and ar.keys() == br.keys() and "context" not in a


def test_online_scheduler_matches_training_cursor_for_consecutive_action_times():
    start, before, info, cat, _, reply, context = fixture(source="LIVE_PLAYER")
    state = {"raw": before, "steps": [], "bodies": []}

    def step(n):
        state["steps"].append(n)
        _, state["raw"], *_ = fixture(
            loop=state["raw"].observation.game_loop + n, source="LIVE_PLAYER"
        )

    ctl = SimpleNamespace(
        observe=lambda: state["raw"],
        step=step,
        query=lambda request: reply,
        actions=lambda request: sc.ResponseAction(result=[1] * len(request.actions)),
    )

    def body(payload, head, options, completed):
        state["bodies"].append(payload["observation_loop"])
        return {
            "ability": 11,
            "source": 0 if 0 in options else SELECT_END,
            "target_mode": "none",
            "queue": False,
            "bundle": BUNDLE_STOP,
        }[head]

    current = start
    elapsed = []
    for row in schedule_targets((8, 9, 10), initial_loop=0, terminal_loop=11, maximum_delay=8):
        assert current.key.loop == row.scheduler_loop
        receipt = run_scheduled_decision(
            ctl,
            current,
            choose_schedule=lambda p, h: ("ACT", row.delay),
            choose_body=body,
            maximum_delay=8,
            context=context,
            episode="episode",
            game_info=info,
            catalog=cat,
        )
        assert receipt.prepared.body_frame.key.loop == row.body_loop
        current = receipt.next_frame
        elapsed.append(receipt.elapsed_loops)
    assert current.key.loop == 11 and elapsed == [9, 1, 1]
    assert state["steps"] == [8, 1, 1, 1] and set(state["bodies"]) == {8, 9, 10}


def test_terminal_after_dispatch_has_actual_time_and_no_next_query():
    start, before, info, cat, _, reply, context = fixture(source="LIVE_PLAYER")
    state = {"raw": before, "queries": 0}

    def step(n):
        assert n == 1
        _, state["raw"], *_ = fixture(loop=1, source="LIVE_PLAYER")
        state["raw"].player_result.add(player_id=1, result=1)

    def query(request):
        state["queries"] += 1
        return reply

    ctl = SimpleNamespace(
        observe=lambda: state["raw"],
        step=step,
        query=query,
        actions=lambda request: sc.ResponseAction(result=[1] * len(request.actions)),
    )

    def body(payload, head, options, completed):
        return {
            "ability": 11,
            "source": 0 if 0 in options else SELECT_END,
            "target_mode": "none",
            "queue": False,
            "bundle": BUNDLE_STOP,
        }[head]

    r = run_scheduled_decision(
        ctl,
        start,
        choose_schedule=lambda p, h: ("ACT", 0),
        choose_body=body,
        maximum_delay=8,
        context=context,
        episode="episode",
        game_info=info,
        catalog=cat,
    )
    assert r.terminated and r.next_frame is None and r.elapsed_loops == 1
    assert r.post_dispatch_termination_loop == 1 and r.dispatch.command_count == 1
    assert state["queries"] == 1  # body only; no actor query after terminal


@pytest.mark.parametrize("fault", ["wrong_loop", "wrong_player"])
def test_post_dispatch_clock_or_player_fault_is_not_a_successful_decision(fault):
    start, before, info, cat, _, reply, context = fixture(source="LIVE_PLAYER")
    state = {"raw": before}

    def step(n):
        _, state["raw"], *_ = fixture(loop=2 if fault == "wrong_loop" else 1, source="LIVE_PLAYER")
        if fault == "wrong_player":
            state["raw"].observation.player_common.player_id = 2

    ctl = SimpleNamespace(
        observe=lambda: state["raw"],
        step=step,
        query=lambda request: reply,
        actions=lambda request: sc.ResponseAction(result=[1] * len(request.actions)),
    )

    def body(payload, head, options, completed):
        return {
            "ability": 11,
            "source": 0 if 0 in options else SELECT_END,
            "target_mode": "none",
            "queue": False,
            "bundle": BUNDLE_STOP,
        }[head]

    with pytest.raises(ValueError, match="post-dispatch|player changed"):
        run_scheduled_decision(
            ctl,
            start,
            choose_schedule=lambda p, h: ("ACT", 0),
            choose_body=body,
            maximum_delay=8,
            context=context,
            episode="episode",
            game_info=info,
            catalog=cat,
        )


def event_fixture():
    _, raw, info, cat, q, r, context = fixture(source="LIVE_PLAYER")
    raw.observation.raw_data.event.dead_units.append(999)
    f = ActionFrame.from_protocol(
        raw, info, cat, BoundQuery.bind(raw.observation, q, r), context=context, episode="episode"
    )
    current = sc.ResponseObservation()
    current.CopyFrom(raw)
    current.observation.raw_data.ClearField("event")
    return f, raw, current, info, cat, r, context


def test_scheduler_allows_only_consumed_death_event_and_retains_complete_guard_wire():
    f, original, current, info, cat, reply, context = event_fixture()
    ctl = SimpleNamespace(observe=lambda: current, query=lambda request: reply)
    prepared = prepare_live_body(
        ctl,
        f,
        delay=0,
        maximum_delay=8,
        context=context,
        episode="episode",
        game_info=info,
        catalog=cat,
    )
    guard = sc.ResponseObservation()
    guard.ParseFromString(prepared.scheduler_check_wire)
    assert not guard.observation.raw_data.event.dead_units
    assert list(original.observation.raw_data.event.dead_units) == [999]
    retained = sc.Observation()
    retained.ParseFromString(f._observation_wire)
    assert list(retained.raw_data.event.dead_units) == [999]
    assert prepared.body_frame.key.loop == 0 and prepared.body_frame.slot_tags == f.slot_tags
    assert (
        prepared.body_frame.policy_observation()["entities"] == f.policy_observation()["entities"]
    )


def test_dispatch_allows_consumed_death_event_without_changing_full_bundle():
    from poorstar.actions.timing import PreparedBody

    f, original, current, *_ = event_fixture()
    sent = []
    targets = tuple(RawCommandTarget.from_protocol(command(f), f) for _ in range(2))
    machine, _ = teacher_force(f, targets)

    def actions(request):
        sent.append(request)
        return sc.ResponseAction(result=[1, 1])

    receipt = dispatch_live_bundle(
        SimpleNamespace(observe=lambda: current, actions=actions),
        PreparedBody(f.key, f, 0),
        machine,
    )
    assert receipt.command_count == 2 and len(sent[0].actions) == 2
    guard = sc.ResponseObservation()
    guard.ParseFromString(receipt.body_check_wire)
    assert not guard.observation.raw_data.event.dead_units
    assert list(original.observation.raw_data.event.dead_units) == [999]


@pytest.mark.parametrize(
    "fault",
    [
        "new_event",
        "replacement_event",
        "health",
        "resource",
        "tag",
        "orders",
        "loop",
        "player",
        "ui",
    ],
)
def test_event_drain_never_hides_new_events_or_any_other_state_change(fault):
    from poorstar.actions.timing import PreparedBody, _current_observation_matches

    f, _, current, info, cat, reply, context = event_fixture()
    o = current.observation
    if fault == "new_event":
        o.raw_data.event.dead_units.extend((999, 1000))
    if fault == "replacement_event":
        o.raw_data.event.dead_units.append(1000)
    if fault == "health":
        o.raw_data.units[0].health = 5
    if fault == "resource":
        o.player_common.minerals += 1
    if fault == "tag":
        o.raw_data.units[0].tag = 1000
    if fault == "orders":
        o.raw_data.units[0].orders.add(ability_id=11)
    if fault == "loop":
        o.game_loop += 1
    if fault == "player":
        o.player_common.player_id = 2
    if fault == "ui":
        o.ui_data.single.unit.unit_type = 777
    assert not _current_observation_matches(f, o)
    ctl = SimpleNamespace(observe=lambda: current, query=lambda request: reply)
    with pytest.raises(ValueError, match="stale"):
        prepare_live_body(
            ctl,
            f,
            delay=0,
            maximum_delay=8,
            context=context,
            episode="episode",
            game_info=info,
            catalog=cat,
        )
    machine, _ = teacher_force(f, (RawCommandTarget.from_protocol(command(f), f),))
    with pytest.raises(ValueError, match="stale"):
        dispatch_live_bundle(ctl, PreparedBody(f.key, f, 0), machine)


def test_event_drain_does_not_bypass_original_frame_integrity_or_unknown_event_fields():
    from poorstar.actions.timing import _current_observation_matches

    f, _, current, *_ = event_fixture()
    with pytest.raises(ValueError, match="corrupt"):
        _current_observation_matches(
            replace(f, key=replace(f.key, observation_sha256="bad")), current.observation
        )
    # Unknown protobuf field 99, varint 1: it must never disappear during comparison.
    current.observation.raw_data.event.MergeFromString(b"\x98\x06\x01")
    assert not _current_observation_matches(f, current.observation)


def test_policy_gets_real_raster_arrays_without_metadata_hash_features():
    import numpy as np

    _, raw, info, cat, q, r, context = fixture()
    image = raw.observation.raw_data.map_state.visibility
    image.bits_per_pixel = 8
    image.size.x = 200
    image.size.y = 160
    image.data = b"\x02" * (200 * 160)
    frame = ActionFrame.from_protocol(
        raw, info, cat, BoundQuery.bind(raw.observation, q, r), context=context, episode="episode"
    )
    observation = frame.policy_observation()
    assert isinstance(observation["spatial_arrays"]["visibility"], np.ndarray)
    assert observation["spatial_arrays"]["visibility"].shape == (160, 200)
    assert np.all(observation["spatial_arrays"]["visibility"] == 2)
    assert (
        "spatial" not in observation
        and "context" not in observation
        and "catalog_sha256" not in observation
    )
    assert observation["scalars"]["own_visible_larva_count"] == 2


def test_abilities_only_catalog_cannot_silently_remove_larva_field_from_policy():
    f, raw, info, full, q, r, context = fixture()
    abilities = sc.ResponseData()
    abilities.abilities.extend(full.abilities)
    partial = EngineActionCatalog.from_protocol(abilities)
    frame = ActionFrame.from_protocol(
        raw,
        info,
        partial,
        BoundQuery.bind(raw.observation, q, r),
        context=context,
        episode="episode",
    )
    with pytest.raises(ValueError, match="unit catalog"):
        frame.policy_observation()
    complete = partial.with_unit_catalog(full)
    restored = ActionFrame.from_protocol(
        raw,
        info,
        complete,
        BoundQuery.bind(raw.observation, q, r),
        context=context,
        episode="episode",
    )
    assert restored.policy_observation() == f.policy_observation()
    full.abilities[0].target = 2
    with pytest.raises(ValueError, match="identical measured abilities"):
        partial.with_unit_catalog(full)


@pytest.mark.parametrize("decision", ["ACT", "WAIT"])
def test_episode_end_during_delay_is_a_terminal_receipt_not_a_body_or_fake_wait(decision):
    start, before, info, cat, _, _, context = fixture(source="LIVE_PLAYER")
    _, after, *_ = fixture(loop=2, source="LIVE_PLAYER")
    after.player_result.add(player_id=1, result=1)
    state = {"raw": before}

    def step(n):
        state["raw"] = after

    ctl = SimpleNamespace(
        observe=lambda: state["raw"],
        step=step,
        query=lambda request: pytest.fail("no body query at terminal"),
        actions=lambda request: pytest.fail("no dispatch at terminal"),
    )
    receipt = run_scheduled_decision(
        ctl,
        start,
        choose_schedule=lambda payload, horizon: (decision, 3),
        choose_body=lambda *args: pytest.fail("no body policy at terminal"),
        maximum_delay=8,
        context=context,
        episode="episode",
        game_info=info,
        catalog=cat,
    )
    assert receipt.decision == decision and receipt.terminated and receipt.dispatch is None
    assert receipt.prepared.elapsed_loops == 2 and receipt.prepared.requested_delay == 3
    assert receipt.prepared.body_frame is None and receipt.prepared.termination_loop == 2
