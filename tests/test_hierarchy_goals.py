"""Tests: test hierarchy goals."""

import json
from copy import deepcopy
from dataclasses import asdict, replace

import numpy as np
import pytest
import torch
from s2clientprotocol import sc2api_pb2 as sc

from poorstar.actions.coarse import CoarseActionProfile
from poorstar.actions.frame import ActionFrame, BoundQuery
from poorstar.hierarchy import (
    DispatchObservation,
    ExecutionFeedback,
    ExternalPreferenceProgram,
    HierarchicalPolicy,
    HierarchyConfig,
    HierarchyPolicySession,
    HierarchyState,
    PreferenceIntervention,
    TaskEffectObserver,
    TaskGoal,
    TaskState,
)
from poorstar.hierarchy.observer import entity_region, region_visibility
from poorstar.hierarchy.state import advance_state
from poorstar.models.policy import BCModelConfig
from tests.test_hierarchy_runtime import model_fixture, update_control
from tests.test_shared_action_contract import fixture


@pytest.fixture(scope="module", autouse=True)
def threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def visible_frame(
    loop=0,
    *,
    tags=(10, 20),
    complete=True,
    positions=None,
    upgrades=(),
    visibility=None,
    episode="episode",
    player=1,
):
    _, raw, info, catalog, request, reply, context = fixture(loop=loop, tags=tags, episode=episode)
    catalog.units.add(unit_id=200, name="MeasuredOtherType")
    catalog.upgrades.add(upgrade_id=7, name="MeasuredUpgrade")
    for ability, name in (
        (3664, "TransportUnloadAll"),
        (3670, "TransportUnloadUnit"),
        (3671, "CancelLast"),
        (3672, "CancelSlot"),
    ):
        catalog.abilities.add(ability_id=ability, target=1, link_name=name)
        for answer in reply.abilities:
            answer.abilities.add(ability_id=ability)
    for index, unit in enumerate(raw.observation.raw_data.units):
        if complete is not None:
            unit.build_progress = 1.0 if complete else 0.5
        unit.health = 20.0 + index
        unit.health_max = 100.0
        if positions:
            unit.pos.x, unit.pos.y = positions[index]
    if upgrades is not None:
        raw.observation.raw_data.player.SetInParent()
        raw.observation.raw_data.player.upgrade_ids.extend(upgrades)
    if visibility is not None:
        image = raw.observation.raw_data.map_state.visibility
        image.bits_per_pixel = 8
        image.size.x = 200
        image.size.y = 160
        image.data = np.asarray(visibility, dtype=np.uint8).tobytes()
    if player != 1:
        raw.observation.player_common.player_id = player
        context = replace(context, player_id=player)
    frame = ActionFrame.from_protocol(
        raw,
        info,
        catalog,
        BoundQuery.bind(raw.observation, request, reply),
        context=context,
        episode=episode,
    )
    return frame, catalog


def goal_config(**kw):
    return HierarchyConfig(
        arm="direct_task",
        manager_period=8,
        task_period=4,
        task_slots=2,
        commitment_horizons=(8, 16),
        task_horizons=(4, 8, 16),
        target_regions=(2, 2),
        grounded_tasks=True,
        **kw,
    )


def declared_task(config, goal, *, task_id="episode:task:1", units=0, start=0, horizon=8, slot=0):
    return TaskState(
        task_id,
        slot,
        "movement",
        start,
        horizon,
        0,
        0.5,
        None,
        None,
        goal=goal,
        squad_id=task_id.replace("task", "squad"),
        allocation_size=units,
    )


def grounded_model(*, arm="direct_task", **kw):
    frame, catalog = visible_frame()
    profile = CoarseActionProfile.from_catalog(catalog)
    config = BCModelConfig.from_catalog(catalog, profile, hidden_size=32, maximum_delay=8)
    hierarchy = replace(goal_config(**kw), arm=arm)
    torch.manual_seed(52)
    return HierarchicalPolicy(config, hierarchy), frame, profile


def grounded_control(model, *, kind="allocated_region", slots=(0,), region=3):
    q = dict(
        operation="assign",
        task_type="movement",
        horizon=8,
        resource_priority=0.5,
        target_region=region,
        goal=TaskGoal(
            kind,
            100 if kind == "own_completed_unit_delta" else 7 if kind == "upgrade_present" else None,
            1 if kind == "own_completed_unit_delta" else None,
            region if kind in ("allocated_region", "region_visible") else None,
        ).to_dict(),
        unit_slots=list(slots),
    )
    manager = (
        update_control(model)["manager"]
        if model.hierarchy_config.arm == "persistent_intent"
        else None
    )
    return dict(manager=manager, tasks=[q, {"operation": "keep"}])


@pytest.mark.parametrize(
    "kind", ["own_completed_unit_delta", "upgrade_present", "allocated_region", "region_visible"]
)
def test_goals_confirm_only_actual_visible_satisfaction_after_registration(kind):
    config = goal_config()
    initial, _ = visible_frame(visibility=np.zeros((160, 200)))
    goal = TaskGoal(
        kind,
        100 if kind == "own_completed_unit_delta" else 7 if kind == "upgrade_present" else None,
        1 if kind == "own_completed_unit_delta" else None,
        3 if kind in ("allocated_region", "region_visible") else None,
    )
    task = declared_task(config, goal, units=1 if kind == "allocated_region" else 0)
    observer = TaskEffectObserver(config, episode="episode", player_id=1)
    observer.register(initial, task, (0,) if task.allocation_size else ())
    assert not any(x.outcome == "confirmed" for x in observer.observe(initial))
    later, _ = visible_frame(
        1,
        tags=(10, 20, 30) if kind == "own_completed_unit_delta" else (10, 20),
        positions=((150, 120), (5, 7)) if kind == "allocated_region" else None,
        upgrades=(7,) if kind == "upgrade_present" else (),
        visibility=np.full((160, 200), 2),
    )
    (outcome,) = observer.observe(later)
    assert outcome.outcome == "confirmed" and outcome.progress == 1.0
    assert outcome.reason == "OBSERVED_DECLARED_GOAL_SATISFACTION"
    assert observer.observe(later) == ()
    public = json.dumps(asdict(outcome))
    assert all(k not in public for k in ("unit_tag", "world_x", "world_y", "members", "salt"))


def test_count_goal_counts_only_completed_own_current_units():
    config = goal_config()
    initial, _ = visible_frame()
    task = declared_task(config, TaskGoal("own_completed_unit_delta", 100, 2))
    observer = TaskEffectObserver(config, episode="episode", player_id=1)
    observer.register(initial, task, ())
    incomplete, _ = visible_frame(1, tags=(10, 20, 30, 40), complete=False)
    assert observer.observe(incomplete) == ()
    partial, _ = visible_frame(2, tags=(10, 20, 30))
    (progress,) = observer.observe(partial)
    assert progress.outcome == "progress" and progress.progress == 0.5
    fewer, _ = visible_frame(3, tags=(10, 20))
    assert observer.observe(fewer) == ()  # evidenced progress is monotone, not fabricated failure.
    later, _ = visible_frame(4, tags=(10, 20, 30, 40))
    assert observer.observe(later)[0].outcome == "confirmed"


@pytest.mark.parametrize(
    "reason", ["deadline", "missing_unit", "missing_field", "already_satisfied"]
)
def test_unknown_or_censored_never_becomes_negative_or_confirmation(reason):
    config = goal_config()
    goal = (
        TaskGoal("allocated_region", region=3)
        if reason == "missing_unit"
        else TaskGoal("upgrade_present", subject=7)
    )
    first, _ = visible_frame(upgrades=(7,) if reason == "already_satisfied" else ())
    task = declared_task(config, goal, units=1 if reason == "missing_unit" else 0)
    observer = TaskEffectObserver(config, episode="episode", player_id=1)
    observer.register(first, task, (0,) if task.allocation_size else ())
    later, _ = visible_frame(
        8 if reason == "deadline" else 1,
        tags=(20,) if reason == "missing_unit" else (10, 20),
        upgrades=None if reason == "missing_field" else (7,),
    )
    (result,) = observer.observe(later)
    assert result.outcome == "censored" and result.progress is None and result.to_feedback() is None


def test_frame_local_members_rebind_by_private_identity_not_slot_or_type():
    config = goal_config()
    first, _ = visible_frame(tags=(20, 30))
    task = declared_task(config, TaskGoal("allocated_region", region=3), units=1)
    observer = TaskEffectObserver(config, episode="episode", player_id=1)
    observer.register(first, task, (0,))  # tag20 has current slot0.
    later, _ = visible_frame(1, tags=(10, 20, 30))
    assert observer.current_allocations(later) == {task.task_id: (1,)}
    assert observer.observe(later) == ()
    foreign, _ = visible_frame(2, tags=(10, 20, 30), positions=((150, 120), (5, 7), (5, 7)))
    assert observer.observe(foreign) == ()  # newcomer in region is not our member.
    moved, _ = visible_frame(3, tags=(10, 20, 30), positions=((5, 7), (150, 120), (5, 7)))
    assert observer.observe(moved)[0].outcome == "confirmed"


@pytest.mark.parametrize("fault", ["other_player", "other_episode", "past", "new_map"])
def test_observer_rejects_foreign_or_noncausal_frame(fault):
    config = goal_config()
    first, _ = visible_frame(1)
    observer = TaskEffectObserver(config, episode="episode", player_id=1)
    task = declared_task(config, TaskGoal("upgrade_present", subject=7), start=1)
    observer.register(first, task, ())
    candidate, _ = visible_frame(
        0 if fault == "past" else 2,
        player=2 if fault == "other_player" else 1,
        episode="other" if fault == "other_episode" else "episode",
    )
    if fault == "new_map":
        candidate = replace(candidate, key=replace(candidate.key, map_sha256="f" * 64))
    with pytest.raises(ValueError):
        observer.observe(candidate)


def test_observer_recovery_and_slot_replacement_are_exact_and_bounded():
    config = goal_config()
    first, _ = visible_frame()
    task = declared_task(config, TaskGoal("allocated_region", region=3), units=1)
    observer = TaskEffectObserver(config, episode="episode", player_id=1)
    observer.register(first, task, (0,))
    recovered = TaskEffectObserver(config, episode="episode", player_id=1)
    recovered.load_state_dict(json.loads(json.dumps(observer.state_dict())))
    later, _ = visible_frame(1, positions=((150, 120), (5, 7)))
    assert recovered.observe(later) == observer.observe(later)
    next_task = declared_task(
        config, TaskGoal("upgrade_present", subject=7), task_id="episode:task:2", start=1
    )
    recovered.synchronize((next_task,))
    recovered.register(later, next_task, ())
    after, _ = visible_frame(2, upgrades=(7,))
    assert {x.task_id for x in recovered.observe(after)} == {next_task.task_id}
    assert len(recovered.state_dict()["tasks"]) == 1
    bad = observer.state_dict()
    bad["episode"] = "other"
    with pytest.raises(ValueError, match="identity"):
        recovered.load_state_dict(bad)


def test_allocations_are_explicit_exclusive_and_never_steal_other_squads():
    config = goal_config()
    frame, _ = visible_frame()
    observer = TaskEffectObserver(config, episode="episode", player_id=1)
    first = declared_task(config, TaskGoal("allocated_region", region=3), units=1)
    second = declared_task(
        config, TaskGoal("allocated_region", region=3), task_id="episode:task:2", units=1, slot=1
    )
    observer.register(frame, first, (0,))
    with pytest.raises(ValueError, match="two active"):
        observer.register(frame, second, (0,))
    observer.register(frame, second, (1,))
    assert observer.current_allocations(frame) == {first.task_id: (0,), second.task_id: (1,)}
    assert (
        observer.allocation_audit(frame, first.task_id, (0,))["status"]
        == "MATCHED_DECLARED_ALLOCATION"
    )
    assert (
        observer.allocation_audit(frame, first.task_id, (1,))["status"]
        == "OTHER_TASK_MEMBER_EMITTED"
    )
    assert observer.allocation_audit(frame, None, (0, 1))["status"] == "EXPLICIT_UNASSIGNED_BUNDLE"


@pytest.mark.parametrize("arm", ["direct_task", "persistent_intent"])
def test_typed_goal_and_allocation_teacher_sample_density_state_and_gradient_parity(arm):
    model, frame, _ = grounded_model(arm=arm)
    original = HierarchyState("episode")
    sample = model.encode_control(
        frame.policy_observation(), control_state=original, deterministic=True
    )
    teacher = model.encode_control(
        frame.policy_observation(), control_state=original, teacher_control=sample.control
    )
    assert sample.control_state == teacher.control_state
    assert torch.equal(sample.control_log_prob, teacher.control_log_prob)
    assert sample.task_allocations == teacher.task_allocations
    teacher = model.encode_control(
        frame.policy_observation(), control_state=original, teacher_control=grounded_control(model)
    )
    teacher.control_log_prob.backward()
    assert model.task_goal_kind_head.weight.grad is not None
    assert model.task_allocation_query.weight.grad is not None or not any(
        q["operation"] == "assign" for q in sample.control["tasks"]
    )


def test_learned_unit_context_changes_decoder_and_critic_without_mask_changes():
    model, frame, profile = grounded_model()
    initial = model.encode_control(
        frame.policy_observation(),
        control_state=HierarchyState("episode"),
        teacher_control=grounded_control(model),
    )
    later, _ = visible_frame(1)
    keep = {"manager": None, "tasks": None}
    state = initial.control_state
    a = model.encode_control(
        later.policy_observation(),
        control_state=state,
        teacher_control=keep,
        task_allocations={"episode:task:1": (0,)},
    )
    b = model.encode_control(
        later.policy_observation(),
        control_state=state,
        teacher_control=keep,
        task_allocations={"episode:task:1": (1,)},
    )
    assert not torch.equal(a.encoded.hidden, b.encoded.hidden)
    assert not torch.equal(a.value, b.value)
    assert frame.ability_options() == later.ability_options()
    with pytest.raises(ValueError, match="rebinding"):
        model.encode_control(later.policy_observation(), control_state=state, teacher_control=keep)
    with pytest.raises(ValueError, match="own-visible"):
        model.evaluate_values(
            later.policy_observation(),
            control_state=state,
            task_allocations={"episode:task:1": (999,)},
        )
    group = dict(model.parameter_groups()["task"])
    assert "task_unit_context.weight" in group


def test_grounded_feedback_ablation_keeps_visible_members_but_hides_goal_outcomes_and_event_trigger():
    model, frame, _ = grounded_model(use_feedback=False)
    first = model.encode_control(
        frame.policy_observation(),
        control_state=HierarchyState("episode"),
        teacher_control=grounded_control(model),
    )
    later, _ = visible_frame(2, positions=((150, 120), (5, 7)))
    event = ExecutionFeedback("visible-goal", "episode:task:1", "confirmed", 2, "d" * 64, 1.0)
    kwargs = dict(
        control_state=first.control_state,
        teacher_control={"manager": None, "tasks": None},
        task_allocations={"episode:task:1": (1,)},
        allow_manager=False,
    )
    a = model.encode_control(later.policy_observation(), feedback=(event,), **kwargs)
    b = model.encode_control(later.policy_observation(), **kwargs)
    assert torch.equal(a.encoded.hidden, b.encoded.hidden) and torch.equal(a.value, b.value)
    assert a.control == b.control and a.control_state.tasks == b.control_state.tasks
    assert a.control_state.tasks[0].active and a.control_state.tasks[0].progress == 0.0


@pytest.mark.parametrize(
    "fault", ["same_member", "extra_field", "already_region", "unknown_unit", "empty_allocated"]
)
def test_goal_teacher_rejects_invalid_or_unsatisfied_domain_parameters(fault):
    model, frame, _ = grounded_model()
    control = grounded_control(model)
    if fault == "same_member":
        control["tasks"][1] = deepcopy(control["tasks"][0])
    if fault == "extra_field":
        control["tasks"][0]["goal"]["future_winner"] = True
    if fault == "already_region":
        control["tasks"][0]["goal"]["region"] = 0
    if fault == "unknown_unit":
        control["tasks"][0]["goal"] = TaskGoal("own_completed_unit_delta", 999, 1).to_dict()
    if fault == "empty_allocated":
        control["tasks"][0]["unit_slots"] = []
    with pytest.raises(ValueError):
        model.encode_control(
            frame.policy_observation(),
            control_state=HierarchyState("episode"),
            teacher_control=control,
        )


def test_grounded_session_requires_real_observer_and_rejects_rpc_forged_confirmation():
    model, frame, profile = grounded_model()
    session = HierarchyPolicySession(model, profile, deterministic=True)
    session.reset("episode")
    with pytest.raises(ValueError, match="frame hook"):
        session.choose_schedule(frame.policy_observation(), 8)
    session.observe_tasks(frame)
    forged = ExecutionFeedback("fake", "episode:task:1", "confirmed", 0, "rpc_accepted", 1.0)
    with pytest.raises(ValueError, match="task observer"):
        session.submit_feedback(forged)
    session.choose_schedule(frame.policy_observation(), 8)
    assert "scheduler_task_allocations" in session.audit_trace
    recovered = HierarchyPolicySession(model, profile, deterministic=True)
    recovered.load_state_dict(session.state_dict())
    assert recovered.control_state == session.control_state
    assert recovered.task_observer.state_dict() == session.task_observer.state_dict()
    later, _ = visible_frame(1)
    assert recovered.observe_tasks(later) == session.observe_tasks(later)


@pytest.mark.parametrize("arm", ["flat", "direct_task", "persistent_intent"])
def test_common_receipts_are_shared_causal_backbone_inputs_and_feedback_off_is_exact(arm):
    model, frame, _ = model_fixture(arm, common_receipts_mode="shared_v1")
    prior = HierarchyState("episode")
    control = update_control(model)
    receipt = DispatchObservation(0, (2, 1, 1, 1), "current-rpc-evidence")
    a = model.encode_control(
        frame.policy_observation(), control_state=prior, teacher_control=control
    )
    b = model.encode_control(
        frame.policy_observation(),
        control_state=prior,
        teacher_control=control,
        dispatch_observation=receipt,
    )
    assert not torch.equal(a.memory_hidden, b.memory_hidden)
    assert torch.equal(a.encoded.entities, b.encoded.entities)
    assert "dispatch_projection.weight" in dict(model.parameter_groups()["backbone"])
    disabled = HierarchicalPolicy(model.config, replace(model.hierarchy_config, use_feedback=False))
    disabled.load_state_dict(model.state_dict())
    x = disabled.encode_control(
        frame.policy_observation(), control_state=prior, teacher_control=control
    )
    y = disabled.encode_control(
        frame.policy_observation(),
        control_state=prior,
        teacher_control=control,
        dispatch_observation=receipt,
    )
    assert torch.equal(x.memory_hidden, y.memory_hidden) and torch.equal(
        x.control_log_prob, y.control_log_prob
    )
    with pytest.raises(ValueError, match="future"):
        model.evaluate_values(
            frame.policy_observation(),
            control_state=prior,
            dispatch_observation=replace(receipt, loop=1),
        )


def test_common_receipt_session_scheduler_body_and_recovery_use_exact_same_input():
    model, frame, profile = model_fixture("flat", common_receipts_mode="shared_v1")
    session = HierarchyPolicySession(model, profile, deterministic=True)
    session.reset("episode")
    receipt = DispatchObservation(0, (1, 1, 0, 1), "real_rpc")
    session.submit_dispatch_observation(receipt)
    session.choose_schedule(frame.policy_observation(), 8)
    session.choose_body(frame.policy_observation(), "ability", profile.ability_options(frame), ())
    assert (
        session.audit_trace["scheduler_dispatch_observation"]
        == session.audit_trace["body_dispatch_observation"]
        == asdict(receipt)
    )
    recovered = HierarchyPolicySession(model, profile, deterministic=True)
    recovered.load_state_dict(session.state_dict())
    assert recovered.dispatch_observation == receipt
    left = session.choose_body(frame.policy_observation(), "queue", (False, True), ())
    right = recovered.choose_body(frame.policy_observation(), "queue", (False, True), ())
    assert left == right and session.action_log_prob == recovered.action_log_prob


@pytest.mark.parametrize("mode", ["stepwise", "fixed_dwell", "minimum_hold"])
def test_pure_external_program_renews_exactly_declared_control_clock(mode):
    options = dict(manager_mode="external")
    if mode == "stepwise":
        options["persistence"] = False
    if mode == "fixed_dwell":
        options.update(commitment_mode="fixed_dwell", fixed_commitment_horizon=8)
    model, _, _ = model_fixture(**options)
    program = ExternalPreferenceProgram(
        model.hierarchy_config,
        episode="episode",
        interventions=(PreferenceIntervention(0, (0.5,) * 9, 8),),
    )
    state = HierarchyState("episode")
    first, *_ = visible_frame()
    override = program.override_for(0, state)
    assert override == program.override_for(0, state)
    control = {"manager": None, "tasks": [{"operation": "keep"}] * 2}
    state = model.encode_control(
        first.policy_observation(),
        control_state=state,
        external_override=override,
        teacher_control=control,
    ).control_state
    for loop in (1, 7, 8, 9):
        newer, *_ = visible_frame(loop)
        override = program.override_for(loop, state)
        expected = mode == "stepwise" or mode == "fixed_dwell" and loop == 8
        assert (override is not None) == expected
        result = model.encode_control(
            newer.policy_observation(),
            control_state=state,
            external_override=override,
            teacher_control=None,
            deterministic=True,
        )
        assert result.manager_log_prob == 0 and not result.manager_decision
        state = result.control_state
    assert (
        program.state_dict()
        == ExternalPreferenceProgram(
            model.hierarchy_config,
            episode="episode",
            interventions=(PreferenceIntervention(0, (0.5,) * 9, 8),),
        ).state_dict()
    )


def test_confidence_prediction_is_pure_prior_context_and_auxiliary_gradient():
    model, frame, _ = model_fixture()
    prior = HierarchyState("episode")
    saved = prior.to_dict()
    value = model.confidence_prediction(frame.policy_observation(), control_state=prior)
    assert prior.to_dict() == saved and 0 < float(value) < 1
    encoded = model._encode_shared(frame.policy_observation())
    latent = model.manager_input(
        torch.cat((encoded.hidden, model.zero_state(), model.zero_state(), model.zero_state()))
    )
    params = torch.nn.functional.softplus(model.manager_confidence_head(latent)) + 1.0
    assert torch.equal(value, params[0] / params.sum())
    (-torch.log(value)).backward()
    assert model.manager_confidence_head.weight.grad.abs().sum() > 0


@pytest.mark.parametrize("mode", ["flat", "direct_task", "stepwise", "fixed_dwell", "external"])
def test_confidence_prediction_rejects_comparators_without_declared_head(mode):
    options = {}
    arm = mode if mode in ("flat", "direct_task") else "persistent_intent"
    if mode == "stepwise":
        options["persistence"] = False
    if mode == "fixed_dwell":
        options["commitment_mode"] = "fixed_dwell"
    if mode == "external":
        options["manager_mode"] = "external"
    model, frame, _ = model_fixture(arm, **options)
    with pytest.raises(ValueError, match="no learned confidence"):
        model.confidence_prediction(
            frame.policy_observation(), control_state=HierarchyState("episode")
        )


@pytest.mark.parametrize("frozen", ["joint", "value_only", "manager_rl"])
def test_complete_grounded_goal_body_feedback_shared_receipt_actor_scorer_and_recovery_parity(
    frozen,
):
    from types import SimpleNamespace

    from poorstar.actions.coarse import coarse_sample
    from poorstar.training.scorer import HierarchyScorer, payload_from_session
    from tests.test_hierarchy_runtime import prepare_session

    model, first, profile = grounded_model(
        arm="persistent_intent", goal_kinds=("allocated_region",), common_receipts_mode="shared_v1"
    )
    with torch.no_grad():
        model.task_operation_head.weight.zero_()
        model.task_operation_head.bias.copy_(torch.tensor([-10.0, 10.0, -10.0]))
        model.task_allocation_query.weight.zero_()
        model.task_allocation_query.bias.zero_()
        model.task_goal_region_head.weight.zero_()
        model.task_goal_region_head.bias.copy_(torch.tensor([-10.0, -10.0, -10.0, 10.0]))
        model.task_horizon_head.weight.zero_()
        model.task_horizon_head.bias.copy_(torch.tensor([-10.0, 10.0, -10.0]))
    allowed = (
        {"value"}
        if frozen == "value_only"
        else {"manager", "value"}
        if frozen == "manager_rl"
        else None
    )
    for group, parameters in model.parameter_groups().items():
        for _, parameter in parameters:
            parameter.requires_grad_(allowed is None or group in allowed)
    session = prepare_session(model, profile)
    session.submit_dispatch_observation(DispatchObservation(0, (1, 1, 0, 1), "fixture-real-rpc"))
    session.observe_tasks(first)
    decision, delay = session.choose_schedule(first.policy_observation(), 8)
    assert (decision, delay) == ("ACT", 2)
    snapshot = session.state_dict()
    task_id = session.control_state.tasks[0].task_id
    body, _ = visible_frame(2, positions=((150, 120), (150, 120)))
    outcomes = session.observe_tasks(body)
    assert outcomes[0].outcome == "confirmed" and outcomes[0].task_id == task_id
    sampled = coarse_sample(
        body,
        lambda head, options, completed: session.choose_body(
            body.policy_observation(), head, options, completed
        ),
        profile,
    )
    assert sampled.protocol_actions()
    assert session.audit_trace["body_feedback"][0]["status"] == "confirmed"
    payload = payload_from_session(session, first, body)
    step = SimpleNamespace(
        payload=payload, manager_decision=session.audit_trace["manager_decision"]
    )
    score = HierarchyScorer(model, profile)(step)
    assert float(score.log_prob) == session.action_log_prob
    assert torch.equal(score.next_hidden[0], session.hidden)
    assert score.next_hidden[1] == session.control_state
    recovered = HierarchyPolicySession(model, profile, deterministic=True)
    recovered.load_state_dict(snapshot)
    assert recovered.observe_tasks(body) == outcomes
    rerun = coarse_sample(
        body,
        lambda head, options, completed: recovered.choose_body(
            body.policy_observation(), head, options, completed
        ),
        profile,
    )
    assert sampled.commands == rerun.commands and recovered.audit_trace == session.audit_trace


def test_real_typed_annotation_sidecar_json_loader_bc_loss_count_and_gradients(tmp_path):
    from poorstar.actions.coarse import coarse_sample
    from poorstar.actions.contract import RawCommandTarget
    from poorstar.data.bc.records import BCDecision
    from poorstar.evaluation.confidence import ConfidenceWindow, build_confidence_target
    from poorstar.evaluation.teachers import build_control_sidecar
    from poorstar.hierarchy import HierarchyBCRecord, hierarchy_bc_objective
    from poorstar.training.config import (
        ControlSidecar,
        control_label_from_dict,
        model_config_sha256,
    )
    from poorstar.training.contracts import TrainingIdentity, digest
    from tests.test_hierarchy_runtime import prepare_session

    model, first, profile = grounded_model(
        arm="persistent_intent", goal_kinds=("allocated_region",), common_receipts_mode="shared_v1"
    )
    with torch.no_grad():
        model.task_operation_head.weight.zero_()
        model.task_operation_head.bias.copy_(torch.tensor([-10.0, 10.0, -10.0]))
        model.task_allocation_query.weight.zero_()
        model.task_allocation_query.bias.zero_()
        model.task_goal_region_head.weight.zero_()
        model.task_goal_region_head.bias.copy_(torch.tensor([-10.0, -10.0, -10.0, 10.0]))
        model.task_horizon_head.weight.zero_()
        model.task_horizon_head.bias.copy_(torch.tensor([-10.0, 10.0, -10.0]))
    session = prepare_session(model, profile)
    session.submit_dispatch_observation(DispatchObservation(0, (1, 1, 0, 1), "fixture-rpc"))
    session.observe_tasks(first)
    session.choose_schedule(first.policy_observation(), 8)
    intent = session.control_state.intent
    body, _ = visible_frame(2, positions=((150, 120), (150, 120)))
    outcomes = session.observe_tasks(body)
    sampled = coarse_sample(
        body,
        lambda head, options, completed: session.choose_body(
            body.policy_observation(), head, options, completed
        ),
        profile,
    )
    trace = session.audit_trace
    row = BCDecision(
        "a" * 64,
        first,
        "ACT",
        2,
        body,
        tuple(RawCommandTarget.from_protocol(a, body) for a in sampled.protocol_actions()),
    )
    window = ConfidenceWindow(
        "episode",
        intent.revision,
        intent.start_loop,
        intent.start_loop + intent.commitment_horizon,
        first.key.observation_sha256,
    )
    target = build_confidence_target(
        window,
        confirmations=[
            dict(
                episode="episode",
                parent_revision=intent.revision,
                loop=o.loop,
                task_id=o.task_id,
                status=o.outcome,
                evidence_sha256=o.evidence_id,
            )
            for o in outcomes
        ],
        cutoff_loop=2,
        cutoff_evidence_sha256=body.key.observation_sha256,
    )
    assert target.mask and target.value == 1.0
    names = (
        "scheduler_control",
        "body_control",
        "body_execution_task",
        "feedback",
        "external_override",
        "scheduler_dispatch_observation",
        "body_dispatch_observation",
        "scheduler_task_allocations",
        "body_task_allocations",
        "body_feedback",
    )
    label = {k: deepcopy(trace[k]) for k in names}
    label.update(
        supervision_id="fixture-observed-controls",
        supervision_kind="actor_control_trace",
        confidence_target=asdict(target),
    )
    # Force the actual JSON representation before both compiler and learner.
    label = json.loads(json.dumps(label))

    def dependency(frame):
        return dict(
            loop=frame.key.loop,
            player_id=frame.key.player_id,
            observation_sha256=frame.key.observation_sha256,
            evidence_sha256=frame.key.observation_sha256,
            availability="ACTOR_VISIBLE_CURRENT_OR_PAST",
        )

    annotation = dict(
        row_id=ControlSidecar.row_id(row),
        label=label,
        scheduler_evidence=[dependency(first)],
        body_evidence=[dependency(first), dependency(body)],
    )
    initial = HierarchyState("episode")
    evidence = dict(
        schema="poorstar_v3_control_annotation_evidence_v1",
        annotation_policy_sha256="f" * 64,
        human_intent_truth=False,
        future_or_privileged_inputs=False,
        annotation_sha256=digest([annotation]),
        row_order=[annotation["row_id"]],
        initial_control_states_sha256=digest({annotation["row_id"]: initial.to_dict()}),
        fixture_only=True,
        actual_engine_frames=0,
    )
    evidence_path = tmp_path / "visible_evidence.json"
    evidence_path.write_text(json.dumps(evidence))
    identity = TrainingIdentity(
        "typed_fixture",
        "persistent_intent",
        1,
        "b" * 64,
        "c" * 64,
        "d" * 64,
        "e" * 64,
        model_config_sha256(model.config, model.hierarchy_config),
        profile.sha256,
        "a" * 64,
    )
    output = tmp_path / "typed_sidecar.json"
    report = build_control_sidecar(
        (row,),
        (annotation,),
        initial_control_state=initial,
        model=model,
        profile=profile,
        identity=identity,
        evidence_path=evidence_path,
        output_path=output,
        annotation_policy_sha256="f" * 64,
    )
    loader = ControlSidecar(
        output, replace(identity, control_sidecar_sha256=report["sha256"]), model.hierarchy_config
    )
    actual_sequence = loader.sequence((row,))
    actual_record = actual_sequence.records[0]
    assert isinstance(actual_record.scheduler_dispatch_observation, DispatchObservation)
    assert isinstance(actual_record.body_feedback[0], ExecutionFeedback)
    assert actual_record.confidence_target == target
    expected_record = HierarchyBCRecord(row, **control_label_from_dict(label))
    expected = hierarchy_bc_objective(model, [(expected_record,)], profile, collect_metrics=False)
    actual = hierarchy_bc_objective(model, [actual_sequence], profile, collect_metrics=False)
    assert torch.equal(actual.loss, expected.loss) and actual.denominators == expected.denominators
    assert actual.denominators["confidence_aux"] == 1
    expected.loss.backward()
    gradients = {k: None if p.grad is None else p.grad.clone() for k, p in model.named_parameters()}
    model.zero_grad()
    actual.loss.backward()
    for name, parameter in model.named_parameters():
        assert (parameter.grad is None) == (gradients[name] is None)
        if parameter.grad is not None:
            assert torch.equal(parameter.grad, gradients[name]), name
