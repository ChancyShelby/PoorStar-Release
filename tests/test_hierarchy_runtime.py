"""Tests: test hierarchy runtime."""

import json
from copy import deepcopy
from dataclasses import replace

import pytest
import torch

from poorstar.actions.coarse import coarse_sample, coarse_teacher_force
from poorstar.actions.contract import Token
from poorstar.hierarchy import (
    ExecutionFeedback,
    ExternalOverride,
    HierarchicalPolicy,
    HierarchyConfig,
    HierarchyPolicySession,
    HierarchyState,
    IntentState,
    TaskState,
    control_objective,
)
from poorstar.hierarchy.state import advance_state
from poorstar.models.policy import BCModelConfig, RecurrentBCPolicy
from tests.test_coarse_bc import setup, target


@pytest.fixture(scope="module", autouse=True)
def threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def model_fixture(arm="persistent_intent", **kwargs):
    frame, profile, catalog, info = setup()
    config = BCModelConfig.from_catalog(catalog, profile, hidden_size=32, maximum_delay=8)
    hierarchy = HierarchyConfig(
        arm=arm,
        manager_period=8,
        task_period=4,
        commitment_horizons=(8, 16),
        task_horizons=(4, 8),
        task_slots=2,
        target_regions=(2, 2),
        **kwargs,
    )
    torch.manual_seed(31)
    model = HierarchicalPolicy(config, hierarchy)
    return model, frame, profile


def update_control(model, *, operation="update", task_operation="assign"):
    c = model.hierarchy_config
    manager = None
    if c.arm == "persistent_intent":
        manager = {"operation": operation}
        if operation == "update":
            manager.update(preference=[0.5] * len(c.preference_names))
            if c.persistence and c.commitment_mode == "minimum_hold":
                manager.update(commitment_horizon=8, confidence=0.7)
    tasks = None if c.arm == "flat" else []
    if tasks is not None:
        for i in range(c.task_slots):
            q = {"operation": task_operation}
            if task_operation == "assign":
                q.update(
                    task_type=c.task_types[i],
                    horizon=8,
                    resource_priority=0.5,
                    target_region=None if i == 0 else i,
                )
            tasks.append(q)
    return {"manager": manager, "tasks": tasks}


def observed(loop):
    frame, *_ = setup(loop=loop)
    return frame.policy_observation()


@pytest.mark.parametrize("arm", ["flat", "direct_task", "persistent_intent"])
def test_three_arms_have_operational_probabilities_and_disjoint_groups(arm):
    model, f, _ = model_fixture(arm)
    out = model.encode_control(
        f.policy_observation(),
        control_state=HierarchyState("episode"),
        teacher_control=update_control(model),
    )
    assert torch.isfinite(out.control_log_prob) and torch.isfinite(out.value)
    assert out.encoded.entities.shape == (2, 32)
    groups = model.parameter_groups()
    names = [n for values in groups.values() for n, _ in values]
    assert len(names) == len(set(names)) == len(dict(model.named_parameters()))
    assert set(names) == set(dict(model.named_parameters()))
    if arm == "flat":
        assert not groups["manager"] and not groups["task"]
    if arm == "direct_task":
        assert not groups["manager"] and out.control_state.intent is None
    if arm == "persistent_intent":
        assert groups["manager"] and out.control_state.intent.source == "learned"


def test_flat_preserves_original_encoder_decoder_and_teacher_losses_exactly():
    model, f, p = model_fixture("flat")
    plain = RecurrentBCPolicy(model.config)
    plain.load_state_dict({n: v for n, v in model.state_dict().items() if n in plain.state_dict()})
    a = plain.encode(f.policy_observation())
    b = model.encode_control(
        f.policy_observation(), control_state=HierarchyState("episode")
    ).encoded
    assert torch.equal(a.hidden, b.hidden) and torch.equal(a.entities, b.entities)
    targets = (target(f), target(f, ability=21, point=(14.125, 30.5)))
    first = plain.teacher_body(a, f, targets, p)
    second = model.teacher_body(b, f, targets, p)
    assert first[1] == second[1]
    assert all(torch.equal(torch.stack(first[0][h]), torch.stack(second[0][h])) for h in first[0])


@pytest.mark.parametrize("arm", ["flat", "direct_task", "persistent_intent"])
def test_foundation_checkpoint_transfer_is_exact_and_declares_new_parameters(arm):
    model, _, _ = model_fixture(arm)
    torch.manual_seed(92)
    source = RecurrentBCPolicy(model.config)
    original_new = {
        k: v.clone() for k, v in model.state_dict().items() if k not in source.state_dict()
    }
    report = model.load_c3_foundation(source)
    assert report["transfer_kind"] == "NEW_ARCHITECTURE_INITIALIZATION"
    assert set(report["shared_parameter_names"]) == set(source.state_dict())
    assert all(torch.equal(model.state_dict()[k], v) for k, v in source.state_dict().items())
    assert all(torch.equal(model.state_dict()[k], v) for k, v in original_new.items())
    with pytest.raises(ValueError, match="original C3"):
        model.load_c3_foundation(model)


def test_foundation_transfer_refuses_mismatched_frozen_model_config():
    model, _, _ = model_fixture()
    source = RecurrentBCPolicy(replace(model.config, maximum_delay=7))
    with pytest.raises(ValueError, match="matching"):
        model.load_c3_foundation(source)


@pytest.mark.parametrize("arm", ["direct_task", "persistent_intent"])
def test_nonflat_plain_encode_fails_instead_of_silently_ignoring_controls(arm):
    model, f, _ = model_fixture(arm)
    with pytest.raises(ValueError, match="explicit encode_control"):
        model.encode(f.policy_observation())


def test_sampled_and_teacher_controls_reproduce_state_distributions_and_gradients():
    model, f, _ = model_fixture()
    initial = HierarchyState("episode")
    torch.manual_seed(713)
    a = model.encode_control(f.policy_observation(), control_state=initial)
    rng = torch.get_rng_state().clone()
    b = model.encode_control(
        f.policy_observation(), control_state=initial, teacher_control=a.control
    )
    assert torch.equal(rng, torch.get_rng_state())
    assert a.control_state == b.control_state
    assert torch.equal(a.encoded.hidden, b.encoded.hidden)
    assert torch.equal(a.control_log_prob, b.control_log_prob)
    assert torch.equal(a.control_entropy, b.control_entropy)
    grads_a = torch.autograd.grad(a.control_log_prob, tuple(model.parameters()), allow_unused=True)
    grads_b = torch.autograd.grad(b.control_log_prob, tuple(model.parameters()), allow_unused=True)
    for x, y in zip(grads_a, grads_b):
        assert (x is None) == (y is None)
        if x is not None:
            assert torch.equal(x, y)


def test_explicit_control_and_all_body_heads_have_finite_trainable_gradients():
    model, f, p = model_fixture()
    out = model.encode_control(
        f.policy_observation(),
        control_state=HierarchyState("episode"),
        teacher_control=update_control(model),
    )
    body, task_lp, _, task_id = model.select_execution_task(
        out, teacher_task_id=out.control_state.tasks[0].task_id
    )
    targets = (
        target(f),
        target(f, ability=31, target=10),
        target(f, ability=21, point=(14.125, 30.5)),
    )
    losses, _, _, _, _ = model.teacher_body(body, f, targets, p)
    loss = control_objective(out) - task_lp + sum(torch.stack(v).mean() for v in losses.values())
    loss.backward()
    groups = model.parameter_groups()
    for name in ("manager", "task", "backbone", "decoder"):
        gradients = [p.grad for _, p in groups[name] if p.grad is not None]
        assert gradients and all(torch.isfinite(g).all() for g in gradients)
        assert sum(float(g.abs().sum()) for g in gradients) > 0
    assert task_id == out.control_state.tasks[0].task_id
    assert not any(p.grad is not None and bool(p.grad.abs().sum()) for _, p in groups["value"])


def test_persistent_keep_uses_real_age_without_resampling_preference():
    model, f, _ = model_fixture()
    a = model.encode_control(
        f.policy_observation(),
        control_state=HierarchyState("episode"),
        teacher_control=update_control(model),
    )
    b = model.encode_control(
        observed(3),
        a.memory_hidden,
        a.control_state,
        teacher_control={"manager": None, "tasks": None},
    )
    assert b.control_state.intent.preference == a.control_state.intent.preference
    assert b.control_state.intent.revision == 1 and b.control_state.intent.age == 3
    assert b.control == {"manager": None, "tasks": None}
    assert float(b.control_log_prob) == 0.0


def test_manager_trainable_eligibility_preserves_forced_keep_audit_without_new_segment():
    model, f, profile = model_fixture()
    q = update_control(model, task_operation="keep")
    q["manager"]["commitment_horizon"] = 16
    initial = HierarchyState("episode")
    rng = torch.get_rng_state().clone()
    assert model.manager_decision_eligible(initial, 0)
    assert torch.equal(rng, torch.get_rng_state()) and initial.last_loop is None
    first = model.encode_control(f.policy_observation(), control_state=initial, teacher_control=q)
    assert first.manager_decision
    assert not model.manager_decision_eligible(first.control_state, 3)
    assert not model.manager_decision_eligible(first.control_state, 8)
    forced = {"manager": {"operation": "keep"}, "tasks": [{"operation": "keep"}] * 2}
    receipt = model.encode_control(
        observed(8), first.memory_hidden, first.control_state, teacher_control=forced
    )
    assert receipt.control["manager"] == {"operation": "keep"}
    assert not receipt.manager_decision and float(receipt.manager_log_prob) == 0.0
    assert float(receipt.manager_entropy) == 0.0
    assert model.manager_decision_eligible(receipt.control_state, 16)
    learned_keep = model.encode_control(
        observed(16), receipt.memory_hidden, receipt.control_state, teacher_control=forced
    )
    assert learned_keep.manager_decision and float(learned_keep.manager_log_prob) < 0
    assert not model.manager_decision_eligible(receipt.control_state, 16, allow_manager=False)
    external = ExternalOverride((0.5,) * 9, 16, 0.7, "human")
    assert not model.manager_decision_eligible(
        receipt.control_state, 16, external_override=external
    )
    model.eval()
    session = HierarchyPolicySession(model, profile, deterministic=True)
    session.reset("episode")
    session.control_state = first.control_state
    session.hidden = first.memory_hidden.detach()
    session.choose_schedule(observed(8), model.config.maximum_delay)
    assert session.audit_trace["scheduler_control"]["manager"] == {"operation": "keep"}
    assert session.audit_trace["manager_decision"] is False


def test_causal_feedback_unblocks_a_real_manager_decision_inside_commitment():
    model, f, _ = model_fixture()
    first = model.encode_control(
        f.policy_observation(),
        control_state=HierarchyState("episode"),
        teacher_control=update_control(model),
    )
    event = ExecutionFeedback(
        "accepted", first.control_state.tasks[0].task_id, "accepted", 1, "actual-rpc-receipt"
    )
    assert model.manager_decision_eligible(first.control_state, 1, feedback=(event,))
    result = model.encode_control(
        observed(1),
        first.memory_hidden,
        first.control_state,
        feedback=(event,),
        teacher_control={"manager": {"operation": "keep"}, "tasks": [{"operation": "keep"}] * 2},
    )
    assert result.manager_decision
    assert not model.manager_decision_eligible(
        first.control_state, 1, feedback=(event,), allow_manager=False
    )


def test_stepwise_preference_is_fresh_and_independent_of_previous_intent():
    model, f, _ = model_fixture(persistence=False)
    first = model.encode_control(
        f.policy_observation(),
        control_state=HierarchyState("episode"),
        teacher_control=update_control(model),
    )
    old = first.control_state
    changed = replace(
        old,
        intent=replace(
            old.intent, preference=(1.0,) * 9, commitment_horizon=999, confidence=0.01, revision=99
        ),
    )
    q = {"manager": {"operation": "update", "preference": [0.25] * 9}, "tasks": None}
    a = model.encode_control(observed(1), first.memory_hidden, old, teacher_control=q)
    b = model.encode_control(observed(1), first.memory_hidden, changed, teacher_control=q)
    assert torch.equal(a.manager_log_prob, b.manager_log_prob)
    assert torch.equal(a.value, b.value) and torch.equal(a.encoded.hidden, b.encoded.hidden)
    assert a.control_state.intent.preference == b.control_state.intent.preference == (0.25,) * 9
    assert a.control_state.intent.start_loop == 1 and a.control_state.intent.age == 0
    assert set(a.control["manager"]) == {"operation", "preference"}
    assert not hasattr(model, "manager_horizon_head") and not hasattr(
        model, "manager_confidence_head"
    )
    assert not hasattr(model, "manager_operation_head")
    with pytest.raises(ValueError, match="outside current causal domain"):
        model.encode_control(
            observed(1),
            first.memory_hidden,
            old,
            teacher_control={"manager": {"operation": "keep"}, "tasks": None},
        )


def test_stepwise_external_preference_requires_explicit_current_intervention():
    model, f, _ = model_fixture(persistence=False, manager_mode="external")
    override = ExternalOverride((0.4,) * 9, 18, 0.7, "first")
    first = model.encode_control(
        f.policy_observation(),
        control_state=HierarchyState("episode"),
        external_override=override,
        deterministic=True,
    )
    with pytest.raises(ValueError, match="fresh intervention"):
        model.encode_control(observed(1), first.memory_hidden, first.control_state)
    again = model.encode_control(
        observed(1),
        first.memory_hidden,
        first.control_state,
        external_override=replace(override, intervention_id="second"),
        teacher_control={"manager": None, "tasks": None},
    )
    assert again.control_state.intent.start_loop == 1 and float(again.manager_log_prob) == 0


def test_stepwise_task_interface_ignores_commitment_age_and_confidence_fields():
    model, f, _ = model_fixture(persistence=False)
    a = model.encode_control(
        f.policy_observation(),
        control_state=HierarchyState("episode"),
        teacher_control=update_control(model),
    )
    intent = a.control_state.intent
    x = model._intent_features(intent)
    y = model._intent_features(replace(intent, age=700, commitment_horizon=1000, confidence=0.001))
    assert torch.equal(x, y)


def test_task_assignment_is_parallel_and_does_not_replace_intent_with_action():
    model, f, _ = model_fixture()
    out = model.encode_control(
        f.policy_observation(),
        control_state=HierarchyState("episode"),
        teacher_control=update_control(model),
    )
    assert len(out.control_state.tasks) == 2
    assert {t.slot for t in out.control_state.tasks} == {0, 1}
    assert all(t.parent_intent == 1 for t in out.control_state.tasks)
    assert out.control_state.tasks[0].target_region is None
    assert set(out.control["manager"]) == {
        "operation",
        "preference",
        "commitment_horizon",
        "confidence",
    }
    assert not any(k in out.control["manager"] for k in ("ability", "tag", "target_unit"))


@pytest.mark.parametrize("feedback", [True, False])
def test_declared_task_deadline_expires_without_fabricating_gameplay_confirmation(feedback):
    model, f, _ = model_fixture(use_feedback=feedback)
    a = model.encode_control(
        f.policy_observation(),
        control_state=HierarchyState("episode"),
        teacher_control=update_control(model),
    ).control_state
    before, _ = advance_state(a, 7, model.hierarchy_config)
    assert all(t.active for t in before.tasks)
    after, events = advance_state(before, 8, model.hierarchy_config)
    assert all(t.status == "timeout" and not t.active and t.progress == 0 for t in after.tasks)
    assert not any(e.status == "confirmed" for e in events)
    assert len(events) == (2 if feedback else 0)
    again, events = advance_state(after, 9, model.hierarchy_config)
    assert not events and all(t.status == "timeout" for t in again.tasks)


def test_external_override_is_continuous_auditable_and_has_no_manager_density():
    model, f, _ = model_fixture()
    override = ExternalOverride(
        tuple(0.123 for _ in model.hierarchy_config.preference_names), 13, 0.9, "paired-z-1"
    )
    out = model.encode_control(
        f.policy_observation(),
        control_state=HierarchyState("episode"),
        external_override=override,
        teacher_control={"manager": None, "tasks": update_control(model)["tasks"]},
    )
    intent = out.control_state.intent
    assert intent.source == "external" and intent.preference == override.preference
    assert intent.commitment_horizon == 13 and float(out.manager_log_prob) == 0.0
    assert out.control["manager"] is None


def test_external_only_manager_requires_initial_override_and_preserves_it():
    model, f, _ = model_fixture(manager_mode="external")
    with pytest.raises(ValueError, match="initial intervention"):
        model.encode_control(f.policy_observation(), control_state=HierarchyState("episode"))
    override = ExternalOverride((0.3,) * 9, 9, 0.8, "z-1")
    a = model.encode_control(
        f.policy_observation(),
        control_state=HierarchyState("episode"),
        external_override=override,
        deterministic=True,
    )
    b = model.encode_control(
        observed(20),
        a.memory_hidden,
        a.control_state,
        teacher_control={"manager": None, "tasks": [{"operation": "keep"}] * 2},
    )
    assert (
        b.control_state.intent.preference == override.preference
        and b.control_state.intent.age == 20
    )


def test_terminate_and_new_update_have_explicit_revisions():
    model, f, _ = model_fixture()
    a = model.encode_control(
        f.policy_observation(),
        control_state=HierarchyState("episode"),
        teacher_control=update_control(model),
    )
    q = {"manager": {"operation": "terminate"}, "tasks": [{"operation": "keep"}] * 2}
    b = model.encode_control(observed(8), a.memory_hidden, a.control_state, teacher_control=q)
    assert (
        not b.control_state.intent.active
        and b.control_state.intent.update_or_termination == "terminate"
    )
    c = model.encode_control(
        observed(16), b.memory_hidden, b.control_state, teacher_control=update_control(model)
    )
    assert c.control_state.intent.active and c.control_state.intent.revision == 2


def test_commitment_rejects_untriggered_early_update():
    model, f, _ = model_fixture()
    a = model.encode_control(
        f.policy_observation(),
        control_state=HierarchyState("episode"),
        teacher_control=update_control(model),
    )
    # A faster Manager test clock is explicitly selected; horizon remains 8.
    model.hierarchy_config = replace(model.hierarchy_config, manager_period=4)
    with pytest.raises(ValueError, match="outside current causal domain"):
        model.encode_control(
            observed(4), a.memory_hidden, a.control_state, teacher_control=update_control(model)
        )


def test_feedback_acceptance_is_distinct_from_confirmation_and_has_causal_evidence():
    model, f, _ = model_fixture()
    a = model.encode_control(
        f.policy_observation(),
        control_state=HierarchyState("episode"),
        teacher_control=update_control(model),
    )
    task = a.control_state.tasks[0]
    e = ExecutionFeedback("rpc-1", task.task_id, "accepted", 1, "dispatch-receipt-hash")
    q = {"manager": {"operation": "keep"}, "tasks": [{"operation": "keep"}] * 2}
    b = model.encode_control(
        observed(1), a.memory_hidden, a.control_state, feedback=(e,), teacher_control=q
    )
    assert b.control_state.tasks[0].status == "accepted" and b.control_state.tasks[0].progress == 0
    assert b.control_state.tasks[0].active
    confirmed = ExecutionFeedback(
        "effect-1", task.task_id, "confirmed", 2, "visible-effect-proof", 1.0
    )
    c = model.encode_control(
        observed(2), b.memory_hidden, b.control_state, feedback=(confirmed,), teacher_control=q
    )
    assert c.control_state.tasks[0].status == "confirmed" and not c.control_state.tasks[0].active


def test_repeated_commands_keep_aggregate_acceptance_and_record_new_receipts():
    model, f, _ = model_fixture()
    a = model.encode_control(
        f.policy_observation(),
        control_state=HierarchyState("episode"),
        teacher_control=update_control(model),
    ).control_state
    task_id = a.tasks[0].task_id
    first = (
        ExecutionFeedback("d1", task_id, "dispatched", 1, "dispatch1"),
        ExecutionFeedback("a1", task_id, "accepted", 1, "response1"),
    )
    state, _ = advance_state(a, 1, model.hierarchy_config, first)
    second = (
        ExecutionFeedback("d2", task_id, "dispatched", 2, "dispatch2"),
        ExecutionFeedback("a2", task_id, "accepted", 2, "response2"),
    )
    state, events = advance_state(state, 2, model.hierarchy_config, second)
    assert state.tasks[0].status == "accepted" and state.tasks[0].active
    assert state.tasks[0].last_feedback_loop == 2 and len(events) == 2
    assert set(state.consumed_feedback) == {"d2", "a2"}
    assert state.feedback_watermark_loop == 2
    old = ExecutionFeedback("old", task_id, "accepted", 1, "oldresponse")
    with pytest.raises(ValueError, match="event time"):
        advance_state(state, 3, model.hierarchy_config, (old,))


def test_feedback_recovery_state_stays_bounded_over_a_long_task():
    model, f, _ = model_fixture()
    model.hierarchy_config = replace(model.hierarchy_config, task_horizons=(10000, 20000))
    q = update_control(model)
    for task in q["tasks"]:
        task["horizon"] = 10000
    state = model.encode_control(
        f.policy_observation(), control_state=HierarchyState("episode"), teacher_control=q
    ).control_state
    for loop in range(1, 1001):
        task_id = state.tasks[0].task_id
        state, _ = advance_state(
            state,
            loop,
            model.hierarchy_config,
            (
                ExecutionFeedback(f"d:{loop}", task_id, "dispatched", loop, f"req:{loop}"),
                ExecutionFeedback(f"a:{loop}", task_id, "accepted", loop, f"resp:{loop}"),
            ),
        )
        assert len(state.consumed_feedback) == 2
    assert state.feedback_watermark_loop == 1000
    assert len(json.dumps(state.to_dict())) < 2200
    restored = HierarchyState.from_dict(
        json.loads(json.dumps(state.to_dict())), model.hierarchy_config
    )
    assert restored == state
    with pytest.raises(ValueError, match="watermark"):
        advance_state(
            restored,
            1001,
            model.hierarchy_config,
            (ExecutionFeedback("a:1", task_id, "accepted", 1, "resp:1"),),
        )


@pytest.mark.parametrize("kind", ["future", "duplicate", "unknown", "empty_evidence", "regression"])
def test_invalid_feedback_is_rejected(kind):
    model, f, _ = model_fixture()
    a = model.encode_control(
        f.policy_observation(),
        control_state=HierarchyState("episode"),
        teacher_control=update_control(model),
    )
    task = a.control_state.tasks[0]
    e = ExecutionFeedback("e", task.task_id, "accepted", 1, "proof")
    state, _ = advance_state(a.control_state, 1, model.hierarchy_config, (e,))
    candidate = dict(
        future=replace(e, event_id="next", loop=3),
        duplicate=e,
        unknown=replace(e, event_id="next", task_id="no-task"),
        empty_evidence=replace(e, event_id="next", evidence_id=""),
        regression=replace(e, event_id="next", status="dispatched"),
    )[kind]
    with pytest.raises(ValueError):
        advance_state(state, 2, model.hierarchy_config, (candidate,))


def test_feedback_ablation_removes_outcomes_and_event_clock_triggers():
    model, f, _ = model_fixture(use_feedback=False)
    a = model.encode_control(
        f.policy_observation(),
        control_state=HierarchyState("episode"),
        teacher_control=update_control(model),
    )
    event = ExecutionFeedback("e", a.control_state.tasks[0].task_id, "rejected", 1, "proof")
    kwargs = dict(teacher_control={"manager": None, "tasks": None})
    x = model.encode_control(
        observed(1), a.memory_hidden, a.control_state, feedback=(event,), **kwargs
    )
    y = model.encode_control(observed(1), a.memory_hidden, a.control_state, **kwargs)
    assert torch.equal(x.encoded.hidden, y.encoded.hidden)
    assert x.control == y.control and x.control_state.tasks == y.control_state.tasks
    assert x.control_state.tasks[0].status == "active"


def test_body_phase_advances_age_and_tasks_without_manager_decision():
    model, f, _ = model_fixture()
    a = model.encode_control(
        f.policy_observation(),
        control_state=HierarchyState("episode"),
        teacher_control=update_control(model),
    )
    b = model.encode_control(
        observed(12),
        a.memory_hidden,
        a.control_state,
        allow_manager=False,
        teacher_control={"manager": None, "tasks": [{"operation": "keep"}] * 2},
    )
    assert b.control_state.intent.age == 12 and b.control_state.intent.revision == 1
    assert b.control["manager"] is None and float(b.manager_log_prob) == 0


@pytest.mark.parametrize("arm", ["flat", "direct_task", "persistent_intent"])
def test_actor_rejects_future_or_undeclared_observation_inputs(arm):
    model, f, _ = model_fixture(arm)
    for key in ("outcome", "future", "raw_tags", "expert_intent"):
        with pytest.raises(ValueError, match="schema drift"):
            model.encode_control(
                dict(f.policy_observation(), **{key: 1}),
                control_state=HierarchyState("episode"),
                deterministic=True,
            )


def test_missing_or_extra_control_labels_fail_closed():
    model, f, _ = model_fixture()
    initial = HierarchyState("episode")
    with pytest.raises(ValueError, match="missing"):
        model.encode_control(
            f.policy_observation(),
            control_state=initial,
            teacher_control={"manager": None, "tasks": None},
            deterministic=True,
        )
    q = update_control(model)
    q["manager"]["future_win"] = 1
    with pytest.raises(ValueError, match="schema drift"):
        model.encode_control(f.policy_observation(), control_state=initial, teacher_control=q)


def test_json_native_state_roundtrip_and_clock_corruption_rejection():
    model, f, _ = model_fixture()
    a = model.encode_control(
        f.policy_observation(),
        control_state=HierarchyState("episode"),
        teacher_control=update_control(model),
    ).control_state
    restored = HierarchyState.from_dict(json.loads(json.dumps(a.to_dict())), model.hierarchy_config)
    assert a == restored
    broken = a.to_dict()
    broken["intent"]["age"] = 19
    with pytest.raises(ValueError, match="actual elapsed"):
        HierarchyState.from_dict(broken, model.hierarchy_config)


def prepare_session(model, profile):
    model.eval()
    with torch.no_grad():
        model.act_head.weight.zero_()
        model.act_head.bias.copy_(torch.tensor([10.0, -10.0]))
        model.delay_head.weight.zero_()
        model.delay_head.bias.fill_(-10.0)
        model.delay_head.bias[2] = 10.0
        model.fixed_heads["bundle"].weight.zero_()
        model.fixed_heads["bundle"].bias.copy_(torch.tensor([-10.0, 10.0]))
        model.ability_head.weight.zero_()
        model.ability_head.bias.fill_(-10.0)
        model.ability_head.bias[model.ability_index[11]] = 10.0
        model.end_head.weight.zero_()
        model.end_head.bias.fill_(20.0)
    session = HierarchyPolicySession(model, profile, deterministic=True)
    session.reset("episode")
    return session


@pytest.mark.parametrize(
    "arm,kwargs",
    [
        ("flat", {}),
        ("direct_task", {}),
        ("persistent_intent", {}),
        ("flat", {"memory_mode": "feedforward"}),
        ("persistent_intent", {"commitment_mode": "fixed_dwell", "fixed_commitment_horizon": 10}),
    ],
)
def test_session_joint_logprob_matches_complete_teacher_replay(arm, kwargs):
    model, f, p = model_fixture(arm, **kwargs)
    session = prepare_session(model, p)
    initial = session.control_state
    decision, delay = session.choose_schedule(f.policy_observation(), 8)
    body_frame, *_ = setup(loop=delay)
    machine = coarse_sample(
        body_frame,
        lambda head, options, completed: session.choose_body(
            body_frame.policy_observation(), head, options, completed
        ),
        p,
    )
    trace = deepcopy(session.audit_trace)
    a = model.encode_control(
        f.policy_observation(), control_state=initial, teacher_control=trace["scheduler_control"]
    )
    act, delays = model.scheduler(a.encoded, decision)
    lp = (
        a.control_log_prob
        + act.log_prob(torch.tensor(("ACT", "WAIT").index(decision)))
        + delays.log_prob(torch.tensor(delay))
    )
    hidden = model.consume_schedule(a.encoded, delay)
    b = model.encode_control(
        body_frame.policy_observation(),
        hidden,
        a.control_state,
        teacher_control=trace["body_control"],
        allow_manager=False,
    )
    encoded, association_lp, _, _ = model.select_execution_task(
        b, teacher_task_id=trace["body_execution_task"]
    )
    lp = lp + b.control_log_prob + association_lp
    prefix = encoded.hidden
    for token in trace["body_tokens"]:
        head, options, value = token["head"], token["options"], token["value"]
        dist = model.distribution(encoded, prefix, head, options)
        term = (
            dist.log_prob(value)
            if head == "target_position"
            else dist.log_prob(torch.tensor(options.index(value)))
        )
        lp = lp + term
        prefix = model.consume(encoded, prefix, Token(head, value))
    assert abs(float(lp) - session.action_log_prob) < 2e-5
    assert torch.equal(prefix, session.hidden)
    assert session.control_state == b.control_state and machine.protocol_actions()


@pytest.mark.parametrize("frozen", ["value_only", "manager_rl"])
def test_frozen_backbone_long_63_source_multi_command_actor_teacher_parity(frozen):
    model, _, _ = model_fixture()
    f, p, _, _ = setup(tags=tuple(range(1, 71)))
    session = prepare_session(model, p)
    with torch.no_grad():
        model.end_head.bias.fill_(-100.0)
    # Construct a genuine two-command causal trace, then fit two tiny linear
    # fixture separators so the unmodified sampler emits that trace. This tests
    # full long-bundle arithmetic, not a synthetic repeated invalid token list.
    a = model.encode_control(
        f.policy_observation(), control_state=HierarchyState("episode"), deterministic=True
    )
    h = model.consume_schedule(a.encoded, 2)
    body, *_ = setup(loop=2, tags=tuple(range(1, 71)))
    b = model.encode_control(
        body.policy_observation(), h, a.control_state, deterministic=True, allow_manager=False
    )
    encoded, _, _, _ = model.select_execution_task(b, deterministic=True)
    prefix = encoded.hidden
    ability_prefixes = []
    bundle_prefixes = []
    count = 0

    def build(head, options, completed):
        nonlocal prefix, count
        dist = model.distribution(encoded, prefix, head, options)
        if head == "ability":
            ability_prefixes.append(prefix.detach().clone())
            value = 11 if count == 0 else 21
        elif head == "bundle":
            bundle_prefixes.append(prefix.detach().clone())
            value = options[0] if count == 0 else options[1]
            count += 1
        elif head == "target_position":
            value = dist.choose(True)
        else:
            value = options[int(dist.logits.argmax())]
        prefix = model.consume(encoded, prefix, Token(head, value))
        return value

    wanted = coarse_sample(body, build, p)
    assert len(wanted.commands) == 2 and all(len(c[1]) == 63 for c in wanted.commands)

    def separator(head, indices, vectors):
        x, y = vectors
        delta = y - x
        norm = delta.square().sum()
        assert float(norm) > 0
        direction = delta * 20.0 / norm
        midpoint = (x + y) / 2
        with torch.no_grad():
            head.weight.zero_()
            head.bias.fill_(-1000.0)
            head.weight[indices[0]] = -direction
            head.weight[indices[1]] = direction
            offset = torch.dot(direction, midpoint)
            head.bias[indices[0]] = offset
            head.bias[indices[1]] = -offset

    separator(
        model.ability_head, (model.ability_index[11], model.ability_index[21]), ability_prefixes
    )
    separator(model.fixed_heads["bundle"], (0, 1), bundle_prefixes)
    allowed = {"value"} if frozen == "value_only" else {"manager", "value"}
    for group, values in model.parameter_groups().items():
        for _, parameter in values:
            parameter.requires_grad_(group in allowed)
    session.reset("episode")
    session.choose_schedule(f.policy_observation(), 8)
    actual = coarse_sample(
        body,
        lambda head, options, completed: session.choose_body(
            body.policy_observation(), head, options, completed
        ),
        p,
    )
    assert actual.commands == wanted.commands
    trace = session.audit_trace
    a = model.encode_control(
        f.policy_observation(),
        control_state=HierarchyState("episode"),
        teacher_control=trace["scheduler_control"],
    )
    act, delay = model.scheduler(a.encoded, "ACT")
    lp = a.control_log_prob + act.log_prob(torch.tensor(0))
    lp = lp + delay.log_prob(torch.tensor(2))
    h = model.consume_schedule(a.encoded, 2)
    b = model.encode_control(
        body.policy_observation(),
        h,
        a.control_state,
        teacher_control=trace["body_control"],
        allow_manager=False,
    )
    encoded, assoc, _, _ = model.select_execution_task(
        b, teacher_task_id=trace["body_execution_task"]
    )
    lp = lp + b.control_log_prob
    lp = lp + assoc
    prefix = encoded.hidden
    for token in trace["body_tokens"]:
        head, options, value = token["head"], token["options"], token["value"]
        distribution = model.distribution(encoded, prefix, head, options)
        term = (
            distribution.log_prob(value)
            if head == "target_position"
            else distribution.log_prob(torch.tensor(options.index(value)))
        )
        lp = lp + term
        prefix = model.consume(encoded, prefix, Token(head, value))
    assert float(lp) == session.action_log_prob
    assert torch.equal(prefix, session.hidden)


def test_partial_body_session_recovery_keeps_next_distribution_and_reset():
    model, f, p = model_fixture()
    session = prepare_session(model, p)
    session.choose_schedule(f.policy_observation(), 8)
    body, *_ = setup(loop=2)
    session.choose_body(body.policy_observation(), "ability", (11, 21), ())
    snapshot = session.state_dict()
    restored = HierarchyPolicySession(model, p, deterministic=True)
    restored.load_state_dict(snapshot)
    first = session.choose_body(body.policy_observation(), "source", (0, 1), ())
    second = restored.choose_body(body.policy_observation(), "source", (0, 1), ())
    assert first == second and session.action_log_prob == restored.action_log_prob
    assert torch.equal(session.hidden, restored.hidden)
    assert session.audit_trace == restored.audit_trace
    restored.reset("new-episode")
    assert restored.hidden is None and restored.control_state == HierarchyState("new-episode")
    assert restored.audit_trace is None and restored.pending_override is None


def test_recovery_rejects_wrong_identity_shapes_and_nonfinite_state():
    model, f, p = model_fixture()
    session = prepare_session(model, p)
    session.choose_schedule(f.policy_observation(), 8)
    for mutate in (
        lambda d: d.update(hierarchy_config_sha256="0" * 64),
        lambda d: d.update(episode="wrong"),
        lambda d: d.update(hidden=torch.zeros(31)),
        lambda d: d.update(hidden=torch.full((32,), float("nan"))),
    ):
        broken = session.state_dict()
        mutate(broken)
        with pytest.raises(ValueError):
            HierarchyPolicySession(model, p, deterministic=True).load_state_dict(broken)


def test_bootstrap_value_does_not_sample_or_advance_controls():
    model, f, _ = model_fixture()
    rng = torch.get_rng_state().clone()
    values = model.evaluate_values(f.policy_observation(), control_state=HierarchyState("episode"))
    assert all(torch.isfinite(v) and v.ndim == 0 for v in values)
    assert torch.equal(rng, torch.get_rng_state())


def test_critic_reads_existing_intent_and_tasks_before_current_choices():
    model, f, _ = model_fixture()
    prior = HierarchyState("episode")
    first = model.encode_control(
        f.policy_observation(), control_state=prior, teacher_control=update_control(model)
    )
    # New actions cannot change the current-state baseline.
    different = update_control(model)
    different["manager"]["preference"] = [1.0] * 9
    second = model.encode_control(
        f.policy_observation(), control_state=prior, teacher_control=different
    )
    assert torch.equal(first.value, second.value)
    values_a = model.evaluate_values(observed(1), control_state=first.control_state)
    values_b = model.evaluate_values(observed(1), control_state=second.control_state)
    assert not torch.equal(values_a[0], values_b[0])
    state = first.control_state
    result = model.encode_control(
        observed(1), control_state=state, teacher_control={"manager": None, "tasks": None}
    )
    assert torch.equal(values_a[0], result.value)
    assert state.intent.age == 0  # bootstrap did not mutate preceding state


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(arm="bad"),
        dict(task_slots=0),
        dict(manager_period=1, task_period=2),
        dict(preference_names=("economy", "economy")),
        dict(preference_levels=(-1.0, 1.0)),
        dict(use_feedback=1),
        dict(confidence_event="future_outcome"),
        dict(memory_mode="bad"),
        dict(arm="direct_task", memory_mode="feedforward"),
        dict(commitment_mode="bad"),
        dict(fixed_commitment_horizon=0),
        dict(commitment_mode="fixed_dwell"),
        dict(arm="persistent_intent", commitment_mode="fixed_dwell", persistence=False),
    ],
)
def test_invalid_hierarchy_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        HierarchyConfig(**kwargs)


def test_fixed_dwell_feedback_cannot_preempt_or_restart_and_exact_boundary_updates():
    model, f, _ = model_fixture(commitment_mode="fixed_dwell", fixed_commitment_horizon=10)
    first = model.encode_control(
        f.policy_observation(),
        control_state=HierarchyState("episode"),
        teacher_control=update_control(model),
    )
    assert first.manager_decision and first.control_state.intent.commitment_horizon == 10
    assert set(first.control["manager"]) == {"operation", "preference"}
    for name in ("manager_operation_head", "manager_horizon_head", "manager_confidence_head"):
        assert not hasattr(model, name)
        assert not any(n.startswith(name) for n, _ in model.parameter_groups()["manager"])
    feedback = ExecutionFeedback(
        "accepted", first.control_state.tasks[0].task_id, "accepted", 1, "observed-response"
    )
    assert not model.manager_decision_eligible(first.control_state, 1, feedback=(feedback,))
    keep = {"manager": {"operation": "keep"}, "tasks": [{"operation": "keep"}] * 2}
    held = model.encode_control(
        observed(1),
        first.memory_hidden,
        first.control_state,
        feedback=(feedback,),
        teacher_control=keep,
    )
    assert held.control_state.intent.start_loop == 0 and held.control_state.intent.age == 1
    assert not held.manager_decision and float(held.manager_log_prob) == 0.0
    with pytest.raises(ValueError, match="outside current causal domain"):
        model.encode_control(
            observed(2),
            held.memory_hidden,
            held.control_state,
            teacher_control={"manager": update_control(model)["manager"], "tasks": None},
        )
    # At task deadlines real clock feedback is still unable to replace Intent.
    expired_tasks = model.encode_control(
        observed(8), held.memory_hidden, held.control_state, teacher_control=keep
    )
    assert expired_tasks.control_state.intent.start_loop == 0
    before = model.encode_control(
        observed(9),
        expired_tasks.memory_hidden,
        expired_tasks.control_state,
        teacher_control={"manager": {"operation": "keep"}, "tasks": None},
    )
    assert not before.manager_decision and before.control_state.intent.age == 9
    assert model.manager_decision_eligible(before.control_state, 10)
    boundary = model.encode_control(
        observed(10),
        before.memory_hidden,
        before.control_state,
        teacher_control={"manager": update_control(model)["manager"], "tasks": None},
    )
    assert boundary.manager_decision and boundary.control_state.intent.start_loop == 10
    assert boundary.control_state.intent.age == 0 and boundary.control_state.intent.revision == 2
    assert boundary.control_state.intent.commitment_horizon == 10
    with pytest.raises(ValueError, match="outside current causal domain"):
        model.encode_control(
            observed(10),
            before.memory_hidden,
            before.control_state,
            teacher_control={"manager": {"operation": "terminate"}, "tasks": None},
        )
    sampled = model.encode_control(
        observed(10), before.memory_hidden, before.control_state, deterministic=True
    )
    replayed = model.encode_control(
        observed(10), before.memory_hidden, before.control_state, teacher_control=sampled.control
    )
    assert sampled.control_state == replayed.control_state
    assert torch.equal(sampled.control_log_prob, replayed.control_log_prob)


def test_fixed_dwell_external_override_is_an_audited_new_declared_dwell():
    model, f, _ = model_fixture(commitment_mode="fixed_dwell", fixed_commitment_horizon=10)
    first = model.encode_control(
        f.policy_observation(),
        control_state=HierarchyState("episode"),
        teacher_control=update_control(model, task_operation="keep"),
    )
    override = ExternalOverride((0.4,) * 9, 3, 0.7, "explicit-user-intervention")
    current = model.encode_control(
        observed(5),
        first.memory_hidden,
        first.control_state,
        external_override=override,
        teacher_control={"manager": None, "tasks": [{"operation": "keep"}] * 2},
    )
    assert current.control_state.intent.source == "external"
    assert (
        current.control_state.intent.start_loop == 5
        and current.control_state.intent.commitment_horizon == 3
    )
    assert not current.manager_decision and float(current.manager_log_prob) == 0.0
    assert not model.manager_decision_eligible(current.control_state, 7)
    assert model.manager_decision_eligible(current.control_state, 8)
    boundary = model.encode_control(
        observed(8),
        current.memory_hidden,
        current.control_state,
        teacher_control={"manager": update_control(model)["manager"], "tasks": None},
    )
    assert boundary.control_state.intent.start_loop == 8
    assert boundary.control_state.intent.commitment_horizon == 10
    assert boundary.control_state.intent.source == "learned"


def test_feedforward_scheduler_and_bootstrap_discard_previous_memory_but_body_keeps_delay():
    model, f, _ = model_fixture("flat", memory_mode="feedforward")
    initial = HierarchyState("episode")
    hidden = torch.linspace(-2, 2, model.config.hidden_size)
    a = model.encode_control(f.policy_observation(), hidden, initial)
    b = model.encode_control(f.policy_observation(), -hidden * 10, initial)
    assert torch.equal(a.memory_hidden, b.memory_hidden)
    assert torch.equal(a.value, b.value)
    assert torch.equal(model.scheduler(a.encoded)[0].logits, model.scheduler(b.encoded)[0].logits)
    values_a = model.evaluate_values(f.policy_observation(), hidden, initial)
    values_b = model.evaluate_values(f.policy_observation(), -hidden * 10, initial)
    assert all(torch.equal(x, y) for x, y in zip(values_a, values_b))
    assert torch.equal(values_a[0], a.value)
    with pytest.raises(ValueError, match="explicit encode_control"):
        model.encode(f.policy_observation(), hidden)
    # FF ablates history between decisions; delay and within-ACT state survive.
    short = model.consume_schedule(a.encoded, 0)
    long = model.consume_schedule(a.encoded, 3)
    x = model.encode_control(observed(3), short, a.control_state, allow_manager=False)
    y = model.encode_control(observed(3), long, a.control_state, allow_manager=False)
    assert not torch.equal(x.memory_hidden, y.memory_hidden)
    assert torch.equal(x.encoded.entities, y.encoded.entities)
    groups = model.parameter_groups()
    assert groups["backbone"] and groups["decoder"] and groups["value"]
    assert not groups["manager"] and not groups["task"]


def test_feedforward_session_next_schedule_is_independent_of_previous_body_prefix():
    model, _, profile = model_fixture("flat", memory_mode="feedforward")
    a = prepare_session(model, profile)
    b = prepare_session(model, profile)
    a.hidden = torch.arange(model.config.hidden_size, dtype=torch.float32)
    b.hidden = -a.hidden * 3
    first = a.choose_schedule(observed(5), model.config.maximum_delay)
    second = b.choose_schedule(observed(5), model.config.maximum_delay)
    assert first == second and a.audit_trace == b.audit_trace
    assert torch.equal(a.hidden, b.hidden)
    saved = a.state_dict()
    recovered = HierarchyPolicySession(model, profile, deterministic=True)
    recovered.load_state_dict(saved)
    assert recovered.choose_schedule(observed(8), model.config.maximum_delay) == a.choose_schedule(
        observed(8), model.config.maximum_delay
    )
    assert recovered.audit_trace == a.audit_trace


@pytest.mark.parametrize(
    "comparator,phase",
    [
        ("feedforward", "JOINT_RL"),
        ("feedforward", "VALUE_ONLY"),
        ("fixed_dwell", "JOINT_RL"),
        ("fixed_dwell", "MANAGER_RL"),
    ],
)
def test_comparators_actual_rollout_scorer_and_optimizer_phase_consistency(comparator, phase):
    from poorstar.training import PPOLearner
    from poorstar.training.config import model_config_sha256
    from poorstar.training.returns import targets
    from poorstar.training.scorer import HierarchyScorer
    from tests.test_training_system_foundation import sampled_sequence, system

    arm = "flat" if comparator == "feedforward" else "persistent_intent"
    old, profile, _ = system(arm, phase)
    hierarchy = replace(
        old.model.hierarchy_config,
        **(
            {"memory_mode": "feedforward"}
            if comparator == "feedforward"
            else {"commitment_mode": "fixed_dwell", "fixed_commitment_horizon": 2}
        ),
    )
    model = HierarchicalPolicy(old.model.config, hierarchy)
    model.load_state_dict(
        {n: v for n, v in old.model.state_dict().items() if n in model.state_dict()}
    )
    identity = replace(
        old.identity, model_config_sha256=model_config_sha256(model.config, hierarchy)
    )
    learner = PPOLearner(model, HierarchyScorer(model, profile), identity, old.config, phase=phase)
    model.eval()
    sequence = sampled_sequence(learner, profile)
    if comparator == "fixed_dwell":
        assert [s.manager_decision for s in sequence.steps] == [True, False, True]
        assert sequence.steps[1].payload.trace["scheduler_control"]["manager"] == {
            "operation": "keep"
        }
        assert targets(sequence, learner.config, manager=True).indices == (0, 2)
    else:
        assert not any(s.manager_decision for s in sequence.steps)
    hidden = None
    for step in sequence.steps:
        score = learner.scorer(step, hidden)
        hidden = score.next_hidden
        assert float(score.log_prob) == step.old_log_prob
        assert float(score.manager_log_prob) == step.old_manager_log_prob
        assert float(score.value) == step.old_value
    result = learner.update([sequence])
    assert result["optimizer_steps"] == 1 and result["changed_parameters"]
