"""Tests: test hierarchy bc."""

from dataclasses import replace

import pytest
import torch

from poorstar.hierarchy import (
    HierarchyBCRecord,
    HierarchyBCSequence,
    HierarchyState,
    hierarchy_bc_objective,
)
from poorstar.models.policy import RecurrentBCPolicy
from poorstar.training.foundation import bc_objective
from tests.test_coarse_bc import decision_rows, setup
from tests.test_hierarchy_runtime import model_fixture, update_control


@pytest.fixture(scope="module", autouse=True)
def threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def records_fixture(model):
    rows, profile, *_ = decision_rows()
    first = HierarchyBCRecord(
        rows[0],
        scheduler_control=update_control(model),
        body_control={"manager": None, "tasks": None},
        body_execution_task="episode:task:1",
        supervision_id="fixture-control-trace-0",
        supervision_kind="actor_control_trace",
    )
    second = HierarchyBCRecord(
        rows[1],
        scheduler_control={"manager": None, "tasks": None},
        supervision_id="fixture-control-trace-1",
        supervision_kind="actor_control_trace",
    )
    return (first, second), profile


def test_flat_adapter_is_identical_to_original_loss_counts_and_gradients():
    model, _, _ = model_fixture("flat")
    rows, profile, *_ = decision_rows()
    original = RecurrentBCPolicy(model.config)
    original.load_state_dict(
        {k: v for k, v in model.state_dict().items() if k in original.state_dict()}
    )
    a = bc_objective(original, [rows], profile, collect_metrics=False)
    b = hierarchy_bc_objective(
        model, [[HierarchyBCRecord(r) for r in rows]], profile, collect_metrics=False
    )
    assert torch.equal(a.loss, b.loss) and a.denominators == b.denominators
    assert all(a.counts[k] == b.counts[k] for k in a.counts)
    assert b.counts["explicit_control_phases"] == 3
    a.loss.backward()
    b.loss.backward()
    for k, p in original.named_parameters():
        q = dict(model.named_parameters())[k]
        assert (p.grad is None) == (q.grad is None)
        if p.grad is not None:
            assert torch.equal(p.grad, q.grad)


def test_feedforward_bc_matches_independent_original_decisions_with_full_act_prefix():
    model, _, _ = model_fixture("flat", memory_mode="feedforward")
    rows, profile, *_ = decision_rows()
    original = RecurrentBCPolicy(model.config)
    original.load_state_dict(
        {k: v for k, v in model.state_dict().items() if k in original.state_dict()}
    )
    # Independent decisions reset cross-decision memory; each ACT still performs
    # delay consumption, fresh body encode and ordinary structured teacher force.
    expected = bc_objective(original, [(row,) for row in rows], profile, collect_metrics=False)
    actual = hierarchy_bc_objective(
        model, [[HierarchyBCRecord(r) for r in rows]], profile, collect_metrics=False
    )
    assert torch.equal(expected.loss, actual.loss)
    assert expected.denominators == actual.denominators
    assert all(torch.equal(expected.head_sums[h], actual.head_sums[h]) for h in expected.head_sums)
    expected.loss.backward()
    actual.loss.backward()
    for name, parameter in original.named_parameters():
        other = dict(model.named_parameters())[name]
        assert (parameter.grad is None) == (other.grad is None)
        if parameter.grad is not None:
            assert torch.equal(parameter.grad, other.grad), name


@pytest.mark.parametrize("arm", ["direct_task", "persistent_intent"])
def test_explicit_hierarchy_bc_trains_controls_and_original_action_heads(arm):
    model, _, _ = model_fixture(arm)
    records, profile = records_fixture(model)
    out = hierarchy_bc_objective(model, [records], profile, collect_metrics=False)
    assert out.denominators["task_control"] == out.denominators["execution_task"] == 1
    assert out.denominators["act"] == out.denominators["delay"] == 2
    assert out.denominators["ability"] == 1
    if arm == "persistent_intent":
        assert out.denominators["manager_control"] == 1
    else:
        assert "manager_control" not in out.denominators
    out.loss.backward()
    for group, values in model.parameter_groups().items():
        if group == "value" or not values:
            continue
        assert any(p.grad is not None and bool(p.grad.abs().sum()) for _, p in values), group


def test_burn_in_padding_have_no_control_or_action_loss():
    model, _, _ = model_fixture()
    records, profile = records_fixture(model)
    burn = replace(records[0], decision=replace(records[0].decision, burn_in=True, loss=False))
    pad = HierarchyBCRecord(replace(records[1].decision, valid=False, loss=False))
    out = hierarchy_bc_objective(model, [(burn, records[1], pad)], profile)
    assert out.denominators == {"act": 1, "delay": 1}
    assert out.counts["burn_in_rows"] == out.counts["padding_rows"] == 1
    assert out.counts["explicit_control_phases"] == 3


def test_forced_commitment_keep_receipt_is_excluded_from_manager_bc_denominator():
    model, _, _ = model_fixture()
    rows, profile, *_ = decision_rows()
    q = update_control(model, task_operation="keep")
    q["manager"]["commitment_horizon"] = 16
    first = HierarchyBCRecord(
        replace(rows[0], decision="WAIT", delay=8, body=None, targets=(), burn_in=True, loss=False),
        scheduler_control=q,
        supervision_id="explicit-burn-in-control",
        supervision_kind="actor_control_trace",
    )
    later, *_ = setup(loop=8)
    forced = HierarchyBCRecord(
        replace(rows[1], scheduler=later),
        scheduler_control={"manager": {"operation": "keep"}, "tasks": [{"operation": "keep"}] * 2},
        supervision_id="explicit-forced-clock-receipt",
        supervision_kind="actor_control_trace",
    )
    out = hierarchy_bc_objective(model, [(first, forced)], profile)
    assert "manager_control" not in out.denominators
    assert out.denominators == {"act": 1, "delay": 1, "task_control": 1}


@pytest.mark.parametrize("kind", ["no_provenance", "no_scheduler", "no_body", "wrong_kind"])
def test_missing_high_level_truth_is_not_replaced_by_sampled_labels(kind):
    model, _, _ = model_fixture()
    records, profile = records_fixture(model)
    broken = dict(
        no_provenance=replace(records[0], supervision_id=None),
        no_scheduler=replace(records[0], scheduler_control=None),
        no_body=replace(records[0], body_control=None),
        wrong_kind=replace(records[0], supervision_kind="native_expert_intent"),
    )[kind]
    rng = torch.get_rng_state().clone()
    with pytest.raises(ValueError):
        hierarchy_bc_objective(model, [(broken, records[1])], profile)
    assert torch.equal(rng, torch.get_rng_state())


def test_mid_episode_window_requires_its_actual_control_snapshot():
    model, _, _ = model_fixture()
    records, profile = records_fixture(model)
    initial = HierarchyState("episode")
    first = records[0].decision
    start = model.encode_control(
        first.scheduler.policy_observation(),
        control_state=initial,
        teacher_control=records[0].scheduler_control,
    )
    hidden = model.consume_schedule(start.encoded, first.delay)
    body = model.encode_control(
        first.body.policy_observation(),
        hidden,
        start.control_state,
        teacher_control=records[0].body_control,
        allow_manager=False,
    )
    with pytest.raises(ValueError, match="causal initial control snapshot"):
        hierarchy_bc_objective(model, [(records[1],)], profile)
    window = HierarchyBCSequence((records[1],), body.control_state)
    out = hierarchy_bc_objective(model, [window], profile)
    assert out.denominators == {"act": 1, "delay": 1}
    assert body.control_state.intent.age == 2


def test_rejected_expert_body_retains_original_scheduler_and_control_accounting():
    from tests.test_coarse_bc import target

    model, _, _ = model_fixture()
    records, profile = records_fixture(model)
    rejected = replace(
        records[0],
        decision=replace(
            records[0].decision, targets=(target(records[0].decision.body, tags=(999,)),)
        ),
    )
    out = hierarchy_bc_objective(model, [(rejected, records[1])], profile)
    assert out.counts["rejected_bundles"] == 1
    assert out.denominators["act"] == 2 and out.denominators["manager_control"] == 1
    assert "ability" not in out.denominators
    assert out.counts["explicit_control_phases"] == 3


def test_wrong_snapshot_episode_is_rejected():
    model, _, _ = model_fixture()
    records, profile = records_fixture(model)
    with pytest.raises(ValueError, match="another episode"):
        hierarchy_bc_objective(
            model, [HierarchyBCSequence(records, HierarchyState("wrong"))], profile
        )
