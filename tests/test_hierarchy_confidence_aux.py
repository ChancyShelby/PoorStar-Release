"""Tests: test hierarchy confidence aux."""

from dataclasses import replace

import pytest
import torch

from poorstar.evaluation.confidence import ConfidenceTarget, ConfidenceWindow
from poorstar.hierarchy import hierarchy_bc_objective
from tests.test_hierarchy_bc import records_fixture
from tests.test_hierarchy_runtime import model_fixture


@pytest.fixture(scope="module", autouse=True)
def threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def target_for(record, *, value=1.0, mask=True):
    manager = record.scheduler_control["manager"]
    frame = record.decision.scheduler
    window = ConfidenceWindow(
        frame.key.episode,
        1,
        frame.key.loop,
        frame.key.loop + manager.get("commitment_horizon", 8),
        frame.key.observation_sha256,
    )
    return ConfidenceTarget(
        window.sha256,
        value if mask else None,
        mask,
        "ENGINE_SCHEMA_FIXTURE_NOT_CAPABILITY",
        ("b" * 64,),
    )


def test_confidence_aux_is_masked_sum_count_and_future_target_does_not_change_action_losses():
    model, _, _ = model_fixture()
    records, profile = records_fixture(model)
    positive = replace(records[0], confidence_target=target_for(records[0]))
    negative = replace(records[0], confidence_target=target_for(records[0], value=0.0))
    a = hierarchy_bc_objective(model, [(positive, records[1])], profile, collect_metrics=False)
    b = hierarchy_bc_objective(model, [(negative, records[1])], profile, collect_metrics=False)
    assert a.denominators["confidence_aux"] == b.denominators["confidence_aux"] == 1
    assert all(
        torch.equal(a.head_sums[k], b.head_sums[k]) for k in a.head_sums if k != "confidence_aux"
    )
    assert not torch.equal(a.head_sums["confidence_aux"], b.head_sums["confidence_aux"])
    model.zero_grad()
    a.head_sums["confidence_aux"].backward()
    assert model.manager_confidence_head.weight.grad.abs().sum() > 0
    assert model.ability_head.weight.grad is None
    censored = replace(records[0], confidence_target=target_for(records[0], mask=False))
    c = hierarchy_bc_objective(model, [(censored, records[1])], profile, collect_metrics=False)
    assert c.denominators["confidence_aux"] == 0 and c.head_sums["confidence_aux"] == 0


@pytest.mark.parametrize("fault", ["observation", "revision", "window_end"])
def test_confidence_target_from_another_observation_or_commitment_is_rejected(fault):
    model, _, _ = model_fixture()
    records, profile = records_fixture(model)
    frame = records[0].decision.scheduler
    window = ConfidenceWindow(
        "episode",
        2 if fault == "revision" else 1,
        0,
        16 if fault == "window_end" else 8,
        "c" * 64 if fault == "observation" else frame.key.observation_sha256,
    )
    target = ConfidenceTarget(window.sha256, 1.0, True, "fixture", ("b" * 64,))
    with pytest.raises(ValueError, match="different observation/commitment/revision"):
        hierarchy_bc_objective(
            model, [(replace(records[0], confidence_target=target), records[1])], profile
        )


@pytest.mark.parametrize("mode", ["direct_task", "stepwise", "fixed_dwell"])
def test_auxiliary_target_requires_a_new_learned_full_lifecycle_commitment(mode):
    opts = {}
    arm = "direct_task" if mode == "direct_task" else "persistent_intent"
    if mode == "stepwise":
        opts["persistence"] = False
    if mode == "fixed_dwell":
        opts["commitment_mode"] = "fixed_dwell"
    model, _, _ = model_fixture(arm, **opts)
    records, profile = records_fixture(model)
    baseline, _, _ = model_fixture()
    source, _ = records_fixture(baseline)
    with pytest.raises(ValueError, match="full-lifecycle"):
        hierarchy_bc_objective(
            model,
            [(replace(records[0], confidence_target=target_for(source[0])), records[1])],
            profile,
        )


def test_burn_in_confidence_target_does_not_add_loss_denominator():
    model, _, _ = model_fixture()
    records, profile = records_fixture(model)
    row = replace(records[0].decision, burn_in=True, loss=False)
    initial = replace(records[0], decision=row, confidence_target=target_for(records[0]))
    out = hierarchy_bc_objective(model, [(initial, records[1])], profile)
    assert "confidence_aux" not in out.denominators
