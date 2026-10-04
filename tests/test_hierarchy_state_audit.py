"""Tests: test hierarchy state audit."""

from copy import deepcopy
from dataclasses import replace

import pytest
import torch

from poorstar.hierarchy.bc import (
    HierarchyBCSequence,
    audit_hierarchy_bc_sequence,
    hierarchy_bc_objective,
)
from tests.test_hierarchy_bc import records_fixture
from tests.test_hierarchy_runtime import model_fixture


@pytest.fixture(autouse=True)
def threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


@pytest.mark.parametrize("arm", ["flat", "direct_task", "persistent_intent"])
def test_opt_in_state_audit_preserves_loss_gradients_parameters_and_rng(arm):
    model, *_ = model_fixture(arm)
    audited = deepcopy(model)
    records, profile = records_fixture(model)
    if arm == "flat":
        records = tuple(
            replace(r, scheduler_control=None, body_control=None, body_execution_task=None)
            for r in records
        )
    initial = {n: p.detach().clone() for n, p in model.named_parameters()}
    rng = torch.get_rng_state().clone()
    ordinary = hierarchy_bc_objective(model, [records], profile, collect_metrics=False)
    after = torch.get_rng_state().clone()
    torch.set_rng_state(rng)
    actual, ledger = audit_hierarchy_bc_sequence(
        audited, HierarchyBCSequence(records), profile, collect_metrics=False
    )
    assert torch.equal(torch.get_rng_state(), after)
    assert torch.equal(ordinary.loss, actual.loss) and ordinary.denominators == actual.denominators
    assert ordinary.counts == actual.counts
    assert all(torch.equal(v, actual.head_sums[h]) for h, v in ordinary.head_sums.items())
    ordinary.loss.backward()
    actual.loss.backward()
    for name, p in model.named_parameters():
        q = dict(audited.named_parameters())[name]
        assert torch.equal(p, initial[name]) and torch.equal(q, initial[name])
        assert (p.grad is None) == (q.grad is None)
        if p.grad is not None:
            assert torch.equal(p.grad, q.grad), name
    assert [e.phase for e in ledger] == ["scheduler", "body", "scheduler"]
    assert ledger[0].post_state_dict == ledger[1].prior_state_dict
    assert ledger[1].post_state_dict == ledger[2].prior_state_dict
    assert ledger[0].scheduler_loop == ledger[1].scheduler_loop


def test_audit_keeps_burnin_and_excludes_padding_without_mutating_prior_snapshot():
    model, *_ = model_fixture()
    records, profile = records_fixture(model)
    burn = replace(records[0], decision=replace(records[0].decision, burn_in=True, loss=False))
    padding = replace(records[1], decision=replace(records[1].decision, valid=False, loss=False))
    _, ledger = audit_hierarchy_bc_sequence(
        model, HierarchyBCSequence((burn, records[1], padding)), profile
    )
    assert len(ledger) == 3 and ledger[0].prior_state_dict["intent"] is None
    assert ledger[-1].post_state_dict["intent"] is not None
    assert ledger[0].prior_state_dict["intent"] is None
    with pytest.raises(ValueError, match="fresh phase ledger"):
        audit_hierarchy_bc_sequence(model, HierarchyBCSequence(records), profile, state_audit=[])
