"""Tests: test bc development."""

import random
from dataclasses import replace

import numpy as np
import pytest
import torch

from poorstar.cli.evaluation.bc import evaluate
from poorstar.data.bc.supervision import DISTAR_SKIP
from poorstar.data.bc.windows import WindowOrder, load_policy_checkpoint, save_checkpoint
from poorstar.models.policy import BCModelConfig, RecurrentBCPolicy
from poorstar.training.foundation import BCTrainer
from tests.test_coarse_bc import decision_rows


@pytest.fixture(autouse=True)
def threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def test_actor_only_checkpoint_load_preserves_training_rng_and_does_not_create_optimizer(
    tmp_path, monkeypatch
):
    rows, p, cat, _ = decision_rows()
    cfg = BCModelConfig.from_catalog(cat, p, hidden_size=32, maximum_delay=8)
    trainer = BCTrainer(
        RecurrentBCPolicy(cfg),
        p,
        learning_rate=0.001,
        supervision_policy=DISTAR_SKIP,
        feature_batch_size=4,
    )
    trainer.step([rows])
    binding = {"source": "frozen"}
    save_checkpoint(tmp_path / "one", trainer, WindowOrder(1, 37), binding)
    fresh = RecurrentBCPolicy(cfg)
    rng = (random.getstate(), np.random.get_state(), torch.get_rng_state().clone())

    def forbidden(*a, **k):
        raise AssertionError("evaluation constructed optimizer")

    monkeypatch.setattr(torch.optim, "AdamW", forbidden)
    loaded = load_policy_checkpoint(
        tmp_path / "one", fresh, binding, supervision_policy=DISTAR_SKIP, feature_batch_size=4
    )
    assert loaded["steps"] == 1 and not fresh.training
    assert all(torch.equal(v, trainer.model.state_dict()[n]) for n, v in fresh.state_dict().items())
    assert rng[0] == random.getstate() and np.array_equal(rng[1][1], np.random.get_state()[1])
    assert torch.equal(rng[2], torch.get_rng_state())
    with pytest.raises(ValueError, match="policy drift"):
        load_policy_checkpoint(
            tmp_path / "one",
            fresh,
            binding,
            supervision_policy="STRICT_WHOLE_BODY_V1",
            feature_batch_size=4,
        )
    with pytest.raises(ValueError, match="source/data"):
        load_policy_checkpoint(
            tmp_path / "one", fresh, {}, supervision_policy=DISTAR_SKIP, feature_batch_size=4
        )
    with (tmp_path / "one/state.pt").open("ab") as f:
        f.write(b"broken")
    with pytest.raises(ValueError, match="corruption"):
        load_policy_checkpoint(
            tmp_path / "one", fresh, binding, supervision_policy=DISTAR_SKIP, feature_batch_size=4
        )


def test_development_census_excludes_burnin_and_refuses_train_or_test():
    rows, p, cat, _ = decision_rows()
    cfg = BCModelConfig.from_catalog(cat, p, hidden_size=32, maximum_delay=8)
    model = RecurrentBCPolicy(cfg)

    class Dataset:
        allowed_split = "development"
        windows = [0, 1]
        rows = 2

        def window(self, i):
            return (rows[0],) if i == 0 else (replace(rows[0], burn_in=True, loss=False), rows[1])

    ds = Dataset()
    settings = dict(
        schema="poorstar_c3_development_evaluator_v1",
        unroll_length=1,
        burn_in_length=1,
        position_tolerance=1.0,
        token_watchdog=1,
    )
    report = evaluate(model, ds, p, settings, supervision_policy=DISTAR_SKIP, feature_batch_size=4)
    assert report["active_unique_rows"] == 2 and report["teacher"]["denominators"]["act"] == 2
    assert report["free_body"]["counts"]["watchdog_failures_all_phases"] == 2
    assert report["free_body"]["evaluated_scheduler_rows"] == 1
    assert report["free_body"]["scheduler_row_coverage_fraction"] == 0.5
    assert report["optimizer_steps"] == 0 and not report["test_read"] and not report["rl_ready"]
    for forbidden in ("train", "test"):
        ds.allowed_split = forbidden
        with pytest.raises(ValueError, match="development split"):
            evaluate(model, ds, p, settings, supervision_policy=DISTAR_SKIP, feature_batch_size=4)
    ds.allowed_split = "development"
    ds.rows = 3
    with pytest.raises(ValueError, match="denominator"):
        evaluate(model, ds, p, settings, supervision_policy=DISTAR_SKIP, feature_batch_size=4)
