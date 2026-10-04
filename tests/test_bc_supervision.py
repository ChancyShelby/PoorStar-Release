"""Tests: test bc supervision."""

from dataclasses import replace

import pytest
import torch

from poorstar.actions.coarse import validate_coarse_bundle
from poorstar.actions.contract import TargetDomainRejection
from poorstar.data.bc.supervision import DISTAR_SKIP, STRICT, prepare_teacher, supervision_census
from poorstar.models.policy import BCModelConfig, RecurrentBCPolicy
from poorstar.training.foundation import BCTrainer, bc_objective
from tests.test_coarse_bc import decision_rows, setup, target


@pytest.fixture(autouse=True)
def threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def test_missing_sources_are_removed_before_selection_cap_without_mutating_original():
    f, p, *_ = setup(maximum_sources=1)
    t = target(f, tags=(999, 20))
    with pytest.raises(TargetDomainRejection):
        prepare_teacher(f, (t,), p, policy=STRICT)
    result = prepare_teacher(f, (t,), p, policy=DISTAR_SKIP)
    n = result.targets[0]
    assert n.raw.source_slots == (1,) and not n.raw.issues
    assert n.original == t and t.source_slots == (None, 1)
    assert n.discarded_source_count == 1 and "UNOBSERVABLE_SOURCES_DROPPED" in n.reasons
    machine, trace, _ = validate_coarse_bundle(f, result.targets, p)
    assert machine.commands[0][1] == (1,) and trace == result.trace


def test_skip_one_command_keeps_other_commands_in_order_and_closes_stop():
    f, p, *_ = setup()
    commands = (target(f, tags=(10,)), target(f, target=999), target(f, tags=(20,), queue=True))
    result = prepare_teacher(f, commands, p, policy=DISTAR_SKIP)
    assert result.skipped == ((1, "UNREPRESENTABLE_ARGUMENT"),)
    assert tuple(n.original for n in result.targets) == (commands[0], commands[2])
    machine, trace, _ = validate_coarse_bundle(f, result.targets, p)
    assert machine.head == "done" and len(machine.commands) == 2
    assert [v for h, o, v in trace if h == "bundle"] == ["BUNDLE_NEXT", "BUNDLE_STOP"]


def test_filter_does_not_grant_unavailable_replacement_or_missing_source():
    f, p, *_ = setup()
    f = replace(f, advertised={s: skills - {3671} for s, skills in f.advertised.items()})
    result = prepare_teacher(
        f, (target(f, ability=3672), target(f, tags=(999,))), p, policy=DISTAR_SKIP
    )
    assert not result.targets and len(result.skipped) == 2
    with pytest.raises(TargetDomainRejection):
        prepare_teacher(
            f,
            (target(f),),
            p,
            policy=DISTAR_SKIP,
            body_rejection="NATIVE_COMMAND_WITHOUT_UNIT_ECHO",
        )


def test_internal_errors_are_not_reclassified_as_skipped_data():
    f, p, *_ = setup()
    with pytest.raises(ValueError, match="current complete"):
        prepare_teacher(
            f,
            (replace(target(f), frame_key=replace(f.key, loop=f.key.loop + 1)),),
            p,
            policy=DISTAR_SKIP,
        )
    with pytest.raises(ValueError, match="unknown supervision"):
        prepare_teacher(f, (target(f),), p, policy="bad")


def test_partial_and_empty_bodies_keep_schedule_and_microbatch_denominators():
    rows, p, cat, _ = decision_rows()
    row = rows[0]
    mixed = replace(row, targets=(target(row.body), target(row.body, tags=(999,))))
    empty = replace(row, targets=(target(row.body, tags=(999,)),))
    cfg = BCModelConfig.from_catalog(cat, p, hidden_size=32, maximum_delay=8)
    torch.manual_seed(29)
    one = RecurrentBCPolicy(cfg)
    two = RecurrentBCPolicy(cfg)
    two.load_state_dict(one.state_dict())
    sequences = [(mixed, rows[1]), (empty, rows[1])]
    a = BCTrainer(one, p, learning_rate=0.001, supervision_policy=DISTAR_SKIP)
    b = BCTrainer(two, p, learning_rate=0.001, supervision_policy=DISTAR_SKIP)
    expected = a.step(sequences)
    actual = b.step_microbatched(sequences)
    assert expected.denominators == actual.denominators
    assert expected.denominators["act"] == 4 and expected.denominators["ability"] == 1
    assert (
        actual.counts["skipped_commands"] == 2
        and actual.counts["partially_supervised_bundles"] == 1
    )
    assert actual.counts["rejected_bundles"] == 1
    assert all(
        torch.allclose(v, dict(two.named_parameters())[n], atol=2e-7, rtol=2e-6)
        for n, v in one.named_parameters()
    )
    burnt = bc_objective(
        two,
        [(replace(empty, burn_in=True, loss=False), rows[1])],
        p,
        supervision_policy=DISTAR_SKIP,
    )
    assert burnt.counts["burn_in_rows"] == 1 and burnt.denominators["act"] == 1


def test_unique_filtered_accounting_keeps_original_denominators():
    rows, p, cat, _ = decision_rows()
    row = replace(rows[0], targets=(target(rows[0].body), target(rows[0].body, tags=(999,))))
    report = supervision_census((row, rows[1]), p)
    assert report["counts"]["requested_commands"] == 2
    assert report["counts"]["retained_commands"] == report["counts"]["skipped_commands"] == 1
    assert report["head_denominators"]["act"] == 2 and report["head_denominators"]["ability"] == 1
    assert not report["causal_rejection_adjudication_required"]
    with pytest.raises(ValueError, match="duplicate"):
        supervision_census((row, row), p)


def test_streaming_census_rejects_reordered_or_discontiguous_episodes():
    rows, p, *_ = decision_rows()
    with pytest.raises(ValueError, match="nonmonotonic"):
        supervision_census(reversed(rows), p)
    other = replace(
        rows[1],
        scheduler=replace(
            rows[1].scheduler, key=replace(rows[1].scheduler.key, episode="another-episode")
        ),
    )
    with pytest.raises(ValueError, match="discontiguous"):
        supervision_census((rows[0], other, rows[1]), p)
    report = supervision_census((rows[0], rows[1], other), p)
    assert report["unique_episodes"] == 2 and report["counts"]["rows"] == 3


def test_large_streaming_census_does_not_retain_per_row_keys():
    import tracemalloc

    rows, p, *_ = decision_rows()
    wait = rows[1]

    def stream():
        for i in range(100000):
            yield replace(
                wait, scheduler=replace(wait.scheduler, key=replace(wait.scheduler.key, loop=i))
            )

    tracemalloc.start()
    try:
        report = supervision_census(stream(), p)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert report["counts"]["rows"] == report["head_denominators"]["act"] == 100000
    assert report["unique_episodes"] == 1 and peak < 8 * 1024**2


def test_free_body_evaluation_uses_declared_filtered_reference_and_keeps_raw_targets():
    from poorstar.evaluation.bc import evaluate_body, evaluate_sequence_bodies

    rows, p, cat, _ = decision_rows()
    row = rows[0]
    targets = (target(row.body), target(row.body, tags=(999,)))
    mixed = replace(row, targets=targets)
    model = RecurrentBCPolicy(BCModelConfig.from_catalog(cat, p, hidden_size=32, maximum_delay=8))
    model.eval()
    # Force a bounded sampled command; reference projection is the property under test.
    with torch.no_grad():
        model.fixed_heads["bundle"].weight.zero_()
        model.fixed_heads["bundle"].bias[:] = torch.tensor([-100.0, 100.0])
    report = evaluate_body(
        model, row.body, targets, p, supervision_policy=DISTAR_SKIP, token_watchdog=1000
    )
    assert report["counts"]["expected_commands"] == 1 and report["skipped_reference_commands"] == 1
    assert report["supervision_policy"] == DISTAR_SKIP and mixed.targets == targets
    seq = evaluate_sequence_bodies(
        model, [(mixed, rows[1])], p, supervision_policy=DISTAR_SKIP, token_watchdog=1000
    )
    assert (
        seq["counts"]["expected_commands"] == 1 and seq["counts"]["skipped_reference_commands"] == 1
    )
    assert not seq["online_scheduler_evaluated"]
