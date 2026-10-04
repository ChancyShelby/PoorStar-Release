"""Tests: test bc throughput."""

import math
from dataclasses import replace

import numpy as np
import pytest
import torch

from poorstar.actions.coarse import coarse_teacher_force
from poorstar.models.policy import ENTITY_FIELDS, RecurrentBCPolicy
from tests.test_coarse_bc import model_fixture, setup, target


def test_advertised_only_coarse_domain_equals_full_alias_expansion():
    f, p, *_ = setup()
    expected = tuple(a for a in f.ability_options() if p.ability(a) == a)
    assert p.ability_options(f) == expected
    _, trace, _ = coarse_teacher_force(f, (target(f), target(f)), p)
    assert all(o == expected for h, o, v in trace if h == "ability")


def test_policy_fast_path_matches_audit_payload_and_isolates_mutations():
    f, p, *_ = setup()
    payload, arrays = f.actor_candidate_payload()
    obs = f.policy_observation()
    assert obs["entities"] == payload["entities"]
    assert obs["own_operations"] == payload["own_operations"]["parents"]
    for name, array in arrays.items():
        assert np.array_equal(array, obs["spatial_arrays"][name])
    obs["own_operations"][0]["orders"].append({"poison": 1})
    obs["entities"][0]["values"]["health"] = 99999
    fresh = f.policy_observation()
    assert fresh["own_operations"] == payload["own_operations"]["parents"]
    assert fresh["entities"] == payload["entities"]


def test_native_number_matrix_matches_scalar_formula_and_rejects_nonfinite():
    f, p, model, *_ = model_fixture()
    obs = f.policy_observation()
    rows = [(e["values"], e["present"]) for e in obs["entities"]]
    reference = []
    for values, present in rows:
        result = []
        for name in ENTITY_FIELDS:
            exists = bool(present.get(name, False))
            v = float(values.get(name, 0)) if exists else 0.0
            if name in ("world_x", "world_y"):
                v /= obs["map_size"][0 if name == "world_x" else 1]
            else:
                v = math.copysign(math.log1p(abs(v)), v)
            result.extend((v, float(exists)))
        reference.append(result)
    actual = model._number_matrix(rows, ENTITY_FIELDS, obs["map_size"])
    assert np.allclose(actual, reference, atol=2e-15, rtol=2e-15)
    # Float32 features, and hence parameter gradients, are exactly identical here.
    assert torch.equal(model._tensor(actual), model._tensor(reference))
    a = model.entity_projection(model._tensor(actual)).sum()
    b = model.entity_projection(model._tensor(reference)).sum()
    assert torch.equal(
        torch.autograd.grad(a, model.entity_projection.weight)[0],
        torch.autograd.grad(b, model.entity_projection.weight)[0],
    )
    values, present = rows[0]
    values = dict(values, health=float("nan"))
    present = dict(present, health=True)
    with pytest.raises(ValueError, match="nonfinite"):
        model._number_matrix([(values, present)], ENTITY_FIELDS, obs["map_size"])


@pytest.mark.parametrize("metadata,fallback", [(103, False), (400, True), (30, False)])
def test_terminal_hint_never_substitutes_header_for_actual_first_result(metadata, fallback):
    from s2clientprotocol import sc2api_pb2 as sc

    from poorstar.data.bc.builder import discover_replay_terminal

    class Controller:
        loop = 0

        def step(self, n):
            self.loop = min(500, self.loop + n)

        def observe(self):
            x = sc.ResponseObservation()
            x.observation.game_loop = self.loop
            x.observation.player_common.player_id = 1
            if self.loop >= 91:
                x.player_result.add(player_id=1, result=sc.Victory)
            return x

    c = Controller()
    raw, r = discover_replay_terminal(
        c,
        player_id=1,
        metadata_duration_loops=metadata,
        step_chunk=32,
        restart=lambda: setattr(c, "loop", 0),
        terminal_probe_margin=16,
    )
    assert r["terminal_loop"] == 91 and raw.observation.game_loop == 91
    assert r["terminal_probe_fallback"] == fallback
    if not fallback:
        assert r["refinement_one_loop_steps"] <= 16


def test_batched_features_match_single_frame_outputs_gradients_and_do_not_mix_frames():
    from poorstar.models.encoder import advance_features, encode_features_batch

    f, p, model, *_ = model_fixture()
    model.train()
    observations = [setup(tags=(10,))[0].policy_observation(), f.policy_observation()]
    observations[1]["scalars"]["minerals"] = 100
    expected = [model.encode(o).hidden for o in observations]
    features = encode_features_batch(model, observations)
    actual = [advance_features(model, x, None).hidden for x in features]
    for a, b in zip(expected, actual):
        assert torch.allclose(a, b, atol=2e-6, rtol=2e-5)
    params = list(model.parameters())
    g1 = torch.autograd.grad(sum(x.square().sum() for x in expected), params, allow_unused=True)
    g2 = torch.autograd.grad(sum(x.square().sum() for x in actual), params, allow_unused=True)
    for a, b in zip(g1, g2):
        assert (a is None) == (b is None)
        if a is not None:
            assert torch.allclose(a, b, atol=3e-6, rtol=3e-4)
    before = features[0][0].detach().clone()
    observations[1]["entities"][0]["values"]["health"] = 10000
    after = encode_features_batch(model, observations)[0][0]
    assert torch.equal(before, after)


def test_batched_objective_preserves_burnin_sparse_denominators_and_exact_resume(tmp_path):
    from poorstar.data.bc.supervision import DISTAR_SKIP
    from poorstar.data.bc.windows import (
        BCRunPlan,
        WindowOrder,
        freeze_binding,
        load_checkpoint,
        save_checkpoint,
    )
    from poorstar.models.policy import BCModelConfig
    from poorstar.training.foundation import BCTrainer, bc_objective
    from tests.test_coarse_bc import decision_rows

    rows, p, cat, _ = decision_rows()
    cfg = BCModelConfig.from_catalog(cat, p, hidden_size=32, maximum_delay=8)
    torch.manual_seed(31)
    one = RecurrentBCPolicy(cfg)
    two = RecurrentBCPolicy(cfg)
    two.load_state_dict(one.state_dict())
    sequences = [(replace(rows[0], burn_in=True, loss=False), rows[1]), rows]
    a = bc_objective(one, sequences, p, collect_metrics=False, supervision_policy=DISTAR_SKIP)
    b = bc_objective(
        two,
        sequences,
        p,
        collect_metrics=False,
        supervision_policy=DISTAR_SKIP,
        feature_batch_size=4,
    )
    assert (
        a.denominators == b.denominators and a.counts == b.counts and a.rejections == b.rejections
    )
    assert torch.allclose(a.loss, b.loss, atol=2e-6, rtol=2e-5)
    a.loss.backward()
    b.loss.backward()
    for n, x in one.named_parameters():
        y = dict(two.named_parameters())[n]
        assert (x.grad is None) == (y.grad is None)
        if x.grad is not None:
            assert torch.allclose(x.grad, y.grad, atol=3e-6, rtol=3e-4)
    trainer = BCTrainer(
        two, p, learning_rate=0.001, feature_batch_size=4, supervision_policy=DISTAR_SKIP
    )
    order = WindowOrder(4, 31)
    plan = BCRunPlan("ENGINEERING", 31, 10, feature_batch_size=4, supervision_policy=DISTAR_SKIP)
    trainer.step_microbatched(sequences)
    binding = freeze_binding(trainer, plan, source_sha256="a" * 64, dataset_sha256="b" * 64)
    save_checkpoint(tmp_path / "one", trainer, order, binding)
    expected = trainer.step_microbatched(sequences).loss
    weights = {n: x.detach().clone() for n, x in two.named_parameters()}
    fresh = BCTrainer(
        RecurrentBCPolicy(cfg),
        p,
        learning_rate=0.1,
        feature_batch_size=4,
        supervision_policy=DISTAR_SKIP,
    )
    load_checkpoint(tmp_path / "one", fresh, order, binding)
    assert torch.equal(expected, fresh.step_microbatched(sequences).loss)
    assert all(torch.equal(x, weights[n]) for n, x in fresh.model.named_parameters())
    wrong = BCTrainer(
        RecurrentBCPolicy(cfg),
        p,
        learning_rate=0.1,
        feature_batch_size=1,
        supervision_policy=DISTAR_SKIP,
    )
    with pytest.raises(ValueError, match="batching drift"):
        load_checkpoint(tmp_path / "one", wrong, order, binding)


def test_native_solid_checksum_truncation_and_trailing_frame_are_not_silently_accepted(tmp_path):
    import zstandard

    from poorstar.data.bc.storage import expand_solid

    wire = zstandard.ZstdCompressor(write_checksum=True).compress(b"unchanged actor wires" * 100)
    bad = tmp_path / "bad.zst"
    bad.write_bytes(wire[:-2])
    with pytest.raises(ValueError, match="truncated"):
        expand_solid(bad)
    bad.write_bytes(wire + b"garbage")
    with pytest.raises(ValueError, match="trailing"):
        expand_solid(bad)
    broken = bytearray(wire)
    broken[-1] ^= 1
    bad.write_bytes(broken)
    with pytest.raises(zstandard.ZstdError):
        expand_solid(bad)


def test_native_solid_real_size_window_exceeds_64kib_but_remains_within_byte_guard(tmp_path):
    import zstandard

    from poorstar.data.bc.storage import expand_solid

    raw = np.arange(2**19, dtype=np.uint32).tobytes()
    wire = zstandard.ZstdCompressor(level=3, write_checksum=True).compress(raw)
    assert zstandard.get_frame_parameters(wire).window_size > 64 * 1024
    p = tmp_path / "real_size.zst"
    p.write_bytes(wire)
    f = expand_solid(p)
    try:
        assert f.read() == raw
    finally:
        f.close()
