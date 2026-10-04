"""Tests: test coarse bc."""

import hashlib
import json
from dataclasses import replace

import numpy as np
import pytest
import torch
from s2clientprotocol import sc2api_pb2 as sc

from poorstar.actions.coarse import (
    CoarseActionProfile,
    coarse_sample,
    coarse_teacher_force,
    normalize_target,
    validate_coarse_bundle,
)
from poorstar.actions.contract import (
    SELECT_END,
    CompleteRequestTarget,
    RawCommandTarget,
    TargetDomainRejection,
    Token,
)
from poorstar.actions.frame import ActionFrame, BoundQuery, SchedulerFrame
from poorstar.data.bc.records import (
    BCDecision,
    BCShardReader,
    BCShardWriter,
    freeze_split,
    sequence_windows,
    validate_split,
)
from poorstar.models.policy import (
    BCModelConfig,
    BCPolicySession,
    RecurrentBCPolicy,
    WorldPointDistribution,
)
from poorstar.training.foundation import BCAggregate, bc_objective
from tests.test_shared_action_contract import command, fixture


@pytest.fixture(scope="module", autouse=True)
def threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def setup(*, loop=0, tags=(10, 20), maximum_sources=63, episode="episode"):
    f, raw, info, cat, q, r, context = fixture(loop=loop, tags=tags, episode=episode)
    for i, u in enumerate(raw.observation.raw_data.units):
        u.pos.x += i
        u.health = 30 + i * 5
        u.health_max = 100
    for ability, name in (
        (3664, "TransportUnloadAll"),
        (3670, "TransportUnloadUnit"),
        (3671, "CancelLast"),
        (3672, "CancelSlot"),
    ):
        cat.abilities.add(ability_id=ability, target=1, link_name=name)
        for answer in r.abilities:
            answer.abilities.add(ability_id=ability)
    frame = ActionFrame.from_protocol(
        raw, info, cat, BoundQuery.bind(raw.observation, q, r), context=context, episode=episode
    )
    return frame, CoarseActionProfile.from_catalog(cat, maximum_sources=maximum_sources), cat, info


def target(frame, **kwargs):
    return RawCommandTarget.from_protocol(command(frame, **kwargs), frame)


@pytest.mark.parametrize("ability,replacement", [(3670, 3664), (3672, 3671)])
def test_native_data_family_conversion_keeps_original_evidence(ability, replacement):
    f, p, cat, info = setup()
    original = target(f, ability=ability, queue=True)
    native = CompleteRequestTarget.retain_native(
        original, {"m_data": {"Data": 126}, "m_cmdFlags": 2}
    )
    n = normalize_target(native, f, p)
    assert n.raw.ability == replacement and n.raw.target_mode == "none" and n.raw.queue is True
    assert n.original == native and n.original.raw.original_wire == original.original_wire
    m, trace, normalized = validate_coarse_bundle(f, (native,), p)
    assert m.protocol_actions()[0].action_raw.unit_command.ability_id == replacement


def test_unknown_data_and_unknown_engine_ability_are_explicit_rejections():
    f, p, *_ = setup()
    t = CompleteRequestTarget.retain_native(target(f), {"m_data": {"Data": 20}})
    with pytest.raises(TargetDomainRejection, match="no authorized"):
        normalize_target(t, f, p)
    with pytest.raises(TargetDomainRejection, match="absent"):
        normalize_target(target(f, ability=9999), f, p)


def test_conversion_does_not_grant_replacement_availability():
    f, p, *_ = setup()
    advertised = {s: skills - {3671} for s, skills in f.advertised.items()}
    f = replace(f, advertised=advertised)
    with pytest.raises(TargetDomainRejection):
        coarse_teacher_force(f, (target(f, ability=3672),), p)


def test_cap_preserves_end_and_teacher_sample_domains():
    f, p, *_ = setup(tags=tuple(range(1, 70)))
    m, trace, n = validate_coarse_bundle(f, (target(f),), p)
    assert len(m.commands[0][1]) == 63 and n[0].discarded_source_count == 6
    rows = [r for r in trace if r[0] == "source"]
    assert len(rows) == 64 and rows[-1][1:] == ((SELECT_END,), SELECT_END)
    assert len(f.slot_tags) == 69  # entity/target domain remains uncapped


def test_unload_conversion_can_remove_inapplicable_passenger_target_only():
    f, p, *_ = setup()
    original = target(f, ability=3670, target=999)
    n = normalize_target(original, f, p)
    assert not n.raw.issues and original.issues == ("UNOBSERVABLE_ENTITY_TARGET",)
    broken = target(f, ability=3670, target=999, tags=(999,))
    with pytest.raises(TargetDomainRejection):
        coarse_teacher_force(f, (broken,), p)


def test_profile_requires_catalog_families_and_exact_catalog_parity():
    f, p, cat, info = setup()
    broken = sc.ResponseData()
    broken.CopyFrom(cat)
    del broken.abilities[-1]
    with pytest.raises(ValueError):
        CoarseActionProfile.from_catalog(broken)
    with pytest.raises(ValueError):
        p.check_frame(replace(f, targets={11: 1}))


def test_sampling_watchdog_is_failure_not_inserted_stop():
    f, p, *_ = setup()
    with pytest.raises(RuntimeError, match="watchdog"):
        coarse_sample(f, lambda h, o, c: o[0], p, token_watchdog=2)


def model_fixture():
    f, p, cat, info = setup()
    cfg = BCModelConfig.from_catalog(cat, p, hidden_size=32, maximum_delay=8)
    torch.manual_seed(7)
    model = RecurrentBCPolicy(cfg)
    return f, p, model, cat, info


def test_all_body_heads_have_finite_nonzero_gradients_without_optimizer():
    f, p, model, *_ = model_fixture()
    encoded = model.encode(f.policy_observation())
    targets = (
        target(f),
        target(f, ability=31, target=10),
        target(f, ability=21, point=(14.125, 30.5)),
    )
    losses, correct, hidden, m, n = model.teacher_body(encoded, f, targets, p)
    loss = sum(torch.stack(v).mean() for v in losses.values())
    loss.backward()
    assert set(losses) == {
        "ability",
        "source",
        "target_mode",
        "target_entity",
        "target_position",
        "queue",
        "bundle",
    }
    for group in (
        "ability_head",
        "source_query",
        "target_query",
        "position_head",
        "entity_projection",
        "memory",
        "spatial_encoder",
    ):
        grads = [
            v.grad
            for name, v in model.named_parameters()
            if name.startswith(group) and v.grad is not None
        ]
        assert (
            grads
            and all(torch.isfinite(v).all() for v in grads)
            and sum(float(v.abs().sum()) for v in grads) > 0
        ), group


def test_teacher_and_online_use_identical_conditional_distributions():
    f, p, model, *_ = model_fixture()
    model.eval()
    obs = f.policy_observation()
    session = BCPolicySession(model, p)
    session.hidden = model.zero_state()
    encoded = model.encode(obs, session.hidden)
    targets = (target(f, ability=21, point=(3.125, 17.5)), target(f))
    machine, trace, _ = coarse_teacher_force(f, targets, p)
    prefix = encoded.hidden
    for head, options, value in trace:
        # Session body setup is the same observation encoding as teacher forcing.
        if session.body is None:
            session.body = model.encode(obs, session.hidden)
            session.prefix = session.body.hidden
        a = model.distribution(encoded, prefix, head, options)
        b = model.distribution(session.body, session.prefix, head, options)
        if head == "target_position":
            assert torch.equal(a.log_prob(value), b.log_prob(value))
        else:
            assert torch.equal(a.logits, b.logits)
        prefix = model.consume(encoded, prefix, Token(head, value))
        session.prefix = model.consume(session.body, session.prefix, Token(head, value))


def test_actor_metadata_or_future_targets_cannot_enter_encoder():
    f, p, model, *_ = model_fixture()
    obs = f.policy_observation()
    initial = model.encode(obs).hidden.detach()
    for key in ("raw_tags", "outcome", "native_data", "frame_hash"):
        with pytest.raises(ValueError):
            model.encode(dict(obs, **{key: 1}))
    # Presence false means arbitrary placeholder values have exactly no effect.
    obs["scalars"]["minerals"] = 999999
    obs["scalar_present"]["minerals"] = False
    a = model.encode(obs).hidden.detach()
    obs["scalars"]["minerals"] = -888
    assert torch.equal(a, model.encode(obs).hidden.detach())


def test_position_density_keeps_fractional_targets_and_sampling_bounds():
    params = torch.tensor([1.0, -2.0, -0.5, 0.5], requires_grad=True)
    dist = WorldPointDistribution(params, (200, 160))
    loss = -dist.log_prob((24.125, 87.5))
    loss.backward()
    assert torch.isfinite(params.grad).all()
    for _ in range(100):
        x, y = dist.choose()
        assert 0 <= x < 200 and 0 <= y < 160
    with pytest.raises(ValueError):
        dist.log_prob((200, 0))


def decision_rows():
    a, p, cat, info = setup(loop=0)
    b, *_ = setup(loop=2)
    c, *_ = setup(loop=3)
    digest = "a" * 64
    return (
        (BCDecision(digest, a, "ACT", 2, b, (target(b),)), BCDecision(digest, c, "WAIT", 1)),
        p,
        cat,
        info,
    )


def test_sparse_body_rejection_preserves_scheduler_supervision():
    rows, p, cat, info = decision_rows()
    model = RecurrentBCPolicy(BCModelConfig.from_catalog(cat, p, hidden_size=32, maximum_delay=8))
    bad = replace(rows[0], targets=(target(rows[0].body, tags=(999,)),))
    result = bc_objective(model, [(bad, rows[1])], p)
    assert result.denominators == {"act": 2, "delay": 2}
    assert (
        result.counts["rejected_bundles"] == 1
        and result.rejections["UNREPRESENTABLE_ARGUMENT"] == 1
    )
    aggregate = BCAggregate()
    aggregate.add(result)
    with pytest.raises(ValueError, match="zero aggregate"):
        aggregate.report(required_heads=("ability",))


def test_burn_in_padding_and_episode_reset_do_not_contribute_loss():
    rows, p, cat, info = decision_rows()
    model = RecurrentBCPolicy(BCModelConfig.from_catalog(cat, p, hidden_size=32, maximum_delay=8))
    burn = replace(rows[0], burn_in=True, loss=False)
    padding = replace(rows[1], valid=False, loss=False)
    result = bc_objective(model, [(burn, rows[1], padding)], p)
    assert result.denominators == {"act": 1, "delay": 1}
    assert result.counts["burn_in_rows"] == result.counts["padding_rows"] == 1
    session = BCPolicySession(model, p)
    session.hidden = model.zero_state() + 1
    session.reset("new")
    assert session.hidden is None and session.episode == "new"


def test_model_errors_are_not_swallowed_as_expert_rejections(monkeypatch):
    rows, p, cat, info = decision_rows()
    model = RecurrentBCPolicy(BCModelConfig.from_catalog(cat, p, hidden_size=32, maximum_delay=8))

    def broken(*args):
        raise TargetDomainRejection("INTERNAL_BUG", "neural implementation failure")

    monkeypatch.setattr(model, "teacher_body", broken)
    with pytest.raises(TargetDomainRejection, match="implementation"):
        bc_objective(model, [rows], p)


def test_split_groups_duplicates_and_two_player_views_together():
    rows = [
        dict(replay_sha256="a" * 64, match_group="one"),
        dict(replay_sha256="a" * 64, match_group="two"),
        dict(replay_sha256="b" * 64, match_group="two"),
        dict(replay_sha256="c" * 64, match_group="three"),
    ]
    split = freeze_split(rows, seed=20261003)
    assert split["assignments"]["a" * 64] == split["assignments"]["b" * 64]
    assert split["groups"]["a" * 64] == split["groups"]["b" * 64]
    assert split == freeze_split(reversed(rows), seed=20261003)
    validate_split(split)
    broken = json.loads(json.dumps(split))
    broken["assignments"]["a" * 64] = "invalid"
    with pytest.raises(ValueError):
        validate_split(broken)


def test_binary_shard_roundtrip_exact_actor_arrays_masks_and_targets(tmp_path):
    rows, p, cat, info = decision_rows()
    split = freeze_split([dict(replay_sha256="a" * 64, match_group="match")], seed=1)
    chosen = split["assignments"]["a" * 64]
    writer = BCShardWriter(
        tmp_path,
        split_manifest=split,
        split=chosen,
        catalog=cat,
        game_info=info,
        source_manifest_sha256="b" * 64,
        extractor_sha256="c" * 64,
        maximum_delay=8,
    )
    for row in rows:
        writer.append(row)
    artifact = writer.close()
    reader = BCShardReader(
        artifact["path"], sha256=artifact["sha256"], split_manifest=split, allowed_split=chosen
    )
    copied = list(reader)
    reader.close()
    for a, b in zip(rows, copied):
        assert a.scheduler.key == b.scheduler.key and a.targets == b.targets
        pa = a.scheduler.policy_observation()
        pb = b.scheduler.policy_observation()
        aa = pa.pop("spatial_arrays")
        bb = pb.pop("spatial_arrays")
        assert pa == pb and aa.keys() == bb.keys()
        assert all(np.array_equal(aa[n], bb[n]) for n in aa)
    with pytest.raises(ValueError):
        BCShardReader(artifact["path"], sha256="0" * 64, split_manifest=split, allowed_split=chosen)


def test_sequence_rejects_clock_gaps_and_wrong_split(tmp_path):
    rows, p, cat, info = decision_rows()
    model = RecurrentBCPolicy(BCModelConfig.from_catalog(cat, p, hidden_size=32, maximum_delay=8))
    with pytest.raises(ValueError, match="clock gap"):
        bc_objective(model, [(rows[0], rows[0])], p)
    split = freeze_split([dict(replay_sha256="b" * 64, match_group="match")], seed=1)
    writer = BCShardWriter(
        tmp_path,
        split_manifest=split,
        split="train",
        catalog=cat,
        game_info=info,
        source_manifest_sha256="b" * 64,
        extractor_sha256="c" * 64,
    )
    with pytest.raises(ValueError, match="different split"):
        writer.append(rows[0])
    writer.abort()


def test_parallel_and_conditional_models_have_matched_parameter_capacity():
    f, p, model, cat, info = model_fixture()
    parallel = RecurrentBCPolicy(replace(model.config, autoregressive=False))
    assert sum(v.numel() for v in model.parameters()) == sum(
        v.numel() for v in parallel.parameters()
    )
    assert model.config.sha256 != parallel.config.sha256


def test_unqueried_scheduler_cannot_be_used_as_a_body():
    f, p, cat, info = setup()
    raw = sc.ResponseObservation()
    raw.observation.ParseFromString(f._observation_wire)
    scheduler = SchedulerFrame.from_protocol(
        raw, info, cat, context=f._context, episode=f.key.episode
    )
    assert scheduler.policy_observation().keys() == f.policy_observation().keys()
    with pytest.raises(ValueError, match="not a query-qualified"):
        coarse_teacher_force(scheduler, (target(scheduler),), p)


def test_sequence_windows_preserve_tail_and_causal_burn_in():
    rows, p, cat, info = decision_rows()
    windows = list(sequence_windows(rows, unroll_length=1, burn_in_length=2))
    assert len(windows) == 2 and len(windows[0]) == 1 and len(windows[1]) == 2
    assert windows[1][0].burn_in and not windows[1][0].loss
    assert windows[1][1] == rows[1]


def test_complete_audit_builder_keeps_missing_echo_as_act(tmp_path):
    import gzip

    from s2clientprotocol import query_pb2

    from poorstar.common.integrity import file_hash, payload_hash
    from poorstar.data.bc.builder import decisions_from_audit
    from poorstar.data.trajectory import encode_bytes
    from poorstar.engine.protocol_wire import ProtocolWireWriter

    trajectory = tmp_path / "trajectory"
    query = tmp_path / "query"
    geometry = tmp_path / "geometry"
    for root in (trajectory, query, geometry):
        root.mkdir()
    f, p, cat, info = setup()
    (geometry / "game_info.protocol").write_bytes(info.SerializeToString())
    (query / "ability_catalog.protocol").write_bytes(cat.SerializeToString())
    writer = ProtocolWireWriter(trajectory / "responses.protocol.gz")
    query_rows = []
    echo_rows = []
    for loop in range(4):
        f, *_ = setup(loop=loop)
        raw = sc.ResponseObservation()
        raw.observation.ParseFromString(f._observation_wire)
        if loop in (1, 3):
            raw.actions.add().CopyFrom(command(f))
            echo_rows.append(dict(kind="unit_command", response_loop=loop))
        if loop == 3:
            raw.player_result.add(player_id=1, result=sc.Victory)
        wire = raw.SerializeToString()
        writer.append(wire)
        # Only ACT body frames have queries. Scheduler-only frames need none.
        if loop in (0, 2):
            req, rep = query_pb2.RequestQuery(), query_pb2.ResponseQuery()
            req.ParseFromString(f._query_request_wire)
            rep.ParseFromString(f._query_response_wire)
            req.abilities.add(unit_tag=999)
            rep.abilities.add(unit_tag=0)
            query_rows.append(
                dict(
                    loop=loop,
                    observation_sha256=hashlib.sha256(wire).hexdigest(),
                    absent_control=999,
                    gone_control=None,
                    queries=[
                        dict(
                            mode="default",
                            request=req.SerializeToString(),
                            response=rep.SerializeToString(),
                        )
                    ],
                )
            )
        elif loop == 1:
            # The original no-echo command has a body query, but still no supervised RAW body.
            req, rep = query_pb2.RequestQuery(), query_pb2.ResponseQuery()
            req.ParseFromString(f._query_request_wire)
            rep.ParseFromString(f._query_response_wire)
            req.abilities.add(unit_tag=999)
            rep.abilities.add(unit_tag=0)
            query_rows.append(
                dict(
                    loop=loop,
                    observation_sha256=hashlib.sha256(wire).hexdigest(),
                    absent_control=999,
                    gone_control=None,
                    queries=[
                        dict(
                            mode="default",
                            request=req.SerializeToString(),
                            response=rep.SerializeToString(),
                        )
                    ],
                )
            )
    writer.close(complete=True)
    (query / "responses.protocol.gz").write_bytes(
        (trajectory / "responses.protocol.gz").read_bytes()
    )
    (query / "queries.jsonl.gz").write_bytes(
        gzip.compress(
            ("\n".join(json.dumps(x, default=encode_bytes) for x in query_rows) + "\n").encode()
        )
    )
    native = [
        dict(
            _event="NNet.Game.SCmdEvent",
            _gameloop=i,
            _userid={"m_userId": 0},
            m_data={"None": None},
        )
        for i in range(3)
    ]
    (trajectory / "original_events.jsonl.gz").write_bytes(
        gzip.compress(("\n".join(json.dumps(e) for e in native) + "\n").encode())
    )
    (trajectory / "engine_actions.jsonl.gz").write_bytes(
        gzip.compress(("\n".join(json.dumps(e) for e in echo_rows) + "\n").encode())
    )
    base = dict(
        replay_sha256="a" * 64, base_build=55958, controlled_player_id=1, controlled_user_id=0
    )

    def publish(root, extras):
        report = dict(base, **extras, artifacts={r.name: file_hash(r) for r in root.iterdir()})
        digest = payload_hash(report)
        report["content_digest"] = digest
        (root / "report.json").write_text(json.dumps(report))
        return digest

    qdigest = publish(query, {})
    publish(trajectory, dict(same_run_query_binding_verified=True, input_query_digest=qdigest))
    publish(geometry, {})
    rows = list(decisions_from_audit(trajectory, query, geometry, cat, maximum_delay=8))
    assert len(rows) == 3 and all(r.decision == "ACT" for r in rows)
    assert (
        rows[1].body_rejection == "NATIVE_COMMAND_WITHOUT_UNIT_ECHO" and rows[1].native_event_wires
    )
    assert all(isinstance(r.scheduler, SchedulerFrame) for r in rows)


def test_last_selection_slot_must_make_end_legal():
    f, p, *_ = setup(maximum_sources=1)
    f = replace(f, advertised={0: frozenset(), 1: f.advertised[1]})
    from poorstar.actions.coarse import CoarsePrefixMachine

    machine = CoarsePrefixMachine(f, p)
    machine.accept(Token("ability", 11))
    assert machine.options() == (1,)
    machine.accept(Token("source", 1))
    assert machine.options() == (SELECT_END,)


def test_partial_rpc_results_are_not_complete_acceptance():
    from poorstar.actions.timing import DispatchReceipt

    f, *_ = setup()
    receipt = DispatchReceipt(f.key, 2, b"", b"", (1, 3))
    assert receipt.accepted_count == 1 and not receipt.all_accepted


def test_coarse_fine_position_density_and_sampling_keep_original_precision():
    from poorstar.models.policy import CoarseFinePointDistribution

    params = torch.zeros(5 * 16 * 16, requires_grad=True)
    dist = CoarseFinePointDistribution(params, (200, 160), (16, 16))
    (-dist.log_prob((24.125, 87.5))).backward()
    assert torch.isfinite(params.grad).all() and params.grad.abs().sum() > 0
    for _ in range(30):
        point = dist.choose()
        assert 0 <= point[0] < 200 and 0 <= point[1] < 160


def test_training_metric_switch_keeps_identical_objective_and_denominators():
    rows, p, cat, info = decision_rows()
    torch.manual_seed(17)
    model = RecurrentBCPolicy(BCModelConfig.from_catalog(cat, p, hidden_size=32, maximum_delay=8))
    a = bc_objective(model, [rows], p)
    b = bc_objective(model, [rows], p, collect_metrics=False)
    assert torch.equal(a.loss, b.loss) and a.denominators == b.denominators
    assert not b.metrics and not b.calibration_bins
    aggregate = BCAggregate()
    aggregate.add(a)
    report = aggregate.report()
    assert report["teacher_conditioned_ece"] and all(
        0 <= x <= 1 for x in report["teacher_conditioned_ece"].values()
    )


def test_direct_phase_extraction_keeps_complete_commands_and_missing_echo():
    from s2clientprotocol import query_pb2

    from poorstar.data.bc.builder import decisions_from_controller

    f, p, cat, info = setup()
    native = [
        dict(_event="NNet.Game.SCmdEvent", _gameloop=i, m_data={"None": None}) for i in range(3)
    ]

    class Controller:
        loop = 0

        def step(self, n):
            self.loop += n

        def observe(self):
            current, *_ = setup(loop=self.loop)
            raw = sc.ResponseObservation()
            raw.observation.ParseFromString(current._observation_wire)
            if self.loop in (1, 3):
                raw.actions.add().CopyFrom(command(current))
            if self.loop == 3:
                raw.player_result.add(player_id=1, result=sc.Victory)
            return raw

        def query(self, req):
            current, *_ = setup(loop=self.loop)
            reply = query_pb2.ResponseQuery()
            reply.ParseFromString(current._query_response_wire)
            return reply

    rows = list(
        decisions_from_controller(
            Controller(),
            info,
            cat,
            context=f._context,
            episode="direct",
            replay_sha256="a" * 64,
            native_events=native,
            terminal_loop=3,
            maximum_delay=8,
        )
    )
    assert len(rows) == 3 and rows[1].body_rejection == "NATIVE_COMMAND_WITHOUT_UNIT_ECHO"
    assert [r.body.key.loop for r in rows] == [0, 1, 2]


def test_empty_query_is_valid_evidence_for_a_zero_entity_frame(tmp_path):
    from s2clientprotocol import query_pb2

    populated, p, cat, info = setup()
    raw = sc.ResponseObservation()
    raw.observation.ParseFromString(populated._observation_wire)
    del raw.observation.raw_data.units[:]
    q = query_pb2.RequestQuery()
    response = query_pb2.ResponseQuery()
    f = ActionFrame.from_protocol(
        raw,
        info,
        cat,
        BoundQuery.bind(raw.observation, q, response),
        context=populated._context,
        episode="empty",
    )
    assert not f._query_request_wire and not f._query_response_wire
    split = freeze_split([dict(replay_sha256="a" * 64, match_group="empty")], seed=1)
    chosen = split["assignments"]["a" * 64]
    writer = BCShardWriter(
        tmp_path,
        split_manifest=split,
        split=chosen,
        catalog=cat,
        game_info=info,
        source_manifest_sha256="b" * 64,
        extractor_sha256="c" * 64,
    )
    writer.append(BCDecision("a" * 64, f, "WAIT", 1))
    artifact = writer.close()
    reader = BCShardReader(
        artifact["path"], sha256=artifact["sha256"], split_manifest=split, allowed_split=chosen
    )
    assert len(list(reader)[0].scheduler.slot_tags) == 0
    reader.close()
