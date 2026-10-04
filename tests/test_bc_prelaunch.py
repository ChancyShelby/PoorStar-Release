"""Tests: test bc prelaunch."""

import json
import math
import random
from dataclasses import replace

import numpy as np
import pytest
import torch

from poorstar.actions.contract import SELECT_END
from poorstar.common.integrity import payload_hash
from poorstar.data.bc.records import BCShardWriter, freeze_split
from poorstar.data.bc.windows import (
    BCRunPlan,
    BCWindowDataset,
    PolicyObservationCache,
    WindowOrder,
    freeze_binding,
    load_checkpoint,
    save_checkpoint,
)
from poorstar.evaluation.action_diagnostics import domain_census
from poorstar.models.policy import BCModelConfig, RecurrentBCPolicy
from poorstar.training.foundation import BCTrainer, bc_objective
from tests.test_coarse_bc import decision_rows, model_fixture, setup, target


@pytest.fixture(autouse=True)
def threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


@pytest.mark.parametrize(
    "head,options", [("source", (1, 0, SELECT_END)), ("target_entity", (1, 0))]
)
def test_vectorized_pointer_preserves_distribution_and_gradients(head, options):
    f, p, model, *_ = model_fixture()
    model.eval()
    encoded = model.encode(f.policy_observation())
    prefix = encoded.hidden
    query = (model.source_query if head == "source" else model.target_query)(prefix)
    old = torch.stack(
        [
            model.end_head(prefix).squeeze(0)
            if o == SELECT_END
            else encoded.entities[o].dot(query) / math.sqrt(len(query))
            for o in options
        ]
    )
    new = model.distribution(encoded, prefix, head, options).logits
    reference = torch.log_softmax(old, 0)
    assert torch.allclose(new, reference, atol=1e-6, rtol=1e-6)
    params = (model.source_query if head == "source" else model.target_query).weight
    a = torch.autograd.grad(-reference[0], params, retain_graph=True)[0]
    b = torch.autograd.grad(-new[0], params)[0]
    assert torch.allclose(a, b, atol=1e-6, rtol=1e-5)


def test_rebuilt_optimizer_resume_is_exact_next_step(tmp_path):
    rows, p, cat, info = decision_rows()
    cfg = BCModelConfig.from_catalog(cat, p, hidden_size=32, maximum_delay=8)
    torch.manual_seed(19)
    random.seed(19)
    np.random.seed(19)
    model = RecurrentBCPolicy(cfg)
    trainer = BCTrainer(model, p, learning_rate=0.001)
    plan = BCRunPlan("ENGINEERING", 19, 10, hidden_size=32)
    order = WindowOrder(4, 19)
    order.take(2)
    trainer.step([rows])
    binding = freeze_binding(trainer, plan, source_sha256="a" * 64, dataset_sha256="b" * 64)
    checkpoint = save_checkpoint(tmp_path / "one", trainer, order, binding)
    expected_inputs = order.take(3)
    random_expected = (random.random(), np.random.rand(), torch.rand(1))
    loss = trainer.step([rows]).loss.detach().clone()
    parameters = {n: v.detach().clone() for n, v in model.named_parameters()}
    recovered = BCTrainer(RecurrentBCPolicy(cfg), p, learning_rate=0.5)
    load_checkpoint(tmp_path / "one", recovered, order, binding)
    assert order.take(3) == expected_inputs
    assert random.random() == random_expected[0] and np.random.rand() == random_expected[1]
    assert torch.equal(torch.rand(1), random_expected[2])
    assert torch.equal(recovered.step([rows]).loss.detach(), loss)
    assert all(torch.equal(v, parameters[n]) for n, v in recovered.model.named_parameters())
    assert recovered.steps == 2 and checkpoint["steps"] == 1
    with pytest.raises(FileExistsError):
        save_checkpoint(tmp_path / "one", trainer, order, binding)
    with pytest.raises(ValueError, match="drift"):
        load_checkpoint(tmp_path / "one", recovered, order, dict(binding, source_sha256="c" * 64))
    with (tmp_path / "one" / "state.pt").open("ab") as f:
        f.write(b"corruption")
    with pytest.raises(ValueError, match="corruption"):
        load_checkpoint(tmp_path / "one", recovered, order, binding)


def test_checkpoint_failed_write_is_never_published(tmp_path, monkeypatch):
    rows, p, cat, info = decision_rows()
    trainer = BCTrainer(
        RecurrentBCPolicy(BCModelConfig.from_catalog(cat, p, hidden_size=32)),
        p,
        learning_rate=0.001,
    )

    def broken(state, path):
        path.write_bytes(b"partial")
        raise OSError("disk full")

    monkeypatch.setattr(torch, "save", broken)
    with pytest.raises(OSError):
        save_checkpoint(tmp_path / "checkpoint", trainer, WindowOrder(1, 1), {})
    assert not (tmp_path / "checkpoint").exists()


def test_unique_domain_census_counts_all_commands_of_rejected_bundle():
    rows, p, cat, info = decision_rows()
    row = rows[0]
    broken = replace(row, targets=(target(row.body), target(row.body, tags=(999,))))
    report = domain_census((broken, rows[1]), p)
    assert report["counts"]["body_commands_without_supervision"] == 2
    assert report["lost_command_fraction"] == 1.0 and report["examples"][0]["command_index"] == 1
    assert report["examples"][0]["issues"] == ["UNOBSERVABLE_SOURCE_REFERENCE"]
    with pytest.raises(ValueError, match="duplicate"):
        domain_census((row, row), p)
    model = RecurrentBCPolicy(BCModelConfig.from_catalog(cat, p, hidden_size=32, maximum_delay=8))
    result = bc_objective(model, [(replace(broken, burn_in=True, loss=False), rows[1])], p)
    assert (
        result.counts.get("active_rejected_bundles", 0) == 0
        and result.counts["rejected_bundles"] == 1
    )


def test_observation_cache_preserves_gradients_and_is_bounded():
    rows, p, cat, info = decision_rows()
    model = RecurrentBCPolicy(BCModelConfig.from_catalog(cat, p, hidden_size=32, maximum_delay=8))
    model.eval()
    torch.manual_seed(5)
    a = bc_objective(model, [rows], p)
    a.loss.backward()
    gradients = {n: v.grad.clone() for n, v in model.named_parameters() if v.grad is not None}
    model.zero_grad(set_to_none=True)
    cache = PolicyObservationCache(2)
    b = bc_objective(model, [rows], p, observation_cache=cache)
    b.loss.backward()
    assert torch.equal(a.loss, b.loss) and len(cache.frames) <= 2
    assert all(
        torch.equal(v.grad, gradients[n]) for n, v in model.named_parameters() if v.grad is not None
    )


def build_dataset(tmp_path):
    rows, p, cat, info = decision_rows()
    split = freeze_split([dict(replay_sha256="a" * 64, match_group="one")], seed=1)
    chosen = split["assignments"]["a" * 64]
    artifacts = []
    for row in rows:
        writer = BCShardWriter(
            tmp_path / chosen,
            split_manifest=split,
            split=chosen,
            catalog=cat,
            game_info=info,
            source_manifest_sha256="b" * 64,
            extractor_sha256="c" * 64,
            maximum_delay=8,
        )
        writer.append(row)
        artifacts.append(writer.close())
    report = dict(shards=artifacts, split_sha256=split["sha256"], purpose="ENGINEERING")
    report["content_digest"] = payload_hash(report)
    (tmp_path / "build.json").write_text(json.dumps(report))
    (tmp_path / "split.json").write_text(json.dumps(split))
    return chosen, report


def test_lazy_window_joins_shard_boundaries_and_retains_burnin(tmp_path):
    chosen, report = build_dataset(tmp_path)
    ds = BCWindowDataset(
        tmp_path / "build.json",
        tmp_path / "split.json",
        unroll_length=1,
        burn_in_length=1,
        allowed_split=chosen,
        maximum_open_shards=1,
    )
    assert ds.rows == 2 and len(ds.windows) == 2
    seq = ds.window(1)
    assert len(seq) == 2 and seq[0].burn_in and not seq[0].loss and seq[1].loss
    assert seq[0].scheduler.key.loop == 0 and seq[1].scheduler.key.loop == 3 and len(ds.open) <= 1
    ds.close()
    report["shards"].append(report["shards"][0])
    report.pop("content_digest")
    report["content_digest"] = payload_hash(report)
    (tmp_path / "build.json").write_text(json.dumps(report))
    with pytest.raises(ValueError, match="duplicate"):
        BCWindowDataset(tmp_path / "build.json", tmp_path / "split.json", allowed_split=chosen)


def test_shuffle_resume_and_schedule_budget_bounds():
    a = WindowOrder(5, 11)
    b = WindowOrder(5, 11)
    assert sorted(a.take(5)) == list(range(5))
    b.cursor = a.cursor
    assert a.take(12) == b.take(12)
    plan = BCRunPlan("ENGINEERING", 11, 8, warmup_steps=2)
    assert plan.lr_at(0) == plan.learning_rate / 2 and plan.lr_at(1) == plan.learning_rate
    assert plan.lr_at(7) == pytest.approx(plan.learning_rate * plan.minimum_lr_ratio)
    with pytest.raises(ValueError, match="exhausted"):
        plan.lr_at(8)


def test_terminal_discovery_uses_engine_truth_when_header_duration_is_wrong():
    from s2clientprotocol import sc2api_pb2 as sc

    from poorstar.data.bc.builder import discover_replay_terminal

    class Controller:
        loop = 0

        def step(self, n):
            self.loop = min(91, self.loop + n)

        def observe(self):
            raw = sc.ResponseObservation()
            raw.observation.game_loop = self.loop
            raw.observation.player_common.player_id = 1
            if self.loop == 91:
                raw.player_result.add(player_id=1, result=sc.Victory)
            return raw

    raw, report = discover_replay_terminal(Controller(), player_id=1, metadata_duration_loops=120)
    assert (
        report["terminal_loop"] == 91
        and report["metadata_duration_loops"] == 120
        and raw.player_result
    )
    with pytest.raises(ValueError, match="player"):
        discover_replay_terminal(Controller(), player_id=2, metadata_duration_loops=120)


def test_terminal_refines_first_result_instead_of_last_chunk_observation():
    from s2clientprotocol import sc2api_pb2 as sc

    from poorstar.data.bc.builder import discover_replay_terminal

    class Controller:
        loop = 0

        def step(self, n):
            self.loop = min(120, self.loop + n)

        def observe(self):
            raw = sc.ResponseObservation()
            raw.observation.game_loop = self.loop
            raw.observation.player_common.player_id = 1
            if self.loop >= 91:
                raw.player_result.add(player_id=1, result=sc.Victory)
            return raw

    c = Controller()
    with pytest.raises(ValueError, match="requires restart"):
        discover_replay_terminal(c, player_id=1, metadata_duration_loops=120, step_chunk=32)
    raw, report = discover_replay_terminal(
        c,
        player_id=1,
        metadata_duration_loops=120,
        step_chunk=32,
        restart=lambda: setattr(c, "loop", 0),
    )
    assert report["terminal_loop"] == 91 and report["chunk_terminal_upper_bound"] == 96
    assert report["last_nonterminal_chunk_loop"] == 64 and report["refinement_one_loop_steps"] == 27
    assert report["terminal_loop_exact"] and int(raw.observation.game_loop) == 91


def test_native_terminal_tail_is_audited_without_moving_playable_command_clocks():
    from poorstar.actions.timing import schedule_targets
    from poorstar.data.bc.builder import partition_native_commands

    events = (
        dict(_event="NNet.Game.SCmdEvent", _gameloop=89, m_data={"TargetUnit": 12}),
        dict(_event="NNet.Game.SCommandManagerStateEvent", _gameloop=91, m_state=1, m_sequence=4),
        dict(_event="NNet.Game.SCmdEvent", _gameloop=92, m_data={"Data": 0}),
        dict(_event="NNet.Game.SGameUserLeaveEvent", _gameloop=120),
    )
    active, tail = partition_native_commands(events, terminal_loop=91)
    assert active == (events[0],) and tail == events[1:3]
    assert active[0] is events[0] and tail[0] is events[1]
    plans = list(
        schedule_targets(
            [e["_gameloop"] for e in active], initial_loop=0, terminal_loop=91, maximum_delay=32
        )
    )
    assert next(p.body_loop for p in plans if p.decision == "ACT") == 89
    assert plans[-1].scheduler_loop + plans[-1].delay == 91
    # The strict scheduler still refuses callers that fail to partition/audit.
    with pytest.raises(ValueError, match="outside nonterminal"):
        list(schedule_targets([89, 91, 92], initial_loop=0, terminal_loop=91, maximum_delay=32))


@pytest.mark.parametrize("loop", [-1, True, 1.5])
def test_terminal_partition_does_not_hide_invalid_native_clocks(loop):
    from poorstar.data.bc.builder import partition_native_commands

    with pytest.raises(ValueError, match="invalid native command clock"):
        partition_native_commands(
            [dict(_event="NNet.Game.SCmdEvent", _gameloop=loop)], terminal_loop=91
        )


def test_resumed_corpus_job_checks_terminal_tail_bytes_and_accounting(tmp_path):
    from poorstar.cli.data.build import completed_job, write_report
    from poorstar.common.integrity import file_hash

    root = tmp_path / "attempt_00000"
    (root / "train").mkdir(parents=True)
    shard = root / "train/one"
    shard.write_bytes(b"shard")
    terminal = root / "extraction_terminal.protocol"
    terminal.write_bytes(b"terminal")
    tail = root / "terminal_native_commands.jsonl.gz"
    tail.write_bytes(b"audited tail")
    job = dict(job_id="job", replay_sha256="a" * 64, player_id=2, split="train")
    report = dict(
        replay_sha256="a" * 64,
        controlled_player_id=2,
        split_sha256="b" * 64,
        source_manifest_sha256="c" * 64,
        engine_base_build=55958,
        full_game=True,
        lossless_solid_compression=True,
        solid_codec="zstd",
        optimizer_steps=0,
        terminal_source="INDEPENDENT_FULL_ENGINE_PASS",
        terminal_loop=91,
        terminal_observation_sha256=file_hash(terminal),
        counts=dict(all_decisions=1),
        shards=[dict(path=str(shard), sha256=file_hash(shard), bytes=5, decisions=1)],
        terminal_native_boundary=dict(
            rule="FIRST_ENGINE_RESULT_NONTERMINAL_BODY_INTERVAL",
            terminal_loop=91,
            original_command_events=3,
            playable_command_events=1,
            commands_at_terminal=1,
            commands_after_terminal=1,
            artifact=tail.name,
            sha256=file_hash(tail),
        ),
    )

    def publish():
        write_report(root / "report.json", report)
        write_report(
            root / "job_completion.json",
            dict(
                job_id="job",
                split_sha256="b" * 64,
                source_manifest_sha256="c" * 64,
                maximum_delay=32,
                shard_decisions=1024,
                solid_codec="zstd",
                report_digest=payload_hash(report),
            ),
        )

    def load():
        return completed_job(
            root,
            job,
            split_sha256="b" * 64,
            source_sha256="c" * 64,
            maximum_delay=32,
            shard_decisions=1024,
            solid_codec="zstd",
        )

    publish()
    assert load() == report
    tail.write_bytes(b"corrupt tail")
    with pytest.raises(ValueError, match="terminal command audit drift"):
        load()
    tail.write_bytes(b"audited tail")
    report["terminal_native_boundary"]["original_command_events"] = 4
    publish()
    with pytest.raises(ValueError, match="terminal command boundary drift"):
        load()


def test_microbatch_uses_full_logical_batch_sparse_head_denominators(monkeypatch):
    rows, p, cat, info = decision_rows()
    cfg = BCModelConfig.from_catalog(cat, p, hidden_size=32, maximum_delay=8)
    torch.manual_seed(4)
    one = RecurrentBCPolicy(cfg)
    two = RecurrentBCPolicy(cfg)
    two.load_state_dict(one.state_dict())
    # Disable dropout for a mathematical gradient comparison; different batch
    # execution orders are otherwise allowed to consume random draws differently.
    for model in (one, two):
        for layer in model.modules():
            if isinstance(layer, torch.nn.Dropout):
                layer.p = 0.0
    a = BCTrainer(one, p, learning_rate=0.001)
    b = BCTrainer(two, p, learning_rate=0.001)
    sequences = [rows, (rows[1],)]
    reference = a.step(sequences)
    actual = b.step_microbatched(sequences)
    assert reference.denominators == actual.denominators
    assert float(reference.loss) == pytest.approx(float(actual.loss), rel=1e-6)
    assert all(
        torch.allclose(v, dict(two.named_parameters())[n], atol=2e-7, rtol=2e-6)
        for n, v in one.named_parameters()
    )


def test_formal_launcher_refuses_engineering_data_before_optimizer():
    from poorstar.cli.training.train_foundation import formal_guard

    class Dataset:
        report = {"purpose": "ENGINEERING", "paper_corpus_gate_passed": False}

    with pytest.raises(ValueError, match="full registered corpus"):
        formal_guard(BCRunPlan("FORMAL", 1, 10), Dataset(), {}, None)


def test_rank_and_conditional_joint_do_not_hide_wrong_source():
    f, p, model, *_ = model_fixture()
    model.eval()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.ability_head.bias[model.ability_index[31]] = 5.0
    encoded = model.encode(f.policy_observation())
    terms, metrics, hidden, machine, normalized = model.teacher_body(
        encoded, f, (target(f, ability=31, tags=(20,), target=20),), p
    )
    assert metrics["ability"] == [1]
    assert metrics["source"] == [0, 0] and metrics["source_mrr"] == [0.5, 0.5]
    assert metrics["J1_teacher_conditioned_command"] == [1]
    assert all(metrics["J" + str(n) + "_teacher_conditioned_command"] == [0] for n in range(2, 8))


@pytest.mark.parametrize("codec", ["xz", "zstd"])
def test_solid_storage_preserves_exact_all_members_and_training_rows(tmp_path, codec):
    from poorstar.data.bc.records import BCShardReader
    from poorstar.data.bc.storage import expand_solid, solid_repack

    chosen, report = build_dataset(tmp_path)
    entry = report["shards"][0]
    artifact = solid_repack(
        entry["path"], sha256=entry["sha256"], output_dir=tmp_path / "packed" / chosen, codec=codec
    )
    split = json.loads((tmp_path / "split.json").read_text())
    a = BCShardReader(
        entry["path"], sha256=entry["sha256"], split_manifest=split, allowed_split=chosen
    )
    b = BCShardReader(
        artifact["path"], sha256=artifact["sha256"], split_manifest=split, allowed_split=chosen
    )
    try:
        assert a.index == b.index
        assert all(a.zip.read(name) == b.zip.read(name) for name in a.zip.namelist())
        x, y = list(a), list(b)
        assert x[0].scheduler.key == y[0].scheduler.key and x[0].targets == y[0].targets
        model = RecurrentBCPolicy(
            BCModelConfig.from_catalog(a.catalog, a.profile, hidden_size=32, maximum_delay=8)
        )
        model.eval()
        assert torch.equal(
            bc_objective(model, [x], a.profile).loss, bc_objective(model, [y], b.profile).loss
        )
        with pytest.raises(ValueError, match="resource guard"):
            expand_solid(artifact["path"], maximum_bytes=8)
    finally:
        a.close()
        b.close()


@pytest.mark.parametrize("codec,suffix", [("xz", ".xz"), ("zstd", ".zst")])
def test_direct_solid_writer_is_readable_without_legacy_archive(tmp_path, codec, suffix):
    from poorstar.data.bc.records import BCShardReader

    rows, p, cat, info = decision_rows()
    split = freeze_split([dict(replay_sha256="a" * 64, match_group="one")], seed=1)
    chosen = split["assignments"]["a" * 64]
    writer = BCShardWriter(
        tmp_path / chosen,
        split_manifest=split,
        split=chosen,
        catalog=cat,
        game_info=info,
        source_manifest_sha256="b" * 64,
        extractor_sha256="c" * 64,
        maximum_delay=8,
        solid_compression=True,
        solid_codec=codec,
    )
    for row in rows:
        writer.append(row)
    result = writer.close()
    assert result["path"].endswith(".bc.zip" + suffix) and not writer.partial.exists()
    reader = BCShardReader(
        result["path"], sha256=result["sha256"], split_manifest=split, allowed_split=chosen
    )
    try:
        assert [r.scheduler.key.loop for r in reader] == [0, 3]
    finally:
        reader.close()


def test_fusion_normalization_prevents_large_branch_erasing_current_observation():
    f, p, model, *_ = model_fixture()
    model.eval()
    obs = f.policy_observation()
    # The observed failure had spatial/fusion amplitudes above 280. A common
    # branch-scale explosion must not make different scalar states identical.
    with torch.no_grad():
        model.spatial_encoder[7].weight.mul_(1000)
        model.spatial_encoder[7].bias.mul_(1000)
    a = model.encode(obs).hidden
    other = f.policy_observation()
    other["scalars"]["minerals"] = 1000
    other["observation_loop"] = 72
    b = model.encode(other).hidden
    assert float((a - b).abs().max()) > 1e-4
    model.zero_grad(set_to_none=True)
    model.encode(other).hidden.sum().backward()
    assert sum(float(p.grad.abs().sum()) for p in model.scalar_projection.parameters()) > 0


def test_geometry_feature_uses_current_position_presence_and_no_identifiers():
    f, p, model, *_ = model_fixture()
    model.eval()
    obs = f.policy_observation()
    # Unknown position values must have no effect on content or geometry branches.
    for e in obs["entities"]:
        e["present"]["world_x"] = False
        e["present"]["world_y"] = False
    a = model.encode(obs).entities.detach()
    for e in obs["entities"]:
        e["values"]["world_x"] = 999999
        e["values"]["world_y"] = -999999
    assert torch.equal(a, model.encode(obs).entities.detach())
    with pytest.raises(ValueError):
        model.encode(dict(obs, raw_tags=[10, 20]))


def test_full_corpus_plan_retains_valid_scope_despite_failed_metadata_row(tmp_path, monkeypatch):
    import gzip

    from poorstar.cli.data.prepare import main
    from poorstar.common.integrity import file_hash

    def write(path, value):
        value["content_digest"] = payload_hash(value)
        path.write_text(json.dumps(value))
        return value["content_digest"]

    entry = dict(
        alias="valid",
        scope_candidate=True,
        file_sha256="a" * 64,
        controlled_zerg_player_ids=[1],
        path=str(tmp_path / "input.SC2Replay"),
        bytes=10,
        map_hash="b" * 64,
        matchup="ZvT",
        header_loops=100,
        history="UNKNOWN_NOT_CONFIRMATORY",
    )
    sd = write(
        tmp_path / "scope.json",
        dict(
            engine_base_build=55958,
            entries=[{"alias": "bad", "problems": ["METADATA_PARSE_FAILURE"]}, entry],
            counts={"compatible_zerg_candidate_perspectives": 1},
        ),
    )
    item = dict(
        alias="valid",
        completion="COMPLETE_NATIVE_EVENT_CENSUS",
        file_sha256="a" * 64,
        controlled_zerg_player_ids=[1],
        native_content_match_key="c" * 64,
    )
    chunk = tmp_path / "chunk.jsonl.gz"
    with gzip.open(chunk, "wt") as f:
        f.write(json.dumps(item) + "\n")
    write(
        tmp_path / "census.json",
        dict(
            source_scope_digest=sd,
            failures=[],
            full_chunk_readback_verified=True,
            chunks=[
                dict(
                    path=chunk.name, sha256=file_hash(chunk), bytes=chunk.stat().st_size, records=1
                )
            ],
        ),
    )
    write(tmp_path / "pilot.json", dict(registered_views=1, shards=[{"bytes": 100}]))
    monkeypatch.setattr(
        "sys.argv",
        [
            "prepare",
            "--scope",
            str(tmp_path / "scope.json"),
            "--native-census",
            str(tmp_path / "census.json"),
            "--pilot-build-report",
            str(tmp_path / "pilot.json"),
            "--output-dir",
            str(tmp_path / "output"),
        ],
    )
    main()
    report = json.loads((tmp_path / "output/report.json").read_text())
    assert (
        report["counts"]["candidate_alias_views"]
        == report["counts"]["unique_byte_identical_views"]
        == 1
    )


def test_parallel_control_blocks_current_cross_head_information_but_keeps_history():
    from poorstar.actions.contract import Token

    f, p, conditional, *_ = model_fixture()
    conditional.eval()
    parallel = RecurrentBCPolicy(replace(conditional.config, autoregressive=False))
    parallel.load_state_dict(conditional.state_dict())
    parallel.eval()
    a = parallel.encode(f.policy_observation())
    b = parallel.encode(f.policy_observation())
    pa = parallel.consume(a, a.hidden, Token("ability", 11))
    pb = parallel.consume(b, b.hidden, Token("ability", 21))
    assert torch.equal(
        parallel.distribution(a, pa, "source", (0, 1)).logits,
        parallel.distribution(b, pb, "source", (0, 1)).logits,
    )
    pa = parallel.consume(a, pa, Token("source", 0))
    pb = parallel.consume(b, pb, Token("source", 1))
    assert torch.equal(
        parallel.distribution(a, pa, "queue", (False, True)).logits,
        parallel.distribution(b, pb, "queue", (False, True)).logits,
    )
    # Both arms retain exactly the same past-action memory for identical tokens.
    x = conditional.encode(f.policy_observation())
    y = parallel.encode(f.policy_observation())
    px = x.hidden
    py = y.hidden
    from poorstar.actions.coarse import coarse_teacher_force

    _, trace, _ = coarse_teacher_force(f, (target(f), target(f, ability=31, target=20)), p)
    for head, options, value in trace:
        px = conditional.consume(x, px, Token(head, value))
        py = parallel.consume(y, py, Token(head, value))
        assert torch.equal(px, py)
    assert y.parallel_command_base is not None and y.parallel_source_state is None
    # This proves the information boundary, not equal FLOPs or formal fairness.


def _full_plan_fixture(root):
    import gzip

    from poorstar.cli.data.build import write_report
    from poorstar.common.integrity import file_hash

    root.mkdir()
    groups = []
    jobs = []
    for index in range(2):
        replay = root / ("replay_" + str(index))
        replay.write_bytes(bytes([index]) * 10)
        sha = file_hash(replay)
        groups.append(dict(replay_sha256=sha, match_group=str(index)))
        jobs.append(
            dict(
                job_id=payload_hash([sha, 1]),
                replay_sha256=sha,
                player_id=1,
                replay=str(replay),
                replay_bytes=10,
            )
        )
    split = freeze_split(groups, seed=12)
    (root / "split.json").write_text(json.dumps(split))
    for job in jobs:
        job["split"] = split["assignments"][job["replay_sha256"]]
    with gzip.open(root / "jobs.jsonl.gz", "wt") as f:
        for job in jobs:
            f.write(json.dumps(job) + "\n")
    write_report(
        root / "report.json",
        dict(
            schema="poorstar_c3_full_corpus_extraction_plan_v1",
            jobs_sha256=file_hash(root / "jobs.jsonl.gz"),
            split_sha256=split["sha256"],
            counts={"unique_byte_identical_views": 2},
        ),
    )
    return jobs, split


def test_full_job_freeze_rejects_split_and_denominator_drift(tmp_path):
    from poorstar.cli.data.build import freeze_jobs, read_verified, write_report

    jobs, split = _full_plan_fixture(tmp_path / "plan")
    assert freeze_jobs(tmp_path / "plan")[2] == jobs
    report, _ = read_verified(tmp_path / "plan/report.json")
    report["counts"]["unique_byte_identical_views"] = 3
    write_report(tmp_path / "plan/report.json", report)
    with pytest.raises(ValueError, match="denominator"):
        freeze_jobs(tmp_path / "plan")


def test_full_coordinator_stops_on_failure_resumes_exact_scope_and_detects_corruption(
    tmp_path, monkeypatch
):
    from poorstar.cli.data import build as coordinator
    from poorstar.common.integrity import file_hash

    jobs, split = _full_plan_fixture(tmp_path / "plan")
    manifest = tmp_path / "source.json"
    manifest.write_text("{}")
    monkeypatch.setattr("poorstar.common.integrity.source_integrity", lambda *x: {"problems": []})
    calls = []
    fail = [True]

    def process(command, log, timeout):
        options = {command[i]: command[i + 1] for i in range(2, len(command) - 1, 2)}
        out = __import__("pathlib").Path(options["--output-dir"])
        out.mkdir()
        calls.append(out)
        if fail[0]:
            fail[0] = False
            return 9
        job = next(j for j in jobs if j["replay"] == options["--replay"])
        target = out / job["split"]
        target.mkdir()
        shard = target / "one.bc.zip.xz"
        shard.write_bytes(b"bound-shard")
        terminal = out / "extraction_terminal.protocol"
        terminal.write_bytes(b"bound-terminal")
        coordinator.write_report(
            out / "report.json",
            dict(
                replay_sha256=job["replay_sha256"],
                controlled_player_id=1,
                split_sha256=split["sha256"],
                source_manifest_sha256=file_hash(manifest),
                engine_base_build=55958,
                full_game=True,
                lossless_solid_compression=True,
                terminal_source="INDEPENDENT_FULL_ENGINE_PASS",
                optimizer_steps=0,
                shards=[
                    dict(
                        path=str(shard),
                        sha256=file_hash(shard),
                        bytes=shard.stat().st_size,
                        decisions=1,
                    )
                ],
                counts={"all_decisions": 1},
                terminal_observation_sha256=file_hash(terminal),
            ),
        )
        return 0

    monkeypatch.setattr(coordinator, "extraction_process", process)
    argv = [
        "build",
        "--plan-dir",
        str(tmp_path / "plan"),
        "--source-manifest",
        str(manifest),
        "--sc2-path",
        str(tmp_path / "engine"),
        "--output-dir",
        str(tmp_path / "output"),
        "--workers",
        "1",
        "--reserve-gib",
        "1",
    ]
    monkeypatch.setattr("sys.argv", argv)
    assert coordinator.main() == 1 and len(calls) == 1  # No second job launch after failure.
    assert (calls[0].parent / "attempt_00000.record.json").exists()
    assert coordinator.main() == 1 and len(calls) == 1  # Failure cannot silently disappear.
    monkeypatch.setattr("sys.argv", argv + ["--retry-failed"])
    assert coordinator.main() == 0 and len(calls) == 3
    report, _ = coordinator.read_verified(tmp_path / "output/report.json")
    assert report["complete_engine_extraction"] and report["completed_views"] == 2
    assert not report["paper_corpus_gate_passed"] and not report["formal_bc_ready"]
    assert calls[1].name == "attempt_00001" and calls[0].exists()
    assert coordinator.main() == 0 and len(calls) == 3  # Valid resume launches no extractor.
    shard = __import__("pathlib").Path(report["shards"][0]["path"])
    shard.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="shard drift"):
        coordinator.main()
    assert len(calls) == 3


def test_shared_corpus_lock_prevents_two_instance_coordinators(tmp_path):
    from poorstar.cli.data.build import corpus_lock

    with corpus_lock(tmp_path):
        with pytest.raises(BlockingIOError):
            with corpus_lock(tmp_path):
                pass
    with corpus_lock(tmp_path):
        pass  # OS released lock; no stale lock deletion.


def test_checkpoint_preserves_whole_run_metrics_and_rejects_missing_aggregate(tmp_path):
    from poorstar.training.foundation import BCAggregate

    rows, p, cat, info = decision_rows()
    cfg = BCModelConfig.from_catalog(cat, p, hidden_size=32, maximum_delay=8)
    trainer = BCTrainer(RecurrentBCPolicy(cfg), p, learning_rate=0.001)
    order = WindowOrder(2, 1)
    aggregate = BCAggregate()
    aggregate.add(trainer.step([rows]))
    expected = aggregate.report()
    save_checkpoint(tmp_path / "full", trainer, order, {}, aggregate=aggregate)
    recovered = BCAggregate()
    load_checkpoint(tmp_path / "full", trainer, order, {}, aggregate=recovered)
    assert recovered.report() == expected
    # A later WAIT-only batch must not erase an earlier sparse entity/position head.
    recovered.add(trainer.step([(rows[1],)]))
    assert recovered.denominators["target_position"] == aggregate.denominators["target_position"]
    assert recovered.denominators["act"] == aggregate.denominators["act"] + 1
    save_checkpoint(tmp_path / "without", trainer, order, {})
    with pytest.raises(ValueError, match="aggregate missing"):
        load_checkpoint(tmp_path / "without", trainer, order, {}, aggregate=BCAggregate())


def test_resource_probe_runs_both_arms_on_same_train_windows(tmp_path, monkeypatch):
    from poorstar.cli.training.benchmark import main

    chosen, _ = build_dataset(tmp_path)
    assert chosen == "train"
    manifest = tmp_path / "source.json"
    manifest.write_text("{}")
    monkeypatch.setattr("poorstar.common.integrity.source_integrity", lambda *x: {"problems": []})
    monkeypatch.setattr(
        "sys.argv",
        [
            "probe",
            "--build-report",
            str(tmp_path / "build.json"),
            "--split-manifest",
            str(tmp_path / "split.json"),
            "--source-manifest",
            str(manifest),
            "--output-dir",
            str(tmp_path / "probe"),
            "--device",
            "cpu",
            "--steps",
            "1",
            "--batch-windows",
            "2",
            "--unroll-length",
            "1",
            "--burn-in-length",
            "0",
        ],
    )
    main()
    report = json.loads((tmp_path / "probe/report.json").read_text())
    assert report["matched_input_and_denominators"] and report["measurement_complete"]
    a, b = report["arms"]
    assert a["parameter_count"] == b["parameter_count"]
    assert a["steps"][0]["denominators"] == b["steps"][0]["denominators"]
    assert a["steps"][0]["active_rows"] == b["steps"][0]["active_rows"] == 2
    assert not report["formal_bc_started"] and not report["full_corpus_throughput_certified"]


def test_extractor_timeout_closes_engine_group_even_when_parent_exits_on_term(monkeypatch):
    import signal
    import subprocess

    from poorstar.cli.data import build as coordinator

    class Child:
        pid = 987654
        calls = 0

        def wait(self, timeout=None):
            self.calls += 1
            if self.calls == 1:
                raise subprocess.TimeoutExpired("extract", timeout)
            return -signal.SIGTERM

    child = Child()
    signals = []
    monkeypatch.setattr(coordinator.subprocess, "Popen", lambda *a, **k: child)
    monkeypatch.setattr(coordinator.os, "killpg", lambda pid, sig: signals.append((pid, sig)))
    with pytest.raises(subprocess.TimeoutExpired):
        coordinator.extraction_process(["extract"], None, 1)
    assert signals == [(child.pid, signal.SIGTERM), (child.pid, signal.SIGKILL)]
