"""Tests: test league loop."""

import json
import subprocess
import sys
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from poorstar.league import (
    LeagueBudget,
    LeagueCoordinator,
    LeagueMember,
    LeagueRunPlan,
    PromotionRules,
    SnapshotRegistry,
    WorkerResult,
    assert_matched_leagues,
)
from poorstar.league.fixture import FixtureDispatcher
from poorstar.league.ownership import read_document, verify_worker_lease, write_document
from poorstar.league.promotion import evaluate_promotion
from poorstar.training import TrainingIdentity
from poorstar.training.contracts import digest


def setup(tmp_path, *, specialized=False, rounds=1, games=1, attempts=2):
    identity = TrainingIdentity(
        "league-mechanism",
        "persistent_intent",
        17,
        "a" * 64,
        "b" * 64,
        "c" * 64,
        "d" * 64,
        "e" * 64,
        "f" * 64,
        "1" * 64,
    )
    weights = tmp_path / "opaque_fixture.bin"
    weights.write_bytes(b"NOT_A_TORCH_ACTOR_OR_GAME_POLICY")
    registry = SnapshotRegistry(tmp_path / "registry")
    main = registry.register(
        weights,
        identity=identity,
        policy_version=0,
        role="intent_specialist" if specialized else "main",
        intent_profile={"economy": 0.5} if specialized else None,
    )
    anchor = registry.register(weights, identity=identity, policy_version=0, role="anchor")
    historical = registry.register(weights, identity=identity, policy_version=0, role="historical")
    budget = LeagueBudget(
        500,
        2000,
        1000000,
        1000000,
        1000000,
        20,
        10000,
        10000,
        100,
        100,
        100,
        50,
        "2" * 64,
        "3" * 64,
        digest({}),
    )
    plan = LeagueRunPlan(
        "intent_specialized" if specialized else "ordinary",
        17,
        (LeagueMember("main", main["snapshot_id"], asdict(identity), 100, 50),),
        (anchor["snapshot_id"],),
        (historical["snapshot_id"],),
        ("FIXTURE_NO_MAP",),
        (11, 12),
        rounds,
        games,
        1,
        attempts,
        budget,
        PromotionRules(0.0, 0.0, 1.0, 0.1, 2),
    )
    return identity, registry, plan


def full_run(coordinator, dispatcher=None):
    dispatcher = dispatcher or FixtureDispatcher()
    values = []
    for _ in range(200):
        value = coordinator.run_next(dispatcher)
        if value is None:
            return values
        values.append(value)
    raise AssertionError("fixture did not exhaust bounded loop")


def test_complete_loop_has_fixed_independent_panel_and_keeps_history(tmp_path):
    identity, registry, plan = setup(tmp_path, rounds=2)
    coordinator = LeagueCoordinator(tmp_path / "run", registry, identity, plan, fixture=True)
    try:
        values = full_run(coordinator)
        report = coordinator.report()
        assert report["run_complete"] and report["operations"] == {"COMPLETE": 22}
        assert report["usage"]["updates"] == 2 and not any(report["compute_exceeded"].values())
        assert (
            report["fixture"]
            and not report["engine_started"]
            and not report["science_h4_e6_confirmed"]
        )
        assert (
            len(registry.list()) == 5
            and registry.read(plan.members[0].snapshot_id)["policy_version"] == 0
        )
        assert registry.read(report["current"]["main"])["policy_version"] == 2
        assert report["promoted"] == report["current"]
        assert (
            len(report["training_history"]) == 3
            and plan.members[0].snapshot_id in report["training_history"]
        )
        assert len([v for v in values if v.get("promoted")]) == 2
        jobs = [json.loads(row[0]) for row in coordinator.db.execute("SELECT spec FROM jobs")]
        assert len([j for j in jobs if j["provenance"] == "learner"]) == 2
        evaluator = [j for j in jobs if j["provenance"] == "evaluator"]
        assert len(evaluator) == 16
        assert {j["seed"] for j in evaluator} == {11, 12}
        assert {j["opponent"] for j in evaluator} == set(plan.anchors + plan.historical)
        training = coordinator.scheduler.payoff(plan.members[0].snapshot_id, plan.historical[0])
        assert training["games"] == 1
    finally:
        coordinator.close()


def test_failed_promotion_preserves_opponent_pool_but_continues_challenger(tmp_path):
    identity, registry, plan = setup(tmp_path, rounds=2)
    plan = replace(plan, promotion=PromotionRules(0.9, 0.9, 0.1, 0.1, 2))
    coordinator = LeagueCoordinator(tmp_path / "run", registry, identity, plan, fixture=True)
    try:
        values = full_run(coordinator)
        report = coordinator.report()
        assert report["run_complete"] and report["promoted"]["main"] == plan.members[0].snapshot_id
        assert registry.read(report["current"]["main"])["policy_version"] == 2
        assert [v["status"] for v in values if "promoted" in v] == ["FAIL", "FAIL"]
        panels = [
            json.loads(raw)
            for (raw,) in coordinator.db.execute("SELECT spec FROM jobs")
            if json.loads(raw)["provenance"] == "evaluator"
        ]
        # Failed promotion cannot silently replace the next-round incumbent.
        assert sum(j["learner"] == plan.members[0].snapshot_id for j in panels) == 8
    finally:
        coordinator.close()


def test_resume_exact_operations_no_extra_payoff_or_compute(tmp_path):
    identity, registry, plan = setup(tmp_path)
    run = tmp_path / "run"
    first = LeagueCoordinator(run, registry, identity, plan, fixture=True)
    first.run_next(FixtureDispatcher())
    usage = first.report()["usage"]
    first.close()
    second = LeagueCoordinator(run, registry, identity, plan, fixture=True)
    try:
        assert second.recover() == 0 and second.report()["usage"] == usage
        full_run(second)
        report = second.report()
        assert report["run_complete"]
        assert second.run_next(FixtureDispatcher()) is None and second.report() == report
    finally:
        second.close()


def test_single_owner_and_mode_or_host_binding_cannot_relabel_fixture(tmp_path):
    identity, registry, plan = setup(tmp_path)
    first = LeagueCoordinator(tmp_path / "run", registry, identity, plan, fixture=True)
    try:
        with pytest.raises(ValueError, match="another local"):
            LeagueCoordinator(tmp_path / "run", registry, identity, plan, fixture=True)
    finally:
        first.close()
    with pytest.raises(ValueError, match="fixture/plan"):
        LeagueCoordinator(tmp_path / "run", registry, identity, plan, fixture=False)


def test_interrupted_fixture_is_charged_full_reserved_frames_and_compute(tmp_path):
    identity, registry, plan = setup(tmp_path)
    run = tmp_path / "run"
    first = LeagueCoordinator(run, registry, identity, plan, fixture=True)

    def crash(request):
        raise RuntimeError("deliberate process fixture interruption")

    with pytest.raises(RuntimeError, match="interruption"):
        first.run_next(crash)
    first.close()
    second = LeagueCoordinator(run, registry, identity, plan, fixture=True)
    try:
        assert second.recover() == 1
        report = second.report()
        assert report["usage"]["actor_compute_millis"] == 10000
        assert report["scheduler"]["budgets"]["learner"]["spent"] == 20
        full_run(second)
        assert second.report()["run_complete"]
        assert second.report()["scheduler"]["statuses"]["INCOMPLETE"] == 1
        assert second.report()["scheduler"]["budgets"]["learner"]["spent"] == 23
    finally:
        second.close()


def test_crash_between_payoff_commit_and_operation_commit_is_idempotent(tmp_path, monkeypatch):
    identity, registry, plan = setup(tmp_path)
    run = tmp_path / "run"
    first = LeagueCoordinator(run, registry, identity, plan, fixture=True)
    original = first._event

    def fail(kind, value):
        if kind == "WORKER_COMMITTED":
            raise RuntimeError("after scheduler commit")
        original(kind, value)

    monkeypatch.setattr(first, "_event", fail)
    with pytest.raises(RuntimeError, match="scheduler commit"):
        first.run_next(FixtureDispatcher())
    assert first.scheduler.payoff(plan.members[0].snapshot_id, plan.historical[0])["games"] == 1
    first.close()
    second = LeagueCoordinator(run, registry, identity, plan, fixture=True)
    try:
        assert second.recover() == 1
        assert (
            second.scheduler.payoff(plan.members[0].snapshot_id, plan.historical[0])["games"] == 1
        )
        assert second.scheduler.report()["budgets"]["learner"]["spent"] == 3
    finally:
        second.close()


@pytest.mark.parametrize(
    "field", ["claim_token", "attempt_id", "request_sha256", "owned_sessions_closed"]
)
def test_worker_result_cannot_cross_owned_attempt(tmp_path, field):
    identity, registry, plan = setup(tmp_path)
    coordinator = LeagueCoordinator(tmp_path / "run", registry, identity, plan, fixture=True)

    def wrong(request):
        result = FixtureDispatcher()(request)
        return replace(result, **{field: False if field == "owned_sessions_closed" else "9" * 64})

    try:
        with pytest.raises(ValueError):
            coordinator.run_next(wrong)
        assert coordinator.scheduler.report()["budgets"]["learner"]["spent"] == 0
        assert coordinator.scheduler.report()["budgets"]["learner"]["reserved"] == 20
    finally:
        coordinator.close()


def test_compute_overspend_retained_and_new_work_blocked_before_queue(tmp_path):
    identity, registry, plan = setup(tmp_path)
    plan = replace(plan, budget=replace(plan.budget, actor_compute_millis=10000))
    coordinator = LeagueCoordinator(tmp_path / "run", registry, identity, plan, fixture=True)

    def overspend(request):
        return replace(FixtureDispatcher()(request), compute_millis=10001)

    try:
        assert coordinator.run_next(overspend)["worker_status"] == "ERROR"
        assert coordinator.report()["usage"]["actor_compute_millis"] == 10001
        assert coordinator.report()["compute_exceeded"]["actor_compute_millis"]
        before = coordinator.db.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]
        with pytest.raises(ValueError, match="compute budget"):
            coordinator.run_next(FixtureDispatcher())
        assert coordinator.db.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == before
    finally:
        coordinator.close()


def test_budget_validation_and_matching_are_treatment_blind(tmp_path):
    identity, registry, ordinary = setup(tmp_path)
    # Same weights, anchors, seeds and numerical budget, a declared specialist
    # occupies the same population slot and parameter allocation.
    specialist = registry.register(
        registry.weights_path(ordinary.members[0].snapshot_id),
        identity=identity,
        policy_version=0,
        role="intent_specialist",
        intent_profile={"economy": 0.5},
    )
    specialized = replace(
        ordinary,
        kind="intent_specialized",
        members=(replace(ordinary.members[0], snapshot_id=specialist["snapshot_id"]),),
    )
    assert assert_matched_leagues(ordinary, specialized) == ordinary.matched_budget_sha256
    with pytest.raises(ValueError, match="matched frames"):
        assert_matched_leagues(
            ordinary,
            replace(
                specialized, budget=replace(specialized.budget, evaluation_compute_millis=1000001)
            ),
        )
    with pytest.raises(ValueError, match="all immutable versions"):
        replace(ordinary, budget=replace(ordinary.budget, population_limit=3))
    with pytest.raises(ValueError, match="parameter budget"):
        replace(ordinary, budget=replace(ordinary.budget, maximum_parameters=99))


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "training", "truncated"])
def test_promotion_requires_complete_fixed_natural_evaluator_panel(tmp_path, mutation):
    _, _, plan = setup(tmp_path)
    panel = [
        dict(opponent=o, map_name=m, seed=s, provenance="evaluator", verdict="win")
        for o in plan.anchors + plan.historical
        for m in plan.maps
        for s in plan.evaluation_seeds
    ]
    changed = [dict(v) for v in panel]
    if mutation == "missing":
        changed.pop()
    elif mutation == "duplicate":
        changed.append(dict(changed[0]))
    elif mutation == "training":
        changed[0]["provenance"] = "learner"
    else:
        changed[0]["verdict"] = "truncated"
    if mutation in ("duplicate", "training"):
        with pytest.raises(ValueError, match="frozen evaluation panel"):
            evaluate_promotion(plan, panel, changed)
    else:
        assert evaluate_promotion(plan, panel, changed)["status"] == "NOT_EVALUATED"


def test_fixed_anchor_regression_blocks_promotion_even_with_history_wins(tmp_path):
    _, _, plan = setup(tmp_path)
    plan = replace(plan, promotion=PromotionRules(0.4, 0.4, 0.1, 0.1, 2))
    old = [
        dict(opponent=o, map_name=m, seed=s, provenance="evaluator", verdict="win")
        for o in plan.anchors + plan.historical
        for m in plan.maps
        for s in plan.evaluation_seeds
    ]
    new = [{**v, "verdict": "loss" if v["opponent"] in plan.anchors else "win"} for v in old]
    assert evaluate_promotion(plan, old, new)["status"] == "FAIL"


def test_lease_drift_rejected_and_independent_evaluator_cannot_train(tmp_path):
    identity, registry, plan = setup(tmp_path)
    coordinator = LeagueCoordinator(tmp_path / "run", registry, identity, plan, fixture=True)

    def change(request):
        lease = read_document(coordinator.root / "dispatch_lease.json")
        lease["claim_token"] = "0" * 64
        write_document(coordinator.root / "dispatch_lease.json", lease)
        with pytest.raises(ValueError, match="lease request"):
            verify_worker_lease(request)
        raise RuntimeError("drift preserved")

    try:
        with pytest.raises(RuntimeError):
            coordinator.run_next(change)
    finally:
        coordinator.close()


def test_cli_fixture_completes_and_default_only_plans(tmp_path):
    identity, registry, plan = setup(tmp_path)
    ip = tmp_path / "identity.json"
    ip.write_text(json.dumps(asdict(identity)))
    pp = tmp_path / "plan.json"
    pp.write_text(json.dumps(asdict(plan)))
    base = [
        sys.executable,
        "poorstar/cli/league/run.py",
        "--identity",
        str(ip),
        "--run-plan",
        str(pp),
        "--registry",
        str(registry.root),
        "--maximum-operations",
        "100",
    ]
    queued = subprocess.run(
        base + ["--output-dir", str(tmp_path / "planned")], capture_output=True, text=True
    )
    assert queued.returncode == 0, queued.stderr
    assert json.loads(queued.stdout)["status"] == "PLAN_ONLY"
    actual = subprocess.run(
        base + ["--output-dir", str(tmp_path / "fixture"), "--fixture-run"],
        capture_output=True,
        text=True,
    )
    assert actual.returncode == 0, actual.stderr
    result = json.loads(actual.stdout)
    assert (
        result["run_complete"]
        and result["status"] == "FIXTURE_ONLY"
        and not result["engine_started"]
    )


def test_owned_nonengine_subprocess_dispatch_registers_and_closes(tmp_path, monkeypatch):
    from poorstar.league.dispatch import SubprocessDispatcher
    from poorstar.training.checkpoint import file_hash

    identity, registry, plan = setup(tmp_path)
    coordinator = LeagueCoordinator(tmp_path / "run", registry, identity, plan, fixture=True)
    worker = tmp_path / "isolated_worker.py"
    worker.write_text(
        """import argparse,sys\nfrom pathlib import Path\nsys.path.insert(0, """
        + repr(str(Path(__file__).resolve().parents[1]))
        + """)\nfrom dataclasses import asdict\nfrom poorstar.league.contracts import WorkerRequest\nfrom poorstar.league.fixture import FixtureDispatcher\nfrom poorstar.league.ownership import read_document,write_document\np=argparse.ArgumentParser();p.add_argument('--request-json');p.add_argument('--result-json');a=p.parse_args()\nr=WorkerRequest(**read_document(a.request_json));result=FixtureDispatcher()(r);write_document(a.result_json,asdict(result),immutable=True)\n"""
    )
    monkeypatch.setattr("poorstar.league.dispatch.authorize_worker", verify_worker_lease)
    dispatcher = SubprocessDispatcher(worker, actor_worker_sha256=file_hash(worker))
    try:
        result = coordinator.run_next(dispatcher)
        assert result["worker_status"] == "COMPLETE" and result["charged_compute_millis"] > 1
        directory = next((coordinator.root / "attempts").glob("*/*"))
        launch = read_document(directory / "worker_launch.json")
        closed = read_document(directory / "worker_exit.json")
        assert closed["all_owned_sessions_closed"] and closed["returncode"] == 0
        assert closed["owner"] == launch and not closed["signalled_once"]
        assert read_document(directory / "actor_report.json")["SC2_engine_launches"] == 0
    finally:
        coordinator.close()


def test_nonzero_owned_worker_preserves_qualified_error_evidence_and_actual_frames(
    tmp_path, monkeypatch
):
    from poorstar.league.dispatch import SubprocessDispatcher
    from poorstar.training.checkpoint import file_hash

    identity, registry, plan = setup(tmp_path)
    coordinator = LeagueCoordinator(tmp_path / "run", registry, identity, plan, fixture=True)
    worker = tmp_path / "isolated_error_worker.py"
    worker.write_text(
        """import argparse,sys\nfrom pathlib import Path\nsys.path.insert(0, """
        + repr(str(Path(__file__).resolve().parents[1]))
        + """)\nfrom dataclasses import asdict,replace\nfrom poorstar.league.contracts import WorkerRequest\nfrom poorstar.league.fixture import FixtureDispatcher\nfrom poorstar.league.ownership import read_document,write_document\nfrom poorstar.training.checkpoint import file_hash\np=argparse.ArgumentParser();p.add_argument('--request-json');p.add_argument('--result-json');a=p.parse_args()\nr=WorkerRequest(**read_document(a.request_json));result=FixtureDispatcher()(r)\nfailure=Path(r.output_dir)/'qualified_fixture_failure.json'\nwrite_document(failure,dict(request_sha256=r.sha256,SC2_engine_launches=0,owned_sessions_closed=True,error='deliberate qualified nonzero exit'),immutable=True)\nartifacts={**result.artifacts,'failure':dict(path=str(failure),sha256=file_hash(failure))}\nresult=replace(result,status='ERROR',compute_millis=1000,artifacts=artifacts)\nwrite_document(a.result_json,asdict(result),immutable=True)\nsys.exit(7)\n"""
    )
    monkeypatch.setattr("poorstar.league.dispatch.authorize_worker", verify_worker_lease)
    dispatcher = SubprocessDispatcher(worker, actor_worker_sha256=file_hash(worker))
    try:
        result = coordinator.run_next(dispatcher)
        assert result["worker_status"] == "ERROR" and result["environment_loops"] == 3
        assert result["charged_compute_millis"] >= 1000
        assert {"report", "rollout_index", "failure"} <= set(result["artifacts"])
        assert (
            file_hash(result["artifacts"]["failure"]["path"])
            == result["artifacts"]["failure"]["sha256"]
        )
        assert read_document(result["artifacts"]["failure"]["path"])["owned_sessions_closed"]
        directory = next((coordinator.root / "attempts").glob("*/*"))
        exit_record = read_document(directory / "worker_exit.json")
        assert exit_record["returncode"] == 7 and exit_record["all_owned_sessions_closed"]
        assert coordinator.scheduler.report()["budgets"]["learner"]["spent"] == 3
        assert (
            coordinator.scheduler.payoff(plan.members[0].snapshot_id, plan.historical[0])["games"]
            == 0
        )
    finally:
        coordinator.close()


@pytest.mark.parametrize("mutation", ["descendant", "owner", "receipt", "alive", "closed"])
def test_recovery_requires_qualified_receipt_and_all_recorded_sessions(
    tmp_path, monkeypatch, mutation
):
    import socket

    from poorstar.league.contracts import WorkerRequest
    from poorstar.league.dispatch import verify_closed_attempt

    identity, registry, plan = setup(tmp_path)
    coordinator = LeagueCoordinator(tmp_path / "run", registry, identity, plan, fixture=True)

    def interrupted(request):
        raise RuntimeError("staged recovery fixture")

    try:
        with pytest.raises(RuntimeError):
            coordinator.run_next(interrupted)
        request = WorkerRequest(
            **json.loads(coordinator.db.execute("SELECT request FROM attempts").fetchone()[0])
        )
        directory = Path(request.output_dir)
        owner = dict(
            host=socket.gethostname(),
            pid=1000001,
            starttime="old",
            sid=1000001,
            cwd=str(tmp_path),
            command=["EXPLICIT_NONENGINE_FIXTURE"],
            request_sha256=request.sha256,
        )
        write_document(directory / "worker_launch.json", owner, immutable=True)
        sessions = dict(owner=owner, tracked_sessions=[1000001, 1000002])
        if mutation == "owner":
            sessions["owner"] = {**owner, "request_sha256": "0" * 64}
        write_document(directory / "worker_sessions.json", sessions)
        write_document(
            directory / "worker_exit.json",
            dict(
                owner=owner,
                tracked_sessions=[1000001],
                all_owned_sessions_closed=mutation != "closed",
                compute_millis=9,
            ),
            immutable=True,
        )
        result = FixtureDispatcher()(request)
        if mutation != "receipt":
            write_document(directory / "worker_result.json", asdict(result), immutable=True)
        monkeypatch.setattr(
            "poorstar.league.dispatch.process_starttime",
            lambda pid: "old" if mutation == "alive" else None,
        )
        monkeypatch.setattr(
            "poorstar.league.dispatch.session_members",
            lambda sid: [{"pid": 3}] if mutation == "descendant" and sid == 1000002 else [],
        )
        with pytest.raises(ValueError):
            verify_closed_attempt(request)
        assert coordinator.scheduler.report()["budgets"]["learner"]["spent"] == 0
    finally:
        coordinator.close()


def test_closed_recovery_keeps_complete_session_union_and_measured_compute(tmp_path, monkeypatch):
    import socket

    from poorstar.league.contracts import WorkerRequest
    from poorstar.league.dispatch import verify_closed_attempt

    identity, registry, plan = setup(tmp_path)
    coordinator = LeagueCoordinator(tmp_path / "run", registry, identity, plan, fixture=True)

    def interrupted(request):
        raise RuntimeError("staged recovery fixture")

    try:
        with pytest.raises(RuntimeError):
            coordinator.run_next(interrupted)
        request = WorkerRequest(
            **json.loads(coordinator.db.execute("SELECT request FROM attempts").fetchone()[0])
        )
        directory = Path(request.output_dir)
        owner = dict(
            host=socket.gethostname(),
            pid=1000001,
            starttime="old",
            sid=1000001,
            request_sha256=request.sha256,
        )
        write_document(directory / "worker_launch.json", owner, immutable=True)
        write_document(
            directory / "worker_sessions.json",
            dict(owner=owner, tracked_sessions=[1000001, 1000002]),
        )
        write_document(
            directory / "worker_exit.json",
            dict(
                owner=owner,
                tracked_sessions=[1000001, 1000003],
                all_owned_sessions_closed=True,
                compute_millis=91,
            ),
            immutable=True,
        )
        write_document(
            directory / "worker_result.json", asdict(FixtureDispatcher()(request)), immutable=True
        )
        seen = []
        monkeypatch.setattr("poorstar.league.dispatch.process_starttime", lambda pid: None)
        monkeypatch.setattr(
            "poorstar.league.dispatch.session_members", lambda sid: seen.append(sid) or []
        )
        assert verify_closed_attempt(request) == 91
        assert set(seen) == {1000001, 1000002, 1000003}
    finally:
        coordinator.close()


def test_authorization_requires_both_distinct_stage_preflights(tmp_path, monkeypatch):
    from poorstar.league.ownership import authorize_worker, worker_execution_digest

    identity, registry, plan = setup(tmp_path)
    coordinator = LeagueCoordinator(tmp_path / "run", registry, identity, plan, fixture=True)
    calls = []

    def gate(path, identity, stage, **kwargs):
        calls.append((stage, kwargs["execution_config_sha256"]))

    monkeypatch.setattr("poorstar.training.gates.validate_launch", gate)

    def worker(request):
        authorize_worker(request)
        assert calls == [
            ("SELFPLAY", worker_execution_digest(request)),
            ("LEAGUE", worker_execution_digest(request)),
        ]
        return FixtureDispatcher()(request)

    try:
        coordinator.run_next(worker)
    finally:
        coordinator.close()


def test_truncated_actor_preserves_bootstrap_trajectory_without_payoff(tmp_path):
    from poorstar.training.checkpoint import file_hash

    identity, registry, plan = setup(tmp_path)
    coordinator = LeagueCoordinator(tmp_path / "run", registry, identity, plan, fixture=True)

    def truncated(request):
        result = FixtureDispatcher()(request)
        path = Path(result.artifacts["report"]["path"])
        report = read_document(path)
        report.update(
            terminated=False, truncated=True, reward=0.0, boundary_reason="resident_step_cap"
        )
        # A fixture writes one final file; no production output is rewritten.
        changed = path.parent / "truncated_report.json"
        write_document(changed, report, immutable=True)
        artifacts = {
            **result.artifacts,
            "report": dict(path=str(changed), sha256=file_hash(changed)),
        }
        return replace(result, status="TRUNCATED", artifacts=artifacts)

    try:
        assert coordinator.run_next(truncated)["worker_status"] == "TRUNCATED"
        assert (
            coordinator.scheduler.payoff(plan.members[0].snapshot_id, plan.historical[0])["games"]
            == 0
        )
        assert coordinator.run_next(FixtureDispatcher())["worker_status"] == "COMPLETE"
    finally:
        coordinator.close()


@pytest.mark.parametrize("games", [1, 2])
def test_learner_adapter_real_cli_continues_optimizer_and_prefix_on_protocol_fixture(
    tmp_path, monkeypatch, games
):
    """One VALUE_ONLY mechanism update, not SC2/RL capability or a formal run."""
    import torch

    from poorstar.league.dispatch import execute_learner_request
    from poorstar.training.checkpoint import file_hash, save_checkpoint
    from poorstar.training.export import export_actor, load_actor
    from poorstar.training.runtime import configure_bc_runtime
    from poorstar.training.storage import RolloutWriter
    from tests.test_training_system_foundation import sampled_sequence, system

    previous_threads = torch.get_num_threads()
    configure_bc_runtime("cpu")
    torch.set_num_threads(1)
    try:
        learner, profile, catalog = system("flat", "VALUE_ONLY")
        learner.config = replace(learner.config, maximum_batch_sequences=games)
        learner.identity = replace(learner.identity, budget_sha256=learner.config.sha256)
        tree = Path(__file__).resolve().parents[1]
        cli = tree / "poorstar/cli/training/train.py"
        # Explicit mechanism source subset, never a complete-suite certificate.
        source = tmp_path / "mechanism_cli_source.json"
        source.write_text(
            json.dumps(
                dict(
                    files=[
                        dict(
                            path=str(cli.relative_to(tree)),
                            sha256=file_hash(cli),
                            bytes=cli.stat().st_size,
                        )
                    ],
                    scope="MECHANISM_CLI_SOURCE_SUBSET_NOT_RELEASE_QUALIFICATION",
                )
            )
        )
        learner.identity = replace(learner.identity, source_sha256=file_hash(source))
        identity = learner.identity
        registry = SnapshotRegistry(tmp_path / "actual_registry")
        exported = export_actor(tmp_path / "initial_actor.pt", learner)
        main = registry.register(exported["path"], identity=identity, policy_version=0, role="main")
        anchor = registry.register(
            exported["path"], identity=identity, policy_version=0, role="anchor"
        )
        history = registry.register(
            exported["path"], identity=identity, policy_version=0, role="historical"
        )
        model_config = tmp_path / "model.json"
        model_config.write_text(
            json.dumps(
                dict(
                    bc=asdict(learner.model.config),
                    hierarchy=asdict(learner.model.hierarchy_config),
                )
            )
        )
        numerical = tmp_path / "ppo.json"
        numerical.write_text(json.dumps(asdict(learner.config)))
        empty = tmp_path / "consumed_prefix.json"
        write_document(
            empty,
            dict(
                schema="poorstar_v3_rollout_index_v1",
                identity_sha256=identity.sha256,
                shards=[],
                update_groups=[],
                physical_environment_loops=0,
            ),
            immutable=True,
        )
        save_checkpoint(
            tmp_path / "bootstrap", learner, sampler_state=dict(cursor=0, consumed_groups=[])
        )
        manifest = tmp_path / "bootstrap/manifest.json"
        total = sum(p.numel() for p in learner.model.parameters())
        trainable = sum(p.numel() for p in learner.model.parameters() if p.requires_grad)
        budget = LeagueBudget(
            500,
            2000,
            1000000,
            1000000,
            1000000,
            20,
            10000,
            30000,
            3,
            10,
            total,
            trainable,
            "2" * 64,
            digest(asdict(learner.config)),
            digest({}),
        )
        execution = dict(
            training_cli=str(cli),
            training_cli_sha256=file_hash(cli),
            stage="VALUE_ONLY",
            model_config=str(model_config),
            training_plan=str(numerical),
            source_manifest=str(source),
            device="cpu",
            bootstrap={
                "flat:71": dict(
                    checkpoint_manifest=dict(path=str(manifest), sha256=file_hash(manifest)),
                    accumulated_rollout_index=dict(path=str(empty), sha256=file_hash(empty)),
                )
            },
        )
        plan = LeagueRunPlan(
            "ordinary",
            71,
            (LeagueMember("main", main["snapshot_id"], asdict(identity), total, trainable),),
            (anchor["snapshot_id"],),
            (history["snapshot_id"],),
            ("FIXTURE_NO_MAP",),
            (11, 12),
            1,
            games,
            1,
            1,
            budget,
            PromotionRules(0, 0, 1, 0.1, 2),
            learner_execution=execution,
        )
        coordinator = LeagueCoordinator(
            tmp_path / "actual_fixture", registry, identity, plan, fixture=True
        )
        monkeypatch.setattr("poorstar.league.dispatch.authorize_worker", verify_worker_lease)

        def actor(request):
            result = FixtureDispatcher()(request)
            directory = Path(request.output_dir)
            sequence = sampled_sequence(
                learner, profile, episode="league-protocol-fixture-" + request.operation_id
            )
            writer = RolloutWriter(
                directory / "actual_shards", identity, catalog, maximum_sources=63, maximum_delay=8
            )
            for step in sequence.steps:
                writer.append(step)
            shard = writer.close()
            index = directory / "actual_rollout_index.json"
            write_document(
                index,
                dict(
                    schema="poorstar_v3_rollout_index_v1",
                    identity_sha256=identity.sha256,
                    job_id=request.payload["job"]["job_id"],
                    physical_environment_loops=3,
                    shards=[shard],
                    update_groups=[[shard["sha256"]]],
                    fixture="PROTOCOL_FIXTURE_NO_SC2",
                ),
                immutable=True,
            )
            return replace(
                result,
                artifacts={
                    **result.artifacts,
                    "rollout_index": dict(path=str(index), sha256=file_hash(index)),
                },
            )

        try:
            for _ in range(games):
                coordinator.run_next(actor)
            result = coordinator.run_next(execute_learner_request)
            assert result["worker_status"] == "COMPLETE" and result["charged_updates"] == 1
            current = registry.read(coordinator.report()["current"]["main"])
            assert current["policy_version"] == 1
            _, loaded_identity, record = load_actor(
                registry.weights_path(current["snapshot_id"]), sha256=current["weights_sha256"]
            )
            assert loaded_identity == identity and record["stage_counters"]["optimizer_steps"] == 1
            training = coordinator._state("training:main")
            prefix = read_document(training["accumulated_rollout_index"]["path"])
            assert (
                prefix["physical_environment_loops"] == 3 * games
                and len(prefix["update_groups"]) == 1
                and len(prefix["update_groups"][0]) == games
            )
            continued = read_document(training["checkpoint_manifest"]["path"])
            assert continued["updates"] == 1
            assert read_document(manifest)["updates"] == 0  # old evidence remains immutable
        finally:
            coordinator.close()
    finally:
        torch.set_num_threads(previous_threads)


def test_formal_gate_failure_reserves_no_job_and_launches_no_worker(tmp_path):
    identity, registry, plan = setup(tmp_path)
    coordinator = LeagueCoordinator(
        tmp_path / "gate_blocked", registry, identity, plan, fixture=False
    )
    try:
        with pytest.raises(ValueError, match="FORMAL"):
            coordinator.run_next(lambda request: pytest.fail("worker should not run"))
        assert coordinator.db.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 0
        assert coordinator.db.execute("SELECT COUNT(*) FROM attempts").fetchone()[0] == 0
        assert coordinator.report()["usage"]["actor_compute_millis"] == 0
    finally:
        coordinator.close()
