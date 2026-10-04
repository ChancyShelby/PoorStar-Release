"""Tests: test collection admission."""

import copy
import signal
import threading
import time
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from poorstar.league.ownership import read_document, write_document
from poorstar.training.admission import validate_rollout_admission
from poorstar.training.checkpoint import file_hash
from poorstar.training.contracts import digest
from poorstar.training.deadline import CollectionBudgetExceeded, CollectionDeadline
from poorstar.training.stage_actor import execute_stage_actor
from tests.test_stage_actor import OwnedProtocolPair, stage_fixture
from tests.test_training_system import learner, sequence


@pytest.fixture(autouse=True)
def one_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def test_blocked_operation_is_interrupted_and_prior_handler_restored():
    previous = signal.getsignal(signal.SIGALRM)
    deadline = CollectionDeadline(40)
    started = time.monotonic()
    try:
        deadline.arm(required=True)
        with pytest.raises(CollectionBudgetExceeded):
            time.sleep(10)
    finally:
        deadline.disarm()
    assert time.monotonic() - started < 2
    assert signal.getsignal(signal.SIGALRM) == previous
    assert signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0)


def test_existing_timer_is_never_replaced():
    previous = signal.getsignal(signal.SIGALRM)
    signal.setitimer(signal.ITIMER_REAL, 60)
    try:
        with pytest.raises(ValueError, match="another active"):
            CollectionDeadline(100).arm(required=True)
        assert signal.getitimer(signal.ITIMER_REAL)[0] > 50
        assert signal.getsignal(signal.SIGALRM) == previous
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)


def test_real_guard_cannot_silently_run_in_worker_thread():
    failures = []

    def worker():
        try:
            CollectionDeadline(100).arm(required=True)
        except ValueError as e:
            failures.append(str(e))

    thread = threading.Thread(target=worker)
    thread.start()
    thread.join()
    assert len(failures) == 1 and "main-thread" in failures[0]


def test_deadline_stops_before_next_engine_event_and_preserves_failure(tmp_path, monkeypatch):
    # No sleep/timer race: the first advance expires the injected wall clock.
    expired = [False]

    class Clock(CollectionDeadline):
        def arm(self, **_):
            pass

        def check(self):
            if expired[0]:
                raise CollectionBudgetExceeded("fixture deadline expired")

    monkeypatch.setattr("poorstar.training.stage_actor.CollectionDeadline", Clock)

    class ExpireAfterAdvance(OwnedProtocolPair):
        def advance(self, loops):
            super().advance(loops)
            expired[0] = True

    _, registry, plan = stage_fixture(tmp_path)
    backend = ExpireAfterAdvance()
    result = execute_stage_actor(plan, registry, tmp_path / "output", protocol_fixture=backend)
    assert result["status"] == "ERROR" and result["all_owned_sessions_closed"]
    assert len(backend.advances) == 1 and backend.closed
    assert "rollout_index" not in result["artifacts"] and "failure" in result["artifacts"]
    assert not (tmp_path / "output/rollout_index.json").exists()
    assert result["actual_engine_frames"] == 0


@pytest.mark.parametrize("boundary", ["complete", "truncated"])
def test_successful_stage_admission_has_exact_receipt(tmp_path, boundary):
    train, registry, plan = stage_fixture(tmp_path)
    result = execute_stage_actor(
        plan,
        registry,
        tmp_path / "output",
        protocol_fixture=OwnedProtocolPair(3 if boundary == "complete" else None),
    )
    assert result["status"] == ("COMPLETE" if boundary == "complete" else "TRUNCATED")
    path = result["artifacts"]["rollout_index"]["path"]
    assert validate_rollout_admission(path, train.identity)["collection_id"] == plan.collection_id


@pytest.mark.parametrize(
    "change", ["status", "overspend", "closure", "source", "identity", "index_hash", "plan"]
)
def test_failed_or_mismatched_final_stage_report_is_not_an_optimizer_input(tmp_path, change):
    train, registry, plan = stage_fixture(tmp_path)
    result = execute_stage_actor(
        plan, registry, tmp_path / "output", protocol_fixture=OwnedProtocolPair()
    )
    index_path = Path(result["artifacts"]["rollout_index"]["path"])
    report_path = index_path.parent / "stage_report.json"
    report = read_document(report_path)
    if change == "status":
        report["status"] = "ERROR"
    elif change == "overspend":
        report["compute_millis"] = plan.maximum_compute_millis + 1
    elif change == "closure":
        report["all_owned_sessions_closed"] = False
    elif change == "source":
        report["source_postflight"]["problems"] = ["DRIFT"]
    elif change == "identity":
        report["identity"]["seed"] += 1
    elif change == "index_hash":
        report["artifacts"]["rollout_index"]["sha256"] = "0" * 64
    else:
        report["plan_sha256"] = "0" * 64
    write_document(report_path, report)
    with pytest.raises(ValueError):
        validate_rollout_admission(index_path, train.identity)


def test_missing_final_report_keeps_valid_raw_shards_ineligible(tmp_path):
    train, registry, plan = stage_fixture(tmp_path)
    result = execute_stage_actor(
        plan, registry, tmp_path / "output", protocol_fixture=OwnedProtocolPair()
    )
    path = Path(result["artifacts"]["rollout_index"]["path"])
    (path.parent / "stage_report.json").rename(path.parent / "old_report.json")
    with pytest.raises(FileNotFoundError):
        validate_rollout_admission(path, train.identity)


def test_formal_bare_index_is_rejected_and_aggregate_preserves_full_receipt_coverage(tmp_path):
    train, registry, plan = stage_fixture(tmp_path)
    result = execute_stage_actor(
        plan, registry, tmp_path / "output", protocol_fixture=OwnedProtocolPair()
    )
    path = Path(result["artifacts"]["rollout_index"]["path"])
    index = read_document(path)
    bare = {
        k: v
        for k, v in index.items()
        if k not in ("collection_id", "collection_plan_sha256", "stage_report_path")
    }
    formal = replace(train.identity, purpose="FORMAL")
    bare["identity_sha256"] = formal.sha256
    write_document(tmp_path / "bare.json", bare, immutable=True)
    with pytest.raises(ValueError, match="lacks successful"):
        validate_rollout_admission(tmp_path / "bare.json", formal)
    merged = {
        k: index[k]
        for k in (
            "schema",
            "identity_sha256",
            "shards",
            "physical_environment_loops",
            "update_groups",
        )
    }
    merged["collection_indices"] = [dict(path=str(path), sha256=file_hash(path))]
    write_document(tmp_path / "merged.json", merged, immutable=True)
    assert (
        validate_rollout_admission(tmp_path / "merged.json", train.identity)["shards"]
        == index["shards"]
    )
    merged["physical_environment_loops"] += 1
    write_document(tmp_path / "merged.json", merged)
    with pytest.raises(ValueError, match="coverage"):
        validate_rollout_admission(tmp_path / "merged.json", train.identity)


@pytest.mark.parametrize(
    "status", ["COMPLETE", "ERROR", "OVER_BUDGET", "DISPATCH_OVER_BUDGET", "DISPATCH_SIGNAL"]
)
def test_owned_worker_receipt_controls_admission(tmp_path, status):
    (tmp_path / "stage").mkdir()
    train, registry, plan = stage_fixture(tmp_path / "stage")
    result = execute_stage_actor(
        plan, registry, tmp_path / "collected", protocol_fixture=OwnedProtocolPair()
    )
    stage_index = read_document(result["artifacts"]["rollout_index"]["path"])
    root = tmp_path / "worker"
    path = root / "actor/rollout_index.json"
    index = {
        k: stage_index[k]
        for k in (
            "schema",
            "identity_sha256",
            "shards",
            "physical_environment_loops",
            "update_groups",
        )
    }
    index["job_id"] = "a" * 64
    write_document(path, index, immutable=True)
    request = dict(
        kind="actor",
        identity=plan.identity,
        maximum_compute_millis=100,
        payload=dict(job=dict(job_id=index["job_id"], provenance="learner")),
    )
    write_document(root / "request.json", request, immutable=True)
    receipt = dict(
        request_sha256=digest(request),
        kind="actor",
        status="ERROR" if status == "ERROR" else "COMPLETE",
        compute_millis=101 if status == "OVER_BUDGET" else 10,
        owned_sessions_closed=True,
        artifacts=dict(rollout_index=dict(path=str(path.resolve()), sha256=file_hash(path))),
    )
    write_document(root / "worker_result.json", receipt, immutable=True)
    if status.startswith("DISPATCH_"):
        write_document(
            root / "worker_exit.json",
            dict(
                returncode=0,
                signalled_once=status == "DISPATCH_SIGNAL",
                all_owned_sessions_closed=True,
                owner=dict(request_sha256=digest(request)),
                compute_millis=101 if status == "DISPATCH_OVER_BUDGET" else 10,
            ),
            immutable=True,
        )
    if status == "COMPLETE":
        assert validate_rollout_admission(path, train.identity)["job_id"] == index["job_id"]
    else:
        with pytest.raises(ValueError, match="worker|dispatcher"):
            validate_rollout_admission(path, train.identity)


def test_ppo_each_epoch_reports_its_actual_microbatch_metrics():
    train = learner(epochs=3, learning_rate=0.01, max_kl=100.0, maximum_batch_sequences=2)
    batches = [sequence(train, "a"), sequence(train, "b")]
    records = []
    original = train._batch

    def capture(*args, **kwargs):
        out = original(*args, **kwargs)
        if torch.is_grad_enabled():
            records.append(copy.deepcopy(out[1]))
        return out

    train._batch = capture
    report = train.update(batches)
    assert len(records) == 6 and report["optimizer_steps"] == 3
    for epoch, metrics in enumerate(report["epochs"]):
        rows = records[2 * epoch : 2 * epoch + 2]
        count = sum(r["samples"] for r in rows)
        for key in ("policy_loss", "value_loss", "entropy"):
            assert metrics[key] == pytest.approx(sum(r[key] for r in rows))
        assert metrics["max_kl"] == max(r["max_kl"] for r in rows)
        assert metrics["mean_kl"] == pytest.approx(
            sum(r["mean_kl"] * r["samples"] for r in rows) / count
        )
    assert report["epochs"][1]["value_loss"] != report["epochs"][0]["value_loss"]
    assert report["epochs"][1]["max_kl"] > 0
