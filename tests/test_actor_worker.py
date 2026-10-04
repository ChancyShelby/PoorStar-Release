"""Tests: test actor worker."""

from dataclasses import asdict, replace
from pathlib import Path

import pytest

from poorstar.league.contracts import WorkerRequest
from poorstar.league.ownership import read_document
from poorstar.training.checkpoint import file_hash
from poorstar.training.contracts import digest
from poorstar.training.export import export_actor
from poorstar.training.job_worker import actor_parameter_counts, execute_actor_request
from tests.test_training_system_actor import ProtocolPair
from tests.test_training_system_foundation import system


@pytest.fixture(autouse=True)
def threads():
    import torch

    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


class OwnedProtocolFixture(ProtocolPair):
    def __init__(self, **_):
        super().__init__(terminal_loop=3)
        self.owned_process_results = []

    def __enter__(self):
        return self

    def close(self):
        self.owned_process_results = [{"pid": -1, "exit_code": 0, "fixture": True}]

    def __exit__(self, *_):
        self.close()


def request_fixture(tmp_path, monkeypatch, provenance="learner"):
    learner, _, _ = system("flat")
    manifest = tmp_path / "source.json"
    manifest.write_text("{}")
    learner.identity = replace(learner.identity, source_sha256=file_hash(manifest))
    artifact = export_actor(tmp_path / "actor.pt", learner)
    names = ("learner", "opponent")
    snapshots = {}
    for name in names:
        snapshots[name] = dict(
            snapshot_id=digest(name),
            weights_path=artifact["path"],
            weights_sha256=artifact["sha256"],
            identity=asdict(learner.identity),
            policy_version=learner.policy_version,
            intent_profile=None,
        )
    job = dict(
        job_id=digest(["job", provenance]),
        learner=digest("learner"),
        opponent=digest("opponent"),
        provenance=provenance,
        map_name="protocol-fixture",
        seed=71,
        maximum_loops=5,
    )
    count = actor_parameter_counts(learner.model, "JOINT_RL")
    payload = dict(
        job=job,
        provenance=provenance,
        snapshots=snapshots,
        member_name="flat",
        parameter_budget=dict(
            **count,
            maximum_parameters=count["parameter_count"],
            maximum_trainable_parameters=count["trainable_parameter_count"],
        ),
    )
    execution = dict(
        source_manifest=str(manifest),
        sc2_path=str(tmp_path),
        version="fixture",
        device="cpu",
        unroll_steps=2,
        burn_in_steps=1,
        maximum_resident_steps=8,
        maximum_sources=63,
        learner_phase="JOINT_RL",
        measurement_period=2,
    )
    output = tmp_path / "attempt"
    output.mkdir()
    request = WorkerRequest(
        "actor",
        "a" * 64,
        asdict(learner.identity),
        "b" * 64,
        "c" * 64,
        "d" * 64,
        {"host": "protocol-fixture", "pid": 1, "starttime": "fixture"},
        str(output),
        10000,
        payload,
        execution,
    )
    from poorstar.common import integrity as audit
    from poorstar.training import actor, job_worker

    monkeypatch.setattr(actor, "SC2PairBackend", OwnedProtocolFixture)
    monkeypatch.setattr(job_worker, "authorize_worker", lambda r: learner.identity)
    monkeypatch.setattr(job_worker, "verify_worker_lease", lambda r: {})
    monkeypatch.setattr(audit, "source_integrity", lambda *_: {"problems": []})
    return request


@pytest.mark.parametrize("provenance", ["learner", "evaluator"])
def test_owned_request_actual_collector_and_complete_shard_readback(
    tmp_path, monkeypatch, provenance
):
    request = request_fixture(tmp_path, monkeypatch, provenance)
    result = execute_actor_request(request)
    result.validate(request)
    assert (
        result.status == "COMPLETE"
        and result.environment_loops == 3
        and result.owned_sessions_closed
    )
    report = read_document(result.artifacts["report"]["path"])
    assert (
        report["job_id"] == request.payload["job"]["job_id"] and report["provenance"] == provenance
    )
    assert not report["capability_evaluated"]
    assert ("rollout_index" in result.artifacts) == (provenance == "learner")
    if provenance == "learner":
        index = read_document(result.artifacts["rollout_index"]["path"])
        assert index["physical_environment_loops"] == 3 and len(index["shards"]) == 2
    else:
        assert not (Path(request.output_dir) / "actor" / "rollouts").exists()
        measured = read_document(result.artifacts["evaluator_metrics"]["path"])
        assert [r["loop"] for r in measured["observations"]] == [0, 2, 3]
        assert (
            measured["provenance"] == "evaluator" and not measured["engine_qualification_inferred"]
        )


def test_worker_rejects_greedy_training_before_process_creation(tmp_path, monkeypatch):
    request = request_fixture(tmp_path, monkeypatch)
    request = replace(request, execution=dict(request.execution, deterministic=True))
    with pytest.raises(ValueError, match="greedy"):
        execute_actor_request(request)
    assert not (Path(request.output_dir) / "actor").exists()


def test_worker_phase_drift_preserves_error_without_engine_or_rollouts(tmp_path, monkeypatch):
    request = request_fixture(tmp_path, monkeypatch)
    request = replace(request, execution=dict(request.execution, learner_phase="MANAGER_RL"))
    result = execute_actor_request(request)
    result.validate(request)
    assert (
        result.status == "ERROR"
        and result.environment_loops is None
        and result.owned_sessions_closed
    )
    failure = read_document(result.artifacts["failure"]["path"])
    assert "phase differs" in failure["error"] and "rollout_index" not in result.artifacts
