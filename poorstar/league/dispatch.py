"""League: dispatch."""

import json
import math
import os
import platform
import signal
import socket
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

from poorstar.league.contracts import WorkerRequest, WorkerResult
from poorstar.league.ownership import (
    authorize_worker,
    process_starttime,
    read_document,
    verify_worker_lease,
    write_document,
)
from poorstar.training.checkpoint import file_hash
from poorstar.training.contracts import TrainingIdentity, digest


def process_table():
    if platform.system() == "Linux":
        rows = []
        for path in Path("/proc").iterdir():
            if not path.name.isdigit():
                continue
            try:
                fields = (path / "stat").read_text().rsplit(")", 1)[1].split()
                rows.append(
                    dict(
                        pid=int(path.name), ppid=int(fields[1]), sid=int(fields[3]), state=fields[0]
                    )
                )
            except (FileNotFoundError, ProcessLookupError):
                pass
        return rows
    raw = subprocess.run(
        ["ps", "-axo", "pid=,ppid=,stat="], text=True, capture_output=True, check=True
    ).stdout
    rows = []
    for line in raw.splitlines():
        a, b, state = line.split()[:3]
        try:
            rows.append(dict(pid=int(a), ppid=int(b), sid=os.getsid(int(a)), state=state))
        except ProcessLookupError:
            pass
    return rows


def session_members(sid):
    return [p for p in process_table() if p["sid"] == sid and not p["state"].startswith("Z")]


class SubprocessDispatcher:
    """No shell, no unowned signal, one registered worker and bounded drain."""

    def __init__(
        self, actor_worker, *, actor_worker_sha256, interpreter=sys.executable, drain_seconds=60.0
    ):
        self.actor_worker = Path(actor_worker)
        self.actor_worker_sha256 = actor_worker_sha256
        self.interpreter = interpreter
        self.drain_seconds = drain_seconds

    def __call__(self, request):
        authorize_worker(request)
        if request.kind == "actor":
            if file_hash(self.actor_worker) != self.actor_worker_sha256:
                raise ValueError("qualified actor worker source drift")
            command = [
                self.interpreter,
                str(self.actor_worker),
                "--request-json",
                str(Path(request.output_dir) / "request.json"),
                "--result-json",
                str(Path(request.output_dir) / "worker_result.json"),
            ]
        else:
            command = [
                self.interpreter,
                "-m",
                "poorstar.league.dispatch",
                "--learner-request",
                str(Path(request.output_dir) / "request.json"),
                "--result",
                str(Path(request.output_dir) / "worker_result.json"),
            ]
        directory = Path(request.output_dir)
        write_document(
            directory / "worker_launch_intent.json",
            dict(request_sha256=request.sha256, command=command),
            immutable=True,
        )
        sessions = set()
        started = time.monotonic()
        signalled = False
        with (directory / "worker.log").open("x") as log:
            worker = subprocess.Popen(
                command,
                cwd=str(Path(__file__).resolve().parents[2]),
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            birth = process_starttime(worker.pid)
            if birth is None:
                # Reap our actual child before reporting the absent /proc row.
                # No signal is permitted without its observed birth. A zero
                # exit additionally needs the qualified worker's closure proof.
                worker.wait(timeout=1.0)
            owner = dict(
                host=socket.gethostname(),
                pid=worker.pid,
                starttime=birth,
                sid=worker.pid,
                cwd=str(Path(__file__).resolve().parents[2]),
                command=command,
                request_sha256=request.sha256,
            )
            write_document(directory / "worker_launch.json", owner, immutable=True)
            sessions.add(worker.pid)
            deadline = started + request.maximum_compute_millis / 1000
            while worker.poll() is None:
                table = process_table()
                descendants = {worker.pid}
                while True:
                    extended = descendants | {p["pid"] for p in table if p["ppid"] in descendants}
                    if extended == descendants:
                        break
                    descendants = extended
                sessions.update(p["sid"] for p in table if p["pid"] in descendants)
                write_document(
                    directory / "worker_sessions.json",
                    dict(owner=owner, tracked_sessions=sorted(sessions)),
                )
                if time.monotonic() >= deadline and not signalled:
                    if process_starttime(worker.pid) != birth:
                        raise ValueError("worker PID changed; no signal authorized")
                    if platform.system() == "Linux":
                        actual = [
                            p.decode()
                            for p in (Path("/proc") / str(worker.pid) / "cmdline")
                            .read_bytes()
                            .split(b"\0")
                            if p
                        ]
                        if (
                            actual != command
                            or os.getsid(worker.pid) != worker.pid
                            or str((Path("/proc") / str(worker.pid) / "cwd").resolve())
                            != owner["cwd"]
                        ):
                            raise ValueError(
                                "worker exact command/session drift; no signal authorized"
                            )
                    worker.send_signal(signal.SIGINT)
                    signalled = True
                    deadline = time.monotonic() + self.drain_seconds
                    write_document(
                        directory / "worker_drain.json",
                        dict(owner=owner, signal="SIGINT", reason="compute_cap"),
                        immutable=True,
                    )
                elif signalled and time.monotonic() >= deadline:
                    raise ValueError(
                        "owned worker failed to drain; manual ownership attention required"
                    )
                time.sleep(0.02)
            returncode = worker.wait()
        if any(session_members(sid) for sid in sessions):
            raise ValueError("worker owned descendant sessions remain alive")
        if not (directory / "worker_result.json").exists():
            # The qualified actor also accounts for SC2 sessions that can be
            # created and reparented between coordinator observations. An
            # absent receipt cannot turn our partial process scan into a
            # complete engine-closure certificate.
            raise ValueError("worker reaped but qualified descendant closure receipt is unresolved")
        result = WorkerResult(**read_document(directory / "worker_result.json"))
        result.validate(request)
        verify_worker_lease(request)
        elapsed = math.ceil((time.monotonic() - started) * 1000)
        write_document(
            directory / "worker_exit.json",
            dict(
                owner=owner,
                returncode=returncode,
                tracked_sessions=sorted(sessions),
                all_owned_sessions_closed=True,
                signalled_once=signalled,
                compute_millis=elapsed,
            ),
            immutable=True,
        )
        if returncode or signalled:
            return WorkerResult(
                request.sha256,
                request.operation_id,
                request.attempt_id,
                request.claim_token,
                request.kind,
                "ERROR",
                max(result.compute_millis, elapsed),
                result.environment_loops,
                True,
                result.artifacts,
            )
        return WorkerResult(
            **{**asdict(result), "compute_millis": max(result.compute_millis, elapsed)}
        )


def verify_closed_attempt(request):
    """Recover only a closed qualified worker and the union of its sessions."""
    directory = Path(request.output_dir)
    if not (directory / "worker_launch.json").exists():
        raise ValueError("unresolved worker launch window; owned closure proof required")
    owner = read_document(directory / "worker_launch.json")
    if (
        owner.get("host") != socket.gethostname()
        or owner.get("request_sha256") != request.sha256
        or type(owner.get("pid")) is not int
        or owner["pid"] <= 0
        or owner.get("sid") != owner["pid"]
    ):
        raise ValueError("crash recovery worker owner/request/session drift")
    if owner.get("starttime") is not None and process_starttime(owner["pid"]) == owner["starttime"]:
        raise ValueError("crash recovery worker still alive")
    sessions = {owner["sid"]}
    for filename in ("worker_sessions.json", "worker_exit.json"):
        path = directory / filename
        if path.exists():
            evidence = read_document(path)
            if evidence.get("owner") != owner or any(
                type(s) is not int or s <= 0 for s in evidence.get("tracked_sessions", ())
            ):
                raise ValueError("crash recovery tracked session owner drift")
            sessions.update(evidence["tracked_sessions"])
    closed_path = directory / "worker_exit.json"
    receipt = directory / "worker_result.json"
    if closed_path.exists():
        closed = read_document(closed_path)
        if (
            closed.get("all_owned_sessions_closed") is not True
            or type(closed.get("compute_millis")) is not int
            or closed["compute_millis"] < 0
        ):
            raise ValueError("crash recovery worker closure certificate drift")
    else:
        closed = None
    if not receipt.exists():
        raise ValueError("crash recovery qualified descendant closure receipt is unresolved")
    result = WorkerResult(**read_document(receipt))
    result.validate(request)
    if any(session_members(sid) for sid in sessions):
        raise ValueError("crash recovery owned session descendants still alive")
    return max(result.compute_millis, 0 if closed is None else closed["compute_millis"])


def execute_learner_request(request):
    """Continue exact optimizer/sampler state; never initialize from an actor anew."""
    authorize_worker(request)
    if request.kind != "learner" or request.payload["provenance"] != "learner":
        raise ValueError("learner adapter rejects evaluator requests")
    execution = request.execution
    directory = Path(request.output_dir)
    previous = request.payload.get("previous_training") or execution.get("bootstrap", {}).get(
        request.identity["arm"] + ":" + str(request.identity["seed"])
    )
    if not previous or not all(
        k in previous for k in ("checkpoint_manifest", "accumulated_rollout_index")
    ):
        raise ValueError(
            "league continuation requires frozen exact optimizer checkpoint and consumed rollout prefix"
        )
    for artifact in (
        previous["checkpoint_manifest"],
        previous["accumulated_rollout_index"],
        *request.payload["rollout_indices"],
    ):
        if file_hash(artifact["path"]) != artifact["sha256"]:
            raise ValueError("learner continuation input drift")
    from poorstar.training.admission import validate_rollout_admission

    training_identity = TrainingIdentity(**request.identity)
    old = validate_rollout_admission(
        previous["accumulated_rollout_index"]["path"], training_identity
    )
    if old["identity_sha256"] != TrainingIdentity(**request.identity).sha256:
        raise ValueError("bootstrap consumed rollout prefix belongs to a different member")
    bootstrap_manifest = read_document(previous["checkpoint_manifest"]["path"])
    if (
        bootstrap_manifest["identity_sha256"] != old["identity_sha256"]
        or file_hash(Path(previous["checkpoint_manifest"]["path"]).parent / "state.pt")
        != bootstrap_manifest["state_sha256"]
    ):
        raise ValueError("bootstrap exact optimizer checkpoint identity/state drift")
    from poorstar.training.export import load_actor

    snapshot = request.payload["previous_snapshot"]
    _, snapshot_identity, snapshot_actor = load_actor(
        snapshot["weights_path"], sha256=snapshot["weights_sha256"], device="cpu"
    )
    if (
        snapshot_identity.sha256 != old["identity_sha256"]
        or snapshot_actor["policy_version"] != snapshot["policy_version"]
        or snapshot_actor["parameter_sha256"] != bootstrap_manifest["parameter_sha256"]
    ):
        raise ValueError("bootstrap checkpoint is not the exact frozen rollout actor")
    import torch

    checkpoint_state = torch.load(
        Path(previous["checkpoint_manifest"]["path"]).parent / "state.pt",
        map_location="cpu",
        weights_only=False,
    )
    if checkpoint_state["learner"]["policy_version"] != snapshot["policy_version"]:
        raise ValueError("bootstrap checkpoint policy version differs from rollout actor")
    run_plan = read_document(Path(request.owner["root"]) / "run_plan.json")
    training_plan = json.loads(Path(execution["training_plan"]).read_text())
    if digest(training_plan) != run_plan["budget"]["learner_contract_sha256"]:
        raise ValueError("matched numeric learner contract drift")
    shards = list(old["shards"])
    groups = list(old["update_groups"])
    loops = old["physical_environment_loops"]
    new_hashes = []
    collection_indices = list(
        old.get("collection_indices", [previous["accumulated_rollout_index"]] if shards else [])
    )
    existing = {s["sha256"] for s in shards}
    for artifact in request.payload["rollout_indices"]:
        index = validate_rollout_admission(artifact["path"], training_identity)
        collection_indices.append(artifact)
        if index["identity_sha256"] != old["identity_sha256"]:
            raise ValueError("new league rollout identity drift")
        if existing & {s["sha256"] for s in index["shards"]}:
            raise ValueError("already-consumed league trajectory cannot be reused")
        from poorstar.training.storage import RolloutReader

        for shard in index["shards"]:
            reader = RolloutReader(
                shard["path"], sha256=shard["sha256"], identity=TrainingIdentity(**request.identity)
            )
            try:
                sequence = reader.sequence()
                sequence.validate(
                    TrainingIdentity(**request.identity),
                    snapshot["policy_version"],
                    0,
                    learning=True,
                )
                if any(
                    step.checkpoint_sha256 != snapshot_actor["parameter_sha256"]
                    for step in sequence.steps
                    if step.valid
                ):
                    raise ValueError(
                        "new league trajectory belongs to a different frozen behavior actor"
                    )
            finally:
                reader.close()
        existing.update(s["sha256"] for s in index["shards"])
        shards.extend(index["shards"])
        new_hashes.extend(s["sha256"] for s in index["shards"])
        loops += index["physical_environment_loops"]
    if (
        len(new_hashes) > training_plan.get("maximum_batch_sequences", 1)
        or request.payload["requested_updates"] != 1
    ):
        raise ValueError(
            "all same-version league rollouts require one bounded logical update group"
        )
    groups.append(new_hashes)
    merged = dict(
        schema="poorstar_v3_rollout_index_v1",
        identity_sha256=old["identity_sha256"],
        shards=shards,
        update_groups=groups,
        physical_environment_loops=loops,
        collection_indices=collection_indices,
    )
    write_document(directory / "accumulated_rollout_index.json", merged, immutable=True)
    identity_path = directory / "worker_identity.json"
    identity_path.write_text(json.dumps(request.identity, sort_keys=True) + "\n")
    cli = Path(execution["training_cli"])
    if file_hash(cli) != execution["training_cli_sha256"]:
        raise ValueError("qualified learner CLI drift")
    output = directory / "learner"
    command = [
        sys.executable,
        str(cli),
        "--stage",
        execution["stage"],
        "--identity",
        str(identity_path),
        "--model-config",
        execution["model_config"],
        "--plan",
        execution["training_plan"],
        "--source-manifest",
        execution["source_manifest"],
        "--output-dir",
        str(output),
        "--resume-from",
        str(Path(previous["checkpoint_manifest"]["path"]).parent),
        "--rollout-index",
        str(directory / "accumulated_rollout_index.json"),
        "--device",
        execution["device"],
        "--checkpoint-every",
        "1",
    ]
    for source, flag in (
        ("training_preflight", "--preflight"),
        ("stage_transition", "--stage-transition"),
    ):
        if execution.get(source):
            command.extend((flag, execution[source]))
    started = time.monotonic()
    with (directory / "learner_cli.log").open("x") as log:
        completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
    verify_worker_lease(request)
    elapsed = math.ceil((time.monotonic() - started) * 1000)
    if completed.returncode or not (output / "report.json").exists():
        return WorkerResult(
            request.sha256,
            request.operation_id,
            request.attempt_id,
            request.claim_token,
            "learner",
            "ERROR",
            elapsed,
            0,
            True,
            {},
        )
    report = read_document(output / "report.json")
    actor = torch.load(report["actor"]["path"], map_location="cpu", weights_only=False)
    before = json.loads(Path(previous["checkpoint_manifest"]["path"]).read_text())["updates"]
    updates = report["updates"] - before
    if (
        updates != request.payload["requested_updates"]
        or actor["policy_version"] != request.payload["expected_next_policy_version"]
    ):
        raise ValueError("learner consumed update count/version outside frozen league request")
    model, actor_identity, verified_actor = load_actor(
        report["actor"]["path"], sha256=report["actor"]["sha256"], device="cpu"
    )
    if (
        actor_identity.sha256 != old["identity_sha256"]
        or verified_actor["policy_version"] != request.payload["expected_next_policy_version"]
    ):
        raise ValueError("learner exported actor exact identity/version drift")
    from poorstar.training.learner import groups_for

    groups = groups_for(model)
    phase = verified_actor["training_phase"]
    allowed = (
        {"value"}
        if phase == "VALUE_ONLY"
        else {"manager", "value"}
        if phase == "MANAGER_RL"
        else set(groups)
        if phase == "JOINT_RL"
        else None
    )
    if allowed is None:
        raise ValueError("league requires a declared outcome-RL phase")
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for name in allowed for _, p in groups[name])
    if (
        total != request.payload["parameter_budget"]["parameter_count"]
        or trainable != request.payload["parameter_budget"]["trainable_parameter_count"]
    ):
        raise ValueError("actual learner parameter allocation differs from frozen matched budget")
    candidates = read_document(report["candidate_index"]["path"])
    last = candidates["candidates"][-1]
    if file_hash(last["path"]) != last["sha256"]:
        raise ValueError("learner candidate publication drift")
    candidate = read_document(last["path"])
    checkpoint = Path(candidate["checkpoint_path"])
    # The adapter report states the operation delta; original CLI evidence and
    # cumulative counters remain immutable and are referenced by hash.
    normalized = dict(
        identity_sha256=report["identity_sha256"],
        updates=updates,
        policy_version=actor["policy_version"],
        parameter_count=total,
        trainable_parameter_count=trainable,
        cli_report=dict(path=str(output / "report.json"), sha256=file_hash(output / "report.json")),
    )
    write_document(directory / "training_report.json", normalized, immutable=True)
    artifacts = dict(
        training_report=dict(
            path=str(directory / "training_report.json"),
            sha256=file_hash(directory / "training_report.json"),
        ),
        actor_export=report["actor"],
        accumulated_rollout_index=dict(
            path=str(directory / "accumulated_rollout_index.json"),
            sha256=file_hash(directory / "accumulated_rollout_index.json"),
        ),
        checkpoint_manifest=dict(
            path=str(checkpoint / "manifest.json"), sha256=file_hash(checkpoint / "manifest.json")
        ),
    )
    return WorkerResult(
        request.sha256,
        request.operation_id,
        request.attempt_id,
        request.claim_token,
        "learner",
        "COMPLETE",
        elapsed,
        0,
        True,
        artifacts,
    )


def main():
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--learner-request", type=Path, required=True)
    p.add_argument("--result", type=Path, required=True)
    args = p.parse_args()
    request = WorkerRequest(**read_document(args.learner_request))
    result = execute_learner_request(request)
    write_document(args.result, asdict(result), immutable=True)


if __name__ == "__main__":
    main()
