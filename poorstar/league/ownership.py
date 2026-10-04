"""League: ownership."""

import fcntl
import json
import os
import platform
import socket
import sqlite3
import subprocess
import uuid
from dataclasses import asdict
from pathlib import Path

from poorstar.training.checkpoint import file_hash
from poorstar.training.contracts import digest


def process_starttime(pid):
    if platform.system() == "Linux":
        try:
            return (Path("/proc") / str(pid) / "stat").read_text().rsplit(")", 1)[1].split()[19]
        except (FileNotFoundError, ProcessLookupError):
            return None
    result = subprocess.run(
        ["ps", "-p", str(pid), "-o", "lstart="], text=True, capture_output=True, check=False
    )
    return result.stdout.strip() or None


def write_document(path, value, *, immutable=False):
    value = dict(value)
    value["content_digest"] = digest(value)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if immutable:
        with path.open("x") as f:
            json.dump(value, f, sort_keys=True, indent=2, allow_nan=False)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
    else:
        temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
        with temporary.open("x") as f:
            json.dump(value, f, sort_keys=True, indent=2, allow_nan=False)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, path)
    return value


def read_document(path):
    value = json.loads(Path(path).read_text())
    check = dict(value)
    expected = check.pop("content_digest", None)
    if digest(check) != expected:
        raise ValueError("immutable league document drift: " + str(path))
    return check


def local_filesystem(path):
    """Resolve the longest Linux mount prefix rather than caller assertions."""
    path = Path(path).resolve()
    if platform.system() == "Linux":
        mounts = []
        for line in Path("/proc/self/mountinfo").read_text().splitlines():
            left, right = line.split(" - ", 1)
            mount = Path(left.split()[4].replace("\\040", " "))
            kind = right.split()[0]
            if path == mount or mount in path.parents:
                mounts.append((len(mount.parts), kind))
        kind = max(mounts)[1] if mounts else None
    elif platform.system() == "Darwin":
        result = subprocess.run(
            ["stat", "-f", "%T", str(path)], text=True, capture_output=True, check=True
        )
        kind = result.stdout.strip()
    else:
        raise ValueError("local SQLite ownership unsupported on this platform")
    if kind is None or any(x in kind.lower() for x in ("nfs", "smb", "cifs", "sshfs", "9p")):
        raise ValueError("league database must be on verified host-local filesystem")
    return kind


class CoordinatorOwner:
    def __init__(self, root, plan_sha256):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.filesystem = local_filesystem(self.root)
        self.file = (self.root / "coordinator.lock").open("a+")
        try:
            fcntl.flock(self.file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self.file.close()
            raise ValueError("another local league coordinator owns this database")
        self.owner = dict(
            host=socket.gethostname(),
            pid=os.getpid(),
            starttime=process_starttime(os.getpid()),
            nonce=uuid.uuid4().hex,
            plan_sha256=plan_sha256,
            root=str(self.root.resolve()),
        )
        write_document(self.root / "owner.json", self.owner)

    def close(self):
        fcntl.flock(self.file, fcntl.LOCK_UN)
        self.file.close()


def verify_worker_lease(request):
    """Check the coordinator's current generation before/after real execution."""
    owner = request.owner
    root = Path(owner["root"])
    if (
        owner["host"] != socket.gethostname()
        or process_starttime(owner["pid"]) != owner["starttime"]
    ):
        raise ValueError("worker coordinator owner is absent/foreign/reused")
    if read_document(root / "owner.json") != owner:
        raise ValueError("worker coordinator owner generation drift")
    lease = read_document(root / "dispatch_lease.json")
    if lease != dict(
        request_sha256=request.sha256,
        operation_id=request.operation_id,
        attempt_id=request.attempt_id,
        claim_token=request.claim_token,
        owner=owner,
        request_path=str(Path(request.output_dir) / "request.json"),
    ):
        raise ValueError("worker dispatch lease request/token drift")
    if read_document(lease["request_path"]) != asdict(request):
        raise ValueError("worker immutable request file drift")
    db = sqlite3.connect((root / "league.sqlite").resolve().as_uri() + "?mode=ro", uri=True)
    try:
        attempt = db.execute(
            "SELECT request,status FROM attempts WHERE id=?", (request.attempt_id,)
        ).fetchone()
        if attempt is None or attempt[1] != "RUNNING" or json.loads(attempt[0]) != asdict(request):
            raise ValueError("worker exact registered attempt is not running")
        if request.kind == "actor":
            job = request.payload["job"]
            row = db.execute(
                "SELECT spec,status,owner FROM jobs WHERE id=?", (job["job_id"],)
            ).fetchone()
            if (
                row is None
                or json.loads(row[0]) != job
                or row[1:] != ("RUNNING", request.claim_token)
            ):
                raise ValueError("worker exact actor job is not exclusively claimed")
    finally:
        db.close()
    with (root / "coordinator.lock").open() as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            pass
        else:
            fcntl.flock(f, fcntl.LOCK_UN)
            raise ValueError("worker coordinator exclusive ownership was released")
    return lease


def worker_execution_digest(request):
    # Gate document references are excluded to avoid a circular self-hash.
    # All engine/device/temporal/learner options and the whole plan remain bound.
    config = {
        k: v
        for k, v in request.execution.items()
        if k not in ("selfplay_preflight", "league_preflight", "training_preflight")
    }
    return digest(
        dict(run_plan_sha256=request.run_plan_sha256, kind=request.kind, execution=config)
    )


def verify_stage_gates(identity, plan_sha256, kind, execution):
    from poorstar.training.gates import validate_launch

    config = {
        k: v
        for k, v in execution.items()
        if k not in ("selfplay_preflight", "league_preflight", "training_preflight")
    }
    configuration = digest(dict(run_plan_sha256=plan_sha256, kind=kind, execution=config))
    validate_launch(
        execution.get("selfplay_preflight"),
        identity,
        "SELFPLAY",
        execution_config_sha256=configuration,
    )
    validate_launch(
        execution.get("league_preflight"), identity, "LEAGUE", execution_config_sha256=configuration
    )
    return configuration


def authorize_worker(request):
    verify_worker_lease(request)
    from poorstar.training import TrainingIdentity

    identity = TrainingIdentity(**request.identity)
    plan = read_document(Path(request.owner["root"]) / "run_plan.json")
    if digest(plan) != request.run_plan_sha256:
        raise ValueError("worker frozen league plan drift")
    member = next(
        (m for m in plan["members"] if m["name"] == request.payload.get("member_name")), None
    )
    if member is None or member["identity"] != request.identity:
        raise ValueError("worker population member identity mismatch")
    expected = plan["actor_execution"] if request.kind == "actor" else plan["learner_execution"]
    if request.execution != expected:
        raise ValueError("worker execution differs from frozen run plan")
    maximum = plan["budget"][
        "maximum_actor_compute_millis"
        if request.kind == "actor"
        else "maximum_learner_compute_millis"
    ]
    if request.maximum_compute_millis != maximum:
        raise ValueError("worker compute reservation differs from frozen run plan")
    if request.kind == "actor":
        actor_config = {
            k: v
            for k, v in request.execution.items()
            if k not in ("selfplay_preflight", "league_preflight", "training_preflight")
        }
        if digest(actor_config) != plan["budget"]["actor_protocol_sha256"]:
            raise ValueError("worker matched actor protocol configuration drift")
    parameter_budget = request.payload.get("parameter_budget")
    if parameter_budget != dict(
        parameter_count=member["parameter_count"],
        trainable_parameter_count=member["trainable_parameter_count"],
        maximum_parameters=plan["budget"]["maximum_parameters"],
        maximum_trainable_parameters=plan["budget"]["maximum_trainable_parameters"],
    ):
        raise ValueError("worker parameter budget differs from frozen run plan")
    verify_stage_gates(identity, request.run_plan_sha256, request.kind, request.execution)
    return identity
