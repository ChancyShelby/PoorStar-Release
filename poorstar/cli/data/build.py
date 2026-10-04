#!/usr/bin/env python
"""Build BC data concurrently from a verified replay extraction plan."""

import argparse
import fcntl
import gzip
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
import uuid
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import contextmanager
from pathlib import Path

from poorstar.common.paths import source_root

TREE = source_root()


def unique_json_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("duplicate JSON object key: " + key)
        value[key] = item
    return value


def read_verified(path):
    from poorstar.common.integrity import payload_hash

    value = json.loads(Path(path).read_text(), object_pairs_hook=unique_json_object)
    digest = value.pop("content_digest")
    if payload_hash(value) != digest:
        raise ValueError("report identity drift: " + str(path))
    return value, digest


def freeze_jobs(plan_dir):
    from poorstar.common.integrity import file_hash, payload_hash
    from poorstar.data.bc.records import validate_split

    plan_dir = Path(plan_dir)
    plan, digest = read_verified(plan_dir / "report.json")
    if plan["schema"] != "poorstar_c3_full_corpus_extraction_plan_v1":
        raise ValueError("full scope plan required")
    if file_hash(plan_dir / "jobs.jsonl.gz") != plan["jobs_sha256"]:
        raise ValueError("job list drift")
    split = json.loads((plan_dir / "split.json").read_text())
    validate_split(split)
    if split["sha256"] != plan["split_sha256"]:
        raise ValueError("plan split drift")
    with gzip.open(plan_dir / "jobs.jsonl.gz", "rt") as f:
        jobs = [json.loads(line) for line in f]
    seen = set()
    for job in jobs:
        key = (job["replay_sha256"], job["player_id"])
        if key in seen or job["job_id"] != payload_hash(list(key)):
            raise ValueError("duplicate/drifted job identity")
        seen.add(key)
        if job["split"] != split["assignments"][key[0]]:
            raise ValueError("job split drift")
    if len(jobs) != plan["counts"]["unique_byte_identical_views"]:
        raise ValueError("full view denominator drift")
    if plan.get("scope") is not None:
        from poorstar.data.corpus.subset import validate_subset_jobs

        validate_subset_jobs(plan, jobs, split)
        if file_hash(plan_dir / "split.json") != plan["scope"]["parent_split_file_sha256"]:
            raise ValueError("inherited split file bytes drift")
    return plan, digest, jobs


def completed_job(
    attempt, job, *, split_sha256, source_sha256, maximum_delay, shard_decisions, solid_codec="xz"
):
    """A cached success is usable only after checking every output byte and binding."""
    from poorstar.common.integrity import file_hash

    attempt = Path(attempt)
    report, digest = read_verified(attempt / "report.json")
    marker, _ = read_verified(attempt / "job_completion.json")
    if marker.pop("solid_codec", "xz") != solid_codec:
        raise ValueError("completed job codec drift")
    expected = dict(
        job_id=job["job_id"],
        split_sha256=split_sha256,
        source_manifest_sha256=source_sha256,
        maximum_delay=maximum_delay,
        shard_decisions=shard_decisions,
        report_digest=digest,
    )
    if marker != expected:
        raise ValueError("completed job configuration drift")
    if (
        report["replay_sha256"] != job["replay_sha256"]
        or report["controlled_player_id"] != job["player_id"]
        or report["split_sha256"] != split_sha256
        or report["source_manifest_sha256"] != source_sha256
        or report["engine_base_build"] != 55958
        or not report["full_game"]
        or not report["lossless_solid_compression"]
        or report["terminal_source"] != "INDEPENDENT_FULL_ENGINE_PASS"
        or report["optimizer_steps"] != 0
        or report.get("solid_codec", "xz") != solid_codec
    ):
        raise ValueError("completed job provenance drift")
    rows = 0
    for shard in report["shards"]:
        path = Path(shard["path"])
        if path.parent.name != job["split"] or not path.resolve().is_relative_to(attempt.resolve()):
            raise ValueError("shard placement drift")
        if file_hash(path) != shard["sha256"] or path.stat().st_size != shard["bytes"]:
            raise ValueError("completed shard drift")
        rows += shard["decisions"]
    if rows < 1 or rows != report["counts"]["all_decisions"]:
        raise ValueError("completed row denominator drift")
    if file_hash(attempt / "extraction_terminal.protocol") != report["terminal_observation_sha256"]:
        raise ValueError("terminal evidence drift")
    boundary = report.get("terminal_native_boundary")
    if boundary is not None:
        if (
            boundary["rule"] != "FIRST_ENGINE_RESULT_NONTERMINAL_BODY_INTERVAL"
            or boundary["terminal_loop"] != report["terminal_loop"]
            or boundary["original_command_events"]
            != boundary["playable_command_events"]
            + boundary["commands_at_terminal"]
            + boundary["commands_after_terminal"]
            or boundary["artifact"] != "terminal_native_commands.jsonl.gz"
        ):
            raise ValueError("terminal command boundary drift")
        if file_hash(attempt / boundary["artifact"]) != boundary["sha256"]:
            raise ValueError("terminal command audit drift")
    return report


@contextmanager
def corpus_lock(root):
    # Shared disk instances must not duplicate jobs. OS releases this advisory
    # lock after a crash; no deletion of an old run is needed for recovery.
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    with (root / "coordinator.lock").open("a") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def write_report(path, value):
    from poorstar.common.integrity import payload_hash

    # Hash the exact JSON value that readers will receive. In particular,
    # integer ability keys become strings before canonical key sorting. Refuse
    # collisions (e.g. 2 and '2') instead of silently discarding statistics.
    value = json.loads(
        json.dumps(dict(value), allow_nan=False), object_pairs_hook=unique_json_object
    )
    value["content_digest"] = payload_hash(value)
    temporary = Path(str(path) + ".partial")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temporary, path)


def begin_dispatch_run(root, evidence, identity, lineage, sources, registered_views, jobs):
    """Invalidate old completion before work; preserve a fresh run registration.

    The mutable current-run reference and completion reference are control
    records. Data identity and per-run evidence are never copied or relabelled.
    Caller must hold the corpus writer lock throughout the run.
    """
    from poorstar.common.integrity import file_hash

    root, evidence = Path(root).resolve(), Path(evidence).resolve()
    run_id = uuid.uuid4().hex
    registration_path = evidence / "dispatch_registration.json"
    if not lineage:
        registration_path = root / "dispatch_runs" / run_id / "registration.json"
        registration_path.parent.mkdir(parents=True, exist_ok=False)
    if registration_path.exists():
        raise ValueError("fresh dispatch registration required")
    _, identity_digest = read_verified(root / "identity.json")
    _, assignment_digest = read_verified(root / "dispatch_assignment.json")
    lineage_digest = None
    lineage_sha = None
    if lineage:
        saved, lineage_digest = read_verified(evidence / "identity.json")
        if saved != lineage:
            raise ValueError("coordinator run identity drift")
        lineage_sha = file_hash(evidence / "identity.json")
    registration = dict(
        schema="poorstar_c3_dispatch_run_registration_v1",
        run_id=run_id,
        registration_path=str(registration_path),
        data_output_dir=str(root),
        data_binding=identity,
        data_identity_digest=identity_digest,
        data_identity_file_sha256=file_hash(root / "identity.json"),
        dispatch_assignment_digest=assignment_digest,
        dispatch_assignment_file_sha256=file_hash(root / "dispatch_assignment.json"),
        registered_views=registered_views,
        requested_job_ids=[j["job_id"] for j in jobs],
        run_evidence_dir=str(evidence),
        report_path=str(evidence / "report.json"),
        run_identity_digest=lineage_digest,
        run_identity_file_sha256=lineage_sha,
        coordinator_binding=lineage,
        source_roles=sources,
    )
    write_report(registration_path, registration)
    write_report(root / "dispatch_run.json", registration)
    _, registration_digest = read_verified(root / "dispatch_run.json")
    pointer = dict(
        schema="poorstar_c3_dispatch_completion_reference_v1",
        status="RUNNING",
        run_id=run_id,
        registration_digest=registration_digest,
        current_registration_file_sha256=file_hash(root / "dispatch_run.json"),
        report=None,
        source_postflight=None,
    )
    write_report(root / "dispatch_completion.json", pointer)
    return registration


def finish_dispatch_run(root, registration, result, postflight):
    """Publish completion only after final report and both source postflights."""
    from poorstar.common.integrity import file_hash

    root = Path(root).resolve()
    current, registration_digest = read_verified(root / "dispatch_run.json")
    if current != registration:
        raise ValueError("current dispatch registration drift")
    report_path = Path(registration["report_path"])
    report, report_digest = read_verified(report_path)
    if report != result or report.get("dispatch_registration_digest") != registration_digest:
        raise ValueError("dispatch report/registration drift")
    if postflight != {
        role: registration["source_roles"][role] for role in ("coordinator", "extractor")
    }:
        raise ValueError("dispatch source postflight drift")
    write_report(
        root / "dispatch_completion.json",
        dict(
            schema="poorstar_c3_dispatch_completion_reference_v1",
            status="COMPLETE" if result["complete_assigned_engine_extraction"] else "INCOMPLETE",
            run_id=registration["run_id"],
            registration_digest=registration_digest,
            current_registration_file_sha256=file_hash(root / "dispatch_run.json"),
            report=dict(
                path=str(report_path),
                content_digest=report_digest,
                file_sha256=file_hash(report_path),
            ),
            source_postflight=postflight,
        ),
    )


def verified_sources(manifest, extractor_root=None, extractor_manifest=None):
    """Bind dispatch code and data-producing code independently, without relabelling data."""
    from poorstar.common.integrity import file_hash, source_integrity

    if (extractor_root is None) != (extractor_manifest is None):
        raise ValueError("extractor code root and manifest must be supplied together")
    pre = source_integrity(TREE, Path(manifest))
    if pre["problems"]:
        raise ValueError("coordinator source preflight failed")
    root = TREE if extractor_root is None else Path(extractor_root).resolve()
    source = Path(manifest) if extractor_manifest is None else Path(extractor_manifest)
    if extractor_root is not None:
        script = (root / "poorstar/cli/data/extract.py").resolve()
        entries = json.loads(source.read_text()).get("files", [])
        if (
            not script.is_relative_to(root)
            or not script.is_file()
            or not any(e["path"] == "poorstar/cli/data/extract.py" for e in entries)
        ):
            raise ValueError("extractor script missing or outside bound source")
    extraction_pre = (
        pre
        if root == TREE and source.resolve() == Path(manifest).resolve()
        else source_integrity(root, source)
    )
    if extraction_pre["problems"]:
        raise ValueError("extractor source preflight failed")
    return (
        root,
        source,
        dict(
            coordinator=pre,
            extractor=extraction_pre,
            coordinator_source_manifest_sha256=file_hash(Path(manifest)),
            extractor_source_manifest_sha256=file_hash(source),
        ),
    )


@contextmanager
def graceful_stop():
    """SIGINT stops submission; worker sessions finish and commit their evidence."""
    stop = threading.Event()
    previous = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, lambda signum, frame: stop.set())
    try:
        yield stop
    finally:
        signal.signal(signal.SIGINT, previous)


def extraction_process(command, log, timeout, *, cwd=None):
    # An extractor timeout must also close its own SC2 descendants. A separate
    # process group avoids touching other users' or the second worker's engines.
    child = subprocess.Popen(
        command,
        cwd=TREE if cwd is None else cwd,
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    try:
        return child.wait(timeout=timeout)
    except BaseException:
        try:
            os.killpg(child.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            child.wait(timeout=15)
        except subprocess.TimeoutExpired:
            pass
        finally:
            # The parent may exit on TERM while an engine descendant remains.
            # Always close any remaining members of OUR process group.
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            child.wait()
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("plan-dir", "source-manifest", "sc2-path", "output-dir"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--workers", type=int, choices=(1, 2, 4), default=2)
    p.add_argument("--extractor-code-root", type=Path)
    p.add_argument("--extractor-source-manifest", type=Path)
    p.add_argument(
        "--run-evidence-dir",
        type=Path,
        help="fresh external coordinator lineage/output; corpus identity remains extractor-bound",
    )
    p.add_argument(
        "--assignment-manifest",
        type=Path,
        help="digest-verified fixed dispatch subset of the SAME full plan; never a smaller corpus gate",
    )
    p.add_argument(
        "--reuse-data-root",
        type=Path,
        action="append",
        default=[],
        help="drained original parent-plan corpus; full-row verified immutable references",
    )
    p.add_argument(
        "--import-only",
        action="store_true",
        help="aggregate complete drained same-scope owners; never launch pending engines",
    )
    p.add_argument("--reserve-gib", type=int, default=32)
    p.add_argument("--maximum-delay", type=int, default=32)
    p.add_argument("--shard-decisions", type=int, default=1024)
    p.add_argument("--job-timeout-seconds", type=int, default=7200)
    p.add_argument("--solid-codec", choices=("xz", "zstd"), default="xz")
    p.add_argument(
        "--retry-failed",
        action="store_true",
        help="append identical infrastructure retry attempts; never overwrite",
    )
    a = p.parse_args()
    a.output_dir = a.output_dir.resolve()
    if a.run_evidence_dir is not None:
        a.run_evidence_dir = a.run_evidence_dir.resolve()
    if a.output_dir.resolve().is_relative_to(TREE):
        p.error("external corpus output required")
    if min(a.reserve_gib, a.maximum_delay, a.shard_decisions, a.job_timeout_seconds) < 1:
        p.error("positive resource bounds required")
    from poorstar.common.integrity import file_hash, payload_hash, source_integrity

    extractor_root, extractor_manifest, sources = verified_sources(
        a.source_manifest, a.extractor_code_root, a.extractor_source_manifest
    )
    if a.output_dir.resolve().is_relative_to(extractor_root):
        p.error("external corpus output required for extractor")
    if a.extractor_code_root is not None and a.run_evidence_dir is None:
        p.error("separate extractor requires fresh coordinator evidence")
    if a.run_evidence_dir is not None and (
        a.run_evidence_dir.resolve().is_relative_to(TREE)
        or a.run_evidence_dir.resolve().is_relative_to(extractor_root)
        or a.run_evidence_dir.resolve() == a.output_dir.resolve()
    ):
        p.error("fresh external coordinator evidence required")
    if a.reuse_data_root and a.run_evidence_dir is None:
        p.error("reuse requires fresh coordinator evidence")
    if a.import_only and (not a.reuse_data_root or a.assignment_manifest is not None):
        p.error("import-only requires complete-scope reuse and no assignment")
    if any(
        a.output_dir.resolve() == r.resolve()
        or a.output_dir.resolve().is_relative_to(r.resolve())
        or r.resolve().is_relative_to(a.output_dir.resolve())
        for r in a.reuse_data_root
    ):
        p.error("reuse source and destination corpus roots must be disjoint")
    source_sha = sources["extractor_source_manifest_sha256"]
    plan, plan_digest, jobs = freeze_jobs(a.plan_dir)
    all_jobs = jobs
    identity = dict(
        schema="poorstar_c3_full_corpus_coordinator_v1",
        plan_digest=plan_digest,
        source_manifest_sha256=source_sha,
        split_sha256=plan["split_sha256"],
        maximum_delay=a.maximum_delay,
        shard_decisions=a.shard_decisions,
        solid_compression=True,
        solid_codec=a.solid_codec,
    )
    from poorstar.data.corpus.assignment import completion_scope, select_assignment

    registered_views = len(jobs)
    assignment_binding = None
    if a.assignment_manifest is not None:
        if a.run_evidence_dir is None:
            p.error("fixed assignment requires fresh coordinator evidence")
        assignment, assignment_digest = read_verified(a.assignment_manifest)
        jobs = select_assignment(assignment, identity, jobs)
        assignment_binding = dict(manifest=assignment, content_digest=assignment_digest)
    start = time.monotonic()
    success = {}
    failed = []
    pending = []
    with corpus_lock(a.output_dir):
        # A root cannot silently switch between full dispatch and an assignment,
        # or between two assignments. This is a control binding, not a relabel of
        # the immutable extractor/data identity or any job-completion marker.
        assignment_path = a.output_dir / "dispatch_assignment.json"
        if assignment_path.exists():
            prior, _ = read_verified(assignment_path)
            if prior != {"assignment": assignment_binding}:
                raise ValueError("resume dispatch assignment drift")
        elif assignment_binding is not None and (a.output_dir / "identity.json").exists():
            raise ValueError("cannot narrow an existing unassigned corpus root")
        else:
            write_report(assignment_path, dict(assignment=assignment_binding))
        lineage = {}
        evidence = a.output_dir if a.run_evidence_dir is None else a.run_evidence_dir
        if a.run_evidence_dir is not None:
            evidence.mkdir(parents=True, exist_ok=False)
            lineage = dict(
                schema="poorstar_c3_coordinator_source_roles_v1",
                sources=sources,
                coordinator_code_root=str(TREE),
                extractor_code_root=str(extractor_root),
                extractor_manifest=str(extractor_manifest),
                data_output_dir=str(a.output_dir),
                data_binding=identity,
                workers=a.workers,
                reserve_gib=a.reserve_gib,
                job_timeout_seconds=a.job_timeout_seconds,
                optimizer_steps=0,
                formal_bc_started=False,
                rl_started=False,
            )
            lineage["dispatch_assignment"] = assignment_binding
            write_report(evidence / "identity.json", lineage)
        identity_path = a.output_dir / "identity.json"
        if identity_path.exists():
            old, _ = read_verified(identity_path)
            if old != identity:
                raise ValueError("resume source/data/configuration drift")
        else:
            write_report(identity_path, identity)
        registration = begin_dispatch_run(
            a.output_dir, evidence, identity, lineage, sources, registered_views, jobs
        )
        import_binding = None
        if a.reuse_data_root:
            from poorstar.data.corpus.subset import import_completed_jobs

            split = json.loads((a.plan_dir / "split.json").read_text())
            success, import_binding = import_completed_jobs(
                plan,
                plan_digest,
                all_jobs,
                split,
                identity,
                a.reuse_data_root,
                a.output_dir / "read_only_import.json",
                requested_job_ids=[job["job_id"] for job in jobs],
                require_source_complete=a.import_only,
            )
            write_report(evidence / "read_only_import.json", import_binding)
        elif (a.output_dir / "read_only_import.json").exists():
            raise ValueError("resume requires the original explicit reuse sources")
        for job in jobs:
            root = a.output_dir / "jobs" / job["job_id"]
            attempts = sorted(
                {
                    x if x.is_dir() else Path(str(x)[: -len(".record.json")])
                    for x in root.glob("attempt_*")
                    if x.is_dir() or x.name.endswith(".record.json")
                }
            )
            if job["job_id"] in success:
                if attempts:
                    raise ValueError("imported job has duplicate destination ownership")
                continue
            if attempts and (attempts[-1] / "job_completion.json").exists():
                success[job["job_id"]] = completed_job(
                    attempts[-1],
                    job,
                    split_sha256=plan["split_sha256"],
                    source_sha256=source_sha,
                    maximum_delay=a.maximum_delay,
                    shard_decisions=a.shard_decisions,
                    solid_codec=a.solid_codec,
                )
            elif attempts and not a.retry_failed:
                failed.append(
                    dict(
                        job_id=job["job_id"],
                        reason="RETRY_FLAG_REQUIRED",
                        attempt=str(attempts[-1]),
                    )
                )
            else:
                pending.append((job, root / ("attempt_" + str(len(attempts)).zfill(5))))
        if a.import_only and pending:
            failed.append(dict(reason="IMPORT_ONLY_SCOPE_INCOMPLETE", pending_views=len(pending)))

        def execute(pair):
            job, attempt = pair
            replay = Path(job["replay"])
            attempt.parent.mkdir(parents=True, exist_ok=True)
            write_report(
                Path(str(attempt) + ".record.json"),
                dict(job_id=job["job_id"], status="STARTED", binding=identity),
            )
            if (
                replay.stat().st_size != job["replay_bytes"]
                or file_hash(replay) != job["replay_sha256"]
            ):
                raise ValueError("replay source drift")
            if shutil.disk_usage(a.output_dir).free < a.reserve_gib * 1024**3:
                raise OSError("disk reserve reached; retain every incomplete attempt")
            command = [
                sys.executable,
                str(extractor_root / "poorstar/cli/data/extract.py"),
                "--replay",
                str(replay),
                "--player-id",
                str(job["player_id"]),
                "--sc2-path",
                str(a.sc2_path),
                "--split-manifest",
                str(a.plan_dir / "split.json"),
                "--source-manifest-sha256",
                source_sha,
                "--output-dir",
                str(attempt),
                "--maximum-delay",
                str(a.maximum_delay),
                "--shard-decisions",
                str(a.shard_decisions),
                "--solid-compression",
                "--solid-codec",
                a.solid_codec,
            ]
            # Log sits beside the extractor's fresh directory, including failures
            # before it can create that directory. Never retry based on model results.
            with Path(str(attempt) + ".log").open("x") as log:
                if a.extractor_code_root is None:
                    returncode = extraction_process(command, log, a.job_timeout_seconds)
                else:
                    returncode = extraction_process(
                        command, log, a.job_timeout_seconds, cwd=extractor_root
                    )
            if returncode:
                raise RuntimeError("extractor exit " + str(returncode))
            _, digest = read_verified(attempt / "report.json")
            write_report(
                attempt / "job_completion.json",
                dict(
                    job_id=job["job_id"],
                    split_sha256=plan["split_sha256"],
                    source_manifest_sha256=source_sha,
                    maximum_delay=a.maximum_delay,
                    shard_decisions=a.shard_decisions,
                    solid_codec=a.solid_codec,
                    report_digest=digest,
                ),
            )
            return completed_job(
                attempt,
                job,
                split_sha256=plan["split_sha256"],
                source_sha256=source_sha,
                maximum_delay=a.maximum_delay,
                shard_decisions=a.shard_decisions,
                solid_codec=a.solid_codec,
            )

        # A failed job stops NEW launches; already active jobs finish and are
        # retained. Restart is explicit and preserves the exact full scope.
        with graceful_stop() as stop, ThreadPoolExecutor(max_workers=a.workers) as pool:
            running = {}
            cursor = 0
            while running or (not failed and not stop.is_set() and cursor < len(pending)):
                while (
                    not failed
                    and not stop.is_set()
                    and len(running) < a.workers
                    and cursor < len(pending)
                ):
                    pair = pending[cursor]
                    cursor += 1
                    running[pool.submit(execute, pair)] = pair
                if not running:
                    break
                done, _ = wait(running, return_when=FIRST_COMPLETED)
                for future in done:
                    job, attempt = running.pop(future)
                    try:
                        success[job["job_id"]] = future.result()
                        write_report(
                            Path(str(attempt) + ".record.json"),
                            dict(job_id=job["job_id"], status="COMPLETE", binding=identity),
                        )
                    except BaseException as e:
                        error = dict(
                            job_id=job["job_id"],
                            attempt=str(attempt),
                            error=f"{type(e).__name__}: {e}",
                        )
                        failed.append(error)
                        write_report(
                            Path(str(attempt) + ".record.json"),
                            dict(error, status="FAILED", binding=identity),
                        )
                progress = dict(
                    identity,
                    requested_views=registered_views,
                    requested_assigned_views=len(jobs),
                    completed_views=len(success),
                    failures=failed,
                    elapsed_seconds=time.monotonic() - start,
                    workers=a.workers,
                    reserve_gib=a.reserve_gib,
                    stop_requested=stop.is_set(),
                    in_flight=len(running),
                    optimizer_steps=0,
                    formal_bc_ready=False,
                    rl_ready=False,
                )
                if lineage:
                    progress["coordinator_binding"] = lineage
                if import_binding:
                    progress["read_only_import_digest"] = payload_hash(import_binding)
                write_report(evidence / "progress.json", progress)
                print(
                    json.dumps(
                        dict(
                            completed_views=len(success),
                            requested_views=len(jobs),
                            failures=len(failed),
                        )
                    ),
                    flush=True,
                )
        postflight = dict(
            coordinator=source_integrity(TREE, a.source_manifest),
            extractor=source_integrity(extractor_root, extractor_manifest),
        )
        if postflight["coordinator"]["problems"]:
            raise ValueError("source postflight failed")
        if postflight["extractor"]["problems"]:
            raise ValueError("extractor source postflight failed")
        if postflight != {role: sources[role] for role in ("coordinator", "extractor")}:
            raise ValueError("source manifest/qualification changed during dispatch")
        reports = [success[j["job_id"]] for j in jobs if j["job_id"] in success]
        counts = {}
        shards = []
        for report in reports:
            shards.extend(report["shards"])
            for k, v in report["counts"].items():
                counts[k] = counts.get(k, 0) + v
        result = dict(
            identity,
            purpose="FORMAL_CORPUS",
            **completion_scope(registered_views, len(jobs), len(reports), failed, stop.is_set()),
            shards=shards,
            counts=counts,
            failures=failed,
            full_game=True,
            engine_base_build=55958,
            all_job_terminal_passes_independent=True,
            elapsed_seconds=time.monotonic() - start,
            workers=a.workers,
            reserve_gib=a.reserve_gib,
            stop_requested=stop.is_set(),
            paper_corpus_gate_passed=False,
            qualification="FULL_DOMAIN_AND_ACTOR_EVALUATOR_PREFLIGHT_STILL_REQUIRED",
            optimizer_steps=0,
            formal_bc_ready=False,
            rl_ready=False,
        )
        if lineage:
            result["coordinator_binding"] = lineage
        result["dispatch_registration_digest"] = payload_hash(registration)
        if import_binding:
            result["read_only_import_digest"] = payload_hash(import_binding)
            result["read_only_imported_views"] = import_binding["imported_views"]
        write_report(evidence / "report.json", result)
        finish_dispatch_run(a.output_dir, registration, result, postflight)
        return 0 if result["complete_assigned_engine_extraction"] else (130 if stop.is_set() else 1)


if __name__ == "__main__":
    raise SystemExit(main())
