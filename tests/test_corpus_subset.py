"""Tests: test corpus subset."""

import gzip
import json
from collections import Counter
from dataclasses import replace
from pathlib import Path

import pytest
from s2clientprotocol import sc2api_pb2 as sc

from poorstar.actions.frame import ActionFrame, BoundQuery
from poorstar.cli.data import build as coordinator
from poorstar.common.integrity import file_hash, payload_hash
from poorstar.data.bc.records import BCDecision, BCShardWriter, freeze_split
from poorstar.data.corpus.assignment import SCHEMA as ASSIGNMENT_SCHEMA
from poorstar.data.corpus.subset import (
    derive_zvz_plan,
    import_completed_jobs,
    read_only_owner_lock,
    verify_completed_rows,
    write_zvz_plan,
)
from tests.test_coarse_bc import setup, target


def plan_fixture(path):
    path.mkdir()
    jobs = []
    groups = []
    for index, matchup in enumerate(("ZvZ", "ZvT", "ZvP")):
        digest = str(index + 1) * 64
        groups.append(dict(replay_sha256=digest, match_group="native" + str(index)))
        for player in (1, 2) if matchup == "ZvZ" else (1,):
            jobs.append(
                dict(
                    job_id=payload_hash([digest, player]),
                    replay_sha256=digest,
                    player_id=player,
                    replay=str(path / ("replay" + str(index))),
                    replay_bytes=10,
                    aliases=["alias" + str(index)],
                    native_content_group="native" + str(index),
                    matchup=matchup,
                )
            )
    split = freeze_split(groups, seed=4)
    (path / "split.json").write_text(json.dumps(split, indent=2) + "\n")
    for job in jobs:
        job["split"] = split["assignments"][job["replay_sha256"]]
    with gzip.open(path / "jobs.jsonl.gz", "wt") as stream:
        for job in jobs:
            stream.write(json.dumps(job) + "\n")
    plan = dict(
        schema="poorstar_c3_full_corpus_extraction_plan_v1",
        split_sha256=split["sha256"],
        jobs_sha256=file_hash(path / "jobs.jsonl.gz"),
        counts={"unique_byte_identical_views": len(jobs)},
        scope_digest="parent-scope",
        native_census_digest="parent-native",
    )
    coordinator.write_report(path / "report.json", plan)
    return plan, payload_hash(plan), jobs, split


def paired_fixture(tmp_path):
    parent, pd, jobs, split = plan_fixture(tmp_path / "parent")
    plan, digest, selected = write_zvz_plan(tmp_path / "parent", tmp_path / "selected")
    identity = dict(
        schema="poorstar_c3_full_corpus_coordinator_v1",
        plan_digest=digest,
        source_manifest_sha256="b" * 64,
        split_sha256=split["sha256"],
        maximum_delay=8,
        shard_decisions=1,
        solid_compression=True,
        solid_codec="xz",
    )
    root = tmp_path / "old"
    root.mkdir()
    (root / "coordinator.lock").touch()
    coordinator.write_report(root / "identity.json", dict(identity, plan_digest=pd))
    return plan, digest, selected, split, identity, root


def complete(root, job, split, identity, *, native_mismatch=False):
    """Two genuine compressed shards, one ACT and one WAIT, ending at loop4."""
    attempt = root / "jobs" / job["job_id"] / "attempt_00000"
    attempt.mkdir(parents=True)
    episode = job["replay_sha256"] + ":player" + str(job["player_id"])
    frames = []
    for loop in (0, 2, 3):
        frame, _, cat, info = setup(loop=loop, episode=episode)
        if job["player_id"] != 1:
            raw = sc.ResponseObservation()
            raw.observation.ParseFromString(frame._observation_wire)
            raw.observation.player_common.player_id = job["player_id"]
            from s2clientprotocol import query_pb2

            request = query_pb2.RequestQuery()
            reply = query_pb2.ResponseQuery()
            request.ParseFromString(frame._query_request_wire)
            reply.ParseFromString(frame._query_response_wire)
            frame = ActionFrame.from_protocol(
                raw,
                info,
                cat,
                BoundQuery.bind(raw.observation, request, reply),
                context=replace(frame._context, player_id=job["player_id"]),
                episode=episode,
            )
        frames.append(frame)
    native = {
        "_event": "NNet.Game.SCmdEvent",
        "_gameloop": 2,
        "m_data": {},
        "_userid": {"m_userId": job["player_id"] - 1},
    }
    rows = (
        BCDecision(
            job["replay_sha256"],
            frames[0],
            "ACT",
            2,
            frames[1],
            (target(frames[1]),),
            native_event_wires=(json.dumps(native).encode(),),
        ),
        BCDecision(job["replay_sha256"], frames[2], "WAIT", 1),
    )
    shards = []
    for row in rows:
        writer = BCShardWriter(
            attempt / job["split"],
            split_manifest=split,
            split=job["split"],
            catalog=cat,
            game_info=info,
            source_manifest_sha256=identity["source_manifest_sha256"],
            extractor_sha256="c" * 64,
            maximum_delay=8,
            solid_compression=True,
        )
        writer.append(row)
        shards.append(writer.close())
    terminal = sc.ResponseObservation()
    terminal.observation.game_loop = 4
    terminal.observation.player_common.player_id = job["player_id"]
    terminal.player_result.add(player_id=1, result=1)
    (attempt / "extraction_terminal.protocol").write_bytes(terminal.SerializeToString())
    with gzip.open(attempt / "terminal_native_commands.jsonl.gz", "wt") as stream:
        stream.write("")
    boundary = dict(
        rule="FIRST_ENGINE_RESULT_NONTERMINAL_BODY_INTERVAL",
        terminal_loop=4,
        original_command_events=1,
        playable_command_events=1,
        commands_at_terminal=0,
        commands_after_terminal=0,
        artifact="terminal_native_commands.jsonl.gz",
        sha256=file_hash(attempt / "terminal_native_commands.jsonl.gz"),
    )
    counts = {"all_decisions": 2, "ACT": 1, "WAIT": 1, "body_not_structurally_rejected": 2}
    report = dict(
        schema="poorstar_c3_direct_decision_extraction_v1",
        replay_sha256=job["replay_sha256"],
        controlled_player_id=job["player_id"],
        split_sha256=split["sha256"],
        source_manifest_sha256=identity["source_manifest_sha256"],
        extractor_sha256="c" * 64,
        engine_base_build=55958,
        full_game=True,
        lossless_solid_compression=True,
        terminal_source="INDEPENDENT_FULL_ENGINE_PASS",
        optimizer_steps=0,
        shards=shards,
        counts=counts,
        terminal_loop=4,
        terminal_observation_sha256=file_hash(attempt / "extraction_terminal.protocol"),
        terminal_native_boundary=boundary,
    )
    coordinator.write_report(attempt / "report.json", report)
    coordinator.write_report(
        attempt / "job_completion.json",
        dict(
            job_id=job["job_id"],
            split_sha256=split["sha256"],
            source_manifest_sha256=identity["source_manifest_sha256"],
            maximum_delay=8,
            shard_decisions=1,
            solid_codec="xz",
            report_digest=payload_hash(report),
        ),
    )
    return attempt, report


def import_fixture(tmp_path):
    plan, pd, jobs, split, identity, root = paired_fixture(tmp_path)
    attempt, report = complete(root, jobs[0], split, identity)
    return plan, pd, jobs, split, identity, root, attempt, report


def run_import(tmp_path, fixture, roots=None):
    plan, pd, jobs, split, identity, root, *_ = fixture
    return import_completed_jobs(
        plan, pd, jobs, split, identity, roots or [root], tmp_path / "ledger.json"
    )


def test_real_paired_selection_preserves_original_split_bytes_ids_order_and_lineage(tmp_path):
    plan, pd, jobs, split = plan_fixture(tmp_path / "parent")
    before = {
        n: file_hash(tmp_path / "parent" / n)
        for n in ("split.json", "report.json", "jobs.jsonl.gz")
    }
    selected, digest, chosen = write_zvz_plan(tmp_path / "parent", tmp_path / "selected")
    assert chosen == jobs[:2]
    assert (
        selected["scope"]["parent_plan_digest"] == pd
        and selected["scope"]["original_registered_views"] == 4
    )
    assert (
        selected["parent_scope_digest"] == "parent-scope"
        and selected["parent_native_census_digest"] == "parent-native"
    )
    assert selected["scope_digest"] != selected["parent_scope_digest"]
    assert (tmp_path / "parent/split.json").read_bytes() == (
        tmp_path / "selected/split.json"
    ).read_bytes()
    assert before == {n: file_hash(tmp_path / "parent" / n) for n in before}
    assert (
        selected["counts"]["unique_replay_files"] == 1
        and selected["counts"]["unique_byte_identical_views"] == 2
    )
    assert not selected["formal_bc_ready"] and not selected["rl_ready"]
    second, _, _ = write_zvz_plan(tmp_path / "parent", tmp_path / "again")
    assert second["jobs_sha256"] == selected["jobs_sha256"]
    with pytest.raises(FileExistsError):
        write_zvz_plan(tmp_path / "parent", tmp_path / "selected")


@pytest.mark.parametrize(
    "case",
    [
        "missing_pair",
        "job_id",
        "player",
        "split",
        "native",
        "mixed",
        "aliases",
        "denominator",
        "digest",
    ],
)
def test_derivation_rejects_incomplete_or_drifted_parent(tmp_path, case):
    plan, pd, jobs, split = plan_fixture(tmp_path / "parent")
    if case == "missing_pair":
        jobs.pop(1)
        plan["counts"]["unique_byte_identical_views"] = len(jobs)
    elif case == "job_id":
        jobs[0]["job_id"] = "bad"
    elif case == "player":
        jobs[0]["player_id"] = True
    elif case == "split":
        jobs[0]["split"] = "test" if jobs[0]["split"] != "test" else "train"
    elif case == "native":
        jobs[0]["native_content_group"] = ""
    elif case == "mixed":
        jobs[1]["matchup"] = "ZvP"
    elif case == "aliases":
        jobs[1]["aliases"] = ["other"]
    elif case == "denominator":
        plan["counts"]["unique_byte_identical_views"] += 1
    elif case == "digest":
        pd = "bad"
    if case != "digest":
        pd = payload_hash(plan)
    with pytest.raises(ValueError):
        derive_zvz_plan(plan, pd, jobs, split)


def test_subset_freeze_rejects_rehashed_matchup_or_scope_forgery(tmp_path):
    (*_,) = paired_fixture(tmp_path)
    plan, digest, jobs = coordinator.freeze_jobs(tmp_path / "selected")
    plan["scope_digest"] = "bad"
    coordinator.write_report(tmp_path / "selected/report.json", plan)
    with pytest.raises(ValueError, match="scope digest"):
        coordinator.freeze_jobs(tmp_path / "selected")


def test_import_full_rows_census_immutable_paths_and_exact_resume(tmp_path):
    fixture = import_fixture(tmp_path)
    plan, pd, jobs, split, identity, root, attempt, original = fixture
    before = {str(p): file_hash(p) for p in root.rglob("*") if p.is_file()}
    reports, ledger = run_import(tmp_path, fixture)
    assert list(reports) == [jobs[0]["job_id"]]
    assert reports[jobs[0]["job_id"]] == original
    assert ledger["imported_views"] == 1 and ledger["imported_rows"] == 2
    assert ledger["imports"][0]["attempt"] == str(attempt)
    q = ledger["imports"][0]["qualification"]
    assert (
        q["rows"] == 2
        and q["playable_native_events"] == 1
        and q["supervision_counts"]["retained_commands"] == 1
    )
    assert not ledger["formal_bc_ready"] and ledger["optimizer_steps"] == 0
    assert before == {str(p): file_hash(p) for p in root.rglob("*") if p.is_file()}
    ledger_hash = file_hash(tmp_path / "ledger.json")
    assert run_import(tmp_path, fixture)[1] == ledger
    assert ledger_hash == file_hash(tmp_path / "ledger.json")


def test_old_owner_must_be_drained_and_lock_file_unchanged(tmp_path):
    fixture = import_fixture(tmp_path)
    root = fixture[5]
    before = file_hash(root / "coordinator.lock")
    with coordinator.corpus_lock(root):
        with pytest.raises(BlockingIOError):
            run_import(tmp_path, fixture)
    assert (
        not (tmp_path / "ledger.json").exists() and file_hash(root / "coordinator.lock") == before
    )
    run_import(tmp_path, fixture)


@pytest.mark.parametrize(
    "field",
    [
        "source_manifest_sha256",
        "split_sha256",
        "maximum_delay",
        "shard_decisions",
        "solid_codec",
        "plan_digest",
    ],
)
def test_source_owner_configuration_cannot_be_relabelled(tmp_path, field):
    fixture = import_fixture(tmp_path)
    root = fixture[5]
    old, _ = coordinator.read_verified(root / "identity.json")
    old[field] = "drift"
    coordinator.write_report(root / "identity.json", old)
    with pytest.raises(ValueError, match="source owner"):
        run_import(tmp_path, fixture)
    assert not (tmp_path / "ledger.json").exists()


@pytest.mark.parametrize(
    "case", ["shard", "terminal", "native_tail", "counts", "player", "marker", "duplicate_shard"]
)
def test_completed_artifact_corruption_never_imports_even_rehashed_reports(tmp_path, case):
    fixture = import_fixture(tmp_path)
    attempt = fixture[6]
    report = fixture[7]
    if case == "shard":
        Path(report["shards"][0]["path"]).write_bytes(b"corrupt")
    elif case == "terminal":
        terminal = sc.ResponseObservation()
        terminal.observation.game_loop = 4
        (attempt / "extraction_terminal.protocol").write_bytes(terminal.SerializeToString())
        report["terminal_observation_sha256"] = file_hash(attempt / "extraction_terminal.protocol")
    elif case == "native_tail":
        report["terminal_native_boundary"]["playable_command_events"] = 0
        report["terminal_native_boundary"]["original_command_events"] = 0
    elif case == "counts":
        report["counts"]["ACT"] = 2
    elif case == "player":
        report["controlled_player_id"] = 2
    elif case == "marker":
        marker, _ = coordinator.read_verified(attempt / "job_completion.json")
        marker["maximum_delay"] = 9
        coordinator.write_report(attempt / "job_completion.json", marker)
    elif case == "duplicate_shard":
        report["shards"].append(report["shards"][0])
        report["counts"]["all_decisions"] = 3
    if case not in ("shard", "marker"):
        coordinator.write_report(attempt / "report.json", report)
        marker, _ = coordinator.read_verified(attempt / "job_completion.json")
        marker["report_digest"] = payload_hash(report)
        coordinator.write_report(attempt / "job_completion.json", marker)
    with pytest.raises(ValueError):
        run_import(tmp_path, fixture)
    assert not (tmp_path / "ledger.json").exists()


def test_latest_partial_is_retained_and_never_counted_as_success(tmp_path):
    fixture = import_fixture(tmp_path)
    attempt = fixture[6]
    (attempt.parent / "attempt_00001.record.json").write_text('{"status":"FAILED"}')
    reports, ledger = run_import(tmp_path, fixture)
    assert not reports and ledger["imported_views"] == 0
    assert ledger["incomplete"][0]["attempt"].endswith("attempt_00001")
    assert attempt.exists()


def test_duplicate_ownership_and_resume_changed_inputs_refuse(tmp_path):
    fixture = import_fixture(tmp_path)
    plan, pd, jobs, split, identity, root, attempt, _ = fixture
    other = tmp_path / "second"
    other.mkdir()
    (other / "coordinator.lock").touch()
    coordinator.write_report(
        other / "identity.json", dict(identity, plan_digest=plan["scope"]["parent_plan_digest"])
    )
    complete(other, jobs[0], split, identity)
    with pytest.raises(ValueError, match="duplicate completed job ownership"):
        run_import(tmp_path, fixture, [root, other])
    run_import(tmp_path, fixture)
    with pytest.raises(ValueError, match="ledger binding drift"):
        run_import(tmp_path, fixture, [other])
    Path(fixture[7]["shards"][0]["path"]).write_bytes(b"drift after first import")
    with pytest.raises(ValueError, match="shard drift"):
        run_import(tmp_path, fixture)


def test_source_symlinks_and_ledger_inside_source_refused(tmp_path):
    fixture = import_fixture(tmp_path)
    root = fixture[5]
    (root / "coordinator.lock").unlink()
    (root / "coordinator.lock").symlink_to(tmp_path / "outside")
    with pytest.raises(ValueError, match="existing real"):
        run_import(tmp_path, fixture)
    plan, pd, jobs, split, identity, *_ = fixture
    with pytest.raises(ValueError, match="outside immutable"):
        import_completed_jobs(plan, pd, jobs, split, identity, [root], root / "ledger.json")


def assigned_owner(path, identity, jobs, selected, *, completed=True):
    path.mkdir()
    (path / "coordinator.lock").touch()
    coordinator.write_report(path / "identity.json", identity)
    manifest = dict(
        schema=ASSIGNMENT_SCHEMA,
        data_binding=identity,
        registered_views=len(jobs),
        job_ids=[job["job_id"] for job in selected],
    )
    control = dict(assignment=dict(manifest=manifest, content_digest=payload_hash(manifest)))
    coordinator.write_report(path / "dispatch_assignment.json", control)
    evidence = path.parent / (path.name + "_run")
    evidence.mkdir()
    sources = {
        role: dict(files_checked=1, manifest_sha256=sha, problems=[])
        for role, sha in (
            ("coordinator", "a" * 64),
            ("extractor", identity["source_manifest_sha256"]),
        )
    }
    sources.update(
        coordinator_source_manifest_sha256="a" * 64,
        extractor_source_manifest_sha256=identity["source_manifest_sha256"],
    )
    lineage = dict(
        schema="poorstar_c3_coordinator_source_roles_v1",
        data_output_dir=str(path),
        data_binding=identity,
        dispatch_assignment=control["assignment"],
        sources=sources,
    )
    coordinator.write_report(evidence / "identity.json", lineage)
    registration = coordinator.begin_dispatch_run(
        path, evidence, identity, lineage, sources, len(jobs), selected
    )
    report = dict(
        identity,
        registered_views=len(jobs),
        complete_assigned_engine_extraction=completed,
        complete_engine_extraction=completed and len(selected) == len(jobs),
        completed_views=len(selected),
        requested_assigned_views=len(selected),
        stop_requested=False,
        failures=[],
        coordinator_binding=lineage,
        dispatch_registration_digest=payload_hash(registration),
        optimizer_steps=0,
        formal_bc_ready=False,
        rl_ready=False,
    )
    coordinator.write_report(evidence / "report.json", report)
    coordinator.finish_dispatch_run(
        path, registration, report, {r: sources[r] for r in ("coordinator", "extractor")}
    )


def test_import_assignment_filters_old_root_without_widening_dispatch(tmp_path):
    fixture = import_fixture(tmp_path)
    plan, pd, jobs, split, identity, root, *_ = fixture
    complete(root, jobs[1], split, identity)
    reports, ledger = import_completed_jobs(
        plan,
        pd,
        jobs,
        split,
        identity,
        [root],
        tmp_path / "ledger.json",
        requested_job_ids=[jobs[1]["job_id"]],
    )
    assert set(reports) == {jobs[1]["job_id"]} and ledger["binding"]["registered_views"] == 2
    assert ledger["binding"]["requested_job_ids"] == [jobs[1]["job_id"]]
    assert ledger["imports"][0]["player_id"] == 2


def test_aggregate_disjoint_complete_assignments_and_follow_original_import_references(tmp_path):
    fixture = import_fixture(tmp_path)
    plan, pd, jobs, split, identity, old, *_ = fixture
    left = tmp_path / "left"
    right = tmp_path / "right"
    assigned_owner(left, identity, jobs, jobs[:1])
    assigned_owner(right, identity, jobs, jobs[1:])
    # Left owns its old immutable reference, right owns a newly completed view.
    import_completed_jobs(
        plan,
        pd,
        jobs,
        split,
        identity,
        [old],
        left / "read_only_import.json",
        requested_job_ids=[jobs[0]["job_id"]],
    )
    complete(right, jobs[1], split, identity)
    reports, ledger = import_completed_jobs(
        plan,
        pd,
        jobs,
        split,
        identity,
        [left, right],
        tmp_path / "aggregate.json",
        require_source_complete=True,
    )
    assert set(reports) == {job["job_id"] for job in jobs} and ledger["imported_views"] == 2
    assert ledger["imported_rows"] == 4
    first = next(entry for entry in ledger["imports"] if entry["job_id"] == jobs[0]["job_id"])
    assert first["source_root"] == str(old) and first["via_owner_root"] == str(left)
    assert first["attempt"].startswith(str(old))


@pytest.mark.parametrize(
    "case", ["incomplete", "overlap", "unassigned", "digest", "reference_hash"]
)
def test_assignment_aggregation_refuses_owner_gap_overlap_and_reference_tampering(tmp_path, case):
    fixture = import_fixture(tmp_path)
    plan, pd, jobs, split, identity, old, *_ = fixture
    left = tmp_path / "left"
    right = tmp_path / "right"
    assigned_owner(left, identity, jobs, jobs[:1], completed=case != "incomplete")
    assigned_owner(right, identity, jobs, jobs[:1] if case == "overlap" else jobs[1:])
    complete(left, jobs[0], split, identity)
    complete(right, jobs[0] if case == "overlap" else jobs[1], split, identity)
    if case == "unassigned":
        complete(left, jobs[1], split, identity)
    elif case == "digest":
        control, _ = coordinator.read_verified(left / "dispatch_assignment.json")
        control["assignment"]["content_digest"] = "bad"
        coordinator.write_report(left / "dispatch_assignment.json", control)
    elif case == "reference_hash":
        # Substitute an import reference in place of the local completion.
        import shutil

        shutil.rmtree(left / "jobs")
        import_completed_jobs(
            plan,
            pd,
            jobs,
            split,
            identity,
            [old],
            left / "read_only_import.json",
            requested_job_ids=[jobs[0]["job_id"]],
        )
        ledger, _ = coordinator.read_verified(left / "read_only_import.json")
        ledger["imports"][0]["original_report_file_sha256"] = "bad"
        coordinator.write_report(left / "read_only_import.json", ledger)
    with pytest.raises(ValueError):
        import_completed_jobs(
            plan,
            pd,
            jobs,
            split,
            identity,
            [left, right],
            tmp_path / "aggregate.json",
            require_source_complete=True,
        )
    assert not (tmp_path / "aggregate.json").exists()


@pytest.mark.parametrize("assigned", [False, True])
def test_coordinator_reuse_runs_zero_engines_preserves_partial_scope_and_resumes(
    tmp_path, monkeypatch, assigned
):
    from tests.test_coordinator_sources import source

    plan, pd, jobs, split, identity, old = paired_fixture(tmp_path)
    own_root = tmp_path / "code"
    own = source(own_root)
    foreign_root = tmp_path / "extractor"
    foreign = source(foreign_root)
    monkeypatch.setattr(coordinator, "TREE", own_root)
    identity["source_manifest_sha256"] = file_hash(foreign)
    coordinator.write_report(
        old / "identity.json", dict(identity, plan_digest=plan["scope"]["parent_plan_digest"])
    )
    for job in jobs:
        complete(old, job, split, identity)
    argv = [
        "build",
        "--plan-dir",
        str(tmp_path / "selected"),
        "--source-manifest",
        str(own),
        "--extractor-code-root",
        str(foreign_root),
        "--extractor-source-manifest",
        str(foreign),
        "--run-evidence-dir",
        str(tmp_path / "run1"),
        "--sc2-path",
        str(tmp_path / "engine"),
        "--output-dir",
        str(tmp_path / "data"),
        "--workers",
        "1",
        "--reserve-gib",
        "1",
        "--maximum-delay",
        "8",
        "--shard-decisions",
        "1",
        "--reuse-data-root",
        str(old),
    ]
    if assigned:
        manifest = dict(
            schema=ASSIGNMENT_SCHEMA,
            data_binding=identity,
            registered_views=2,
            job_ids=[jobs[0]["job_id"]],
        )
        coordinator.write_report(tmp_path / "assignment.json", manifest)
        argv.extend(["--assignment-manifest", str(tmp_path / "assignment.json")])
    monkeypatch.setattr("sys.argv", argv)

    def no_engine(*args, **kwargs):
        raise AssertionError("verified imports must not launch SC2")

    monkeypatch.setattr(coordinator, "extraction_process", no_engine)
    assert coordinator.main() == 0
    report, _ = coordinator.read_verified(tmp_path / "run1/report.json")
    assert report["registered_views"] == 2 and report["completed_views"] == (1 if assigned else 2)
    assert report["complete_engine_extraction"] is (not assigned)
    assert report["read_only_imported_views"] == report["completed_views"]
    assert not report["formal_bc_ready"] and not report["rl_ready"]
    assert not (tmp_path / "data/jobs").exists()
    completion, _ = coordinator.read_verified(tmp_path / "data/dispatch_completion.json")
    assert completion["status"] == "COMPLETE" and completion["report"]["path"] == str(
        tmp_path / "run1/report.json"
    )
    assert not (tmp_path / "data/report.json").exists()
    before = file_hash(tmp_path / "data/read_only_import.json")
    argv[argv.index("--run-evidence-dir") + 1] = str(tmp_path / "run2")
    assert coordinator.main() == 0 and file_hash(tmp_path / "data/read_only_import.json") == before
    resumed, _ = coordinator.read_verified(tmp_path / "data/dispatch_completion.json")
    assert resumed["run_id"] != completion["run_id"] and resumed["report"]["path"] == str(
        tmp_path / "run2/report.json"
    )
    assert file_hash(tmp_path / "run1/report.json") == completion["report"]["file_sha256"]


def test_import_only_incomplete_full_scope_never_launches_pending_engine(tmp_path, monkeypatch):
    from tests.test_coordinator_sources import source

    plan, pd, jobs, split, identity, old = paired_fixture(tmp_path)
    own_root = tmp_path / "code"
    own = source(own_root)
    foreign_root = tmp_path / "extractor"
    foreign = source(foreign_root)
    monkeypatch.setattr(coordinator, "TREE", own_root)
    identity["source_manifest_sha256"] = file_hash(foreign)
    part = tmp_path / "part"
    assigned_owner(part, identity, jobs, jobs[:1])
    complete(part, jobs[0], split, identity)
    argv = [
        "build",
        "--plan-dir",
        str(tmp_path / "selected"),
        "--source-manifest",
        str(own),
        "--extractor-code-root",
        str(foreign_root),
        "--extractor-source-manifest",
        str(foreign),
        "--run-evidence-dir",
        str(tmp_path / "run1"),
        "--sc2-path",
        str(tmp_path / "engine"),
        "--output-dir",
        str(tmp_path / "data"),
        "--workers",
        "1",
        "--reserve-gib",
        "1",
        "--maximum-delay",
        "8",
        "--shard-decisions",
        "1",
        "--reuse-data-root",
        str(part),
        "--import-only",
    ]
    monkeypatch.setattr("sys.argv", argv)

    def no_engine(*args, **kwargs):
        raise AssertionError("import-only never starts SC2")

    monkeypatch.setattr(coordinator, "extraction_process", no_engine)
    assert coordinator.main() == 1
    report, _ = coordinator.read_verified(tmp_path / "run1/report.json")
    assert not report["complete_engine_extraction"] and report["completed_views"] == 1
    assert report["failures"] == [dict(reason="IMPORT_ONLY_SCOPE_INCOMPLETE", pending_views=1)]


def test_actual_separate_evidence_partition_owners_aggregate_full_rows_without_engines_and_resume(
    tmp_path, monkeypatch
):
    from tests.test_coordinator_sources import source

    plan, pd, jobs, split, identity, old = paired_fixture(tmp_path)
    own_root = tmp_path / "code"
    own = source(own_root)
    foreign_root = tmp_path / "extractor"
    foreign = source(foreign_root)
    monkeypatch.setattr(coordinator, "TREE", own_root)
    identity["source_manifest_sha256"] = file_hash(foreign)
    coordinator.write_report(
        old / "identity.json", dict(identity, plan_digest=plan["scope"]["parent_plan_digest"])
    )
    for job in jobs:
        complete(old, job, split, identity)
    immutable_before = {str(p): file_hash(p) for p in old.rglob("*") if p.is_file()}
    base = [
        "build",
        "--plan-dir",
        str(tmp_path / "selected"),
        "--source-manifest",
        str(own),
        "--extractor-code-root",
        str(foreign_root),
        "--extractor-source-manifest",
        str(foreign),
        "--sc2-path",
        str(tmp_path / "engine"),
        "--workers",
        "1",
        "--reserve-gib",
        "1",
        "--maximum-delay",
        "8",
        "--shard-decisions",
        "1",
    ]

    def no_engine(*args, **kwargs):
        raise AssertionError("complete read-only import must never launch SC2")

    monkeypatch.setattr(coordinator, "extraction_process", no_engine)
    parts = []
    reports = []
    for index, job in enumerate(jobs):
        part = tmp_path / ("part" + str(index))
        parts.append(part)
        evidence = tmp_path / ("part_run" + str(index))
        reports.append(evidence / "report.json")
        manifest = dict(
            schema=ASSIGNMENT_SCHEMA,
            data_binding=identity,
            registered_views=2,
            job_ids=[job["job_id"]],
        )
        assignment = tmp_path / ("assignment" + str(index) + ".json")
        coordinator.write_report(assignment, manifest)
        argv = base + [
            "--output-dir",
            str(part),
            "--run-evidence-dir",
            str(evidence),
            "--assignment-manifest",
            str(assignment),
            "--reuse-data-root",
            str(old),
        ]
        monkeypatch.setattr("sys.argv", argv)
        assert coordinator.main() == 0
        result, _ = coordinator.read_verified(evidence / "report.json")
        assert (
            result["complete_assigned_engine_extraction"]
            and not result["complete_engine_extraction"]
        )
        assert result["completed_views"] == 1 and result["read_only_imported_views"] == 1
        assert not (part / "jobs").exists() and not (part / "report.json").exists()
    # A stale report at the data root is ignored; actual owner reports are external.
    coordinator.write_report(
        parts[0] / "report.json", dict(identity, completed_views=0, failures=["old failure"])
    )
    preserved = {str(p): file_hash(p) for p in reports}
    argv = base + [
        "--output-dir",
        str(tmp_path / "aggregate"),
        "--run-evidence-dir",
        str(tmp_path / "aggregate_run1"),
        "--reuse-data-root",
        str(parts[0]),
        "--reuse-data-root",
        str(parts[1]),
        "--import-only",
    ]
    monkeypatch.setattr("sys.argv", argv)
    assert coordinator.main() == 0
    result, _ = coordinator.read_verified(tmp_path / "aggregate_run1/report.json")
    ledger, _ = coordinator.read_verified(tmp_path / "aggregate/read_only_import.json")
    assert (
        result["registered_views"] == 2
        and result["completed_views"] == 2
        and result["complete_engine_extraction"]
    )
    assert result["counts"]["all_decisions"] == 4 and ledger["imported_rows"] == 4
    # Member coverage includes ACT/WAIT metadata and every bound frame protocol.
    assert all(
        e["qualification"]["rows"] == 2 and e["qualification"]["members"] == 15
        for e in ledger["imports"]
    )
    assert all(e["source_root"] == str(old) for e in ledger["imports"])
    assert {b["current_completion"]["report_path"] for b in ledger["source_bindings"]} == set(
        map(str, reports)
    )
    ledger_hash = file_hash(tmp_path / "aggregate/read_only_import.json")
    argv[argv.index("--run-evidence-dir") + 1] = str(tmp_path / "aggregate_run2")
    assert (
        coordinator.main() == 0
        and file_hash(tmp_path / "aggregate/read_only_import.json") == ledger_hash
    )
    assert immutable_before == {str(p): file_hash(p) for p in old.rglob("*") if p.is_file()}
    assert preserved == {str(p): file_hash(p) for p in reports}
    assert not (tmp_path / "aggregate/jobs").exists()


@pytest.mark.parametrize(
    "case",
    [
        "missing_pointer",
        "changed_report",
        "changed_location",
        "wrong_scope",
        "wrong_source_roles",
        "postflight_failed",
        "stale_generation",
        "running_restart",
        "changed_registration",
    ],
)
def test_completion_locator_fails_closed_without_using_stale_root_reports(tmp_path, case):
    fixture = import_fixture(tmp_path)
    plan, pd, jobs, split, identity, old, *_ = fixture
    left = tmp_path / "left"
    right = tmp_path / "right"
    assigned_owner(left, identity, jobs, jobs[:1])
    assigned_owner(right, identity, jobs, jobs[1:])
    complete(left, jobs[0], split, identity)
    complete(right, jobs[1], split, identity)
    pointer, _ = coordinator.read_verified(left / "dispatch_completion.json")
    registration, _ = coordinator.read_verified(left / "dispatch_run.json")
    external = Path(pointer["report"]["path"])
    # Even a byte-for-byte successful legacy root report cannot rescue the locator.
    (left / "report.json").write_bytes(external.read_bytes())
    if case == "missing_pointer":
        (left / "dispatch_completion.json").unlink()
    elif case == "changed_report":
        external.write_text(external.read_text() + " ")
    elif case == "changed_location":
        pointer["report"]["path"] = str(left / "report.json")
        coordinator.write_report(left / "dispatch_completion.json", pointer)
    elif case == "wrong_scope":
        report, digest = coordinator.read_verified(external)
        report["plan_digest"] = "wrong-scope"
        coordinator.write_report(external, report)
        pointer["report"].update(
            content_digest=payload_hash(report), file_sha256=file_hash(external)
        )
        coordinator.write_report(left / "dispatch_completion.json", pointer)
    elif case == "wrong_source_roles":
        report, _ = coordinator.read_verified(external)
        report["coordinator_binding"]["sources"]["extractor_source_manifest_sha256"] = (
            "wrong-source"
        )
        coordinator.write_report(external, report)
        pointer["report"].update(
            content_digest=payload_hash(report), file_sha256=file_hash(external)
        )
        coordinator.write_report(left / "dispatch_completion.json", pointer)
    elif case == "postflight_failed":
        pointer["source_postflight"]["extractor"]["problems"] = ["SOURCE_DRIFT"]
        coordinator.write_report(left / "dispatch_completion.json", pointer)
    elif case in ("stale_generation", "running_restart"):
        evidence = tmp_path / "new_run"
        evidence.mkdir()
        lineage = registration["coordinator_binding"]
        coordinator.write_report(evidence / "identity.json", lineage)
        coordinator.begin_dispatch_run(
            left, evidence, identity, lineage, registration["source_roles"], len(jobs), jobs[:1]
        )
        if case == "stale_generation":
            coordinator.write_report(left / "dispatch_completion.json", pointer)
    elif case == "changed_registration":
        registration["requested_job_ids"] = [jobs[1]["job_id"]]
        coordinator.write_report(left / "dispatch_run.json", registration)
        pointer["registration_digest"] = payload_hash(registration)
        pointer["current_registration_file_sha256"] = file_hash(left / "dispatch_run.json")
        coordinator.write_report(left / "dispatch_completion.json", pointer)
    with pytest.raises(ValueError):
        import_completed_jobs(
            plan,
            pd,
            jobs,
            split,
            identity,
            [left, right],
            tmp_path / "aggregate.json",
            require_source_complete=True,
        )
    assert not (tmp_path / "aggregate.json").exists() and (left / "report.json").exists()


def test_source_manifest_change_with_valid_files_never_publishes_complete_report(
    tmp_path, monkeypatch
):
    import poorstar.common.integrity as audit
    from tests.test_coordinator_sources import source

    plan, pd, jobs, split, identity, old = paired_fixture(tmp_path)
    own_root = tmp_path / "code"
    own = source(own_root)
    foreign_root = tmp_path / "extractor"
    foreign = source(foreign_root)
    monkeypatch.setattr(coordinator, "TREE", own_root)
    identity["source_manifest_sha256"] = file_hash(foreign)
    coordinator.write_report(
        old / "identity.json", dict(identity, plan_digest=plan["scope"]["parent_plan_digest"])
    )
    for job in jobs:
        complete(old, job, split, identity)
    original = audit.source_integrity
    own_checks = 0

    def change_manifest_at_postflight(tree, manifest):
        nonlocal own_checks
        if Path(tree) == own_root:
            own_checks += 1
            if own_checks == 2:
                document = json.loads(own.read_text())
                document["annotation"] = "changed after registration"
                own.write_text(json.dumps(document))
        return original(tree, manifest)

    monkeypatch.setattr(audit, "source_integrity", change_manifest_at_postflight)
    argv = [
        "build",
        "--plan-dir",
        str(tmp_path / "selected"),
        "--source-manifest",
        str(own),
        "--extractor-code-root",
        str(foreign_root),
        "--extractor-source-manifest",
        str(foreign),
        "--run-evidence-dir",
        str(tmp_path / "run"),
        "--sc2-path",
        str(tmp_path / "engine"),
        "--output-dir",
        str(tmp_path / "data"),
        "--workers",
        "1",
        "--reserve-gib",
        "1",
        "--maximum-delay",
        "8",
        "--shard-decisions",
        "1",
        "--reuse-data-root",
        str(old),
    ]
    monkeypatch.setattr("sys.argv", argv)

    def no_engine(*args, **kwargs):
        raise AssertionError("complete imports must not start SC2")

    monkeypatch.setattr(coordinator, "extraction_process", no_engine)
    with pytest.raises(ValueError, match="source manifest/qualification changed"):
        coordinator.main()
    assert original(own_root, own)["problems"] == []
    pointer, _ = coordinator.read_verified(tmp_path / "data/dispatch_completion.json")
    assert pointer["status"] == "RUNNING" and pointer["report"] is None
    assert not (tmp_path / "run/report.json").exists()
