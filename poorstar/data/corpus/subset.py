"""Data: corpus: subset."""

import fcntl
import gzip
import json
from collections import Counter, defaultdict
from contextlib import ExitStack, contextmanager
from pathlib import Path

from poorstar.common.integrity import file_hash, payload_hash
from poorstar.data.bc.records import validate_split

SUBSET_SCHEMA = "poorstar_c3_inherited_matchup_scope_v1"
REUSE_SCHEMA = "poorstar_c3_read_only_completed_job_import_v1"


def verified_current_completion(root, identity, control, jobs, owned):
    """Resolve the current owner's external final report under its shared lock.

    A stale root/report.json never substitutes for completion. A new run first
    invalidates completion, so an interrupted or failed restart cannot reuse an
    earlier success even when its immutable report still exists.
    """
    from poorstar.cli.data.build import read_verified

    root = Path(root).resolve()

    def real_file(path):
        path = Path(path)
        if path.is_symlink() or not path.is_file():
            raise ValueError(
                "current dispatch completion requires an existing real file: " + str(path)
            )
        return path

    pointer_path = real_file(root / "dispatch_completion.json")
    pointer, pointer_digest = read_verified(pointer_path)
    if (
        set(pointer)
        != {
            "schema",
            "status",
            "run_id",
            "registration_digest",
            "current_registration_file_sha256",
            "report",
            "source_postflight",
        }
        or pointer["schema"] != "poorstar_c3_dispatch_completion_reference_v1"
        or pointer["status"] != "COMPLETE"
    ):
        raise ValueError("complete current dispatch reference required")
    run_path = real_file(root / "dispatch_run.json")
    run, run_digest = read_verified(run_path)
    if (
        set(run)
        != {
            "schema",
            "run_id",
            "registration_path",
            "data_output_dir",
            "data_binding",
            "data_identity_digest",
            "data_identity_file_sha256",
            "dispatch_assignment_digest",
            "dispatch_assignment_file_sha256",
            "registered_views",
            "requested_job_ids",
            "run_evidence_dir",
            "report_path",
            "run_identity_digest",
            "run_identity_file_sha256",
            "coordinator_binding",
            "source_roles",
        }
        or pointer["registration_digest"] != run_digest
        or pointer["run_id"] != run.get("run_id")
        or pointer["current_registration_file_sha256"] != file_hash(run_path)
        or run.get("schema") != "poorstar_c3_dispatch_run_registration_v1"
    ):
        raise ValueError("stale current dispatch completion registration")
    registration_path = real_file(run["registration_path"])
    registered, _ = read_verified(registration_path)
    if registered != run or file_hash(registration_path) != file_hash(run_path):
        raise ValueError("immutable dispatch registration drift")
    data, data_digest = read_verified(root / "identity.json")
    dispatch, dispatch_digest = read_verified(root / "dispatch_assignment.json")
    if (
        data != identity
        or dispatch != control
        or run["data_binding"] != identity
        or run["data_output_dir"] != str(root)
        or run["data_identity_digest"] != data_digest
        or run["data_identity_file_sha256"] != file_hash(root / "identity.json")
        or run["dispatch_assignment_digest"] != dispatch_digest
        or run["dispatch_assignment_file_sha256"] != file_hash(root / "dispatch_assignment.json")
        or run["registered_views"] != len(jobs)
        or run["requested_job_ids"] != [j["job_id"] for j in jobs if j["job_id"] in owned]
    ):
        raise ValueError("current dispatch completion data/assignment binding drift")
    evidence = Path(run["run_evidence_dir"])
    # Complete sources use the required fresh independent coordinator evidence.
    if (
        not evidence.is_absolute()
        or evidence.is_symlink()
        or evidence.resolve() == root
        or registration_path != evidence / "dispatch_registration.json"
        or run["report_path"] != str(evidence / "report.json")
        or not run["coordinator_binding"]
    ):
        raise ValueError("fresh independent coordinator completion evidence required")
    lineage_path = real_file(evidence / "identity.json")
    lineage, lineage_digest = read_verified(lineage_path)
    if (
        lineage != run["coordinator_binding"]
        or lineage_digest != run["run_identity_digest"]
        or file_hash(lineage_path) != run["run_identity_file_sha256"]
        or lineage.get("schema") != "poorstar_c3_coordinator_source_roles_v1"
        or lineage.get("data_binding") != identity
        or lineage.get("data_output_dir") != str(root)
        or lineage.get("dispatch_assignment") != control["assignment"]
        or lineage.get("sources") != run["source_roles"]
    ):
        raise ValueError("current dispatch completion source-role identity drift")
    roles = run["source_roles"]
    if roles.get("extractor_source_manifest_sha256") != identity[
        "source_manifest_sha256"
    ] or pointer["source_postflight"] != {
        role: roles.get(role) for role in ("coordinator", "extractor")
    }:
        raise ValueError("current dispatch completion source postflight drift")
    for role in ("coordinator", "extractor"):
        integrity = roles.get(role, {})
        if (
            integrity.get("problems") != []
            or type(integrity.get("files_checked")) is not int
            or integrity["files_checked"] < 1
            or integrity.get("manifest_sha256") != roles.get(role + "_source_manifest_sha256")
        ):
            raise ValueError("current dispatch completion source qualification missing")
    reference = pointer["report"]
    if (
        not isinstance(reference, dict)
        or set(reference) != {"path", "content_digest", "file_sha256"}
        or reference["path"] != run["report_path"]
    ):
        raise ValueError("current dispatch completion report location drift")
    report_path = real_file(reference["path"])
    completed, completed_digest = read_verified(report_path)
    if (
        completed_digest != reference["content_digest"]
        or file_hash(report_path) != reference["file_sha256"]
        or any(completed.get(k) != v for k, v in identity.items())
        or completed.get("dispatch_registration_digest") != run_digest
        or completed.get("coordinator_binding") != lineage
        or type(completed.get("registered_views")) is not int
        or completed["registered_views"] != len(jobs)
        or completed.get("complete_assigned_engine_extraction") is not True
        or completed.get("complete_engine_extraction") is not (len(owned) == len(jobs))
        or completed.get("stop_requested") is not False
        or completed.get("failures") != []
        or type(completed.get("completed_views")) is not int
        or completed["completed_views"] != len(owned)
        or type(completed.get("requested_assigned_views")) is not int
        or completed["requested_assigned_views"] != len(owned)
        or type(completed.get("optimizer_steps")) is not int
        or completed["optimizer_steps"] != 0
        or completed.get("formal_bc_ready") is not False
        or completed.get("rl_ready") is not False
    ):
        raise ValueError("complete drained source assignment/report required")
    return dict(
        completion_reference_digest=pointer_digest,
        completion_reference_file_sha256=file_hash(pointer_path),
        registration_digest=run_digest,
        report_path=str(report_path),
        report_digest=completed_digest,
        report_file_sha256=file_hash(report_path),
    )


def derive_zvz_plan(parent, parent_digest, jobs, split):
    """Select whole ZvZ replay groups in original order; never resample splits."""
    if parent["schema"] != "poorstar_c3_full_corpus_extraction_plan_v1":
        raise ValueError("original full plan required")
    if parent.get("scope") is not None:
        raise ValueError("derive from original full plan, not another subset")
    if payload_hash(parent) != parent_digest:
        raise ValueError("parent plan digest drift")
    if validate_split(split) != parent["split_sha256"]:
        raise ValueError("parent split drift")
    if len(jobs) != parent["counts"]["unique_byte_identical_views"]:
        raise ValueError("parent job denominator drift")
    seen = set()
    groups = defaultdict(list)
    native_splits = {}
    native_matchups = {}
    for job in jobs:
        replay, player = job["replay_sha256"], job["player_id"]
        if type(player) is not int or player not in (1, 2):
            raise ValueError("one versus one player identity required")
        if job["job_id"] != payload_hash([replay, player]) or job["job_id"] in seen:
            raise ValueError("duplicate or drifted job identity")
        seen.add(job["job_id"])
        if split["assignments"].get(replay) != job["split"]:
            raise ValueError("job split drift")
        native = job.get("native_content_group")
        if not isinstance(native, str) or not native:
            raise ValueError("explicit native content group required")
        if job.get("matchup") not in ("ZvZ", "ZvT", "ZvP"):
            raise ValueError("original matchup required")
        if native_splits.setdefault(native, job["split"]) != job["split"]:
            raise ValueError("native group split leakage")
        if native_matchups.setdefault(native, job["matchup"]) != job["matchup"]:
            raise ValueError("native group matchup drift")
        aliases = job.get("aliases")
        if not isinstance(aliases, list) or not aliases or len(set(aliases)) != len(aliases):
            raise ValueError("explicit unique aliases required")
        groups[replay].append(job)
    for members in groups.values():
        matchups = {j["matchup"] for j in members}
        if len(matchups) != 1 or len({j["native_content_group"] for j in members}) != 1:
            raise ValueError("replay group metadata drift")
        if members[0]["matchup"] == "ZvZ":
            if {j["player_id"] for j in members} != {1, 2} or len(members) != 2:
                raise ValueError("ZvZ requires both complete registered player views")
            if members[0]["aliases"] != members[1]["aliases"]:
                raise ValueError("ZvZ paired aliases drift")
    chosen = [dict(job) for job in jobs if job["matchup"] == "ZvZ"]
    if not chosen:
        raise ValueError("empty ZvZ scope")
    replay_ids = {j["replay_sha256"] for j in chosen}
    alias_views = sum(len(j["aliases"]) for j in chosen)
    native_groups = {j["native_content_group"] for j in chosen}
    report = dict(
        schema=parent["schema"],
        purpose="OWNER_AUTHORIZED_ZVZ_SCOPE_INHERITED_SPLIT",
        scope=dict(
            schema=SUBSET_SCHEMA,
            matchup="ZvZ",
            parent_plan_digest=parent_digest,
            parent_jobs_sha256=parent["jobs_sha256"],
            original_registered_views=len(jobs),
            split_policy="BYTE_IDENTICAL_PARENT_SPLIT_NO_REASSIGNMENT",
            perspective_policy="BOTH_PLAYERS_PER_ZVZ_REPLAY",
            omitted_matchups=["ZvT", "ZvP"],
        ),
        split_sha256=parent["split_sha256"],
        seed=parent.get("seed", split["seed"]),
        parent_scope_digest=parent.get("scope_digest"),
        parent_native_census_digest=parent.get("native_census_digest"),
        counts=dict(
            candidate_matches=alias_views // 2,
            unique_replay_files=len(replay_ids),
            candidate_alias_views=alias_views,
            unique_byte_identical_views=len(chosen),
            byte_identical_alias_duplicates=alias_views - len(chosen),
            native_content_groups=len(native_groups),
        ),
        split_view_counts=dict(Counter(j["split"] for j in chosen)),
        matchup_view_counts={"ZvZ": len(chosen)},
        history_quarantine=False,
        native_content_groups_used_for_split_not_view_deletion=True,
        complete_engine_extraction=False,
        completed_views=0,
        optimizer_steps=0,
        formal_bc_ready=False,
        rl_ready=False,
    )
    report["scope_digest"] = payload_hash(
        dict(scope=report["scope"], job_ids=[j["job_id"] for j in chosen])
    )
    return report, chosen


def write_zvz_plan(parent_dir, output_dir):
    """Publish a fresh deterministic job list and the exact original split bytes."""
    from poorstar.cli.data.build import freeze_jobs, write_report

    parent_dir, output_dir = Path(parent_dir), Path(output_dir)
    if output_dir.exists():
        raise FileExistsError("fresh subset output required")
    parent, digest, jobs = freeze_jobs(parent_dir)
    split = json.loads((parent_dir / "split.json").read_text())
    report, chosen = derive_zvz_plan(parent, digest, jobs, split)
    output_dir.mkdir(parents=True)
    (output_dir / "split.json").write_bytes((parent_dir / "split.json").read_bytes())
    wire = "".join(
        json.dumps(job, sort_keys=True, separators=(",", ":")) + "\n" for job in chosen
    ).encode()
    with (output_dir / "jobs.jsonl.gz").open("xb") as stream:
        with gzip.GzipFile(fileobj=stream, mode="wb", filename="", mtime=0) as compressed:
            compressed.write(wire)
    report["jobs_sha256"] = file_hash(output_dir / "jobs.jsonl.gz")
    report["jobs_bytes"] = (output_dir / "jobs.jsonl.gz").stat().st_size
    report["scope"]["parent_split_file_sha256"] = file_hash(parent_dir / "split.json")
    report["scope_digest"] = payload_hash(
        dict(scope=report["scope"], job_ids=[j["job_id"] for j in chosen])
    )
    write_report(output_dir / "report.json", report)
    return freeze_jobs(output_dir)


def validate_subset_jobs(plan, jobs, split):
    """The selected denominator must be the complete paired ZvZ scope it declares."""
    scope = plan.get("scope")
    if scope is None:
        return
    if (
        scope.get("schema") != SUBSET_SCHEMA
        or scope.get("matchup") != "ZvZ"
        or scope.get("split_policy") != "BYTE_IDENTICAL_PARENT_SPLIT_NO_REASSIGNMENT"
        or scope.get("perspective_policy") != "BOTH_PLAYERS_PER_ZVZ_REPLAY"
        or scope.get("omitted_matchups") != ["ZvT", "ZvP"]
        or scope.get("original_registered_views", 0) < len(jobs)
    ):
        raise ValueError("subset scope contract drift")
    # Reuse the complete grouping validation with a temporary original-plan view;
    # it preserves the actual job/split values and never publishes that view.
    original = {k: v for k, v in plan.items() if k != "scope"}
    derived, chosen = derive_zvz_plan(original, payload_hash(original), jobs, split)
    if (
        chosen != jobs
        or derived["counts"] != plan["counts"]
        or derived["split_view_counts"] != plan["split_view_counts"]
    ):
        raise ValueError("subset paired ZvZ grouping/denominator drift")
    if plan["matchup_view_counts"] != {"ZvZ": len(jobs)}:
        raise ValueError("subset matchup denominator drift")
    if plan["scope_digest"] != payload_hash(dict(scope=scope, job_ids=[j["job_id"] for j in jobs])):
        raise ValueError("subset scope digest drift")


@contextmanager
def read_only_owner_lock(root):
    """Exclude production writers while allowing concurrent immutable readers."""
    root = Path(root).resolve()
    path = root / "coordinator.lock"
    if not path.is_file() or path.is_symlink():
        raise ValueError("existing real source corpus owner lock required")
    with path.open("rb") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_SH | fcntl.LOCK_NB)
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def verify_completed_rows(
    attempt, job, *, split, source_sha256, maximum_delay, shard_decisions, solid_codec
):
    """Exhaust every member/row, native tail and terminal; run the real teacher census.

    No model is fitted. Native unit tags remain exclusively in the original external
    artifacts. Only counts and hashes are published in the import qualification.
    """
    from s2clientprotocol import sc2api_pb2 as sc

    from poorstar.cli.data.build import completed_job
    from poorstar.data.bc.records import BCShardReader
    from poorstar.data.bc.supervision import DISTAR_SKIP, supervision_census
    from poorstar.data.trajectory import decode_bytes

    report = completed_job(
        attempt,
        job,
        split_sha256=split["sha256"],
        source_sha256=source_sha256,
        maximum_delay=maximum_delay,
        shard_decisions=shard_decisions,
        solid_codec=solid_codec,
    )
    attempt = Path(attempt)
    if report.get("schema") != "poorstar_c3_direct_decision_extraction_v1":
        raise ValueError("original complete extraction schema required")
    terminal = sc.ResponseObservation()
    terminal.ParseFromString((attempt / "extraction_terminal.protocol").read_bytes())
    if (
        not terminal.player_result
        or int(terminal.observation.game_loop) != report["terminal_loop"]
        or int(terminal.observation.player_common.player_id) != job["player_id"]
    ):
        raise ValueError("terminal clock/player/result drift")
    boundary = report.get("terminal_native_boundary")
    if boundary is None:
        raise ValueError("complete native terminal boundary required")
    tail = Counter()
    previous = -1
    with gzip.open(attempt / boundary["artifact"], "rt") as stream:
        for line in stream:
            event = json.loads(line, object_hook=decode_bytes)
            loop = event["_gameloop"]
            if (
                type(loop) is not int
                or loop < report["terminal_loop"]
                or loop < previous
                or not (
                    event["_event"].endswith("SCmdEvent")
                    or (
                        event["_event"].endswith("SCommandManagerStateEvent")
                        and event.get("m_state") == 1
                    )
                )
            ):
                raise ValueError("native terminal tail event drift")
            previous = loop
            tail[
                "commands_at_terminal"
                if loop == report["terminal_loop"]
                else "commands_after_terminal"
            ] += 1
    for name in ("commands_at_terminal", "commands_after_terminal"):
        if tail[name] != boundary[name]:
            raise ValueError("native terminal tail denominator drift")
    counts = Counter()
    native_count = 0
    cursor = 0
    profile = None
    profile_hash = None
    member_count = 0
    shard_paths = set()

    def rows():
        nonlocal cursor, native_count, profile, profile_hash, member_count
        for shard in report["shards"]:
            path = Path(shard["path"]).resolve()
            if path in shard_paths:
                raise ValueError("duplicate imported shard")
            shard_paths.add(path)
            reader = BCShardReader(
                path, sha256=shard["sha256"], split_manifest=split, allowed_split=job["split"]
            )
            try:
                index = reader.index
                if (
                    index["source_manifest_sha256"] != source_sha256
                    or index["extractor_sha256"] != report["extractor_sha256"]
                    or index["maximum_delay"] != maximum_delay
                    or len(index["decisions"]) != shard["decisions"]
                    or not 1 <= shard["decisions"] <= shard_decisions
                ):
                    raise ValueError("imported shard configuration/row denominator drift")
                for name in index["members"]:
                    reader._read(name)
                member_count += len(index["members"])
                if profile is None:
                    profile = reader.profile
                    profile_hash = profile.sha256
                elif reader.profile.sha256 != profile_hash:
                    raise ValueError("mixed imported action profile")
                for row in reader:
                    if (
                        row.replay_sha256 != job["replay_sha256"]
                        or row.scheduler.key.player_id != job["player_id"]
                        or row.scheduler.key.episode
                        != job["replay_sha256"] + ":player" + str(job["player_id"])
                        or row.scheduler.key.loop != cursor
                    ):
                        raise ValueError("imported replay/player/full clock coverage drift")
                    cursor += row.delay + int(row.decision == "ACT")
                    if cursor > report["terminal_loop"]:
                        raise ValueError("imported row crosses engine terminal")
                    for wire in row.native_event_wires:
                        event = json.loads(wire, object_hook=decode_bytes)
                        if event["_gameloop"] != row.body.key.loop:
                            raise ValueError("imported native/body time alignment drift")
                        native_count += 1
                    for name in (
                        "all_decisions",
                        row.decision,
                        row.body_rejection or "body_not_structurally_rejected",
                    ):
                        counts[name] += 1
                    yield row
            finally:
                reader.close()

    # The first profile is needed before the lazy census iterator advances.
    first = report["shards"][0]
    reader = BCShardReader(
        first["path"], sha256=first["sha256"], split_manifest=split, allowed_split=job["split"]
    )
    try:
        initial_profile = reader.profile
    finally:
        reader.close()
    census = supervision_census(rows(), initial_profile, policy=DISTAR_SKIP)
    if dict(counts) != report["counts"] or cursor != report["terminal_loop"]:
        raise ValueError("complete imported row counts/terminal coverage drift")
    if native_count != boundary["playable_command_events"]:
        raise ValueError("playable native command denominator drift")
    qualification = dict(
        method="ALL_MEMBERS_ROWS_NATIVE_TAIL_TERMINAL_AND_DISTAR_TEACHER_CENSUS",
        rows=counts["all_decisions"],
        members=member_count,
        playable_native_events=native_count,
        terminal_native_events=sum(tail.values()),
        terminal_loop=report["terminal_loop"],
        extraction_counts=dict(counts),
        profile_sha256=profile_hash,
        supervision_census_digest=payload_hash(census),
        supervision_counts=census["counts"],
        supervision_policy=DISTAR_SKIP,
        optimizer_steps=0,
        capability_evaluated=False,
    )
    qualification["content_digest"] = payload_hash(qualification)
    return report, qualification


def import_completed_jobs(
    plan,
    plan_digest,
    jobs,
    split,
    identity,
    source_roots,
    ledger_path,
    *,
    requested_job_ids=None,
    require_source_complete=False,
):
    """Verify drained owners and publish references to unchanged source artifacts.

    Resume re-verifies all completed inputs; incomplete attempts are recorded and
    remain pending. Duplicate ownership across independent roots is rejected.
    """
    from poorstar.cli.data.build import read_verified, write_report

    if (
        identity["plan_digest"] != plan_digest
        or identity["split_sha256"] != split["sha256"]
        or validate_split(split) != plan["split_sha256"]
    ):
        raise ValueError("import destination plan/split drift")
    if plan.get("scope", {}).get("schema") != SUBSET_SCHEMA:
        raise ValueError("explicit inherited subset plan required for import")
    all_jobs = jobs
    known = {job["job_id"]: job for job in jobs}
    if len(known) != len(jobs):
        raise ValueError("duplicate destination job identity")
    requested = list(known) if requested_job_ids is None else list(requested_job_ids)
    if len(set(requested)) != len(requested) or set(requested) - set(known):
        raise ValueError("unknown or duplicate import assignment job")
    jobs = [job for job in all_jobs if job["job_id"] in set(requested)]
    roots = [Path(root).resolve() for root in source_roots]
    if not roots or len(set(roots)) != len(roots):
        raise ValueError("unique explicit import source roots required")
    roots.sort(key=str)
    ledger_path = Path(ledger_path)
    if any(ledger_path.resolve().is_relative_to(root) for root in roots):
        raise ValueError("ledger must be outside immutable source roots")
    binding = dict(
        destination=identity,
        source_roots=[str(root) for root in roots],
        parent_plan_digest=plan["scope"]["parent_plan_digest"],
        registered_views=len(all_jobs),
        requested_job_ids=[job["job_id"] for job in jobs],
        require_source_complete=require_source_complete,
    )
    prior = None
    if ledger_path.exists():
        prior, _ = read_verified(ledger_path)
        if prior.get("schema") != REUSE_SCHEMA or prior.get("binding") != binding:
            raise ValueError("resume import ledger binding drift")
    imports = []
    incomplete = []
    reports = {}
    root_bindings = []
    with ExitStack() as stack:
        locked = set()
        for root in roots:
            stack.enter_context(read_only_owner_lock(root))
            locked.add(root)
        for root in roots:
            old, old_digest = read_verified(root / "identity.json")
            expected = dict(identity, plan_digest=plan["scope"]["parent_plan_digest"])
            if old not in (expected, identity):
                raise ValueError("source owner extractor/split/delay/codec/parent-plan drift")
            owned = set(known)
            completion_binding = None
            assignment_path = root / "dispatch_assignment.json"
            if old == identity:
                if not assignment_path.exists():
                    raise ValueError("same-scope source dispatch ownership required")
                control, _ = read_verified(assignment_path)
                if set(control) != {"assignment"}:
                    raise ValueError("source dispatch ownership fields drift")
                if control["assignment"] is not None:
                    envelope = control["assignment"]
                    if (
                        set(envelope) != {"manifest", "content_digest"}
                        or payload_hash(envelope["manifest"]) != envelope["content_digest"]
                    ):
                        raise ValueError("source assignment digest drift")
                    from poorstar.data.corpus.assignment import select_assignment

                    owned = {
                        job["job_id"]
                        for job in select_assignment(envelope["manifest"], old, all_jobs)
                    }
                for path in (root / "jobs").glob("*"):
                    if path.is_dir() and path.name not in owned:
                        raise ValueError("source contains unassigned job ownership")
                if require_source_complete:
                    completion_binding = verified_current_completion(
                        root, old, control, all_jobs, owned
                    )
            elif require_source_complete:
                raise ValueError("complete aggregate sources must have the same scientific scope")
            root_bindings.append(
                dict(
                    path=str(root),
                    identity_digest=old_digest,
                    identity_file_sha256=file_hash(root / "identity.json"),
                )
            )
            if completion_binding is not None:
                root_bindings[-1]["current_completion"] = completion_binding
            referenced = {}
            if (root / "read_only_import.json").exists():
                source_ledger, _ = read_verified(root / "read_only_import.json")
                source_binding = source_ledger.get("binding", {})
                if (
                    source_ledger.get("schema") != REUSE_SCHEMA
                    or source_binding.get("destination") != old
                    or source_binding.get("registered_views") != len(all_jobs)
                    or set(source_binding.get("requested_job_ids", [])) != owned
                    or source_binding.get("parent_plan_digest")
                    != plan["scope"]["parent_plan_digest"]
                    or source_ledger.get("imported_views") != len(source_ledger.get("imports", []))
                ):
                    raise ValueError("source import ledger ownership/binding drift")
                for entry in source_ledger["imports"]:
                    key = entry["job_id"]
                    if key in referenced or key not in owned:
                        raise ValueError("duplicate or unassigned source import reference")
                    origin = Path(entry["source_root"]).resolve()
                    if str(origin) not in source_binding["source_roots"] or origin == root:
                        raise ValueError("source import reference provenance drift")
                    if origin not in locked:
                        stack.enter_context(read_only_owner_lock(origin))
                        locked.add(origin)
                    origin_identity, _ = read_verified(origin / "identity.json")
                    if origin_identity not in (expected, identity):
                        raise ValueError("referenced original owner configuration drift")
                    attempt = Path(entry["attempt"]).resolve()
                    if (
                        not attempt.is_relative_to(origin / "jobs" / key)
                        or file_hash(attempt / "report.json")
                        != entry["original_report_file_sha256"]
                        or file_hash(attempt / "job_completion.json")
                        != entry["original_completion_file_sha256"]
                    ):
                        raise ValueError("referenced original artifact drift")
                    referenced[key] = (attempt, origin)
            for job in jobs:
                if job["job_id"] not in owned:
                    continue
                job_root = root / "jobs" / job["job_id"]
                reference = referenced.get(job["job_id"])
                if not job_root.exists() and reference is None:
                    continue
                if job_root.is_symlink() or not job_root.resolve().is_relative_to(root):
                    raise ValueError("source job placement escape")
                attempts = sorted(
                    {
                        x if x.is_dir() else Path(str(x)[: -len(".record.json")])
                        for x in job_root.glob("attempt_*")
                        if x.is_dir() or x.name.endswith(".record.json")
                    }
                )
                if reference is not None:
                    if attempts:
                        raise ValueError("source imported/local duplicate job ownership")
                    attempt, origin = reference
                elif attempts:
                    attempt, origin = attempts[-1], root
                else:
                    continue
                if attempt.is_symlink() or not attempt.resolve().is_relative_to(origin):
                    raise ValueError("source attempt placement escape")
                if not (attempt / "job_completion.json").exists():
                    incomplete.append(
                        dict(
                            job_id=job["job_id"],
                            source_root=str(root),
                            attempt=str(attempt),
                            reason="NO_LATEST_COMPLETE_MARKER",
                        )
                    )
                    continue
                report, qualification = verify_completed_rows(
                    attempt,
                    job,
                    split=split,
                    source_sha256=identity["source_manifest_sha256"],
                    maximum_delay=identity["maximum_delay"],
                    shard_decisions=identity["shard_decisions"],
                    solid_codec=identity["solid_codec"],
                )
                if job["job_id"] in reports:
                    raise ValueError("duplicate completed job ownership across source roots")
                reports[job["job_id"]] = report
                _, report_digest = read_verified(attempt / "report.json")
                imports.append(
                    dict(
                        job_id=job["job_id"],
                        source_root=str(origin),
                        via_owner_root=str(root),
                        attempt=str(attempt),
                        original_report_digest=report_digest,
                        original_report_file_sha256=file_hash(attempt / "report.json"),
                        original_completion_file_sha256=file_hash(attempt / "job_completion.json"),
                        replay_sha256=job["replay_sha256"],
                        player_id=job["player_id"],
                        split=job["split"],
                        qualification=qualification,
                    )
                )
        order = {job["job_id"]: i for i, job in enumerate(jobs)}
        imports.sort(key=lambda entry: order[entry["job_id"]])
        value = dict(
            schema=REUSE_SCHEMA,
            binding=binding,
            source_bindings=root_bindings,
            imports=imports,
            incomplete=incomplete,
            imported_views=len(imports),
            imported_rows=sum(entry["qualification"]["rows"] for entry in imports),
            immutable_source_artifacts=True,
            source_data_relabelled=False,
            optimizer_steps=0,
            formal_bc_ready=False,
            rl_ready=False,
        )
        if prior is not None:
            if prior != value:
                raise ValueError("immutable import ledger/source completion drift")
        else:
            ledger_path.parent.mkdir(parents=True, exist_ok=True)
            write_report(ledger_path, value)
    return reports, value
