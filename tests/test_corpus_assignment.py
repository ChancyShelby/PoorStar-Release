"""Tests: test corpus assignment."""

from pathlib import Path

import pytest

from poorstar.cli.data import build as coordinator
from poorstar.common.integrity import file_hash
from poorstar.data.corpus.assignment import SCHEMA
from tests.test_coordinator_sources import complete, setup


def assignment_fixture(tmp_path, monkeypatch):
    tree, own, root, foreign, jobs, split, argv = setup(tmp_path, monkeypatch)
    _, digest, frozen = coordinator.freeze_jobs(tmp_path / "plan")
    identity = dict(
        schema="poorstar_c3_full_corpus_coordinator_v1",
        plan_digest=digest,
        source_manifest_sha256=file_hash(foreign),
        split_sha256=split["sha256"],
        maximum_delay=32,
        shard_decisions=1024,
        solid_compression=True,
        solid_codec="xz",
    )
    value = dict(
        schema=SCHEMA,
        data_binding=identity,
        registered_views=len(frozen),
        job_ids=[jobs[-1]["job_id"]],
    )
    manifest = tmp_path / "assignment.json"
    coordinator.write_report(manifest, value)
    argv.extend(["--assignment-manifest", str(manifest)])
    return jobs, split, argv, value, manifest


@pytest.mark.parametrize(
    "field",
    [
        "plan_digest",
        "source_manifest_sha256",
        "split_sha256",
        "maximum_delay",
        "shard_decisions",
        "solid_codec",
    ],
)
def test_assignment_cannot_change_data_binding_before_output(tmp_path, monkeypatch, field):
    jobs, split, argv, value, manifest = assignment_fixture(tmp_path, monkeypatch)
    value["data_binding"][field] = "drift"
    coordinator.write_report(manifest, value)
    with pytest.raises(ValueError, match="assignment source/plan/split/configuration drift"):
        coordinator.main()
    assert not (tmp_path / "data").exists() and not (tmp_path / "run1").exists()


@pytest.mark.parametrize(
    "case", ["empty", "duplicate", "unknown", "denominator", "extra", "tamper"]
)
def test_assignment_invalid_scope_never_starts_engine(tmp_path, monkeypatch, case):
    jobs, split, argv, value, manifest = assignment_fixture(tmp_path, monkeypatch)
    if case == "empty":
        value["job_ids"] = []
    elif case == "duplicate":
        value["job_ids"] *= 2
    elif case == "unknown":
        value["job_ids"] = ["not-a-registered-job"]
    elif case == "denominator":
        value["registered_views"] = 1
    elif case == "extra":
        value["unregistered_option"] = True
    coordinator.write_report(manifest, value)
    if case == "tamper":
        manifest.write_text(manifest.read_text().replace(jobs[-1]["job_id"], "0" * 64))
    with pytest.raises(ValueError):
        coordinator.main()
    assert not (tmp_path / "data").exists() and not (tmp_path / "run1").exists()


def test_partial_dispatch_complete_is_not_full_corpus_and_resume_preserves_markers(
    tmp_path, monkeypatch
):
    jobs, split, argv, value, manifest = assignment_fixture(tmp_path, monkeypatch)
    calls = []

    def process(command, log, timeout, *, cwd):
        calls.append(command)
        complete(command, jobs, split)
        return 0

    monkeypatch.setattr(coordinator, "extraction_process", process)
    assert coordinator.main() == 0 and len(calls) == 1
    assert calls[0][calls[0].index("--replay") + 1] == jobs[-1]["replay"]
    report, _ = coordinator.read_verified(tmp_path / "run1/report.json")
    assert report["registered_views"] == 2 and report["requested_assigned_views"] == 1
    assert report["completed_views"] == 1 and report["complete_assigned_engine_extraction"]
    assert not report["complete_engine_extraction"] and not report["paper_corpus_gate_passed"]
    assert (
        not report["formal_bc_ready"] and not report["rl_ready"] and report["optimizer_steps"] == 0
    )
    identity = tmp_path / "data/identity.json"
    marker = next((tmp_path / "data").rglob("job_completion.json"))
    before = (file_hash(identity), file_hash(marker))
    argv[argv.index("--run-evidence-dir") + 1] = str(tmp_path / "run2")
    assert coordinator.main() == 0 and len(calls) == 1
    assert before == (file_hash(identity), file_hash(marker))
    # Removing the assignment cannot silently expand the root on restart.
    argv[argv.index("--run-evidence-dir") + 1] = str(tmp_path / "run3")
    del argv[-2:]
    with pytest.raises(ValueError, match="resume dispatch assignment drift"):
        coordinator.main()
    assert len(calls) == 1 and not (tmp_path / "run3").exists()


def test_different_assignment_refuses_same_root(tmp_path, monkeypatch):
    jobs, split, argv, value, manifest = assignment_fixture(tmp_path, monkeypatch)

    def process(command, log, timeout, *, cwd):
        complete(command, jobs, split)
        return 0

    monkeypatch.setattr(coordinator, "extraction_process", process)
    assert coordinator.main() == 0
    value["job_ids"] = [jobs[0]["job_id"]]
    coordinator.write_report(manifest, value)
    argv[argv.index("--run-evidence-dir") + 1] = str(tmp_path / "run2")
    with pytest.raises(ValueError, match="resume dispatch assignment drift"):
        coordinator.main()
    assert not (tmp_path / "run2").exists()


def test_existing_full_root_cannot_be_relabelled_as_assignment(tmp_path, monkeypatch):
    jobs, split, argv, value, manifest = assignment_fixture(tmp_path, monkeypatch)
    data = tmp_path / "data"
    data.mkdir()
    coordinator.write_report(data / "identity.json", value["data_binding"])
    before = file_hash(data / "identity.json")
    with pytest.raises(ValueError, match="cannot narrow"):
        coordinator.main()
    assert (
        file_hash(data / "identity.json") == before
        and not (data / "dispatch_assignment.json").exists()
    )
