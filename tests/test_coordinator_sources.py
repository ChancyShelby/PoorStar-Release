"""Tests: test coordinator sources."""

import json
import os
import signal
import sys
from pathlib import Path

import pytest

from poorstar.cli.data import build as coordinator
from poorstar.common.integrity import file_hash
from tests.test_bc_prelaunch import _full_plan_fixture


def source(root, text='print("bound extractor")\n'):
    (root / "poorstar/cli/data").mkdir(parents=True)
    p = root / "poorstar/cli/data/extract.py"
    p.write_text(text)
    manifest = root.parent / (root.name + "_manifest.json")
    manifest.write_text(
        json.dumps(
            {
                "files": [
                    dict(
                        path="poorstar/cli/data/extract.py",
                        sha256=file_hash(p),
                        bytes=p.stat().st_size,
                    )
                ]
            }
        )
    )
    return manifest


def setup(tmp_path, monkeypatch):
    tree = tmp_path / "coordinator"
    own = source(tree)
    extraction = tmp_path / "extractor"
    foreign = source(extraction, 'print("independent data source")\n')
    monkeypatch.setattr(coordinator, "TREE", tree)
    jobs, split = _full_plan_fixture(tmp_path / "plan")
    argv = [
        "build",
        "--plan-dir",
        str(tmp_path / "plan"),
        "--source-manifest",
        str(own),
        "--extractor-code-root",
        str(extraction),
        "--extractor-source-manifest",
        str(foreign),
        "--run-evidence-dir",
        str(tmp_path / "run1"),
        "--sc2-path",
        str(tmp_path / "engine"),
        "--output-dir",
        str(tmp_path / "data"),
        "--workers",
        "4",
        "--reserve-gib",
        "1",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    return tree, own, extraction, foreign, jobs, split, argv


def complete(command, jobs, split):
    options = {command[i]: command[i + 1] for i in range(2, len(command) - 1, 2)}
    out = Path(options["--output-dir"])
    out.mkdir()
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
            source_manifest_sha256=options["--source-manifest-sha256"],
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


@pytest.mark.parametrize("missing", ["root", "manifest"])
def test_independent_extractor_requires_paired_binding(tmp_path, monkeypatch, missing):
    tree, own, root, foreign, *_ = setup(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="supplied together"):
        coordinator.verified_sources(
            own, None if missing == "root" else root, None if missing == "manifest" else foreign
        )


@pytest.mark.parametrize("which", ["coordinator", "extractor"])
def test_drift_in_either_source_refuses_before_output(tmp_path, monkeypatch, which):
    tree, own, root, foreign, *_ = setup(tmp_path, monkeypatch)
    ((tree if which == "coordinator" else root) / "poorstar/cli/data/extract.py").write_text(
        "drift"
    )
    with pytest.raises(ValueError, match=which + " source preflight"):
        coordinator.main()
    assert not (tmp_path / "data").exists() and not (tmp_path / "run1").exists()


def test_extractor_symlink_escape_and_unlisted_script_refused(tmp_path, monkeypatch):
    tree, own, root, foreign, *_ = setup(tmp_path, monkeypatch)
    p = root / "poorstar/cli/data/extract.py"
    p.unlink()
    p.symlink_to(tree / "poorstar/cli/data/extract.py")
    with pytest.raises(ValueError, match="script missing or outside"):
        coordinator.verified_sources(own, root, foreign)
    p.unlink()
    p.write_text("restored")
    foreign.write_text('{"files": []}')
    with pytest.raises(ValueError, match="script missing or outside"):
        coordinator.verified_sources(own, root, foreign)


def test_child_source_cwd_and_resume_keep_existing_data_identity(tmp_path, monkeypatch):
    tree, own, root, foreign, jobs, split, argv = setup(tmp_path, monkeypatch)
    calls = []

    def process(command, log, timeout, *, cwd):
        assert cwd == root and command[1] == str(root / "poorstar/cli/data/extract.py")
        assert command[command.index("--source-manifest-sha256") + 1] == file_hash(foreign)
        calls.append(command)
        complete(command, jobs, split)
        return 0

    monkeypatch.setattr(coordinator, "extraction_process", process)
    assert coordinator.main() == 0 and len(calls) == 2
    data_identity = tmp_path / "data/identity.json"
    original = file_hash(data_identity)
    binding, _ = coordinator.read_verified(data_identity)
    assert binding["source_manifest_sha256"] == file_hash(foreign) != file_hash(own)
    report, _ = coordinator.read_verified(tmp_path / "run1/report.json")
    roles = report["coordinator_binding"]["sources"]
    assert roles["extractor_source_manifest_sha256"] == file_hash(foreign)
    assert roles["coordinator_source_manifest_sha256"] == file_hash(own)
    assert not report["formal_bc_ready"] and report["optimizer_steps"] == 0
    argv[argv.index("--run-evidence-dir") + 1] = str(tmp_path / "run2")
    assert coordinator.main() == 0 and len(calls) == 2
    assert original == file_hash(data_identity)  # Verified reuse, no relabel/re-extraction.
    with pytest.raises(FileExistsError):
        coordinator.main()
    shard = next((tmp_path / "data").rglob("one.bc.zip.xz"))
    shard.write_bytes(b"corrupt")
    argv[argv.index("--run-evidence-dir") + 1] = str(tmp_path / "run3")
    with pytest.raises(ValueError, match="shard drift"):
        coordinator.main()
    assert len(calls) == 2


def test_source_change_mid_run_refuses_completion_claim(tmp_path, monkeypatch):
    tree, own, root, foreign, jobs, split, argv = setup(tmp_path, monkeypatch)
    argv[argv.index("--workers") + 1] = "1"

    def process(command, log, timeout, *, cwd):
        complete(command, jobs, split)
        (root / "poorstar/cli/data/extract.py").write_text("changed")
        return 0

    monkeypatch.setattr(coordinator, "extraction_process", process)
    with pytest.raises(ValueError, match="extractor source postflight"):
        coordinator.main()
    assert not (tmp_path / "run1/report.json").exists()


def test_sigint_drains_commits_restores_handler_and_resumes_pending(tmp_path, monkeypatch):
    tree, own, root, foreign, jobs, split, argv = setup(tmp_path, monkeypatch)
    argv[argv.index("--workers") + 1] = "1"
    calls = []
    previous = signal.getsignal(signal.SIGINT)

    def process(command, log, timeout, *, cwd):
        calls.append(command)
        os.kill(os.getpid(), signal.SIGINT)
        complete(command, jobs, split)
        return 0

    monkeypatch.setattr(coordinator, "extraction_process", process)
    assert coordinator.main() == 130
    assert signal.getsignal(signal.SIGINT) == previous
    report, _ = coordinator.read_verified(tmp_path / "run1/report.json")
    assert report["stop_requested"] and not report["complete_engine_extraction"]
    assert report["completed_views"] == 1 and not report["failures"] and len(calls) == 1
    assert len(list((tmp_path / "data").rglob("job_completion.json"))) == 1
    with coordinator.corpus_lock(tmp_path / "data"):
        pass

    def finish(command, log, timeout, *, cwd):
        calls.append(command)
        complete(command, jobs, split)
        return 0

    monkeypatch.setattr(coordinator, "extraction_process", finish)
    argv[argv.index("--run-evidence-dir") + 1] = str(tmp_path / "run2")
    assert coordinator.main() == 0 and len(calls) == 2
    report, _ = coordinator.read_verified(tmp_path / "run2/report.json")
    assert report["completed_views"] == 2 and not report["stop_requested"]


def test_actual_child_uses_bound_directory_and_independent_session(tmp_path):
    with (tmp_path / "child.log").open("w+") as log:
        assert (
            coordinator.extraction_process(
                [
                    sys.executable,
                    "-c",
                    "import os; print(os.getcwd()); print(os.getpid()==os.getsid(0))",
                ],
                log,
                10,
                cwd=tmp_path,
            )
            == 0
        )
        log.seek(0)
        assert log.read().splitlines() == [str(tmp_path), "True"]


def test_stop_handler_restored_on_context_failure():
    previous = signal.getsignal(signal.SIGINT)
    with pytest.raises(RuntimeError):
        with coordinator.graceful_stop() as stop:
            os.kill(os.getpid(), signal.SIGINT)
            assert stop.is_set()
            raise RuntimeError("owned failure")
    assert signal.getsignal(signal.SIGINT) == previous


@pytest.mark.parametrize("option", ["--run-evidence-dir", "--output-dir"])
def test_output_inside_extractor_source_refused(tmp_path, monkeypatch, option):
    tree, own, root, foreign, jobs, split, argv = setup(tmp_path, monkeypatch)
    argv[argv.index(option) + 1] = str(root / "forbidden")
    with pytest.raises(SystemExit):
        coordinator.main()
    assert not (root / "forbidden").exists()
