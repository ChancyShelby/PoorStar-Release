"""League: fixture."""

from pathlib import Path

from poorstar.league.contracts import WorkerResult
from poorstar.league.ownership import verify_worker_lease, write_document
from poorstar.training.checkpoint import file_hash
from poorstar.training.contracts import TrainingIdentity


class FixtureDispatcher:
    def __call__(self, request):
        verify_worker_lease(request)
        directory = Path(request.output_dir)
        caveat = "SCRIPTED_MECHANISM_FIXTURE_NOT_GAMEPLAY_OR_OPTIMIZATION"
        if request.kind == "actor":
            job = request.payload["job"]
            report = dict(
                provenance=job["provenance"],
                environment_loops=3,
                terminated=True,
                truncated=False,
                reward=0.0,
                fixture=caveat,
                SC2_engine_launches=0,
                optimizer_updates_performed=0,
            )
            write_document(directory / "actor_report.json", report, immutable=True)
            artifacts = dict(
                report=dict(
                    path=str(directory / "actor_report.json"),
                    sha256=file_hash(directory / "actor_report.json"),
                )
            )
            if job["provenance"] == "learner":
                write_document(
                    directory / "rollout_index.json",
                    dict(
                        job_id=job["job_id"],
                        identity_sha256=TrainingIdentity(**request.identity).sha256,
                        physical_environment_loops=3,
                        shards=[],
                        update_groups=[],
                        fixture=caveat,
                    ),
                    immutable=True,
                )
                artifacts["rollout_index"] = dict(
                    path=str(directory / "rollout_index.json"),
                    sha256=file_hash(directory / "rollout_index.json"),
                )
            frames = 3
        else:
            previous = request.payload["previous_snapshot"]
            (directory / "actor_fixture.bin").write_bytes(
                Path(previous["weights_path"]).read_bytes() + request.attempt_id.encode()
            )
            write_document(
                directory / "training_report.json",
                dict(
                    identity_sha256=TrainingIdentity(**request.identity).sha256,
                    updates=request.payload["requested_updates"],
                    policy_version=request.payload["expected_next_policy_version"],
                    fixture=caveat,
                    SC2_engine_launches=0,
                    optimizer_updates_performed=0,
                ),
                immutable=True,
            )
            artifacts = dict(
                training_report=dict(
                    path=str(directory / "training_report.json"),
                    sha256=file_hash(directory / "training_report.json"),
                ),
                actor_export=dict(
                    path=str(directory / "actor_fixture.bin"),
                    sha256=file_hash(directory / "actor_fixture.bin"),
                ),
            )
            frames = 0
        return WorkerResult(
            request.sha256,
            request.operation_id,
            request.attempt_id,
            request.claim_token,
            request.kind,
            "COMPLETE",
            1,
            frames,
            True,
            artifacts,
        )
