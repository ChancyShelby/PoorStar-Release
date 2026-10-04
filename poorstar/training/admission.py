"""Training: admission."""

from dataclasses import asdict
from pathlib import Path

from poorstar.league.ownership import read_document
from poorstar.training.checkpoint import file_hash
from poorstar.training.contracts import digest


def validate_rollout_admission(path, identity, *, _seen=None):
    path = Path(path).resolve()
    seen = set() if _seen is None else _seen
    if path in seen:
        raise ValueError("cyclic collection receipt provenance")
    seen = seen | {path}
    index = read_document(path)
    if index.get("identity_sha256") != identity.sha256:
        raise ValueError("rollout admission identity drift")
    shards = index.get("shards", [])
    if not shards:
        return index  # zero-update bootstrap has no experience
    if identity.purpose == "FORMAL" and index.get("fixture"):
        raise ValueError("protocol fixtures cannot enter formal rollout learning")
    if (
        "collection_id" in index
        or "collection_plan_sha256" in index
        or "stage_report_path" in index
    ):
        report_path = Path(index.get("stage_report_path", "")).resolve()
        if report_path != path.parent / "stage_report.json":
            raise ValueError("stage collection requires its final sibling report")
        report = read_document(report_path)
        plan = read_document(path.parent / "stage_plan.json")
        artifact = report.get("artifacts", {}).get("rollout_index", {})
        binding = report.get("binding", {})
        if (
            report.get("schema") != "poorstar_v3_stage_actor_collection_report_v1"
            or report.get("status") not in ("COMPLETE", "TRUNCATED")
            or report.get("error") is not None
            or report.get("all_owned_sessions_closed") is not True
            or report.get("provenance") != "learner"
            or report.get("identity") != asdict(identity)
            or report.get("identity_sha256") != identity.sha256
            or report.get("plan_sha256") != digest(plan)
            or report.get("plan_sha256") != index.get("collection_plan_sha256")
            or binding.get("collection_id") != index.get("collection_id")
            or plan.get("collection_id") != index.get("collection_id")
            or binding.get("maximum_compute_millis") != plan.get("maximum_compute_millis")
            or type(report.get("compute_millis")) is not int
            or not 0 <= report["compute_millis"] <= plan["maximum_compute_millis"]
            or report.get("environment_loops") != index.get("physical_environment_loops")
            or artifact.get("sha256") != file_hash(path)
            or Path(artifact.get("path", "")).resolve() != path
        ):
            raise ValueError("failed/unbound/over-budget stage collection cannot enter learning")
        for key in ("source_preflight", "source_postflight"):
            evidence = report.get(key) or {}
            if (
                evidence.get("problems") != []
                or evidence.get("manifest_sha256") != identity.source_sha256
            ):
                raise ValueError("stage collection source receipt is unresolved")
        if identity.purpose == "FORMAL" and (
            report.get("fixture_scope") is not None
            or report.get("engine_launch_attempted") is not True
        ):
            raise ValueError("formal stage collection requires actual engine provenance")
    elif "job_id" in index:
        if identity.purpose == "ENGINEERING" and index.get("fixture") == "PROTOCOL_FIXTURE_NO_SC2":
            return index  # existing arithmetic seam; no formal admission
        # The registered worker's result lives above its actor artifact folder.
        result = read_document(path.parent.parent / "worker_result.json")
        request = read_document(path.parent.parent / "request.json")
        artifact = result.get("artifacts", {}).get("rollout_index", {})
        if (
            result.get("status") not in ("COMPLETE", "TRUNCATED")
            or result.get("owned_sessions_closed") is not True
            or result.get("request_sha256") != digest(request)
            or request.get("identity") != asdict(identity)
            or request.get("kind") != "actor"
            or result.get("kind") != "actor"
            or request.get("payload", {}).get("job", {}).get("provenance") != "learner"
            or request.get("payload", {}).get("job", {}).get("job_id") != index["job_id"]
            or type(result.get("compute_millis")) is not int
            or not 0 <= result["compute_millis"] <= request["maximum_compute_millis"]
            or artifact.get("sha256") != file_hash(path)
            or Path(artifact.get("path", "")).resolve() != path
        ):
            raise ValueError("failed/unbound/over-budget owned worker cannot enter learning")
        exit_path = path.parent.parent / "worker_exit.json"
        if exit_path.exists():
            exited = read_document(exit_path)
            if (
                exited.get("returncode") != 0
                or exited.get("signalled_once") is not False
                or exited.get("all_owned_sessions_closed") is not True
                or exited.get("owner", {}).get("request_sha256") != digest(request)
                or type(exited.get("compute_millis")) is not int
                or not 0 <= exited["compute_millis"] <= request["maximum_compute_millis"]
            ):
                raise ValueError("failed/over-budget dispatcher exit cannot enter learning")
        elif identity.purpose == "FORMAL":
            raise ValueError("formal worker lacks final dispatcher closure receipt")
    elif "collection_indices" in index:
        admitted = []
        loops = 0
        for reference in index["collection_indices"]:
            if file_hash(reference["path"]) != reference["sha256"]:
                raise ValueError("collection input receipt hash drift")
            source = validate_rollout_admission(reference["path"], identity, _seen=seen)
            admitted.extend(source["shards"])
            loops += source["physical_environment_loops"]
        if admitted != shards or loops != index.get("physical_environment_loops"):
            raise ValueError("aggregated rollout collection coverage drift")
    elif identity.purpose == "FORMAL":
        raise ValueError("formal rollout lacks successful collection receipts")
    return index
