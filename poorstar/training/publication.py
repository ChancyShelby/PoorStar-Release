"""Training: publication."""

import json
from dataclasses import asdict
from pathlib import Path

from poorstar.training.checkpoint import file_hash, save_checkpoint
from poorstar.training.contracts import digest
from poorstar.training.export import export_actor


def publish_candidate(root, learner, *, sampler_state=None):
    root = Path(root)
    checkpoint = root / ("checkpoint_" + str(learner.updates).zfill(8))
    manifest = save_checkpoint(checkpoint, learner, sampler_state=sampler_state)
    actor = export_actor(
        root / ("actor_version_" + str(learner.policy_version).zfill(8) + ".pt"), learner
    )
    result = dict(
        schema="poorstar_v3_checkpoint_candidate_v1",
        identity=asdict(learner.identity),
        identity_sha256=learner.identity.sha256,
        stage=learner.phase,
        policy_version=learner.policy_version,
        optimizer_steps=learner.optimizer_steps,
        updates=learner.updates,
        parameter_sha256=learner.parameter_sha256,
        actor=actor,
        checkpoint_path=str(checkpoint),
        checkpoint_manifest_sha256=file_hash(checkpoint / "manifest.json"),
        checkpoint_state_sha256=manifest["state_sha256"],
        evaluated=False,
        selected=False,
    )
    result["content_digest"] = digest(result)
    path = root / ("candidate_" + str(learner.policy_version).zfill(8) + ".json")
    with path.open("x") as output:
        json.dump(result, output, sort_keys=True, indent=2)
        output.write("\n")
    return dict(path=str(path), sha256=file_hash(path), candidate=result)


def publish_candidate_index(path, identity, stage, candidates):
    candidates = tuple(candidates)
    versions = [c["candidate"]["policy_version"] for c in candidates]
    if not candidates or versions != sorted(set(versions)):
        raise ValueError("candidate publications must be unique chronological policy versions")
    for item in candidates:
        record = item["candidate"]
        if (
            record["identity_sha256"] != identity.sha256
            or record["stage"] != stage
            or file_hash(item["path"]) != item["sha256"]
            or file_hash(record["actor"]["path"]) != record["actor"]["sha256"]
        ):
            raise ValueError("checkpoint candidate identity/content drift")
    document = dict(
        schema="poorstar_v3_checkpoint_candidate_index_v1",
        identity_sha256=identity.sha256,
        stage=stage,
        candidates=list(candidates),
        capability_evaluated=False,
        selected=False,
    )
    document["content_digest"] = digest(document)
    with Path(path).open("x") as output:
        json.dump(document, output, sort_keys=True, indent=2)
        output.write("\n")
    return dict(path=str(path), sha256=file_hash(path), content_digest=document["content_digest"])
