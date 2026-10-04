"""Training: export."""

import os
from dataclasses import asdict
from pathlib import Path

import torch

from poorstar.common.parameters import parameter_digest
from poorstar.training.checkpoint import file_hash
from poorstar.training.config import model_config_sha256


def export_actor(path, learner):
    from poorstar.training.stages import cumulative_counters

    path = Path(path)
    dataset_window = (
        dict(
            unroll_length=learner.config.unroll_length, burn_in_length=learner.config.burn_in_length
        )
        if learner.phase == "BC"
        else None
        if getattr(learner, "stage_transition", None) is None
        else learner.stage_transition.get("dataset_window")
    )
    with path.open("xb") as f:
        torch.save(
            dict(
                schema="poorstar_v3_hierarchical_actor_v1",
                identity=asdict(learner.identity),
                bc_config=asdict(learner.model.config),
                hierarchy_config=asdict(learner.model.hierarchy_config),
                policy_version=learner.policy_version,
                parameter_sha256=learner.parameter_sha256,
                training_phase=learner.phase,
                stage_counters=dict(
                    updates=learner.updates,
                    optimizer_steps=learner.optimizer_steps,
                    environment_loops=learner.environment_loops,
                ),
                cumulative_counters=cumulative_counters(learner),
                dataset_window=dataset_window,
                stage_transition=getattr(learner, "stage_transition", None),
                parameters={
                    n: v.detach().cpu().clone() for n, v in learner.model.state_dict().items()
                },
            ),
            f,
        )
        f.flush()
        os.fsync(f.fileno())
    return dict(path=str(path), sha256=file_hash(path), parameter_sha256=learner.parameter_sha256)


def load_actor(path, *, sha256, device="cpu"):
    from poorstar.hierarchy import HierarchicalPolicy, HierarchyConfig
    from poorstar.models.policy import BCModelConfig
    from poorstar.training.contracts import TrainingIdentity

    if file_hash(path) != sha256:
        raise ValueError("actor file hash drift")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema") != "poorstar_v3_hierarchical_actor_v1":
        raise ValueError("declared optimizer-free actor export required")
    identity = TrainingIdentity(**payload["identity"])
    low = BCModelConfig(**payload["bc_config"])
    high = HierarchyConfig(**payload["hierarchy_config"])
    if model_config_sha256(low, high) != identity.model_config_sha256:
        raise ValueError("actor architecture identity drift")
    if parameter_digest(payload["parameters"]) != payload["parameter_sha256"]:
        raise ValueError("actor parameter corruption")
    # Parameters are replaced in full. Initializing their temporary values must
    # not move an existing learner/actor's sampling stream as a loading sideeffect.
    # Construction is on CPU; .to(device) performs no stochastic operation.
    with torch.random.fork_rng(devices=[]):
        model = HierarchicalPolicy(low, high).to(device)
    model.load_state_dict(payload["parameters"])
    model.eval()
    return model, identity, payload
