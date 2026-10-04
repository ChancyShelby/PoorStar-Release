"""Persist optimizer state, counters and random state for exact recovery."""

import copy
import hashlib
import json
import os
import random
import shutil
import tempfile
from pathlib import Path

import numpy as np
import torch

from poorstar.training.contracts import digest


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def capture_rng():
    state = dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state())
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng(state):
    if ("cuda" in state) != torch.cuda.is_available():
        raise ValueError("RNG recovery device family drift")
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state:
        if len(state["cuda"]) != torch.cuda.device_count():
            raise ValueError("RNG recovery CUDA topology drift")
        torch.cuda.set_rng_state_all(state["cuda"])


def runtime(model):
    from poorstar.training.runtime import runtime_profile

    return dict(
        runtime_profile(str(next(model.parameters()).device)), torch_threads=torch.get_num_threads()
    )


def save_checkpoint(path, learner, *, sampler_state=None, actor_states=None):
    path = Path(path)
    if path.exists():
        raise FileExistsError("checkpoint is immutable: " + str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".training-", dir=path.parent))
    try:
        payload = dict(
            schema="poorstar_v3_training_checkpoint_v1",
            learner=learner.state_dict(),
            rng=capture_rng(),
            sampler_state=sampler_state,
            actor_states=actor_states,
            runtime=runtime(learner.model),
        )
        with (temporary / "state.pt").open("wb") as f:
            torch.save(payload, f)
            f.flush()
            os.fsync(f.fileno())
        manifest = dict(
            schema=payload["schema"],
            identity_sha256=learner.identity.sha256,
            state_sha256=file_hash(temporary / "state.pt"),
            parameter_sha256=learner.parameter_sha256,
            runtime=payload["runtime"],
            updates=learner.updates,
        )
        manifest["content_digest"] = digest(manifest)
        with (temporary / "manifest.json").open("x") as f:
            json.dump(manifest, f, sort_keys=True, indent=2, allow_nan=False)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        # Directory rename exposes both members together; old attempts survive.
        os.rename(temporary, path)
        return manifest
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def load_checkpoint(path, learner):
    path = Path(path)
    manifest = json.loads((path / "manifest.json").read_text())
    check = dict(manifest)
    expected = check.pop("content_digest")
    if digest(check) != expected or manifest["state_sha256"] != file_hash(path / "state.pt"):
        raise ValueError("checkpoint corruption or manifest drift")
    if manifest["identity_sha256"] != learner.identity.sha256 or manifest["runtime"] != runtime(
        learner.model
    ):
        raise ValueError("exact resume identity/runtime drift")
    state = torch.load(
        path / "state.pt", map_location=next(learner.model.parameters()).device, weights_only=False
    )
    if state["schema"] != manifest["schema"] or state["runtime"] != manifest["runtime"]:
        raise ValueError("checkpoint payload identity drift")
    before, rng = copy.deepcopy(learner.state_dict()), capture_rng()
    try:
        learner.load_state_dict(state["learner"])
        if (
            learner.parameter_sha256 != manifest["parameter_sha256"]
            or learner.updates != manifest["updates"]
        ):
            raise ValueError("checkpoint parameter/counter corruption")
        # CPU RNG byte tensors must not follow map_location to a CUDA device.
        state["rng"]["torch"] = state["rng"]["torch"].cpu()
        if "cuda" in state["rng"]:
            state["rng"]["cuda"] = [v.cpu() for v in state["rng"]["cuda"]]
        restore_rng(state["rng"])
    except BaseException:
        learner.stage_transition = before.get("stage_transition")
        learner.load_state_dict(before)
        restore_rng(rng)
        raise
    return dict(
        sampler_state=state["sampler_state"], actor_states=state["actor_states"], manifest=manifest
    )
