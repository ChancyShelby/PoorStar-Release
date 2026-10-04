"""Indexed causal BC windows, observation caches and foundation checkpoint state."""

import json
import math
import os
import random
import tempfile
from bisect import bisect_right
from collections import OrderedDict
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import torch

from poorstar.common.integrity import file_hash, payload_hash
from poorstar.data.bc.records import BCShardReader, validate_split


class PolicyObservationCache:
    """LRU of player-visible values only, never learned activations or targets."""

    def __init__(self, maximum_frames=256):
        if type(maximum_frames) is not int or maximum_frames < 1:
            raise ValueError("positive cache bound required")
        self.maximum_frames = maximum_frames
        self.frames = OrderedDict()

    def get(self, frame):
        # Query evidence remains on each frame/target but is never an actor
        # feature. Identical scheduler/body observation wires may safely share
        # their visible values without re-parsing every entity a second time.
        k = frame.key
        key = (k.episode, k.player_id, k.loop, k.observation_sha256, k.catalog_sha256, k.map_sha256)
        if key not in self.frames:
            self.frames[key] = frame.policy_observation()
        self.frames.move_to_end(key)
        while len(self.frames) > self.maximum_frames:
            self.frames.popitem(last=False)
        return self.frames[key]


@dataclass(frozen=True)
class BCRunPlan:
    purpose: str
    seed: int
    total_steps: int
    learning_rate: float = 0.0003
    minimum_lr_ratio: float = 0.1
    warmup_steps: int = 0
    gradient_clip: float = 10.0
    batch_windows: int = 1
    unroll_length: int = 8
    burn_in_length: int = 4
    hidden_size: int = 128
    autoregressive: bool = True
    supervision_policy: str = "STRICT_WHOLE_BODY_V1"
    feature_batch_size: int = 1
    runtime_mode: str = "DETERMINISTIC_V1"

    def __post_init__(self):
        if self.purpose not in ("ENGINEERING", "FORMAL"):
            raise ValueError("explicit purpose required")
        if type(self.autoregressive) is not bool:
            raise ValueError("explicit decoder arm required")
        if self.runtime_mode != "DETERMINISTIC_V1":
            raise ValueError("unknown BC runtime mode")
        from poorstar.data.bc.supervision import POLICIES

        if self.supervision_policy not in POLICIES:
            raise ValueError("unknown supervision policy")
        for k in (
            "seed",
            "total_steps",
            "warmup_steps",
            "batch_windows",
            "unroll_length",
            "burn_in_length",
            "hidden_size",
            "feature_batch_size",
        ):
            if type(getattr(self, k)) is not int or getattr(self, k) < 0:
                raise ValueError("integer run bounds required")
        if (
            min(
                self.total_steps,
                self.batch_windows,
                self.unroll_length,
                self.hidden_size,
                self.feature_batch_size,
            )
            < 1
        ):
            raise ValueError("positive run bounds required")
        if self.warmup_steps >= self.total_steps:
            raise ValueError("warmup must leave at least one nonwarmup step")
        if any(
            not math.isfinite(v)
            for v in (self.learning_rate, self.minimum_lr_ratio, self.gradient_clip)
        ):
            raise ValueError("finite optimizer bounds required")
        if self.learning_rate <= 0 or self.gradient_clip <= 0 or not 0 < self.minimum_lr_ratio <= 1:
            raise ValueError("invalid optimizer bounds")

    @property
    def sha256(self):
        return payload_hash(asdict(self))

    def lr_at(self, completed_steps):
        if not 0 <= completed_steps < self.total_steps:
            raise ValueError("optimizer budget exhausted")
        if completed_steps < self.warmup_steps:
            return self.learning_rate * (completed_steps + 1) / self.warmup_steps
        span = max(1, self.total_steps - self.warmup_steps - 1)
        progress = (completed_steps - self.warmup_steps) / span
        return self.learning_rate * (
            self.minimum_lr_ratio
            + (1 - self.minimum_lr_ratio) * (1 + math.cos(math.pi * progress)) / 2
        )


class WindowOrder:
    """Stateless epoch shuffle, recoverable without consuming global RNG."""

    def __init__(self, windows, seed):
        if type(windows) is not int or windows < 1:
            raise ValueError("nonempty window census required")
        self.windows = windows
        self.seed = seed
        self.cursor = 0
        self._cached_epoch = None
        self._indices = None

    def take(self, count):
        if type(count) is not int or count < 1:
            raise ValueError("positive batch required")
        values = []
        for _ in range(count):
            epoch, offset = divmod(self.cursor, self.windows)
            if self._cached_epoch != epoch:
                self._indices = list(range(self.windows))
                random.Random(f"{self.seed}:{epoch}").shuffle(self._indices)
                self._cached_epoch = epoch
            values.append(self._indices[offset])
            self.cursor += 1
        return values


class BCWindowDataset:
    """Validate metadata once; open at most a few compressed archives on demand.

    Original rows are not loaded into RAM during indexing. Burn-in is reconstructed
    across shard boundaries, never across replay/player/episode or a clock gap.
    """

    def __init__(
        self,
        build_report,
        split_manifest,
        *,
        unroll_length=8,
        burn_in_length=4,
        allowed_split="train",
        maximum_open_shards=2,
    ):
        if allowed_split not in ("train", "development", "test"):
            raise ValueError("explicit split required")
        if unroll_length < 1 or burn_in_length < 0 or maximum_open_shards < 1:
            raise ValueError("invalid input scheduling bounds")
        self.split = json.loads(Path(split_manifest).read_text())
        validate_split(self.split)
        self.report = json.loads(Path(build_report).read_text())
        self.build_digest = self.report.pop("content_digest")
        if payload_hash(self.report) != self.build_digest:
            raise ValueError("build report hash drift")
        if self.report["split_sha256"] != self.split["sha256"]:
            raise ValueError("build/split drift")
        self.allowed_split = allowed_split
        self.maximum_open_shards = maximum_open_shards
        self.entries = []
        self.windows = []
        self.open = OrderedDict()
        self.rows = 0
        self.shard_offsets = []
        seen_shards = set()
        seen_episodes = set()
        episode = None
        next_loop = None
        episode_start = 0
        active_start = 0
        self.catalog = None
        self.profile = None
        self.maximum_delay = None
        for entry in self.report["shards"]:
            if Path(entry["path"]).parent.name != allowed_split:
                continue
            if entry["sha256"] in seen_shards:
                raise ValueError("duplicate shard in build report")
            seen_shards.add(entry["sha256"])
            number = len(self.entries)
            self.entries.append(entry)
            self.shard_offsets.append(self.rows)
            reader = self._reader(number)
            if self.catalog is None:
                self.catalog = reader.catalog
                self.profile = reader.profile
                self.maximum_delay = reader.index["maximum_delay"]
            elif (
                self.profile.sha256 != reader.profile.sha256
                or self.maximum_delay != reader.index["maximum_delay"]
            ):
                raise ValueError("mixed action profiles or scheduler horizons")
            if entry["decisions"] != len(reader.index["decisions"]):
                raise ValueError("shard row denominator drift")
            for index, record in enumerate(reader.index["decisions"]):
                k = record["scheduler"]["original_frame_key"]
                identity = (record["replay_sha256"], k["player_id"], k["episode"])
                if self.split["assignments"].get(identity[0]) != allowed_split:
                    raise ValueError("split leaked in row metadata")
                if not record["valid"] or record["burn_in"] or not record["loss"]:
                    raise ValueError("original archive masks changed")
                if identity != episode:
                    if self.rows > active_start:
                        self.windows.append(
                            (
                                max(episode_start, active_start - burn_in_length),
                                active_start,
                                self.rows,
                            )
                        )
                    if identity in seen_episodes:
                        raise ValueError("duplicate/discontiguous original episode")
                    seen_episodes.add(identity)
                    if k["loop"] != 0:
                        raise ValueError("complete episode must start at zero")
                    episode = identity
                    next_loop = None
                    episode_start = self.rows
                    active_start = self.rows
                if next_loop is not None and next_loop != k["loop"]:
                    raise ValueError("dataset clock gap")
                next_loop = k["loop"] + record["delay"] + (record["decision"] == "ACT")
                self.rows += 1
                if self.rows - active_start == unroll_length:
                    self.windows.append(
                        (max(episode_start, active_start - burn_in_length), active_start, self.rows)
                    )
                    active_start = self.rows
        if self.rows > active_start:
            self.windows.append(
                (max(episode_start, active_start - burn_in_length), active_start, self.rows)
            )
        if not self.windows:
            raise ValueError("no registered windows for requested split")
        self.sha256 = payload_hash(
            dict(
                build=self.build_digest,
                split=self.split["sha256"],
                allowed_split=allowed_split,
                windows=self.windows,
                unroll_length=unroll_length,
                burn_in_length=burn_in_length,
            )
        )

    def _reader(self, number):
        if number not in self.open:
            e = self.entries[number]
            self.open[number] = BCShardReader(
                Path(e["path"]),
                sha256=e["sha256"],
                split_manifest=self.split,
                allowed_split=self.allowed_split,
            )
        self.open.move_to_end(number)
        while len(self.open) > self.maximum_open_shards:
            _, reader = self.open.popitem(last=False)
            reader.close()
        return self.open[number]

    def window(self, index):
        start, active, end = self.windows[index]
        rows = []
        for ordinal in range(start, end):
            shard = bisect_right(self.shard_offsets, ordinal) - 1
            row = self._reader(shard).row_at(ordinal - self.shard_offsets[shard])
            rows.append(replace(row, burn_in=True, loss=False) if ordinal < active else row)
        return tuple(rows)

    def close(self):
        for reader in self.open.values():
            reader.close()
        self.open.clear()


def freeze_binding(trainer, plan, *, source_sha256, dataset_sha256):
    return dict(
        plan_sha256=plan.sha256,
        source_sha256=source_sha256,
        dataset_sha256=dataset_sha256,
        model_config_sha256=trainer.model.config.sha256,
        profile_sha256=trainer.profile.sha256,
    )


def save_checkpoint(path, trainer, order, binding, *, aggregate=None):
    """Commit complete model/optimizer/RNG/cursor state by an atomic directory rename."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".checkpoint-", dir=path.parent))
    from poorstar.training.runtime import runtime_profile

    state = dict(
        model=trainer.model.state_dict(),
        optimizer=trainer.optimizer.state_dict(),
        steps=trainer.steps,
        gradient_clip=trainer.gradient_clip,
        training=trainer.model.training,
        order=dict(windows=order.windows, seed=order.seed, cursor=order.cursor),
        python_rng=random.getstate(),
        numpy_rng=np.random.get_state(),
        torch_rng=torch.get_rng_state(),
        cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        run_aggregate=aggregate.state_dict() if aggregate is not None else None,
        supervision_policy=trainer.supervision_policy,
        feature_batch_size=trainer.feature_batch_size,
        runtime_profile=runtime_profile(trainer.model.device),
    )
    torch.save(state, temporary / "state.pt")
    report = dict(
        schema="poorstar_c3_bc_checkpoint_v1",
        binding=binding,
        state_sha256=file_hash(temporary / "state.pt"),
        state_bytes=(temporary / "state.pt").stat().st_size,
        steps=trainer.steps,
    )
    report["content_digest"] = payload_hash(report)
    (temporary / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    for name in ("state.pt", "report.json"):
        with (temporary / name).open("rb") as f:
            os.fsync(f.fileno())
    os.rename(temporary, path)
    return report


def load_checkpoint(path, trainer, order, binding, *, aggregate=None):
    path = Path(path)
    report = json.loads((path / "report.json").read_text())
    digest = report.pop("content_digest")
    if payload_hash(report) != digest or report["schema"] != "poorstar_c3_bc_checkpoint_v1":
        raise ValueError("checkpoint report drift")
    if report["binding"] != binding:
        raise ValueError("checkpoint source/data/plan/profile/config drift")
    if (
        file_hash(path / "state.pt") != report["state_sha256"]
        or (path / "state.pt").stat().st_size != report["state_bytes"]
    ):
        raise ValueError("checkpoint state corruption")
    state = torch.load(path / "state.pt", map_location="cpu", weights_only=False)
    from poorstar.training.runtime import runtime_profile

    if state.get("runtime_profile") != runtime_profile(trainer.model.device):
        raise ValueError("checkpoint runtime drift; exact resume requires the same backend")
    if state.get("supervision_policy", "STRICT_WHOLE_BODY_V1") != trainer.supervision_policy:
        raise ValueError("checkpoint supervision policy drift")
    if state.get("feature_batch_size", 1) != trainer.feature_batch_size:
        raise ValueError("checkpoint feature batching drift")
    if state["steps"] != report["steps"] or state["gradient_clip"] != trainer.gradient_clip:
        raise ValueError("checkpoint training bounds drift")
    if (state["order"]["windows"], state["order"]["seed"]) != (order.windows, order.seed):
        raise ValueError("sampler identity drift")
    if aggregate is not None:
        if state.get("run_aggregate") is None:
            raise ValueError("whole-run aggregate missing from resumed checkpoint")
        aggregate.load_state_dict(state["run_aggregate"])
    trainer.model.load_state_dict(state["model"])
    trainer.optimizer.load_state_dict(state["optimizer"])
    trainer.steps = state["steps"]
    trainer.model.train(state["training"])
    order.cursor = state["order"]["cursor"]
    random.setstate(state["python_rng"])
    np.random.set_state(state["numpy_rng"])
    torch.set_rng_state(state["torch_rng"])
    if state["cuda_rng"]:
        if len(state["cuda_rng"]) != torch.cuda.device_count():
            raise ValueError("CUDA topology changed for exact resume")
        torch.cuda.set_rng_state_all(state["cuda_rng"])
    return dict(report, content_digest=digest)


def load_policy_checkpoint(path, model, binding, *, supervision_policy, feature_batch_size):
    """Verified actor-only read; evaluation cannot restore optimizer or training RNGs."""
    path = Path(path)
    report = json.loads((path / "report.json").read_text())
    digest = report.pop("content_digest")
    if payload_hash(report) != digest or report["schema"] != "poorstar_c3_bc_checkpoint_v1":
        raise ValueError("checkpoint report drift")
    if report["binding"] != binding:
        raise ValueError("checkpoint source/data/plan/profile/config drift")
    if (
        file_hash(path / "state.pt") != report["state_sha256"]
        or (path / "state.pt").stat().st_size != report["state_bytes"]
    ):
        raise ValueError("checkpoint state corruption")
    state = torch.load(path / "state.pt", map_location="cpu", weights_only=False)
    if state.get("supervision_policy", "STRICT_WHOLE_BODY_V1") != supervision_policy:
        raise ValueError("checkpoint supervision policy drift")
    if state.get("feature_batch_size", 1) != feature_batch_size:
        raise ValueError("checkpoint feature batching drift")
    if state["steps"] != report["steps"]:
        raise ValueError("checkpoint step drift")
    model.load_state_dict(state["model"])
    model.eval()
    return dict(report, content_digest=digest)
