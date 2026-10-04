"""Typed identities, rollout records and numeric optimizer configuration."""

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from typing import Any

import torch


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def is_sha(value):
    return (
        isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)
    )


@dataclass(frozen=True)
class TrainingIdentity:
    experiment: str
    arm: str
    seed: int
    source_sha256: str
    dataset_sha256: str
    split_sha256: str
    scope_sha256: str
    model_config_sha256: str
    profile_sha256: str
    budget_sha256: str
    purpose: str = "ENGINEERING"
    initializer_sha256: str = "0" * 64
    control_sidecar_sha256: str = "0" * 64
    corpus_qualification_sha256: str = "0" * 64
    stage_transition_sha256: str = "0" * 64
    matchup: str = "ZvZ"
    schema: str = "poorstar_v3_training_identity_v1"

    def __post_init__(self):
        if self.purpose not in ("ENGINEERING", "FORMAL") or self.matchup != "ZvZ":
            raise ValueError("declared ZvZ purpose required")
        if not self.experiment or not self.arm or type(self.seed) is not int or self.seed < 0:
            raise ValueError("experiment, arm and nonnegative seed required")
        for name, value in asdict(self).items():
            if name.endswith("_sha256") and not is_sha(value):
                raise ValueError("content hash required: " + name)

    @property
    def sha256(self):
        return digest(asdict(self))


@dataclass(frozen=True)
class PPOConfig:
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    clip_ratio: float = 0.2
    value_clip: float = 0.2
    value_coefficient: float = 0.5
    entropy_coefficient: float = 0.0
    gradient_clip: float = 0.5
    max_kl: float = 0.03
    gamma_per_loop: float = 1.0
    lambda_per_loop: float = 0.999
    normalize_advantage: bool = True
    epochs: int = 1
    maximum_policy_lag: int = 0
    maximum_updates: int = 1
    maximum_environment_loops: int = 1
    maximum_sequence_steps: int = 128
    maximum_batch_sequences: int = 1

    def __post_init__(self):
        values = asdict(self)
        if any(not math.isfinite(v) for v in values.values() if isinstance(v, float)):
            raise ValueError("finite optimizer coefficients required")
        if self.learning_rate <= 0 or self.gradient_clip <= 0 or self.max_kl <= 0:
            raise ValueError("positive optimizer bounds required")
        if not 0 < self.clip_ratio < 1 or not 0 < self.value_clip < 1:
            raise ValueError("clip coefficients outside (0,1)")
        if min(self.value_coefficient, self.entropy_coefficient, self.weight_decay) < 0:
            raise ValueError("nonnegative loss weights required")
        if not 0 < self.gamma_per_loop <= 1 or not 0 < self.lambda_per_loop <= 1:
            raise ValueError("per-game-loop discounts outside (0,1]")
        for name in (
            "epochs",
            "maximum_updates",
            "maximum_environment_loops",
            "maximum_sequence_steps",
            "maximum_batch_sequences",
        ):
            if type(values[name]) is not int or values[name] < 1:
                raise ValueError("positive integer optimizer/frame budgets required")
        if type(self.maximum_policy_lag) is not int or self.maximum_policy_lag < 0:
            raise ValueError("nonnegative integer policy lag required")
        if type(self.normalize_advantage) is not bool:
            raise ValueError("explicit normalization required")

    @property
    def sha256(self):
        return digest(asdict(self))


@dataclass(frozen=True)
class RolloutStep:
    episode_id: str
    index: int
    game_loop: int
    next_game_loop: int
    policy_version: int
    checkpoint_sha256: str
    observation_sha256: str
    mask_sha256: str
    payload: Any = field(repr=False)
    old_log_prob: float = 0.0
    old_manager_log_prob: float = 0.0
    old_value: float = 0.0
    old_manager_value: float = 0.0
    next_value: float = 0.0
    next_manager_value: float = 0.0
    reward: float = 0.0
    reward_loop: int | None = None
    terminated: bool = False
    truncated: bool = False
    boundary_reason: str = ""
    valid: bool = True
    burn_in: bool = False
    loss: bool = True
    manager_decision: bool = False
    provenance: str = "learner"
    behavior: str = "sampled"
    hidden_fallback_count: int = 0

    @property
    def duration(self):
        return self.next_game_loop - self.game_loop

    def validate(self):
        for name in ("index", "game_loop", "next_game_loop", "policy_version"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError("nonnegative integer rollout clocks/versions required")
        if not self.episode_id or not all(
            is_sha(getattr(self, n))
            for n in ("checkpoint_sha256", "observation_sha256", "mask_sha256")
        ):
            raise ValueError("episode and content-bound behavior/observation/masks required")
        for name in ("valid", "burn_in", "loss", "manager_decision", "terminated", "truncated"):
            if type(getattr(self, name)) is not bool:
                raise ValueError("explicit rollout booleans required")
        if self.terminated and self.truncated:
            raise ValueError("terminal and truncation are mutually exclusive")
        if (not self.valid or self.burn_in) and self.loss:
            raise ValueError("padding/burn-in cannot contribute loss")
        if self.valid and self.duration <= 0:
            raise ValueError("valid decision must advance actual game loops")
        if self.provenance not in ("learner", "evaluator") or self.behavior not in (
            "sampled",
            "deterministic",
        ):
            raise ValueError("explicit trajectory provenance/behavior required")
        if self.hidden_fallback_count != 0:
            raise ValueError("hidden executor replacement forbidden")
        for name in (
            "old_log_prob",
            "old_manager_log_prob",
            "old_value",
            "old_manager_value",
            "next_value",
            "next_manager_value",
            "reward",
        ):
            if not math.isfinite(getattr(self, name)):
                raise ValueError("nonfinite rollout field: " + name)
        # This infrastructure deliberately admits outcome rewards only.
        if self.terminated:
            if self.boundary_reason != "engine_terminal" or self.reward not in (-1.0, 0.0, 1.0):
                raise ValueError("resolved natural outcome required")
            if self.next_value != 0 or self.next_manager_value != 0:
                raise ValueError("natural terminal bootstrap must be zero")
        elif self.reward != 0:
            raise ValueError("intermediate/shaped reward not authorized")
        if self.truncated:
            from poorstar.training.boundaries import TRUNCATION_REASONS

            if self.boundary_reason not in TRUNCATION_REASONS | {"unroll_cut", "resident_step_cap"}:
                raise ValueError("declared infrastructure boundary required")
        elif not self.terminated and self.boundary_reason:
            raise ValueError("boundary reason without a boundary")
        if self.reward != 0 and (
            type(self.reward_loop) is not int
            or not self.game_loop <= self.reward_loop <= self.next_game_loop
        ):
            raise ValueError("outcome reward needs its measured game loop")


@dataclass(frozen=True)
class RolloutSequence:
    steps: tuple[RolloutStep, ...]
    identity_sha256: str
    initial_hidden: Any = field(default=None, repr=False)
    starts_at_episode_boundary: bool = True

    def validate(self, identity, policy_version, maximum_policy_lag, *, learning=True):
        if self.identity_sha256 != identity.sha256 or not self.steps:
            raise ValueError("rollout identity drift or empty sequence")
        real = [s for s in self.steps if s.valid]
        if not real:
            raise ValueError("no valid rollout decisions")
        if self.starts_at_episode_boundary:
            if self.initial_hidden is not None or real[0].index != 0:
                raise ValueError("episode start must reset memory")
        elif self.initial_hidden is None and not real[0].burn_in:
            raise ValueError("window needs recorded recurrent state or causal burn-in")
        if len({(s.policy_version, s.checkpoint_sha256) for s in real}) != 1:
            raise ValueError("policy checkpoint changes inside recurrent unroll")
        padding_seen = False
        for i, s in enumerate(self.steps):
            s.validate()
            if not s.valid:
                padding_seen = True
                continue
            if padding_seen:
                raise ValueError("padding can only follow real decisions")
            lag = policy_version - s.policy_version
            if not 0 <= lag <= maximum_policy_lag:
                raise ValueError("behavior policy is stale or from the future")
            if learning and (s.provenance != "learner" or s.behavior != "sampled"):
                raise ValueError("evaluation/argmax trajectory cannot train policy")
            if i:
                previous = self.steps[i - 1]
                if previous.terminated or previous.truncated:
                    raise ValueError("unroll crossed episode boundary")
                if (
                    s.episode_id != previous.episode_id
                    or s.index != previous.index + 1
                    or s.game_loop != previous.next_game_loop
                ):
                    raise ValueError("recurrent unroll episode/clock gap")
                if (
                    previous.next_value != s.old_value
                    or previous.next_manager_value != s.old_manager_value
                ):
                    raise ValueError("behavior value bootstrap discontinuity")
        if not real[-1].terminated and not real[-1].truncated:
            raise ValueError("unroll end requires measured truncation or natural terminal")


@dataclass
class StepScore:
    log_prob: torch.Tensor
    entropy: torch.Tensor
    value: torch.Tensor
    next_hidden: Any
    manager_log_prob: torch.Tensor
    manager_entropy: torch.Tensor
    manager_value: torch.Tensor
    observation_sha256: str
    mask_sha256: str

    def validate(self, step):
        if (
            self.observation_sha256 != step.observation_sha256
            or self.mask_sha256 != step.mask_sha256
        ):
            raise ValueError("teacher/actor observation or action domain drift")
        for name in (
            "log_prob",
            "entropy",
            "value",
            "manager_log_prob",
            "manager_entropy",
            "manager_value",
        ):
            value = getattr(self, name)
            if (
                not isinstance(value, torch.Tensor)
                or value.numel() != 1
                or not torch.isfinite(value).all()
            ):
                raise ValueError("finite differentiable scalar required: " + name)
