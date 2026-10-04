"""Hierarchy: interventions."""

import hashlib
import json
from dataclasses import asdict, dataclass

from poorstar.hierarchy.state import ExternalOverride


@dataclass(frozen=True)
class PreferenceIntervention:
    loop: int
    preference: tuple[float, ...]
    commitment_horizon: int
    confidence: float = 0.5


class ExternalPreferenceSchedule:
    def __init__(self, config, *, episode, interventions):
        if config.arm != "persistent_intent" or config.manager_mode != "external":
            raise ValueError("explicit external-only Preference arm required")
        interventions = tuple(interventions)
        if not interventions or interventions[0].loop != 0 or not episode:
            raise ValueError("predeclared external schedule requires its loop-zero profile")
        previous = -1
        for item in interventions:
            if (
                not isinstance(item, PreferenceIntervention)
                or type(item.loop) is not int
                or item.loop <= previous
            ):
                raise ValueError("ordered unique external intervention clocks required")
            ExternalOverride(
                item.preference, item.commitment_horizon, item.confidence, "validate"
            ).validate(config)
            if (
                config.commitment_mode == "fixed_dwell"
                and item.commitment_horizon != config.fixed_commitment_horizon
            ):
                raise ValueError(
                    "external fixed-dwell comparison requires the same declared fixed horizon"
                )
            previous = item.loop
        self.config = config
        self.episode = episode
        self.interventions = interventions
        self.last_loop = None
        self.last_index = None

    @property
    def sha256(self):
        value = dict(
            config_sha256=self.config.sha256,
            episode=self.episode,
            interventions=[asdict(item) for item in self.interventions],
        )
        return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()

    def override_for(self, loop, state):
        state.validate(self.config)
        if (
            state.episode != self.episode
            or type(loop) is not int
            or loop < 0
            or (self.last_loop is not None and loop <= self.last_loop)
        ):
            raise ValueError("one chronological external schedule call per scheduler required")
        index = max(i for i, item in enumerate(self.interventions) if item.loop <= loop)
        changed = self.last_index != index
        expired = (
            state.intent is None
            or not state.intent.active
            or (
                self.config.commitment_mode == "fixed_dwell"
                and loop - state.intent.start_loop >= state.intent.commitment_horizon
            )
        )
        issue = changed or expired or not self.config.persistence
        self.last_loop = loop
        self.last_index = index
        if not issue:
            return None
        item = self.interventions[index]
        return ExternalOverride(
            item.preference,
            item.commitment_horizon,
            item.confidence,
            f"external_schedule:{self.sha256}:{loop}:{index}",
        )

    def state_dict(self):
        return dict(
            schema_version="poorstar_v3_external_preference_schedule_v1",
            schedule_sha256=self.sha256,
            last_loop=self.last_loop,
            last_index=self.last_index,
        )

    def load_state_dict(self, value):
        if (
            set(value) != set(self.state_dict())
            or value["schema_version"] != self.state_dict()["schema_version"]
            or value["schedule_sha256"] != self.sha256
        ):
            raise ValueError("external Preference schedule recovery identity mismatch")
        loop, index = value["last_loop"], value["last_index"]
        if (loop is None) != (index is None) or (
            loop is not None
            and (
                type(loop) is not int
                or loop < 0
                or type(index) is not int
                or index != max(i for i, item in enumerate(self.interventions) if item.loop <= loop)
            )
        ):
            raise ValueError("external Preference schedule recovery clock mismatch")
        self.last_loop = loop
        self.last_index = index


class ExternalPreferenceProgram(ExternalPreferenceSchedule):
    """Pure renewal function; identity/clock are entirely in the prior state.

    This variant is the actor integration API. Calling it again with the same
    inputs returns the same explicit intervention, enabling exact recovery.
    """

    def override_for(self, loop, state):
        state.validate(self.config)
        if (
            state.episode != self.episode
            or type(loop) is not int
            or loop < 0
            or (state.last_loop is not None and loop < state.last_loop)
        ):
            raise ValueError("external program requires current causal episode/clock")
        index = max(i for i, item in enumerate(self.interventions) if item.loop <= loop)
        item = self.interventions[index]
        intent = state.intent
        if intent is not None and intent.source != "external":
            raise ValueError("external-only program cannot silently adopt a learned Intent")
        if intent is not None:
            previous = max(
                (p for p in self.interventions if p.loop <= intent.start_loop), key=lambda p: p.loop
            )
            if (
                tuple(intent.preference) != tuple(previous.preference)
                or intent.commitment_horizon != previous.commitment_horizon
                or intent.confidence != previous.confidence
            ):
                raise ValueError(
                    "prior external profile is not declared by this intervention program"
                )
        changed = intent is None or item.loop > intent.start_loop
        expired = (
            intent is None
            or not intent.active
            or (
                self.config.commitment_mode == "fixed_dwell"
                and loop - intent.start_loop >= intent.commitment_horizon
            )
        )
        if not (changed or expired or not self.config.persistence):
            return None
        revision = 1 if intent is None else intent.revision + 1
        return ExternalOverride(
            item.preference,
            item.commitment_horizon,
            item.confidence,
            f"external_program:{self.sha256}:{loop}:{revision}:{index}",
        )

    def state_dict(self):
        return dict(
            schema_version="poorstar_v3_external_preference_program_v1", program_sha256=self.sha256
        )

    def load_state_dict(self, value):
        if value != self.state_dict():
            raise ValueError("external Preference program recovery identity mismatch")
