"""Evaluation: interventions."""

from dataclasses import asdict

from poorstar.hierarchy.interventions import PreferenceIntervention
from poorstar.hierarchy.state import ExternalOverride
from poorstar.training.contracts import digest


class ScheduledPreferenceInterventions:
    def __init__(self, config, *, episode, interventions):
        if config.arm != "persistent_intent" or not episode:
            raise ValueError("Preference intervention requires the declared Intent arm")
        self.config = config
        self.episode = episode
        self.interventions = tuple(interventions)
        if not self.interventions:
            raise ValueError("explicit frozen intervention schedule required")
        previous = -1
        for item in self.interventions:
            if (
                not isinstance(item, PreferenceIntervention)
                or type(item.loop) is not int
                or item.loop <= previous
            ):
                raise ValueError("ordered unique nonnegative intervention clocks required")
            ExternalOverride(
                item.preference, item.commitment_horizon, item.confidence, "validate"
            ).validate(config)
            if (
                config.commitment_mode == "fixed_dwell"
                and item.commitment_horizon != config.fixed_commitment_horizon
            ):
                raise ValueError("intervention cannot change the fixed-dwell comparison horizon")
            previous = item.loop
        self.last_loop = None
        self.last_applied_index = -1

    @property
    def sha256(self):
        return digest(
            dict(
                config_sha256=self.config.sha256,
                episode=self.episode,
                interventions=[asdict(i) for i in self.interventions],
            )
        )

    def override_for(self, loop, state):
        state.validate(self.config)
        if (
            state.episode != self.episode
            or type(loop) is not int
            or loop < 0
            or self.last_loop is not None
            and loop <= self.last_loop
        ):
            raise ValueError(
                "one chronological intervention call per actual scheduler clock required"
            )
        self.last_loop = loop
        eligible = [
            i
            for i, item in enumerate(self.interventions)
            if item.loop <= loop and i > self.last_applied_index
        ]
        if not eligible:
            return None
        if len(eligible) > 1:
            raise ValueError(
                "actor crossed multiple registered intervention cells without observing them"
            )
        index = eligible[0]
        self.last_applied_index = index
        item = self.interventions[index]
        return ExternalOverride(
            item.preference,
            item.commitment_horizon,
            item.confidence,
            f"registered_intervention:{self.sha256}:{index}:{loop}",
        )

    def state_dict(self):
        return dict(
            schema="poorstar_v3_registered_intervention_clock_v1",
            program_sha256=self.sha256,
            last_loop=self.last_loop,
            last_applied_index=self.last_applied_index,
        )

    def load_state_dict(self, value):
        expected = self.state_dict()
        if (
            set(value) != set(expected)
            or value["schema"] != expected["schema"]
            or value["program_sha256"] != self.sha256
        ):
            raise ValueError("intervention recovery program identity drift")
        loop, index = value["last_loop"], value["last_applied_index"]
        if (
            loop is not None
            and (type(loop) is not int or loop < 0)
            or type(index) is not int
            or not -1 <= index < len(self.interventions)
            or loop is None
            and index != -1
            or index >= 0
            and self.interventions[index].loop > loop
        ):
            raise ValueError("intervention recovery clock drift")
        self.last_loop = loop
        self.last_applied_index = index


def visible_input_digest(observation):
    """Hash exact neural input attributes, excluding transient tags/raw metadata."""
    arrays = {
        name: dict(
            shape=list(array.shape),
            dtype=str(array.dtype),
            sha256=__import__("hashlib").sha256(array.tobytes()).hexdigest(),
        )
        for name, array in sorted(observation["spatial_arrays"].items())
    }
    return digest(
        dict(
            scalars=observation["scalars"],
            scalar_present=observation["scalar_present"],
            entities=observation["entities"],
            upgrades=observation["upgrades"],
            own_operations=observation["own_operations"],
            spatial_present=observation["spatial_present"],
            spatial=arrays,
            map_size=observation["map_size"],
            raster_frame=observation["raster_frame"],
        )
    )
