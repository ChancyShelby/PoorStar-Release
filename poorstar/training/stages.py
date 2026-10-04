"""Training: stages."""

import json
from dataclasses import asdict
from pathlib import Path

from poorstar.training.checkpoint import file_hash
from poorstar.training.contracts import digest, is_sha

PHASES = ("BC", "VALUE_ONLY", "MANAGER_RL", "JOINT_RL")
ZERO_SHA = "0" * 64


def successor_binding(identity):
    # The manifest's file hash belongs in identity. Excluding just that field
    # avoids a circular hash while binding every other experiment/data field.
    value = asdict(identity)
    value.pop("stage_transition_sha256")
    return value


def stage_budget(config, stage):
    return dict(
        optimizer_steps=config.total_steps
        if stage == "BC"
        else config.maximum_updates * config.epochs,
        environment_loops=0 if stage == "BC" else config.maximum_environment_loops,
    )


def actor_counters(actor):
    values = actor.get("stage_counters")
    if not isinstance(values, dict) or set(values) != {
        "updates",
        "optimizer_steps",
        "environment_loops",
    }:
        raise ValueError("predecessor actor lacks explicit stage counters")
    if any(type(v) is not int or v < 0 for v in values.values()):
        raise ValueError("invalid predecessor stage counters")
    total = actor.get("cumulative_counters")
    if not isinstance(total, dict) or set(total) != {"optimizer_steps", "environment_loops"}:
        raise ValueError("predecessor actor lacks cumulative stage budget ledger")
    if any(type(v) is not int or v < values[k] for k, v in total.items()):
        raise ValueError("invalid cumulative predecessor counters")
    return values, total


def permitted_transfer(predecessor, successor, *, arm, manager_available):
    if predecessor not in PHASES or successor not in PHASES:
        raise ValueError("declared predecessor/successor training phase required")
    allowed = {("BC", "VALUE_ONLY"), ("VALUE_ONLY", "MANAGER_RL"), ("MANAGER_RL", "JOINT_RL")}
    # No-Manager comparators must not invent a Manager warm-up stage.
    if arm in ("flat", "direct_task") or not manager_available:
        allowed.add(("VALUE_ONLY", "JOINT_RL"))
    if (predecessor, successor) not in allowed:
        raise ValueError("training stage skipped or reversed")
    if successor == "MANAGER_RL" and not manager_available:
        raise ValueError("Manager stage needs actual learned Manager parameters")


def transition_document(
    identity, stage, prior_identity, prior_actor, actor_sha256, config, *, manager_available
):
    """Build a reviewable external manifest; does not write or train anything."""
    permitted_transfer(
        prior_actor.get("training_phase"),
        stage,
        arm=identity.arm,
        manager_available=manager_available,
    )
    if config.sha256 != identity.budget_sha256:
        raise ValueError("successor stage numeric budget identity drift")
    for name in (
        "experiment",
        "arm",
        "seed",
        "dataset_sha256",
        "split_sha256",
        "scope_sha256",
        "model_config_sha256",
        "profile_sha256",
        "purpose",
        "matchup",
    ):
        if getattr(prior_identity, name) != getattr(identity, name):
            raise ValueError("stage transfer changes frozen experiment/data/model: " + name)
    if not is_sha(actor_sha256) or identity.initializer_sha256 != actor_sha256:
        raise ValueError("stage actor initializer hash mismatch")
    counters, totals = actor_counters(prior_actor)
    window = prior_actor.get("dataset_window")
    if window is not None:
        validate_dataset_window(window)
    elif identity.purpose == "FORMAL":
        raise ValueError("formal predecessor lacks its frozen BC dataset window recipe")
    budget = stage_budget(config, stage)
    document = dict(
        schema="poorstar_v3_stage_transition_v1",
        stage=stage,
        predecessor_stage=prior_actor["training_phase"],
        successor_binding=successor_binding(identity),
        predecessor_identity_sha256=prior_identity.sha256,
        predecessor_actor_sha256=actor_sha256,
        predecessor_parameter_sha256=prior_actor["parameter_sha256"],
        predecessor_policy_version=prior_actor["policy_version"],
        predecessor_stage_counters=counters,
        dataset_window=window,
        predecessor_cumulative_counters=totals,
        stage_budget=budget,
        maximum_cumulative_counters={k: totals[k] + budget[k] for k in totals},
        fresh_optimizer=True,
        fresh_stage_counters=True,
        capability_inferred=False,
    )
    document["content_digest"] = digest(document)
    return document


def read_transition(path, identity, stage, config, *, manager_available=True):
    if config.sha256 != identity.budget_sha256:
        raise ValueError("successor stage numeric budget identity drift")
    if (
        not path
        or identity.stage_transition_sha256 == ZERO_SHA
        or file_hash(path) != identity.stage_transition_sha256
    ):
        raise ValueError("stage transition is missing or not frozen in identity")
    document = json.loads(Path(path).read_text())
    validate_transition_state(
        document, identity, stage, config, manager_available=manager_available
    )
    return document


def validate_transition_state(document, identity, stage, config, *, manager_available):
    """Validate the embedded checkpoint contract, independent of file spelling.

    The CLI additionally verifies the externally frozen manifest file SHA.
    Canonical content validation here prevents a direct checkpoint API from
    silently adopting an unrelated or skipped-stage transition.
    """
    if document is None:
        if identity.stage_transition_sha256 != ZERO_SHA:
            raise ValueError("checkpoint lacks its identity-bound stage transition")
        return
    if identity.stage_transition_sha256 == ZERO_SHA:
        raise ValueError("checkpoint stage transition is not frozen in identity")
    if config.sha256 != identity.budget_sha256:
        raise ValueError("successor stage numeric budget identity drift")
    value = dict(document)
    expected = value.pop("content_digest", None)
    if (
        digest(value) != expected
        or document.get("schema") != "poorstar_v3_stage_transition_v1"
        or document.get("stage") != stage
        or document.get("successor_binding") != successor_binding(identity)
        or document.get("stage_budget") != stage_budget(config, stage)
        or document.get("fresh_optimizer") is not True
        or document.get("fresh_stage_counters") is not True
        or document.get("capability_inferred") is not False
    ):
        raise ValueError("stage transition contract drift")
    permitted_transfer(
        document.get("predecessor_stage"),
        stage,
        arm=identity.arm,
        manager_available=manager_available,
    )
    if document.get("dataset_window") is not None:
        validate_dataset_window(document["dataset_window"])
    elif identity.purpose == "FORMAL":
        raise ValueError("formal transition lacks its frozen BC dataset window recipe")
    counters, totals = actor_counters(
        dict(
            stage_counters=document.get("predecessor_stage_counters"),
            cumulative_counters=document.get("predecessor_cumulative_counters"),
        )
    )
    if (
        not is_sha(document.get("predecessor_identity_sha256"))
        or document.get("predecessor_actor_sha256") != identity.initializer_sha256
        or not is_sha(document.get("predecessor_parameter_sha256"))
        or type(document.get("predecessor_policy_version")) is not int
        or document["predecessor_policy_version"] < 0
        or document.get("maximum_cumulative_counters")
        != {k: totals[k] + stage_budget(config, stage)[k] for k in totals}
    ):
        raise ValueError("stage transition predecessor/cumulative budget drift")


def validate_dataset_window(window):
    if (
        not isinstance(window, dict)
        or set(window) != {"unroll_length", "burn_in_length"}
        or type(window["unroll_length"]) is not int
        or window["unroll_length"] < 1
        or type(window["burn_in_length"]) is not int
        or window["burn_in_length"] < 0
    ):
        raise ValueError("frozen original BC dataset window recipe required")


def validate_transition(
    document,
    identity,
    stage,
    prior_identity,
    prior_actor,
    actor_sha256,
    config,
    *,
    manager_available,
):
    expected = transition_document(
        identity,
        stage,
        prior_identity,
        prior_actor,
        actor_sha256,
        config,
        manager_available=manager_available,
    )
    if document != expected:
        raise ValueError("stage predecessor identity/phase/budget drift")
    return document


def initialize_stage(learner, transition):
    if learner.updates or learner.optimizer_steps or learner.optimizer.state:
        raise ValueError("stage transfer requires a fresh optimizer and stage counters")
    if learner.parameter_sha256 != transition["predecessor_parameter_sha256"]:
        raise ValueError("stage transfer weights are not the frozen predecessor")
    learner.stage_transition = transition
    learner.policy_version = transition["predecessor_policy_version"]
    if hasattr(learner, "behavior_registry"):
        learner.behavior_registry = {learner.policy_version: learner.parameter_sha256}


def cumulative_counters(learner):
    prior = (
        {}
        if getattr(learner, "stage_transition", None) is None
        else learner.stage_transition["predecessor_cumulative_counters"]
    )
    return dict(
        optimizer_steps=prior.get("optimizer_steps", 0) + learner.optimizer_steps,
        environment_loops=prior.get("environment_loops", 0) + learner.environment_loops,
    )


def validate_recovered_counters(state, config, phase):
    maximum = config.total_steps if phase == "BC" else config.maximum_updates
    epochs = 1 if phase == "BC" else config.epochs
    for name in ("updates", "optimizer_steps", "policy_version", "environment_loops"):
        if type(state[name]) is not int or state[name] < 0:
            raise ValueError("recovered stage counters must be nonnegative integers")
    transition = state.get("stage_transition")
    inherited = 0 if transition is None else transition["predecessor_policy_version"]
    if (
        state["updates"] > maximum
        or state["optimizer_steps"] != state["updates"] * epochs
        or state["policy_version"] != inherited + state["updates"]
        or state["environment_loops"] > (0 if phase == "BC" else config.maximum_environment_loops)
    ):
        raise ValueError("recovered optimizer/version/frame counter drift")
