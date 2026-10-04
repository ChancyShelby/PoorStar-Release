"""Training: gates."""

import json
from dataclasses import asdict
from pathlib import Path

from poorstar.training.checkpoint import file_hash
from poorstar.training.contracts import digest

REQUIRED = {
    "BC": (
        "complete_zvz_corpus",
        "source_identity",
        "mac_complete_tests",
        "cpu_complete_tests",
        "cuda_complete_tests",
        "supervision_census",
        "recurrent_resume",
        "actor_parity",
        "gpu_resources",
        "matched_arms_seeds_budget",
        "frozen_evaluator",
    ),
    "VALUE_ONLY": (
        "supervised_capability",
        "bot_feasibility",
        "actor_parity",
        "outcome_provenance",
        "matched_arms_seeds_budget",
        "frozen_evaluator",
    ),
    "MANAGER_RL": (
        "supervised_capability",
        "bot_feasibility",
        "task_capability",
        "intent_causal_gate",
        "value_pretraining",
        "rollout_probability_parity",
        "matched_arms_seeds_budget",
        "frozen_evaluator",
    ),
    "JOINT_RL": (
        "supervised_capability",
        "bot_feasibility",
        "value_pretraining",
        "rollout_probability_parity",
        "matched_arms_seeds_budget",
        "frozen_evaluator",
    ),
    "SELFPLAY": (
        "supervised_capability",
        "bot_feasibility",
        "outcome_rl",
        "two_player_actor_parity",
        "process_cleanup",
        "immutable_snapshots",
        "matched_arms_seeds_budget",
        "frozen_evaluator",
    ),
    "LEAGUE": (
        "selfplay_capability",
        "fixed_anchors",
        "payoff_provenance",
        "promotion_rules",
        "matched_arms_seeds_budget",
        "frozen_evaluator",
    ),
}


def validate_launch(path, identity, stage, *, execution_config_sha256=None):
    if stage not in REQUIRED:
        raise ValueError("unknown training stage")
    if identity.purpose != "FORMAL":
        raise ValueError("formal engine launch requires FORMAL identity")
    if path is None:
        raise ValueError("launch blocked: frozen evidence preflight is missing")
    document = json.loads(Path(path).read_text())
    check = dict(document)
    expected = check.pop("content_digest", None)
    if (
        digest(check) != expected
        or document.get("schema") != "poorstar_v3_training_launch_preflight_v1"
    ):
        raise ValueError("launch preflight schema/hash drift")
    if (
        document.get("identity_sha256") != identity.sha256
        or document.get("stage") != stage
        or document.get("status") != "READY"
    ):
        raise ValueError("launch preflight identity/stage not ready")
    if (
        execution_config_sha256 is not None
        and document.get("execution_config_sha256") != execution_config_sha256
    ):
        raise ValueError("launch numeric execution configuration drift")
    for name in REQUIRED[stage]:
        evidence = document.get("evidence", {}).get(name)
        if not evidence or document.get("gates", {}).get(name) != "PASS":
            raise ValueError("launch blocked: " + name)
        if file_hash(evidence["path"]) != evidence["sha256"]:
            raise ValueError("evidence content drift: " + name)
        report = json.loads(Path(evidence["path"]).read_text())
        if (
            report.get("status") != "PASS"
            or report.get("gate") != name
            or report.get("binding") != evidence.get("binding")
        ):
            raise ValueError("evidence verdict/binding mismatch: " + name)
        if evidence.get("binding") != asdict(identity):
            raise ValueError(
                "full experiment/source/data/split/model/profile/budget/arm/seed binding required: "
                + name
            )
    return document
