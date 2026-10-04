"""Evaluation: metrics."""

import math
from collections import Counter
from dataclasses import asdict

import numpy as np

from poorstar.training.contracts import digest


class VisibleMetricRecorder:
    def __init__(self, *, player_id=None):
        self.player_id = player_id
        self.observations = []
        self.controls = []
        self.confidence_predictions = []

    def observe(self, frame):
        if self.player_id is None:
            self.player_id = frame.key.player_id
        if (
            frame.key.player_id != self.player_id
            or self.observations
            and frame.key.loop <= self.observations[-1]["loop"]
        ):
            raise ValueError("one chronological controlled-player metric observation required")
        observation = frame.policy_observation()
        metrics = {}
        for name, value in observation["scalars"].items():
            metrics[name] = value if observation["scalar_present"][name] else None
        metrics["observed_upgrades"] = (
            None if observation["upgrades"] is None else len(observation["upgrades"])
        )
        # ActionFrame.policy_observation verifies its uncapped entity census
        # against slot_tags before returning; it intentionally omits overflow.
        completed = Counter()
        incomplete = Counter()
        coverage = len(observation["entities"]) == len(frame.slot_tags)
        for entity in observation["entities"]:
            v, p = entity["values"], entity["present"]
            if not all(
                p.get(n) for n in ("alliance", "display_type", "unit_type", "build_progress")
            ):
                coverage = False
                continue
            if v["alliance"] == 1 and v["display_type"] == 1:
                (completed if v["build_progress"] >= 1 else incomplete)[str(v["unit_type"])] += 1
        metrics["completed_own_units"] = sum(completed.values()) if coverage else None
        metrics["incomplete_own_units"] = sum(incomplete.values()) if coverage else None
        array = observation["spatial_arrays"].get("visibility")
        metrics["visible_native_raster_fraction"] = (
            float(np.count_nonzero(array == 2) / array.size)
            if observation["spatial_present"].get("visibility") and array is not None and array.size
            else None
        )
        row = dict(
            loop=frame.key.loop,
            player_id=self.player_id,
            observation_sha256=frame.key.observation_sha256,
            metrics=metrics,
            completed_own_unit_types=dict(completed) if coverage else None,
            complete_entity_coverage=coverage,
        )
        row["content_digest"] = digest(row)
        self.observations.append(row)

    def control(self, frame, state, trace, phase):
        if phase not in ("scheduler", "body") or frame.key.player_id != self.player_id:
            raise ValueError("current controlled-player phase measurement required")
        manager = trace["scheduler_control"].get("manager") if phase == "scheduler" else None
        row = dict(
            episode=state.episode,
            loop=frame.key.loop,
            phase=phase,
            preference=None if state.intent is None else list(state.intent.preference),
            operation=None if manager is None else manager.get("operation"),
            manager_decision=trace["manager_decision"] if phase == "scheduler" else False,
            tasks={
                t.task_id: dict(task_type=t.task_type, status=t.status, progress=t.progress)
                for t in state.tasks
            },
            observation_sha256=frame.key.observation_sha256,
        )
        row["evidence_sha256"] = digest(row)
        self.controls.append(row)

    def confidence(self, frame, state, probability):
        """Record a pure prior-context model mean for an actual learned update.

        The collector computes probability before conditioning on the selected
        Intent, and supplies the new state only to bind its commitment window.
        Neither sampled confidence nor future outcomes are prediction inputs.
        """
        from poorstar.evaluation.confidence import ConfidenceWindow

        intent = state.intent
        if (
            frame.key.player_id != self.player_id
            or intent is None
            or intent.source != "learned"
            or intent.update_or_termination != "update"
            or not intent.active
            or intent.age != 0
            or intent.start_loop != frame.key.loop
        ):
            raise ValueError("actual current learned Intent update required for confidence ledger")
        value = float(probability)
        if not math.isfinite(value) or not 0 < value < 1:
            raise ValueError("actual differentiable Beta mean must be inside (0,1)")
        window = ConfidenceWindow(
            state.episode,
            intent.revision,
            intent.start_loop,
            intent.start_loop + intent.commitment_horizon,
            frame.key.observation_sha256,
        )
        window.validate()
        if any(v["window_sha256"] == window.sha256 for v in self.confidence_predictions):
            raise ValueError("duplicate confidence commitment prediction")
        row = dict(
            window=asdict(window),
            window_sha256=window.sha256,
            prediction=value,
            player_id=self.player_id,
            prediction_kind="PURE_PRECHOICE_MANAGER_BETA_MEAN",
        )
        row["evidence_sha256"] = digest(row)
        self.confidence_predictions.append(row)

    def report(self, actor_report):
        receipts = [r for r in actor_report["receipts"] if r["player_id"] == self.player_id]
        goals = [r for r in actor_report["goal_observations"] if r["player_id"] == self.player_id]
        from poorstar.evaluation.confidence import ConfidenceWindow, build_confidence_target

        predictions = []
        confirmations = [
            dict(
                episode=actor_report["episode_id"],
                parent_revision=g["parent_revision"],
                loop=g["loop"],
                task_id=g["task_id"],
                status="confirmed",
                evidence_sha256=g["evidence_id"],
            )
            for g in goals
            if g["outcome"] == "confirmed"
        ]
        for prediction in self.confidence_predictions:
            window = ConfidenceWindow(**prediction["window"])
            if window.episode != actor_report["episode_id"]:
                raise ValueError("confidence commitment belongs to another actual episode")
            target = build_confidence_target(
                window,
                confirmations=confirmations,
                cutoff_loop=actor_report["environment_loops"],
                cutoff_evidence_sha256=digest(actor_report),
            )
            predictions.append(dict(prediction, target=asdict(target)))
        return dict(
            schema="poorstar_v3_visible_evaluator_metrics_v1",
            provenance="evaluator",
            observations=self.observations,
            controls=self.controls,
            dispatch_counts=dict(
                commands=sum(r["commands"] for r in receipts),
                accepted=sum(r["accepted"] for r in receipts),
                rejected=sum(r["commands"] - r["accepted"] for r in receipts),
            ),
            goal_observation_counts=dict(Counter(r["outcome"] for r in goals)),
            goal_observations=goals,
            confidence_predictions=predictions,
            confidence_negative_coverage_certificates_available=False,
            terminal_outcome=actor_report["reward"] if actor_report["terminated"] else None,
            physical_environment_loops=actor_report["environment_loops"],
            interpretation="FIXED_CLOCK_VISIBLE_SAMPLES_NOT_CONTINUOUS_OR_HIDDEN_STATE_TRUTH",
            human_intent_labels_inferred=False,
            engine_qualification_inferred=False,
        )
