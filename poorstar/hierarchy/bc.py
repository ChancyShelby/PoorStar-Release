"""Hierarchy: bc."""

import math
from collections import defaultdict
from dataclasses import dataclass

import torch
from torch import nn

from poorstar.data.bc.records import BCDecision
from poorstar.data.bc.supervision import STRICT
from poorstar.hierarchy.model import HierarchicalPolicy
from poorstar.hierarchy.state import HierarchyState
from poorstar.training.foundation import bc_objective


@dataclass(frozen=True)
class HierarchyBCRecord:
    decision: BCDecision
    scheduler_control: dict | None = None
    body_control: dict | None = None
    body_execution_task: str | None = None
    feedback: tuple = ()
    external_override: object | None = None
    supervision_id: str | None = None
    supervision_kind: str | None = None
    scheduler_dispatch_observation: object | None = None
    body_dispatch_observation: object | None = None
    scheduler_task_allocations: dict | None = None
    body_task_allocations: dict | None = None
    body_feedback: tuple = ()
    confidence_target: object | None = None

    def validate(self, model):
        self.decision.validate(model.config.maximum_delay)
        if not self.decision.valid:
            return
        if model.hierarchy_config.arm != "flat":
            if (
                not isinstance(self.supervision_id, str)
                or not self.supervision_id
                or self.supervision_kind
                not in (
                    "actor_control_trace",
                    "weak_visible_task_labels",
                    "declared_external_intervention",
                )
            ):
                raise ValueError("explicit high-level supervision provenance required")
            if self.scheduler_control is None:
                raise ValueError(
                    "missing high-level scheduler supervision; no expert Intent is inferred"
                )
            if self.decision.decision == "ACT" and self.body_control is None:
                raise ValueError("missing high-level body supervision")
        if self.decision.decision == "WAIT" and (
            self.body_control is not None or self.body_execution_task is not None
        ):
            raise ValueError("WAIT has no body control/association supervision")
        if self.decision.decision == "WAIT" and (
            self.body_feedback
            or self.body_task_allocations is not None
            or self.body_dispatch_observation is not None
        ):
            raise ValueError("WAIT has no body observations/allocations/feedback")
        if self.confidence_target is not None:
            from poorstar.evaluation.confidence import ConfidenceTarget

            if not isinstance(self.confidence_target, ConfidenceTarget):
                raise ValueError("typed confidence auxiliary target required")
            self.confidence_target.validate()


@dataclass(frozen=True)
class HierarchyBCSequence:
    records: tuple[HierarchyBCRecord, ...]
    initial_control_state: HierarchyState | None = None


@dataclass(frozen=True)
class BCPhaseStateAudit:
    replay_sha256: str
    episode: str
    player_id: int
    scheduler_loop: int
    phase: str
    phase_loop: int
    prior_state_dict: dict
    post_state_dict: dict


class _PhaseAdapter(nn.Module):
    def __init__(self, model, sequences, state_audit=None):
        super().__init__()
        self.model = model
        self.config = model.config
        self.queue = []
        self.index = 0
        self.control_state = None
        self.control_losses = defaultdict(list)
        self.confidence_predictions = []
        self.confidence_targets = []
        self.state_audit = state_audit
        for sequence in sequences:
            previous = None
            for record in sequence.records:
                row = record.decision
                if not row.valid:
                    continue
                identity = (
                    row.replay_sha256,
                    row.scheduler.key.episode,
                    row.scheduler.key.player_id,
                )
                start = identity != previous
                initial = sequence.initial_control_state if previous is None else None
                if (
                    start
                    and model.hierarchy_config.arm != "flat"
                    and row.scheduler.key.loop > 0
                    and initial is None
                ):
                    raise ValueError(
                        "mid-episode hierarchical BC window requires its causal initial control snapshot"
                    )
                previous = identity
                self.queue.append((record, "scheduler", start, initial))
                if row.decision == "ACT":
                    self.queue.append((record, "body", False, None))

    @property
    def device(self):
        return self.model.device

    def scheduler(self, *args, **kwargs):
        return self.model.scheduler(*args, **kwargs)

    def consume_schedule(self, *args, **kwargs):
        return self.model.consume_schedule(*args, **kwargs)

    def teacher_body(self, *args, **kwargs):
        return self.model.teacher_body(*args, **kwargs)

    def encode(self, observation, hidden=None):
        if self.index >= len(self.queue):
            raise ValueError("BC phase queue exhausted")
        record, phase, reset, initial = self.queue[self.index]
        self.index += 1
        row = record.decision
        frame = row.scheduler if phase == "scheduler" else row.body
        if observation["observation_loop"] != frame.key.loop:
            raise ValueError("BC control label/observation phase mismatch")
        if reset:
            self.control_state = (
                initial if initial is not None else HierarchyState(frame.key.episode)
            )
            if self.control_state.episode != frame.key.episode:
                raise ValueError("BC initial controls belong to another episode")
        prior = self.control_state
        receipt = (
            record.scheduler_dispatch_observation
            if phase == "scheduler"
            else record.body_dispatch_observation
        )
        feedback = record.feedback if phase == "scheduler" else record.body_feedback
        result = self.model.encode_control(
            observation,
            hidden,
            prior,
            teacher_control=record.scheduler_control
            if phase == "scheduler"
            else record.body_control,
            feedback=feedback,
            external_override=record.external_override if phase == "scheduler" else None,
            allow_manager=phase == "scheduler",
            dispatch_observation=receipt,
            task_allocations=record.scheduler_task_allocations
            if phase == "scheduler"
            else record.body_task_allocations,
        )
        self.control_state = result.control_state
        if self.state_audit is not None:
            self.state_audit.append(
                BCPhaseStateAudit(
                    row.replay_sha256,
                    frame.key.episode,
                    frame.key.player_id,
                    row.scheduler.key.loop,
                    phase,
                    frame.key.loop,
                    prior.to_dict(),
                    result.control_state.to_dict(),
                )
            )
        active = row.loss and not row.burn_in
        if phase == "scheduler" and record.confidence_target is not None:
            from poorstar.evaluation.confidence import ConfidenceWindow

            c = self.model.hierarchy_config
            control = result.control["manager"]
            if (
                control is None
                or control["operation"] != "update"
                or not c.persistence
                or c.commitment_mode != "minimum_hold"
                or c.manager_mode != "learned"
            ):
                raise ValueError(
                    "confidence targets require a new learned full-lifecycle Intent update"
                )
            intent = result.control_state.intent
            window = ConfidenceWindow(
                frame.key.episode,
                intent.revision,
                intent.start_loop,
                intent.start_loop + intent.commitment_horizon,
                frame.key.observation_sha256,
            )
            window.validate()
            if window.sha256 != record.confidence_target.window_sha256:
                raise ValueError(
                    "confidence target belongs to a different observation/commitment/revision"
                )
            if active:
                prediction = self.model.confidence_prediction(
                    observation,
                    hidden,
                    prior,
                    feedback=feedback,
                    dispatch_observation=receipt,
                    task_allocations=record.scheduler_task_allocations,
                )
                self.confidence_predictions.append(prediction)
                self.confidence_targets.append(record.confidence_target)
        if active:
            if result.manager_decision:
                self.control_losses["manager_control"].append(-result.manager_log_prob)
            if result.control["tasks"] is not None:
                self.control_losses["task_control"].append(-result.task_log_prob)
        if phase == "body":
            encoded, lp, _, _ = self.model.select_execution_task(
                result, teacher_task_id=record.body_execution_task
            )
            if active and self.model.hierarchy_config.arm != "flat":
                self.control_losses["execution_task"].append(-lp)
            return encoded
        return result.encoded


def hierarchy_bc_objective(
    model,
    sequences,
    profile,
    *,
    head_weights=None,
    control_weights=None,
    collect_metrics=True,
    observation_cache=None,
    supervision_policy=STRICT,
    state_audit=None,
):
    """`sequences` are HierarchyBCSequence or tuples of HierarchyBCRecord. Caller
    supplies frozen provenance-bound controls; unknown labels remain unavailable
    and fail here instead of becoming random/heuristic expert truth. Plain flat
    BC can use records wrapping the unchanged BCDecision with no controls.
    """
    if not isinstance(model, HierarchicalPolicy):
        raise ValueError("hierarchical BC requires the named C3 adapter policy")
    sequences = tuple(
        q if isinstance(q, HierarchyBCSequence) else HierarchyBCSequence(tuple(q))
        for q in sequences
    )
    for q in sequences:
        for r in q.records:
            if not isinstance(r, HierarchyBCRecord):
                raise ValueError("explicit HierarchyBCRecord required")
            r.validate(model)
    weights = {} if control_weights is None else control_weights
    if set(weights) - {
        "manager_control",
        "task_control",
        "execution_task",
        "confidence_aux",
    } or any(
        not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0 for v in weights.values()
    ):
        raise ValueError("finite declared control-objective weights required")
    if state_audit is not None and not isinstance(state_audit, list):
        raise ValueError("state audit requires an explicit append-only list")
    adapter = _PhaseAdapter(model, sequences, state_audit)
    result = bc_objective(
        adapter,
        [tuple(r.decision for r in q.records) for q in sequences],
        profile,
        head_weights=head_weights,
        collect_metrics=collect_metrics,
        observation_cache=observation_cache,
        supervision_policy=supervision_policy,
        feature_batch_size=1,
    )
    if adapter.index != len(adapter.queue):
        raise ValueError("unconsumed BC control phases")
    for head, values in adapter.control_losses.items():
        total = torch.stack(values).sum()
        count = len(values)
        result.head_sums[head] = total
        result.denominators[head] = count
        result.loss = result.loss + total / count * weights.get(head, 1.0)
    if adapter.confidence_predictions:
        from poorstar.evaluation.confidence import confidence_auxiliary_loss

        total, count = confidence_auxiliary_loss(
            torch.stack(adapter.confidence_predictions), adapter.confidence_targets
        )
        result.head_sums["confidence_aux"] = total
        result.denominators["confidence_aux"] = count
        if count:
            result.loss = result.loss + total / count * weights.get("confidence_aux", 1.0)
    if not torch.isfinite(result.loss):
        raise ValueError("nonfinite hierarchical BC objective")
    result.counts["explicit_control_phases"] = adapter.index
    return result


def audit_hierarchy_bc_sequence(model, sequence, profile, **kwargs):
    """Return the original loss plus snapshots from its actual teacher phases.

    No second forward, sampling, optimizer or inferred high-level label occurs.
    Padding contributes no snapshot; burn-in remains part of the causal trace.
    """
    if "state_audit" in kwargs:
        raise ValueError("audit wrapper owns its fresh phase ledger")
    ledger = []
    result = hierarchy_bc_objective(model, (sequence,), profile, state_audit=ledger, **kwargs)
    return result, tuple(ledger)
