"""Evaluation: confidence."""

import math
from dataclasses import dataclass

import torch

from poorstar.training.contracts import digest, is_sha

EVENT = "parent_task_confirmed_before_commitment_end"


@dataclass(frozen=True)
class ConfidenceWindow:
    episode: str
    parent_revision: int
    start_loop: int
    end_loop: int
    prediction_observation_sha256: str
    event: str = EVENT

    def validate(self):
        if (
            not self.episode
            or type(self.parent_revision) is not int
            or self.parent_revision < 1
            or type(self.start_loop) is not int
            or self.start_loop < 0
            or type(self.end_loop) is not int
            or self.end_loop <= self.start_loop
            or not is_sha(self.prediction_observation_sha256)
            or self.event != EVENT
        ):
            raise ValueError("exact commitment/event/current observation identity required")

    @property
    def sha256(self):
        return digest(self.__dict__)


@dataclass(frozen=True)
class ConfidenceTarget:
    window_sha256: str
    value: float | None
    mask: bool
    reason: str
    evidence_sha256: tuple[str, ...]

    def validate(self):
        if (
            not is_sha(self.window_sha256)
            or type(self.mask) is not bool
            or not self.reason
            or not self.evidence_sha256
            or any(not is_sha(h) for h in self.evidence_sha256)
            or (self.mask and self.value not in (0.0, 1.0))
            or (not self.mask and self.value is not None)
        ):
            raise ValueError("explicit censored/observed confidence target required")


def build_confidence_target(
    window, *, confirmations=(), coverage=(), cutoff_loop=None, cutoff_evidence_sha256=None
):
    """Intervals are complete observer coverage, not sparse observation timestamps.

    confirmation fields: episode,parent_revision,loop,task_id,status,evidence_sha256.
    coverage fields: episode,parent_revision,start_loop,end_loop,complete,evidence_sha256.
    Only coverage certified by the task-aware observer can support a negative.
    """
    window.validate()
    evidence = []
    cutoff = window.end_loop if cutoff_loop is None else cutoff_loop
    if type(cutoff) is not int or cutoff < window.start_loop:
        raise ValueError("explicit chronological commitment observation cutoff required")
    if cutoff_evidence_sha256 is not None:
        if not is_sha(cutoff_evidence_sha256):
            raise ValueError("content-bound cutoff evidence required")
        evidence.append(cutoff_evidence_sha256)
    positive = False
    for event in confirmations:
        if set(event) != {
            "episode",
            "parent_revision",
            "loop",
            "task_id",
            "status",
            "evidence_sha256",
        }:
            raise ValueError("typed visible confirmation provenance required")
        if not is_sha(event["evidence_sha256"]) or not event["task_id"]:
            raise ValueError("confirmation task and evidence required")
        if type(event["loop"]) is not int or event["loop"] < 0 or event["loop"] > cutoff:
            raise ValueError("future confirmation cannot enter a target build")
        if event["status"] not in ("confirmed", "accepted", "rejected", "timeout", "censored"):
            raise ValueError("declared observer status required")
        if (
            event["episode"] == window.episode
            and event["parent_revision"] == window.parent_revision
        ):
            if window.start_loop <= event["loop"] < window.end_loop:
                evidence.append(event["evidence_sha256"])
                positive |= event["status"] == "confirmed"
    intervals = []
    for item in coverage:
        if set(item) != {
            "episode",
            "parent_revision",
            "start_loop",
            "end_loop",
            "complete",
            "evidence_sha256",
        }:
            raise ValueError("explicit observer coverage certificate required")
        begin, end = item["start_loop"], item["end_loop"]
        if (
            type(begin) is not int
            or type(end) is not int
            or begin < 0
            or end < begin
            or end > cutoff
            or type(item["complete"]) is not bool
            or not is_sha(item["evidence_sha256"])
        ):
            raise ValueError("invalid/future observer coverage")
        if item["episode"] == window.episode and item["parent_revision"] == window.parent_revision:
            evidence.append(item["evidence_sha256"])
            if item["complete"]:
                intervals.append((max(begin, window.start_loop), min(end, window.end_loop)))
    if not evidence:
        raise ValueError("even a censored target requires explicit observation/cutoff evidence")
    cursor = window.start_loop
    for begin, end in sorted(intervals):
        if begin > cursor:
            break
        cursor = max(cursor, end)
    complete = cutoff >= window.end_loop and cursor >= window.end_loop
    target = ConfidenceTarget(
        window.sha256,
        1.0 if positive else 0.0 if complete else None,
        positive or complete,
        "VISIBLE_PARENT_GOAL_CONFIRMED"
        if positive
        else "FULL_COMMITMENT_OBSERVED_WITHOUT_CONFIRMATION"
        if complete
        else "CENSORED_COMMITMENT_OR_OBSERVER_COVERAGE",
        tuple(sorted(set(evidence))),
    )
    target.validate()
    return target


def confidence_auxiliary_loss(predictions, targets):
    """Differentiable BCE sum and eligible denominator; no outcome actor input."""
    if (
        predictions.ndim != 1
        or len(predictions) != len(targets)
        or not torch.isfinite(predictions).all()
    ):
        raise ValueError("one finite current prediction per commitment required")
    if torch.any((predictions <= 0) | (predictions >= 1)):
        raise ValueError("confidence probabilities must be strictly inside (0,1)")
    for target in targets:
        target.validate()
    eligible = [i for i, target in enumerate(targets) if target.mask]
    if not eligible:
        return predictions.sum() * 0.0, 0
    values = predictions.new_tensor([targets[i].value for i in eligible])
    return torch.nn.functional.binary_cross_entropy(
        predictions[eligible], values, reduction="sum"
    ), len(eligible)


def confidence_calibration(predictions, targets, *, bin_edges):
    edges = tuple(bin_edges)
    if len(edges) < 2 or edges[0] != 0 or edges[-1] != 1 or tuple(sorted(set(edges))) != edges:
        raise ValueError("frozen complete probability bins required")
    if len(predictions) != len(targets):
        raise ValueError("prediction/target length mismatch")
    bins = [[] for _ in edges[:-1]]
    observed = []
    censored = 0
    for prediction, target in zip(predictions, targets):
        target.validate()
        if not math.isfinite(prediction) or not 0 <= prediction <= 1:
            raise ValueError("finite probability required")
        if not target.mask:
            censored += 1
            continue
        observed.append((prediction, target.value))
        for index, (low, high) in enumerate(zip(edges, edges[1:])):
            if low <= prediction < high or index == len(bins) - 1 and prediction == 1:
                bins[index].append((prediction, target.value))
                break
    if not observed:
        return dict(
            status="NO_ELIGIBLE_TARGETS",
            denominator=0,
            censored=censored,
            brier=None,
            ece=None,
            bins=[],
        )
    rows = []
    for values, low, high in zip(bins, edges, edges[1:]):
        rows.append(
            dict(
                lower=low,
                upper=high,
                count=len(values),
                mean_prediction=None if not values else sum(p for p, _ in values) / len(values),
                observed_rate=None if not values else sum(y for _, y in values) / len(values),
            )
        )
    n = len(observed)
    return dict(
        status="MEASURED",
        denominator=n,
        censored=censored,
        brier=sum((p - y) ** 2 for p, y in observed) / n,
        ece=sum(
            x["count"] * abs(x["mean_prediction"] - x["observed_rate"]) for x in rows if x["count"]
        )
        / n,
        bins=rows,
        calibrated_claim=False,
    )


def fit_temperature(predictions, targets, *, split, evidence_sha256, iterations=50):
    """Prospective train/development-only calibration, separate from paper test."""
    if split not in ("train", "development") or not is_sha(evidence_sha256):
        raise ValueError("calibration cannot fit paper test; frozen training evidence required")
    if type(iterations) is not int or iterations < 1:
        raise ValueError("frozen finite calibration iterations required")
    values = torch.as_tensor(predictions, dtype=torch.float64)
    _, count = confidence_auxiliary_loss(values, targets)
    if count == 0:
        raise ValueError("calibration fitting requires observed event targets")
    mask = [i for i, target in enumerate(targets) if target.mask]
    logits = torch.logit(values[mask]).detach()
    labels = values.new_tensor([targets[i].value for i in mask])
    log_temperature = torch.zeros((), dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.Adam([log_temperature], lr=0.05)
    for _ in range(iterations):
        optimizer.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits / log_temperature.exp(), labels
        )
        loss.backward()
        optimizer.step()
    temperature = float(log_temperature.exp().detach())
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("nonfinite calibration artifact")
    artifact = dict(
        schema="poorstar_v3_confidence_temperature_v1",
        event=EVENT,
        split=split,
        evidence_sha256=evidence_sha256,
        iterations=iterations,
        eligible=count,
        temperature=temperature,
        original_predictions_unchanged=True,
        paper_calibration_evaluated=False,
    )
    artifact["content_digest"] = digest(artifact)
    return artifact
