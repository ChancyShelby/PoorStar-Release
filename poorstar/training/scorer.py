"""Match actor sampling context to differentiable PPO action rescoring."""

import math
from copy import deepcopy
from dataclasses import dataclass

import torch

from poorstar.actions.coarse import CoarsePrefixMachine
from poorstar.actions.contract import Token
from poorstar.training.contracts import StepScore, digest


@dataclass(frozen=True)
class PolicyStepPayload:
    scheduler_frame: object
    body_frame: object | None
    trace: dict
    interrupted_before_body: bool = False
    dispatch_results: tuple[int, ...] = ()
    memory_before: torch.Tensor | None = None

    @property
    def observation_sha256(self):
        return digest(
            [
                self.scheduler_frame.key.observation_sha256,
                None if self.body_frame is None else self.body_frame.key.observation_sha256,
            ]
        )

    @property
    def mask_sha256(self):
        value = dict(
            scheduler=self.scheduler_frame.key.__dict__,
            body=None if self.body_frame is None else self.body_frame.key.__dict__,
            tokens=[dict(head=t["head"], options=t["options"]) for t in self.trace["body_tokens"]],
            scheduler_control=self.trace["scheduler_control"],
            body_control=self.trace["body_control"],
        )
        for name in (
            "scheduler_dispatch_observation",
            "body_dispatch_observation",
            "scheduler_task_allocations",
            "body_task_allocations",
            "body_feedback",
        ):
            if name in self.trace:
                value[name] = self.trace[name]
        return digest(value)

    def validate(self, model, profile):
        t = self.trace
        if self.memory_before is not None and (
            not isinstance(self.memory_before, torch.Tensor)
            or self.memory_before.shape != (model.config.hidden_size,)
            or not torch.isfinite(self.memory_before).all()
        ):
            raise ValueError("bound finite recurrent state required")
        expected = {
            "prior_control_state",
            "scheduler_loop",
            "decision",
            "delay",
            "scheduler_control",
            "body_control",
            "body_execution_task",
            "feedback",
            "external_override",
            "body_tokens",
            "manager_decision",
            "manager_log_prob",
            "task_log_prob",
            "value",
            "manager_value",
            "action_log_prob",
        }
        if model.hierarchy_config.common_receipts_mode == "shared_v1":
            expected |= {"scheduler_dispatch_observation", "body_dispatch_observation"}
        if model.hierarchy_config.grounded_tasks:
            expected |= {"scheduler_task_allocations", "body_task_allocations", "body_feedback"}
        if set(t) != expected or t["scheduler_loop"] != self.scheduler_frame.key.loop:
            raise ValueError("explicit sampled phase/control trace required")
        if self.scheduler_frame.terminal or t["decision"] not in ("ACT", "WAIT"):
            raise ValueError("nonterminal current scheduler required")
        if type(t["delay"]) is not int or not 0 <= t["delay"] <= model.config.maximum_delay:
            raise ValueError("delay outside current scheduler vocabulary")
        from poorstar.hierarchy.state import ExecutionFeedback, ExternalOverride, HierarchyState

        prior = HierarchyState.from_dict(t["prior_control_state"], model.hierarchy_config)
        events = tuple(ExecutionFeedback(**e) for e in t["feedback"])
        from poorstar.hierarchy.state import DispatchObservation

        for phase, frame in (("scheduler", self.scheduler_frame), ("body", self.body_frame)):
            receipt = t.get(phase + "_dispatch_observation")
            if receipt is not None:
                if frame is None:
                    raise ValueError("missing body cannot contain common receipt")
                DispatchObservation.from_dict(receipt).validate(frame.key.loop)
        if model.hierarchy_config.grounded_tasks:
            if self.body_frame is None and (
                t["body_task_allocations"] is not None or t["body_feedback"]
            ):
                raise ValueError("missing body cannot contain goal allocation/feedback")
            for event in t["body_feedback"]:
                ExecutionFeedback(**event).validate(self.body_frame.key.loop)
        override = t["external_override"]
        if override is not None:
            override = dict(override)
            override["preference"] = tuple(override["preference"])
            override = ExternalOverride(**override)
        eligible = model.manager_decision_eligible(
            prior,
            t["scheduler_loop"],
            feedback=events,
            external_override=override,
            allow_manager=True,
        )
        if t["manager_decision"] != eligible:
            raise ValueError("actual trainable Manager event eligibility mismatch")
        if t["external_override"] is not None and t["manager_decision"]:
            raise ValueError("external intervention has no learned Manager density")
        if t["decision"] == "WAIT":
            if (
                not t["delay"]
                or self.body_frame is not None
                or t["body_tokens"]
                or self.interrupted_before_body
            ):
                raise ValueError("WAIT may not contain a fabricated body")
        elif self.body_frame is None:
            if (
                not self.interrupted_before_body
                or t["body_tokens"]
                or t["body_control"] is not None
            ):
                raise ValueError("missing ACT body needs explicit terminal interruption")
        else:
            a, b = self.scheduler_frame.key, self.body_frame.key
            if (
                self.body_frame.terminal
                or (a.episode, a.player_id) != (b.episode, b.player_id)
                or b.loop != a.loop + t["delay"]
            ):
                raise ValueError("ACT body phase/episode mismatch")
            profile.check_frame(self.body_frame)


def position_entropy(dist):
    """Exact entropy of disjoint coarse regions with truncated-normal offsets."""
    from poorstar.models.policy import WorldPointDistribution

    entropies = []
    for parameters in dist.fine:
        d = WorldPointDistribution(parameters, dist.cell_size)
        alpha = -d.normal.loc / d.normal.scale
        beta = (d.size - d.normal.loc) / d.normal.scale
        phi_a = torch.exp(-alpha.square() / 2) / math.sqrt(2 * math.pi)
        phi_b = torch.exp(-beta.square() / 2) / math.sqrt(2 * math.pi)
        entropy = (
            torch.log(d.normal.scale * d.mass)
            + 0.5 * math.log(2 * math.pi * math.e)
            + (alpha * phi_a - beta * phi_b) / (2 * d.mass)
        ).sum()
        entropies.append(entropy)
    return dist.coarse.entropy() + (dist.coarse.probs * torch.stack(entropies)).sum()


class HierarchyScorer:
    def __init__(self, model, profile):
        self.model, self.profile = model, profile

    def __call__(self, step, hidden=None):
        from poorstar.hierarchy.state import ExecutionFeedback, ExternalOverride, HierarchyState

        model = self.model
        payload = step.payload
        payload.validate(model, self.profile)
        trace = payload.trace
        prior = HierarchyState.from_dict(trace["prior_control_state"], model.hierarchy_config)
        memory = hidden
        if isinstance(hidden, tuple):
            memory, prior_from_rollout = hidden
            if prior.to_dict() != prior_from_rollout.to_dict():
                raise ValueError("recurrent control state discontinuity")
        if isinstance(memory, torch.Tensor):
            if memory.shape != (model.config.hidden_size,) or not torch.isfinite(memory).all():
                raise ValueError("recurrent snapshot shape/value drift")
            memory = memory.to(model.device)
        feedback = tuple(ExecutionFeedback(**e) for e in trace["feedback"])
        override = trace["external_override"]
        if override is not None:
            override = dict(override)
            override["preference"] = tuple(override["preference"])
            override = ExternalOverride(**override)
        from poorstar.hierarchy.state import DispatchObservation

        def receipt(phase):
            value = trace.get(phase + "_dispatch_observation")
            return None if value is None else DispatchObservation.from_dict(value)

        encoded = model.encode_control(
            payload.scheduler_frame.policy_observation(),
            memory,
            prior,
            feedback=feedback,
            external_override=override,
            teacher_control=trace["scheduler_control"],
            allow_manager=True,
            dispatch_observation=receipt("scheduler"),
            task_allocations=trace.get("scheduler_task_allocations"),
        )
        if encoded.manager_decision != step.manager_decision:
            raise ValueError("Manager event eligibility differs from behavior")
        act, delays = model.scheduler(encoded.encoded, trace["decision"])
        lp = encoded.control_log_prob + act.log_prob(
            torch.tensor(("ACT", "WAIT").index(trace["decision"]), device=model.device)
        )
        lp = lp + delays.log_prob(torch.tensor(trace["delay"], device=model.device))
        entropy = encoded.control_entropy + act.entropy() + delays.entropy()
        memory = model.consume_schedule(encoded.encoded, trace["delay"])
        state = encoded.control_state
        if payload.body_frame is not None:
            body = model.encode_control(
                payload.body_frame.policy_observation(),
                memory,
                state,
                teacher_control=trace["body_control"],
                allow_manager=False,
                feedback=tuple(ExecutionFeedback(**e) for e in trace.get("body_feedback", [])),
                dispatch_observation=receipt("body"),
                task_allocations=trace.get("body_task_allocations"),
            )
            executable, task_lp, task_entropy, task_id = model.select_execution_task(
                body, teacher_task_id=trace["body_execution_task"]
            )
            body.encoded = executable
            state = body.control_state
            prefix = body.encoded.hidden
            lp = lp + body.control_log_prob
            entropy = entropy + body.control_entropy
            lp = lp + task_lp
            entropy = entropy + task_entropy
            machine = CoarsePrefixMachine(payload.body_frame, self.profile)
            for token in trace["body_tokens"]:
                head, options, value = token["head"], tuple(token["options"]), token["value"]
                if isinstance(value, list):
                    value = tuple(value)
                if machine.head != head or machine.options() != options:
                    raise ValueError("sampled/replayed C3 token mask mismatch")
                distribution = model.distribution(body.encoded, prefix, head, options)
                if head == "target_position":
                    logp = distribution.log_prob(value)
                    ent = position_entropy(distribution)
                else:
                    if value not in options:
                        raise ValueError("recorded action outside current C3 domain")
                    logp = distribution.log_prob(
                        torch.tensor(options.index(value), device=model.device)
                    )
                    ent = distribution.entropy()
                lp = lp + logp
                entropy = entropy + ent
                machine.accept(Token(head, value))
                prefix = model.consume(body.encoded, prefix, Token(head, value))
            if machine.head != "done":
                raise ValueError("incomplete structured bundle cannot train policy")
            memory = prefix
        return StepScore(
            lp,
            entropy,
            encoded.value,
            (memory, state),
            encoded.manager_log_prob,
            encoded.manager_entropy,
            encoded.manager_value,
            payload.observation_sha256,
            payload.mask_sha256,
        )


def payload_from_session(
    session,
    scheduler_frame,
    body_frame,
    *,
    interrupted_before_body=False,
    dispatch_results=(),
    memory_before=None,
):
    return PolicyStepPayload(
        scheduler_frame,
        body_frame,
        deepcopy(session.audit_trace),
        interrupted_before_body,
        tuple(dispatch_results),
        memory_before,
    )
