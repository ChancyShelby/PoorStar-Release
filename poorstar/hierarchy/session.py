"""Manage persistent control state, interventions and action decisions."""

import math
from copy import deepcopy
from dataclasses import asdict

import torch

from poorstar.actions.contract import Token
from poorstar.hierarchy.observer import TaskEffectObserver
from poorstar.hierarchy.state import (
    DispatchObservation,
    ExecutionFeedback,
    ExternalOverride,
    HierarchyState,
)
from poorstar.models.policy import BCPolicySession, EncodedObservation


class HierarchyPolicySession(BCPolicySession):
    def __init__(self, model, profile, *, deterministic=False):
        super().__init__(model, profile, deterministic=deterministic)
        self.control_state = None
        self.pending_feedback = []
        self.pending_override = None
        self.audit_trace = None
        self._action_log_prob_tensor = model._tensor(0.0)
        self.dispatch_observation = None
        self.task_observer = None
        self._task_frame = None

    def reset(self, episode):
        if not isinstance(episode, str) or not episode:
            raise ValueError("explicit episode identity required")
        super().reset(episode)
        self.control_state = HierarchyState(episode)
        self.pending_feedback = []
        self.pending_override = None
        self.audit_trace = None
        self.action_log_prob = 0.0
        self._action_log_prob_tensor = self.model._tensor(0.0)
        self.dispatch_observation = None
        self.task_observer = None
        self._task_frame = None

    def submit_dispatch_observation(self, receipt):
        if self.model.hierarchy_config.common_receipts_mode != "shared_v1" or not isinstance(
            receipt, DispatchObservation
        ):
            raise ValueError("declared task-independent dispatch channel required")
        receipt.validate(receipt.loop)
        if self.dispatch_observation is not None and receipt.loop < self.dispatch_observation.loop:
            raise ValueError("common dispatch receipt clock cannot regress")
        self.dispatch_observation = receipt

    def observe_tasks(self, frame):
        """Call on each actual current scheduler/body observation, before choice.

        Returns public evidenced outcomes for the immutable rollout ledger. Only
        progress/confirmed enter pending Task feedback; censored is never a
        negative target. This hook chooses no action and invokes no engine call.
        """
        if not self.model.hierarchy_config.grounded_tasks or self.control_state is None:
            raise ValueError("reset grounded Task session before observation")
        if self.task_observer is None:
            self.task_observer = TaskEffectObserver(
                self.model.hierarchy_config, episode=self.episode, player_id=frame.key.player_id
            )
        self.task_observer.synchronize(self.control_state.tasks)
        outcomes = self.task_observer.observe(frame)
        self._task_frame = frame
        self.pending_feedback.extend(
            event for outcome in outcomes if (event := outcome.to_feedback()) is not None
        )
        return outcomes

    def _allocations(self, observation):
        if not self.model.hierarchy_config.grounded_tasks:
            return None
        if self._task_frame is None or self._task_frame.key.loop != observation["observation_loop"]:
            raise ValueError("grounded Task inference requires its current environment frame hook")
        return self.task_observer.current_allocations(self._task_frame)

    def _register_tasks(self, result):
        if not self.model.hierarchy_config.grounded_tasks:
            return
        self.task_observer.synchronize(result.control_state.tasks)
        records = result.control["tasks"]
        if records is None:
            return
        tasks = {t.slot: t for t in result.control_state.tasks}
        for slot, record in enumerate(records):
            if record["operation"] == "assign":
                self.task_observer.register(self._task_frame, tasks[slot], record["unit_slots"])

    def submit_feedback(self, *events):
        if any(not isinstance(e, ExecutionFeedback) for e in events):
            raise ValueError("typed evidence-backed task feedback required")
        if self.model.hierarchy_config.grounded_tasks and any(
            e.status in ("progress", "confirmed") for e in events
        ):
            raise ValueError(
                "grounded goal progress/confirmation must come from the current task observer"
            )
        self.pending_feedback.extend(events)

    def intervene(self, override):
        if self.model.hierarchy_config.arm != "persistent_intent":
            raise ValueError("external intervention requires the Intent arm")
        if not isinstance(override, ExternalOverride):
            raise ValueError("typed external intervention required")
        override.validate(self.model.hierarchy_config)
        if self.pending_override is not None:
            raise ValueError("pending intervention must be consumed before replacement")
        self.pending_override = override

    @torch.no_grad()
    def choose_schedule(self, observation, maximum_delay):
        if self.control_state is None or self.episode is None:
            raise ValueError("reset session with episode identity before inference")
        if maximum_delay != self.model.config.maximum_delay:
            raise ValueError("online/training delay horizon mismatch")
        self.body = self.prefix = None
        prior = self.control_state.to_dict()
        feedback = tuple(self.pending_feedback)
        override = self.pending_override
        allocations = self._allocations(observation)
        result = self.model.encode_control(
            observation,
            self.hidden,
            self.control_state,
            feedback=feedback,
            external_override=override,
            deterministic=self.deterministic,
            allow_manager=True,
            dispatch_observation=self.dispatch_observation,
            task_allocations=allocations,
        )
        self.control_state = result.control_state
        self._register_tasks(result)
        self.pending_feedback = []
        self.pending_override = None
        act, _ = self.model.scheduler(result.encoded)
        i = int(act.logits.argmax()) if self.deterministic else int(act.sample())
        decision = ("ACT", "WAIT")[i]
        _, delays = self.model.scheduler(result.encoded, decision)
        delay = int(delays.logits.argmax()) if self.deterministic else int(delays.sample())
        self.hidden = self.model.consume_schedule(result.encoded, delay)
        self._action_log_prob_tensor = result.control_log_prob + act.log_prob(
            torch.tensor(i, device=self.model.device)
        )
        self._action_log_prob_tensor = self._action_log_prob_tensor + delays.log_prob(
            torch.tensor(delay, device=self.model.device)
        )
        self.action_log_prob = float(self._action_log_prob_tensor)
        self.audit_trace = dict(
            prior_control_state=prior,
            scheduler_loop=observation["observation_loop"],
            decision=decision,
            delay=delay,
            scheduler_control=deepcopy(result.control),
            body_control=None,
            feedback=[asdict(e) for e in feedback],
            external_override=None if override is None else asdict(override),
            body_tokens=[],
            body_execution_task=None,
            manager_decision=result.manager_decision,
            manager_log_prob=float(result.manager_log_prob),
            task_log_prob=float(result.task_log_prob),
            value=float(result.value),
            manager_value=float(result.manager_value),
            action_log_prob=self.action_log_prob,
        )
        if self.model.hierarchy_config.common_receipts_mode == "shared_v1":
            self.audit_trace.update(
                scheduler_dispatch_observation=None
                if self.dispatch_observation is None
                else asdict(self.dispatch_observation),
                body_dispatch_observation=None,
            )
        if self.model.hierarchy_config.grounded_tasks:
            self.audit_trace.update(
                scheduler_task_allocations=deepcopy(allocations),
                body_task_allocations=None,
                body_feedback=[],
            )
        return decision, delay

    @torch.no_grad()
    def choose_body(self, observation, head, options, completed):
        if self.audit_trace is None:
            raise ValueError("body requires a preceding scheduler decision")
        if self.body is None:
            allocations = self._allocations(observation)
            # New visible goal observations at the delayed body phase remain
            # causal; the scorer must replay the recorded body receipts too.
            feedback = (
                tuple(self.pending_feedback) if self.model.hierarchy_config.grounded_tasks else ()
            )
            result = self.model.encode_control(
                observation,
                self.hidden,
                self.control_state,
                deterministic=self.deterministic,
                allow_manager=False,
                feedback=feedback,
                dispatch_observation=self.dispatch_observation,
                task_allocations=allocations,
            )
            self.control_state = result.control_state
            if self.model.hierarchy_config.grounded_tasks:
                self.pending_feedback = []
            self._register_tasks(result)
            self.body, task_lp, _, task_id = self.model.select_execution_task(
                result, deterministic=self.deterministic
            )
            self.prefix = self.body.hidden
            self.last_body_loop = observation["observation_loop"]
            self.audit_trace["body_control"] = deepcopy(result.control)
            self.audit_trace["body_execution_task"] = task_id
            self.audit_trace["task_log_prob"] += float(result.task_log_prob + task_lp)
            self._action_log_prob_tensor = self._action_log_prob_tensor + result.control_log_prob
            self._action_log_prob_tensor = self._action_log_prob_tensor + task_lp
            self.action_log_prob = float(self._action_log_prob_tensor)
            if self.model.hierarchy_config.common_receipts_mode == "shared_v1":
                self.audit_trace["body_dispatch_observation"] = (
                    None if self.dispatch_observation is None else asdict(self.dispatch_observation)
                )
            if self.model.hierarchy_config.grounded_tasks:
                self.audit_trace["body_task_allocations"] = deepcopy(allocations)
                self.audit_trace["body_feedback"] = [asdict(e) for e in feedback]
        elif observation["observation_loop"] != self.last_body_loop:
            raise ValueError("body observation phase changed")
        dist = self.model.distribution(self.body, self.prefix, head, options)
        if head == "target_position":
            value = dist.choose(self.deterministic)
            lp = dist.log_prob(value)
        else:
            i = int(dist.logits.argmax()) if self.deterministic else int(dist.sample())
            value = options[i]
            lp = dist.log_prob(torch.tensor(i, device=self.model.device))
        self._action_log_prob_tensor = self._action_log_prob_tensor + lp
        self.action_log_prob = float(self._action_log_prob_tensor)
        self.prefix = self.model.consume(self.body, self.prefix, Token(head, value))
        self.hidden = self.prefix
        self.audit_trace["body_tokens"].append(dict(head=head, options=tuple(options), value=value))
        self.audit_trace["action_log_prob"] = self.action_log_prob
        return value

    def state_dict(self):
        def tensor(x):
            return None if x is None else x.detach().cpu().clone()

        body = (
            None
            if self.body is None
            else dict(
                hidden=tensor(self.body.hidden),
                entities=tensor(self.body.entities),
                map_size=self.body.map_size,
                parallel_command_base=tensor(self.body.parallel_command_base),
                parallel_source_state=tensor(self.body.parallel_source_state),
            )
        )
        result = dict(
            schema_version="poorstar_v3_hierarchy_session_v1",
            model_config_sha256=self.model.config.sha256,
            hierarchy_config_sha256=self.model.hierarchy_config.sha256,
            deterministic=self.deterministic,
            episode=self.episode,
            hidden=tensor(self.hidden),
            prefix=tensor(self.prefix),
            body=body,
            last_body_loop=self.last_body_loop,
            control_state=None if self.control_state is None else self.control_state.to_dict(),
            pending_feedback=[asdict(e) for e in self.pending_feedback],
            pending_override=None
            if self.pending_override is None
            else asdict(self.pending_override),
            audit_trace=deepcopy(self.audit_trace),
            action_log_prob=self.action_log_prob,
        )
        if self.model.hierarchy_config.common_receipts_mode == "shared_v1":
            result["dispatch_observation"] = (
                None if self.dispatch_observation is None else asdict(self.dispatch_observation)
            )
        if self.model.hierarchy_config.grounded_tasks:
            result["task_observer"] = (
                None if self.task_observer is None else self.task_observer.state_dict()
            )
        return result

    def load_state_dict(self, data):
        if set(data) != set(self.state_dict()):
            raise ValueError("session recovery schema drift")
        if (
            data["schema_version"] != "poorstar_v3_hierarchy_session_v1"
            or data["model_config_sha256"] != self.model.config.sha256
            or data["hierarchy_config_sha256"] != self.model.hierarchy_config.sha256
            or data["deterministic"] != self.deterministic
        ):
            raise ValueError("session recovery configuration mismatch")
        h = self.model.config.hidden_size

        def tensor(x, *, entity=False):
            if x is None:
                return None
            if not isinstance(x, torch.Tensor) or not torch.isfinite(x).all():
                raise ValueError("finite tensor recovery required")
            if (x.ndim != 2 or x.shape[-1] != h) if entity else x.shape != (h,):
                raise ValueError("recurrent/entity recovery shape drift")
            return x.to(device=self.model.device, dtype=next(self.model.parameters()).dtype).clone()

        state = (
            None
            if data["control_state"] is None
            else HierarchyState.from_dict(data["control_state"], self.model.hierarchy_config)
        )
        if (state is None) != (data["episode"] is None) or (
            state is not None and state.episode != data["episode"]
        ):
            raise ValueError("session recovery episode mismatch")
        body = data["body"]
        if body is not None:
            if set(body) != {
                "hidden",
                "entities",
                "map_size",
                "parallel_command_base",
                "parallel_source_state",
            }:
                raise ValueError("body recovery schema drift")
            size = tuple(body["map_size"])
            if len(size) != 2 or any(type(n) is not int or n < 1 for n in size):
                raise ValueError("body recovery map dimensions drift")
            body = EncodedObservation(
                tensor(body["hidden"]),
                tensor(body["entities"], entity=True),
                size,
                tensor(body["parallel_command_base"]),
                tensor(body["parallel_source_state"]),
            )
        feedback = [ExecutionFeedback(**e) for e in data["pending_feedback"]]
        override = data["pending_override"]
        if override is not None:
            override = dict(override)
            override["preference"] = tuple(override["preference"])
            override = ExternalOverride(**override)
            override.validate(self.model.hierarchy_config)
        if not math.isfinite(data["action_log_prob"]):
            raise ValueError("finite recovered log probability required")
        hidden = tensor(data["hidden"])
        prefix = tensor(data["prefix"])
        self.episode = data["episode"]
        self.hidden = hidden
        self.prefix = prefix
        self.body = body
        self.last_body_loop = data["last_body_loop"]
        self.control_state = state
        self.pending_feedback = feedback
        self.pending_override = override
        self.audit_trace = deepcopy(data["audit_trace"])
        self.action_log_prob = data["action_log_prob"]
        self._action_log_prob_tensor = self.model._tensor(data["action_log_prob"])
        if self.model.hierarchy_config.common_receipts_mode == "shared_v1":
            receipt = data["dispatch_observation"]
            self.dispatch_observation = (
                None if receipt is None else DispatchObservation.from_dict(receipt)
            )
            if self.dispatch_observation is not None:
                self.dispatch_observation.validate(self.dispatch_observation.loop)
        if self.model.hierarchy_config.grounded_tasks:
            value = data["task_observer"]
            self.task_observer = None
            if value is not None:
                self.task_observer = TaskEffectObserver(
                    self.model.hierarchy_config, episode=self.episode, player_id=value["player_id"]
                )
                self.task_observer.load_state_dict(value)
            self._task_frame = None
