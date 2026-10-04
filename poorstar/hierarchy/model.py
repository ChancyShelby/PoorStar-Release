"""Condition the shared action policy on Intent and Task decisions."""

import math
from dataclasses import dataclass, replace

import torch
from torch import nn
from torch.distributions import Beta, Categorical

from poorstar.hierarchy.goals import TaskGoal, current_own_slots
from poorstar.hierarchy.observer import completed_count, entity_region, region_visibility
from poorstar.hierarchy.state import (
    DISPATCH_STATUSES,
    FEEDBACK_STATUSES,
    TASK_STATUSES,
    DispatchObservation,
    ExecutionFeedback,
    ExternalOverride,
    HierarchyConfig,
    HierarchyState,
    IntentState,
    TaskState,
    advance_state,
)
from poorstar.models.policy import EncodedObservation, RecurrentBCPolicy

_SAMPLE = object()


def _ordinary_attention_input(module, arguments):
    # Frozen policy stages still need the SAME actor/teacher attention algebra.
    # A temporary differentiable feature leaf disables Torch's fused inference
    # path even when every encoder parameter is frozen. No persistent parameter
    # or dummy budget-matching parameter is added.
    if torch.is_grad_enabled() and not arguments[0].requires_grad:
        return (arguments[0].detach().requires_grad_(True), *arguments[1:])
    return None


@dataclass
class HierarchyEncoding:
    encoded: EncodedObservation
    memory_hidden: torch.Tensor
    control_state: HierarchyState
    control_log_prob: torch.Tensor
    control_entropy: torch.Tensor
    manager_log_prob: torch.Tensor
    manager_entropy: torch.Tensor
    task_log_prob: torch.Tensor
    task_entropy: torch.Tensor
    value: torch.Tensor
    manager_value: torch.Tensor
    control: dict
    manager_decision: bool
    task_allocations: dict | None = None


class HierarchicalPolicy(RecurrentBCPolicy):
    def __init__(self, config, hierarchy_config=HierarchyConfig()):
        super().__init__(config)
        self.hierarchy_config = hierarchy_config
        self.entity_attention.register_forward_pre_hook(_ordinary_attention_input)
        h = config.hidden_size
        c = hierarchy_config
        if c.common_receipts_mode == "shared_v1":
            self.dispatch_projection = nn.Linear(len(DISPATCH_STATUSES) + 1, h, bias=False)
        self.value_head = nn.Linear(h, 1)
        self.manager_value_head = nn.Linear(h, 1)
        if c.arm != "flat":
            self.value_context = nn.Sequential(
                nn.Linear((3 if c.arm == "persistent_intent" and c.persistence else 2) * h, h),
                nn.Tanh(),
            )
        if c.arm != "flat":
            self.task_embedding = nn.Embedding(len(c.task_types), h)
            self.task_region_embedding = nn.Embedding(math.prod(c.target_regions) + 1, h)
            self.task_state_projection = nn.Linear(5 + len(TASK_STATUSES), h)
            self.feedback_projection = nn.Linear(len(FEEDBACK_STATUSES), h, bias=False)
            self.task_input = nn.Sequential(
                nn.Linear((3 if c.arm == "persistent_intent" else 2) * h, h), nn.Tanh()
            )
            self.task_policy_core = nn.GRUCell(h, h)
            self.task_slot_embedding = nn.Embedding(c.task_slots, h)
            self.task_operation_head = nn.Linear(h, 3)
            self.task_type_head = nn.Linear(h, len(c.task_types))
            self.task_horizon_head = nn.Linear(h, len(c.task_horizons))
            self.task_priority_head = nn.Linear(h, len(c.preference_levels))
            self.task_region_head = nn.Linear(h, math.prod(c.target_regions) + 1)
            self.task_condition = nn.Sequential(nn.Linear(2 * h, 2 * h), nn.Tanh())
            self.task_choice_embedding = nn.Embedding(3, h)
            self.task_execution_query = nn.Linear(h, h)
            self.task_idle_embedding = nn.Parameter(torch.randn(h) * 0.02)
            if c.grounded_tasks:
                self.task_goal_kind_head = nn.Linear(h, len(c.goal_kinds))
                self.task_goal_unit_head = nn.Linear(h, len(config.unit_types))
                if config.upgrades:
                    self.task_goal_upgrade_head = nn.Linear(h, len(config.upgrades))
                self.task_goal_quantity_head = nn.Linear(h, len(c.goal_quantities))
                self.task_goal_region_head = nn.Linear(h, math.prod(c.target_regions))
                self.task_goal_embedding = nn.Embedding(len(c.goal_kinds), h)
                self.task_goal_unit_embedding = nn.Embedding(len(config.unit_types), h)
                if config.upgrades:
                    self.task_goal_upgrade_embedding = nn.Embedding(len(config.upgrades), h)
                self.task_goal_projection = nn.Linear(2, h)
                self.task_allocation_query = nn.Linear(h, h)
                self.task_allocation_end = nn.Parameter(torch.randn(h) * 0.02)
                self.task_allocation_core = nn.GRUCell(h, h)
                self.task_unit_context = nn.Linear(h, h, bias=False)
        if c.arm == "persistent_intent":
            self.intent_projection = nn.Sequential(
                nn.Linear(len(c.preference_names) + (4 if c.persistence else 0), h), nn.Tanh()
            )
            self.manager_input = nn.Sequential(
                nn.Linear((4 if c.persistence else 3) * h, h), nn.Tanh()
            )
            self.manager_preference_head = nn.Linear(
                h, len(c.preference_names) * len(c.preference_levels)
            )
            if c.persistence and c.commitment_mode == "minimum_hold":
                self.manager_operation_head = nn.Linear(h, 3)
                self.manager_horizon_head = nn.Linear(h, len(c.commitment_horizons))
                self.manager_confidence_head = nn.Linear(h, 2)

    def encode(self, observation, hidden=None):
        if (
            self.hierarchy_config.arm != "flat"
            or self.hierarchy_config.memory_mode != "recurrent"
            or self.hierarchy_config.common_receipts_mode != "legacy"
        ):
            raise ValueError("hierarchical policy requires explicit encode_control state/choices")
        return super().encode(observation, hidden)

    def parameter_groups(self):
        """Exhaustive disjoint named parameters for stage-specific optimizers."""
        groups = {k: [] for k in ("backbone", "decoder", "manager", "task", "value")}
        decoder = (
            "ability_embedding.",
            "ability_head.",
            "act_head.",
            "delay_head.",
            "delay_embedding.",
            "schedule_core.",
            "prefix_core.",
            "source_query.",
            "target_query.",
            "end_embedding",
            "end_head.",
            "position_head.",
            "position_embedding.",
            "fixed_heads.",
            "fixed_embeddings.",
        )
        for name, p in self.named_parameters():
            if name.startswith(("value_head.", "manager_value_head.", "value_context.")):
                group = "value"
            elif name.startswith(("manager_", "intent_")):
                group = "manager"
            elif name.startswith(("task_", "feedback_")):
                group = "task"
            elif name.startswith(decoder):
                group = "decoder"
            else:
                group = "backbone"
            groups[group].append((name, p))
        return {k: tuple(v) for k, v in groups.items()}

    def load_c3_foundation(self, source):
        """This is an explicit architecture transition, not exact recovery of an old
        training run. The caller records the parent checkpoint/source identities.
        """
        if not isinstance(source, RecurrentBCPolicy) or source.config != self.config:
            raise ValueError("matching C3 foundation configuration required")
        if isinstance(source, HierarchicalPolicy):
            raise ValueError("foundation transfer requires the original C3 policy")
        own = self.state_dict()
        shared = source.state_dict()
        if not set(shared).issubset(own) or any(own[k].shape != v.shape for k, v in shared.items()):
            raise ValueError("C3 foundation parameter contract drift")
        own.update({k: v.detach().clone() for k, v in shared.items()})
        self.load_state_dict(own, strict=True)
        return {
            "shared_parameter_names": tuple(sorted(shared)),
            "new_parameter_names": tuple(sorted(set(own) - set(shared))),
            "transfer_kind": "NEW_ARCHITECTURE_INITIALIZATION",
        }

    def _zero(self):
        return self.value_head.weight.sum() * 0.0

    def _encode_shared(self, observation, hidden=None, *, phase="scheduler"):

        if phase not in ("scheduler", "body"):
            raise ValueError("explicit scheduler/body encoding phase required")
        if self.hierarchy_config.memory_mode == "feedforward" and phase == "scheduler":
            # Cross-decision recurrence is the sole foundation ablation. Within
            # ACT, chosen delay/body/token state remains fully autoregressive.
            hidden = None
        gradient = torch.is_grad_enabled()
        with torch.enable_grad():
            encoded = super().encode(observation, hidden)
        if not gradient:
            encoded = EncodedObservation(
                encoded.hidden.detach(), encoded.entities.detach(), encoded.map_size
            )
        return encoded

    def _values(self, base, state, *, entities=None, task_allocations=None):
        if self.hierarchy_config.arm == "flat":
            context = base
        else:
            parts = [
                base,
                self._task_summary(state, entities=entities, task_allocations=task_allocations),
            ]
            if (
                self.hierarchy_config.arm == "persistent_intent"
                and self.hierarchy_config.persistence
            ):
                parts.append(self._intent_features(state.intent))
            context = self.value_context(torch.cat(parts))
        return self.value_head(context).squeeze(-1), self.manager_value_head(context).squeeze(-1)

    def _validate_allocations(self, state, observation, allocations):
        c = self.hierarchy_config
        if not c.grounded_tasks:
            if allocations is not None:
                raise ValueError("allocation input requires declared grounded Tasks")
            return None
        if allocations is not None and not isinstance(allocations, dict):
            raise ValueError("explicit current task allocation mapping required")
        mapping = {} if allocations is None else allocations
        known = {t.task_id: t for t in state.tasks}
        active = {k: t for k, t in known.items() if t.active}
        own = current_own_slots(observation)
        if set(mapping) - set(known) or any(
            t.allocation_size and k not in mapping for k, t in active.items()
        ):
            raise ValueError("grounded Task phase requires current frame allocation rebinding")
        members = []
        result = {}
        for task_id, slots in mapping.items():
            if (
                not isinstance(slots, (tuple, list))
                or len(slots) > known[task_id].allocation_size
                or any(type(s) is not int or s not in own for s in slots)
            ):
                raise ValueError("allocation rebinding outside current own-visible frame")
            if task_id in active:
                result[task_id] = tuple(slots)
                members.extend(slots)
        if len(set(members)) != len(members):
            raise ValueError("overlapping current task/squad unit ownership")
        for task in state.tasks:
            task.goal.validate(c, unit_types=self.config.unit_types, upgrades=self.config.upgrades)
        return result

    def _with_dispatch(self, encoded, observation, receipt):
        c = self.hierarchy_config
        if receipt is not None:
            if c.common_receipts_mode != "shared_v1" or not isinstance(
                receipt, DispatchObservation
            ):
                raise ValueError("declared typed common dispatch observation required")
            receipt.validate(observation["observation_loop"])
        if c.common_receipts_mode == "shared_v1" and c.use_feedback and receipt is not None:
            features = [math.log1p(n) for n in receipt.counts] + [
                math.log1p(observation["observation_loop"] - receipt.loop)
            ]
            return EncodedObservation(
                encoded.hidden + self.dispatch_projection(self._tensor(features)),
                encoded.entities,
                encoded.map_size,
            )
        return encoded

    def evaluate_values(
        self,
        observation,
        hidden=None,
        control_state=None,
        *,
        feedback=(),
        dispatch_observation=None,
        task_allocations=None,
    ):
        """Visible-state bootstrap without sampling/advancing strategic controls."""
        if self.hierarchy_config.arm != "flat" and not isinstance(control_state, HierarchyState):
            raise ValueError("hierarchical bootstrap requires explicit existing control state")
        state = control_state
        if state is not None:
            state, _ = advance_state(
                state, observation["observation_loop"], self.hierarchy_config, feedback
            )
        elif feedback:
            raise ValueError("flat bootstrap cannot receive task feedback")
        task_allocations = self._validate_allocations(state, observation, task_allocations)
        encoded = self._with_dispatch(
            self._encode_shared(observation, hidden), observation, dispatch_observation
        )
        return self._values(
            encoded.hidden, state, entities=encoded.entities, task_allocations=task_allocations
        )

    def confidence_prediction(
        self,
        observation,
        hidden=None,
        control_state=None,
        *,
        feedback=(),
        dispatch_observation=None,
        task_allocations=None,
    ):
        """Differentiable declared-event Beta mean from pre-choice visible context.

        This does not sample controls or consume outcome targets. Calibration and
        observed-event label masks are separate caller-owned auxiliary learning.
        """
        c = self.hierarchy_config
        if (
            c.arm != "persistent_intent"
            or not c.persistence
            or c.manager_mode != "learned"
            or c.commitment_mode != "minimum_hold"
            or not isinstance(control_state, HierarchyState)
        ):
            raise ValueError("configuration has no learned confidence prediction head")
        state, events = advance_state(control_state, observation["observation_loop"], c, feedback)
        task_allocations = self._validate_allocations(state, observation, task_allocations)
        encoded = self._with_dispatch(
            self._encode_shared(observation, hidden), observation, dispatch_observation
        )
        latent = self.manager_input(
            torch.cat(
                (
                    encoded.hidden,
                    self._intent_features(state.intent),
                    self._task_summary(
                        state, entities=encoded.entities, task_allocations=task_allocations
                    ),
                    self._feedback_features(events),
                )
            )
        )
        parameters = torch.nn.functional.softplus(self.manager_confidence_head(latent)) + 1.0
        return parameters[0] / parameters.sum()

    def select_execution_task(self, encoding, *, teacher_task_id=_SAMPLE, deterministic=False):
        """Select a task for this body bundle, with an explicit unassigned option."""
        c = self.hierarchy_config
        if c.arm == "flat":
            if teacher_task_id not in (_SAMPLE, None):
                raise ValueError("flat policy cannot associate a task")
            return encoding.encoded, self._zero(), self._zero(), None
        tasks = [t for t in encoding.control_state.tasks if t.active]
        embeddings = [
            self._task_features(
                t,
                entities=encoding.encoded.entities,
                members=(encoding.task_allocations or {}).get(t.task_id),
            )
            for t in tasks
        ] + [self.task_idle_embedding]
        vectors = torch.stack(embeddings)
        intent = (
            self._intent_features(encoding.control_state.intent)
            if c.arm == "persistent_intent"
            else self.zero_state()
        )
        query = self.task_execution_query(encoding.memory_hidden + intent)
        logits = vectors.mv(query) / math.sqrt(len(query))
        options = tuple(t.task_id for t in tasks) + (None,)
        task_id, lp, entropy = self._choice(logits, options, teacher_task_id, deterministic)
        selected = embeddings[options.index(task_id)]
        base = encoding.memory_hidden
        scale, shift = self.task_condition(torch.cat((base, selected))).chunk(2)
        conditioned = base * (1.0 + 0.1 * scale) + 0.1 * shift
        return (
            EncodedObservation(conditioned, encoding.encoded.entities, encoding.encoded.map_size),
            lp,
            entropy,
            task_id,
        )

    def _intent_features(self, intent):
        c = self.hierarchy_config
        if intent is None or not intent.active:
            return self.zero_state()
        lifecycle = (
            (
                math.log1p(intent.commitment_horizon),
                math.log1p(intent.age),
                intent.confidence,
                float(intent.active),
            )
            if c.persistence
            else ()
        )
        values = (*intent.preference, *lifecycle)
        return self.intent_projection(self._tensor(values))

    def _task_features(self, task, *, entities=None, members=None):
        c = self.hierarchy_config
        status = [float(task.status == s) for s in TASK_STATUSES]
        values = (
            math.log1p(task.horizon),
            math.log1p(task.age),
            task.resource_priority,
            task.progress,
            float(task.active),
            *status,
        )
        region = math.prod(c.target_regions) if task.target_region is None else task.target_region
        result = (
            self.task_embedding(
                torch.tensor(c.task_types.index(task.task_type), device=self.device)
            )
            + self.task_region_embedding(torch.tensor(region, device=self.device))
            + self.task_state_projection(self._tensor(values))
        )
        if c.grounded_tasks:
            goal = task.goal
            result = result + self.task_goal_embedding(
                self._tensor(c.goal_kinds.index(goal.kind)).long()
            )
            if goal.kind == "own_completed_unit_delta":
                result = result + self.task_goal_unit_embedding(
                    self._tensor(self.config.unit_types.index(goal.subject)).long()
                )
            elif goal.kind == "upgrade_present":
                result = result + self.task_goal_upgrade_embedding(
                    self._tensor(self.config.upgrades.index(goal.subject)).long()
                )
            if goal.region is not None:
                result = result + self.task_region_embedding(self._tensor(goal.region).long())
            result = result + self.task_goal_projection(
                self._tensor((math.log1p(goal.quantity or 0), math.log1p(task.allocation_size)))
            )
            if entities is not None and task.active and task.allocation_size:
                if members is None:
                    raise ValueError(
                        "current task-unit context requires explicit allocation rebinding"
                    )
                if members:
                    result = result + self.task_unit_context(entities[list(members)].mean(0))
        return result

    def _task_summary(self, state, *, active_only=False, entities=None, task_allocations=None):
        tasks = [t for t in state.tasks if t.active or not active_only]
        return (
            torch.stack(
                [
                    self._task_features(
                        t, entities=entities, members=(task_allocations or {}).get(t.task_id)
                    )
                    for t in tasks
                ]
            ).mean(0)
            if tasks
            else self.zero_state()
        )

    def _feedback_features(self, feedback):
        if not self.hierarchy_config.use_feedback or not feedback:
            return self.zero_state()
        counts = [sum(e.status == s for e in feedback) for s in FEEDBACK_STATUSES]
        return self.feedback_projection(self._tensor([math.log1p(n) for n in counts]))

    def _choice(self, logits, options, teacher, deterministic):
        dist = Categorical(logits=logits, validate_args=False)
        if teacher is _SAMPLE:
            index = int(dist.logits.argmax()) if deterministic else int(dist.sample())
            value = options[index]
        else:
            if teacher not in options:
                raise ValueError("control teacher choice outside current causal domain")
            index = options.index(teacher)
            value = teacher
        return value, dist.log_prob(torch.tensor(index, device=self.device)), dist.entropy()

    def _manager_domain(self, state, loop, feedback):
        """Actual causal operation domain; None means no Manager receipt."""
        c = self.hierarchy_config
        if c.commitment_mode == "fixed_dwell":
            active = state.intent is not None and state.intent.active
            return (
                ("keep",)
                if active and state.intent.age < state.intent.commitment_horizon
                else ("update",)
            )
        due = (
            state.last_manager_loop is None
            or not c.persistence
            or loop - state.last_manager_loop >= c.manager_period
        )
        if not (due or feedback):
            return None
        active = state.intent is not None and state.intent.active
        if active and c.persistence:
            if state.intent.age < state.intent.commitment_horizon and not feedback:
                return ("keep",)
            return ("keep", "update", "terminate")
        return ("update",)

    def manager_decision_eligible(
        self, control_state, loop, *, feedback=(), external_override=None, allow_manager=True
    ):
        """Pure trainable decision eligibility, without encoding or sampling.

        A forced commitment keep is an auditable clock receipt, with zero policy
        density and no new SMDP return segment. A singleton update still samples
        the Preference and therefore is a trainable Manager decision.
        """
        c = self.hierarchy_config
        if type(allow_manager) is not bool or not isinstance(control_state, HierarchyState):
            raise ValueError("explicit strategic state/phase eligibility required")
        state, events = advance_state(control_state, loop, c, feedback)
        if c.arm != "persistent_intent" or not allow_manager or c.manager_mode == "external":
            return False
        if external_override is not None:
            if not isinstance(external_override, ExternalOverride):
                raise ValueError("typed external override required")
            external_override.validate(c)
            return False
        domain = self._manager_domain(state, loop, events)
        return domain is not None and domain != ("keep",)

    def _manager(
        self,
        base,
        state,
        loop,
        feedback,
        override,
        teacher,
        deterministic,
        *,
        entities=None,
        task_allocations=None,
    ):
        c = self.hierarchy_config
        zero = self._zero()
        if override is not None:
            if not isinstance(override, ExternalOverride):
                raise ValueError("typed external override required")
            override.validate(c)
            if teacher is not None:
                raise ValueError("external override is an intervention, not a manager action label")
            revision = 1 if state.intent is None else state.intent.revision + 1
            intent = IntentState(
                override.preference,
                override.commitment_horizon,
                loop,
                0,
                override.confidence,
                "external",
                "external",
                revision,
            )
            return replace(state, intent=intent, last_manager_loop=loop), zero, zero, None
        if c.manager_mode == "external":
            if teacher is not None:
                raise ValueError("external-only arm cannot score a learned manager choice")
            if state.intent is None:
                raise ValueError("external-only manager requires an initial intervention")
            if not c.persistence:
                raise ValueError(
                    "stateless external Preference requires a fresh intervention each scheduler"
                )
            if (
                c.commitment_mode == "fixed_dwell"
                and state.intent.age >= state.intent.commitment_horizon
            ):
                raise ValueError(
                    "fixed-dwell external Preference requires a fresh intervention at expiry"
                )
            return state, zero, zero, None
        operation_options = self._manager_domain(state, loop, feedback)
        if operation_options is None:
            if teacher is not None:
                raise ValueError("manager label outside a scheduled/event decision")
            return state, zero, zero, None
        parts = [base]
        if c.persistence:
            parts.append(self._intent_features(state.intent))
        parts.extend(
            (
                self._task_summary(state, entities=entities, task_allocations=task_allocations),
                self._feedback_features(feedback),
            )
        )
        latent = self.manager_input(torch.cat(parts))
        full = ("keep", "update", "terminate")
        learned_lifecycle = c.persistence and c.commitment_mode == "minimum_hold"
        logits = (
            self.manager_operation_head(latent)[[full.index(v) for v in operation_options]]
            if learned_lifecycle
            else self._tensor([0.0])
        )
        if teacher is not None and (not isinstance(teacher, dict) or "operation" not in teacher):
            raise ValueError("explicit manager action record required")
        op, lp, ent = self._choice(
            logits,
            operation_options,
            _SAMPLE if teacher is None else teacher["operation"],
            deterministic,
        )
        record = {"operation": op}
        if op == "update":
            expected = (
                {"operation", "preference", "commitment_horizon", "confidence"}
                if learned_lifecycle
                else {"operation", "preference"}
            )
            if teacher is not None and set(teacher) != expected:
                raise ValueError("manager update record schema drift")
            logits = self.manager_preference_head(latent).reshape(len(c.preference_names), -1)
            preference = []
            if teacher is not None and len(teacher["preference"]) != len(c.preference_names):
                raise ValueError("teacher Preference dimension drift")
            for i, row in enumerate(logits):
                z, p, e = self._choice(
                    row,
                    c.preference_levels,
                    _SAMPLE if teacher is None else teacher["preference"][i],
                    deterministic,
                )
                preference.append(z)
                lp = lp + p
                ent = ent + e
            if learned_lifecycle:
                horizon, p, e = self._choice(
                    self.manager_horizon_head(latent),
                    c.commitment_horizons,
                    _SAMPLE if teacher is None else teacher["commitment_horizon"],
                    deterministic,
                )
                lp = lp + p
                ent = ent + e
                parameters = (
                    torch.nn.functional.softplus(self.manager_confidence_head(latent)) + 1.0
                )
                dist = Beta(parameters[0], parameters[1], validate_args=False)
                if teacher is None:
                    confidence = (
                        float(dist.mean.detach()) if deterministic else float(dist.sample())
                    )
                else:
                    confidence = teacher["confidence"]
                if (
                    not isinstance(confidence, (float, int))
                    or not math.isfinite(confidence)
                    or not 0 < confidence < 1
                ):
                    raise ValueError("learned confidence density requires an interior probability")
                lp = lp + dist.log_prob(self._tensor(confidence))
                ent = ent + dist.entropy()
            else:
                # Clock/confidence bookkeeping is explicit, not a prediction:
                # fixed dwell uses its registered horizon, stepwise uses one.
                horizon = c.fixed_commitment_horizon if c.commitment_mode == "fixed_dwell" else 1
                confidence = 0.5
            revision = 1 if state.intent is None else state.intent.revision + 1
            intent = IntentState(
                tuple(preference),
                horizon,
                loop,
                0,
                float(confidence),
                "update",
                "learned",
                revision,
            )
            record.update(preference=preference)
            if learned_lifecycle:
                record.update(commitment_horizon=horizon, confidence=float(confidence))
        else:
            if teacher is not None and set(teacher) != {"operation"}:
                raise ValueError("keep/terminate record cannot contain hindsight parameters")
            intent = replace(state.intent, update_or_termination=op, active=(op != "terminate"))
        return replace(state, intent=intent, last_manager_loop=loop), lp, ent, record

    def _goal_options(self, observation, own):
        c = self.hierarchy_config
        by_slot = {e["slot"]: e for e in observation["entities"]}
        regions = tuple(range(math.prod(c.target_regions)))
        kind_options = []
        for kind in c.goal_kinds:
            if kind == "own_completed_unit_delta":
                valid = any(
                    completed_count(observation, t) is not None for t in self.config.unit_types
                )
            elif kind == "upgrade_present":
                valid = observation["upgrades"] is not None and any(
                    u not in observation["upgrades"] for u in self.config.upgrades
                )
            elif kind == "allocated_region":
                valid = (
                    bool(own)
                    and all(
                        entity_region(by_slot[s], observation, c.target_regions) is not None
                        for s in own
                    )
                    and len(regions) > 1
                )
            else:
                valid = any(
                    (f := region_visibility(observation, r, c.target_regions)) is not None
                    and f < 1.0
                    for r in regions
                )
            if valid:
                kind_options.append(kind)
        return tuple(kind_options)

    def _ground_task(self, latent, observation, entities, reserved, teacher, deterministic):
        """Probability-bearing goals and unit membership on one causal frame."""
        c = self.hierarchy_config
        lp = self._zero()
        entropy = self._zero()
        own = tuple(s for s in current_own_slots(observation) if s not in reserved)
        by_slot = {e["slot"]: e for e in observation["entities"]}
        regions = tuple(range(math.prod(c.target_regions)))
        kind_options = self._goal_options(observation, own)
        if not kind_options:
            raise ValueError("no observable unsatisfied typed goal in current Task domain")
        qgoal = None if teacher is None else TaskGoal.from_dict(teacher["goal"])
        if qgoal is not None:
            qgoal.validate(c, unit_types=self.config.unit_types, upgrades=self.config.upgrades)
        logits = self.task_goal_kind_head(latent)[[c.goal_kinds.index(k) for k in kind_options]]
        kind, p, e = self._choice(
            logits, tuple(kind_options), _SAMPLE if qgoal is None else qgoal.kind, deterministic
        )
        lp = lp + p
        entropy = entropy + e
        slots = []
        qslots = None if teacher is None else teacher["unit_slots"]
        if qslots is not None and (
            not isinstance(qslots, (tuple, list))
            or len(set(qslots)) != len(qslots)
            or len(qslots) > c.maximum_task_units
            or any(type(s) is not int for s in qslots)
        ):
            raise ValueError("explicit unique bounded current allocation sequence required")
        allocation_latent = latent
        while True:
            remaining = tuple(s for s in own if s not in slots)
            must_select = kind == "allocated_region" and not slots
            options = remaining if must_select else remaining + (None,)
            if len(slots) >= c.maximum_task_units:
                options = (None,)
            vectors = torch.stack(
                [entities[s] if s is not None else self.task_allocation_end for s in options]
            )
            query = self.task_allocation_query(allocation_latent)
            logits = vectors.mv(query) / math.sqrt(len(query))
            wanted = (
                _SAMPLE
                if qslots is None
                else qslots[len(slots)]
                if len(slots) < len(qslots)
                else None
            )
            chosen, p, e = self._choice(logits, options, wanted, deterministic)
            lp = lp + p
            entropy = entropy + e
            if chosen is None:
                break
            slots.append(chosen)
            allocation_latent = self.task_allocation_core(entities[chosen], allocation_latent)
        # Goal parameters are conditioned on the selected members, not on a
        # strategic script or the later decoded action.
        latent = allocation_latent
        subject = quantity = region = None
        if kind == "own_completed_unit_delta":
            domain = tuple(
                t for t in self.config.unit_types if completed_count(observation, t) is not None
            )
            subject, p, e = self._choice(
                self.task_goal_unit_head(latent)[[self.config.unit_types.index(t) for t in domain]],
                domain,
                _SAMPLE if qgoal is None else qgoal.subject,
                deterministic,
            )
            lp = lp + p
            entropy = entropy + e
            quantity, p, e = self._choice(
                self.task_goal_quantity_head(latent),
                c.goal_quantities,
                _SAMPLE if qgoal is None else qgoal.quantity,
                deterministic,
            )
            lp = lp + p
            entropy = entropy + e
        elif kind == "upgrade_present":
            domain = tuple(u for u in self.config.upgrades if u not in observation["upgrades"])
            subject, p, e = self._choice(
                self.task_goal_upgrade_head(latent)[
                    [self.config.upgrades.index(u) for u in domain]
                ],
                domain,
                _SAMPLE if qgoal is None else qgoal.subject,
                deterministic,
            )
            lp = lp + p
            entropy = entropy + e
        else:
            if kind == "allocated_region":
                domain = tuple(
                    r
                    for r in regions
                    if any(
                        entity_region(by_slot[s], observation, c.target_regions) != r for s in slots
                    )
                )
            else:
                domain = tuple(
                    r
                    for r in regions
                    if (f := region_visibility(observation, r, c.target_regions)) is not None
                    and f < 1.0
                )
            region, p, e = self._choice(
                self.task_goal_region_head(latent)[list(domain)],
                domain,
                _SAMPLE if qgoal is None else qgoal.region,
                deterministic,
            )
            lp = lp + p
            entropy = entropy + e
        goal = TaskGoal(kind, subject, quantity, region)
        goal.validate(c, unit_types=self.config.unit_types, upgrades=self.config.upgrades)
        return goal, tuple(slots), lp, entropy

    def _tasks(
        self,
        base,
        state,
        loop,
        feedback,
        teacher,
        deterministic,
        *,
        observation=None,
        entities=None,
        task_allocations=None,
    ):
        c = self.hierarchy_config
        zero = self._zero()
        due = state.last_task_loop is None or loop - state.last_task_loop >= c.task_period
        if not (due or feedback):
            if teacher is not None:
                raise ValueError("task label outside a scheduled/event decision")
            return state, zero, zero, None, task_allocations
        if teacher is not None and (
            not isinstance(teacher, (list, tuple)) or len(teacher) != c.task_slots
        ):
            raise ValueError("one explicit action per parallel task slot required")
        summary = self._task_summary(state, entities=entities, task_allocations=task_allocations)
        parts = [base, summary]
        if c.arm == "persistent_intent":
            parts.append(self._intent_features(state.intent))
        latent = self.task_input(torch.cat(parts)) + self._feedback_features(feedback)
        slots = {t.slot: t for t in state.tasks}
        result = []
        records = []
        lp = zero
        entropy = zero
        revision = state.task_revision
        allocations = {} if task_allocations is None else dict(task_allocations)
        if c.grounded_tasks:
            active = {t.task_id: t for t in state.tasks if t.active}
            own = current_own_slots(observation)
            if set(allocations) - {t.task_id for t in state.tasks} or any(
                t.allocation_size and t.task_id not in allocations for t in active.values()
            ):
                raise ValueError("grounded Task phase requires current frame allocation rebinding")
            allocations = {k: v for k, v in allocations.items() if k in active}
            all_members = []
            for task_id, members in allocations.items():
                if (
                    not isinstance(members, (tuple, list))
                    or len(members) > active[task_id].allocation_size
                    or any(type(s) is not int or s not in own for s in members)
                ):
                    raise ValueError("allocation rebinding outside current own-visible frame")
                all_members.extend(members)
            if len(set(all_members)) != len(all_members):
                raise ValueError("overlapping current task/squad unit ownership")
        for slot in range(c.task_slots):
            q = None if teacher is None else teacher[slot]
            if q is not None and (not isinstance(q, dict) or "operation" not in q):
                raise ValueError("explicit task slot action required")
            old = slots.get(slot)
            options = (
                ("keep", "assign", "terminate")
                if old is not None and old.active
                else ("keep", "assign")
            )
            if c.grounded_tasks:
                reserved = {
                    s
                    for task_id, members in allocations.items()
                    if old is None or task_id != old.task_id
                    for s in members
                }
                available = tuple(s for s in current_own_slots(observation) if s not in reserved)
                if not self._goal_options(observation, available):
                    options = tuple(op for op in options if op != "assign")
            latent = self.task_policy_core(
                self.task_slot_embedding(torch.tensor(slot, device=self.device)), latent
            )
            all_ops = ("keep", "assign", "terminate")
            logits = self.task_operation_head(latent)[[all_ops.index(v) for v in options]]
            op, p, e = self._choice(
                logits, options, _SAMPLE if q is None else q["operation"], deterministic
            )
            lp = lp + p
            entropy = entropy + e
            record = {"operation": op}
            if op == "assign":
                fields = {"operation", "task_type", "horizon", "resource_priority", "target_region"}
                if c.grounded_tasks:
                    fields.update(("goal", "unit_slots"))
                if q is not None and set(q) != fields:
                    raise ValueError("task assignment record schema drift")
                values = {}
                heads = (
                    (self.task_type_head, "task_type", c.task_types),
                    (self.task_horizon_head, "horizon", c.task_horizons),
                    (self.task_priority_head, "resource_priority", c.preference_levels),
                    (
                        self.task_region_head,
                        "target_region",
                        tuple(range(math.prod(c.target_regions))) + (None,),
                    ),
                )
                for head, name, domain in heads:
                    value, p, e = self._choice(
                        head(latent), domain, _SAMPLE if q is None else q[name], deterministic
                    )
                    values[name] = value
                    lp = lp + p
                    entropy = entropy + e
                revision += 1
                goal = None
                unit_slots = ()
                if c.grounded_tasks:
                    if old is not None:
                        allocations.pop(old.task_id, None)
                    reserved = {s for members in allocations.values() for s in members}
                    goal, unit_slots, p, e = self._ground_task(
                        latent, observation, entities, reserved, q, deterministic
                    )
                    lp = lp + p
                    entropy = entropy + e
                task = TaskState(
                    f"{state.episode}:task:{revision}",
                    slot,
                    values["task_type"],
                    loop,
                    values["horizon"],
                    0,
                    values["resource_priority"],
                    values["target_region"],
                    state.intent.revision
                    if state.intent is not None and state.intent.active
                    else None,
                    goal=goal,
                    squad_id=f"{state.episode}:squad:{revision}" if c.grounded_tasks else None,
                    allocation_size=len(unit_slots),
                )
                if c.grounded_tasks:
                    allocations[task.task_id] = unit_slots
                    record.update(goal=goal.to_dict(), unit_slots=list(unit_slots))
                result.append(task)
                record.update(values)
            else:
                if q is not None and set(q) != {"operation"}:
                    raise ValueError("keep/terminate task record has undeclared parameters")
                if old is not None:
                    result.append(replace(old, status="terminated") if op == "terminate" else old)
                    if c.grounded_tasks and op == "terminate":
                        allocations.pop(old.task_id, None)
            records.append(record)
            latent = self.task_policy_core(
                self.task_choice_embedding(torch.tensor(all_ops.index(op), device=self.device)),
                latent,
            )
        return (
            replace(state, tasks=tuple(result), last_task_loop=loop, task_revision=revision),
            lp,
            entropy,
            records,
            allocations,
        )

    def encode_control(
        self,
        observation,
        hidden=None,
        control_state=None,
        *,
        feedback=(),
        external_override=None,
        teacher_control=None,
        deterministic=False,
        allow_manager=True,
        dispatch_observation=None,
        task_allocations=None,
    ):
        """Pure phase transition. State must belong to the current episode.

        teacher_control is {manager: None|record, tasks: None|slot records}; it is
        consumed solely as sampled-action reevaluation/teacher targets. All logits
        are computed from the current visible observation and preceding state.
        """
        c = self.hierarchy_config
        if type(allow_manager) is not bool:
            raise ValueError("explicit strategic phase eligibility required")
        if not isinstance(control_state, HierarchyState):
            raise ValueError("explicit HierarchyState with episode identity required")
        if teacher_control is not None and set(teacher_control) != {"manager", "tasks"}:
            raise ValueError("control teacher schema drift")
        state, events = advance_state(control_state, observation["observation_loop"], c, feedback)
        manager_decision = (
            c.arm == "persistent_intent"
            and allow_manager
            and c.manager_mode == "learned"
            and external_override is None
            and self._manager_domain(state, observation["observation_loop"], events)
            not in (None, ("keep",))
        )
        encoded = self._with_dispatch(
            self._encode_shared(
                observation, hidden, phase="scheduler" if allow_manager else "body"
            ),
            observation,
            dispatch_observation,
        )
        task_allocations = self._validate_allocations(state, observation, task_allocations)
        base = encoded.hidden
        # Values precede the sampled strategic/task actions, and depend on the
        # currently existing visible controls. They cannot condition on the
        # chosen action whose advantage they will estimate.
        value, manager_value = self._values(
            base, state, entities=encoded.entities, task_allocations=task_allocations
        )
        manager_lp = task_lp = manager_entropy = task_entropy = self._zero()
        manager_record = task_record = None
        if c.arm == "flat":
            if (
                feedback
                or external_override is not None
                or teacher_control not in (None, {"manager": None, "tasks": None})
            ):
                raise ValueError("flat policy cannot receive hierarchical controls/feedback")
        else:
            manager_teacher = None if teacher_control is None else teacher_control["manager"]
            if c.arm == "persistent_intent" and allow_manager:
                state, manager_lp, manager_entropy, manager_record = self._manager(
                    base,
                    state,
                    observation["observation_loop"],
                    events,
                    external_override,
                    manager_teacher,
                    deterministic,
                    entities=encoded.entities,
                    task_allocations=task_allocations,
                )
            elif external_override is not None or manager_teacher is not None:
                raise ValueError("Intent intervention/labels outside a strategic decision phase")
            state, task_lp, task_entropy, task_record, task_allocations = self._tasks(
                base,
                state,
                observation["observation_loop"],
                events,
                None if teacher_control is None else teacher_control["tasks"],
                deterministic,
                observation=observation,
                entities=encoded.entities,
                task_allocations=task_allocations,
            )

            tasks = self._task_summary(
                state,
                active_only=True,
                entities=encoded.entities,
                task_allocations=task_allocations,
            )
            scale, shift = self.task_condition(torch.cat((base, tasks))).chunk(2)
            conditioned = base * (1.0 + 0.1 * scale) + 0.1 * shift
            encoded = EncodedObservation(conditioned, encoded.entities, encoded.map_size)
        state.validate(c)
        if teacher_control is not None:
            if (teacher_control["manager"] is None) != (manager_record is None):
                raise ValueError("manager decision missing from teacher control trace")
            if (teacher_control["tasks"] is None) != (task_record is None):
                raise ValueError("task decisions missing from teacher control trace")
        return HierarchyEncoding(
            encoded,
            base,
            state,
            manager_lp + task_lp,
            manager_entropy + task_entropy,
            manager_lp,
            manager_entropy,
            task_lp,
            task_entropy,
            value,
            manager_value,
            {"manager": manager_record, "tasks": task_record},
            manager_decision,
            task_allocations if c.grounded_tasks else None,
        )


def control_objective(encoding, *, manager=True, tasks=True):
    """Explicit-control BC NLL; caller supplies provenance-qualified labels.

    Calling with sampled controls is a mechanism check, not replay Intent
    supervision. High-level ground truth is never invented by this function.
    """
    result = encoding.control_log_prob * 0.0
    if manager and encoding.manager_decision:
        result = result - encoding.manager_log_prob
    if tasks and encoding.control["tasks"] is not None:
        result = result - encoding.task_log_prob
    return result
