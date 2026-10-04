"""Training: bc."""

import copy
import math
from collections import Counter
from dataclasses import asdict

import torch

from poorstar.common.parameters import parameter_digest
from poorstar.hierarchy.bc import hierarchy_bc_objective
from poorstar.training.contracts import digest
from poorstar.training.foundation import BCAggregate, BCBatchResult
from poorstar.training.learner import groups_for


class HierarchicalBCAggregate(BCAggregate):
    """Hierarchicalbcaggregate."""

    def report(self, *, required_heads=()):
        missing = [h for h in required_heads if self.denominators[h] == 0]
        if missing:
            raise ValueError("zero aggregate supervision denominator: " + ",".join(missing))
        view = copy.copy(self)
        view.denominators = Counter({h: n for h, n in self.denominators.items() if n})
        result = BCAggregate.report(view, required_heads=required_heads)
        result["denominators"] = dict(self.denominators)
        result["unsupervised_heads"] = sorted(h for h, n in self.denominators.items() if not n)
        return result


def validate_bc_contract(
    model,
    profile,
    identity,
    plan,
    *,
    catalog=None,
    maximum_delay=None,
    head_weights=None,
    control_weights=None,
):
    """Reject static BC drift before creating Adam or starting an update."""
    from poorstar.models.policy import HEADS, BCModelConfig
    from poorstar.training.config import model_config_sha256

    heads = {} if head_weights is None else dict(head_weights)
    controls = {} if control_weights is None else dict(control_weights)
    if (
        identity.seed != plan.seed
        or identity.purpose != plan.purpose
        or identity.budget_sha256
        != digest(dict(plan=asdict(plan), head_weights=heads, control_weights=controls))
    ):
        raise ValueError("BC plan/objective/seed identity drift")
    if (
        model.config.hidden_size != plan.hidden_size
        or model.config.autoregressive != plan.autoregressive
        or plan.feature_batch_size != 1
        or profile.sha256 != identity.profile_sha256
        or model_config_sha256(model.config, model.hierarchy_config) != identity.model_config_sha256
    ):
        raise ValueError("BC architecture/feature/action profile plan drift")
    if maximum_delay is not None and model.config.maximum_delay != maximum_delay:
        raise ValueError("BC model/data scheduler horizon drift")
    if catalog is not None:
        options = asdict(model.config)
        for name in ("abilities", "unit_types", "upgrades"):
            options.pop(name)
        if BCModelConfig.from_catalog(catalog, profile, **options) != model.config:
            raise ValueError("BC model/data vocabulary drift")
    for values, allowed in (
        (heads, set(HEADS) | {"act", "delay"}),
        (controls, {"manager_control", "task_control", "execution_task", "confidence_aux"}),
    ):
        if set(values) - allowed or any(
            type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in values.values()
        ):
            raise ValueError("finite declared BC objective weights required")
    if (
        identity.purpose == "FORMAL"
        and model.hierarchy_config.arm == "persistent_intent"
        and "confidence_aux" not in controls
    ):
        raise ValueError(
            "formal Intent BC requires an explicit frozen confidence auxiliary weight, including zero"
        )


class HierarchicalBCLearner:
    def __init__(
        self,
        model,
        profile,
        identity,
        plan,
        *,
        head_weights=None,
        control_weights=None,
        observation_cache=None,
    ):
        validate_bc_contract(
            model,
            profile,
            identity,
            plan,
            head_weights=head_weights,
            control_weights=control_weights,
        )
        self.model, self.profile, self.identity, self.config = model, profile, identity, plan
        self.head_weights = {} if head_weights is None else dict(head_weights)
        self.control_weights = {} if control_weights is None else dict(control_weights)
        expected_budget = digest(
            dict(
                plan=asdict(plan),
                head_weights=self.head_weights,
                control_weights=self.control_weights,
            )
        )
        if (
            identity.budget_sha256 != expected_budget
            or identity.seed != plan.seed
            or identity.purpose != plan.purpose
        ):
            raise ValueError("BC plan/objective/seed identity drift")
        self.observation_cache = observation_cache
        self.groups = groups_for(model)
        self.trainable_names = tuple(
            n for group, members in self.groups.items() if group != "value" for n, _ in members
        )
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(name in self.trainable_names)
        self.optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], lr=plan.learning_rate
        )
        self.updates = self.optimizer_steps = self.policy_version = self.environment_loops = 0
        self.phase = "BC"
        self.aggregate = HierarchicalBCAggregate()
        self.stage_transition = None

    @property
    def parameter_sha256(self):
        return parameter_digest(self.model.state_dict())

    def objective(self, sequences, *, metrics=False):
        return hierarchy_bc_objective(
            self.model,
            sequences,
            self.profile,
            head_weights=self.head_weights,
            control_weights=self.control_weights,
            collect_metrics=metrics,
            observation_cache=self.observation_cache,
            supervision_policy=self.config.supervision_policy,
        )

    def update(self, sequences):
        sequences = tuple(sequences)
        if (
            self.updates >= self.config.total_steps
            or not sequences
            or len(sequences) > self.config.batch_windows
        ):
            raise ValueError("BC optimizer/logical batch budget exhausted")
        if any(
            len(s.records) > self.config.unroll_length + self.config.burn_in_length
            for s in sequences
        ):
            raise ValueError("BC recurrent unroll exceeds frozen activation bound")
        self.model.train()
        # Count exact denominators across the logical batch before allocating
        # one unroll's graph. Mean-of-microbatch losses would distort sparseheads.
        denominators = Counter()
        with torch.no_grad():
            for sequence in sequences:
                denominators.update(self.objective([sequence]).denominators)
        if not denominators.get("act"):
            raise ValueError("no supervised rows; BC update forbidden")
        from poorstar.training.checkpoint import capture_rng, restore_rng

        old = (
            copy.deepcopy(self.model.state_dict()),
            copy.deepcopy(self.optimizer.state_dict()),
            capture_rng(),
        )
        sums = Counter()
        counts = Counter()
        rejections = Counter()
        actual_denominators = Counter()
        weights = self.head_weights | self.control_weights
        try:
            self.optimizer.zero_grad(set_to_none=True)
            for group in self.optimizer.param_groups:
                group["lr"] = self.config.lr_at(self.updates)
            loss_value = 0.0
            for sequence in sequences:
                result = self.objective([sequence])
                if any(
                    result.denominators[h] == 0 and float(v.detach()) != 0.0
                    for h, v in result.head_sums.items()
                ):
                    raise ValueError("unsupervised BC head has a nonzero numerator")
                loss = sum(
                    (
                        v / denominators[h] * weights.get(h, 1.0)
                        for h, v in result.head_sums.items()
                        if denominators[h] > 0
                    ),
                    next(self.model.parameters()).sum() * 0.0,
                )
                loss.backward()
                loss_value += float(loss.detach())
                sums.update({h: float(v.detach()) for h, v in result.head_sums.items()})
                counts.update(result.counts)
                rejections.update(result.rejections)
                actual_denominators.update(result.denominators)
                del loss, result
            if denominators != actual_denominators:
                raise ValueError("hierarchical BC supervision denominator drift")
            norm = torch.nn.utils.clip_grad_norm_(
                [p for p in self.model.parameters() if p.requires_grad],
                self.config.gradient_clip,
                error_if_nonfinite=True,
            )
            self.optimizer.step()
            if any(not torch.isfinite(p).all() for p in self.model.parameters()):
                raise ValueError("nonfinite BC parameters")
        except BaseException:
            self.model.load_state_dict(old[0])
            self.optimizer.load_state_dict(old[1])
            restore_rng(old[2])
            self.optimizer.zero_grad(set_to_none=True)
            raise
        result = BCBatchResult(
            torch.tensor(loss_value, device=self.model.device),
            {h: torch.tensor(v, device=self.model.device) for h, v in sums.items()},
            dict(denominators),
            {},
            dict(rejections),
            dict(counts),
        )
        self.aggregate.add(result)
        self.updates += 1
        self.optimizer_steps += 1
        self.policy_version += 1
        return dict(
            schema="poorstar_v3_hierarchical_bc_update_v1",
            identity_sha256=self.identity.sha256,
            update=self.updates,
            optimizer_steps=self.optimizer_steps,
            loss=loss_value,
            head_loss={h: sums[h] / n for h, n in denominators.items() if n},
            unsupervised_heads=sorted(h for h, n in denominators.items() if not n),
            denominators=dict(denominators),
            counts=dict(counts),
            rejections=dict(rejections),
            gradient_norm=float(norm),
            parameter_sha256=self.parameter_sha256,
            capability_evaluated=False,
            rl_ready=False,
        )

    def state_dict(self):
        return dict(
            model=self.model.state_dict(),
            optimizer=self.optimizer.state_dict(),
            phase=self.phase,
            identity=asdict(self.identity),
            plan=asdict(self.config),
            head_weights=self.head_weights,
            control_weights=self.control_weights,
            trainable_names=self.trainable_names,
            updates=self.updates,
            optimizer_steps=self.optimizer_steps,
            policy_version=self.policy_version,
            environment_loops=0,
            aggregate=self.aggregate.state_dict(),
            stage_transition=self.stage_transition,
        )

    def load_state_dict(self, state):
        from poorstar.training.stages import validate_recovered_counters

        validate_recovered_counters(state, self.config, self.phase)
        if self.stage_transition is not None and self.stage_transition != state.get(
            "stage_transition"
        ):
            raise ValueError("BC stage transition drift")
        for name, expected in (
            ("identity", asdict(self.identity)),
            ("plan", asdict(self.config)),
            ("phase", self.phase),
            ("head_weights", self.head_weights),
            ("control_weights", self.control_weights),
        ):
            if state[name] != expected:
                raise ValueError("BC recovery identity/objective drift: " + name)
        self.stage_transition = state.get("stage_transition")
        if (
            tuple(state["trainable_names"]) != self.trainable_names
            or not 0 <= state["updates"] <= self.config.total_steps
        ):
            raise ValueError("BC recovered parameter groups/budget drift")
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.updates = self.optimizer_steps = self.policy_version = state["updates"]
        self.aggregate.load_state_dict(state["aggregate"])


def load_foundation_initialization(
    path, model, *, expected_sha256, source_report=None, expected_binding=None
):
    """Load foundation initialization."""
    import json
    from pathlib import Path

    from poorstar.training.checkpoint import file_hash

    if file_hash(path) != expected_sha256:
        raise ValueError("C3 initial actor checkpoint hash drift")
    if source_report is None or expected_binding is None:
        raise ValueError(
            "C3 migration requires a frozen checkpoint report and exact source/data/model/profile binding"
        )
    from poorstar.training.contracts import is_sha

    required = {
        "plan_sha256",
        "source_sha256",
        "dataset_sha256",
        "model_config_sha256",
        "profile_sha256",
    }
    if set(expected_binding) != required or any(
        not is_sha(expected_binding[name]) for name in required
    ):
        raise ValueError("C3 migration requires the complete original checkpoint binding")
    report = json.loads(Path(source_report).read_text())
    check = dict(report)
    expected = check.pop("content_digest", None)
    if (
        digest(check) != expected
        or report.get("schema") != "poorstar_c3_bc_checkpoint_v1"
        or report.get("binding") != expected_binding
        or report.get("state_sha256") != expected_sha256
        or expected_binding.get("model_config_sha256") != model.config.sha256
        or report.get("state_bytes") != Path(path).stat().st_size
    ):
        raise ValueError("C3 migration configuration/report drift")
    state = torch.load(path, map_location="cpu", weights_only=False)
    source = state["model"]
    from poorstar.models.policy import RecurrentBCPolicy

    with torch.random.fork_rng(devices=[]):
        expected_tensors = RecurrentBCPolicy(model.config).state_dict()
    expected_keys = set(expected_tensors)
    if set(source) != expected_keys:
        raise ValueError("C3 initializer must exhaust the unchanged lower policy")
    if state.get("steps") != report.get("steps"):
        raise ValueError("C3 migration original checkpoint step drift")
    for name, target in expected_tensors.items():
        value = source[name]
        if (
            not isinstance(value, torch.Tensor)
            or value.shape != target.shape
            or value.dtype != target.dtype
            or not torch.isfinite(value).all()
        ):
            raise ValueError("C3 migration tensor shape/dtype/finite drift: " + name)
    missing, unexpected = model.load_state_dict(source, strict=False)
    new_keys = set(model.state_dict()) - expected_keys
    if unexpected or set(missing) != new_keys:
        raise ValueError("C3 hierarchy migration parameter coverage drift")
    return dict(
        lower_parameter_sha256=parameter_digest(source), new_upper_parameters=sorted(new_keys)
    )
