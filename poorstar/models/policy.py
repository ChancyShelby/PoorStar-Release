"""Shared recurrent observation encoder and autoregressive action policy."""

import hashlib
import json
import math
from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical, Normal

from poorstar.actions.coarse import CoarsePrefixMachine, coarse_teacher_force
from poorstar.actions.contract import BUNDLE_NEXT, BUNDLE_STOP, SELECT_END, Token

SCALARS = (
    "minerals",
    "vespene",
    "food_cap",
    "food_used",
    "food_army",
    "food_workers",
    "idle_worker_count",
    "army_count",
    "warp_gate_count",
    "own_visible_larva_count",
)
ENTITY_FIELDS = (
    "alliance",
    "display_type",
    "world_x",
    "world_y",
    "health",
    "health_max",
    "shield",
    "shield_max",
    "energy",
    "energy_max",
    "radius",
    "build_progress",
    "facing",
    "cloak",
    "is_blip",
    "is_flying",
    "is_burrowed",
    "is_powered",
    "is_active",
    "cargo_space_taken",
    "cargo_space_max",
    "attack_upgrade_level",
    "armor_upgrade_level",
    "shield_upgrade_level",
    "buff_duration_remain",
    "buff_duration_max",
    "weapon_cooldown",
    "assigned_harvesters",
    "ideal_harvesters",
    "mineral_contents",
    "vespene_contents",
)
PASSENGER_FIELDS = ("health", "health_max", "shield", "shield_max", "energy", "energy_max")
SPATIAL_FIELDS = ("visibility", "creep", "pathing", "placement", "terrain_height")
HEADS = ("ability", "source", "target_mode", "target_entity", "target_position", "queue", "bundle")
FIXED = {
    "target_mode": ("none", "entity", "position"),
    "queue": (False, True),
    "bundle": (BUNDLE_NEXT, BUNDLE_STOP),
}


@dataclass(frozen=True)
class BCModelConfig:
    abilities: tuple[int, ...]
    unit_types: tuple[int, ...]
    upgrades: tuple[int, ...]
    hidden_size: int = 128
    maximum_delay: int = 32
    autoregressive: bool = True
    position_grid: tuple[int, int] = (16, 16)
    normalized_fusion: bool = True
    entity_position_frequencies: int = 6
    deterministic_spatial_pool: bool = True
    schema_version: str = "poorstar_c3_recurrent_coarse_bc_v3"

    def __post_init__(self):
        for values in (self.abilities, self.unit_types, self.upgrades):
            if tuple(sorted(set(values))) != values or any(
                type(x) is not int or x < 0 for x in values
            ):
                raise ValueError("sorted unique catalog vocabulary required")
        if (
            not self.abilities
            or not self.unit_types
            or self.hidden_size < 16
            or self.hidden_size % 4
        ):
            raise ValueError("nonempty vocabulary and valid attention capacity required")
        if self.maximum_delay < 1:
            raise ValueError("positive scheduler horizon required")
        if type(self.normalized_fusion) is not bool:
            raise ValueError("explicit fusion normalization required")
        if type(self.deterministic_spatial_pool) is not bool:
            raise ValueError("explicit spatial pool backend required")
        if (
            type(self.entity_position_frequencies) is not int
            or not 0 <= self.entity_position_frequencies <= 8
        ):
            raise ValueError("declared bounded entity position encoding required")
        if len(self.position_grid) != 2 or any(
            type(n) is not int or n < 1 for n in self.position_grid
        ):
            raise ValueError("declared coarse position grid required")

    @classmethod
    def from_catalog(cls, catalog, profile, **kwargs):
        from poorstar.actions.frame import EngineActionCatalog

        if EngineActionCatalog.from_protocol(catalog) != profile.catalog:
            raise ValueError("model vocabulary and coarse profile catalog differ")
        abilities = tuple(
            sorted(
                a
                for a in profile.catalog.targets
                if a > 0 and profile.ability(a) == a and profile.catalog.targets[a] is not None
            )
        )
        return cls(
            abilities,
            tuple(sorted(int(u.unit_id) for u in catalog.units)),
            tuple(sorted(int(u.upgrade_id) for u in catalog.upgrades)),
            **kwargs,
        )

    @property
    def sha256(self):
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()


class WorldPointDistribution:
    """Independent normals truncated to the current world rectangle, with exact density.

    Sampling uses inverse CDF, not clipping a normal (which would introduce boundary mass).
    Actor protocol conversion rounds once to float32, as does the original SC2 command.
    """

    def __init__(self, parameters, map_size):
        self.map_size = tuple(map_size)
        size = torch.as_tensor(map_size, device=parameters.device, dtype=parameters.dtype)
        self.size = size
        self.normal = Normal(
            parameters[:2].sigmoid() * size,
            (parameters[2:].sigmoid() * 0.45 + 0.05) * size,
            validate_args=False,
        )
        zero = torch.zeros_like(size)
        self.low = self.normal.cdf(zero)
        self.high = self.normal.cdf(size)
        self.mass = self.high - self.low

    def log_prob(self, point):
        x = torch.as_tensor(point, device=self.size.device, dtype=self.size.dtype)
        if len(point) != 2 or any(
            not math.isfinite(float(v)) or not 0 <= v < bound
            for v, bound in zip(point, self.map_size)
        ):
            raise ValueError("point outside declared world rectangle")
        return (self.normal.log_prob(x) - self.mass.log()).sum()

    def choose(self, deterministic=False):
        if deterministic:
            x = self.normal.loc
        else:
            uniform = torch.rand_like(self.size)
            probability = (self.low + uniform * self.mass).clamp(1e-6, 1 - 1e-6)
            x = self.normal.icdf(probability)
        # Protocol float32 rounding can hit the exclusive upper endpoint.
        upper = torch.nextafter(self.size, torch.zeros_like(self.size))
        x = torch.minimum(torch.maximum(x, torch.zeros_like(x)), upper)
        return tuple(float(v) for v in x.detach().cpu())


class CoarseFinePointDistribution:
    """Categorical world regions plus exact continuous within-region offsets.

    Each original float coordinate identifies a region without rounding the target.
    Fine density includes its world-space Jacobian. This retains multimodal locations.
    """

    def __init__(self, parameters, map_size, grid):
        self.map_size = map_size
        self.grid = grid
        n = grid[0] * grid[1]
        self.coarse = Categorical(logits=parameters[:n], validate_args=False)
        self.fine = parameters[n:].reshape(n, 4)
        self.cell_size = (map_size[0] / grid[0], map_size[1] / grid[1])

    def _index(self, point):
        if len(point) != 2 or any(
            not math.isfinite(float(v)) or not 0 <= v < bound
            for v, bound in zip(point, self.map_size)
        ):
            raise ValueError("point outside declared world rectangle")
        x = int(point[0] / self.cell_size[0])
        y = int(point[1] / self.cell_size[1])
        return min(y, self.grid[1] - 1) * self.grid[0] + min(x, self.grid[0] - 1)

    def log_prob(self, point):
        index = self._index(point)
        x = index % self.grid[0]
        y = index // self.grid[0]
        offset = (float(point[0]) - x * self.cell_size[0], float(point[1]) - y * self.cell_size[1])
        dist = WorldPointDistribution(self.fine[index], self.cell_size)
        return self.coarse.log_prob(torch.tensor(index, device=self.fine.device)) + dist.log_prob(
            offset
        )

    def choose(self, deterministic=False):
        index = int(self.coarse.logits.argmax()) if deterministic else int(self.coarse.sample())
        dx, dy = WorldPointDistribution(self.fine[index], self.cell_size).choose(deterministic)
        point = torch.tensor(
            (
                (index % self.grid[0]) * self.cell_size[0] + dx,
                (index // self.grid[0]) * self.cell_size[1] + dy,
            ),
            device=self.fine.device,
            dtype=self.fine.dtype,
        )
        bounds = torch.tensor(self.map_size, device=self.fine.device, dtype=self.fine.dtype)
        point = torch.minimum(point, torch.nextafter(bounds, torch.zeros_like(bounds)))
        return tuple(float(v) for v in point.cpu())


@dataclass
class EncodedObservation:
    hidden: torch.Tensor
    entities: torch.Tensor
    map_size: tuple[int, int]
    parallel_command_base: torch.Tensor | None = None
    parallel_source_state: torch.Tensor | None = None


class RecurrentBCPolicy(nn.Module):
    def __init__(self, config: BCModelConfig):
        super().__init__()
        self.config = config
        h = config.hidden_size
        self.ability_index = {v: i for i, v in enumerate(config.abilities)}
        self.type_index = {v: i for i, v in enumerate(config.unit_types)}
        self.upgrade_index = {v: i for i, v in enumerate(config.upgrades)}
        self.type_embedding = nn.Embedding(len(config.unit_types) + 1, h)
        self.ability_embedding = nn.Embedding(len(config.abilities) + 1, h)
        self.upgrade_embedding = nn.Embedding(len(config.upgrades) + 1, h)
        self.entity_projection = nn.Linear(len(ENTITY_FIELDS) * 2, h)
        self.entity_position_projection = (
            nn.Linear(4 * config.entity_position_frequencies, h, bias=False)
            if config.entity_position_frequencies
            else None
        )
        self.order_projection = nn.Linear(3, h)
        self.passenger_projection = nn.Linear(len(PASSENGER_FIELDS) * 2, h)
        self.order_core = nn.GRUCell(h, h)
        self.passenger_core = nn.GRUCell(h, h)
        self.buff_projection = nn.Linear(2, h)
        self.entity_attention = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(h, 4, h * 2, dropout=0, batch_first=True),
            2,
            enable_nested_tensor=False,
        )
        self.scalar_projection = nn.Sequential(nn.Linear(len(SCALARS) * 2 + 3, h), nn.ReLU())
        from poorstar.training.runtime import DeterministicAdaptiveAvgPool2d

        pool = (
            DeterministicAdaptiveAvgPool2d((8, 8))
            if config.deterministic_spatial_pool
            else nn.AdaptiveAvgPool2d((8, 8))
        )
        self.spatial_encoder = nn.Sequential(
            nn.Conv2d(5, 16, 3, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(2, ceil_mode=True),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU(),
            pool,
            nn.Flatten(),
            nn.Linear(16 * 64 + 5, h),
            nn.ReLU(),
        )
        # Spatial presence is concatenated after convolution, before its projection.
        self.feature_norms = (
            nn.ModuleList([nn.LayerNorm(h) for _ in range(4)]) if config.normalized_fusion else None
        )
        self.fusion = nn.Linear(h * 4, h)
        self.fusion_norm = nn.LayerNorm(h) if config.normalized_fusion else nn.Identity()
        self.memory = nn.GRUCell(h, h)
        self.act_head = nn.Linear(h, 2)
        self.delay_head = nn.Linear(h, self.config.maximum_delay + 1)
        self.delay_embedding = nn.Embedding(self.config.maximum_delay + 1, h)
        self.schedule_core = nn.GRUCell(h, h)
        self.prefix_core = nn.GRUCell(h, h)
        self.source_query = nn.Linear(h, h)
        self.target_query = nn.Linear(h, h)
        self.ability_head = nn.Linear(h, len(config.abilities))
        self.end_embedding = nn.Parameter(torch.randn(h) * 0.02)
        self.end_head = nn.Linear(h, 1)
        self.position_head = nn.Linear(h, 5 * config.position_grid[0] * config.position_grid[1])
        self.position_embedding = nn.Linear(2, h)
        self.fixed_heads = nn.ModuleDict({k: nn.Linear(h, len(v)) for k, v in FIXED.items()})
        self.fixed_embeddings = nn.ModuleDict(
            {k: nn.Embedding(len(v), h) for k, v in FIXED.items()}
        )

    @property
    def device(self):
        return next(self.parameters()).device

    def zero_state(self):
        return torch.zeros(self.config.hidden_size, device=self.device)

    def _tensor(self, value):
        return torch.as_tensor(value, device=self.device, dtype=torch.float32)

    def _ability_embedding(self, ability):
        return self.ability_embedding(
            torch.tensor(
                self.ability_index.get(ability, len(self.ability_index)), device=self.device
            )
        )

    def _type_embedding(self, unit_type):
        if unit_type not in self.type_index:
            raise ValueError("unit type absent from frozen catalog vocabulary")
        return self.type_embedding(torch.tensor(self.type_index[unit_type], device=self.device))

    @staticmethod
    def _number_matrix(rows, names, size):
        """Same present/value transform, with NumPy's native numerical loops.

        Only declared player-visible fields enter this temporary matrix. Float64
        transforms precede the same float32 tensor conversion as the scalar path.
        """
        present = np.asarray([[bool(p.get(n, False)) for n in names] for v, p in rows], dtype=bool)
        values = np.asarray(
            [[float(v.get(n, 0)) if p.get(n, False) else 0.0 for n in names] for v, p in rows],
            dtype=np.float64,
        )
        if not np.isfinite(values).all():
            raise ValueError("nonfinite actor input")
        transformed = np.copysign(np.log1p(np.abs(values)), values)
        for n, bound in zip(("world_x", "world_y"), size):
            if n in names:
                transformed[:, names.index(n)] = values[:, names.index(n)] / bound
        return np.stack((transformed, present), axis=-1).reshape(len(rows), len(names) * 2)

    def encode(self, observation, hidden=None):
        required = {
            "observation_loop",
            "scalars",
            "scalar_present",
            "upgrades",
            "entities",
            "own_operations",
            "spatial_arrays",
            "spatial_present",
            "map_size",
            "raster_frame",
        }
        if set(observation) != required:
            raise ValueError("policy observation schema drift or undeclared features")
        from poorstar.engine.observation import NATIVE_RASTER_FRAME

        if observation["raster_frame"] != NATIVE_RASTER_FRAME:
            raise ValueError("undeclared raster conversion")
        size = tuple(observation["map_size"])
        if len(size) != 2 or min(size) < 1:
            raise ValueError("current map dimensions required")

        def numbers(values, present, names):
            result = []
            for name in names:
                p = bool(present.get(name, False))
                v = float(values.get(name, 0)) if p else 0.0
                if not math.isfinite(v):
                    raise ValueError("nonfinite actor input")
                if name in ("world_x", "world_y"):
                    v /= size[0 if name == "world_x" else 1]
                else:
                    v = math.copysign(math.log1p(abs(v)), v)
                result.extend((v, float(p)))
            return result

        own = {p["parent_slot"]: p for p in observation["own_operations"]}
        entity_values = []
        entity_types = []
        buff_values = []
        orders = []
        passengers = []
        position_values = []
        for i, e in enumerate(observation["entities"]):
            if e["slot"] != i or any(k in e["values"] for k in ("tag", "unit_tag", "is_selected")):
                raise ValueError("noncontiguous/privileged entity features")
            values = e["values"]
            present = e["present"]
            entity_values.append((values, present))
            # Nearby visible units can differ only by about one world unit. A
            # single x/map_size scalar was almost erased by entity attention.
            # Multiscale continuous geometry uses the same current coordinates;
            # tags, slot numbers, targets and future selection never enter it.
            position = []
            for name, bound in zip(("world_x", "world_y"), size):
                exists = bool(present.get(name, False))
                value = float(values.get(name, 0)) if exists else 0.0
                for level in range(self.config.entity_position_frequencies):
                    angle = 2 * math.pi * (2**level) * value / bound
                    position.extend((math.sin(angle) * exists, math.cos(angle) * exists))
            position_values.append(position)
            unit_type = int(values["unit_type"])
            if unit_type not in self.type_index:
                raise ValueError("unknown unit type in actor input")
            entity_types.append(self.type_index[unit_type])
            buffs = e["buff_ids"]
            buff_values.append([math.log1p(len(buffs)), math.log1p(sum(buffs))])
            operations = own.get(i, {})
            order_rows = []
            for order in operations.get("orders", ()):
                features = [
                    float(order["ability_present"]),
                    float(order["progress_present"]),
                    float(order["progress"]) if order["progress_present"] else 0.0,
                ]
                order_rows.append(
                    (
                        self.ability_index.get(int(order["ability_id"]), len(self.ability_index)),
                        features,
                    )
                )
            orders.append(order_rows)
            passenger_rows = []
            for passenger in operations.get("passengers", ()):
                pv, pp = passenger["values"], passenger["present"]
                if pp.get("unit_type") and int(pv["unit_type"]) not in self.type_index:
                    raise ValueError("unknown passenger type")
                tid = (
                    self.type_index[int(pv["unit_type"])]
                    if pp.get("unit_type")
                    else len(self.type_index)
                )
                passenger_rows.append((tid, numbers(pv, pp, PASSENGER_FIELDS)))
            passengers.append(passenger_rows)

        def operation_states(rows, embedding, projection, core, feature_count):
            # Pad only local operation sequences; no entity/source is capped. Each
            # recurrent update is batched over all parents rather than one GPU launch
            # per individual unit. Inactive padding never advances the local state.
            count = len(rows)
            state = torch.zeros((count, self.config.hidden_size), device=self.device)
            for j in range(max((len(r) for r in rows), default=0)):
                entries = [
                    r[j] if j < len(r) else (embedding.num_embeddings - 1, [0.0] * feature_count)
                    for r in rows
                ]
                ids = torch.tensor([e[0] for e in entries], device=self.device)
                inputs = embedding(ids) + projection(self._tensor([e[1] for e in entries]))
                next_state = core(inputs, state)
                mask = torch.tensor([j < len(r) for r in rows], device=self.device).unsqueeze(1)
                state = torch.where(mask, next_state, state)
            return state

        if entity_values:
            base = self.entity_projection(
                self._tensor(self._number_matrix(entity_values, ENTITY_FIELDS, size))
            )
            position = (
                self.entity_position_projection(self._tensor(position_values))
                if self.entity_position_projection is not None
                else torch.zeros_like(base)
            )
            base = base + position
            base = (
                base
                + self.type_embedding(torch.tensor(entity_types, device=self.device))
                + self.buff_projection(self._tensor(buff_values))
            )
            base = base + operation_states(
                orders, self.ability_embedding, self.order_projection, self.order_core, 3
            )
            base = base + operation_states(
                passengers,
                self.type_embedding,
                self.passenger_projection,
                self.passenger_core,
                len(PASSENGER_FIELDS) * 2,
            )
            encoded = self.entity_attention(base.unsqueeze(0)).squeeze(0) + position
            pooled = encoded.mean(0)
        else:
            encoded = torch.empty((0, self.config.hidden_size), device=self.device)
            pooled = self.zero_state()
        scalars = numbers(observation["scalars"], observation["scalar_present"], SCALARS)
        scalars.extend(
            (math.log1p(observation["observation_loop"]), math.log1p(size[0]), math.log1p(size[1]))
        )
        scalar = self.scalar_projection(self._tensor(scalars))
        upgrade = self.zero_state()
        if observation["upgrades"] is not None:
            for uid in observation["upgrades"]:
                if uid not in self.upgrade_index:
                    raise ValueError("unknown upgrade in actor input")
                upgrade = upgrade + self.upgrade_embedding(
                    torch.tensor(self.upgrade_index[uid], device=self.device)
                )
        arrays = observation["spatial_arrays"]
        presence = observation["spatial_present"]
        if set(arrays) - set(SPATIAL_FIELDS):
            raise ValueError("unknown spatial channel")
        spatial = []
        for name in SPATIAL_FIELDS:
            if bool(presence[name]) != (name in arrays):
                raise ValueError("spatial presence/array drift")
            array = (
                np.asarray(arrays[name])
                if name in arrays
                else np.zeros((size[1], size[0]), dtype=np.uint8)
            )
            if array.shape != (size[1], size[0]) or not np.isfinite(array).all():
                raise ValueError("native spatial shape/value drift")
            spatial.append(
                array.astype(np.float32)
                / (255 if name == "terrain_height" else 3 if name == "visibility" else 1)
            )
        pixels = self._tensor(np.stack(spatial)).unsqueeze(0)
        convolution = self.spatial_encoder[:7](pixels).squeeze(0)
        spatial = self.spatial_encoder[7:](
            torch.cat((convolution, self._tensor([presence[n] for n in SPATIAL_FIELDS])))
        )
        parts = (scalar, pooled, spatial, upgrade)
        if self.feature_norms is not None:
            parts = tuple(norm(value) for norm, value in zip(self.feature_norms, parts))
        fused = torch.tanh(self.fusion_norm(self.fusion(torch.cat(parts))))
        hidden = self.memory(fused, self.zero_state() if hidden is None else hidden)
        return EncodedObservation(hidden, encoded, size)

    def scheduler(self, encoded, decision=None):
        act = Categorical(logits=self.act_head(encoded.hidden), validate_args=False)
        logits = self.delay_head(encoded.hidden)
        if decision == "WAIT":
            logits = logits.clone()
            logits[0] = -torch.inf
        return act, Categorical(logits=logits, validate_args=False)

    def consume_schedule(self, encoded, delay):
        return self.schedule_core(
            self.delay_embedding(torch.tensor(delay, device=self.device)), encoded.hidden
        )

    def distribution(self, encoded, prefix, head, options):
        if not self.config.autoregressive:
            base = (
                encoded.hidden
                if encoded.parallel_command_base is None
                else encoded.parallel_command_base
            )
            prefix = (
                (base if encoded.parallel_source_state is None else encoded.parallel_source_state)
                if head == "source"
                else base
            )
        if head == "target_position":
            return CoarseFinePointDistribution(
                self.position_head(prefix), encoded.map_size, self.config.position_grid
            )
        if not options:
            raise ValueError("empty current neural action domain")
        if head == "ability":
            try:
                indices = [self.ability_index[a] for a in options]
            except KeyError as e:
                raise ValueError("ability domain outside frozen vocabulary") from e
            logits = self.ability_head(prefix)[indices]
        elif head in ("source", "target_entity"):
            query = (self.source_query if head == "source" else self.target_query)(prefix)
            # One matrix product, preserving frame-local option order and END. The
            # previous per-candidate dot loop launched hundreds of CUDA kernels
            # for every selected-unit token without changing the distribution.
            scores = encoded.entities.mv(query) / math.sqrt(len(query))
            scores = torch.cat((scores, self.end_head(prefix).reshape(1)))
            indices = [len(encoded.entities) if o == SELECT_END else o for o in options]
            logits = scores[indices]
        else:
            logits = self.fixed_heads[head](prefix)[[FIXED[head].index(o) for o in options]]
        return Categorical(logits=logits, validate_args=False)

    def consume(self, encoded, prefix, token):
        h, v = token.head, token.value
        if h == "ability":
            embedding = self._ability_embedding(v)
        elif h in ("source", "target_entity"):
            embedding = self.end_embedding if v == SELECT_END else encoded.entities[v]
        elif h == "target_position":
            embedding = self.position_embedding(self._tensor(v) / self._tensor(encoded.map_size))
        else:
            embedding = self.fixed_embeddings[h](
                torch.tensor(FIXED[h].index(v), device=self.device)
            )
        history = self.prefix_core(embedding, prefix)
        if not self.config.autoregressive:
            base = (
                encoded.hidden
                if encoded.parallel_command_base is None
                else encoded.parallel_command_base
            )
            if h == "source":
                source = (
                    base if encoded.parallel_source_state is None else encoded.parallel_source_state
                )
                encoded.parallel_source_state = self.prefix_core(embedding, source)
            elif h == "bundle":
                encoded.parallel_command_base = history
                encoded.parallel_source_state = None
        return history

    def teacher_body(self, encoded, frame, targets, profile, *, collect_metrics=True):
        encoded.parallel_command_base = None
        encoded.parallel_source_state = None
        machine, trace, normalized = coarse_teacher_force(frame, targets, profile)
        prefix = encoded.hidden
        losses = {}
        correct = {}
        command_metrics = {}
        for head, options, value in trace:
            dist = self.distribution(encoded, prefix, head, options)
            if head == "target_position":
                nll = -dist.log_prob(value)
                if collect_metrics:
                    chosen = dist.choose(True)
                    metric = math.dist(chosen, value)
                    expected_cell = dist._index(value)
                    correct.setdefault("position_coarse_cell_accuracy", []).append(
                        int(int(dist.coarse.logits.argmax()) == expected_cell)
                    )
                    cx = expected_cell % dist.grid[0]
                    cy = expected_cell // dist.grid[0]
                    fine = WorldPointDistribution(dist.fine[expected_cell], dist.cell_size).choose(
                        True
                    )
                    local = (value[0] - cx * dist.cell_size[0], value[1] - cy * dist.cell_size[1])
                    correct.setdefault("position_fine_teacher_cell_l2", []).append(
                        math.dist(fine, local)
                    )
                    correct.setdefault("position_within_1_world_unit", []).append(
                        int(metric <= 1.0)
                    )
            else:
                index = options.index(value)
                nll = -dist.log_prob(torch.tensor(index, device=self.device))
                if collect_metrics:
                    metric = int(dist.logits.argmax().item() == index)
                    correct.setdefault(head + "_confidence", []).append(
                        float(dist.probs.max().detach())
                    )
                    one_hot = torch.zeros_like(dist.probs)
                    one_hot[index] = 1
                    correct.setdefault(head + "_brier", []).append(
                        float(((dist.probs - one_hot) ** 2).sum().detach())
                    )
                    if head in ("source", "target_entity"):
                        ranking = torch.argsort(dist.logits, descending=True, stable=True)
                        rank = int((ranking == index).nonzero()[0]) + 1
                        correct.setdefault(head + "_mrr", []).append(1.0 / rank)
                        correct.setdefault(head + "_top5", []).append(int(rank <= 5))
                        correct.setdefault(head + "_candidate_count", []).append(len(options))
                        bucket = (
                            "1"
                            if len(options) == 1
                            else "2_8"
                            if len(options) <= 8
                            else "9_32"
                            if len(options) <= 32
                            else "33_128"
                            if len(options) <= 128
                            else "129_plus"
                        )
                        correct.setdefault(head + "_accuracy_candidates_" + bucket, []).append(
                            metric
                        )
            losses.setdefault(head, []).append(nll)
            if collect_metrics:
                correct.setdefault(head, []).append(metric)
                command_metrics.setdefault(head, []).append(
                    metric <= 1.0 if head == "target_position" else bool(metric)
                )
                if head == "bundle":
                    joint = True
                    for number, name in enumerate(HEADS, 1):
                        joint = joint and all(command_metrics.get(name, ()))
                        correct.setdefault(
                            "J" + str(number) + "_teacher_conditioned_command", []
                        ).append(int(joint))
                    command_metrics = {}
            prefix = self.consume(encoded, prefix, Token(head, value))
        return losses, correct, prefix, machine, normalized


class BCPolicySession:
    """Same model for online schedule/body callbacks, with explicit episode reset."""

    def __init__(self, model, profile, *, deterministic=False):
        self.model = model
        self.profile = profile
        self.deterministic = deterministic
        self.episode = None
        self.hidden = None
        self.body = None
        self.prefix = None
        self.last_body_loop = None
        self.action_log_prob = 0.0

    def reset(self, episode):
        self.episode = episode
        self.hidden = None
        self.body = None
        self.prefix = None
        self.last_body_loop = None

    @torch.no_grad()
    def choose_schedule(self, observation, maximum_delay):
        if maximum_delay != self.model.config.maximum_delay:
            raise ValueError("online/training delay horizon mismatch")
        self.body = None
        self.prefix = None
        encoded = self.model.encode(observation, self.hidden)
        act, _ = self.model.scheduler(encoded)
        idx = int(act.logits.argmax()) if self.deterministic else int(act.sample())
        decision = ("ACT", "WAIT")[idx]
        _, delay_dist = self.model.scheduler(encoded, decision)
        delay = int(delay_dist.logits.argmax()) if self.deterministic else int(delay_dist.sample())
        self.hidden = self.model.consume_schedule(encoded, delay)
        self.action_log_prob = float(
            act.log_prob(torch.tensor(idx, device=self.model.device))
            + delay_dist.log_prob(torch.tensor(delay, device=self.model.device))
        )
        return decision, delay

    @torch.no_grad()
    def choose_body(self, observation, head, options, completed):
        if self.body is None:
            self.body = self.model.encode(observation, self.hidden)
            self.prefix = self.body.hidden
            self.last_body_loop = observation["observation_loop"]
        elif observation["observation_loop"] != self.last_body_loop:
            raise ValueError("body observation phase changed")
        dist = self.model.distribution(self.body, self.prefix, head, options)
        if head == "target_position":
            value = dist.choose(self.deterministic)
            logp = dist.log_prob(value)
        else:
            i = int(dist.logits.argmax()) if self.deterministic else int(dist.sample())
            value = options[i]
            logp = dist.log_prob(torch.tensor(i, device=self.model.device))
        self.action_log_prob += float(logp)
        self.prefix = self.model.consume(self.body, self.prefix, Token(head, value))
        self.hidden = self.prefix
        return value

    def run(self, controller, frame, *, context, game_info, catalog):
        from poorstar.actions.coarse import coarse_sample
        from poorstar.actions.timing import run_scheduled_decision

        if self.model.training:
            raise ValueError("live policy must be in evaluation mode")
        if self.episode != frame.key.episode:
            self.reset(frame.key.episode)
        return run_scheduled_decision(
            controller,
            frame,
            choose_schedule=self.choose_schedule,
            choose_body=self.choose_body,
            maximum_delay=self.model.config.maximum_delay,
            context=context,
            episode=frame.key.episode,
            game_info=game_info,
            catalog=catalog,
            body_sampler=lambda f, c: coarse_sample(f, c, self.profile),
        )
