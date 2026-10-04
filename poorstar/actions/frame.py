"""Bind legal action candidates to an exact player observation and query frame."""

import hashlib
import math
from dataclasses import dataclass, field
from types import MappingProxyType

from poorstar.actions.abilities import canonical_ability
from poorstar.engine.observation import ExtractionContext, has, snapshot_from_protocol
from poorstar.engine.own_operations import own_operations_from_protocol
from poorstar.engine.query_masks import current_own_answers
from poorstar.geometry.entities import Entity, in_player_frame, order_entities


def wire_hash(message):
    return hashlib.sha256(message.SerializeToString(deterministic=True)).hexdigest()


def _current_entities(observation):
    """Same slot sort as sensors; do not build all sensor attributes twice for references."""
    entities = []
    seen = set()
    for u in observation.raw_data.units:
        if (
            not all(has(u, k) for k in ("tag", "alliance", "display_type", "unit_type", "pos"))
            or not has(u.pos, "x")
            or not has(u.pos, "y")
        ):
            raise ValueError("current entity reference/type/position presence required")
        tag = int(u.tag)
        if tag <= 0 or tag in seen:
            raise ValueError("invalid duplicated current RAW reference")
        seen.add(tag)
        x, y = float(u.pos.x), float(u.pos.y)
        if not math.isfinite(x) or not math.isfinite(y):
            raise ValueError("nonfinite current entity position")
        e = Entity(tag, int(u.alliance), int(u.unit_type), int(u.display_type), x, y, {})
        if in_player_frame(e):
            entities.append(e)
    return tuple(order_entities(entities))


@dataclass(frozen=True)
class FrameKey:
    episode: str
    player_id: int
    loop: int
    observation_sha256: str
    query_sha256: str
    catalog_sha256: str
    map_sha256: str


@dataclass(frozen=True)
class BoundQuery:
    """Explicit same-observation receipt; no independent-run tag/ordinal join."""

    observation_sha256: str
    loop: int
    request_wire: bytes = field(repr=False)
    response_wire: bytes = field(repr=False)
    control_tags: tuple[int, ...] = field(default=(), repr=False)

    @classmethod
    def bind(cls, observation, request, response, *, control_tags=()):
        current_own_answers(observation, request, response, control_tags=control_tags)
        return cls(
            wire_hash(observation),
            int(observation.game_loop),
            request.SerializeToString(),
            response.SerializeToString(),
            tuple(control_tags),
        )


@dataclass(frozen=True)
class EngineActionCatalog:
    sha256: str
    targets: object
    remaps: object
    larva_type: int | None

    @classmethod
    def from_protocol(cls, catalog):
        remaps, targets = {}, {}
        for a in catalog.abilities:
            if not has(a, "ability_id") or a.ability_id in remaps:
                raise ValueError("unique explicit engine ability identities required")
            if a.ability_id == 0 and (
                not has(a, "link_name") or a.link_name != "Null" or a.remaps_to_ability_id != 0
            ):
                raise ValueError("only the measured catalog Null placeholder may have ID zero")
            remaps[int(a.ability_id)] = int(a.remaps_to_ability_id)
            targets[int(a.ability_id)] = int(a.target) if has(a, "target") else None
        for a in remaps:
            canonical_ability(a, remaps)
        larvae = [int(u.unit_id) for u in catalog.units if u.name == "Larva"]
        if len(larvae) > 1:
            raise ValueError("ambiguous catalog Larva identity")
        return cls(
            wire_hash(catalog),
            MappingProxyType(targets),
            MappingProxyType(remaps),
            larvae[0] if larvae else None,
        )

    def with_unit_catalog(self, catalog):
        """Bind a full same-ability catalog when the query-run requested abilities only."""
        full = type(self).from_protocol(catalog)
        if self.targets != full.targets or self.remaps != full.remaps or full.larva_type is None:
            raise ValueError(
                "full unit catalog must have identical measured abilities and a unique Larva identity"
            )
        digest = hashlib.sha256((self.sha256 + "\n" + full.sha256).encode()).hexdigest()
        return type(self)(digest, self.targets, self.remaps, full.larva_type)


@dataclass(frozen=True)
class ArgumentReferences:
    """Current references WITHOUT availability: codec audit only, never a sampler frame."""

    key: FrameKey
    map_size: tuple[int, int]
    terminal: bool
    slot_tags: tuple[int, ...] = field(repr=False)
    own_visible_slots: frozenset[int]

    @classmethod
    def from_protocol(cls, response, game_info, *, context, episode):
        context.validate()
        obs = response.observation
        if (
            not isinstance(episode, str)
            or not episode
            or not all(has(obs, k) for k in ("game_loop", "raw_data", "player_common"))
            or not has(obs.player_common, "player_id")
            or obs.player_common.player_id != context.player_id
        ):
            raise ValueError("current controlled player/episode/loop/raw observation required")
        entities = _current_entities(obs)
        if not has(game_info, "start_raw") or not has(game_info.start_raw, "map_size"):
            raise ValueError("declared current map required")
        size = game_info.start_raw.map_size
        if not has(size, "x") or not has(size, "y") or min(size.x, size.y) <= 0:
            raise ValueError("invalid map size")
        key = FrameKey(
            episode,
            context.player_id,
            int(obs.game_loop),
            wire_hash(obs),
            "QUERY_NOT_COLLECTED",
            "CATALOG_NOT_BOUND",
            wire_hash(game_info),
        )
        return cls(
            key,
            (int(size.x), int(size.y)),
            bool(response.player_result),
            tuple(e.tag for e in entities),
            frozenset(i for i, e in enumerate(entities) if e.alliance == 1 and e.display_type == 1),
        )


@dataclass(frozen=True)
class ActionFrame:
    key: FrameKey
    map_size: tuple[int, int]
    terminal: bool
    slot_tags: tuple[int, ...] = field(repr=False)
    own_visible_slots: frozenset[int]
    advertised: object = field(repr=False)
    targets: object
    remaps: object = field(repr=False)
    _observation_wire: bytes = field(repr=False)
    _game_info_wire: bytes = field(repr=False)
    _context: ExtractionContext = field(repr=False)
    larva_type: int | None
    # Exact query evidence is retained for training-shard reproduction, never actor features.
    _query_request_wire: bytes = field(default=b"", repr=False)
    _query_response_wire: bytes = field(default=b"", repr=False)
    _query_control_tags: tuple[int, ...] = field(default=(), repr=False)

    @classmethod
    def from_protocol(
        cls,
        response,
        game_info,
        catalog,
        query: BoundQuery,
        *,
        context: ExtractionContext,
        episode: str,
    ):
        from s2clientprotocol import query_pb2

        context.validate()
        if not isinstance(episode, str) or not episode.strip():
            raise ValueError("explicit episode identity required")
        obs = response.observation
        if (
            not all(has(obs, k) for k in ("game_loop", "raw_data", "player_common"))
            or not has(obs.player_common, "player_id")
            or obs.player_common.player_id != context.player_id
        ):
            raise ValueError("exact current controlled player/loop/raw observation required")
        if query.loop != obs.game_loop or query.observation_sha256 != wire_hash(obs):
            raise ValueError("query belongs to a different observation phase")
        request, reply = query_pb2.RequestQuery(), query_pb2.ResponseQuery()
        request.ParseFromString(query.request_wire)
        reply.ParseFromString(query.response_wire)
        if request.ignore_resource_requirements:
            raise ValueError("ignore-resource query is audit-only, not the current sampler domain")
        answers = current_own_answers(obs, request, reply, control_tags=query.control_tags)
        if not has(game_info, "start_raw") or not has(game_info.start_raw, "map_size"):
            raise ValueError("declared world map size required")
        size = game_info.start_raw.map_size
        if not has(size, "x") or not has(size, "y") or min(size.x, size.y) <= 0:
            raise ValueError("invalid world map size")
        entities = _current_entities(obs)
        tags = tuple(e.tag for e in entities)
        if len(tags) != len(set(tags)) or any(t <= 0 for t in tags):
            raise ValueError("invalid or duplicated current entity identity")
        profile = (
            catalog
            if isinstance(catalog, EngineActionCatalog)
            else EngineActionCatalog.from_protocol(catalog)
        )
        remaps, targets = profile.remaps, profile.targets
        advertised = {}
        own = frozenset(
            i for i, e in enumerate(entities) if e.alliance == 1 and e.display_type == 1
        )
        for i in own:
            advertised[i] = frozenset(canonical_ability(a, remaps) for a, _ in answers[tags[i]])
        return cls(
            FrameKey(
                episode,
                context.player_id,
                int(obs.game_loop),
                wire_hash(obs),
                hashlib.sha256(query.request_wire + query.response_wire).hexdigest(),
                profile.sha256,
                wire_hash(game_info),
            ),
            (int(size.x), int(size.y)),
            bool(response.player_result),
            tags,
            own,
            MappingProxyType(advertised),
            MappingProxyType(targets),
            MappingProxyType(remaps),
            obs.SerializeToString(),
            game_info.SerializeToString(),
            context,
            profile.larva_type,
            query.request_wire,
            query.response_wire,
            query.control_tags,
        )

    def source_can_issue(self, slot, ability):
        return (
            slot in self.own_visible_slots
            and ability > 0
            and ability in self.remaps
            and canonical_ability(ability, self.remaps) in self.advertised[slot]
        )

    def ability_options(self):
        available = set().union(*self.advertised.values()) if self.advertised else set()
        return tuple(
            sorted(
                a
                for a in self.targets
                if a > 0
                and self.targets[a] is not None
                and canonical_ability(a, self.remaps) in available
            )
        )

    def actor_candidate_payload(self, *, _for_policy=False):
        """No response actions/results, selection scalar/presence, tags or hidden enemy orders."""
        from s2clientprotocol import sc2api_pb2 as sc

        obs, info = sc.Observation(), sc.ResponseGameInfo()
        obs.ParseFromString(self._observation_wire)
        info.ParseFromString(self._game_info_wire)
        snap = snapshot_from_protocol(
            obs, info, context=self._context, entity_cap=max(1, len(obs.raw_data.units))
        )
        if snap.overflow or len(snap.entities) != len(self.slot_tags):
            raise ValueError("uncapped frame/sensor ordering mismatch")
        payload = snap.metadata()
        # Extraction source/version/player assertions belong to dispatch provenance, not
        # a replay-versus-live indicator exposed to the policy callback.
        payload.pop("context")
        payload["scalars"].pop("larva_count")
        payload["scalar_present"].pop("larva_count")
        # Actual raster arrays remain separate binary members; metadata binds their hashes.
        payload["schema_version"] = "poorstar_c3_uncapped_current_action_frame_candidate_v1"
        operations = own_operations_from_protocol(
            obs, context=self._context, entity_slots={t: i for i, t in enumerate(self.slot_tags)}
        )
        payload["own_operations"] = (
            dict(parents=operations.parents)
            if _for_policy
            else operations.actor_candidate_payload()
        )
        payload["scalars"]["own_visible_larva_count"] = sum(
            p["unit_type"] == self.larva_type for p in operations.parents
        )
        payload["scalar_present"]["own_visible_larva_count"] = self.larva_type is not None
        return payload, snap.spatial

    def policy_observation(self):
        """Actual arrays/attributes only; no provenance/image hashes stand in for features."""
        if self.larva_type is None:
            raise ValueError(
                "a complete verified engine unit catalog is needed for policy observations"
            )
        # The operation object is newly parsed and not retained anywhere: no
        # deepcopy is needed to isolate a fresh policy callback's values.
        payload, rasters = self.actor_candidate_payload(_for_policy=True)
        return dict(
            observation_loop=payload["observation_loop"],
            scalars=payload["scalars"],
            scalar_present=payload["scalar_present"],
            upgrades=payload["upgrades"],
            entities=payload["entities"],
            own_operations=payload["own_operations"]["parents"],
            spatial_arrays=rasters,
            spatial_present=payload["spatial_present"],
            map_size=payload["map_size"],
            raster_frame=payload["raster_frame"],
        )


class SchedulerFrame(ActionFrame):
    """Current actor observation WITHOUT an ability query; cannot be used as an action body.

    Scheduler predicts only ACT/WAIT/delay. Argument queries are collected on the fresh
    body observation after that delay. Existing audited streams queried grid/event frames,
    so demanding a query on every post-ACT scheduler frame would needlessly block BC.
    """

    query_qualified = False

    @classmethod
    def from_protocol(cls, response, game_info, catalog, *, context, episode):
        refs = ArgumentReferences.from_protocol(
            response, game_info, context=context, episode=episode
        )
        profile = (
            catalog
            if isinstance(catalog, EngineActionCatalog)
            else EngineActionCatalog.from_protocol(catalog)
        )
        key = FrameKey(
            refs.key.episode,
            refs.key.player_id,
            refs.key.loop,
            refs.key.observation_sha256,
            "SCHEDULER_QUERY_NOT_REQUIRED",
            profile.sha256,
            refs.key.map_sha256,
        )
        return cls(
            key,
            refs.map_size,
            refs.terminal,
            refs.slot_tags,
            refs.own_visible_slots,
            MappingProxyType({}),
            profile.targets,
            profile.remaps,
            response.observation.SerializeToString(),
            game_info.SerializeToString(),
            context,
            profile.larva_type,
        )
