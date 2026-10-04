"""BC decision records, compressed shard IO and replay-disjoint splits."""

import hashlib
import json
import tempfile
import zipfile
from dataclasses import dataclass, replace
from pathlib import Path

from poorstar.actions.coarse import CoarseActionProfile
from poorstar.actions.contract import CompleteRequestTarget, RawCommandTarget
from poorstar.actions.frame import ActionFrame, BoundQuery, EngineActionCatalog, SchedulerFrame
from poorstar.data.sensors import _digest, _json, _publish
from poorstar.data.trajectory import decode_bytes
from poorstar.engine.observation import ExtractionContext

SCHEMA = "poorstar_c3_bc_decision_shard_v1"


def freeze_split(replays, *, seed, train_fraction=0.8, development_fraction=0.1):
    """Assign whole connected replay/match groups BEFORE windowing or evaluation.

    A union finds duplicate replay hashes across differently named match groups, so
    either shared replay identity OR shared match identity keeps records together.
    """
    if not 0 < train_fraction < 1 or not 0 < development_fraction < 1 - train_fraction:
        raise ValueError("positive train/development/test fractions required")
    parent = {}
    rows = []

    def root(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for row in replays:
        digest = row["replay_sha256"]
        group = row["match_group"]
        if not _digest(digest) or not isinstance(group, str) or not group:
            raise ValueError("replay hash and explicit match/dedup group required")
        a, b = root("replay:" + digest), root("match:" + group)
        parent[max(a, b)] = min(a, b)
        rows.append((digest, group))
    components = {}
    for node in parent:
        components.setdefault(root(node), []).append(node)
    component_ids = {
        r: hashlib.sha256(_json(sorted(nodes))).hexdigest() for r, nodes in components.items()
    }
    assignments = {}
    groups = {}
    for digest, group in sorted(set(rows)):
        component = component_ids[root("replay:" + digest)]
        fraction = (
            int(hashlib.sha256((str(seed) + ":" + component).encode()).hexdigest(), 16) / 2**256
        )
        split = (
            "train"
            if fraction < train_fraction
            else "development"
            if fraction < train_fraction + development_fraction
            else "test"
        )
        assignments[digest] = split
        groups[digest] = component
    value = dict(
        schema="poorstar_c3_prospective_split_v1",
        seed=seed,
        fractions=[train_fraction, development_fraction, 1 - train_fraction - development_fraction],
        assignments=assignments,
        groups=groups,
        historical_exposure="UNKNOWN_ALLOWED",
    )
    value["sha256"] = hashlib.sha256(_json(value)).hexdigest()
    return value


def validate_split(manifest):
    value = dict(manifest)
    digest = value.pop("sha256")
    if hashlib.sha256(_json(value)).hexdigest() != digest:
        raise ValueError("split manifest hash drift")
    assignments, groups = value["assignments"], value["groups"]
    if set(assignments) != set(groups):
        raise ValueError("split group membership drift")
    seen = {}
    for replay, split in assignments.items():
        if not _digest(replay) or split not in ("train", "development", "test"):
            raise ValueError("invalid split assignment")
        if seen.setdefault(groups[replay], split) != split:
            raise ValueError("replay/match leakage")
    return digest


@dataclass(frozen=True)
class BCDecision:
    replay_sha256: str
    scheduler: ActionFrame
    decision: str
    delay: int
    body: ActionFrame | None = None
    targets: tuple = ()
    valid: bool = True
    burn_in: bool = False
    loss: bool = True
    body_rejection: str | None = None
    native_event_wires: tuple[bytes, ...] = ()

    def validate(self, maximum_delay):
        if not _digest(self.replay_sha256) or self.scheduler.terminal:
            raise ValueError("nonterminal scheduler and replay content identity required")
        if any(type(x) is not bool for x in (self.valid, self.burn_in, self.loss)):
            raise ValueError("explicit boolean sequence masks required")
        if (self.burn_in or not self.valid) and self.loss:
            raise ValueError("padding/burn-in cannot contribute loss")
        if (
            self.decision not in ("ACT", "WAIT")
            or type(self.delay) is not int
            or not 0 <= self.delay <= maximum_delay
        ):
            raise ValueError("invalid scheduler target")
        if self.decision == "WAIT":
            if (
                self.delay == 0
                or self.body is not None
                or self.targets
                or self.body_rejection
                or self.native_event_wires
            ):
                raise ValueError("WAIT must advance without future body")
        elif self.body is None or (not self.targets and self.body_rejection is None):
            raise ValueError("ACT must retain complete fresh body and targets")
        else:
            a, b = self.scheduler.key, self.body.key
            if self.body.terminal or getattr(self.body, "query_qualified", True) is False:
                raise ValueError("ACT body must be nonterminal and query-qualified")
            if (a.episode, a.player_id) != (
                b.episode,
                b.player_id,
            ) or b.loop != a.loop + self.delay:
                raise ValueError("scheduler/body episode or clock mismatch")
            if any(t.frame_key != b for t in self.targets):
                raise ValueError("body target/frame drift")
        if self.body_rejection not in (
            None,
            "NATIVE_COMMAND_WITHOUT_UNIT_ECHO",
            "UNQUALIFIED_NATIVE_DATA_FAMILY",
        ):
            raise ValueError("undeclared structural body rejection")
        if self.body_rejection and not self.native_event_wires:
            raise ValueError("native rejection must retain original events")


class BCShardWriter:
    def __init__(
        self,
        output_dir,
        *,
        split_manifest,
        split,
        catalog,
        game_info,
        source_manifest_sha256,
        extractor_sha256,
        maximum_delay=32,
        maximum_sources=63,
        solid_compression=False,
        solid_codec="xz",
    ):
        if solid_codec not in ("xz", "zstd"):
            raise ValueError("declared solid codec required")
        self.root = Path(output_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        if split not in ("train", "development", "test"):
            raise ValueError("named split required")
        if not _digest(source_manifest_sha256) or not _digest(extractor_sha256):
            raise ValueError("source/extractor hashes required")
        self.split_manifest = split_manifest
        self.split = split
        split_hash = validate_split(split_manifest)
        self.catalog = catalog
        self.info = game_info
        self.profile = CoarseActionProfile.from_catalog(catalog, maximum_sources=maximum_sources)
        self.maximum_delay = maximum_delay
        self.solid_compression = solid_compression
        self.solid_codec = solid_codec
        self.index = dict(
            schema=SCHEMA,
            artifact_kind="RAW_TRAJECTORY",
            split=split,
            split_sha256=split_hash,
            source_manifest_sha256=source_manifest_sha256,
            extractor_sha256=extractor_sha256,
            profile_sha256=self.profile.sha256,
            maximum_delay=maximum_delay,
            maximum_sources=maximum_sources,
            decisions=[],
            members={},
        )
        tmp = tempfile.NamedTemporaryFile(
            dir=self.root, prefix=".bc-", suffix=".partial", delete=False
        )
        self.partial = Path(tmp.name)
        tmp.close()
        self.zip = zipfile.ZipFile(self.partial, "w")
        self.closed = False
        self._member("catalog.protocol", catalog.SerializeToString())
        self._member("game_info.protocol", game_info.SerializeToString())
        self.previous = None

    def _member(self, name, wire):
        digest = hashlib.sha256(wire).hexdigest()
        if name in self.index["members"]:
            if self.index["members"][name] != digest:
                raise ValueError("BC member collision")
            return
        info = zipfile.ZipInfo(name, (1980, 1, 1, 0, 0, 0))
        info.compress_type = zipfile.ZIP_STORED if self.solid_compression else zipfile.ZIP_DEFLATED
        info.external_attr = 0o100600 << 16
        self.zip.writestr(info, wire, compresslevel=3)
        self.index["members"][name] = digest

    def _frame(self, frame):
        self.profile.check_frame(frame)
        if (
            frame.key.map_sha256
            != hashlib.sha256(self.info.SerializeToString(deterministic=True)).hexdigest()
        ):
            raise ValueError("shard map/frame mismatch")
        prefix = "frames/" + hashlib.sha256(_json(frame.key.__dict__)).hexdigest()
        self._member(prefix + "/observation.protocol", frame._observation_wire)
        query_qualified = getattr(frame, "query_qualified", True)
        if query_qualified:
            self._member(prefix + "/query_request.protocol", frame._query_request_wire)
            self._member(prefix + "/query_response.protocol", frame._query_response_wire)
        return dict(
            prefix=prefix,
            episode=frame.key.episode,
            context=frame._context.__dict__,
            original_frame_key=frame.key.__dict__,
            control_tags=frame._query_control_tags,
            query_qualified=query_qualified,
        )

    def append(self, row):
        if self.closed:
            raise ValueError("closed shard")
        row.validate(self.maximum_delay)
        if self.split_manifest["assignments"].get(row.replay_sha256) != self.split:
            raise ValueError("row belongs to a different split")
        key = (row.replay_sha256, row.scheduler.key.player_id, row.scheduler.key.episode)
        if self.previous and self.previous[0] == key and row.scheduler.key.loop != self.previous[1]:
            raise ValueError("sequence clock gap or overlapping decisions")
        record = dict(
            replay_sha256=row.replay_sha256,
            scheduler=self._frame(row.scheduler),
            decision=row.decision,
            delay=row.delay,
            valid=row.valid,
            burn_in=row.burn_in,
            loss=row.loss,
            targets=[],
        )
        record["body_rejection"] = row.body_rejection
        record["native"] = []
        for wire in row.native_event_wires:
            name = "native/" + hashlib.sha256(wire).hexdigest() + ".json"
            self._member(name, wire)
            record["native"].append(name)
        if row.body:
            record["body"] = self._frame(row.body)
            for target in row.targets:
                raw = target.raw if isinstance(target, CompleteRequestTarget) else target
                name = "targets/" + hashlib.sha256(raw.original_wire).hexdigest() + ".protocol"
                self._member(name, raw.original_wire)
                entry = dict(wire=name)
                if isinstance(target, CompleteRequestTarget):
                    native_name = (
                        "native/" + hashlib.sha256(target.native_event_wire).hexdigest() + ".json"
                    )
                    self._member(native_name, target.native_event_wire)
                    entry["native"] = native_name
                record["targets"].append(entry)
        self.index["decisions"].append(record)
        self.previous = (key, row.scheduler.key.loop + row.delay + (row.decision == "ACT"))

    def close(self):
        if self.closed:
            raise ValueError("shard already closed")
        if not self.index["decisions"]:
            raise ValueError("empty BC shard")
        self.zip.writestr("index.json", _json(self.index))
        self.zip.close()
        self.closed = True
        with self.partial.open("rb") as f:
            digest = (
                hashlib.file_digest(f, "sha256").hexdigest()
                if hasattr(hashlib, "file_digest")
                else hashlib.sha256(f.read()).hexdigest()
            )
        if self.solid_compression:
            from poorstar.data.bc.storage import solid_repack

            artifact = solid_repack(
                self.partial, sha256=digest, output_dir=self.root, codec=self.solid_codec
            )
            # Only this writer's private, unpublished staging file is removed.
            # Previous archives and every failed attempt remain untouched.
            self.partial.unlink()
            return artifact
        path = self.root / (digest + ".bc.zip")
        _publish(self.partial, path)
        return dict(
            path=str(path),
            sha256=digest,
            decisions=len(self.index["decisions"]),
            bytes=path.stat().st_size,
        )

    def abort(self):
        if not self.closed:
            self.zip.close()
            self.closed = True
        # Keep partial artifact as a failed attempt; never publish it as complete.


class BCShardReader:
    def __init__(self, path, *, sha256, split_manifest, allowed_split):
        self.path = Path(path)
        if hashlib.sha256(self.path.read_bytes()).hexdigest() != sha256:
            raise ValueError("BC archive hash mismatch")
        self.expanded = None
        if self.path.name.endswith((".xz", ".zst")):
            from poorstar.data.bc.storage import expand_solid

            self.expanded = expand_solid(self.path)
        self.zip = zipfile.ZipFile(self.expanded or path)
        self.index = json.loads(self.zip.read("index.json"))
        if (
            self.index["schema"] != SCHEMA
            or self.index["split"] != allowed_split
            or self.index["split_sha256"] != validate_split(split_manifest)
        ):
            self.zip.close()
            raise ValueError("shard schema/split mismatch")
        self.split_manifest = split_manifest
        if set(self.zip.namelist()) != set(self.index["members"]) | {"index.json"} or len(
            self.zip.namelist()
        ) != len(set(self.zip.namelist())):
            raise ValueError("BC archive member manifest mismatch")
        from s2clientprotocol import sc2api_pb2 as sc

        self.catalog = sc.ResponseData()
        self.catalog.ParseFromString(self._read("catalog.protocol"))
        self.info = sc.ResponseGameInfo()
        self.info.ParseFromString(self._read("game_info.protocol"))
        self.profile = CoarseActionProfile.from_catalog(
            self.catalog, maximum_sources=self.index["maximum_sources"]
        )
        if self.profile.sha256 != self.index["profile_sha256"]:
            raise ValueError("BC profile hash mismatch")

    def _read(self, name):
        wire = self.zip.read(name)
        if hashlib.sha256(wire).hexdigest() != self.index["members"][name]:
            raise ValueError("BC member hash drift")
        return wire

    def _frame(self, record):
        from s2clientprotocol import query_pb2
        from s2clientprotocol import sc2api_pb2 as sc

        raw = sc.ResponseObservation()
        raw.observation.ParseFromString(self._read(record["prefix"] + "/observation.protocol"))
        context = ExtractionContext(**record["context"])
        if record["query_qualified"]:
            request = query_pb2.RequestQuery()
            reply = query_pb2.ResponseQuery()
            request.ParseFromString(self._read(record["prefix"] + "/query_request.protocol"))
            reply.ParseFromString(self._read(record["prefix"] + "/query_response.protocol"))
            receipt = BoundQuery.bind(
                raw.observation, request, reply, control_tags=tuple(record["control_tags"])
            )
            frame = ActionFrame.from_protocol(
                raw,
                self.info,
                self.profile.catalog,
                receipt,
                context=context,
                episode=record["episode"],
            )
        else:
            frame = SchedulerFrame.from_protocol(
                raw, self.info, self.profile.catalog, context=context, episode=record["episode"]
            )
        original = record["original_frame_key"]
        for key in (
            "episode",
            "player_id",
            "loop",
            "observation_sha256",
            "query_sha256",
            "map_sha256",
        ):
            if getattr(frame.key, key) != original[key]:
                raise ValueError("BC frame source identity drift")
        return frame

    def __iter__(self):
        for index in range(len(self.index["decisions"])):
            yield self.row_at(index)

    def row_at(self, index):
        from s2clientprotocol import sc2api_pb2 as sc

        if type(index) is not int or not 0 <= index < len(self.index["decisions"]):
            raise IndexError(index)
        for record in (self.index["decisions"][index],):
            if (
                self.split_manifest["assignments"].get(record["replay_sha256"])
                != self.index["split"]
            ):
                raise ValueError("BC replay leaked across split")
            scheduler = self._frame(record["scheduler"])
            body = self._frame(record["body"]) if "body" in record else None
            targets = []
            for entry in record["targets"]:
                action = sc.Action()
                action.ParseFromString(self._read(entry["wire"]))
                target = RawCommandTarget.from_protocol(action, body)
                if "native" in entry:
                    event = json.loads(self._read(entry["native"]), object_hook=decode_bytes)
                    target = CompleteRequestTarget.retain_native(target, event)
                targets.append(target)
            row = BCDecision(
                record["replay_sha256"],
                scheduler,
                record["decision"],
                record["delay"],
                body,
                tuple(targets),
                record["valid"],
                record["burn_in"],
                record["loss"],
                record["body_rejection"],
                tuple(self._read(n) for n in record["native"]),
            )
            row.validate(self.index["maximum_delay"])
            return row

    def close(self):
        self.zip.close()
        if self.expanded is not None:
            self.expanded.close()


def sequence_windows(rows, *, unroll_length, burn_in_length):
    """Bound memory across shard boundaries, with causal overlapping burn-in.

    Windows never cross replay/player/episode identity. The final short window is
    retained instead of dropped; ragged batches need no fabricated padding frame.
    """
    if (
        type(unroll_length) is not int
        or unroll_length < 1
        or type(burn_in_length) is not int
        or burn_in_length < 0
    ):
        raise ValueError("positive unroll and nonnegative burn-in lengths required")
    episode = None
    history = []
    pending = []
    next_loop = None

    def window():
        return tuple(replace(r, burn_in=True, loss=False) for r in history) + tuple(pending)

    for row in rows:
        identity = (row.replay_sha256, row.scheduler.key.player_id, row.scheduler.key.episode)
        if identity != episode:
            if pending:
                yield window()
            episode = identity
            history = []
            pending = []
            next_loop = None
        if next_loop is not None and row.scheduler.key.loop != next_loop:
            raise ValueError("sequence stream clock gap")
        next_loop = row.scheduler.key.loop + row.delay + (row.decision == "ACT")
        pending.append(row)
        if len(pending) == unroll_length:
            yield window()
            history = (history + pending)[-burn_in_length:] if burn_in_length else []
            pending = []
    if pending:
        yield window()
