"""Training: storage."""

import hashlib
import io
import json
import os
import tempfile
import zipfile
from dataclasses import asdict, replace
from pathlib import Path

from poorstar.training.contracts import RolloutSequence, RolloutStep, digest
from poorstar.training.scorer import PolicyStepPayload


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


class RolloutWriter:
    def __init__(
        self,
        root,
        identity,
        catalog,
        *,
        maximum_sources,
        maximum_delay,
        initial_hidden=None,
        starts_at_episode_boundary=True,
    ):
        from poorstar.actions.coarse import CoarseActionProfile

        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        handle = tempfile.NamedTemporaryFile(
            dir=self.root, prefix=".rollout-", suffix=".partial", delete=False
        )
        self.partial = Path(handle.name)
        handle.close()
        self.zip = zipfile.ZipFile(
            self.partial, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=3
        )
        self.identity = identity
        self.maximum_delay = maximum_delay
        self.closed = False
        self.profile = CoarseActionProfile.from_catalog(catalog, maximum_sources=maximum_sources)
        if self.profile.sha256 != identity.profile_sha256:
            self.zip.close()
            raise ValueError("rollout/action profile identity drift")
        self.index = dict(
            schema="poorstar_v3_hierarchical_rollout_shard_v1",
            artifact_kind="RAW_TRAJECTORY",
            identity_sha256=identity.sha256,
            profile_sha256=self.profile.sha256,
            maximum_sources=maximum_sources,
            maximum_delay=maximum_delay,
            records=[],
            members={},
        )
        self.steps = []
        self.initial_hidden = initial_hidden
        self.starts_at_episode_boundary = starts_at_episode_boundary
        self.index["starts_at_episode_boundary"] = starts_at_episode_boundary
        self.index["initial_hidden"] = None
        if initial_hidden is not None:
            self._memory("initial_hidden.pt", initial_hidden)
            self.index["initial_hidden"] = "initial_hidden.pt"
        self._member("catalog.protocol", catalog.SerializeToString())

    def _memory(self, name, value):
        import torch

        if (
            not isinstance(value, torch.Tensor)
            or value.ndim != 1
            or not torch.isfinite(value).all()
        ):
            raise ValueError("finite vector recurrent state required")
        target = io.BytesIO()
        torch.save(value.detach().cpu().clone(), target)
        self._member(name, target.getvalue())

    def _member(self, name, wire):
        value = hashlib.sha256(wire).hexdigest()
        if name in self.index["members"]:
            if self.index["members"][name] != value:
                raise ValueError("rollout member collision")
            return
        info = zipfile.ZipInfo(name, (1980, 1, 1, 0, 0, 0))
        info.compress_type = zipfile.ZIP_DEFLATED
        self.zip.writestr(info, wire, compresslevel=3)
        self.index["members"][name] = value

    def _frame(self, frame):
        key = frame.key.__dict__
        prefix = "frames/" + digest(key)
        self._member(prefix + "/observation.protocol", frame._observation_wire)
        self._member(prefix + "/game_info.protocol", frame._game_info_wire)
        qualified = getattr(frame, "query_qualified", True)
        if qualified:
            self._member(prefix + "/query_request.protocol", frame._query_request_wire)
            self._member(prefix + "/query_response.protocol", frame._query_response_wire)
        return dict(
            prefix=prefix,
            key=key,
            context=asdict(frame._context),
            query_qualified=qualified,
            control_tags=frame._query_control_tags,
        )

    def append(self, step):
        if self.closed:
            raise ValueError("rollout writer closed")
        step.validate()
        if not isinstance(step.payload, PolicyStepPayload):
            raise ValueError("named C3 payload required")
        if self.steps:
            prev = self.steps[-1]
            if (
                prev.terminated
                or prev.truncated
                or step.episode_id != prev.episode_id
                or step.index != prev.index + 1
                or step.game_loop != prev.next_game_loop
            ):
                raise ValueError("rollout writer episode/clock boundary violation")
        p = step.payload
        record = {k: v for k, v in step.__dict__.items() if k != "payload"}
        record["payload"] = dict(
            scheduler=self._frame(p.scheduler_frame),
            body=None if p.body_frame is None else self._frame(p.body_frame),
            trace=p.trace,
            interrupted_before_body=p.interrupted_before_body,
            dispatch_results=p.dispatch_results,
            memory_before=None,
        )
        if p.memory_before is not None:
            name = "memory/" + str(len(self.steps)).zfill(8) + ".pt"
            self._memory(name, p.memory_before)
            record["payload"]["memory_before"] = name
        name = "decisions/" + str(len(self.steps)).zfill(8) + ".json"
        self._member(name, _json(record))
        self.index["records"].append(name)
        self.steps.append(step)

    def close(self, *, status="COMPLETE"):
        if self.closed:
            raise ValueError("rollout already closed")
        if status not in ("COMPLETE", "ABORTED", "ERROR"):
            raise ValueError("explicit rollout completion state required")
        if status == "COMPLETE":
            RolloutSequence(
                tuple(self.steps),
                self.identity.sha256,
                self.initial_hidden,
                self.starts_at_episode_boundary,
            ).validate(
                self.identity, self.steps[0].policy_version if self.steps else 0, 0, learning=False
            )
        self.index["status"] = status
        self.index["terminal"] = (
            None
            if not self.steps
            else dict(terminated=self.steps[-1].terminated, truncated=self.steps[-1].truncated)
        )
        self.zip.writestr("index.json", _json(self.index))
        self.zip.close()
        self.closed = True
        with self.partial.open("rb") as f:
            os.fsync(f.fileno())
        # Verify every member before publication, including ERROR/ABORTED evidence.
        with zipfile.ZipFile(self.partial) as z:
            if (
                set(z.namelist()) != set(self.index["members"]) | {"index.json"}
                or len(z.namelist()) != len(self.index["members"]) + 1
            ):
                raise ValueError("rollout archive member drift")
            for name, expected in self.index["members"].items():
                if hashlib.sha256(z.read(name)).hexdigest() != expected:
                    raise ValueError("rollout member readback corruption")
        from poorstar.training.checkpoint import file_hash

        value = file_hash(self.partial)
        destination = self.root / (value + ".rollout.zip")
        if destination.exists():
            if file_hash(destination) != value:
                raise ValueError("rollout archive collision")
            self.partial.unlink()
        else:
            self.partial.rename(destination)
        return dict(
            path=str(destination),
            sha256=value,
            steps=len(self.steps),
            status=status,
            schema=self.index["schema"],
            identity_sha256=self.identity.sha256,
        )


class RolloutReader:
    def __init__(self, path, *, sha256, identity, maximum_expanded_bytes=4 * 1024**3):
        from s2clientprotocol import sc2api_pb2 as sc

        from poorstar.actions.coarse import CoarseActionProfile
        from poorstar.training.checkpoint import file_hash

        if file_hash(path) != sha256:
            raise ValueError("rollout archive content drift")
        self.zip = zipfile.ZipFile(path)
        try:
            infos = self.zip.infolist()
            if sum(i.file_size for i in infos) > maximum_expanded_bytes or len(
                {i.filename for i in infos}
            ) != len(infos):
                raise ValueError("rollout archive resource/duplicate member violation")
            self.index = json.loads(self.zip.read("index.json"))
            self.identity = identity
            if (
                self.index["schema"] != "poorstar_v3_hierarchical_rollout_shard_v1"
                or self.index["identity_sha256"] != identity.sha256
                or self.index["status"] != "COMPLETE"
            ):
                raise ValueError("rollout is partial, historical schema or foreign identity")
            if set(self.zip.namelist()) != set(self.index["members"]) | {"index.json"}:
                raise ValueError("rollout member manifest mismatch")
            self.catalog = sc.ResponseData()
            self.catalog.ParseFromString(self._read("catalog.protocol"))
            self.profile = CoarseActionProfile.from_catalog(
                self.catalog, maximum_sources=self.index["maximum_sources"]
            )
            if self.profile.sha256 != identity.profile_sha256:
                raise ValueError("rollout profile drift")
        except BaseException:
            self.zip.close()
            raise

    def _read(self, name):
        wire = self.zip.read(name)
        if hashlib.sha256(wire).hexdigest() != self.index["members"][name]:
            raise ValueError("rollout member corruption")
        return wire

    def _frame(self, record):
        from s2clientprotocol import query_pb2
        from s2clientprotocol import sc2api_pb2 as sc

        from poorstar.actions.frame import ActionFrame, BoundQuery, SchedulerFrame
        from poorstar.engine.observation import ExtractionContext

        prefix = record["prefix"]
        raw = sc.ResponseObservation()
        raw.observation.ParseFromString(self._read(prefix + "/observation.protocol"))
        info = sc.ResponseGameInfo()
        info.ParseFromString(self._read(prefix + "/game_info.protocol"))
        context = ExtractionContext(**record["context"])
        if record["query_qualified"]:
            q, r = query_pb2.RequestQuery(), query_pb2.ResponseQuery()
            q.ParseFromString(self._read(prefix + "/query_request.protocol"))
            r.ParseFromString(self._read(prefix + "/query_response.protocol"))
            receipt = BoundQuery.bind(
                raw.observation, q, r, control_tags=tuple(record["control_tags"])
            )
            frame = ActionFrame.from_protocol(
                raw,
                info,
                self.profile.catalog,
                receipt,
                context=context,
                episode=record["key"]["episode"],
            )
        else:
            frame = SchedulerFrame.from_protocol(
                raw, info, self.profile.catalog, context=context, episode=record["key"]["episode"]
            )
        if frame.key.__dict__ != record["key"]:
            raise ValueError("rollout frame reconstruction identity drift")
        return frame

    def sequence(self):
        import torch

        def memory(name):
            if name is None:
                return None
            value = torch.load(io.BytesIO(self._read(name)), map_location="cpu", weights_only=True)
            if (
                not isinstance(value, torch.Tensor)
                or value.ndim != 1
                or not torch.isfinite(value).all()
            ):
                raise ValueError("recurrent memory record corruption")
            return value

        steps = []
        for name in self.index["records"]:
            r = json.loads(self._read(name))
            p = r.pop("payload")
            payload = PolicyStepPayload(
                self._frame(p["scheduler"]),
                None if p["body"] is None else self._frame(p["body"]),
                p["trace"],
                p["interrupted_before_body"],
                tuple(p["dispatch_results"]),
                memory(p["memory_before"]),
            )
            step = RolloutStep(payload=payload, **r)
            if (
                step.observation_sha256 != payload.observation_sha256
                or step.mask_sha256 != payload.mask_sha256
            ):
                raise ValueError("rollout payload/decision binding drift")
            steps.append(step)
        sequence = RolloutSequence(
            tuple(steps),
            self.identity.sha256,
            memory(self.index["initial_hidden"]),
            self.index["starts_at_episode_boundary"],
        )
        sequence.validate(self.identity, steps[0].policy_version if steps else 0, 0, learning=False)
        return sequence

    def close(self):
        self.zip.close()
