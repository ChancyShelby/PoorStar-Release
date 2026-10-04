"""Data: sensors."""

from __future__ import annotations

import dataclasses
import hashlib
import io
import json
import os
import tempfile
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

from poorstar.common.integrity import file_hash, payload_hash
from poorstar.engine.observation import SENSOR_VERSION, ExtractionContext, SensorSnapshot

SHARD_SCHEMA = "poorstar_v3_c3_sensor_shard_candidate_v1"


class SensorStorageError(ValueError):
    pass


def _json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _publish(source: Path, destination: Path) -> None:
    """Link atomically without clobbering an existing artifact, then fsync the directory."""
    os.link(source, destination)
    source.unlink()
    if os.name == "posix":
        descriptor = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def _write_record(path: Path, value: Any) -> None:
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".record-", delete=False) as stream:
        scratch = Path(stream.name)
        stream.write(_json(value))
        stream.flush()
        os.fsync(stream.fileno())
    try:
        _publish(scratch, path)
    except BaseException:
        scratch.unlink(missing_ok=True)
        raise


def _digest(value: str) -> bool:
    return (
        isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)
    )


class SensorShardWriter:
    def __init__(self, output_dir: Path, *, provenance: dict[str, Any]) -> None:
        self.root = Path(output_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        # Audit shards never pretend to be a frozen training split.
        required = ("replay_sha256", "source_manifest_sha256", "extractor_revision")
        if any(not _digest(provenance.get(name)) for name in required):
            raise SensorStorageError("replay/source/extractor SHA256 provenance is required")
        if provenance.get("purpose") != "FIELD_AUDIT" or provenance.get("split_hash") is not None:
            raise SensorStorageError(
                "candidate extraction is FIELD_AUDIT only, without a training split"
            )
        self.provenance = json.loads(_json(provenance))
        stream = tempfile.NamedTemporaryFile(
            dir=self.root, prefix=".attempt-", suffix=".partial", delete=False
        )
        self.partial = Path(stream.name)
        stream.close()
        self.attempt = self.partial.name
        self.archive = zipfile.ZipFile(
            self.partial, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=3
        )
        self.members: dict[str, dict[str, Any]] = {}
        self.frames: list[dict[str, Any]] = []
        self.context: dict[str, Any] | None = None
        self.map_size: tuple[int, int] | None = None
        self._started = False
        self._closed = False
        self._published: dict[str, Any] | None = None
        self.missing_scalars: dict[str, int] = {}
        self.missing_spatial: dict[str, int] = {}
        self.overflow_frames = 0

    def __enter__(self) -> "SensorShardWriter":
        return self

    def __exit__(self, kind: Any, error: Any, traceback: Any) -> None:
        if not self._closed:
            self.abort("ERROR" if kind else "ABORTED", str(error) if error else "not finalized")

    def _member(self, name: str, content: bytes) -> None:
        sha = hashlib.sha256(content).hexdigest()
        if name in self.members:
            if self.members[name] != {"sha256": sha, "bytes": len(content)}:
                raise SensorStorageError("member collision")
            return
        info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
        info.compress_type = zipfile.ZIP_DEFLATED
        info.create_system = 3
        info.external_attr = 0o100600 << 16
        self.archive.writestr(info, content, compresslevel=3)
        self.members[name] = {"sha256": sha, "bytes": len(content)}

    def append(self, snapshot: SensorSnapshot) -> None:
        if self._closed:
            raise SensorStorageError("writer is closed")
        snapshot.context.validate()
        if snapshot.schema_version != SENSOR_VERSION:
            raise SensorStorageError("unsupported sensor schema")
        if snapshot.observation_loop < 0 or (
            self.frames and snapshot.observation_loop <= self.frames[-1]["observation_loop"]
        ):
            raise SensorStorageError("frame loops must be strictly increasing")
        context = dataclasses.asdict(snapshot.context)
        if self._started and (context != self.context or snapshot.map_size != self.map_size):
            raise SensorStorageError("player/version/source/map changed inside the episode")
        if set(snapshot.spatial) != {k for k, v in snapshot.spatial_present.items() if v}:
            raise SensorStorageError("spatial presence mask disagrees with arrays")
        header = snapshot.metadata()
        header["spatial_members"] = {}
        for name, array in sorted(snapshot.spatial.items()):
            if array.ndim != 2 or array.dtype.kind not in "iu" or array.dtype.hasobject:
                raise SensorStorageError("spatial arrays must be 2D integer rasters")
            buffer = io.BytesIO()
            np.save(buffer, array, allow_pickle=False)
            content = buffer.getvalue()
            member = f"arrays/{hashlib.sha256(content).hexdigest()}.npy"
            self._member(member, content)
            header["spatial_members"][name] = member
        member = f"frames/{snapshot.observation_loop:012d}.json"
        self._member(member, _json(header))
        self.frames.append({"observation_loop": snapshot.observation_loop, "member": member})
        self.context, self.map_size, self._started = context, snapshot.map_size, True
        for key, present in snapshot.scalar_present.items():
            self.missing_scalars[key] = self.missing_scalars.get(key, 0) + int(not present)
        for key, present in snapshot.spatial_present.items():
            self.missing_spatial[key] = self.missing_spatial.get(key, 0) + int(not present)
        self.overflow_frames += int(snapshot.overflow > 0)

    def finalize(self, *, completion: dict[str, Any]) -> dict[str, Any]:
        if self._closed or not self.frames:
            raise SensorStorageError("cannot finalize closed or empty extraction")
        if completion.get("status") != "STREAM_EXHAUSTED":
            raise SensorStorageError("truncated/limited stream cannot be finalized")
        index = {
            "schema": SHARD_SCHEMA,
            "sensor_schema": SENSOR_VERSION,
            "purpose": "FIELD_AUDIT",
            "training_ready": False,
            "provenance": self.provenance,
            "context": self.context,
            "completion": completion,
            "frames": self.frames,
            "missing_scalar_frames": self.missing_scalars,
            "missing_spatial_frames": self.missing_spatial,
            "entity_overflow_frames": self.overflow_frames,
        }
        self._member("index.json", _json(index))
        self.archive.close()
        with self.partial.open("rb") as stream:
            os.fsync(stream.fileno())
        sha = file_hash(self.partial)
        path = self.root / f"{sha}.sensor.zip"
        manifest = {
            "schema": SHARD_SCHEMA,
            "state": "COMPLETE",
            "training_ready": False,
            "shard": path.name,
            "sha256": sha,
            "bytes": self.partial.stat().st_size,
            "index_sha256": self.members["index.json"]["sha256"],
            "members": self.members,
            "frame_count": len(self.frames),
            "provenance": self.provenance,
        }
        manifest["content_digest"] = payload_hash(manifest)
        _publish(self.partial, path)
        self._published = {"shard": path.name, "sha256": sha}
        _write_record(self.root / f"{sha}.manifest.json", manifest)
        self._closed = True
        return manifest

    def abort(self, state: str = "ABORTED", reason: str = "interrupted") -> None:
        if self._closed:
            return
        if state not in ("ABORTED", "ERROR"):
            raise SensorStorageError("invalid failure state")
        self.archive.close()
        _write_record(
            self.root / f"{self.attempt}.{state.lower()}.json",
            {
                "schema": SHARD_SCHEMA,
                "state": state,
                "attempt": self.attempt,
                "frame_count": len(self.frames),
                "reason": reason,
                "provenance": self.provenance,
                "published_before_failure": self._published,
                "training_ready": False,
            },
        )
        self._closed = True


class SensorShardReader:
    """Verify the complete artifact and then retrieve individual indexed frames."""

    def __init__(self, manifest_path: Path) -> None:
        manifest_path = Path(manifest_path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        stable = {k: v for k, v in manifest.items() if k != "content_digest"}
        if manifest.get("content_digest") != payload_hash(stable):
            raise SensorStorageError("manifest digest mismatch")
        if manifest.get("state") != "COMPLETE" or manifest.get("schema") != SHARD_SCHEMA:
            raise SensorStorageError("not a completed candidate shard")
        sha = manifest["sha256"]
        if not _digest(sha) or manifest["shard"] != f"{sha}.sensor.zip":
            raise SensorStorageError("invalid content address")
        path = manifest_path.parent / manifest["shard"]
        if path.stat().st_size != manifest["bytes"] or file_hash(path) != sha:
            raise SensorStorageError("shard bytes/hash mismatch")
        self.manifest = manifest
        self.archive = zipfile.ZipFile(path, "r")
        try:
            names = self.archive.namelist()
            if len(names) != len(set(names)) or set(names) != set(manifest["members"]):
                raise SensorStorageError("member set mismatch")
            self.index = json.loads(self._read("index.json"))
            if manifest["index_sha256"] != manifest["members"]["index.json"]["sha256"]:
                raise SensorStorageError("index hash mismatch")
            if (
                self.index.get("schema") != SHARD_SCHEMA
                or self.index.get("sensor_schema") != SENSOR_VERSION
                or self.index.get("purpose") != "FIELD_AUDIT"
                or self.index.get("training_ready") is not False
                or manifest.get("training_ready") is not False
                or self.index["completion"].get("status") != "STREAM_EXHAUSTED"
                or self.index["provenance"] != manifest["provenance"]
            ):
                raise SensorStorageError("index contract mismatch")
            self.frame_index = {
                int(f["observation_loop"]): f["member"] for f in self.index["frames"]
            }
            loops = [int(f["observation_loop"]) for f in self.index["frames"]]
            if (
                len(loops) != manifest["frame_count"]
                or not loops
                or loops != sorted(set(loops))
                or loops[0] < 0
                or any(
                    member != f"frames/{loop:012d}.json"
                    for loop, member in self.frame_index.items()
                )
            ):
                raise SensorStorageError("invalid frame index")
        except BaseException:
            self.archive.close()
            raise

    def __enter__(self) -> "SensorShardReader":
        return self

    def __exit__(self, *args: Any) -> None:
        self.archive.close()

    def _read(self, member: str) -> bytes:
        expected = self.manifest["members"].get(member)
        if expected is None or self.archive.getinfo(member).file_size != expected["bytes"]:
            raise SensorStorageError("missing/incorrect-sized member")
        content = self.archive.read(member)
        if hashlib.sha256(content).hexdigest() != expected["sha256"]:
            raise SensorStorageError("member hash mismatch")
        return content

    def frame(self, loop: int) -> SensorSnapshot:
        header = json.loads(self._read(self.frame_index[loop]))
        if (
            header["observation_loop"] != loop
            or header["context"] != self.index["context"]
            or header["schema_version"] != SENSOR_VERSION
        ):
            raise SensorStorageError("frame binding mismatch")
        arrays = {}
        if set(header["spatial_members"]) != {k for k, v in header["spatial_present"].items() if v}:
            raise SensorStorageError("frame presence mismatch")
        for name, member in header["spatial_members"].items():
            content = self._read(member)
            if member != f"arrays/{hashlib.sha256(content).hexdigest()}.npy":
                raise SensorStorageError("array content address mismatch")
            array = np.load(io.BytesIO(content), allow_pickle=False)
            meta = header["spatial"][name]
            if (
                list(array.shape) != meta["shape"]
                or str(array.dtype) != meta["dtype"]
                or hashlib.sha256(array.tobytes()).hexdigest() != meta["sha256"]
            ):
                raise SensorStorageError("array/header mismatch")
            arrays[name] = array
        context = ExtractionContext(**header["context"])
        context.validate()
        return SensorSnapshot(
            loop,
            context,
            header["scalars"],
            header["scalar_present"],
            tuple(header["upgrades"]) if header["upgrades"] is not None else None,
            tuple(header["entities"]),
            int(header["overflow"]),
            arrays,
            header["spatial_present"],
            tuple(header["map_size"]) if header["map_size"] is not None else None,
            raster_frame=header["raster_frame"],
        )

    def verify(self) -> None:
        for member in self.manifest["members"]:
            self._read(member)
        for loop in self.frame_index:
            self.frame(loop)
