#!/usr/bin/env python
"""Cli: data: sensors."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import sys
from pathlib import Path

from poorstar.common.paths import source_root

TREE = source_root()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", required=True, type=Path)
    parser.add_argument("--sc2-path", required=True, type=Path)
    parser.add_argument("--sc2-version", required=True)
    parser.add_argument("--player-id", required=True, type=int)
    parser.add_argument(
        "--entity-cap",
        required=True,
        type=int,
        help="declared audit cap; every overflow is counted, never silently accepted",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    if not args.replay.is_file():
        parser.error("a physical SC2Replay file is required")
    if not args.sc2_path.is_dir():
        parser.error("a physical SC2 engine installation is required")
    if args.entity_cap <= 0:
        parser.error("entity cap must be positive")
    if args.output_dir.resolve().is_relative_to(TREE):
        parser.error("large sensor shards belong in a separate data root")
    os.environ["SC2PATH"] = str(args.sc2_path.resolve())
    os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
    from poorstar.common.integrity import file_hash, payload_hash
    from poorstar.data.sensors import SensorShardReader, SensorShardWriter
    from poorstar.engine.replay_adapter import ReplaySensorAdapter

    new_sources = (
        "poorstar/engine/observation.py",
        "poorstar/data/sensors.py",
        "poorstar/common/integrity.py",
        "poorstar/engine/replay_adapter.py",
        "poorstar/cli/data/sensors.py",
    )
    reused = (
        "poorstar/engine/replay_reader.py",
        "poorstar/engine/unit_fields.py",
        "poorstar/geometry/entities.py",
        "requirements-mac-bc.lock.txt",
    )
    sources = {name: file_hash(TREE / name) for name in new_sources + reused}
    replay_sha = file_hash(args.replay)
    provenance = {
        "purpose": "FIELD_AUDIT",
        "split_hash": None,
        "replay_sha256": replay_sha,
        "source_manifest_sha256": payload_hash(sources),
        "source_manifest": sources,
        "extractor_revision": payload_hash({n: sources[n] for n in new_sources}),
        "entity_cap": args.entity_cap,
        "step_mul": 1,
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "packages": {
                name: importlib.metadata.version(name)
                for name in ("numpy", "PySC2", "s2clientprotocol", "protobuf")
            },
        },
    }
    with SensorShardWriter(args.output_dir, provenance=provenance) as writer:
        with ReplaySensorAdapter(
            sc2_path=args.sc2_path,
            version=args.sc2_version,
            player_id=args.player_id,
            entity_cap=args.entity_cap,
        ) as adapter:
            for snapshot in adapter.snapshots(args.replay):
                writer.append(snapshot)
            if file_hash(args.replay) != replay_sha:
                raise ValueError("source replay changed during extraction")
            if any(file_hash(TREE / name) != sha for name, sha in sources.items()):
                raise ValueError("extractor/dependency source changed during extraction")
            manifest = writer.finalize(
                completion={
                    "status": "STREAM_EXHAUSTED",
                    "engine_version": adapter.engine_version,
                    "engine_base_build": adapter.engine_base_build,
                    "terminal_result_frame": "OMITTED_BY_MIGRATED_READER",
                    "world_to_raster": "NOT_ENGINE_VERIFIED",
                }
            )
    manifest_path = args.output_dir / f"{manifest['sha256']}.manifest.json"
    with SensorShardReader(manifest_path) as reader:
        reader.verify()
    print(
        json.dumps(
            {
                "status": "CANDIDATE_SENSORS_PERSISTED_AND_READBACK_VERIFIED",
                "manifest": str(manifest_path),
                "frames": manifest["frame_count"],
                "training_ready": False,
                "rl_ready": False,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
