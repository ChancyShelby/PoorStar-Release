"""Common: integrity."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

FIELDS = ("ability", "source", "target_mode", "target_entity", "target_position", "queue")


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def payload_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def source_integrity(tree: Path, manifest: Path) -> dict[str, Any]:
    document = json.loads(manifest.read_text(encoding="utf-8"))
    problems = []
    entries = document.get("files", [])
    if not entries:
        problems.append("EMPTY_SOURCE_MANIFEST")
    for entry in entries:
        path = (tree / entry["path"]).resolve()
        if not path.is_relative_to(tree.resolve()):
            problems.append(f"SOURCE_OUTSIDE_TREE:{entry['path']}")
        elif not path.is_file():
            problems.append(f"MISSING_SOURCE:{entry['path']}")
        elif file_hash(path) != entry["sha256"] or path.stat().st_size != entry["bytes"]:
            problems.append(f"SOURCE_DRIFT:{entry['path']}")
    return {
        "files_checked": len(entries),
        "manifest_sha256": file_hash(manifest),
        "problems": problems,
    }


def label_census(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Count full labels and the frozen first-command/first-source projection separately."""
    counts: Counter = Counter()
    bundles: Counter = Counter()
    sources: Counter = Counter()
    modes: Counter = Counter()
    abilities: Counter = Counter()
    all_fields: Counter = Counter()
    projected_fields: Counter = Counter()
    failures: Counter = Counter()
    identities: set[tuple[str, int]] = set()
    previous: dict[str, int] = {}
    for row in rows:
        if row["split"] == "test":
            raise ValueError("test supervision is excluded from this development census")
        identity = (str(row["replay_id"]), int(row["decision_loop"]))
        if identity in identities:
            failures["duplicate_decisions"] += 1
        identities.add(identity)
        if identity[0] in previous and identity[1] <= previous[identity[0]]:
            failures["non_monotonic_decisions"] += 1
        previous[identity[0]] = identity[1]
        counts["rows"] += 1
        counts[str(row["act_wait_class"])] += 1
        counts["commands_observed"] += int(row["command_count"])
        counts["unsupported_commands"] += int(row["unsupported_commands"])
        labels = row.get("labels", [])
        if labels:
            bundles[len(labels)] += 1
            counts["labelled_rows"] += 1
            counts["labelled_commands"] += len(labels)
            counts["commands_beyond_first"] += len(labels) - 1
            counts["first_command_sources_beyond_first"] += max(
                0, len(labels[0].get("source_indices", [])) - 1
            )
            for field in FIELDS:
                projected_fields[field] += int(
                    all(
                        label["command_mask"] == 1 and label["field_masks"].get(field, 0) == 1
                        for label in labels
                    )
                )
        for index, label in enumerate(labels):
            if int(label["command_index"]) != index:
                failures["command_order"] += 1
            response = int(label["response_loop"])
            if not identity[1] < response <= identity[1] + 8:
                failures["causal_window"] += 1
            if int(label["observation_loop"]) != identity[1]:
                failures["observation_binding"] += 1
            if int(label["command_loop"]) != response - 1:
                failures["anchor"] += 1
            selection = label.get("source_indices", [])
            sources[len(selection)] += 1
            if len(selection) != len(set(selection)):
                failures["duplicate_sources"] += 1
            modes[str(label["target_mode"])] += 1
            abilities[str(label["ability_id"])] += 1
            for field in FIELDS:
                all_fields[field] += int(
                    label["command_mask"] == 1 and label["field_masks"].get(field, 0) == 1
                )
    return {
        "counts": dict(sorted(counts.items())),
        "bundle_length_histogram": dict(sorted(bundles.items())),
        "source_count_histogram": dict(sorted(sources.items())),
        "target_mode_counts": dict(sorted(modes.items())),
        "ability_counts": dict(sorted(abilities.items())),
        "all_command_field_denominators": dict(sorted(all_fields.items())),
        "c2a2_projection_field_denominators": dict(sorted(projected_fields.items())),
        "failures": dict(sorted(failures.items())),
    }


def replay_inventory(root: Path | None) -> dict[str, Any]:
    """Byte and semantic-member deduplication without interpreting actions or running SC2.

    Semantic keys hash replay.details + game.events + tracker.events. Missing members
    are not silently dropped from the key; their inventory is unverified instead.
    """
    if root is None or not root.is_dir():
        return {"status": "NOT_AVAILABLE", "entries": [], "files": 0}
    import mpyq

    entries = []
    byte_groups: dict[str, list[str]] = {}
    semantic_groups: dict[str, list[str]] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() != ".sc2replay":
            continue
        digest = file_hash(path)
        entry: dict[str, Any] = {
            "relative_path": str(path.relative_to(root)),
            "bytes": path.stat().st_size,
            "sha256": digest,
        }
        byte_groups.setdefault(digest, []).append(entry["relative_path"])
        try:
            with path.open("rb") as stream:
                archive = mpyq.MPQArchive(stream)
                metadata = archive.read_file("replay.gamemetadata.json")
                entry["metadata"] = json.loads(metadata) if metadata else None
                member_hashes = {}
                for member in ("replay.details", "replay.game.events", "replay.tracker.events"):
                    content = archive.read_file(member)
                    if content is None:
                        raise ValueError(f"missing semantic member {member}")
                    member_hashes[member] = hashlib.sha256(content).hexdigest()
            entry["member_sha256"] = member_hashes
            entry["semantic_sha256"] = payload_hash(member_hashes)
            semantic_groups.setdefault(entry["semantic_sha256"], []).append(entry["relative_path"])
            entry["status"] = "ARCHIVE_READABLE_ENGINE_COMPATIBILITY_UNVERIFIED"
        except Exception as error:
            entry["status"] = "ARCHIVE_METADATA_INCOMPLETE"
            entry["error"] = f"{type(error).__name__}: {error}"
        entries.append(entry)
    return {
        "status": "INVENTORIED" if entries else "NOT_AVAILABLE",
        "files": len(entries),
        "entries": entries,
        "byte_duplicate_groups": [v for v in byte_groups.values() if len(v) > 1],
        "semantic_duplicate_groups": [v for v in semantic_groups.values() if len(v) > 1],
        "player_match_disjointness": "NOT_VERIFIED",
        "engine_decodability": "NOT_VERIFIED",
    }


def physical_inventory(root: Path | None, suffix: str) -> dict[str, Any]:
    if root is None or not root.is_dir():
        return {"status": "NOT_AVAILABLE", "files": 0, "entries": []}
    entries = [
        {
            "relative_path": str(p.relative_to(root)),
            "bytes": p.stat().st_size,
            "sha256": file_hash(p),
        }
        for p in sorted(root.rglob(f"*{suffix}"))
        if p.is_file()
    ]
    return {
        "status": "PHYSICAL_FILES_FOUND_NOT_PARITY_VERIFIED" if entries else "NOT_AVAILABLE",
        "files": len(entries),
        "entries": entries,
    }
