"""Engine: sensor audit."""

from __future__ import annotations

import math
from collections import Counter
from typing import Iterable

from poorstar.common.integrity import payload_hash
from poorstar.engine.observation import SensorSnapshot


def audit_snapshots(
    snapshots: Iterable[SensorSnapshot], *, larva_unit_type: int | None = None
) -> dict:
    fields: dict[str, Counter] = {}
    larva_present_gaps: Counter = Counter()
    larva_by_mod8: dict[int, Counter] = {index: Counter() for index in range(8)}
    missing_spatial: Counter = Counter()
    transforms = {
        name: Counter() for name in ("native_y_equals_floor_world_y", "native_y_flips_world_y")
    }
    loops = []
    previous_present = None
    overflow = 0
    outside = 0
    larva_comparison: Counter = Counter()
    for frame in snapshots:
        if loops and frame.observation_loop <= loops[-1]:
            raise ValueError("audit requires strictly increasing observation loops")
        loops.append(frame.observation_loop)
        overflow += int(frame.overflow > 0)
        for name, present in frame.scalar_present.items():
            fields.setdefault(name, Counter())["present" if present else "absent"] += 1
        present = frame.scalar_present["larva_count"]
        if larva_unit_type is not None:
            own_larvae = sum(
                entity["values"].get("alliance") == 1
                and entity["values"].get("display_type") == 1
                and entity["values"].get("unit_type") == larva_unit_type
                for entity in frame.entities
            )
            if present:
                larva_comparison[
                    "present_matches_own_visible_count"
                    if frame.scalars["larva_count"] == own_larvae
                    else "present_disagrees_with_own_visible_count"
                ] += 1
            else:
                larva_comparison[
                    "absent_own_visible_count_zero"
                    if own_larvae == 0
                    else "absent_own_visible_count_positive"
                ] += 1
        larva_by_mod8[frame.observation_loop % 8]["present" if present else "absent"] += 1
        if present:
            if previous_present is not None:
                larva_present_gaps[frame.observation_loop - previous_present] += 1
            previous_present = frame.observation_loop
        for name, available in frame.spatial_present.items():
            missing_spatial[name] += int(not available)
        visibility = frame.spatial.get("visibility")
        if visibility is None:
            continue
        height, width = visibility.shape
        for entity in frame.entities:
            values, presence = entity["values"], entity["present"]
            if values.get("alliance") != 1 or values.get("display_type") != 1:
                continue
            if not presence.get("world_x") or not presence.get("world_y"):
                continue
            x, y = math.floor(values["world_x"]), math.floor(values["world_y"])
            if not (0 <= x < width and 0 <= y < height):
                outside += 1
                continue
            for name, row in (
                ("native_y_equals_floor_world_y", y),
                ("native_y_flips_world_y", height - 1 - y),
            ):
                transforms[name]["own_visible_entity_pixels"] += 1
                transforms[name]["pixel_reports_visible"] += int(visibility[row, x] == 2)
                transforms[name]["pixel_not_visible"] += int(visibility[row, x] != 2)
    if not loops:
        raise ValueError("no real stored observations supplied")
    result = {
        "schema": "poorstar_v3_c3_sensor_field_pixel_audit_v1",
        "frames": len(loops),
        "first_loop": loops[0],
        "last_loop": loops[-1],
        "scalar_presence": {name: dict(counts) for name, counts in sorted(fields.items())},
        "larva_presence_by_loop_mod8": {
            str(key): dict(value) for key, value in larva_by_mod8.items()
        },
        "larva_present_gap_histogram": {
            str(key): value for key, value in sorted(larva_present_gaps.items())
        },
        "larva_count_crosscheck": {
            "unit_type": larva_unit_type,
            "frames": dict(larva_comparison),
            "absent_scalar_semantics": "NOT_INFERRED_FROM_ENTITY_CORRELATION",
        },
        "missing_spatial_frames": dict(sorted(missing_spatial.items())),
        "entity_overflow_frames": overflow,
        "own_visible_points_outside_map": outside,
        "world_raster_candidate_correspondence": {
            name: dict(values) for name, values in transforms.items()
        },
        "world_to_raster_verdict": "DIAGNOSTIC_ONLY_LIVE_TARGET_DISPATCH_NOT_VERIFIED",
        "training_ready": False,
        "rl_ready": False,
    }
    result["content_digest"] = payload_hash(result)
    return result
