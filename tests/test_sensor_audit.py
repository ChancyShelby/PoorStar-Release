from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from poorstar.engine.sensor_audit import audit_snapshots
from tests.test_sensors_and_storage import snapshot


def frame(loop=1):
    item = snapshot(loop)
    entity = {
        "slot": 0,
        "values": {"alliance": 1, "display_type": 1, "world_x": 1.2, "world_y": 1.2},
        "present": {"world_x": True, "world_y": True},
        "buff_ids": [],
    }
    visibility = np.zeros((6, 5), dtype=np.uint8)
    visibility[1, 1] = 2
    return dataclasses.replace(
        item,
        entities=(entity,),
        map_size=(5, 6),
        spatial={"visibility": visibility},
        spatial_present={"visibility": True},
    )


def test_entire_stream_counts_both_coordinate_candidates_without_promoting():
    report = audit_snapshots([frame(1), frame(2), frame(3)])
    assert report["frames"] == 3
    transforms = report["world_raster_candidate_correspondence"]
    assert transforms["native_y_equals_floor_world_y"]["pixel_reports_visible"] == 3
    assert transforms["native_y_flips_world_y"]["pixel_not_visible"] == 3
    assert not report["training_ready"] and not report["rl_ready"]
    assert "world_x" not in str(report)


def test_scalar_absence_is_a_count_not_zero_or_a_discarded_frame():
    first, second = frame(1), frame(8)
    presence = dict(first.scalar_present, larva_count=False)
    report = audit_snapshots([dataclasses.replace(first, scalar_present=presence), second])
    assert report["frames"] == 2
    assert report["scalar_presence"]["larva_count"] == {"absent": 1, "present": 1}
    assert report["larva_presence_by_loop_mod8"]["1"] == {"absent": 1}


def test_outside_map_and_overflow_are_not_hidden():
    item = frame()
    item.entities[0]["values"]["world_x"] = 9
    report = audit_snapshots([dataclasses.replace(item, overflow=2)])
    assert report["own_visible_points_outside_map"] == 1 and report["entity_overflow_frames"] == 1


def test_empty_or_reversed_stream_is_refused():
    with pytest.raises(ValueError, match="no real"):
        audit_snapshots([])
    with pytest.raises(ValueError, match="increasing"):
        audit_snapshots([frame(2), frame(1)])


def test_larva_crosscheck_keeps_absence_distinct_even_when_entity_count_is_zero():
    item = frame()
    item.entities[0]["values"]["unit_type"] = 151
    positive = dataclasses.replace(
        item,
        scalars=dict(item.scalars, larva_count=1),
        scalar_present=dict(item.scalar_present, larva_count=True),
    )
    absent = dataclasses.replace(
        frame(2), scalar_present=dict(item.scalar_present, larva_count=False)
    )
    report = audit_snapshots([positive, absent], larva_unit_type=151)
    assert report["larva_count_crosscheck"]["frames"] == {
        "present_matches_own_visible_count": 1,
        "absent_own_visible_count_zero": 1,
    }
    assert report["scalar_presence"]["larva_count"] == {"present": 1, "absent": 1}
