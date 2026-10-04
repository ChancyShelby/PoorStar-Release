import struct

import pytest

from poorstar.engine.entity_columns import decode_entities, encode_entities


def entity(slot=0, buffs=None):
    return dict(
        slot=slot,
        values={"unit_type": 151, "health": -0.0, "world_x": 1.0000000000000002},
        present={"unit_type": True, "health": False, "world_x": True},
        buff_ids=buffs or [],
    )


def test_all_rows_integer_types_presence_ragged_buffs_and_float_bits_roundtrip():
    rows = [entity(2, [4, 5]), entity(9), entity(17, [6])]
    rows[1]["values"]["health"] = 0  # Absent protobuf floats can carry an integer zero placeholder.
    recovered = decode_entities(encode_entities(rows))
    assert recovered == rows
    for before, after in zip(rows, recovered):
        for name in ("health", "world_x"):
            assert struct.pack("<d", before["values"][name]) == struct.pack(
                "<d", after["values"][name]
            )
        assert type(after["values"]["unit_type"]) is int
    assert decode_entities(encode_entities([])) == []


@pytest.mark.parametrize(
    "change",
    [
        lambda x: x.update(raw_tag=123),
        lambda x: x["values"].update(is_selected=1),
        lambda x: x["values"].update(health=float("nan")),
        lambda x: x["present"].update(health=0),
    ],
)
def test_unsupported_inputs_are_rejected_without_dropping_fields(change):
    row = entity()
    change(row)
    with pytest.raises(ValueError):
        encode_entities([row])


def test_schema_drift_and_duplicate_slots_are_not_truncated():
    with pytest.raises(ValueError, match="duplicate"):
        encode_entities([entity(), entity()])
    second = entity(1)
    second["values"]["new_field"] = 1
    with pytest.raises(ValueError, match="schema"):
        encode_entities([entity(), second])
