"""Engine: entity columns."""

import io
import math

import numpy as np

FORBIDDEN = {"tag", "unit_tag", "raw_tag", "target_unit_tag", "is_selected", "orders"}


def encode_entities(entities):
    if not entities:
        names, presence = [], []
    else:
        sample = entities[0]
        names = sorted(sample["values"])
        presence = sorted(sample["present"])
        if (
            any(type(v) not in (int, float) for v in sample["values"].values())
            or set(names) & FORBIDDEN
        ):
            raise ValueError("unsupported or forbidden entity feature")
    slots, iv, fv, kinds, pv, offsets, buffs = [], [], [], [], [], [0], []
    for entity in entities:
        if set(entity) != {"slot", "values", "present", "buff_ids"}:
            raise ValueError("unknown entity record field")
        values, present = entity["values"], entity["present"]
        if set(values) != set(names) or set(present) != set(presence):
            raise ValueError("heterogeneous entity field schema")
        if any(type(v) not in (int, float) for v in values.values()):
            raise ValueError("unsupported numeric field types")
        if any(type(v) is float and not math.isfinite(v) for v in values.values()):
            raise ValueError("non-finite entity feature")
        if any(type(present[k]) is not bool for k in presence):
            raise ValueError("presence must be boolean")
        if (
            type(entity["slot"]) is not int
            or entity["slot"] < 0
            or any(type(x) is not int or x < 0 for x in entity["buff_ids"])
        ):
            raise ValueError("slot/buff IDs must be nonnegative integers")
        slots.append(entity["slot"])
        iv.append([values[k] if type(values[k]) is int else 0 for k in names])
        fv.append([values[k] if type(values[k]) is float else 0.0 for k in names])
        kinds.append([type(values[k]) is float for k in names])
        pv.append([present[k] for k in presence])
        buffs.extend(entity["buff_ids"])
        offsets.append(len(buffs))
    if len(set(slots)) != len(slots):
        raise ValueError("duplicate entity slots")
    arrays = dict(
        slots=np.asarray(slots, dtype="<i8"),
        value_names=np.asarray(names, dtype="U"),
        presence_names=np.asarray(presence, dtype="U"),
        integers=np.asarray(iv, dtype="<i8").reshape(len(entities), len(names)),
        floats=np.asarray(fv, dtype="<f8").reshape(len(entities), len(names)),
        is_float=np.asarray(kinds, dtype="?").reshape(len(entities), len(names)),
        present=np.asarray(pv, dtype="?").reshape(len(entities), len(presence)),
        buff_offsets=np.asarray(offsets, dtype="<i8"),
        buff_ids=np.asarray(buffs, dtype="<i8"),
    )
    out = io.BytesIO()
    np.savez(out, **arrays)
    return out.getvalue()


def decode_entities(wire):
    with np.load(io.BytesIO(wire), allow_pickle=False) as archive:
        expected = {
            "slots",
            "value_names",
            "presence_names",
            "integers",
            "floats",
            "is_float",
            "present",
            "buff_offsets",
            "buff_ids",
        }
        if set(archive.files) != expected:
            raise ValueError("column member schema mismatch")
        arrays = {k: archive[k] for k in expected}
    slots, iv, fv, pv, offsets, buffs = (
        arrays[k] for k in ("slots", "integers", "floats", "present", "buff_offsets", "buff_ids")
    )
    names, presence = (arrays[k].tolist() for k in ("value_names", "presence_names"))
    kinds = arrays["is_float"]
    if any(
        arrays[k].dtype.kind != "U" or arrays[k].ndim != 1
        for k in ("value_names", "presence_names")
    ):
        raise ValueError("column names must be strings")
    if (
        len(set(names)) != len(names)
        or len(set(presence)) != len(presence)
        or set(names) & FORBIDDEN
    ):
        raise ValueError("duplicate or forbidden column names")
    n = len(slots)
    if (
        slots.ndim != 1
        or slots.dtype != np.dtype("<i8")
        or iv.dtype != np.dtype("<i8")
        or fv.dtype != np.dtype("<f8")
        or pv.dtype != np.dtype("?")
        or offsets.dtype != np.dtype("<i8")
        or buffs.dtype != np.dtype("<i8")
        or iv.shape != (n, len(names))
        or fv.shape != (n, len(names))
        or kinds.shape != (n, len(names))
        or kinds.dtype != np.dtype("?")
        or pv.shape != (n, len(presence))
        or offsets.shape != (n + 1,)
        or buffs.ndim != 1
        or offsets[0] != 0
        or offsets[-1] != len(buffs)
        or np.any(np.diff(offsets) < 0)
        or np.any(slots < 0)
        or np.any(buffs < 0)
        or len(set(slots.tolist())) != n
        or not np.all(np.isfinite(fv))
        or np.any(iv[kinds] != 0)
        or np.any(fv[~kinds] != 0)
    ):
        raise ValueError("column dtype/shape/value contract mismatch")
    return [
        dict(
            slot=int(slots[i]),
            values={
                k: float(fv[i, j]) if kinds[i, j] else int(iv[i, j]) for j, k in enumerate(names)
            },
            present={k: bool(pv[i, j]) for j, k in enumerate(presence)},
            buff_ids=buffs[offsets[i] : offsets[i + 1]].tolist(),
        )
        for i in range(n)
    ]
