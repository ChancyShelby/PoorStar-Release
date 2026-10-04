"""Batched encoding for player-visible entities, passengers and spatial inputs."""

import math

import numpy as np
import torch

from poorstar.engine.observation import NATIVE_RASTER_FRAME
from poorstar.models.policy import ENTITY_FIELDS, PASSENGER_FIELDS, SCALARS, SPATIAL_FIELDS


def encode_features_batch(model, observations):
    if not observations:
        raise ValueError("nonempty observation batch required")
    required = {
        "observation_loop",
        "scalars",
        "scalar_present",
        "upgrades",
        "entities",
        "own_operations",
        "spatial_arrays",
        "spatial_present",
        "map_size",
        "raster_frame",
    }
    size = tuple(observations[0]["map_size"])
    h = model.config.hidden_size
    if len(size) != 2 or min(size) < 1:
        raise ValueError("current map dimensions required")
    entity_rows = []
    types = []
    positions = []
    buffs = []
    orders = []
    passengers = []
    lengths = []
    scalar_rows = []
    upgrade_rows = []
    pixel_rows = []
    presence_rows = []
    for obs in observations:
        if set(obs) != required or obs["raster_frame"] != NATIVE_RASTER_FRAME:
            raise ValueError("policy observation schema drift or undeclared raster")
        if tuple(obs["map_size"]) != size:
            raise ValueError("batch requires equal map dimensions")
        own = {p["parent_slot"]: p for p in obs["own_operations"]}
        lengths.append(len(obs["entities"]))
        for i, e in enumerate(obs["entities"]):
            if e["slot"] != i or any(k in e["values"] for k in ("tag", "unit_tag", "is_selected")):
                raise ValueError("noncontiguous/privileged entity features")
            v, p = e["values"], e["present"]
            entity_rows.append((v, p))
            uid = int(v["unit_type"])
            if uid not in model.type_index:
                raise ValueError("unknown unit type in actor input")
            types.append(model.type_index[uid])
            position = []
            for name, bound in zip(("world_x", "world_y"), size):
                exists = bool(p.get(name, False))
                value = float(v.get(name, 0)) if exists else 0.0
                for level in range(model.config.entity_position_frequencies):
                    angle = 2 * math.pi * (2**level) * value / bound
                    position.extend((math.sin(angle) * exists, math.cos(angle) * exists))
            positions.append(position)
            buffs.append([math.log1p(len(e["buff_ids"])), math.log1p(sum(e["buff_ids"]))])
            op = own.get(i, {})
            orders.append(
                [
                    (
                        model.ability_index.get(int(o["ability_id"]), len(model.ability_index)),
                        [
                            float(o["ability_present"]),
                            float(o["progress_present"]),
                            float(o["progress"]) if o["progress_present"] else 0.0,
                        ],
                    )
                    for o in op.get("orders", ())
                ]
            )
            passenger_rows = []
            for a in op.get("passengers", ()):
                pv, pp = a["values"], a["present"]
                if pp.get("unit_type") and int(pv["unit_type"]) not in model.type_index:
                    raise ValueError("unknown passenger type")
                tid = (
                    model.type_index[int(pv["unit_type"])]
                    if pp.get("unit_type")
                    else len(model.type_index)
                )
                vals = []
                for name in PASSENGER_FIELDS:
                    exists = bool(pp.get(name, False))
                    value = float(pv.get(name, 0)) if exists else 0.0
                    if not math.isfinite(value):
                        raise ValueError("nonfinite actor input")
                    vals.extend((math.copysign(math.log1p(abs(value)), value), float(exists)))
                passenger_rows.append((tid, vals))
            passengers.append(passenger_rows)
        values = model._number_matrix([(obs["scalars"], obs["scalar_present"])], SCALARS, size)[
            0
        ].tolist()
        values.extend(
            (math.log1p(obs["observation_loop"]), math.log1p(size[0]), math.log1p(size[1]))
        )
        scalar_rows.append(values)
        upgrade_rows.append(obs["upgrades"])
        arrays = obs["spatial_arrays"]
        presence = obs["spatial_present"]
        if set(arrays) - set(SPATIAL_FIELDS):
            raise ValueError("unknown spatial channel")
        pixels = []
        for name in SPATIAL_FIELDS:
            if bool(presence[name]) != (name in arrays):
                raise ValueError("spatial presence/array drift")
            a = (
                np.asarray(arrays[name])
                if name in arrays
                else np.zeros((size[1], size[0]), dtype=np.uint8)
            )
            if a.shape != (size[1], size[0]) or not np.isfinite(a).all():
                raise ValueError("native spatial shape/value drift")
            pixels.append(
                a.astype(np.float32)
                / (255 if name == "terrain_height" else 3 if name == "visibility" else 1)
            )
        pixel_rows.append(np.stack(pixels))
        presence_rows.append([presence[n] for n in SPATIAL_FIELDS])

    def operation_states(rows, embedding, projection, core, width):
        state = torch.zeros((len(rows), h), device=model.device)
        for j in range(max((len(r) for r in rows), default=0)):
            entries = [
                r[j] if j < len(r) else (embedding.num_embeddings - 1, [0.0] * width) for r in rows
            ]
            inputs = embedding(
                torch.tensor([e[0] for e in entries], device=model.device)
            ) + projection(model._tensor([e[1] for e in entries]))
            mask = torch.tensor([j < len(r) for r in rows], device=model.device).unsqueeze(1)
            state = torch.where(mask, core(inputs, state), state)
        return state

    if entity_rows:
        base = model.entity_projection(
            model._tensor(model._number_matrix(entity_rows, ENTITY_FIELDS, size))
        )
        position = (
            model.entity_position_projection(model._tensor(positions))
            if model.entity_position_projection is not None
            else torch.zeros_like(base)
        )
        base = base + position
        base = (
            base
            + model.type_embedding(torch.tensor(types, device=model.device))
            + model.buff_projection(model._tensor(buffs))
        )
        base = base + operation_states(
            orders, model.ability_embedding, model.order_projection, model.order_core, 3
        )
        base = base + operation_states(
            passengers,
            model.type_embedding,
            model.passenger_projection,
            model.passenger_core,
            len(PASSENGER_FIELDS) * 2,
        )
        # Only nonempty frames enter attention, avoiding an all-masked softmax.
        chunks = list(base.split(lengths))
        pos_chunks = list(position.split(lengths))
        nonempty = [i for i, n in enumerate(lengths) if n]
        padded = torch.nn.utils.rnn.pad_sequence([chunks[i] for i in nonempty], batch_first=True)
        mask = (
            torch.arange(padded.shape[1], device=model.device)[None, :]
            >= torch.tensor([lengths[i] for i in nonempty], device=model.device)[:, None]
        )
        attended = model.entity_attention(padded, src_key_padding_mask=mask)
        entities = [torch.empty((0, h), device=model.device) for _ in observations]
        for j, i in enumerate(nonempty):
            entities[i] = attended[j, : lengths[i]] + pos_chunks[i]
    else:
        entities = [torch.empty((0, h), device=model.device) for _ in observations]
    pooled = torch.stack([e.mean(0) if len(e) else model.zero_state() for e in entities])
    scalar = model.scalar_projection(model._tensor(scalar_rows))
    upgrades = []
    for ids in upgrade_rows:
        u = model.zero_state()
        if ids is not None:
            for uid in ids:
                if uid not in model.upgrade_index:
                    raise ValueError("unknown upgrade in actor input")
                u = u + model.upgrade_embedding(
                    torch.tensor(model.upgrade_index[uid], device=model.device)
                )
        upgrades.append(u)
    pixels = model._tensor(np.stack(pixel_rows))
    convolution = model.spatial_encoder[:7](pixels)
    spatial = model.spatial_encoder[7:](
        torch.cat((convolution, model._tensor(presence_rows)), dim=1)
    )
    parts = (scalar, pooled, spatial, torch.stack(upgrades))
    if model.feature_norms is not None:
        parts = tuple(norm(v) for norm, v in zip(model.feature_norms, parts))
    fused = torch.tanh(model.fusion_norm(model.fusion(torch.cat(parts, dim=1))))
    return [(fused[i], entities[i], size) for i in range(len(observations))]


def advance_features(model, features, hidden):
    from poorstar.models.policy import EncodedObservation

    fused, entities, size = features
    return EncodedObservation(
        model.memory(fused, model.zero_state() if hidden is None else hidden), entities, size
    )
