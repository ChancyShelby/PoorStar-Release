"""Deterministic identities for model parameters."""

from __future__ import annotations

import hashlib
import struct
from typing import Mapping

import torch


def _tensor_bytes(tensor: torch.Tensor) -> bytes:
    """The raw little-endian bytes of a tensor, without requiring a working NumPy bridge."""

    flat = tensor.detach().to("cpu").contiguous()
    header = struct.pack("<B", flat.element_size()) + struct.pack("<q", flat.numel())
    try:
        return header + flat.numpy().tobytes()
    except (RuntimeError, TypeError):
        as_bytes = flat.view(torch.uint8).reshape(-1)
        return header + bytes(as_bytes.tolist())


def parameter_digest(parameters: Mapping[str, torch.Tensor]) -> str:
    """A digest over every parameter, name and bytes, in sorted name order."""

    digest = hashlib.sha256()
    for name in sorted(parameters):
        digest.update(name.encode("utf-8"))
        digest.update(b"\x00")
        digest.update(_tensor_bytes(parameters[name]))
        digest.update(b"\x1e")
    return digest.hexdigest()


def parameter_digests(parameters: Mapping[str, torch.Tensor]) -> dict[str, str]:
    """The per-parameter digest, so a change can be attributed to a named parameter."""

    return {
        name: hashlib.sha256(_tensor_bytes(tensor)).hexdigest()
        for name, tensor in sorted(parameters.items())
    }
