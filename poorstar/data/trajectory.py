"""Data: trajectory."""

from __future__ import annotations

import base64

BYTE_KEY = "$c3_protocol_bytes_base64"


def encode_bytes(value):
    if not isinstance(value, bytes):
        raise TypeError(f"unsupported raw protocol value: {type(value).__name__}")
    return {BYTE_KEY: base64.b64encode(value).decode("ascii")}


def decode_bytes(value):
    if set(value) == {BYTE_KEY}:
        return base64.b64decode(value[BYTE_KEY], validate=True)
    return value
