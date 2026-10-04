import json

import pytest

from poorstar.data.trajectory import BYTE_KEY, decode_bytes, encode_bytes


def test_binary_protocol_strings_roundtrip_without_utf8_assumption():
    record = {"nested": [{"original": b"\xff\x00text"}, b""], "number": 45}
    assert json.loads(json.dumps(record, default=encode_bytes), object_hook=decode_bytes) == record


def test_other_protocol_dicts_are_preserved():
    value = {BYTE_KEY: "AA==", "another_key": 1}
    assert decode_bytes(value) == value


def test_invalid_byte_encoding_does_not_silently_repair():
    with pytest.raises(ValueError):
        decode_bytes({BYTE_KEY: "!!!!"})


def test_unexpected_types_are_not_stringified():
    with pytest.raises(TypeError):
        encode_bytes(object())
