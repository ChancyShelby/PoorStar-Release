import gzip
import struct

import pytest

from poorstar.engine.protocol_wire import (
    MAGIC,
    MAX_RECORD_BYTES,
    ProtocolWireWriter,
    read_protocol_wire,
)


def test_exact_bytes_complete_roundtrip_including_non_utf8(tmp_path):
    p = tmp_path / "wire.gz"
    writer = ProtocolWireWriter(p)
    records = [b"\xff\x00abc", b"complete\x00response"]
    for r in records:
        writer.append(r)
    writer.close(complete=True)
    assert list(read_protocol_wire(p)) == records


def test_aborted_stream_is_not_a_complete_game(tmp_path):
    p = tmp_path / "wire.gz"
    writer = ProtocolWireWriter(p)
    writer.append(b"record")
    writer.close(complete=False)
    with pytest.raises(ValueError, match="footer"):
        list(read_protocol_wire(p))


@pytest.mark.parametrize(
    "tail",
    [
        struct.pack("<I", 5) + b"a",
        struct.pack("<IQ", 0, 1),
        struct.pack("<IQ", 0, 0) + b"extra",
        struct.pack("<I", MAX_RECORD_BYTES + 1),
    ],
)
def test_invalid_record_and_footer_are_rejected(tmp_path, tail):
    p = tmp_path / "wire.gz"
    with gzip.open(p, "wb") as f:
        f.write(MAGIC + tail)
    with pytest.raises(ValueError):
        list(read_protocol_wire(p))


def test_truncated_compressed_bytes_are_not_complete(tmp_path):
    p = tmp_path / "wire.gz"
    writer = ProtocolWireWriter(p)
    writer.append(b"record")
    writer.close(complete=True)
    p.write_bytes(p.read_bytes()[:-6])
    with pytest.raises(EOFError):
        list(read_protocol_wire(p))


def test_writer_refuses_empty_records_and_duplicate_close(tmp_path):
    p = tmp_path / "wire.gz"
    writer = ProtocolWireWriter(p)
    with pytest.raises(ValueError):
        writer.append(b"")
    writer.close(complete=True)
    with pytest.raises(ValueError):
        writer.append(b"x")
    with pytest.raises(ValueError):
        writer.close(complete=True)
