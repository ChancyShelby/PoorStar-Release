"""Engine: protocol wire."""

from __future__ import annotations

import gzip
import struct
from pathlib import Path

MAGIC = b"C3_RAW_PROTOCOL_WIRE_V1\n"
MAX_RECORD_BYTES = 16 * 1024 * 1024


class ProtocolWireWriter:
    def __init__(self, path: Path):
        self.stream = gzip.open(path, "xb", compresslevel=1)
        self.stream.write(MAGIC)
        self.count = 0
        self.finished = False

    def append(self, wire: bytes):
        if self.finished or not 0 < len(wire) <= MAX_RECORD_BYTES:
            raise ValueError("invalid wire size or append after completion")
        self.stream.write(struct.pack("<I", len(wire)))
        self.stream.write(wire)
        self.count += 1

    def close(self, *, complete: bool):
        if self.finished:
            raise ValueError("wire writer already closed")
        try:
            if complete:
                self.stream.write(struct.pack("<IQ", 0, self.count))
        finally:
            self.stream.close()
            self.finished = True


def read_protocol_wire(path: Path):
    """Yield exact serialized objects; completion is proved only by exhausting the iterator."""
    with gzip.open(path, "rb") as stream:
        if stream.read(len(MAGIC)) != MAGIC:
            raise ValueError("RAW_TRAJECTORY wire header mismatch")
        count = 0
        while True:
            length = stream.read(4)
            if len(length) != 4:
                raise ValueError("wire stream has no complete footer")
            size = struct.unpack("<I", length)[0]
            if size == 0:
                footer = stream.read(8)
                if len(footer) != 8 or struct.unpack("<Q", footer)[0] != count:
                    raise ValueError("wire footer count mismatch")
                if stream.read(1):
                    raise ValueError("unexpected data after wire footer")
                return
            if size > MAX_RECORD_BYTES:
                raise ValueError("wire record exceeds declared limit")
            wire = stream.read(size)
            if len(wire) != size:
                raise ValueError("truncated wire record")
            count += 1
            yield wire
