"""Data: bc: storage."""

import hashlib
import json
import lzma
import shutil
import tempfile
import zipfile
from pathlib import Path

from poorstar.common.integrity import file_hash
from poorstar.data.sensors import _publish


def expand_solid(path, *, maximum_bytes=4 * 1024**3):
    """Anonymous local temporary file; CRC-exhausted, bounded expanded archive."""
    target = tempfile.TemporaryFile()
    count = 0
    try:
        if str(path).endswith(".zst"):
            import zstandard

            with open(path, "rb") as source:
                # Guard the declared native window in BYTES. The installed
                # backend's max_window_size argument did not match documented
                # KiB units and rejected otherwise valid real shards.
                header = source.read(18)
                if zstandard.get_frame_parameters(header).window_size > 64 * 1024**2:
                    raise ValueError("solid zstd window resource guard exceeded")
                source.seek(0)
                decoder = zstandard.ZstdDecompressor().decompressobj()
                # Bound output bursts even for a very compressible input frame.
                for compressed in iter(lambda: source.read(1024), b""):
                    if decoder.eof:
                        raise ValueError("unexpected trailing solid frame")
                    block = decoder.decompress(compressed)
                    count += len(block)
                    if count > maximum_bytes:
                        raise ValueError("solid shard expansion resource guard exceeded")
                    target.write(block)
                    if decoder.unused_data:
                        raise ValueError("unexpected trailing solid frame")
            if not decoder.eof:
                raise ValueError("truncated solid zstd frame")
        else:
            with lzma.open(path, "rb") as source:
                while True:
                    block = source.read(1024**2)
                    if not block:
                        break
                    count += len(block)
                    if count > maximum_bytes:
                        raise ValueError("solid shard expansion resource guard exceeded")
                    target.write(block)
        target.seek(0)
        return target
    except BaseException:
        target.close()
        raise


def solid_repack(path, *, sha256, output_dir, codec="xz"):
    """Hash every original member, including exact index bytes; never alter inputs."""
    if codec not in ("xz", "zstd"):
        raise ValueError("declared solid codec required")
    path = Path(path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if file_hash(path) != sha256:
        raise ValueError("repack source archive drift")
    expanded = expand_solid(path) if path.name.endswith((".xz", ".zst")) else None
    with zipfile.ZipFile(expanded or path) as source:
        index_wire = source.read("index.json")
        index = json.loads(index_wire)
        expected = set(index["members"]) | {"index.json"}
        if set(source.namelist()) != expected or len(source.namelist()) != len(expected):
            raise ValueError("repack member manifest drift")
        # ZIP_STORED makes similarities between consecutive protocol frames
        # visible to the outer dictionary compressor. Index/schema/wires stay exact.
        with tempfile.TemporaryFile() as stored:
            with zipfile.ZipFile(stored, "w") as target:
                for name in source.namelist():
                    wire = source.read(name)
                    if (
                        name != "index.json"
                        and hashlib.sha256(wire).hexdigest() != index["members"][name]
                    ):
                        raise ValueError("repack member wire drift")
                    info = zipfile.ZipInfo(name, (1980, 1, 1, 0, 0, 0))
                    info.external_attr = 0o100600 << 16
                    target.writestr(info, wire)
            stored.seek(0)
            temporary = tempfile.NamedTemporaryFile(
                dir=output_dir, prefix=".solid-", suffix=".partial", delete=False
            )
            partial = Path(temporary.name)
            temporary.close()
            if codec == "xz":
                with lzma.open(partial, "wb", preset=6, check=lzma.CHECK_CRC64) as target:
                    shutil.copyfileobj(stored, target, 1024**2)
            else:
                import zstandard

                # One native compression context per operation; never oversubscribe
                # the SC2 workers. Content checksum plus member SHA readback required.
                with partial.open("wb") as target:
                    size = stored.seek(0, 2)
                    stored.seek(0)
                    zstandard.ZstdCompressor(level=3, threads=0, write_checksum=True).copy_stream(
                        stored, target, size=size
                    )
    if expanded is not None:
        expanded.close()
    suffix = ".xz" if codec == "xz" else ".zst"
    # Expansion dispatch uses the suffix, so retain it on the private staging file.
    staged = Path(str(partial) + suffix)
    partial.rename(staged)
    partial = staged
    digest = file_hash(partial)
    destination = output_dir / (digest + ".bc.zip" + suffix)
    # A complete decompression and byte-for-byte member verification is mandatory
    # before the repack is published as a usable shard.
    readback = expand_solid(partial)
    try:
        with zipfile.ZipFile(readback) as target:
            if set(target.namelist()) != expected or target.read("index.json") != index_wire:
                raise ValueError("solid index/readback drift")
            for name, member_sha in index["members"].items():
                if hashlib.sha256(target.read(name)).hexdigest() != member_sha:
                    raise ValueError("solid member/readback drift")
    finally:
        readback.close()
    _publish(partial, destination)
    return dict(
        path=str(destination),
        sha256=digest,
        decisions=len(index["decisions"]),
        bytes=destination.stat().st_size,
        original_archive_sha256=sha256,
        original_archive_bytes=path.stat().st_size,
        member_wires_and_index_byte_identical=True,
        storage="LOSSLESS_SOLID_ZIP_" + ("XZ" if codec == "xz" else "ZSTD") + "_V1",
    )
