"""
Draco-based streaming serialization and deserialization.

Uses dracoreduced3dgs encoding + cloudpickle for extra data + length-prefix
framing + zstd compression for efficient streaming of DracoPayload objects.

WARNING: cloudpickle uses pickle under the hood. Do NOT use with untrusted
data sources. Only use with trusted data (local files, same-process, etc.).
"""

import struct
from typing import Iterator

import cloudpickle as pickle
import zstandard as zstd

from ..deserializer import AbstractDeserializer
from ..serializer import AbstractSerializer
from .interface import DracoPayload

from . import dracoreduced3dgs

# 4-byte big-endian length prefix (max 4GB per object)
_LEN = struct.Struct(">I")


class DracoSerializer(AbstractSerializer):
    """
    Streaming serializer using Draco encoding + cloudpickle + length-prefix framing + zstd.

    Each DracoPayload is encoded as follows:
    1. The main point cloud data is encoded using dracoreduced3dgs.encode
    2. The extra field (if present) is pickled with cloudpickle
    3. Both are combined with length-prefix framing:
       [draco_len][draco_bytes][extra_len][extra_bytes]
       (extra_len is 0 if no extra data)
    4. The combined data is compressed incrementally using zstd streaming compression.
    """

    def __init__(
        self,
        zstd_level: int = 7,
        draco_level: int = 0,
        qp: int = 30,
        qscale: int = 30,
        qrotation: int = 30,
        qopacity: int = 30,
        qfeaturedc: int = 30,
        qfeaturerest: int = 30,
    ):
        """
        Initialize the serializer.

        Args:
            zstd_level: Zstd compression level (1-22). Default is 7.
            draco_level: Draco compression level (0-10). Default is 0.
            qp: Quantization bits for positions (0 to disable).
            qscale: Quantization bits for scales (0 to disable).
            qrotation: Quantization bits for rotations (0 to disable).
            qopacity: Quantization bits for opacities (0 to disable).
            qfeaturedc: Quantization bits for features_dc (0 to disable).
            qfeaturerest: Quantization bits for features_rest (0 to disable).
        """
        self._compressor = zstd.ZstdCompressor(level=zstd_level).compressobj()
        self._draco_level = draco_level
        self._qp = qp
        self._qscale = qscale
        self._qrotation = qrotation
        self._qopacity = qopacity
        self._qfeaturedc = qfeaturedc
        self._qfeaturerest = qfeaturerest

    def serialize_frame(self, payload: DracoPayload) -> Iterator[bytes]:
        """
        Serialize and compress a DracoPayload object.

        Args:
            payload: A DracoPayload instance to serialize.

        Yields:
            Compressed byte chunks (may yield zero chunks if buffered).
        """
        # Encode point cloud data using Draco
        draco_encoded = dracoreduced3dgs.encode(
            payload.positions,
            payload.scales,
            payload.rotations,
            payload.opacities,
            payload.features_dc,
            payload.features_rest,
            self._draco_level,
            self._qp,
            self._qscale,
            self._qrotation,
            self._qopacity,
            self._qfeaturedc,
            self._qfeaturerest,
        )

        # Pickle the extra field if present
        if payload.extra is not None:
            extra_pickled = pickle.dumps(payload.extra, protocol=pickle.DEFAULT_PROTOCOL)
        else:
            extra_pickled = b""

        # Combine with length-prefix framing:
        # [draco_len][draco_bytes][extra_len][extra_bytes]
        framed = (
            _LEN.pack(len(draco_encoded))
            + draco_encoded
            + _LEN.pack(len(extra_pickled))
            + extra_pickled
        )

        # Compress incrementally
        out = self._compressor.compress(framed)
        if out:
            yield out

    def flush(self) -> Iterator[bytes]:
        """
        Flush remaining compressed data.

        Yields:
            Final compressed byte chunks.
        """
        tail = self._compressor.flush()
        if tail:
            yield tail


class DracoDeserializer(AbstractDeserializer):
    """
    Streaming deserializer using zstd + length-prefix framing + Draco decoding + cloudpickle.

    Decompresses incoming bytes incrementally, buffers until a complete
    length-prefixed frame is available, then decodes and yields DracoPayloads.
    """

    def __init__(self):
        """Initialize the deserializer."""
        self._decompressor = zstd.ZstdDecompressor().decompressobj()
        self._buffer = bytearray()

    def deserialize_frame(self, data: bytes) -> Iterator[DracoPayload]:
        """
        Decompress and deserialize bytes to DracoPayload objects.

        Args:
            data: Compressed bytes to deserialize.

        Yields:
            Complete DracoPayload objects as they become available.
        """
        # Decompress and add to buffer
        decompressed = self._decompressor.decompress(data)
        if decompressed:
            self._buffer.extend(decompressed)

        # Yield complete frames
        yield from self._extract_payloads()

    def flush(self) -> Iterator[DracoPayload]:
        """
        Flush any remaining buffered data.

        Yields:
            Any remaining DracoPayload objects in the buffer.
        """
        # Try to extract any remaining complete payloads
        yield from self._extract_payloads()

        # If there's leftover data, it's incomplete/corrupted
        if self._buffer:
            raise ValueError(
                f"Incomplete data in buffer: {len(self._buffer)} bytes remaining"
            )

    def _extract_payloads(self) -> Iterator[DracoPayload]:
        """
        Extract complete payloads from the buffer.

        Yields:
            Complete DracoPayload objects.
        """
        while True:
            # Need at least draco length prefix
            if len(self._buffer) < _LEN.size:
                break

            # Read draco length
            (draco_len,) = _LEN.unpack(self._buffer[: _LEN.size])

            # Check if we have complete draco data + extra length prefix
            min_frame_size = _LEN.size + draco_len + _LEN.size
            if len(self._buffer) < min_frame_size:
                break

            # Read extra length
            extra_len_offset = _LEN.size + draco_len
            (extra_len,) = _LEN.unpack(
                self._buffer[extra_len_offset: extra_len_offset + _LEN.size]
            )

            # Check if we have complete frame
            total_frame_size = min_frame_size + extra_len
            if len(self._buffer) < total_frame_size:
                break

            # Extract draco bytes
            draco_bytes = bytes(self._buffer[_LEN.size: _LEN.size + draco_len])

            # Extract extra bytes
            extra_offset = extra_len_offset + _LEN.size
            extra_bytes = bytes(self._buffer[extra_offset: extra_offset + extra_len])

            # Remove consumed data from buffer
            del self._buffer[:total_frame_size]

            # Decode draco data
            pc = dracoreduced3dgs.decode(draco_bytes)

            # Unpickle extra if present
            extra = pickle.loads(extra_bytes) if extra_bytes else None

            # Construct and yield DracoPayload
            yield DracoPayload(
                positions=pc.positions,
                scales=pc.scales,
                rotations=pc.rotations,
                opacities=pc.opacities,
                features_dc=pc.features_dc,
                features_rest=pc.features_rest,
                extra=extra,
            )
