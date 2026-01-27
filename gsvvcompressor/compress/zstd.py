"""
Zstandard-based streaming serialization and deserialization.

Uses cloudpickle + length-prefix framing + zstd compression for efficient
streaming of Payload objects. This approach can serialize almost anything
including nested custom classes, torch.Tensor, and numpy.ndarray.

WARNING: cloudpickle uses pickle under the hood. Do NOT use with untrusted
data sources. Only use with trusted data (local files, same-process, etc.).
"""

import struct
from typing import Iterator

import cloudpickle as pickle
import zstandard as zstd

from ..deserializer import AbstractDeserializer
from ..payload import Payload
from ..serializer import AbstractSerializer

# 4-byte big-endian length prefix (max 4GB per object)
_LEN = struct.Struct(">I")


class ZstdSerializer(AbstractSerializer):
    """
    Streaming serializer using cloudpickle + length-prefix framing + zstd.

    Each Payload is pickled, prefixed with its length, then compressed
    incrementally using zstd streaming compression.
    """

    def __init__(self, level: int = 7):
        """
        Initialize the serializer.

        Args:
            level: Zstd compression level (1-22). Default is 7.
        """
        self._compressor = zstd.ZstdCompressor(level=level).compressobj()

    def serialize_frame(self, payload: Payload) -> Iterator[bytes]:
        """
        Serialize and compress a Payload object.

        Args:
            payload: A Payload instance to serialize.

        Yields:
            Compressed byte chunks (may yield zero chunks if buffered).
        """
        # Pickle the payload
        pickled = pickle.dumps(payload, protocol=pickle.DEFAULT_PROTOCOL)
        # Add length prefix framing
        framed = _LEN.pack(len(pickled)) + pickled
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


class ZstdDeserializer(AbstractDeserializer):
    """
    Streaming deserializer using zstd + length-prefix framing + cloudpickle.

    Decompresses incoming bytes incrementally, buffers until a complete
    length-prefixed frame is available, then unpickles and yields Payloads.
    """

    def __init__(self):
        """Initialize the deserializer."""
        self._decompressor = zstd.ZstdDecompressor().decompressobj()
        self._buffer = bytearray()

    def deserialize_frame(self, data: bytes) -> Iterator[Payload]:
        """
        Decompress and deserialize bytes to Payload objects.

        Args:
            data: Compressed bytes to deserialize.

        Yields:
            Complete Payload objects as they become available.
        """
        # Decompress and add to buffer
        decompressed = self._decompressor.decompress(data)
        if decompressed:
            self._buffer.extend(decompressed)

        # Yield complete frames
        yield from self._extract_payloads()

    def flush(self) -> Iterator[Payload]:
        """
        Flush any remaining buffered data.

        Yields:
            Any remaining Payload objects in the buffer.
        """
        # Try to extract any remaining complete payloads
        yield from self._extract_payloads()

        # If there's leftover data, it's incomplete/corrupted
        if self._buffer:
            raise ValueError(
                f"Incomplete data in buffer: {len(self._buffer)} bytes remaining"
            )

    def _extract_payloads(self) -> Iterator[Payload]:
        """
        Extract complete payloads from the buffer.

        Yields:
            Complete Payload objects.
        """
        while True:
            # Need at least length prefix
            if len(self._buffer) < _LEN.size:
                break

            # Read length
            (length,) = _LEN.unpack(self._buffer[: _LEN.size])

            # Check if we have complete frame
            if len(self._buffer) < _LEN.size + length:
                break

            # Extract and unpickle payload
            pickled = bytes(self._buffer[_LEN.size: _LEN.size + length])
            del self._buffer[: _LEN.size + length]

            yield pickle.loads(pickled)
