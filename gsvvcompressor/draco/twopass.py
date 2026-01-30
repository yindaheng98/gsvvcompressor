"""
Two-pass Draco-based serialization and deserialization.

This module accumulates multiple frames during serialization, then compresses
all data as a single block during flush. This enables better compression by
treating all frames as one continuous point cloud.

WARNING: cloudpickle uses pickle under the hood. Do NOT use with untrusted
data sources. Only use with trusted data (local files, same-process, etc.).
"""

import struct
from dataclasses import dataclass
from typing import Iterator, List, Optional, Self

import cloudpickle as pickle
import numpy as np
import zstandard as zstd

from ..deserializer import AbstractDeserializer
from ..payload import Payload
from ..serializer import AbstractSerializer
from .interface import DracoPayload

from . import dracoreduced3dgs

# 4-byte big-endian length prefix (max 4GB per object)
_LEN = struct.Struct(">I")


@dataclass
class TwoPassDracoPayload(DracoPayload):
    """
    Payload containing data in Draco-compatible format with point count.

    This extends DracoPayload with a num_points field to track the number
    of points in each frame, enabling reconstruction after batched compression.

    Attributes:
        positions: Point positions, shape (N, 3), dtype float32/float64
        scales: Scale indices or values, shape (N, 1), dtype int32 for VQ indices
        rotations: Rotation indices or values, shape (N, 2), dtype int32 for VQ indices
        opacities: Opacity indices or values, shape (N, 1), dtype int32 for VQ indices
        features_dc: DC feature indices or values, shape (N, 1), dtype int32 for VQ indices
        features_rest: Rest feature indices or values, shape (N, 9), dtype int32 for VQ indices
        extra: Optional additional payload for codec-specific data
        num_points: Number of points in this frame
    """

    num_points: int = 0

    def to(self, device) -> Self:
        """
        Move the Payload to the specified device.

        Since TwoPassDracoPayload uses numpy arrays (CPU-only), only the extra
        payload (if present) is moved to the target device.

        Args:
            device: The target device (e.g., 'cpu', 'cuda', torch.device).

        Returns:
            A new TwoPassDracoPayload with the extra payload moved to the target device.
        """
        return TwoPassDracoPayload(
            positions=self.positions,
            scales=self.scales,
            rotations=self.rotations,
            opacities=self.opacities,
            features_dc=self.features_dc,
            features_rest=self.features_rest,
            extra=self.extra.to(device) if self.extra is not None else None,
            num_points=self.num_points,
        )


class TwoPassDracoSerializer(AbstractSerializer):
    """
    Two-pass serializer that accumulates frames and compresses as a single block.

    This serializer collects all frame data during serialize_frame calls,
    concatenating positions, scales, rotations, etc. The actual compression
    happens only during flush(), where all data is encoded as one Draco
    point cloud and compressed with zstd.

    Frame format after flush:
    [num_frames][num_points_per_frame...][draco_len][draco_bytes][extras_len][extras_bytes]
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
        self._zstd_level = zstd_level
        self._draco_level = draco_level
        self._qp = qp
        self._qscale = qscale
        self._qrotation = qrotation
        self._qopacity = qopacity
        self._qfeaturedc = qfeaturedc
        self._qfeaturerest = qfeaturerest

        # Accumulators for frame data
        self._positions_list: List[np.ndarray] = []
        self._scales_list: List[np.ndarray] = []
        self._rotations_list: List[np.ndarray] = []
        self._opacities_list: List[np.ndarray] = []
        self._features_dc_list: List[np.ndarray] = []
        self._features_rest_list: List[np.ndarray] = []
        self._num_points_list: List[int] = []
        self._extras_list: List[Optional[Payload]] = []

    def serialize_frame(self, payload: DracoPayload) -> Iterator[bytes]:
        """
        Accumulate a DracoPayload for later compression.

        This method does not yield any bytes - all compression happens in flush().

        Args:
            payload: A DracoPayload instance to accumulate.

        Yields:
            Nothing (compression is deferred to flush).
        """
        num_points = payload.positions.shape[0]

        self._positions_list.append(payload.positions)
        self._scales_list.append(payload.scales)
        self._rotations_list.append(payload.rotations)
        self._opacities_list.append(payload.opacities)
        self._features_dc_list.append(payload.features_dc)
        self._features_rest_list.append(payload.features_rest)
        self._num_points_list.append(num_points)
        self._extras_list.append(payload.extra)

        # Yield nothing - compression happens in flush
        return
        yield  # Make this a generator

    def flush(self) -> Iterator[bytes]:
        """
        Compress all accumulated frames as a single block.

        Concatenates all frame data, encodes with Draco, pickles extras,
        and compresses everything with zstd.

        Yields:
            Compressed byte chunks containing all frames.
        """
        if not self._positions_list:
            return

        # Concatenate all arrays
        all_positions = np.concatenate(self._positions_list, axis=0)
        all_scales = np.concatenate(self._scales_list, axis=0)
        all_rotations = np.concatenate(self._rotations_list, axis=0)
        all_opacities = np.concatenate(self._opacities_list, axis=0)
        all_features_dc = np.concatenate(self._features_dc_list, axis=0)
        all_features_rest = np.concatenate(self._features_rest_list, axis=0)

        # Encode all point cloud data using Draco
        draco_encoded = dracoreduced3dgs.encode(
            all_positions,
            all_scales,
            all_rotations,
            all_opacities,
            all_features_dc,
            all_features_rest,
            self._draco_level,
            self._qp,
            self._qscale,
            self._qrotation,
            self._qopacity,
            self._qfeaturedc,
            self._qfeaturerest,
        )

        # Pickle extras list
        extras_pickled = pickle.dumps(self._extras_list, protocol=pickle.DEFAULT_PROTOCOL)

        # Build frame structure:
        # [num_frames][num_points_per_frame...][draco_len][draco_bytes][extras_len][extras_bytes]
        num_frames = len(self._num_points_list)

        frame_header = _LEN.pack(num_frames)
        for n in self._num_points_list:
            frame_header += _LEN.pack(n)

        framed = (
            frame_header
            + _LEN.pack(len(draco_encoded))
            + draco_encoded
            + _LEN.pack(len(extras_pickled))
            + extras_pickled
        )

        # Compress with zstd
        compressor = zstd.ZstdCompressor(level=self._zstd_level)
        compressed = compressor.compress(framed)

        yield compressed

        # Clear accumulators
        self._positions_list.clear()
        self._scales_list.clear()
        self._rotations_list.clear()
        self._opacities_list.clear()
        self._features_dc_list.clear()
        self._features_rest_list.clear()
        self._num_points_list.clear()
        self._extras_list.clear()


class TwoPassDracoDeserializer(AbstractDeserializer):
    """
    Deserializer for two-pass Draco-compressed frame blocks.

    This deserializer takes a single compressed block containing multiple frames
    and returns them as a list of TwoPassDracoPayload objects.
    """

    def __init__(self):
        """Initialize the deserializer."""
        self._buffer = bytearray()

    def deserialize_frame(self, data: bytes) -> Iterator[TwoPassDracoPayload]:
        """
        Decompress and deserialize a block of frames.

        Args:
            data: Compressed bytes containing all frames.

        Yields:
            TwoPassDracoPayload objects for each frame in the block.
        """
        # Accumulate data
        self._buffer.extend(data)

        # Try to decompress and parse
        if not self._buffer:
            return

        try:
            # Decompress with zstd
            decompressor = zstd.ZstdDecompressor()
            decompressed = decompressor.decompress(bytes(self._buffer))
        except zstd.ZstdError:
            # Not enough data yet, wait for more
            return

        # Clear buffer after successful decompression
        self._buffer.clear()

        # Parse frame structure
        offset = 0

        # Read num_frames
        if len(decompressed) < offset + _LEN.size:
            raise ValueError("Incomplete data: missing num_frames")
        (num_frames,) = _LEN.unpack(decompressed[offset: offset + _LEN.size])
        offset += _LEN.size

        # Read num_points for each frame
        num_points_list = []
        for _ in range(num_frames):
            if len(decompressed) < offset + _LEN.size:
                raise ValueError("Incomplete data: missing num_points")
            (n,) = _LEN.unpack(decompressed[offset: offset + _LEN.size])
            num_points_list.append(n)
            offset += _LEN.size

        # Read draco data
        if len(decompressed) < offset + _LEN.size:
            raise ValueError("Incomplete data: missing draco_len")
        (draco_len,) = _LEN.unpack(decompressed[offset: offset + _LEN.size])
        offset += _LEN.size

        if len(decompressed) < offset + draco_len:
            raise ValueError("Incomplete data: missing draco_bytes")
        draco_bytes = decompressed[offset: offset + draco_len]
        offset += draco_len

        # Read extras
        if len(decompressed) < offset + _LEN.size:
            raise ValueError("Incomplete data: missing extras_len")
        (extras_len,) = _LEN.unpack(decompressed[offset: offset + _LEN.size])
        offset += _LEN.size

        if len(decompressed) < offset + extras_len:
            raise ValueError("Incomplete data: missing extras_bytes")
        extras_bytes = decompressed[offset: offset + extras_len]
        offset += extras_len

        # Decode draco data
        pc = dracoreduced3dgs.decode(draco_bytes)

        # Unpickle extras list
        extras_list: List[Optional[Payload]] = pickle.loads(extras_bytes)

        # Split arrays by num_points
        all_positions = pc.positions
        all_scales = pc.scales
        all_rotations = pc.rotations
        all_opacities = pc.opacities
        all_features_dc = pc.features_dc
        all_features_rest = pc.features_rest

        start_idx = 0
        for i, num_points in enumerate(num_points_list):
            end_idx = start_idx + num_points

            yield TwoPassDracoPayload(
                positions=all_positions[start_idx:end_idx],
                scales=all_scales[start_idx:end_idx],
                rotations=all_rotations[start_idx:end_idx],
                opacities=all_opacities[start_idx:end_idx],
                features_dc=all_features_dc[start_idx:end_idx],
                features_rest=all_features_rest[start_idx:end_idx],
                extra=extras_list[i] if i < len(extras_list) else None,
                num_points=num_points,
            )

            start_idx = end_idx

    def flush(self) -> Iterator[TwoPassDracoPayload]:
        """
        Flush any remaining buffered data.

        Yields:
            Any remaining TwoPassDracoPayload objects.
        """
        # If there's leftover data, it's incomplete/corrupted
        if self._buffer:
            raise ValueError(
                f"Incomplete data in buffer: {len(self._buffer)} bytes remaining"
            )

        return
        yield  # Make this a generator
