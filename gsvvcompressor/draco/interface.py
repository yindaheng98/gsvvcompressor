"""
Draco-capable interframe codec interface.

This module provides a base interface for interframe codecs that can output
Draco-compatible payloads for efficient compression.
"""

from abc import abstractmethod
from dataclasses import dataclass
from typing import Optional, Self

import numpy as np

from ..payload import Payload
from ..interframe.interface import (
    InterframeCodecInterface,
    InterframeCodecContext,
)


@dataclass
class DracoPayload(Payload):
    """
    Payload containing data in Draco-compatible format.

    This payload structure matches the format expected by Draco compression
    for reduced 3DGS data. All arrays should be numpy arrays.

    Attributes:
        positions: Point positions, shape (N, 3), dtype float32/float64
        scales: Scale indices or values, shape (N, 1), dtype int32 for VQ indices
        rotations: Rotation indices or values, shape (N, 2), dtype int32 for VQ indices
            (rotation_re, rotation_im)
        opacities: Opacity indices or values, shape (N, 1), dtype int32 for VQ indices
        features_dc: DC feature indices or values, shape (N, 1), dtype int32 for VQ indices
        features_rest: Rest feature indices or values, shape (N, 9),
            dtype int32 for VQ indices (3 sh_degrees * 3)
        extra: Optional additional payload for codec-specific data not covered by Draco format
    """
    positions: np.ndarray  # (N, 3) float
    scales: np.ndarray  # (N, 1) int32
    rotations: np.ndarray  # (N, 2) int32
    opacities: np.ndarray  # (N, 1) int32
    features_dc: np.ndarray  # (N, 1) int32
    features_rest: np.ndarray  # (N, 9) int32
    extra: Optional[Payload] = None

    def to(self, device) -> Self:
        """
        Move the Payload to the specified device.

        Since DracoPayload uses numpy arrays (CPU-only), only the extra
        payload (if present) is moved to the target device.

        Args:
            device: The target device (e.g., 'cpu', 'cuda', torch.device).

        Returns:
            A new DracoPayload with the extra payload moved to the target device.
        """
        return DracoPayload(
            positions=self.positions,
            scales=self.scales,
            rotations=self.rotations,
            opacities=self.opacities,
            features_dc=self.features_dc,
            features_rest=self.features_rest,
            extra=self.extra.to(device) if self.extra is not None else None,
        )


class DracoCapableInterframeCodecInterface(InterframeCodecInterface):
    """
    Abstract interface for interframe codecs that output Draco-compatible payloads.

    This interface extends InterframeCodecInterface to ensure that all encode/decode
    methods work with DracoPayload. Subclasses must implement conversion methods
    between their internal payload format and DracoPayload.

    The workflow is:
        Encoding:
            1. Call super's encode method to get internal payload
            2. Convert internal payload to DracoPayload via abstract conversion method

        Decoding:
            1. Convert DracoPayload to internal payload via abstract conversion method
            2. Call super's decode method with internal payload
    """

    # =========================================================================
    # Abstract conversion methods - must be implemented by subclasses
    # =========================================================================

    @abstractmethod
    def interframe_payload_to_draco(self, payload: Payload) -> DracoPayload:
        """
        Convert the output of encode_interframe to a DracoPayload.

        Args:
            payload: The internal payload from the parent class's encode_interframe.

        Returns:
            A DracoPayload containing the data in Draco-compatible format.
        """
        pass

    @abstractmethod
    def keyframe_payload_to_draco(self, payload: Payload) -> DracoPayload:
        """
        Convert the output of encode_keyframe to a DracoPayload.

        Args:
            payload: The internal payload from the parent class's encode_keyframe.

        Returns:
            A DracoPayload containing the data in Draco-compatible format.
        """
        pass

    @abstractmethod
    def draco_to_interframe_payload(self, draco_payload: DracoPayload) -> Payload:
        """
        Convert a DracoPayload to the internal payload format for decode_interframe.

        Args:
            draco_payload: The DracoPayload to convert.

        Returns:
            An internal payload suitable for the parent class's decode_interframe.
        """
        pass

    @abstractmethod
    def draco_to_keyframe_payload(self, draco_payload: DracoPayload) -> Payload:
        """
        Convert a DracoPayload to the internal payload format for decode_keyframe.

        Args:
            draco_payload: The DracoPayload to convert.

        Returns:
            An internal payload suitable for the parent class's decode_keyframe.
        """
        pass

    # =========================================================================
    # Overridden encode/decode methods that work with DracoPayload
    # =========================================================================

    def decode_interframe(
        self, payload: DracoPayload, prev_context: InterframeCodecContext
    ) -> InterframeCodecContext:
        """
        Decode a DracoPayload to reconstruct the next frame's context.

        This method first converts the DracoPayload to the internal format,
        then delegates to the parent class's decode_interframe.

        Args:
            payload: The DracoPayload containing the delta data.
            prev_context: The context of the previous frame.

        Returns:
            The reconstructed context for the current frame.
        """
        internal_payload = self.draco_to_interframe_payload(payload)
        return super().decode_interframe(internal_payload, prev_context)

    def encode_interframe(
        self,
        prev_context: InterframeCodecContext,
        next_context: InterframeCodecContext,
    ) -> DracoPayload:
        """
        Encode the difference between two consecutive frames as a DracoPayload.

        This method delegates to the parent class's encode_interframe,
        then converts the result to a DracoPayload.

        Args:
            prev_context: The context of the previous frame.
            next_context: The context of the next frame.

        Returns:
            A DracoPayload containing the delta information.
        """
        internal_payload = super().encode_interframe(prev_context, next_context)
        return self.interframe_payload_to_draco(internal_payload)

    def decode_keyframe(self, payload: DracoPayload) -> InterframeCodecContext:
        """
        Decode a DracoPayload keyframe to create initial context.

        This method first converts the DracoPayload to the internal format,
        then delegates to the parent class's decode_keyframe.

        Args:
            payload: The DracoPayload containing full keyframe data.

        Returns:
            The context for the first/key frame.
        """
        internal_payload = self.draco_to_keyframe_payload(payload)
        return super().decode_keyframe(internal_payload)

    def encode_keyframe(self, context: InterframeCodecContext) -> DracoPayload:
        """
        Encode the first frame as a DracoPayload keyframe.

        This method delegates to the parent class's encode_keyframe,
        then converts the result to a DracoPayload.

        Args:
            context: The context of the first frame.

        Returns:
            A DracoPayload containing the full keyframe data.
        """
        internal_payload = super().encode_keyframe(context)
        return self.keyframe_payload_to_draco(internal_payload)
