"""
Draco-capable interframe codec interface.

This module provides interfaces for interframe codecs that can output
Draco-compatible payloads for efficient compression.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Self

import numpy as np

from gaussian_splatting import GaussianModel

from ..payload import Payload
from ..interframe.interface import (
    InterframeCodecInterface,
    InterframeCodecContext,
    InterframeEncoderInitConfig,
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


class DracoInterframeCodecTranscodingInterface(ABC):
    """
    Abstract interface for transcoding between internal payloads and DracoPayload.

    This interface defines the conversion methods needed to transform codec-specific
    payloads to/from DracoPayload format. Implementations of this interface handle
    the data format conversion without implementing the actual encoding/decoding logic.
    """

    @abstractmethod
    def interframe_payload_to_draco(self, payload: Payload) -> DracoPayload:
        """
        Convert an interframe payload to a DracoPayload.

        Args:
            payload: The internal payload from encode_interframe.

        Returns:
            A DracoPayload containing the data in Draco-compatible format.
        """
        pass

    @abstractmethod
    def keyframe_payload_to_draco(self, payload: Payload) -> DracoPayload:
        """
        Convert a keyframe payload to a DracoPayload.

        Args:
            payload: The internal payload from encode_keyframe.

        Returns:
            A DracoPayload containing the data in Draco-compatible format.
        """
        pass

    @abstractmethod
    def draco_to_interframe_payload(self, draco_payload: DracoPayload) -> Payload:
        """
        Convert a DracoPayload to an interframe payload.

        Args:
            draco_payload: The DracoPayload to convert.

        Returns:
            An internal payload suitable for decode_interframe.
        """
        pass

    @abstractmethod
    def draco_to_keyframe_payload(self, draco_payload: DracoPayload) -> Payload:
        """
        Convert a DracoPayload to a keyframe payload.

        Args:
            draco_payload: The DracoPayload to convert.

        Returns:
            An internal payload suitable for decode_keyframe.
        """
        pass


class DracoInterframeCodecInterface(InterframeCodecInterface):
    """
    Interframe codec that wraps another codec and transcodes to/from DracoPayload.

    This class uses composition to combine an InterframeCodecInterface (for the
    actual encoding/decoding logic) with a DracoInterframeCodecTranscodingInterface
    (for payload format conversion).

    The workflow is:
        Encoding:
            1. Call inner codec's encode method to get internal payload
            2. Convert internal payload to DracoPayload via transcoding interface

        Decoding:
            1. Convert DracoPayload to internal payload via transcoding interface
            2. Call inner codec's decode method with internal payload
    """

    def __init__(
        self,
        codec: InterframeCodecInterface,
        transcoder: DracoInterframeCodecTranscodingInterface,
    ):
        """
        Initialize the DracoInterframeCodecInterface.

        Args:
            codec: The inner codec that performs actual encoding/decoding.
            transcoder: The transcoding interface for payload conversion.
        """
        self._codec = codec
        self._transcoder = transcoder

    # =========================================================================
    # Encode/decode methods - transcode then delegate
    # =========================================================================

    def decode_interframe(
        self, payload: DracoPayload, prev_context: InterframeCodecContext
    ) -> InterframeCodecContext:
        """
        Decode a DracoPayload to reconstruct the next frame's context.

        This method first converts the DracoPayload to the internal format,
        then delegates to the inner codec's decode_interframe.

        Args:
            payload: The DracoPayload containing the delta data.
            prev_context: The context of the previous frame.

        Returns:
            The reconstructed context for the current frame.
        """
        internal_payload = self._transcoder.draco_to_interframe_payload(payload)
        return self._codec.decode_interframe(internal_payload, prev_context)

    def encode_interframe(
        self,
        prev_context: InterframeCodecContext,
        next_context: InterframeCodecContext,
    ) -> DracoPayload:
        """
        Encode the difference between two consecutive frames as a DracoPayload.

        This method delegates to the inner codec's encode_interframe,
        then converts the result to a DracoPayload.

        Args:
            prev_context: The context of the previous frame.
            next_context: The context of the next frame.

        Returns:
            A DracoPayload containing the delta information.
        """
        internal_payload = self._codec.encode_interframe(prev_context, next_context)
        return self._transcoder.interframe_payload_to_draco(internal_payload)

    def decode_keyframe(self, payload: DracoPayload) -> InterframeCodecContext:
        """
        Decode a DracoPayload keyframe to create initial context.

        This method first converts the DracoPayload to the internal format,
        then delegates to the inner codec's decode_keyframe.

        Args:
            payload: The DracoPayload containing full keyframe data.

        Returns:
            The context for the first/key frame.
        """
        internal_payload = self._transcoder.draco_to_keyframe_payload(payload)
        return self._codec.decode_keyframe(internal_payload)

    def decode_keyframe_for_encode(
        self, payload: DracoPayload, context: InterframeCodecContext
    ) -> InterframeCodecContext:
        """
        Decode a DracoPayload keyframe during encoding to avoid error accumulation.

        This method first converts the DracoPayload to the internal format,
        then delegates to the inner codec's decode_keyframe_for_encode.

        Args:
            payload: The DracoPayload that was just encoded.
            context: The original context used for encoding this keyframe.

        Returns:
            The reconstructed context as the decoder would produce it.
        """
        internal_payload = self._transcoder.draco_to_keyframe_payload(payload)
        return self._codec.decode_keyframe_for_encode(internal_payload, context)

    def decode_interframe_for_encode(
        self, payload: DracoPayload, prev_context: InterframeCodecContext
    ) -> InterframeCodecContext:
        """
        Decode a DracoPayload interframe during encoding to avoid error accumulation.

        This method first converts the DracoPayload to the internal format,
        then delegates to the inner codec's decode_interframe_for_encode.

        Args:
            payload: The DracoPayload that was just encoded.
            prev_context: The previous frame's context (reconstructed version).

        Returns:
            The reconstructed context as the decoder would produce it.
        """
        internal_payload = self._transcoder.draco_to_interframe_payload(payload)
        return self._codec.decode_interframe_for_encode(internal_payload, prev_context)

    def encode_keyframe(self, context: InterframeCodecContext) -> DracoPayload:
        """
        Encode the first frame as a DracoPayload keyframe.

        This method delegates to the inner codec's encode_keyframe,
        then converts the result to a DracoPayload.

        Args:
            context: The context of the first frame.

        Returns:
            A DracoPayload containing the full keyframe data.
        """
        internal_payload = self._codec.encode_keyframe(context)
        return self._transcoder.keyframe_payload_to_draco(internal_payload)

    # =========================================================================
    # Other methods - delegate directly to inner codec
    # =========================================================================

    def keyframe_to_context(
        self, frame: GaussianModel, init_config: InterframeEncoderInitConfig
    ) -> InterframeCodecContext:
        """
        Convert a keyframe to a Context.

        Delegates directly to the inner codec.

        Args:
            frame: The GaussianModel frame to convert.
            init_config: Encoder initialization configuration.

        Returns:
            The corresponding Context representation.
        """
        return self._codec.keyframe_to_context(frame, init_config)

    def interframe_to_context(
        self,
        frame: GaussianModel,
        prev_context: InterframeCodecContext,
    ) -> InterframeCodecContext:
        """
        Convert a frame to a Context using the previous context as reference.

        Delegates directly to the inner codec.

        Args:
            frame: The GaussianModel frame to convert.
            prev_context: The context from the previous frame.

        Returns:
            The corresponding Context representation.
        """
        return self._codec.interframe_to_context(frame, prev_context)

    def context_to_frame(
        self, context: InterframeCodecContext, frame: GaussianModel
    ) -> GaussianModel:
        """
        Convert a Context back to a frame.

        Delegates directly to the inner codec.

        Args:
            context: The Context to convert.
            frame: An empty GaussianModel or one from previous pipeline steps.

        Returns:
            The modified GaussianModel with the frame data.
        """
        return self._codec.context_to_frame(context, frame)
