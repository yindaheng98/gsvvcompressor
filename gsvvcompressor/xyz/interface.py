"""
XYZ quantization-based inter-frame codec interface.

This module provides an inter-frame codec that only operates on the xyz
coordinates of a GaussianModel using quantization.
"""

from dataclasses import dataclass
from typing import Optional, Self

import torch

from gaussian_splatting import GaussianModel

from ..payload import Payload
from ..interframe import InterframeEncoderInitConfig, InterframeCodecContext, InterframeCodecInterface
from .quant import XYZQuantConfig, compute_quant_config, quantize_xyz, dequantize_xyz


@dataclass
class XYZQuantInterframeCodecConfig(InterframeEncoderInitConfig):
    """
    Configuration parameters for XYZ quantization-based inter-frame codec.

    This dataclass holds the initialization settings for xyz coordinate
    quantization.

    Attributes:
        k: Which nearest neighbor to use for step size estimation (1 = nearest).
        sample_size: Number of points to sample for NN estimation.
        seed: Random seed for reproducible sampling.
        quantile: Quantile of NN distances to use for dense scale estimation.
        alpha: Scaling factor for step size.
        min_step: Optional minimum step size.
        max_step: Optional maximum step size.
        tolerance: Tolerance for inter-frame change detection. Only coordinates
            with absolute difference > tolerance are considered changed.
    """
    k: int = 1
    sample_size: Optional[int] = 10000
    seed: Optional[int] = 42
    quantile: float = 0.05
    alpha: float = 0.2
    min_step: Optional[float] = None
    max_step: Optional[float] = None
    tolerance: int = 0


@dataclass
class XYZQuantInterframeCodecContext(InterframeCodecContext):
    """
    Context data for XYZ quantization-based inter-frame encoding/decoding.

    This dataclass holds the quantization state including the quantization
    configuration and quantized xyz coordinates.

    Attributes:
        quant_config: The quantization configuration (step_size, origin).
        quantized_xyz: The quantized xyz coordinates, shape (N, 3), dtype int32.
        tolerance: Tolerance for inter-frame change detection.
    """
    quant_config: XYZQuantConfig
    quantized_xyz: torch.Tensor  # shape (N, 3), dtype int32
    tolerance: int = 0


@dataclass
class XYZQuantKeyframePayload(Payload):
    """
    Payload for XYZ quantization keyframe data.

    Contains the quantization configuration and full quantized xyz coordinates.

    Attributes:
        quant_config: The quantization configuration.
        quantized_xyz: The quantized xyz coordinates.
        tolerance: Tolerance for inter-frame change detection.
    """
    quant_config: XYZQuantConfig
    quantized_xyz: torch.Tensor
    tolerance: int = 0

    def to(self, device) -> Self:
        """
        Move the Payload to the specified device.

        Args:
            device: The target device (e.g., 'cpu', 'cuda', torch.device).

        Returns:
            A new XYZQuantKeyframePayload instance on the target device.
        """
        return XYZQuantKeyframePayload(
            quant_config=XYZQuantConfig(
                step_size=self.quant_config.step_size,
                origin=self.quant_config.origin.to(device),
            ),
            quantized_xyz=self.quantized_xyz.to(device),
            tolerance=self.tolerance,
        )


@dataclass
class XYZQuantInterframePayload(Payload):
    """
    Payload for XYZ quantization interframe data.

    Contains only the changed quantized xyz coordinates for subsequent frames.
    The quantization configuration is inherited from the keyframe context.

    Attributes:
        xyz_mask: Boolean tensor indicating which xyz values changed, shape (N,).
        quantized_xyz: Only the changed quantized xyz values (sparse), shape (M, 3).
    """
    xyz_mask: torch.Tensor
    quantized_xyz: torch.Tensor

    def to(self, device) -> Self:
        """
        Move the Payload to the specified device.

        Args:
            device: The target device (e.g., 'cpu', 'cuda', torch.device).

        Returns:
            A new XYZQuantInterframePayload instance on the target device.
        """
        return XYZQuantInterframePayload(
            xyz_mask=self.xyz_mask.to(device),
            quantized_xyz=self.quantized_xyz.to(device),
        )


class XYZQuantInterframeCodecInterface(InterframeCodecInterface):
    """
    XYZ quantization-based inter-frame encoding/decoding interface.

    This interface uses coordinate quantization to compress the xyz coordinates
    of GaussianModel. The keyframe computes the quantization configuration,
    and subsequent frames use the same configuration to quantize their coordinates.

    Only operates on xyz coordinates; other GaussianModel attributes are not modified.
    """

    def decode_interframe(
        self,
        payload: XYZQuantInterframePayload,
        prev_context: XYZQuantInterframeCodecContext,
    ) -> XYZQuantInterframeCodecContext:
        """
        Decode a delta payload to reconstruct the next frame's context.

        Applies the changed xyz values from the payload to the previous context.

        Args:
            payload: The delta payload containing changed quantized xyz with mask.
            prev_context: The context of the previous frame (contains quant_config).

        Returns:
            The reconstructed context for the current frame.
        """
        # Clone previous quantized_xyz and apply changes
        new_quantized_xyz = prev_context.quantized_xyz.clone()
        new_quantized_xyz[payload.xyz_mask] = payload.quantized_xyz

        return XYZQuantInterframeCodecContext(
            quant_config=prev_context.quant_config,
            quantized_xyz=new_quantized_xyz,
            tolerance=prev_context.tolerance,
        )

    def encode_interframe(
        self,
        prev_context: XYZQuantInterframeCodecContext,
        next_context: XYZQuantInterframeCodecContext,
    ) -> XYZQuantInterframePayload:
        """
        Encode the difference between two consecutive frames.

        Compares prev and next contexts to find changed xyz coordinates and stores
        only the changed values with their corresponding mask.

        Args:
            prev_context: The context of the previous frame.
            next_context: The context of the next frame.

        Returns:
            A payload containing only changed quantized xyz with mask.
        """
        prev_xyz = prev_context.quantized_xyz
        next_xyz = next_context.quantized_xyz
        tolerance = prev_context.tolerance

        # Find changed xyz coordinates (any coordinate difference > tolerance)
        diff = (prev_xyz - next_xyz).abs()
        mask = (diff > tolerance).any(dim=-1)
        changed_xyz = next_xyz[mask]

        return XYZQuantInterframePayload(
            xyz_mask=mask,
            quantized_xyz=changed_xyz,
        )

    def decode_keyframe(self, payload: XYZQuantKeyframePayload) -> XYZQuantInterframeCodecContext:
        """
        Decode a keyframe payload to create initial context.

        Args:
            payload: The keyframe payload containing quant_config and quantized xyz.

        Returns:
            The context for the first/key frame.
        """
        return XYZQuantInterframeCodecContext(
            quant_config=payload.quant_config,
            quantized_xyz=payload.quantized_xyz,
            tolerance=payload.tolerance,
        )

    def encode_keyframe(self, context: XYZQuantInterframeCodecContext) -> XYZQuantKeyframePayload:
        """
        Encode the first frame as a keyframe.

        Args:
            context: The context of the first frame.

        Returns:
            A payload containing the quant_config and quantized xyz.
        """
        return XYZQuantKeyframePayload(
            quant_config=context.quant_config,
            quantized_xyz=context.quantized_xyz,
            tolerance=context.tolerance,
        )

    def keyframe_to_context(
        self,
        frame: GaussianModel,
        init_config: XYZQuantInterframeCodecConfig,
    ) -> XYZQuantInterframeCodecContext:
        """
        Convert a keyframe to a XYZQuantInterframeCodecContext.

        Computes quantization configuration from the frame's xyz coordinates
        and quantizes them.

        Args:
            frame: The GaussianModel frame to convert.
            init_config: Configuration parameters for quantization.

        Returns:
            The corresponding XYZQuantInterframeCodecContext representation.
        """
        xyz = frame.get_xyz

        # Compute quantization configuration from xyz coordinates
        quant_config = compute_quant_config(
            points=xyz,
            k=init_config.k,
            sample_size=init_config.sample_size,
            seed=init_config.seed,
            quantile=init_config.quantile,
            alpha=init_config.alpha,
            min_step=init_config.min_step,
            max_step=init_config.max_step,
        )

        # Quantize xyz coordinates
        quantized_xyz = quantize_xyz(xyz, quant_config)

        return XYZQuantInterframeCodecContext(
            quant_config=quant_config,
            quantized_xyz=quantized_xyz,
            tolerance=init_config.tolerance,
        )

    def interframe_to_context(
        self,
        frame: GaussianModel,
        prev_context: XYZQuantInterframeCodecContext,
    ) -> XYZQuantInterframeCodecContext:
        """
        Convert a frame to a XYZQuantInterframeCodecContext using the previous context's config.

        Uses the quantization config from the previous context to quantize xyz coordinates.

        Args:
            frame: The GaussianModel frame to convert.
            prev_context: The context from the previous frame.

        Returns:
            The corresponding XYZQuantInterframeCodecContext representation.
        """
        xyz = frame.get_xyz

        # Quantize xyz using the existing quant_config from keyframe
        quantized_xyz = quantize_xyz(xyz, prev_context.quant_config)

        return XYZQuantInterframeCodecContext(
            quant_config=prev_context.quant_config,
            quantized_xyz=quantized_xyz,
            tolerance=prev_context.tolerance,
        )

    def context_to_frame(
        self,
        context: XYZQuantInterframeCodecContext,
        frame: GaussianModel,
    ) -> GaussianModel:
        """
        Convert a XYZQuantInterframeCodecContext back to a GaussianModel frame.

        Dequantizes the xyz coordinates and sets them on the frame.
        Only modifies xyz coordinates; other attributes are not touched.

        Args:
            context: The XYZQuantInterframeCodecContext to convert.
            frame: An empty GaussianModel or one from previous pipeline steps.
                This frame will be modified in-place with the xyz data.

        Returns:
            The modified GaussianModel with the xyz data.
        """
        # Dequantize xyz coordinates
        xyz = dequantize_xyz(
            context.quantized_xyz,
            context.quant_config,
            dtype=frame.get_xyz.dtype,
        )

        # Set xyz on the frame
        frame._xyz = torch.nn.Parameter(xyz.to(frame.get_xyz.device))

        return frame
