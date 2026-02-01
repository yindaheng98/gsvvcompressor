"""
VQ + XYZ quantization + Draco compression encoder/decoder combination.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Self

import numpy as np
import torch

from ..draco.interface import (
    DracoInterframeCodecInterface,
    DracoInterframeCodecTranscodingInterface,
    DracoPayload,
)
from ..draco.serialize import DracoSerializer, DracoDeserializer
from ..interframe.combine import (
    CombinedInterframeEncoderInitConfig,
    CombinedPayload,
)
from ..interframe.encoder import InterframeEncoder
from ..interframe.decoder import InterframeDecoder
from ..payload import Payload
from ..vq.interface import (
    VQInterframeCodecConfig,
    VQKeyframePayload,
)
from ..vq.singlemask import VQMergeMaskInterframePayload
from ..xyz.interface import (
    XYZQuantInterframeCodecConfig,
    XYZQuantKeyframePayload,
    XYZQuantInterframePayload,
)
from ..xyz.quant import XYZQuantConfig
from .registry import register_encoder, register_decoder
from .vq_xyz_1mask import VQXYZQuantMergeMaskInterframeCodecInterface


# =============================================================================
# Helper functions for converting between VQ ids_dict and Draco numpy arrays
# =============================================================================

def vq_ids_dict_to_draco_arrays(
    ids_dict: Dict[str, torch.Tensor],
    max_sh_degree: int,
) -> tuple:
    """
    Convert VQ ids_dict to Draco format numpy arrays.

    Args:
        ids_dict: Dictionary of VQ indices.
        max_sh_degree: Maximum SH degree (for features_rest).

    Returns:
        Tuple of (scales, rotations, opacities, features_dc, features_rest) numpy arrays.
    """
    assert max_sh_degree == 3, "Only max_sh_degree=3 is supported."

    scales = ids_dict["scaling"].cpu().numpy().reshape(-1, 1).astype(np.int32)
    rotations = np.column_stack([
        ids_dict["rotation_re"].cpu().numpy(),
        ids_dict["rotation_im"].cpu().numpy()
    ]).astype(np.int32)
    opacities = ids_dict["opacity"].cpu().numpy().reshape(-1, 1).astype(np.int32)
    features_dc = ids_dict["features_dc"].cpu().numpy().reshape(-1, 1).astype(np.int32)
    features_rest_list = [
        ids_dict[f"features_rest_{sh_degree}"].cpu().numpy()
        for sh_degree in range(max_sh_degree)
    ]
    features_rest = np.column_stack(features_rest_list).astype(np.int32)

    return scales, rotations, opacities, features_dc, features_rest


def draco_arrays_to_vq_ids_dict(
    scales: np.ndarray,
    rotations: np.ndarray,
    opacities: np.ndarray,
    features_dc: np.ndarray,
    features_rest: np.ndarray,
    max_sh_degree: int,
) -> Dict[str, torch.Tensor]:
    """
    Convert Draco format numpy arrays back to VQ ids_dict.

    Args:
        scales: Scale indices, shape (N, 1).
        rotations: Rotation indices, shape (N, 2).
        opacities: Opacity indices, shape (N, 1).
        features_dc: DC feature indices, shape (N, 1).
        features_rest: Rest feature indices, shape (N, max_sh_degree*3).
        max_sh_degree: Maximum SH degree.

    Returns:
        Dictionary of VQ indices as torch tensors.
    """
    assert max_sh_degree == 3, "Only max_sh_degree=3 is supported."

    ids_dict = {
        'scaling': torch.from_numpy(scales.flatten()),
        'rotation_re': torch.from_numpy(rotations[:, 0].copy()),
        'rotation_im': torch.from_numpy(rotations[:, 1].copy()),
        'opacity': torch.from_numpy(opacities.flatten()),
        'features_dc': torch.from_numpy(features_dc.flatten()).unsqueeze(-1),
    }

    for sh_degree in range(max_sh_degree):
        start_idx = sh_degree * 3
        end_idx = start_idx + 3
        ids_dict[f'features_rest_{sh_degree}'] = torch.from_numpy(
            features_rest[:, start_idx:end_idx].copy()
        )

    return ids_dict


# =============================================================================
# Extra payload classes for Draco format
# =============================================================================

@dataclass
class VQXYZDracoKeyframeExtra(Payload):
    """
    Extra data for VQ+XYZ keyframe that cannot be stored in Draco format.

    Attributes:
        quant_config: XYZ quantization configuration.
        codebook_dict: VQ codebooks.
        max_sh_degree: Maximum SH degree.
    """
    quant_config: XYZQuantConfig
    codebook_dict: Dict[str, torch.Tensor]
    max_sh_degree: int

    def to(self, device) -> Self:
        return VQXYZDracoKeyframeExtra(
            quant_config=XYZQuantConfig(
                step_size=self.quant_config.step_size,
                origin=self.quant_config.origin.to(device),
            ),
            codebook_dict={k: v.to(device) for k, v in self.codebook_dict.items()},
            max_sh_degree=self.max_sh_degree,
        )


@dataclass
class VQXYZDracoInterframeExtra(Payload):
    """
    Extra data for VQ+XYZ interframe that cannot be stored in Draco format.

    Attributes:
        ids_mask: Boolean tensor indicating which positions changed.
    """
    ids_mask: torch.Tensor

    def to(self, device) -> Self:
        return VQXYZDracoInterframeExtra(
            ids_mask=self.ids_mask.to(device),
        )


class VQXYZDracoInterframeCodecTranscodingInterface(DracoInterframeCodecTranscodingInterface):
    """
    Transcoding interface for VQ + XYZ combined codec to/from DracoPayload.

    For keyframes:
        - Positions: quantized XYZ coordinates (int32)
        - Scales, rotations, opacities, features_dc, features_rest: VQ indices
        - Extra: quant_config, codebook_dict, max_sh_degree, tolerance

    For interframes:
        - Positions: sparse quantized XYZ (only changed positions)
        - Scales, rotations, opacities, features_dc, features_rest: sparse VQ indices
        - Extra: ids_mask (boolean tensor indicating changed positions)
    """

    def keyframe_payload_to_draco(self, payload: CombinedPayload) -> DracoPayload:
        """
        Convert a VQ+XYZ keyframe CombinedPayload to DracoPayload.

        Args:
            payload: CombinedPayload containing [XYZQuantKeyframePayload, VQKeyframePayload]

        Returns:
            DracoPayload with full point cloud data and extra metadata.
        """
        xyz_payload: XYZQuantKeyframePayload = payload.payloads[0]
        vq_payload: VQKeyframePayload = payload.payloads[1]

        # Convert quantized XYZ to numpy
        positions = xyz_payload.quantized_xyz.cpu().numpy().astype(np.int32)

        # Convert VQ ids_dict to Draco arrays using helper function
        scales, rotations, opacities, features_dc, features_rest = vq_ids_dict_to_draco_arrays(
            vq_payload.ids_dict, vq_payload.max_sh_degree
        )

        # Store extra data
        extra = VQXYZDracoKeyframeExtra(
            quant_config=xyz_payload.quant_config,
            codebook_dict=vq_payload.codebook_dict,
            max_sh_degree=vq_payload.max_sh_degree,
        )

        return DracoPayload(
            positions=positions,
            scales=scales,
            rotations=rotations,
            opacities=opacities,
            features_dc=features_dc,
            features_rest=features_rest,
            extra=extra,
        )

    def draco_to_keyframe_payload(self, draco_payload: DracoPayload) -> CombinedPayload:
        """
        Convert a DracoPayload back to VQ+XYZ keyframe CombinedPayload.

        Args:
            draco_payload: DracoPayload with point cloud data and extra metadata.

        Returns:
            CombinedPayload containing [XYZQuantKeyframePayload, VQKeyframePayload]
        """
        extra: VQXYZDracoKeyframeExtra = draco_payload.extra

        # Reconstruct quantized XYZ
        quantized_xyz = torch.from_numpy(draco_payload.positions.copy())

        # Reconstruct VQ ids_dict using helper function
        ids_dict = draco_arrays_to_vq_ids_dict(
            draco_payload.scales,
            draco_payload.rotations,
            draco_payload.opacities,
            draco_payload.features_dc,
            draco_payload.features_rest,
            extra.max_sh_degree,
        )

        # Create XYZ keyframe payload
        xyz_payload = XYZQuantKeyframePayload(
            quant_config=extra.quant_config,
            quantized_xyz=quantized_xyz,
        )

        # Create VQ keyframe payload
        vq_payload = VQKeyframePayload(
            ids_dict=ids_dict,
            codebook_dict=extra.codebook_dict,
            max_sh_degree=extra.max_sh_degree,
        )

        return CombinedPayload(payloads=[xyz_payload, vq_payload])

    def interframe_payload_to_draco(self, payload: CombinedPayload) -> DracoPayload:
        """
        Convert a VQ+XYZ interframe CombinedPayload to DracoPayload.

        Args:
            payload: CombinedPayload containing [XYZQuantInterframePayload, VQMergeMaskInterframePayload]

        Returns:
            DracoPayload with sparse changed data and mask in extra.
        """
        xyz_payload: XYZQuantInterframePayload = payload.payloads[0]
        vq_payload: VQMergeMaskInterframePayload = payload.payloads[1]
        assert torch.equal(
            xyz_payload.xyz_mask, vq_payload.ids_mask
        ), "Masks in XYZ and VQ payloads must be the same."

        # Convert sparse quantized XYZ to numpy
        positions = xyz_payload.quantized_xyz.cpu().numpy().astype(np.int32)

        # Convert sparse VQ ids_dict to Draco arrays using helper function
        # Note: max_sh_degree=3 is assumed (asserted in helper function)
        scales, rotations, opacities, features_dc, features_rest = vq_ids_dict_to_draco_arrays(
            vq_payload.ids_dict, max_sh_degree=3
        )

        # Store mask in extra (both payloads should have the same merged mask)
        extra = VQXYZDracoInterframeExtra(
            ids_mask=vq_payload.ids_mask,
        )

        return DracoPayload(
            positions=positions,
            scales=scales,
            rotations=rotations,
            opacities=opacities,
            features_dc=features_dc,
            features_rest=features_rest,
            extra=extra,
        )

    def draco_to_interframe_payload(self, draco_payload: DracoPayload) -> CombinedPayload:
        """
        Convert a DracoPayload back to VQ+XYZ interframe CombinedPayload.

        Args:
            draco_payload: DracoPayload with sparse changed data and mask in extra.

        Returns:
            CombinedPayload containing [XYZQuantInterframePayload, VQMergeMaskInterframePayload]
        """
        extra: VQXYZDracoInterframeExtra = draco_payload.extra

        # Reconstruct sparse quantized XYZ
        quantized_xyz = torch.from_numpy(draco_payload.positions.copy())

        # Reconstruct sparse VQ ids_dict using helper function
        ids_dict = draco_arrays_to_vq_ids_dict(
            draco_payload.scales,
            draco_payload.rotations,
            draco_payload.opacities,
            draco_payload.features_dc,
            draco_payload.features_rest,
            max_sh_degree=3,
        )

        # Create XYZ interframe payload with merged mask
        xyz_payload = XYZQuantInterframePayload(
            xyz_mask=extra.ids_mask,
            quantized_xyz=quantized_xyz,
        )

        # Create VQ interframe payload with merged mask
        vq_payload = VQMergeMaskInterframePayload(
            ids_mask=extra.ids_mask,
            ids_dict=ids_dict,
        )

        return CombinedPayload(payloads=[xyz_payload, vq_payload])


def VQXYZDracoEncoder(
    vq_config: VQInterframeCodecConfig,
    xyz_config: XYZQuantInterframeCodecConfig,
    zstd_level: int = 7,
    draco_level: int = 0,
    qp: int = 0,
    qscale: int = 0,
    qrotation: int = 0,
    qopacity: int = 0,
    qfeaturedc: int = 0,
    qfeaturerest: int = 0,
    payload_device: Optional[str] = None,
) -> InterframeEncoder:
    """Create an encoder combining VQ + XYZ quantization + Draco compression."""

    # Use merged mask interface for better compression efficiency
    combined_interface = VQXYZQuantMergeMaskInterframeCodecInterface()

    # Create transcoding interface
    transcoder = VQXYZDracoInterframeCodecTranscodingInterface()

    # Wrap with Draco interface
    draco_interface = DracoInterframeCodecInterface(combined_interface, transcoder)

    combined_config = CombinedInterframeEncoderInitConfig(
        init_configs=[xyz_config, vq_config]
    )

    serializer = DracoSerializer(
        zstd_level=zstd_level,
        draco_level=draco_level,
        qp=qp,
        qscale=qscale,
        qrotation=qrotation,
        qopacity=qopacity,
        qfeaturedc=qfeaturedc,
        qfeaturerest=qfeaturerest,
    )

    return InterframeEncoder(
        serializer=serializer,
        interface=draco_interface,
        init_config=combined_config,
        payload_device=payload_device,
    )


def VQXYZDracoDecoder(
    payload_device: Optional[str] = None,
    device: Optional[str] = None,
) -> InterframeDecoder:
    """Create a decoder for VQ + XYZ quantization + Draco compressed data."""

    # Use merged mask interface for better compression efficiency
    combined_interface = VQXYZQuantMergeMaskInterframeCodecInterface()

    # Create transcoding interface
    transcoder = VQXYZDracoInterframeCodecTranscodingInterface()

    # Wrap with Draco interface
    draco_interface = DracoInterframeCodecInterface(combined_interface, transcoder)

    deserializer = DracoDeserializer()

    return InterframeDecoder(
        deserializer=deserializer,
        interface=draco_interface,
        payload_device=payload_device,
        device=device,
    )


@dataclass
class VQXYZDracoEncoderConfig:
    """Configuration for VQ + XYZ + Draco encoder."""
    vq: VQInterframeCodecConfig = field(default_factory=VQInterframeCodecConfig)
    xyz: XYZQuantInterframeCodecConfig = field(default_factory=XYZQuantInterframeCodecConfig)
    zstd_level: int = 7
    draco_level: int = 0
    qp: int = 30
    qscale: int = 30
    qrotation: int = 30
    qopacity: int = 30
    qfeaturedc: int = 30
    qfeaturerest: int = 30
    payload_device: Optional[str] = None


@dataclass
class VQXYZDracoDecoderConfig:
    """Configuration for VQ + XYZ + Draco decoder."""
    payload_device: Optional[str] = None
    device: Optional[str] = None


def build_vqxyzdraco_encoder(config: VQXYZDracoEncoderConfig) -> InterframeEncoder:
    """Build encoder from configuration."""
    return VQXYZDracoEncoder(
        vq_config=config.vq,
        xyz_config=config.xyz,
        zstd_level=config.zstd_level,
        draco_level=config.draco_level,
        qp=config.qp,
        qscale=config.qscale,
        qrotation=config.qrotation,
        qopacity=config.qopacity,
        qfeaturedc=config.qfeaturedc,
        qfeaturerest=config.qfeaturerest,
        payload_device=config.payload_device,
    )


def build_vqxyzdraco_decoder(config: VQXYZDracoDecoderConfig) -> InterframeDecoder:
    """Build decoder from configuration."""
    return VQXYZDracoDecoder(
        payload_device=config.payload_device,
        device=config.device,
    )


# Register
register_encoder(
    "vqxyzdraco",
    build_vqxyzdraco_encoder,
    VQXYZDracoEncoderConfig,
    "VQ + XYZ quantization + Draco compression",
)

register_decoder(
    "vqxyzdraco",
    build_vqxyzdraco_decoder,
    VQXYZDracoDecoderConfig,
    "VQ + XYZ quantization + Draco decompression",
)
