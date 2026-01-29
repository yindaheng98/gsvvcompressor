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
    CombinedInterframeCodecInterface,
    CombinedInterframeEncoderInitConfig,
    CombinedPayload,
)
from ..interframe.encoder import InterframeEncoder
from ..interframe.decoder import InterframeDecoder
from ..payload import Payload
from ..vq.interface import (
    VQInterframeCodecInterface,
    VQInterframeCodecConfig,
    VQKeyframePayload,
    VQInterframePayload,
)
from ..xyz.interface import (
    XYZQuantInterframeCodecInterface,
    XYZQuantInterframeCodecConfig,
    XYZQuantKeyframePayload,
    XYZQuantInterframePayload,
)
from ..xyz.quant import XYZQuantConfig
from .registry import register_encoder, register_decoder


@dataclass
class VQXYZDracoKeyframeExtra(Payload):
    """
    Extra data for VQ+XYZ keyframe that cannot be stored in Draco format.

    Attributes:
        quant_config: XYZ quantization configuration.
        codebook_dict: VQ codebooks.
        max_sh_degree: Maximum SH degree.
        tolerance: Tolerance for inter-frame change detection.
    """
    quant_config: XYZQuantConfig
    codebook_dict: Dict[str, torch.Tensor]
    max_sh_degree: int
    tolerance: int

    def to(self, device) -> Self:
        return VQXYZDracoKeyframeExtra(
            quant_config=XYZQuantConfig(
                step_size=self.quant_config.step_size,
                origin=self.quant_config.origin.to(device),
            ),
            codebook_dict={k: v.to(device) for k, v in self.codebook_dict.items()},
            max_sh_degree=self.max_sh_degree,
            tolerance=self.tolerance,
        )


class VQXYZDracoInterframeCodecTranscodingInterface(DracoInterframeCodecTranscodingInterface):
    """
    Transcoding interface for VQ + XYZ combined codec to/from DracoPayload.

    For keyframes:
        - Positions: quantized XYZ coordinates (int32)
        - Scales, rotations, opacities, features_dc, features_rest: VQ indices
        - Extra: quant_config, codebook_dict, max_sh_degree, tolerance

    For interframes:
        - The sparse change data cannot be efficiently represented in Draco format
        - Store the entire CombinedPayload in extra with minimal Draco fields
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

        assert vq_payload.max_sh_degree == 3, "Only max_sh_degree=3 is supported."

        ids_dict = vq_payload.ids_dict
        max_sh_degree = vq_payload.max_sh_degree

        # Extract model attributes (reduced 3DGS format) directly from model and ids_dict
        positions = xyz_payload.quantized_xyz.cpu().numpy().astype(np.int32)  # (N, 3)
        scales = ids_dict["scaling"].cpu().numpy().reshape(-1, 1).astype(np.int32)  # (N, 1)
        rotations = np.column_stack([
            ids_dict["rotation_re"].cpu().numpy(),
            ids_dict["rotation_im"].cpu().numpy()
        ]).astype(np.int32)  # (N, 2)
        opacities = ids_dict["opacity"].cpu().numpy().reshape(-1, 1).astype(np.int32)  # (N, 1)
        features_dc = ids_dict["features_dc"].cpu().numpy().reshape(-1, 1).astype(np.int32)  # (N, 1)
        # features_rest: combine all sh_degrees into (N, 9)
        features_rest_list = [ids_dict[f"features_rest_{sh_degree}"].cpu().numpy() for sh_degree in range(max_sh_degree)]
        features_rest = np.column_stack(features_rest_list).astype(np.int32)  # (N, 9)

        # Store extra data
        extra = VQXYZDracoKeyframeExtra(
            quant_config=xyz_payload.quant_config,
            codebook_dict=vq_payload.codebook_dict,
            max_sh_degree=max_sh_degree,
            tolerance=xyz_payload.tolerance,
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

        assert extra.max_sh_degree == 3, "Only max_sh_degree=3 is supported."

        max_sh_degree = extra.max_sh_degree

        # Reconstruct quantized XYZ
        quantized_xyz = torch.from_numpy(draco_payload.positions)

        # Reconstruct VQ ids_dict
        ids_dict = {
            'scaling': torch.from_numpy(draco_payload.scales.flatten()),
            'rotation_re': torch.from_numpy(draco_payload.rotations[:, 0]),
            'rotation_im': torch.from_numpy(draco_payload.rotations[:, 1]),
            'opacity': torch.from_numpy(draco_payload.opacities.flatten()),
            'features_dc': torch.from_numpy(draco_payload.features_dc.flatten()).unsqueeze(-1),
        }

        # Reconstruct features_rest
        features_rest = draco_payload.features_rest  # (N, 9)
        for sh_degree in range(max_sh_degree):
            start_idx = sh_degree * 3
            end_idx = start_idx + 3
            ids_dict[f'features_rest_{sh_degree}'] = torch.from_numpy(
                features_rest[:, start_idx:end_idx]
            )

        # Create XYZ keyframe payload
        xyz_payload = XYZQuantKeyframePayload(
            quant_config=extra.quant_config,
            quantized_xyz=quantized_xyz,
            tolerance=extra.tolerance,
        )

        # Create VQ keyframe payload
        vq_payload = VQKeyframePayload(
            ids_dict=ids_dict,
            codebook_dict=extra.codebook_dict,
            max_sh_degree=extra.max_sh_degree,
        )

        return CombinedPayload(payloads=[xyz_payload, vq_payload])

    def interframe_payload_to_draco(self, payload: CombinedPayload) -> DracoPayload:
        pass  # TODO

    def draco_to_interframe_payload(self, draco_payload: DracoPayload) -> CombinedPayload:
        pass  # TODO


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
    vq_interface = VQInterframeCodecInterface()
    xyz_interface = XYZQuantInterframeCodecInterface()
    combined_interface = CombinedInterframeCodecInterface([xyz_interface, vq_interface])

    # Create transcoding interface
    transcoder = VQXYZDracoInterframeCodecTranscodingInterface()

    # Wrap with Draco interface
    draco_interface = DracoInterframeCodecInterface(combined_interface, transcoder)

    combined_config = CombinedInterframeEncoderInitConfig(
        init_configs=[xyz_config, vq_config]
    )

    serializer = DracoSerializer(
        level=zstd_level,
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
) -> InterframeDecoder:
    """Create a decoder for VQ + XYZ quantization + Draco compressed data."""
    vq_interface = VQInterframeCodecInterface()
    xyz_interface = XYZQuantInterframeCodecInterface()
    combined_interface = CombinedInterframeCodecInterface([xyz_interface, vq_interface])

    # Create transcoding interface
    transcoder = VQXYZDracoInterframeCodecTranscodingInterface()

    # Wrap with Draco interface
    draco_interface = DracoInterframeCodecInterface(combined_interface, transcoder)

    deserializer = DracoDeserializer()

    return InterframeDecoder(
        deserializer=deserializer,
        interface=draco_interface,
        payload_device=payload_device,
    )


@dataclass
class VQXYZDracoEncoderConfig:
    """Configuration for VQ + XYZ + Draco encoder."""
    vq: VQInterframeCodecConfig = field(default_factory=VQInterframeCodecConfig)
    xyz: XYZQuantInterframeCodecConfig = field(default_factory=XYZQuantInterframeCodecConfig)
    zstd_level: int = 7
    draco_level: int = 0
    qp: int = 0
    qscale: int = 0
    qrotation: int = 0
    qopacity: int = 0
    qfeaturedc: int = 0
    qfeaturerest: int = 0
    payload_device: Optional[str] = None


@dataclass
class VQXYZDracoDecoderConfig:
    """Configuration for VQ + XYZ + Draco decoder."""
    payload_device: Optional[str] = None


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
