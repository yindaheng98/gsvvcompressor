"""
VQ + XYZ quantization + Two-Pass Draco compression encoder/decoder combination.

This module uses two-pass Draco compression which accumulates all frames
and compresses them as a single block for better compression efficiency.
"""

from dataclasses import dataclass, field
from typing import Optional

from ..draco.interface import DracoInterframeCodecInterface
from ..draco.twopass import TwoPassDracoSerializer, TwoPassDracoDeserializer
from ..interframe.combine import CombinedInterframeEncoderInitConfig
from ..interframe.encoder import InterframeEncoder
from ..interframe.decoder import InterframeDecoder
from ..vq.interface import VQInterframeCodecConfig
from ..xyz.interface import XYZQuantInterframeCodecConfig
from .registry import register_encoder, register_decoder
from .vq_xyz_1mask import VQXYZQuantMergeMaskInterframeCodecInterface
from .vq_xyz_draco import VQXYZDracoInterframeCodecTranscodingInterface


def VQXYZDraco2PassEncoder(
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
    """Create an encoder combining VQ + XYZ quantization + Two-Pass Draco compression."""

    # Use merged mask interface for better compression efficiency
    combined_interface = VQXYZQuantMergeMaskInterframeCodecInterface()

    # Create transcoding interface
    transcoder = VQXYZDracoInterframeCodecTranscodingInterface()

    # Wrap with Draco interface
    draco_interface = DracoInterframeCodecInterface(combined_interface, transcoder)

    combined_config = CombinedInterframeEncoderInitConfig(
        init_configs=[xyz_config, vq_config]
    )

    serializer = TwoPassDracoSerializer(
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


def VQXYZDraco2PassDecoder(
    payload_device: Optional[str] = None,
) -> InterframeDecoder:
    """Create a decoder for VQ + XYZ quantization + Two-Pass Draco compressed data."""

    # Use merged mask interface for better compression efficiency
    combined_interface = VQXYZQuantMergeMaskInterframeCodecInterface()

    # Create transcoding interface
    transcoder = VQXYZDracoInterframeCodecTranscodingInterface()

    # Wrap with Draco interface
    draco_interface = DracoInterframeCodecInterface(combined_interface, transcoder)

    deserializer = TwoPassDracoDeserializer()

    return InterframeDecoder(
        deserializer=deserializer,
        interface=draco_interface,
        payload_device=payload_device,
    )


@dataclass
class VQXYZDraco2PassEncoderConfig:
    """Configuration for VQ + XYZ + Two-Pass Draco encoder."""
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
class VQXYZDraco2PassDecoderConfig:
    """Configuration for VQ + XYZ + Two-Pass Draco decoder."""
    payload_device: Optional[str] = None


def build_vqxyzdraco2pass_encoder(config: VQXYZDraco2PassEncoderConfig) -> InterframeEncoder:
    """Build encoder from configuration."""
    return VQXYZDraco2PassEncoder(
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


def build_vqxyzdraco2pass_decoder(config: VQXYZDraco2PassDecoderConfig) -> InterframeDecoder:
    """Build decoder from configuration."""
    return VQXYZDraco2PassDecoder(
        payload_device=config.payload_device,
    )


# Register
register_encoder(
    "vqxyzdraco2pass",
    build_vqxyzdraco2pass_encoder,
    VQXYZDraco2PassEncoderConfig,
    "VQ + XYZ quantization + Two-Pass Draco compression",
)

register_decoder(
    "vqxyzdraco2pass",
    build_vqxyzdraco2pass_decoder,
    VQXYZDraco2PassDecoderConfig,
    "VQ + XYZ quantization + Two-Pass Draco decompression",
)
