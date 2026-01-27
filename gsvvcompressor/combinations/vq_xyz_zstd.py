"""
VQ + XYZ quantization + Zstd compression encoder/decoder combination.
"""

from dataclasses import dataclass, field
from typing import Optional

from ..compress.zstd import ZstdSerializer, ZstdDeserializer
from ..interframe.combine import (
    CombinedInterframeCodecInterface,
    CombinedInterframeEncoderInitConfig,
)
from ..interframe.encoder import InterframeEncoder
from ..interframe.decoder import InterframeDecoder
from ..vq.interface import VQInterframeCodecInterface, VQInterframeCodecConfig
from ..xyz.interface import XYZQuantInterframeCodecInterface, XYZQuantInterframeCodecConfig
from .registry import register_encoder, register_decoder


def VQXYZZstdEncoder(
    vq_config: VQInterframeCodecConfig,
    xyz_config: XYZQuantInterframeCodecConfig,
    zstd_level: int = 7,
    payload_device: Optional[str] = None,
) -> InterframeEncoder:
    """Create an encoder combining VQ + XYZ quantization + Zstd compression."""
    vq_interface = VQInterframeCodecInterface()
    xyz_interface = XYZQuantInterframeCodecInterface()
    combined_interface = CombinedInterframeCodecInterface([xyz_interface, vq_interface])
    combined_config = CombinedInterframeEncoderInitConfig(
        init_configs=[xyz_config, vq_config]
    )
    serializer = ZstdSerializer(level=zstd_level)
    return InterframeEncoder(
        serializer=serializer,
        interface=combined_interface,
        init_config=combined_config,
        payload_device=payload_device,
    )


def VQXYZZstdDecoder(
    payload_device: Optional[str] = None,
) -> InterframeDecoder:
    """Create a decoder for VQ + XYZ quantization + Zstd compressed data."""
    vq_interface = VQInterframeCodecInterface()
    xyz_interface = XYZQuantInterframeCodecInterface()
    combined_interface = CombinedInterframeCodecInterface([xyz_interface, vq_interface])
    deserializer = ZstdDeserializer()
    return InterframeDecoder(
        deserializer=deserializer,
        interface=combined_interface,
        payload_device=payload_device,
    )


@dataclass
class VQXYZZstdEncoderConfig:
    """Configuration for VQ + XYZ + Zstd encoder."""
    vq: VQInterframeCodecConfig = field(default_factory=VQInterframeCodecConfig)
    xyz: XYZQuantInterframeCodecConfig = field(default_factory=XYZQuantInterframeCodecConfig)
    zstd_level: int = 7
    payload_device: Optional[str] = None


@dataclass
class VQXYZZstdDecoderConfig:
    """Configuration for VQ + XYZ + Zstd decoder."""
    payload_device: Optional[str] = None


def build_vqxyzzstd_encoder(config: VQXYZZstdEncoderConfig) -> InterframeEncoder:
    """Build encoder from configuration."""
    return VQXYZZstdEncoder(
        vq_config=config.vq,
        xyz_config=config.xyz,
        zstd_level=config.zstd_level,
        payload_device=config.payload_device,
    )


def build_vqxyzzstd_decoder(config: VQXYZZstdDecoderConfig) -> InterframeDecoder:
    """Build decoder from configuration."""
    return VQXYZZstdDecoder(
        payload_device=config.payload_device,
    )


# Register
register_encoder(
    "vqxyzzstd",
    build_vqxyzzstd_encoder,
    VQXYZZstdEncoderConfig,
    "VQ + XYZ quantization + Zstd compression",
)

register_decoder(
    "vqxyzzstd",
    build_vqxyzzstd_decoder,
    VQXYZZstdDecoderConfig,
    "VQ + XYZ quantization + Zstd decompression",
)
