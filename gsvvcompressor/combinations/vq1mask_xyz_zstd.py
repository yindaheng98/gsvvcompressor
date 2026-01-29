"""
VQ + XYZ quantization with single merged mask + Zstd compression encoder/decoder combination.

This module provides a combined codec where VQ and XYZ share a single merged mask
for interframe encoding, improving compression efficiency.
"""

from dataclasses import dataclass, field
from typing import Optional

from ..compress.zstd import ZstdSerializer, ZstdDeserializer
from ..interframe.combine import CombinedInterframeEncoderInitConfig
from ..interframe.encoder import InterframeEncoder
from ..interframe.decoder import InterframeDecoder
from ..vq.interface import VQInterframeCodecConfig
from ..xyz.interface import XYZQuantInterframeCodecConfig
from .registry import register_encoder, register_decoder
from .vq_xyz import VQXYZQuantMergeMaskInterframeCodecInterface


def VQXYZ1MaskZstdEncoder(
    vq_config: VQInterframeCodecConfig,
    xyz_config: XYZQuantInterframeCodecConfig,
    zstd_level: int = 7,
    payload_device: Optional[str] = None,
) -> InterframeEncoder:
    """Create an encoder combining VQ + XYZ quantization with merged mask + Zstd compression."""
    combined_interface = VQXYZQuantMergeMaskInterframeCodecInterface()
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


def VQXYZ1MaskZstdDecoder(
    payload_device: Optional[str] = None,
) -> InterframeDecoder:
    """Create a decoder for VQ + XYZ quantization with merged mask + Zstd compressed data."""
    combined_interface = VQXYZQuantMergeMaskInterframeCodecInterface()
    deserializer = ZstdDeserializer()
    return InterframeDecoder(
        deserializer=deserializer,
        interface=combined_interface,
        payload_device=payload_device,
    )


@dataclass
class VQXYZ1MaskZstdEncoderConfig:
    """Configuration for VQ + XYZ (merged mask) + Zstd encoder."""
    vq: VQInterframeCodecConfig = field(default_factory=VQInterframeCodecConfig)
    xyz: XYZQuantInterframeCodecConfig = field(default_factory=XYZQuantInterframeCodecConfig)
    zstd_level: int = 7
    payload_device: Optional[str] = None


@dataclass
class VQXYZ1MaskZstdDecoderConfig:
    """Configuration for VQ + XYZ (merged mask) + Zstd decoder."""
    payload_device: Optional[str] = None


def build_vqxyz1maskzstd_encoder(config: VQXYZ1MaskZstdEncoderConfig) -> InterframeEncoder:
    """Build encoder from configuration."""
    return VQXYZ1MaskZstdEncoder(
        vq_config=config.vq,
        xyz_config=config.xyz,
        zstd_level=config.zstd_level,
        payload_device=config.payload_device,
    )


def build_vqxyz1maskzstd_decoder(config: VQXYZ1MaskZstdDecoderConfig) -> InterframeDecoder:
    """Build decoder from configuration."""
    return VQXYZ1MaskZstdDecoder(
        payload_device=config.payload_device,
    )


# Register
register_encoder(
    "vqxyz1maskzstd",
    build_vqxyz1maskzstd_encoder,
    VQXYZ1MaskZstdEncoderConfig,
    "VQ + XYZ quantization (merged mask) + Zstd compression",
)

register_decoder(
    "vqxyz1maskzstd",
    build_vqxyz1maskzstd_decoder,
    VQXYZ1MaskZstdDecoderConfig,
    "VQ + XYZ quantization (merged mask) + Zstd decompression",
)
