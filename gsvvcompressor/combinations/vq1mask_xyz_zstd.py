"""
VQ (single merged mask) + XYZ quantization + Zstd compression encoder/decoder combination.
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
from ..vq.interface import VQInterframeCodecConfig
from ..vq.singlemask import VQMergeMaskInterframeCodecInterface
from ..xyz.interface import XYZQuantInterframeCodecInterface, XYZQuantInterframeCodecConfig
from .vq_xyz_zstd import VQXYZZstdEncoderConfig, VQXYZZstdDecoderConfig
from .registry import register_encoder, register_decoder


def VQ1MaskXYZZstdEncoder(
    vq_config: VQInterframeCodecConfig,
    xyz_config: XYZQuantInterframeCodecConfig,
    zstd_level: int = 7,
    payload_device: Optional[str] = None,
) -> InterframeEncoder:
    """Create an encoder combining VQ (single merged mask) + XYZ quantization + Zstd compression."""
    vq_interface = VQMergeMaskInterframeCodecInterface()
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


def VQ1MaskXYZZstdDecoder(
    payload_device: Optional[str] = None,
) -> InterframeDecoder:
    """Create a decoder for VQ (single merged mask) + XYZ quantization + Zstd compressed data."""
    vq_interface = VQMergeMaskInterframeCodecInterface()
    xyz_interface = XYZQuantInterframeCodecInterface()
    combined_interface = CombinedInterframeCodecInterface([xyz_interface, vq_interface])
    deserializer = ZstdDeserializer()
    return InterframeDecoder(
        deserializer=deserializer,
        interface=combined_interface,
        payload_device=payload_device,
    )


def build_vq1maskxyzzstd_encoder(config: VQXYZZstdEncoderConfig) -> InterframeEncoder:
    """Build encoder from configuration."""
    return VQ1MaskXYZZstdEncoder(
        vq_config=config.vq,
        xyz_config=config.xyz,
        zstd_level=config.zstd_level,
        payload_device=config.payload_device,
    )


def build_vq1maskxyzzstd_decoder(config: VQXYZZstdDecoderConfig) -> InterframeDecoder:
    """Build decoder from configuration."""
    return VQ1MaskXYZZstdDecoder(
        payload_device=config.payload_device,
    )


# Register
register_encoder(
    "vq1maskxyzzstd",
    build_vq1maskxyzzstd_encoder,
    VQXYZZstdEncoderConfig,
    "VQ (single merged mask) + XYZ quantization + Zstd compression",
)

register_decoder(
    "vq1maskxyzzstd",
    build_vq1maskxyzzstd_decoder,
    VQXYZZstdDecoderConfig,
    "VQ (single merged mask) + XYZ quantization + Zstd decompression",
)
