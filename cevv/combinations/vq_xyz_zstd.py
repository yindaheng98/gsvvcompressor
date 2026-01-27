"""
VQ + XYZ quantization + Zstd compression encoder/decoder combination.

This module provides factory functions to create encoders and decoders that combine:
- VQ (Vector Quantization) for attributes (rotation, opacity, scaling, features)
- XYZ quantization for coordinate compression
- Zstd compression for serialization
"""

from ..compress.zstd import ZstdSerializer, ZstdDeserializer
from ..interframe.combine import (
    CombinedInterframeCodecInterface,
    CombinedInterframeEncoderInitConfig,
)
from ..interframe.encoder import InterframeEncoder
from ..interframe.decoder import InterframeDecoder
from ..vq.interface import VQInterframeCodecInterface, VQInterframeCodecConfig
from ..xyz.interface import XYZQuantInterframeCodecInterface, XYZQuantInterframeCodecConfig


def create_vq_xyz_zstd_encoder(
    vq_config: VQInterframeCodecConfig = VQInterframeCodecConfig(),
    xyz_config: XYZQuantInterframeCodecConfig = XYZQuantInterframeCodecConfig(),
    zstd_level: int = 7,
    payload_device=None,
) -> InterframeEncoder:
    """
    Create an encoder combining VQ + XYZ quantization + Zstd compression.

    This encoder uses:
    - VQInterframeCodecInterface for vector quantization of attributes
    - XYZQuantInterframeCodecInterface for coordinate quantization
    - ZstdSerializer for compression

    Args:
        vq_config: Configuration for VQ codec. If None, uses default VQInterframeCodecConfig().
        xyz_config: Configuration for XYZ codec. If None, uses default XYZQuantInterframeCodecConfig().
        zstd_level: Zstd compression level (1-22). Default is 7.
        payload_device: The target device for encoded Payloads before
            serialization (e.g., 'cpu', 'cuda'). If None, no device
            transfer is performed.

    Returns:
        An InterframeEncoder instance configured with the combined codec.
    """
    # Create codec interfaces
    vq_interface = VQInterframeCodecInterface()
    xyz_interface = XYZQuantInterframeCodecInterface()

    # Combine interfaces (XYZ first to set coordinates, then VQ for attributes)
    combined_interface = CombinedInterframeCodecInterface([xyz_interface, vq_interface])
    combined_config = CombinedInterframeEncoderInitConfig(
        init_configs=[xyz_config, vq_config]
    )

    # Create serializer
    serializer = ZstdSerializer(level=zstd_level)

    return InterframeEncoder(
        serializer=serializer,
        interface=combined_interface,
        init_config=combined_config,
        payload_device=payload_device,
    )


def create_vq_xyz_zstd_decoder(
    payload_device=None,
) -> InterframeDecoder:
    """
    Create a decoder for VQ + XYZ quantization + Zstd compressed data.

    This decoder uses:
    - ZstdDeserializer for decompression
    - XYZQuantInterframeCodecInterface for coordinate dequantization
    - VQInterframeCodecInterface for attribute dequantization

    Args:
        zstd_level: Zstd compression level (not used for decompression, kept for API symmetry).
        payload_device: The target device for input Payloads before
            unpacking (e.g., 'cpu', 'cuda'). If None, no device
            transfer is performed.

    Returns:
        An InterframeDecoder instance configured with the combined codec.
    """
    # Create codec interfaces
    vq_interface = VQInterframeCodecInterface()
    xyz_interface = XYZQuantInterframeCodecInterface()

    # Combine interfaces (same order as encoder: XYZ first, then VQ)
    combined_interface = CombinedInterframeCodecInterface([xyz_interface, vq_interface])

    # Create deserializer
    deserializer = ZstdDeserializer()

    return InterframeDecoder(
        deserializer=deserializer,
        interface=combined_interface,
        payload_device=payload_device,
    )
