"""
VQ + XYZ quantization with single merged mask + Zstd compression encoder/decoder combination.

This module provides a combined codec where VQ and XYZ share a single merged mask
for interframe encoding, improving compression efficiency.
"""

from dataclasses import dataclass, field
from typing import Optional

from ..compress.zstd import ZstdSerializer, ZstdDeserializer
from ..interframe.combine import (
    CombinedInterframeCodecInterface,
    CombinedInterframeCodecContext,
    CombinedInterframeEncoderInitConfig,
    CombinedPayload,
)
from ..interframe.encoder import InterframeEncoder
from ..interframe.decoder import InterframeDecoder
from ..vq.interface import (
    VQInterframeCodecConfig,
    VQInterframeCodecContext,
)
from ..vq.singlemask import VQMergeMaskInterframeCodecInterface, VQMergeMaskInterframePayload
from ..xyz.interface import (
    XYZQuantInterframeCodecInterface,
    XYZQuantInterframeCodecConfig,
    XYZQuantInterframeCodecContext,
    XYZQuantInterframePayload,
)
from .registry import register_encoder, register_decoder


class VQXYZQuantMergeMaskInterframeCodecInterface(CombinedInterframeCodecInterface):
    """
    Combined VQ + XYZ codec with single merged mask for interframe encoding.

    This interface extends CombinedInterframeCodecInterface but uses a single merged mask
    (OR of XYZ mask and all VQ attribute masks) for interframe payloads. Both XYZ and VQ
    payloads use the same mask, improving compression efficiency.

    Assumes interfaces order: [xyz_interface, vq_interface]
    """

    def __init__(self):
        super().__init__([xyz_interface, vq_interface])
        xyz_interface = XYZQuantInterframeCodecInterface()
        vq_interface = VQMergeMaskInterframeCodecInterface()
        self.xyz_interface = xyz_interface
        self.vq_interface = vq_interface

    def encode_interframe(
        self,
        prev_context: CombinedInterframeCodecContext,
        next_context: CombinedInterframeCodecContext,
    ) -> CombinedPayload:
        """
        Encode the difference between two consecutive frames.

        Calls super to get individual payloads, then merges masks (OR) and
        re-extracts data using the merged mask.

        Args:
            prev_context: The context of the previous frame.
            next_context: The context of the next frame.

        Returns:
            A CombinedPayload with XYZ and VQ payloads using the same merged mask.
        """
        # Get original payloads from parent
        original_payload: CombinedPayload = super().encode_interframe(prev_context, next_context)
        xyz_original: XYZQuantInterframePayload = original_payload.payloads[0]
        vq_original: VQMergeMaskInterframePayload = original_payload.payloads[1]

        # Merge masks: XYZ mask OR VQ mask
        merged_mask = xyz_original.xyz_mask | vq_original.ids_mask

        # Get next context data
        xyz_next_context: XYZQuantInterframeCodecContext = next_context.contexts[0]
        vq_next_context: VQInterframeCodecContext = next_context.contexts[1]

        # Create XYZ payload with merged mask
        xyz_payload = XYZQuantInterframePayload(
            xyz_mask=merged_mask,
            quantized_xyz=xyz_next_context.quantized_xyz[merged_mask],
        )

        # Create VQ payload with merged mask
        changed_ids_dict = {}
        for key, next_ids in vq_next_context.ids_dict.items():
            changed_ids_dict[key] = next_ids[merged_mask]

        vq_payload = VQMergeMaskInterframePayload(
            ids_mask=merged_mask,
            ids_dict=changed_ids_dict,
        )

        return CombinedPayload(payloads=[xyz_payload, vq_payload])


def VQ1MaskXYZZstdEncoder(
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


def VQ1MaskXYZZstdDecoder(
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
class VQ1MaskXYZZstdEncoderConfig:
    """Configuration for VQ + XYZ (merged mask) + Zstd encoder."""
    vq: VQInterframeCodecConfig = field(default_factory=VQInterframeCodecConfig)
    xyz: XYZQuantInterframeCodecConfig = field(default_factory=XYZQuantInterframeCodecConfig)
    zstd_level: int = 7
    payload_device: Optional[str] = None


@dataclass
class VQ1MaskXYZZstdDecoderConfig:
    """Configuration for VQ + XYZ (merged mask) + Zstd decoder."""
    payload_device: Optional[str] = None


def build_vq1maskxyzzstd_encoder(config: VQ1MaskXYZZstdEncoderConfig) -> InterframeEncoder:
    """Build encoder from configuration."""
    return VQ1MaskXYZZstdEncoder(
        vq_config=config.vq,
        xyz_config=config.xyz,
        zstd_level=config.zstd_level,
        payload_device=config.payload_device,
    )


def build_vq1maskxyzzstd_decoder(config: VQ1MaskXYZZstdDecoderConfig) -> InterframeDecoder:
    """Build decoder from configuration."""
    return VQ1MaskXYZZstdDecoder(
        payload_device=config.payload_device,
    )


# Register
register_encoder(
    "vq1maskxyzzstd",
    build_vq1maskxyzzstd_encoder,
    VQ1MaskXYZZstdEncoderConfig,
    "VQ + XYZ quantization (merged mask) + Zstd compression",
)

register_decoder(
    "vq1maskxyzzstd",
    build_vq1maskxyzzstd_decoder,
    VQ1MaskXYZZstdDecoderConfig,
    "VQ + XYZ quantization (merged mask) + Zstd decompression",
)
