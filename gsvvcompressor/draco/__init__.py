"""
Draco-based compression for reduced 3DGS data.

This module provides:
- DracoPayload: Payload structure for Draco-compatible data
- DracoCapableInterframeCodecInterface: Interface for codecs that output DracoPayload
- DracoSerializer/DracoDeserializer: Streaming serialization using Draco encoding
"""

from .interface import DracoCapableInterframeCodecInterface, DracoPayload
from .serialize import DracoDeserializer, DracoSerializer

__all__ = [
    "DracoPayload",
    "DracoCapableInterframeCodecInterface",
    "DracoSerializer",
    "DracoDeserializer",
]