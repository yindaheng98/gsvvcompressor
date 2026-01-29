"""
Draco-based compression for reduced 3DGS data.

This module provides:
- DracoPayload: Payload structure for Draco-compatible data
- DracoInterframeCodecTranscodingInterface: Abstract interface for payload transcoding
- DracoInterframeCodecInterface: Wrapper codec that transcodes to/from DracoPayload
- DracoSerializer/DracoDeserializer: Streaming serialization using Draco encoding
"""

from .interface import (
    DracoInterframeCodecInterface,
    DracoInterframeCodecTranscodingInterface,
    DracoPayload,
)
from .serialize import DracoDeserializer, DracoSerializer

__all__ = [
    "DracoPayload",
    "DracoInterframeCodecTranscodingInterface",
    "DracoInterframeCodecInterface",
    "DracoSerializer",
    "DracoDeserializer",
]
