"""
Draco-based compression for reduced 3DGS data.

This module provides:
- DracoPayload: Payload structure for Draco-compatible data
- TwoPassDracoPayload: DracoPayload with num_points field for two-pass serialization
- DracoInterframeCodecTranscodingInterface: Abstract interface for payload transcoding
- DracoInterframeCodecInterface: Wrapper codec that transcodes to/from DracoPayload
- DracoSerializer/DracoDeserializer: Streaming serialization using Draco encoding
- TwoPassDracoSerializer/TwoPassDracoDeserializer: Two-pass serialization for batched compression
"""

from .interface import (
    DracoInterframeCodecInterface,
    DracoInterframeCodecTranscodingInterface,
    DracoPayload,
)
from .serialize import DracoDeserializer, DracoSerializer
from .twopass import (
    TwoPassDracoDeserializer,
    TwoPassDracoPayload,
    TwoPassDracoSerializer,
)

__all__ = [
    "DracoPayload",
    "TwoPassDracoPayload",
    "DracoInterframeCodecTranscodingInterface",
    "DracoInterframeCodecInterface",
    "DracoSerializer",
    "DracoDeserializer",
    "TwoPassDracoSerializer",
    "TwoPassDracoDeserializer",
]
