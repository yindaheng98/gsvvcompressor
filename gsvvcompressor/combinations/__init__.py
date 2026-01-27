"""
Combination modules for composing multiple codec components.
"""

from .registry import (
    ENCODERS,
    DECODERS,
    register_encoder,
    register_decoder,
    EncoderEntry,
    DecoderEntry,
)

# Import to trigger registration
from . import vq_xyz_zstd

from .vq_xyz_zstd import (
    VQXYZZstdEncoderConfig,
    VQXYZZstdDecoderConfig,
    VQXYZZstdEncoder,
    VQXYZZstdDecoder,
    build_vqxyzzstd_encoder,
    build_vqxyzzstd_decoder,
)

__all__ = [
    "ENCODERS",
    "DECODERS",
    "register_encoder",
    "register_decoder",
    "EncoderEntry",
    "DecoderEntry",
    "VQXYZZstdEncoderConfig",
    "VQXYZZstdDecoderConfig",
    "VQXYZZstdEncoder",
    "VQXYZZstdDecoder",
    "build_vqxyzzstd_encoder",
    "build_vqxyzzstd_decoder",
]
