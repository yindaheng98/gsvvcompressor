"""
Combination modules for composing multiple codec components.
"""

from .vq_xyz_zstd import create_vq_xyz_zstd_encoder, create_vq_xyz_zstd_decoder

# Registry of encoder factory functions
# Key: combination name, Value: encoder factory function
ENCODERS = {
    "vq_xyz_zstd": create_vq_xyz_zstd_encoder,
}

# Registry of decoder factory functions
# Key: combination name, Value: decoder factory function
DECODERS = {
    "vq_xyz_zstd": create_vq_xyz_zstd_decoder,
}

__all__ = [
    "create_vq_xyz_zstd_encoder",
    "create_vq_xyz_zstd_decoder",
    "ENCODERS",
    "DECODERS",
]
