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
from . import vq_xyz_1mask_zstd
from . import vq_xyz_draco
from . import vq_xyz_draco_2pass

from .vq_xyz_zstd import (
    VQXYZZstdEncoderConfig,
    VQXYZZstdDecoderConfig,
    VQXYZZstdEncoder,
    VQXYZZstdDecoder,
    build_vqxyzzstd_encoder,
    build_vqxyzzstd_decoder,
)
from .vq_xyz_1mask_zstd import (
    VQXYZ1MaskZstdEncoder,
    VQXYZ1MaskZstdDecoder,
    build_vqxyz1maskzstd_encoder,
    build_vqxyz1maskzstd_decoder,
)
from .vq_xyz_draco import (
    VQXYZDracoEncoderConfig,
    VQXYZDracoDecoderConfig,
    VQXYZDracoEncoder,
    VQXYZDracoDecoder,
    build_vqxyzdraco_encoder,
    build_vqxyzdraco_decoder,
)
from .vq_xyz_draco_2pass import (
    VQXYZDraco2PassEncoderConfig,
    VQXYZDraco2PassDecoderConfig,
    VQXYZDraco2PassEncoder,
    VQXYZDraco2PassDecoder,
    build_vqxyzdraco2pass_encoder,
    build_vqxyzdraco2pass_decoder,
)

__all__ = [
    "ENCODERS",
    "DECODERS",
    "register_encoder",
    "register_decoder",
    "EncoderEntry",
    "DecoderEntry",
    # VQ + XYZ + Zstd
    "VQXYZZstdEncoderConfig",
    "VQXYZZstdDecoderConfig",
    "VQXYZZstdEncoder",
    "VQXYZZstdDecoder",
    "build_vqxyzzstd_encoder",
    "build_vqxyzzstd_decoder",
    # VQ (single merged mask) + XYZ + Zstd
    "VQXYZ1MaskZstdEncoder",
    "VQXYZ1MaskZstdDecoder",
    "build_vqxyz1maskzstd_encoder",
    "build_vqxyz1maskzstd_decoder",
    # VQ + XYZ + Draco
    "VQXYZDracoEncoderConfig",
    "VQXYZDracoDecoderConfig",
    "VQXYZDracoEncoder",
    "VQXYZDracoDecoder",
    "build_vqxyzdraco_encoder",
    "build_vqxyzdraco_decoder",
    # VQ + XYZ + Two-Pass Draco
    "VQXYZDraco2PassEncoderConfig",
    "VQXYZDraco2PassDecoderConfig",
    "VQXYZDraco2PassEncoder",
    "VQXYZDraco2PassDecoder",
    "build_vqxyzdraco2pass_encoder",
    "build_vqxyzdraco2pass_decoder",
]
