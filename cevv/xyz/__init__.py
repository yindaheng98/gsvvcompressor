"""
XYZ coordinate processing utilities for volumetric video compression.

This module provides functions for adaptive quantization of 3D point coordinates
based on local point cloud density.
"""

from .quant import (
    XYZQuantConfig,
    compute_quant_config,
    quantize_xyz,
    dequantize_xyz,
    estimate_quantization_error,
)

__all__ = [
    'XYZQuantConfig',
    'compute_quant_config',
    'quantize_xyz',
    'dequantize_xyz',
    'estimate_quantization_error',
]
