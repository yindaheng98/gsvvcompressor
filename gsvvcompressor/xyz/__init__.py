"""
XYZ coordinate processing utilities for volumetric video compression.

This module provides functions for adaptive quantization of 3D point coordinates
based on local point cloud density.
"""

from .quant import XYZQuantConfig
from .interface import (
    XYZQuantInterframeCodecConfig,
    XYZQuantInterframeCodecContext,
    XYZQuantKeyframePayload,
    XYZQuantInterframePayload,
    XYZQuantInterframeCodecInterface,
)

__all__ = [
    # quant.py
    'XYZQuantConfig',
    # interface.py
    'XYZQuantInterframeCodecConfig',
    'XYZQuantInterframeCodecContext',
    'XYZQuantKeyframePayload',
    'XYZQuantInterframePayload',
    'XYZQuantInterframeCodecInterface',
]
