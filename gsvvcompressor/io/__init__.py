"""
IO module for reading and writing GaussianModel frames and bytes data.
"""

from .gaussian_model import FrameReader, FrameWriter
from .bytes import BytesReader, BytesWriter
from .config import (
    FrameReaderConfig,
    FrameWriterConfig,
    build_frame_reader,
    build_frame_writer,
)

__all__ = [
    "FrameReader",
    "FrameWriter",
    "FrameReaderConfig",
    "FrameWriterConfig",
    "build_frame_reader",
    "build_frame_writer",
    "BytesReader",
    "BytesWriter",
]
