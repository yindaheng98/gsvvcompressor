"""
IO module for reading and writing GaussianModel sequences and bytes data.

This module provides classes for:
- GaussianModelSequenceReader: Read GaussianModel frames from a sequence of files
- GaussianModelSequenceWriter: Write GaussianModel frames to a sequence of files
- BytesReader: Read bytes data from a file
- BytesWriter: Write bytes data to a file
"""

from .gaussian_model import GaussianModelSequenceReader, GaussianModelSequenceWriter
from .bytes import BytesReader, BytesWriter

__all__ = [
    "GaussianModelSequenceReader",
    "GaussianModelSequenceWriter",
    "BytesReader",
    "BytesWriter",
]
