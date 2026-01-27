"""Zstandard-based streaming serialization and deserialization."""

from .zstd import ZstdDeserializer, ZstdSerializer

__all__ = ["ZstdSerializer", "ZstdDeserializer"]
