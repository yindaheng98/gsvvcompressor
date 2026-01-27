"""
Global registry for encoder and decoder combinations.
"""

from dataclasses import dataclass
from typing import Callable, Dict, Type

from ..encoder import AbstractEncoder
from ..decoder import AbstractDecoder


@dataclass
class EncoderEntry:
    """Entry in the encoder registry."""
    name: str
    factory: Callable[..., AbstractEncoder]
    config_class: Type
    description: str = ""


@dataclass
class DecoderEntry:
    """Entry in the decoder registry."""
    name: str
    factory: Callable[..., AbstractDecoder]
    config_class: Type
    description: str = ""


# Global registries
ENCODERS: Dict[str, EncoderEntry] = {}
DECODERS: Dict[str, DecoderEntry] = {}


def register_encoder(
    name: str,
    factory: Callable[..., AbstractEncoder],
    config_class: Type,
    description: str = "",
) -> None:
    """Register an encoder."""
    ENCODERS[name] = EncoderEntry(name, factory, config_class, description)


def register_decoder(
    name: str,
    factory: Callable[..., AbstractDecoder],
    config_class: Type,
    description: str = "",
) -> None:
    """Register a decoder."""
    DECODERS[name] = DecoderEntry(name, factory, config_class, description)
