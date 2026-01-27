"""
Command-line interface for gsvvcompressor.

This module provides encode and decode subcommands for each registered
encoder/decoder combination.

Usage:
    python -m gsvvcompressor encode <codec_name> [options]
    python -m gsvvcompressor decode <codec_name> [options]

Example:
    python -m gsvvcompressor encode vqxyzzstd \
        input.first_frame_path=data/frame_0000.ply \
        input.subsequent_format="data/frame_{:04d}.ply" \
        output.path=compressed.bin

    python -m gsvvcompressor decode vqxyzzstd \
        input.path=compressed.bin \
        output.first_frame_path=decoded/frame_0000.ply \
        output.subsequent_format="decoded/frame_{:04d}.ply"

Hydra options:
    --help              Show configuration schema
    --cfg job           Show resolved configuration
    --info config       Show config search path
"""

import logging
import sys
from dataclasses import field, make_dataclass
from typing import Iterable, Iterator, List, Type, TypeVar

import hydra
from hydra.core.config_store import ConfigStore
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig

from .combinations import ENCODERS, DECODERS
from .io import (
    FrameReaderConfig,
    FrameWriterConfig,
    BytesReaderConfig,
    BytesWriterConfig,
    build_frame_reader,
    build_frame_writer,
    build_bytes_reader,
    build_bytes_writer,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Progress Logging Utilities
# =============================================================================

def iter_with_progress(iterable: Iterator, desc: str) -> Iterator:
    """Wrap an iterable to log progress for each item."""
    for i, item in enumerate(iterable):
        logger.info(f"{desc} {i}")
        yield item
    logger.info(f"{desc} finished")


def iter_with_size_logging(iterable: Iterator[bytes], desc: str) -> Iterator[bytes]:
    """Wrap a bytes iterable to log the size of each item."""
    total_size = 0
    for i, item in enumerate(iterable):
        size = len(item)
        total_size += size
        logger.info(f"{desc} {i}: {size} bytes (total: {total_size} bytes)")
        yield item
    logger.info(f"{desc} finished, total size: {total_size} bytes")


# =============================================================================
# Dynamic Config Generation
# =============================================================================

def make_encode_config(codec_config_cls: Type) -> Type:
    """Dynamically create an encode config dataclass for a given codec."""
    return make_dataclass(
        f"Encode{codec_config_cls.__name__}",
        [
            ("input", FrameReaderConfig, field(default_factory=FrameReaderConfig)),
            ("output", BytesWriterConfig, field(default_factory=BytesWriterConfig)),
            ("codec", codec_config_cls, field(default_factory=codec_config_cls)),
        ],
    )


def make_decode_config(codec_config_cls: Type) -> Type:
    """Dynamically create a decode config dataclass for a given codec."""
    return make_dataclass(
        f"Decode{codec_config_cls.__name__}",
        [
            ("input", BytesReaderConfig, field(default_factory=BytesReaderConfig)),
            ("output", FrameWriterConfig, field(default_factory=FrameWriterConfig)),
            ("codec", codec_config_cls, field(default_factory=codec_config_cls)),
        ],
    )


# =============================================================================
# Encode/Decode Functions
# =============================================================================

def do_encode(cfg: DictConfig, codec_name: str) -> None:
    """Execute the encoding process with resolved config."""
    encoder_entry = ENCODERS[codec_name]

    # Build components from config
    frame_reader = build_frame_reader(cfg.input)
    bytes_writer = build_bytes_writer(cfg.output)

    # Build encoder from codec config
    encoder = encoder_entry.factory(cfg.codec)

    logger.info(f"Encoding with {codec_name}...")
    logger.info(f"  Input: {cfg.input.first_frame_path}")
    logger.info(f"  Output: {cfg.output.path}")

    # Read frames and encode
    frame_stream = frame_reader.read()
    encoded_stream = iter_with_progress(encoder.encode_stream(frame_stream), "Encoding frame")

    # Write encoded bytes with size logging
    bytes_writer.write(iter_with_size_logging(encoded_stream, "Writing chunk"))

    logger.info("Encoding complete!")


def do_decode(cfg: DictConfig, codec_name: str) -> None:
    """Execute the decoding process with resolved config."""
    decoder_entry = DECODERS[codec_name]

    # Build components from config
    bytes_reader = build_bytes_reader(cfg.input)
    frame_writer = build_frame_writer(cfg.output)

    # Build decoder from codec config
    decoder = decoder_entry.factory(cfg.codec)

    logger.info(f"Decoding with {codec_name}...")
    logger.info(f"  Input: {cfg.input.path}")
    logger.info(f"  Output: {cfg.output.first_frame_path}")

    # Read bytes and decode
    bytes_stream = bytes_reader.read()
    decoded_stream = iter_with_progress(decoder.decode_stream(bytes_stream), "Decoding frame")

    # Write decoded frames
    frame_writer.write(decoded_stream)

    logger.info("Decoding complete!")


def run_encode(codec_name: str, hydra_args: List[str]) -> None:
    """Run encoding with Hydra configuration."""
    encoder_entry = ENCODERS[codec_name]
    config_cls = make_encode_config(encoder_entry.config_class)

    # Clear and register config
    GlobalHydra.instance().clear()
    cs = ConfigStore.instance()
    cs.store(name="config", node=config_cls)

    @hydra.main(version_base=None, config_path=None, config_name="config")
    def _main(cfg: DictConfig) -> None:
        do_encode(cfg, codec_name)

    sys.argv = [sys.argv[0]] + hydra_args
    _main()


def run_decode(codec_name: str, hydra_args: List[str]) -> None:
    """Run decoding with Hydra configuration."""
    decoder_entry = DECODERS[codec_name]
    config_cls = make_decode_config(decoder_entry.config_class)

    # Clear and register config
    GlobalHydra.instance().clear()
    cs = ConfigStore.instance()
    cs.store(name="config", node=config_cls)

    @hydra.main(version_base=None, config_path=None, config_name="config")
    def _main(cfg: DictConfig) -> None:
        do_decode(cfg, codec_name)

    sys.argv = [sys.argv[0]] + hydra_args
    _main()


# =============================================================================
# Main Entry Point
# =============================================================================

def main() -> None:
    """Main entry point for the CLI."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    if len(sys.argv) < 2 or sys.argv[1] in ("--help", "-h", "help"):
        print("Usage: python -m gsvvcompressor <encode|decode> <codec> [options]")
        print(f"  Encoders: {list(ENCODERS.keys())}")
        print(f"  Decoders: {list(DECODERS.keys())}")
        print("Use --help after codec name for configuration options.")
        sys.exit(0 if len(sys.argv) >= 2 else 1)

    command = sys.argv[1]
    if command not in ("encode", "decode"):
        sys.exit(f"Error: Unknown command '{command}'. Use 'encode' or 'decode'.")

    if len(sys.argv) < 3:
        sys.exit(f"Error: Please specify a codec for {command}. Available: {list(ENCODERS.keys() if command == 'encode' else DECODERS.keys())}")

    codec_name = sys.argv[2]
    registry = ENCODERS if command == "encode" else DECODERS
    if codec_name not in registry:
        sys.exit(f"Error: Unknown {command}r '{codec_name}'. Available: {list(registry.keys())}")

    # Remaining args go to Hydra (--help, --cfg, overrides, etc.)
    hydra_args = sys.argv[3:]

    if command == "encode":
        run_encode(codec_name, hydra_args)
    else:
        run_decode(codec_name, hydra_args)


if __name__ == "__main__":
    main()
