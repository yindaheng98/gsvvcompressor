from dataclasses import dataclass

from omegaconf import MISSING
from .gaussian_model import FrameReader, FrameWriter
from .bytes import BytesReader, BytesWriter


@dataclass
class FrameReaderConfig:
    """Configuration for reading GaussianModel frame sequences."""
    first_frame_path: str = MISSING
    subsequent_format: str = MISSING
    start_index: int = 2
    sh_degree: int = 3


@dataclass
class FrameWriterConfig:
    """Configuration for writing GaussianModel frame sequences."""
    first_frame_path: str = MISSING
    subsequent_format: str = MISSING
    start_index: int = 2


def build_frame_reader(config: FrameReaderConfig) -> FrameReader:
    """Build a FrameReader from configuration."""
    return FrameReader(
        first_frame_path=config.first_frame_path,
        subsequent_format=config.subsequent_format,
        start_index=config.start_index,
        sh_degree=config.sh_degree,
    )


def build_frame_writer(config: FrameWriterConfig) -> FrameWriter:
    """Build a FrameWriter from configuration."""
    return FrameWriter(
        first_frame_path=config.first_frame_path,
        subsequent_format=config.subsequent_format,
        start_index=config.start_index,
    )


@dataclass
class BytesReaderConfig:
    """Configuration for reading bytes data from a file."""
    path: str = MISSING
    chunk_size: int = BytesReader.DEFAULT_CHUNK_SIZE


@dataclass
class BytesWriterConfig:
    """Configuration for writing bytes data to a file."""
    path: str = MISSING


def build_bytes_reader(config: BytesReaderConfig) -> BytesReader:
    """Build a BytesReader from configuration."""
    return BytesReader(
        path=config.path,
        chunk_size=config.chunk_size,
    )


def build_bytes_writer(config: BytesWriterConfig) -> BytesWriter:
    """Build a BytesWriter from configuration."""
    return BytesWriter(
        path=config.path,
    )
