from dataclasses import dataclass

from omegaconf import MISSING
from .gaussian_model import FrameReader, FrameWriter


@dataclass
class FrameReaderConfig:
    """Configuration for reading GaussianModel frame sequences."""
    first_frame_path: str = MISSING
    subsequent_format: str = MISSING
    start_index: int = 1
    sh_degree: int = 3


@dataclass
class FrameWriterConfig:
    """Configuration for writing GaussianModel frame sequences."""
    first_frame_path: str = MISSING
    subsequent_format: str = MISSING
    start_index: int = 1


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
