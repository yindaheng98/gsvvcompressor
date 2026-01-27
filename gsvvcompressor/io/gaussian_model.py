"""
Frame reader and writer classes for GaussianModel sequences.
"""

import os
from dataclasses import dataclass
from typing import Iterator, Optional

from gaussian_splatting import GaussianModel
from omegaconf import MISSING


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


class FrameReader:
    """
    Reader for GaussianModel frames from a series of files.

    Example:
        reader = FrameReader(
            first_frame_path="data/frame_0000.ply",
            subsequent_format="data/frame_{:04d}.ply",
            start_index=1
        )
        for frame in reader.read():
            process(frame)
    """

    def __init__(
        self,
        first_frame_path: str,
        subsequent_format: str,
        start_index: int,
        sh_degree: int = 3,
    ):
        self._first_frame_path = first_frame_path
        self._subsequent_format = subsequent_format
        self._start_index = start_index
        self._sh_degree = sh_degree

    def read(self) -> Iterator[GaussianModel]:
        """Read GaussianModel frames from the file sequence."""
        if not os.path.exists(self._first_frame_path):
            raise FileNotFoundError(
                f"First frame file not found: {self._first_frame_path}"
            )

        model = GaussianModel(sh_degree=self._sh_degree)
        model.load_ply(self._first_frame_path)
        yield model

        index = self._start_index
        while True:
            path = self._subsequent_format.format(index)
            if not os.path.exists(path):
                break
            model = GaussianModel(sh_degree=self._sh_degree)
            model.load_ply(path)
            yield model
            index += 1


class FrameWriter:
    """
    Writer for GaussianModel frames to a series of files.

    Example:
        writer = FrameWriter(
            first_frame_path="output/frame_0000.ply",
            subsequent_format="output/frame_{:04d}.ply",
            start_index=1
        )
        writer.write(frame_iterator)
    """

    def __init__(
        self,
        first_frame_path: str,
        subsequent_format: str,
        start_index: int,
    ):
        self._first_frame_path = first_frame_path
        self._subsequent_format = subsequent_format
        self._start_index = start_index

    def write(self, frames: Iterator[GaussianModel]) -> None:
        """Write GaussianModel frames to the file sequence."""
        is_first = True
        index = self._start_index

        for frame in frames:
            if is_first:
                path = self._first_frame_path
                is_first = False
            else:
                path = self._subsequent_format.format(index)
                index += 1

            if os.path.exists(path):
                raise FileExistsError(f"File already exists: {path}")

            parent_dir = os.path.dirname(path)
            if parent_dir and not os.path.exists(parent_dir):
                os.makedirs(parent_dir, exist_ok=True)

            frame.save_ply(path)


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
