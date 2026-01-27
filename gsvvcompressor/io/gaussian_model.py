"""
GaussianModel sequence reader and writer classes.

This module provides classes for reading and writing GaussianModel frames
from/to a sequence of files.
"""

import os
from typing import Iterator

from gaussian_splatting import GaussianModel


class GaussianModelSequenceReader:
    """
    Reader for GaussianModel sequences from a series of files.

    This class reads GaussianModel frames from a sequence of files, where the
    first frame has a specific path and subsequent frames follow a format string
    pattern.

    Example:
        reader = GaussianModelSequenceReader(
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
        """
        Initialize the GaussianModel sequence reader.

        Args:
            first_frame_path: Path to the first frame file.
            subsequent_format: Format string for subsequent frame paths.
                Should contain a format placeholder for the frame index,
                e.g., "data/frame_{:04d}.ply".
            start_index: The starting frame index for subsequent frames.
        """
        self._first_frame_path = first_frame_path
        self._subsequent_format = subsequent_format
        self._start_index = start_index
        self._sh_degree = sh_degree

    def read(self) -> Iterator[GaussianModel]:
        """
        Read GaussianModel frames from the file sequence.

        Yields:
            GaussianModel instances loaded from each file in the sequence.
            Stops when a file does not exist.

        Raises:
            FileNotFoundError: If the first frame file does not exist.
        """
        # Check and yield first frame
        if not os.path.exists(self._first_frame_path):
            raise FileNotFoundError(
                f"First frame file not found: {self._first_frame_path}"
            )

        model = GaussianModel(sh_degree=self._sh_degree)
        model.load_ply(self._first_frame_path)
        yield model

        # Yield subsequent frames
        index = self._start_index
        while True:
            path = self._subsequent_format.format(index)
            if not os.path.exists(path):
                break

            model = GaussianModel(sh_degree=self._sh_degree)
            model.load_ply(path)
            yield model
            index += 1


class GaussianModelSequenceWriter:
    """
    Writer for GaussianModel sequences to a series of files.

    This class writes GaussianModel frames to a sequence of files, where the
    first frame has a specific path and subsequent frames follow a format string
    pattern.

    Example:
        writer = GaussianModelSequenceWriter(
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
        """
        Initialize the GaussianModel sequence writer.

        Args:
            first_frame_path: Path to write the first frame file.
            subsequent_format: Format string for subsequent frame paths.
                Should contain a format placeholder for the frame index,
                e.g., "output/frame_{:04d}.ply".
            start_index: The starting frame index for subsequent frames.
        """
        self._first_frame_path = first_frame_path
        self._subsequent_format = subsequent_format
        self._start_index = start_index

    def write(self, frames: Iterator[GaussianModel]) -> None:
        """
        Write GaussianModel frames to the file sequence.

        Args:
            frames: An iterator yielding GaussianModel instances to write.

        Raises:
            FileExistsError: If any target file already exists.
        """
        is_first = True
        index = self._start_index

        for frame in frames:
            if is_first:
                path = self._first_frame_path
                is_first = False
            else:
                path = self._subsequent_format.format(index)
                index += 1

            # Check if file exists before writing
            if os.path.exists(path):
                raise FileExistsError(f"File already exists: {path}")

            # Create parent directory if needed
            parent_dir = os.path.dirname(path)
            if parent_dir and not os.path.exists(parent_dir):
                os.makedirs(parent_dir, exist_ok=True)

            frame.save_ply(path)
