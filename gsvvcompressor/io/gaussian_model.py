"""
Frame reader and writer classes for GaussianModel sequences.
"""

import os
from typing import Iterator

from gaussian_splatting import GaussianModel


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
        max_frames: int | None = None,
    ):
        self._first_frame_path = first_frame_path
        self._subsequent_format = subsequent_format
        self._start_index = start_index
        self._sh_degree = sh_degree
        self._max_frames = max_frames

    def read(self) -> Iterator[GaussianModel]:
        """Read GaussianModel frames from the file sequence."""
        model = GaussianModel(sh_degree=self._sh_degree)
        model.load_ply(self._first_frame_path)
        yield model

        frame_count = 1
        if self._max_frames is not None and frame_count >= self._max_frames:
            return

        index = self._start_index
        while True:
            path = self._subsequent_format.format(index)
            model = GaussianModel(sh_degree=self._sh_degree)
            model.load_ply(path)
            yield model
            frame_count += 1
            if self._max_frames is not None and frame_count >= self._max_frames:
                break
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
        overwrite: bool = False,
    ):
        self._first_frame_path = first_frame_path
        self._subsequent_format = subsequent_format
        self._start_index = start_index
        self._overwrite = overwrite

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

            if not self._overwrite and os.path.exists(path):
                raise FileExistsError(f"File already exists: {path}")

            parent_dir = os.path.dirname(path)
            if parent_dir and not os.path.exists(parent_dir):
                os.makedirs(parent_dir, exist_ok=True)

            frame.save_ply(path)
