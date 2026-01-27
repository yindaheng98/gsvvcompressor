"""
Bytes data reader and writer classes.

This module provides classes for reading and writing bytes data from/to files.
"""

import os
from typing import Iterator


class BytesReader:
    """
    Reader for bytes data from a file.

    This class reads bytes data from a file and yields it as an iterator.
    Each yielded item is a bytes chunk from the file.

    Example:
        reader = BytesReader("data/compressed.bin")
        for chunk in reader.read():
            process(chunk)
    """

    # Default chunk size for reading (64KB)
    DEFAULT_CHUNK_SIZE = 64 * 1024

    def __init__(self, path: str, chunk_size: int = DEFAULT_CHUNK_SIZE):
        """
        Initialize the bytes reader.

        Args:
            path: Path to the file to read.
            chunk_size: Size of each chunk to yield in bytes.
                Default is 64KB.
        """
        self._path = path
        self._chunk_size = chunk_size

    def read(self) -> Iterator[bytes]:
        """
        Read bytes data from the file.

        Yields:
            bytes chunks from the file.

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        if not os.path.exists(self._path):
            raise FileNotFoundError(f"File not found: {self._path}")

        with open(self._path, "rb") as f:
            while True:
                chunk = f.read(self._chunk_size)
                if not chunk:
                    break
                yield chunk


class BytesWriter:
    """
    Writer for bytes data to a file.

    This class writes bytes data from an iterator to a file.

    Example:
        writer = BytesWriter("output/compressed.bin")
        writer.write(bytes_iterator)
    """

    def __init__(self, path: str):
        """
        Initialize the bytes writer.

        Args:
            path: Path to the file to write.
        """
        self._path = path

    def write(self, data: Iterator[bytes]) -> None:
        """
        Write bytes data to the file.

        Args:
            data: An iterator yielding bytes chunks to write.

        Raises:
            FileExistsError: If the target file already exists.
        """
        # Check if file exists before writing
        if os.path.exists(self._path):
            raise FileExistsError(f"File already exists: {self._path}")

        # Create parent directory if needed
        parent_dir = os.path.dirname(self._path)
        if parent_dir and not os.path.exists(parent_dir):
            os.makedirs(parent_dir, exist_ok=True)

        with open(self._path, "wb") as f:
            for chunk in data:
                f.write(chunk)
