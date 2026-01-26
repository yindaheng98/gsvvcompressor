from abc import ABC, abstractmethod
from typing import Iterator, Optional

from gaussian_splatting import GaussianModel


class AbstractEncoder(ABC):
    """
    Abstract base class for compression algorithms.

    Subclasses must implement the `encode_frame` method to define
    the specific compression logic for GaussianModel sequences.
    """

    @abstractmethod
    def encode_frame(self, frame: Optional[GaussianModel]) -> tuple[bytes, bool]:
        """
        Encode a single frame of GaussianModel.

        Args:
            frame: A GaussianModel instance to encode, or None to flush
                   remaining encoded data from the internal buffer.

        Returns:
            A tuple of (encoded_bytes, is_finished):
            - encoded_bytes: The encoded byte data (can be empty, i.e., length 0).
            - is_finished: True if all previously input GaussianModels have been
                           fully encoded; False if more data is pending.
        """
        pass

    def encode_stream(self, stream: Iterator[GaussianModel]) -> Iterator[bytes]:
        """
        Encode a stream of GaussianModel frames.

        This method calls `encode_frame` for each frame from the input iterator,
        yields encoded bytes as they become available, and continues calling
        `encode_frame` with None until the encoder signals completion.

        Args:
            stream: An iterator that yields GaussianModel instances to encode.

        Yields:
            Encoded bytes for each processed frame or flush operation.
        """
        is_finished = False

        # Encode each frame from the stream
        for frame in stream:
            encoded_bytes, is_finished = self.encode_frame(frame)
            if encoded_bytes:
                yield encoded_bytes

        # Flush remaining data until encoder signals completion
        while not is_finished:
            encoded_bytes, is_finished = self.encode_frame(None)
            if encoded_bytes:
                yield encoded_bytes
