from abc import ABC, abstractmethod
from typing import Iterator, List

from gaussian_splatting import GaussianModel


class AbstractDecoder(ABC):
    """
    Abstract base class for decompression algorithms.

    Subclasses must implement the `decode_frame` method to define
    the specific decompression logic for byte sequences into GaussianModel.
    """

    @abstractmethod
    def decode_frame(self, data: bytes) -> tuple[List[GaussianModel], bool]:
        """
        Decode a single chunk of compressed data.

        Args:
            data: A bytes object containing compressed data to decode.
                  If the length is 0, it signals the end of the input stream,
                  prompting the decoder to flush remaining buffered data.

        Returns:
            A tuple of (gaussian_models, is_finished):
            - gaussian_models: A list of decoded GaussianModel instances.
                               Can be empty if no complete frames are available yet.
            - is_finished: True if all previously input bytes have been fully
                           decoded; False if more data is pending in the buffer.
        """
        pass

    def decode_stream(self, stream: Iterator[bytes]) -> Iterator[GaussianModel]:
        """
        Decode a stream of compressed byte chunks into GaussianModel frames.

        This method calls `decode_frame` for each chunk from the input iterator,
        yields decoded GaussianModel instances as they become available, and
        continues calling `decode_frame` with empty bytes until the decoder
        signals completion.

        Args:
            stream: An iterator that yields bytes objects to decode.

        Yields:
            Decoded GaussianModel instances for each processed chunk or flush operation.
        """
        is_finished = False

        # Decode each chunk from the stream
        for data in stream:
            gaussian_models, is_finished = self.decode_frame(data)
            yield from gaussian_models

        # Flush remaining data until decoder signals completion
        while not is_finished:
            gaussian_models, is_finished = self.decode_frame(b"")
            yield from gaussian_models
