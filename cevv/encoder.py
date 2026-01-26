from abc import ABC, abstractmethod
from typing import Optional

from gaussian_splatting import GaussianModel


class Encoder(ABC):
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

    def encode_frames(self, frames: list[GaussianModel]) -> bytes:
        """
        Encode a sequence of GaussianModel frames.

        This method calls `encode_frame` for each frame in the input list,
        concatenates all the encoded bytes, and continues calling `encode_frame`
        with None until the encoder signals completion.

        Args:
            frames: A list of GaussianModel instances to encode.

        Returns:
            The concatenated encoded bytes for the entire sequence.
        """
        result = bytearray()
        is_finished = False

        # Encode each frame in the sequence
        for frame in frames:
            encoded_bytes, is_finished = self.encode_frame(frame)
            result.extend(encoded_bytes)

        # Flush remaining data until encoder signals completion
        while not is_finished:
            encoded_bytes, is_finished = self.encode_frame(None)
            result.extend(encoded_bytes)

        return bytes(result)
