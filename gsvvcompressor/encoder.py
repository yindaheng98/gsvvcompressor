from abc import ABC, abstractmethod
from typing import Iterator

from gaussian_splatting import GaussianModel

from .payload import Payload
from .serializer import AbstractSerializer


class AbstractEncoder(ABC):
    """
    Abstract base class for compression algorithms.

    This encoder uses a two-stage process:
    1. Pack frames into Payload objects (via `pack`)
    2. Serialize Payloads to bytes (via the serializer)

    Subclasses must implement `pack` and `flush_pack`, and provide a serializer.
    This design separates frame packing logic from serialization format.
    """

    def __init__(self, serializer: AbstractSerializer, payload_device=None):
        """
        Initialize the encoder.

        Args:
            serializer: The serializer to use for converting Payload to bytes.
            payload_device: The target device for encoded Payloads before
                serialization (e.g., 'cpu', 'cuda'). If None, no device
                transfer is performed.
        """
        self._serializer = serializer
        self._payload_device = payload_device

    @abstractmethod
    def pack(self, frame: GaussianModel) -> Iterator[Payload]:
        """
        Pack a single frame into Payload objects.

        This method transforms the input frame into Payload objects that
        can be serialized.

        Args:
            frame: A GaussianModel instance to pack.

        Yields:
            Packed Payload instances. May yield zero, one, or multiple
            payloads. When the iterator is exhausted, all payloads for this
            frame have been yielded.
        """
        pass

    @abstractmethod
    def flush_pack(self) -> Iterator[Payload]:
        """
        Flush any remaining buffered payloads from the packing stage.

        This method should be called after all frames have been packed
        to ensure any remaining buffered payloads are output.

        Yields:
            Remaining buffered Payload instances. May yield zero, one, or
            multiple payloads until all buffered data has been flushed.
        """
        pass

    def encode_frame(self, frame: GaussianModel) -> Iterator[bytes]:
        """
        Encode a single frame of GaussianModel.

        This method packs the frame into Payloads and then serializes them.

        Args:
            frame: A GaussianModel instance to encode.

        Yields:
            Encoded byte chunks. May yield zero, one, or multiple chunks.
            When the iterator is exhausted, all data for this frame has been
            encoded.
        """
        for payload in self.pack(frame):
            if self._payload_device is not None:
                payload = payload.to(self._payload_device)
            yield from self._serializer.serialize_frame(payload)

    def flush(self) -> Iterator[bytes]:
        """
        Flush any remaining buffered data from both packing and serialization.

        This method should be called after all frames have been encoded
        to ensure any remaining buffered data is output.

        Yields:
            Remaining buffered byte chunks. May yield zero, one, or multiple
            chunks until all buffered data has been flushed.
        """
        # Flush packing stage and serialize any remaining payloads
        for payload in self.flush_pack():
            if self._payload_device is not None:
                payload = payload.to(self._payload_device)
            yield from self._serializer.serialize_frame(payload)

        # Flush serialization stage
        yield from self._serializer.flush()

    def encode_stream(self, stream: Iterator[GaussianModel]) -> Iterator[bytes]:
        """
        Encode a stream of GaussianModel frames.

        This method packs each frame and serializes the payloads.
        It handles the flush logic for both packing and serialization stages.

        Args:
            stream: An iterator that yields GaussianModel instances to encode.

        Yields:
            Encoded bytes for each packed frame or flush operation.
        """
        for frame in stream:
            yield from self.encode_frame(frame)

        yield from self.flush()
