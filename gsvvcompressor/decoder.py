from abc import ABC, abstractmethod
from typing import Iterator, Self

import torch

from gaussian_splatting import GaussianModel

from .deserializer import AbstractDeserializer
from .payload import Payload


class AbstractDecoder(ABC):
    """
    Abstract base class for decompression algorithms.

    This decoder uses a two-stage process:
    1. Deserialize bytes to Payload objects (via the deserializer)
    2. Unpack Payloads to frames (via `unpack`)

    Subclasses must implement `unpack` and `flush_unpack`, and provide
    a deserializer. This design separates frame unpacking logic from
    deserialization format.
    """

    def __init__(
        self,
        deserializer: AbstractDeserializer,
        payload_device: str | torch.device | None = None,
        device: str | torch.device | None = None,
    ):
        """
        Initialize the decoder.

        Args:
            deserializer: The deserializer to use for converting bytes to Payload.
            payload_device: The target device for input Payloads before
                unpacking (e.g., 'cpu', 'cuda'). If None, no device
                transfer is performed.
            device: The target device for output GaussianModel frames
                (e.g., 'cpu', 'cuda'). If None, no device transfer is performed.
        """
        self._deserializer = deserializer
        self._payload_device = payload_device
        self._device = device

    def to(self, device: str | torch.device | None) -> Self:
        """Set the device for decoded models.

        Args:
            device: The device to move models to after decoding.
                    Can be a string like "cuda" or "cpu", a torch.device, or None.

        Returns:
            self for method chaining.
        """
        self._device = device
        return self

    @abstractmethod
    def unpack(self, payload: Payload) -> Iterator[GaussianModel]:
        """
        Unpack frame(s) from a Payload.

        This method reconstructs the original frame(s) from the Payload object.

        Args:
            payload: A Payload instance to unpack.

        Yields:
            Unpacked GaussianModel instances. May yield zero, one, or multiple
            models. When the iterator is exhausted, all frames for this payload
            have been yielded.
        """
        pass

    @abstractmethod
    def flush_unpack(self) -> Iterator[GaussianModel]:
        """
        Flush any remaining buffered frames from the unpacking stage.

        This method should be called after all payloads have been unpacked
        to ensure any remaining buffered frames are output.

        Yields:
            Remaining buffered GaussianModel instances. May yield zero, one,
            or multiple models until all buffered data has been flushed.
        """
        pass

    def decode_frame(self, data: bytes) -> Iterator[GaussianModel]:
        """
        Decode a single chunk of compressed data.

        This method deserializes the bytes to Payloads and then unpacks
        the original frames.

        Args:
            data: A bytes object containing compressed data to decode.

        Yields:
            Decoded GaussianModel instances. May yield zero, one, or multiple
            models. When the iterator is exhausted, all frames for this data
            chunk have been decoded.
        """
        for payload in self._deserializer.deserialize_frame(data):
            if self._payload_device is not None:
                payload = payload.to(self._payload_device)
            for model in self.unpack(payload):
                if self._device is not None:
                    model.to(self._device)
                yield model

    def flush(self) -> Iterator[GaussianModel]:
        """
        Flush any remaining buffered data from both deserialization and unpacking.

        This method should be called after all data has been decoded
        to ensure any remaining buffered frames are output.

        Yields:
            Remaining buffered GaussianModel instances. May yield zero, one,
            or multiple models until all buffered data has been flushed.
        """
        # Flush deserialization stage and unpack any remaining payloads
        for payload in self._deserializer.flush():
            if self._payload_device is not None:
                payload = payload.to(self._payload_device)
            for model in self.unpack(payload):
                if self._device is not None:
                    model.to(self._device)
                yield model

        # Flush unpacking stage
        for model in self.flush_unpack():
            if self._device is not None:
                model.to(self._device)
            yield model

    def decode_stream(self, stream: Iterator[bytes]) -> Iterator[GaussianModel]:
        """
        Decode a stream of compressed byte chunks into GaussianModel frames.

        This method processes each chunk, deserializing and unpacking frames.
        It handles the flush logic for both deserialization and unpacking stages.

        Args:
            stream: An iterator that yields bytes objects to decode.

        Yields:
            Decoded GaussianModel instances for each processed chunk or flush.
        """
        for data in stream:
            yield from self.decode_frame(data)

        yield from self.flush()
