from abc import ABC, abstractmethod
from typing import Iterator

from .payload import Payload


class AbstractSerializer(ABC):
    """
    Abstract base class for serializing Payload objects to bytes.

    Subclasses must implement `serialize_frame` and `flush` methods to define
    the specific serialization logic for Payload objects.
    """

    @abstractmethod
    def serialize_frame(self, payload: Payload) -> Iterator[bytes]:
        """
        Serialize a single Payload object to bytes.

        Args:
            payload: A Payload instance to serialize.

        Yields:
            Serialized byte chunks. May yield zero, one, or multiple chunks.
            When the iterator is exhausted, all data for this payload has been
            serialized.
        """
        pass

    @abstractmethod
    def flush(self) -> Iterator[bytes]:
        """
        Flush any remaining buffered data.

        This method should be called after all payloads have been serialized
        to ensure any remaining buffered data is output.

        Yields:
            Remaining buffered byte chunks. May yield zero, one, or multiple
            chunks until all buffered data has been flushed.
        """
        pass
