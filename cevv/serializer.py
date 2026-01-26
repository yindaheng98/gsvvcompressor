from abc import ABC, abstractmethod
from typing import Iterator, Optional

from .payload import Payload


class AbstractSerializer(ABC):
    """
    Abstract base class for serializing Payload objects to bytes.

    Subclasses must implement the `serialize_frame` method to define
    the specific serialization logic for Payload objects.
    """

    @abstractmethod
    def serialize_frame(self, payload: Optional[Payload]) -> Iterator[bytes]:
        """
        Serialize a single Payload object to bytes.

        Args:
            payload: A Payload instance to serialize, or None to flush
                     remaining data from the internal buffer.

        Yields:
            Serialized byte chunks. May yield zero, one, or multiple chunks.
            When the iterator is exhausted, all data for this payload has been
            serialized. For flush (payload=None), yields any remaining buffered
            data until fully flushed.
        """
        pass
