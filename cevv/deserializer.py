from abc import ABC, abstractmethod
from typing import Iterator

from .payload import Payload


class AbstractDeserializer(ABC):
    """
    Abstract base class for deserializing bytes to Payload objects.

    Subclasses must implement `deserialize_frame` and `flush` methods to define
    the specific deserialization logic for byte sequences into Payload.
    """

    @abstractmethod
    def deserialize_frame(self, data: bytes) -> Iterator[Payload]:
        """
        Deserialize a chunk of bytes to Payload objects.

        Args:
            data: A bytes object containing serialized data to deserialize.

        Yields:
            Deserialized Payload instances. May yield zero, one, or multiple
            payloads depending on available data. When the iterator is exhausted,
            all complete payloads from this data chunk have been yielded.
        """
        pass

    @abstractmethod
    def flush(self) -> Iterator[Payload]:
        """
        Flush any remaining buffered data.

        This method should be called after all data has been deserialized
        to ensure any remaining buffered payloads are output.

        Yields:
            Remaining buffered Payload instances. May yield zero, one, or
            multiple payloads until all buffered data has been flushed.
        """
        pass
