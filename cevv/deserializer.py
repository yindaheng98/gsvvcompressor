from abc import ABC, abstractmethod
from typing import Iterator

from .payload import Payload


class AbstractDeserializer(ABC):
    """
    Abstract base class for deserializing bytes to Payload objects.

    Subclasses must implement the `deserialize_frame` method to define
    the specific deserialization logic for byte sequences into Payload.
    """

    @abstractmethod
    def deserialize_frame(self, data: bytes) -> Iterator[Payload]:
        """
        Deserialize a chunk of bytes to Payload objects.

        Args:
            data: A bytes object containing serialized data to deserialize.
                  If the length is 0, it signals the end of the input stream,
                  prompting the deserializer to flush remaining buffered data.

        Yields:
            Deserialized Payload instances. May yield zero, one, or multiple
            payloads depending on available data. When the iterator is exhausted,
            all complete payloads from this data chunk have been yielded.
            For flush (data=b""), yields any remaining buffered payloads.
        """
        pass
