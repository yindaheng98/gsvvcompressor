from abc import ABC, abstractmethod
from typing import List

from .payload import Payload


class AbstractDeserializer(ABC):
    """
    Abstract base class for deserializing bytes to Payload objects.

    Subclasses must implement the `deserialize_frame` method to define
    the specific deserialization logic for byte sequences into Payload.
    """

    @abstractmethod
    def deserialize_frame(self, data: bytes) -> tuple[List[Payload], bool]:
        """
        Deserialize a chunk of bytes to Payload objects.

        Args:
            data: A bytes object containing serialized data to deserialize.
                  If the length is 0, it signals the end of the input stream,
                  prompting the deserializer to flush remaining buffered data.

        Returns:
            A tuple of (payloads, is_finished):
            - payloads: A list of deserialized Payload instances.
                        Can be empty if no complete payloads are available yet.
            - is_finished: True if all previously input bytes have been fully
                           deserialized; False if more data is pending.
        """
        pass
