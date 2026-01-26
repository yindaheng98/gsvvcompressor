from abc import ABC, abstractmethod
from typing import Optional

from .payload import Payload


class AbstractSerializer(ABC):
    """
    Abstract base class for serializing Payload objects to bytes.

    Subclasses must implement the `serialize_frame` method to define
    the specific serialization logic for Payload objects.
    """

    @abstractmethod
    def serialize_frame(self, payload: Optional[Payload]) -> tuple[bytes, bool]:
        """
        Serialize a single Payload object to bytes.

        Args:
            payload: A Payload instance to serialize, or None to flush
                     remaining data from the internal buffer.

        Returns:
            A tuple of (serialized_bytes, is_finished):
            - serialized_bytes: The serialized byte data (can be empty).
            - is_finished: True if all previously input Payloads have been
                           fully serialized; False if more data is pending.
        """
        pass
