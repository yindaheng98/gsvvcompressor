from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Self


@dataclass
class Payload(ABC):
    """
    Abstract base class for frame payload data.

    This represents the intermediate data structure after processing a frame,
    before serialization. Subclasses should define specific fields for their
    encoding scheme.

    The name "Payload" reflects its role as the processed data that will be
    passed to a serializer for conversion to bytes.
    """

    @abstractmethod
    def to(self, device) -> Self:
        """
        Move the Payload to the specified device for further processing.

        Subclasses must implement this method to handle device transfer
        (e.g., moving tensors to GPU/CPU).

        Args:
            device: The target device (e.g., 'cpu', 'cuda', torch.device).

        Returns:
            A new Payload instance on the target device, or self if already
            on the target device.
        """
        pass
