from abc import ABC
from dataclasses import dataclass


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
    pass
