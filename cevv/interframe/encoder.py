from typing import Iterator, Optional, Type

from gaussian_splatting import GaussianModel

from ..encoder import AbstractEncoder
from ..payload import Payload
from ..serializer import AbstractSerializer
from .interface import InterframeContext, InterframeInterface


class InterframeEncoder(AbstractEncoder):
    """
    Encoder that uses inter-frame compression.

    This encoder maintains an internal Context state and encodes frames
    as differences from the previous frame. The first frame is encoded
    as a keyframe, and subsequent frames are encoded as deltas.
    """

    def __init__(
        self,
        serializer: AbstractSerializer,
        interface: Type[InterframeInterface],
    ):
        """
        Initialize the inter-frame encoder.

        Args:
            serializer: The serializer to use for converting Payload to bytes.
            interface: The InterframeInterface class that provides encoding methods.
        """
        super().__init__(serializer)
        self._interface = interface
        self._prev_context: Optional[InterframeContext] = None

    def pack(self, frame: GaussianModel) -> Iterator[Payload]:
        """
        Pack a single frame into Payload objects using inter-frame encoding.

        The first frame is encoded as a keyframe. Subsequent frames are
        encoded as deltas from the previous frame.

        Args:
            frame: A GaussianModel instance to pack.

        Yields:
            Packed Payload instances.
        """
        if self._prev_context is None:
            # First frame: convert and encode as keyframe
            current_context = self._interface.keyframe_to_context(frame)
            payload = self._interface.encode_keyframe(current_context)
            # Decode back to get reconstructed context (avoid error accumulation)
            reconstructed_context = self._interface.decode_keyframe(payload)
        else:
            # Subsequent frames: convert and encode as delta from previous
            current_context = self._interface.interframe_to_context(
                frame, self._prev_context
            )
            payload = self._interface.encode_interframe(
                self._prev_context, current_context
            )
            # Decode back to get reconstructed context (avoid error accumulation)
            reconstructed_context = self._interface.decode_interframe(
                payload, self._prev_context
            )

        # Use reconstructed context as previous for next frame
        self._prev_context = reconstructed_context

        yield payload

    def flush_pack(self) -> Iterator[Payload]:
        """
        Flush any remaining buffered payloads from the packing stage.

        For inter-frame encoding, there are no buffered payloads to flush.

        Yields:
            No payloads (empty iterator).
        """
        return
        yield  # Make this a generator
