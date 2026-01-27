from typing import Iterator, Optional

from gaussian_splatting import GaussianModel

from ..decoder import AbstractDecoder
from ..deserializer import AbstractDeserializer
from ..payload import Payload
from .interface import InterframeCodecContext, InterframeCodecInterface


class InterframeDecoder(AbstractDecoder):
    """
    Decoder that uses inter-frame decompression.

    This decoder maintains an internal Context state and decodes frames
    by applying deltas to the previous frame's context. The first frame
    is decoded as a keyframe, and subsequent frames are decoded as deltas.
    """

    def __init__(
        self,
        deserializer: AbstractDeserializer,
        interface: InterframeCodecInterface,
        payload_device=None,
    ):
        """
        Initialize the inter-frame decoder.

        Args:
            deserializer: The deserializer to use for converting bytes to Payload.
            interface: The InterframeCodecInterface instance that provides decoding methods.
            payload_device: The target device for input Payloads before
                unpacking (e.g., 'cpu', 'cuda'). If None, no device
                transfer is performed.
        """
        super().__init__(deserializer, payload_device)
        self._interface = interface
        self._prev_context: Optional[InterframeCodecContext] = None

    def create_empty_frame(self) -> GaussianModel:
        """
        Create an empty GaussianModel for frame reconstruction.

        This method is called before `context_to_frame` to provide a frame
        that will be populated with data. Override this method if your
        GaussianModel contains custom data or attributes that are not part of
        the standard GaussianModel(), such as additional fields or custom
        initialization logic.

        Returns:
            An empty GaussianModel instance.
        """
        return GaussianModel()

    def unpack(self, payload: Payload) -> Iterator[GaussianModel]:
        """
        Unpack frame(s) from a Payload using inter-frame decoding.

        The first payload is decoded as a keyframe. Subsequent payloads
        are decoded as deltas from the previous frame.

        Args:
            payload: A Payload instance to unpack.

        Yields:
            Unpacked GaussianModel instances.
        """
        if self._prev_context is None:
            # First frame: decode as keyframe
            current_context = self._interface.decode_keyframe(payload)
        else:
            # Subsequent frames: decode as delta from previous
            current_context = self._interface.decode_interframe(
                payload, self._prev_context
            )

        # Update the previous context for next frame
        self._prev_context = current_context

        # Convert context back to frame
        frame = self.create_empty_frame()
        yield self._interface.context_to_frame(current_context, frame)

    def flush_unpack(self) -> Iterator[GaussianModel]:
        """
        Flush any remaining buffered frames from the unpacking stage.

        For inter-frame decoding, there are no buffered frames to flush.

        Yields:
            No frames (empty iterator).
        """
        return
        yield  # Make this a generator
