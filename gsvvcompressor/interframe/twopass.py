from abc import abstractmethod
from dataclasses import dataclass
from typing import Iterator, List, Optional

from gaussian_splatting import GaussianModel

from ..encoder import AbstractEncoder
from ..payload import Payload
from ..serializer import AbstractSerializer
from .interface import (
    InterframeCodecContext,
    InterframeCodecInterface,
    InterframeEncoderInitConfig,
)
from .encoder import InterframeEncoder


@dataclass
class PassOneContext:
    """
    Context data from the first pass of two-pass encoding.

    This dataclass holds the information gathered during the first pass
    over all frames, which is then used to optimize the encoding in the
    second pass. For example, this could contain statistics about the
    entire video sequence, codebook data, or other global information.

    Subclasses should define specific fields for their encoding scheme.
    """
    pass


class TwoPassInterframeCodecInterface(InterframeCodecInterface):
    """
    Abstract interface for two-pass inter-frame encoding/decoding algorithms.

    This interface extends InterframeCodecInterface to support two-pass encoding,
    where the first pass gathers information about all frames, and the second
    pass uses this information to optimize encoding.

    Design Guidelines:
        - PassOneContext: Store information gathered during the first pass
          (e.g., global statistics, codebook data, quality parameters).
        - The keyframe_to_context method receives PassOneContext instead of
          InterframeEncoderInitConfig, allowing it to use first-pass data.
    """

    @abstractmethod
    def keyframe_to_context_pass_one(
        self,
        frame: GaussianModel,
        init_config: InterframeEncoderInitConfig,
    ) -> PassOneContext:
        """
        Process the keyframe during the first pass to initialize PassOneContext.

        This method is called by the encoder when processing the first frame
        during the first pass. It initializes the PassOneContext with information
        from the keyframe.

        Args:
            frame: The GaussianModel keyframe to process.
            init_config: Encoder initialization configuration.

        Returns:
            PassOneContext initialized from the keyframe.
        """
        pass

    @abstractmethod
    def interframe_to_context_pass_one(
        self,
        frame: GaussianModel,
        prev_pass_one_context: PassOneContext,
    ) -> PassOneContext:
        """
        Process a frame during the first pass to update PassOneContext.

        This method is called by the encoder when processing subsequent frames
        during the first pass. It updates the PassOneContext with information
        gathered from the current frame.

        Args:
            frame: The GaussianModel frame to process.
            prev_pass_one_context: The PassOneContext from processing previous frames.

        Returns:
            Updated PassOneContext incorporating information from this frame.
        """
        pass

    @abstractmethod
    def keyframe_to_context(
        self,
        frame: GaussianModel,
        pass_one_context: PassOneContext,
    ) -> InterframeCodecContext:
        """
        Convert a keyframe to a Context using first-pass information.

        This method is called by the encoder when processing the first frame
        in the second pass. The pass_one_context provides information gathered
        during the first pass for optimized encoding.

        Args:
            frame: The GaussianModel frame to convert.
            pass_one_context: Context from the first pass containing
                global encoding information.

        Returns:
            The corresponding Context representation.
        """
        pass


class TwoPassInterframeEncoder(InterframeEncoder):
    """
    Encoder that uses two-pass inter-frame compression.

    This encoder collects frames during pack() calls (pass one), then
    performs the actual encoding in flush_pack() (pass two).

    Pass one: Gather information from all frames to build PassOneContext.
    Pass two: Encode all frames using the gathered information.
    """

    def __init__(
        self,
        serializer: AbstractSerializer,
        interface: TwoPassInterframeCodecInterface,
        init_config: InterframeEncoderInitConfig,
        payload_device=None,
    ):
        """
        Initialize the two-pass inter-frame encoder.

        Args:
            serializer: The serializer to use for converting Payload to bytes.
            interface: The TwoPassInterframeCodecInterface instance that
                provides encoding methods.
            init_config: Configuration parameters for encoder initialization.
        """
        super().__init__(serializer=serializer, interface=interface, init_config=init_config, payload_device=payload_device)
        self._interface = interface
        self._frames: List[GaussianModel] = []
        self._pass_one_context: Optional[PassOneContext] = None

    def pack(self, frame: GaussianModel) -> Iterator[Payload]:
        """
        Perform pass one on the frame and store it for pass two.

        During pass one, this method processes each frame to gather encoding
        information and stores the frame for later encoding in flush_pack().

        Args:
            frame: A GaussianModel instance to process.

        Yields:
            No payloads during pass one (empty iterator).
        """
        # Pass one: gather information
        if self._pass_one_context is None:
            # First frame: keyframe
            self._pass_one_context = self._interface.keyframe_to_context_pass_one(
                frame, self._init_config
            )
        else:
            # Subsequent frames: interframe
            self._pass_one_context = self._interface.interframe_to_context_pass_one(
                frame, self._pass_one_context
            )

        # Store frame for pass two
        self._frames.append(frame)

        # No payloads during pass one
        return
        yield  # Make this a generator

    def flush_pack(self) -> Iterator[Payload]:
        """
        Perform pass two: encode all stored frames using pass one information.

        This method encodes all frames that were collected during pack() calls,
        using the PassOneContext gathered during pass one.

        Yields:
            Packed Payload instances for all frames.
        """
        if not self._frames or self._pass_one_context is None:
            return
            yield  # Make this a generator

        prev_context: Optional[InterframeCodecContext] = None

        for frame in self._frames:
            if prev_context is None:
                # First frame: convert and encode as keyframe using pass_one_context
                current_context = self._interface.keyframe_to_context(
                    frame, self._pass_one_context
                )
                payload = self._interface.encode_keyframe(current_context)
                # Decode back to get reconstructed context (avoid error accumulation)
                reconstructed_context = self._interface.decode_keyframe_for_encode(
                    payload, current_context
                )
            else:
                # Subsequent frames: convert and encode as delta from previous
                current_context = self._interface.interframe_to_context(
                    frame, prev_context
                )
                payload = self._interface.encode_interframe(
                    prev_context, current_context
                )
                # Decode back to get reconstructed context (avoid error accumulation)
                reconstructed_context = self._interface.decode_interframe_for_encode(
                    payload, prev_context
                )

            # Use reconstructed context as previous for next frame
            prev_context = reconstructed_context

            yield payload

        # Clear stored frames after encoding
        self._frames.clear()
