from abc import ABC, abstractmethod
from dataclasses import dataclass

from gaussian_splatting import GaussianModel

from ..payload import Payload


@dataclass
class InterframeEncoderInitConfig:
    """
    Configuration for initializing the encoder when encoding the first frame (keyframe).

    This dataclass is passed to `keyframe_to_context()` when the encoder processes
    the first frame. It contains encoder-specific settings that determine how the
    encoding context is initialized, such as quality settings, compression levels,
    or algorithm parameters.

    Subclasses should define specific fields for their encoding scheme.

    Note:
        This config is only used by the encoder during keyframe processing.
        Configurations shared by both encoder and decoder should be stored
        in InterframeCodecInterface.
    """
    pass


@dataclass
class InterframeCodecContext:
    """
    Context data for inter-frame encoding/decoding.

    This dataclass holds the state information needed to encode/decode
    frames relative to a reference frame. Both encoder and decoder
    maintain their own InterframeCodecContext instances.

    Subclasses should define specific fields for their encoding scheme.

    Note:
        This is for encoder/decoder runtime state (e.g., reference frame data,
        accumulated statistics). Encoder initialization config should be stored
        in InterframeEncoderInitConfig. Shared configurations should be stored
        in InterframeCodecInterface.
    """
    pass


class InterframeCodecInterface(ABC):
    """
    Abstract interface for inter-frame encoding/decoding algorithms.

    This interface defines the methods required to implement inter-frame
    compression, where frames are encoded as differences from previous frames.

    Design Guidelines:
        - InterframeCodecInterface: Store configurations shared by both encoder
          and decoder (e.g., algorithm parameters that affect both encoding and
          decoding, such as quantization tables or codebook sizes).
        - InterframeCodecContext: Store encoder/decoder runtime state (e.g.,
          reference frame data, accumulated statistics). Each side maintains
          its own context instance.
        - InterframeEncoderInitConfig: Store encoder initialization config that
          is passed to `keyframe_to_context()` when encoding the first frame
          (e.g., quality settings, compression levels).
    """

    @abstractmethod
    def decode_interframe(self, payload: Payload, prev_context: InterframeCodecContext) -> InterframeCodecContext:
        """
        Decode a delta payload to reconstruct the next frame's context.

        Args:
            payload: The delta payload containing the difference data.
            prev_context: The context of the previous frame.

        Returns:
            The reconstructed context for the current frame.
        """
        pass

    @abstractmethod
    def encode_interframe(self, prev_context: InterframeCodecContext, next_context: InterframeCodecContext) -> Payload:
        """
        Encode the difference between two consecutive frames.

        Args:
            prev_context: The context of the previous frame.
            next_context: The context of the next frame.

        Returns:
            A payload containing the delta information.
        """
        pass

    @abstractmethod
    def decode_keyframe(self, payload: Payload) -> InterframeCodecContext:
        """
        Decode a keyframe payload to create initial context.

        Args:
            payload: The keyframe payload containing full frame data.

        Returns:
            The context for the first/key frame.
        """
        pass

    @abstractmethod
    def encode_keyframe(self, context: InterframeCodecContext) -> Payload:
        """
        Encode the first frame as a keyframe.

        Args:
            context: The context of the first frame.

        Returns:
            A payload containing the full keyframe data.
        """
        pass

    @abstractmethod
    def keyframe_to_context(self, frame: GaussianModel, init_config: InterframeEncoderInitConfig) -> InterframeCodecContext:
        """
        Convert a keyframe to a Context.

        This method is called by the encoder when processing the first frame.
        The init_config provides encoder-specific settings for initializing
        the encoding context.

        Args:
            frame: The GaussianModel frame to convert.
            init_config: Encoder initialization configuration.

        Returns:
            The corresponding Context representation.
        """
        pass

    @abstractmethod
    def interframe_to_context(
        self,
        frame: GaussianModel,
        prev_context: InterframeCodecContext,
    ) -> InterframeCodecContext:
        """
        Convert a frame to a Context using the previous context as reference.

        Args:
            frame: The GaussianModel frame to convert.
            prev_context: The context from the previous frame.

        Returns:
            The corresponding Context representation.
        """
        pass

    @abstractmethod
    def context_to_frame(self, context: InterframeCodecContext) -> GaussianModel:
        """
        Convert a Context back to a frame.

        Args:
            context: The Context to convert.

        Returns:
            The corresponding GaussianModel frame.
        """
        pass
