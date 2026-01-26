from abc import ABC, abstractmethod
from dataclasses import dataclass

from gaussian_splatting import GaussianModel

from ..payload import Payload


@dataclass
class InterframeCodecConfig:
    """
    Configuration parameters for initializing inter-frame context from a keyframe.

    This dataclass holds the initialization settings needed when converting
    a keyframe to context. Subclasses should define specific fields for
    their encoding scheme.
    """
    pass


@dataclass
class InterframeCodecContext:
    """
    Context data for inter-frame encoding/decoding.

    This dataclass holds the state information needed to encode/decode
    frames relative to a reference frame. Subclasses should define
    specific fields for their encoding scheme.
    """
    pass


class InterframeCodecInterface(ABC):
    """
    Abstract interface for inter-frame encoding/decoding algorithms.

    This interface defines the methods required to implement inter-frame
    compression, where frames are encoded as differences from previous frames.

    All methods are abstract static methods that must be implemented by
    concrete subclasses.
    """

    @staticmethod
    @abstractmethod
    def decode_interframe(payload: Payload, prev_context: InterframeCodecContext) -> InterframeCodecContext:
        """
        Decode a delta payload to reconstruct the next frame's context.

        Args:
            payload: The delta payload containing the difference data.
            prev_context: The context of the previous frame.

        Returns:
            The reconstructed context for the current frame.
        """
        pass

    @staticmethod
    @abstractmethod
    def encode_interframe(prev_context: InterframeCodecContext, next_context: InterframeCodecContext) -> Payload:
        """
        Encode the difference between two consecutive frames.

        Args:
            prev_context: The context of the previous frame.
            next_context: The context of the next frame.

        Returns:
            A payload containing the delta information.
        """
        pass

    @staticmethod
    @abstractmethod
    def decode_keyframe(payload: Payload) -> InterframeCodecContext:
        """
        Decode a keyframe payload to create initial context.

        Args:
            payload: The keyframe payload containing full frame data.

        Returns:
            The context for the first/key frame.
        """
        pass

    @staticmethod
    @abstractmethod
    def encode_keyframe(context: InterframeCodecContext) -> Payload:
        """
        Encode the first frame as a keyframe.

        Args:
            context: The context of the first frame.

        Returns:
            A payload containing the full keyframe data.
        """
        pass

    @staticmethod
    @abstractmethod
    def keyframe_to_context(frame: GaussianModel, init_config: InterframeCodecConfig) -> InterframeCodecContext:
        """
        Convert a keyframe to a Context.

        Args:
            frame: The GaussianModel frame to convert.
            init_config: Configuration parameters for initialization.

        Returns:
            The corresponding Context representation.
        """
        pass

    @staticmethod
    @abstractmethod
    def interframe_to_context(
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

    @staticmethod
    @abstractmethod
    def context_to_frame(context: InterframeCodecContext) -> GaussianModel:
        """
        Convert a Context back to a frame.

        Args:
            context: The Context to convert.

        Returns:
            The corresponding GaussianModel frame.
        """
        pass
