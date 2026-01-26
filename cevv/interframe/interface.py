from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

from gaussian_splatting import GaussianModel

from ..payload import Payload


@dataclass
class InterframeContext:
    """
    Context data for inter-frame encoding/decoding.

    This dataclass holds the state information needed to encode/decode
    frames relative to a reference frame. Subclasses should define
    specific fields for their encoding scheme.
    """
    pass


class InterframeInterface(ABC):
    """
    Abstract interface for inter-frame encoding/decoding algorithms.

    This interface defines the methods required to implement inter-frame
    compression, where frames are encoded as differences from previous frames.

    All methods are abstract static methods that must be implemented by
    concrete subclasses.
    """

    @staticmethod
    @abstractmethod
    def decode_interframe(payload: Payload, prev_context: InterframeContext) -> InterframeContext:
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
    def encode_interframe(prev_context: InterframeContext, next_context: InterframeContext) -> Payload:
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
    def decode_keyframe(payload: Payload) -> InterframeContext:
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
    def encode_keyframe(context: InterframeContext) -> Payload:
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
    def frame_to_context(
        frame: GaussianModel,
        prev_context: Optional[InterframeContext] = None,
    ) -> InterframeContext:
        """
        Convert a frame to a Context.

        Args:
            frame: The GaussianModel frame to convert.
            prev_context: Optional context from the previous frame,
                which may be used for reference during conversion.

        Returns:
            The corresponding Context representation.
        """
        pass

    @staticmethod
    @abstractmethod
    def context_to_frame(context: InterframeContext) -> GaussianModel:
        """
        Convert a Context back to a frame.

        Args:
            context: The Context to convert.

        Returns:
            The corresponding GaussianModel frame.
        """
        pass
