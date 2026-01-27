from dataclasses import dataclass
from typing import List, Self

from gaussian_splatting import GaussianModel

from ..payload import Payload
from .interface import (
    InterframeCodecInterface,
    InterframeCodecContext,
    InterframeEncoderInitConfig,
)


@dataclass
class CombinedPayload(Payload):
    """
    Combined payload containing multiple sub-codec payloads.
    """
    payloads: List[Payload]

    def to(self, device) -> Self:
        """
        Move the Payload to the specified device.

        Args:
            device: The target device (e.g., 'cpu', 'cuda', torch.device).

        Returns:
            A new CombinedPayload instance with all sub-payloads on the target device.
        """
        return CombinedPayload(
            payloads=[p.to(device) for p in self.payloads],
        )


@dataclass
class CombinedInterframeEncoderInitConfig(InterframeEncoderInitConfig):
    """
    Combined encoder initialization configuration containing multiple sub-codec configs.
    """
    init_configs: List[InterframeEncoderInitConfig]


@dataclass
class CombinedInterframeCodecContext(InterframeCodecContext):
    """
    Combined context containing multiple sub-codec contexts.
    """
    contexts: List[InterframeCodecContext]


class CombinedInterframeCodecInterface(InterframeCodecInterface):
    """
    A codec that combines multiple InterframeCodecInterface instances.

    Each method calls the corresponding method on all sub-codecs and combines
    their outputs.
    """

    def __init__(self, interfaces: List[InterframeCodecInterface]):
        """
        Initialize the combined codec with a list of sub-codecs.

        Args:
            interfaces: List of InterframeCodecInterface instances to combine.
        """
        if not interfaces:
            raise ValueError("At least one interface must be provided")
        self.interfaces = interfaces

    def decode_interframe(
        self, payload: CombinedPayload, prev_context: CombinedInterframeCodecContext
    ) -> CombinedInterframeCodecContext:
        """
        Decode a delta payload to reconstruct the next frame's context.

        Calls decode_interframe on all sub-codecs and combines their contexts.

        Args:
            payload: The delta payload containing individual payloads for each sub-codec.
            prev_context: The context of the previous frame containing individual contexts for each sub-codec.

        Returns:
            A CombinedInterframeCodecContext containing all sub-codec contexts.
        """
        # Call decode_interframe on each sub-codec
        contexts = []
        for interface, payload, prev_context in zip(self.interfaces, payload.payloads, prev_context.contexts):
            context = interface.decode_interframe(payload, prev_context)
            contexts.append(context)
        return CombinedInterframeCodecContext(contexts=contexts)

    def encode_interframe(
        self,
        prev_context: CombinedInterframeCodecContext,
        next_context: CombinedInterframeCodecContext,
    ) -> CombinedPayload:
        """
        Encode the difference between two consecutive frames.

        Calls encode_interframe on all sub-codecs and combines their payloads.

        Args:
            prev_context: The context of the previous frame containing individual contexts for each sub-codec.
            next_context: The context of the next frame containing individual contexts for each sub-codec.

        Returns:
            A CombinedPayload containing all sub-codec payloads.
        """
        # Call encode_interframe on each sub-codec
        payloads = []
        for interface, prev_context, next_context in zip(self.interfaces, prev_context.contexts, next_context.contexts):
            payload = interface.encode_interframe(prev_context, next_context)
            payloads.append(payload)
        return CombinedPayload(payloads=payloads)

    def decode_keyframe(self, payload: CombinedPayload) -> CombinedInterframeCodecContext:
        """
        Decode a keyframe payload to create initial context.

        Calls decode_keyframe on all sub-codecs and combines their contexts.

        Args:
            payload: The keyframe payload containing individual payloads for each sub-codec.

        Returns:
            A CombinedInterframeCodecContext containing all sub-codec contexts.
        """
        # Call decode_keyframe on each sub-codec
        contexts = []
        for interface, payload in zip(self.interfaces, payload.payloads):
            context = interface.decode_keyframe(payload)
            contexts.append(context)
        return CombinedInterframeCodecContext(contexts=contexts)

    def encode_keyframe(
        self, context: CombinedInterframeCodecContext
    ) -> CombinedPayload:
        """
        Encode the first frame as a keyframe.

        Calls encode_keyframe on all sub-codecs and combines their payloads.

        Args:
            context: The context of the first frame containing individual contexts for each sub-codec.

        Returns:
            A CombinedPayload containing all sub-codec payloads.
        """
        # Call encode_keyframe on each sub-codec
        payloads = []
        for interface, context in zip(self.interfaces, context.contexts):
            payload = interface.encode_keyframe(context)
            payloads.append(payload)
        return CombinedPayload(payloads=payloads)

    def keyframe_to_context(
        self, frame: GaussianModel, init_config: CombinedInterframeEncoderInitConfig
    ) -> CombinedInterframeCodecContext:
        """
        Convert a keyframe to a Context.

        Calls keyframe_to_context on all sub-codecs and combines their contexts.

        Args:
            frame: The GaussianModel frame to convert.
            init_config: Encoder initialization configuration containing individual configs for each sub-codec.

        Returns:
            A CombinedInterframeCodecContext containing all sub-codec contexts.
        """
        # Call keyframe_to_context on each sub-codec
        contexts = []
        for interface, init_config in zip(self.interfaces, init_config.init_configs):
            context = interface.keyframe_to_context(frame, init_config)
            contexts.append(context)
        return CombinedInterframeCodecContext(contexts=contexts)

    def interframe_to_context(
        self,
        frame: GaussianModel,
        prev_context: CombinedInterframeCodecContext,
    ) -> CombinedInterframeCodecContext:
        """
        Convert a frame to a Context using the previous context as reference.

        Calls interframe_to_context on all sub-codecs and combines their contexts.

        Args:
            frame: The GaussianModel frame to convert.
            prev_context: The context from the previous frame containing individual contexts for each sub-codec.

        Returns:
            A CombinedInterframeCodecContext containing all sub-codec contexts.
        """
        # Call interframe_to_context on each sub-codec
        contexts = []
        for interface, prev_context in zip(self.interfaces, prev_context.contexts):
            context = interface.interframe_to_context(frame, prev_context)
            contexts.append(context)
        return CombinedInterframeCodecContext(contexts=contexts)

    def context_to_frame(
        self, context: CombinedInterframeCodecContext, frame: GaussianModel
    ) -> GaussianModel:
        """
        Convert a Context back to a frame.

        Calls context_to_frame on all sub-codecs sequentially, applying each
        context to the frame in order.

        Args:
            context: The Context to convert containing individual contexts for each sub-codec.
            frame: An empty GaussianModel or one from previous pipeline steps.
                This frame will be modified in-place by each sub-codec.

        Returns:
            The modified GaussianModel with the frame data.
        """
        # Apply each context sequentially to the frame
        for interface, context in zip(self.interfaces, context.contexts):
            frame = interface.context_to_frame(context, frame)
        return frame
